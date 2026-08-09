#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path

from rm75_app.perception import sam6d_pose_provider as provider
from rm75_app.paths import RUNTIME_DIR


def _write(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _response(payload: dict, request_id=None) -> dict:
    if request_id is not None:
        payload["request_id"] = request_id
    return payload


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _default_provider_args() -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["sam6d_resident_worker.py", "--object-name", "lvmukuai"]
        return provider.parse_args()
    finally:
        sys.argv = old_argv


def _apply_payload_args(args: argparse.Namespace, payload: dict) -> argparse.Namespace:
    out = copy.copy(args)
    for key, value in dict(payload.get("provider_args") or {}).items():
        setattr(out, key, value)
    for key in (
        "foundationpose_root",
        "output_root",
        "sam6d_root",
        "template_cache_root",
        "pem_feature_cache_root",
        "pem_det_score_thresh",
        "pem_save_visualization",
        "pem_run_mode",
        "mask_mode",
        "camera_width",
        "camera_height",
        "camera_fps",
        "camera_serial",
        "warmup_frames",
        "camera_extrinsic_opencv_path",
        "use_direct_camera_extrinsic",
        "sam3_full_scene_mask_confirm",
        "sam3_full_scene_keep_multi_instances",
        "sam3_require_full_scene_masks",
        "sam3_show_full_scene_mask_window",
        "sam3_max_masks_per_item",
        "sam3_instance_index",
        "min_mask_area",
        "sam3_morph_kernel",
        "full_scene_pem_visualization",
        "skip_pem",
        "post_pem_mask_refine",
        "post_pem_mask_refine_objects",
        "post_pem_mask_refine_trigger_px",
    ):
        if key in payload:
            setattr(out, key, payload[key])
    return out


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


class SAM6DResidentWorker:
    def __init__(self, base_args: argparse.Namespace) -> None:
        self.base_args = base_args
        self.runner = None
        self.sam6d_root = None

    def _runner_once(self, args: argparse.Namespace):
        sam6d_root = Path(args.sam6d_root).expanduser().resolve()
        if self.runner is not None and self.sam6d_root == sam6d_root:
            return self.runner
        self.sam6d_root = sam6d_root
        self.runner = provider._get_sam6d_pem_runner(args, sam6d_root)
        return self.runner

    def warmup(self, payload: dict) -> dict:
        args = _apply_payload_args(self.base_args, payload)
        runner = self._runner_once(args)
        t0 = time.perf_counter()
        runner._load_model_once()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        result = {
            "model_loaded": True,
            "model_warmup_ms": elapsed_ms,
            "sam6d_root": str(Path(args.sam6d_root).expanduser()),
        }
        object_names = list(payload.get("object_names") or [])
        if object_names:
            result["template_features"] = self.preload_template_features(payload, args=args)
        return result

    def preload_template_features(self, payload: dict, *, args: argparse.Namespace | None = None) -> dict:
        args = _apply_payload_args(self.base_args, payload) if args is None else args
        runner = self._runner_once(args)
        runner._load_model_once()
        root = Path(args.sam6d_root).expanduser().resolve()
        cache_root = Path(payload.get("preload_output_root") or "/tmp/sam6d_resident_preload").expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)
        results = []
        t_all = time.perf_counter()
        for raw_name in list(payload.get("object_names") or []):
            item_args = copy.copy(args)
            item_args.object_name = raw_name
            t0 = time.perf_counter()
            try:
                object_name, _, mesh_file, mesh_scale = provider.resolve_object_inputs(item_args)
                run_dir = cache_root / object_name
                run_dir.mkdir(parents=True, exist_ok=True)
                cad_path = provider.export_cad_mm(mesh_file, mesh_scale, run_dir / "cad_mm.ply")
                template_dir = provider.prepare_templates(
                    item_args,
                    root,
                    run_dir,
                    cad_path,
                    object_name=object_name,
                    mesh_file=mesh_file,
                    mesh_scale=mesh_scale,
                )
                feature_cache_file = provider._pem_feature_cache_file(
                    item_args,
                    root,
                    object_name,
                    mesh_file,
                    mesh_scale,
                    template_dir,
                )
                runner._get_template_features(template_dir, feature_cache_file)
                results.append(
                    {
                        "object_name": object_name,
                        "ok": True,
                        "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
                        "feature_cache_file": None if feature_cache_file is None else str(feature_cache_file),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "object_name": str(raw_name),
                        "ok": False,
                        "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
                        "error": repr(exc),
                    }
                )
        return {
            "ok_count": sum(1 for item in results if item.get("ok")),
            "object_count": len(results),
            "elapsed_ms": (time.perf_counter() - t_all) * 1000.0,
            "results": results,
        }

    def _sam3_items_for_objects(self, args: argparse.Namespace, object_names: list[str]) -> list[dict]:
        items = []
        for name in object_names:
            item_args = copy.copy(args)
            item_args.object_name = name
            object_name, prompt, _, _ = provider.resolve_object_inputs(item_args)
            for variant_index, variant_prompt in enumerate(provider.sam3_text_prompt_variants(object_name, prompt)):
                items.append(
                    {
                        "id": f"{object_name}_p{variant_index}",
                        "object_name": object_name,
                        "mode": "text",
                        "prompt": variant_prompt,
                    }
                )
        return items

    def capture_frame(self, payload: dict) -> dict:
        args = _apply_payload_args(self.base_args, payload)
        object_names = list(payload.get("object_names") or [])
        if not object_names:
            raise ValueError("capture_frame requires object_names")
        provider.validate_object_name_args(object_names)
        output_root = Path(getattr(args, "output_root", provider.DEFAULT_OUTPUT_ROOT)).expanduser()
        scene_dir = output_root / f"{_now_stamp()}_resident_full_scene_{len(object_names)}objects_pid{os.getpid()}"
        shared_frame_dir = scene_dir / "shared_frame"
        shared_frame_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        frame = provider.capture_realsense_frame(args)
        rgb_path, depth_path, camera_path = provider.save_sam6d_input_frame(frame, shared_frame_dir)
        items = self._sam3_items_for_objects(args, object_names)
        items_path = scene_dir / "sam3_full_scene_text" / "sam3_items.json"
        items_path.parent.mkdir(parents=True, exist_ok=True)
        items_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
        run_args_path = scene_dir / "run_args.json"
        run_args_path.write_text(
            json.dumps({"mode": "resident", "object_names": object_names, "args": vars(args)}, indent=2, default=str),
            encoding="utf-8",
        )
        return {
            "scene_dir": str(scene_dir),
            "frame_dir": str(shared_frame_dir),
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "camera_path": str(camera_path),
            "items_path": str(items_path),
            "items": items,
            "object_names": object_names,
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }

    def _load_sam3_result_map(self, result_path: Path, args: argparse.Namespace) -> dict:
        payload = json.loads(Path(result_path).expanduser().read_text())
        result_map = {}
        multi_instances = bool(getattr(args, "sam3_full_scene_keep_multi_instances", False))
        max_instances = max(1, int(getattr(args, "sam3_max_masks_per_item", 1)))
        for item in payload.get("results", []):
            if not item.get("ok", True):
                continue
            name = provider.normalize_object_name(item.get("object_name") or item.get("id") or "")
            if not name:
                continue
            if multi_instances:
                result_map.setdefault(name, []).append(item)
                continue
            old = result_map.get(name)
            old_score = float(old.get("selected", {}).get("model_score", -1.0)) if isinstance(old, dict) else -1.0
            new_score = float(item.get("selected", {}).get("model_score", -1.0))
            if old is None or new_score > old_score:
                result_map[name] = item
        if multi_instances:
            for key, value in list(result_map.items()):
                result_map[key] = provider._dedupe_sam3_instances(value, max_instances)
        return result_map

    def pose_from_sam3(self, payload: dict) -> dict:
        args = _apply_payload_args(self.base_args, payload)
        object_names = list(payload.get("object_names") or [])
        if not object_names:
            raise ValueError("pose_from_sam3 requires object_names")
        provider.validate_object_name_args(object_names)

        frame_dir = Path(payload["frame_dir"]).expanduser()
        args.rgb_path = str(frame_dir / "rgb.png")
        args.depth_path = str(frame_dir / "depth.png")
        args.camera_path = str(frame_dir / "camera.json")
        args.mask_mode = "sam3_text"
        args.bbox = None
        args.pem_run_mode = "inprocess"
        frame = provider.load_offline_frame(args)

        scene_dir = Path(payload.get("scene_dir") or frame_dir.parent).expanduser()
        sam3_dir = scene_dir / "sam3_full_scene_text"
        sam3_dir.mkdir(parents=True, exist_ok=True)
        sam3_result_path = Path(payload["sam3_result_path"]).expanduser()
        result_map = self._load_sam3_result_map(sam3_result_path, args)
        vis_path = provider.save_sam3_full_scene_mask_visualization(frame, sam3_dir, object_names, result_map)
        missing = [provider.normalize_object_name(name) or str(name) for name in object_names if (provider.normalize_object_name(name) or str(name)) not in result_map]
        print(f"[sam3 resident] full-scene mask overlay: {vis_path}")
        print(f"[sam3 resident] found masks: {sorted(result_map.keys())}")
        if missing:
            print(f"[sam3 resident] missing masks: {missing}")
        if missing and bool(getattr(args, "sam3_require_full_scene_masks", False)):
            raise RuntimeError(f"SAM3 resident masks are incomplete; missing={missing}; overlay={vis_path}")

        sam6d_root = Path(args.sam6d_root).expanduser().resolve()
        runner = self._runner_once(args)
        runner._load_model_once()
        detector_cache = {"sam3_text_results": result_map, "sam3_text_full_scene_attempted": True}
        results = []
        t0 = time.perf_counter()
        attempt_index = 0
        for name in object_names:
            normalized_name = provider.normalize_object_name(name) or str(name)
            precomputed = result_map.get(normalized_name)
            instance_count = len(precomputed) if isinstance(precomputed, list) else 1
            for instance_index in range(instance_count):
                attempt_index += 1
                item_args = copy.copy(args)
                item_args.object_name = name
                item_args.sam3_instance_index = instance_index
                safe_name = provider.normalize_object_name(name) or provider._safe_cache_name(name)
                item_dir = scene_dir / f"{attempt_index:02d}_{safe_name}_i{instance_index:02d}"
                try:
                    result = provider._run_single_object_pose(item_args, frame, sam6d_root, item_dir, detector_cache=detector_cache)
                    result["ok"] = True
                except Exception as exc:
                    result = {
                        "object_name": name,
                        "sam3_instance_index": instance_index,
                        "ok": False,
                        "error": repr(exc),
                        "run_dir": str(item_dir),
                    }
                    print(f"[sam6d-resident] object={name} instance={instance_index} failed: {exc!r}")
                results.append(result)

        summary = {
            "scene_dir": str(scene_dir),
            "object_count": len(results),
            "ok_count": sum(1 for item in results if item.get("ok")),
            "results": results,
            "sam3_mask_overlay": str(vis_path),
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }
        if not bool(getattr(args, "skip_pem", False)) and bool(getattr(args, "full_scene_pem_visualization", True)):
            vis_info = provider.save_full_scene_pem_visualization(args, frame, results, scene_dir / "full_scene_pem_overlay.png")
            summary["full_scene_pem_visualization"] = vis_info
            print(f"[sam6d-resident] full-scene PEM overlay: {vis_info['path']} rendered={len(vis_info['rendered'])}")
        summary_path = scene_dir / "full_scene_pose_results.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["result_path"] = str(summary_path)
        print(f"[sam6d-resident] full scene result: {summary_path}")
        print(f"[sam6d-resident] ok_count={summary['ok_count']}/{summary['object_count']}")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Resident SAM6D PEM JSON-lines worker.")
    parser.add_argument("--sam6d-root", type=str, default=provider.DEFAULT_SAM6D_ROOT)
    parser.add_argument("--template-cache-root", type=str, default=str(RUNTIME_DIR / "sam6d_template_cache"))
    parser.add_argument("--pem-feature-cache-root", type=str, default=str(RUNTIME_DIR / "sam6d_pem_feature_cache"))
    parser.add_argument("--pem-det-score-thresh", type=float, default=0.2)
    args = parser.parse_args()
    base_args = _default_provider_args()
    base_args.sam6d_root = args.sam6d_root
    base_args.template_cache_root = args.template_cache_root
    base_args.pem_feature_cache_root = args.pem_feature_cache_root
    base_args.pem_det_score_thresh = args.pem_det_score_thresh
    base_args.pem_save_visualization = False
    worker = SAM6DResidentWorker(base_args)
    _write({"ok": True, "event": "started", "worker": "sam6d"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            cmd = str(payload.get("cmd") or "")
            if cmd == "warmup":
                result = worker.warmup(payload)
            elif cmd == "preload_templates":
                result = worker.preload_template_features(payload)
            elif cmd == "capture_frame":
                result = worker.capture_frame(payload)
            elif cmd == "pose_from_sam3":
                result = worker.pose_from_sam3(payload)
            elif cmd == "shutdown":
                _write(_response({"ok": True, "cmd": cmd}, payload.get("request_id")))
                return
            else:
                raise ValueError(f"unknown cmd: {cmd}")
            _write(_response({"ok": True, "cmd": cmd, "result": result}, payload.get("request_id")))
        except Exception as exc:
            _log(traceback.format_exc())
            _write(_response({"ok": False, "error": repr(exc)}, locals().get("payload", {}).get("request_id") if isinstance(locals().get("payload"), dict) else None))


if __name__ == "__main__":
    main()
