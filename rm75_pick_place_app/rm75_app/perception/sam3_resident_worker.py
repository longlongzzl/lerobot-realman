#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from rm75_app.perception import sam3_mask_provider as sam3_provider


def _write(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _response(payload: dict, request_id=None) -> dict:
    if request_id is not None:
        payload["request_id"] = request_id
    return payload


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class SAM3ResidentWorker:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.config: dict = {}

    def _load_once(self, config: dict) -> dict:
        checkpoint_path = str(config.get("checkpoint_path") or sam3_provider.DEFAULT_SAM3_CHECKPOINT_PATH)
        device = str(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        resolution = int(config.get("resolution") or 1008)
        confidence_threshold = float(config.get("confidence_threshold") or 0.35)
        key = {
            "checkpoint_path": checkpoint_path,
            "device": device,
            "resolution": resolution,
            "confidence_threshold": confidence_threshold,
        }
        if self.processor is not None and self.config == key:
            return {"loaded": False, "config": key, "elapsed_ms": 0.0}

        sam3_provider._sanitize_proxy_env_for_hf()
        from sam3.model.sam3_image_processor import Sam3Processor
        import sam3.model_builder as sam3_model_builder

        # The official loader opens the 3.3GB checkpoint as a file object, which
        # prevents torch.load(mmap=True). On 16GB hosts that creates a large RAM
        # peak and can fill swap. Memory mapping keeps checkpoint pages reclaimable.
        def load_checkpoint_mmap(model, path):
            ckpt = torch.load(str(Path(path).expanduser()), map_location="cpu", weights_only=True, mmap=True)
            if "model" in ckpt and isinstance(ckpt["model"], dict):
                ckpt = ckpt["model"]
            image_ckpt = {
                name.replace("detector.", ""): value
                for name, value in ckpt.items()
                if "detector" in name
            }
            if model.inst_interactive_predictor is not None:
                image_ckpt.update(
                    {
                        name.replace("tracker.", "inst_interactive_predictor.model."): value
                        for name, value in ckpt.items()
                        if "tracker" in name
                    }
                )
            model.load_state_dict(image_ckpt, strict=False)
            del image_ckpt, ckpt
            gc.collect()

        sam3_model_builder._load_checkpoint = load_checkpoint_mmap
        build_sam3_image_model = sam3_model_builder.build_sam3_image_model

        t0 = time.perf_counter()
        _log(f"[sam3 resident] loading model device={device} resolution={resolution}")
        model = build_sam3_image_model(
            checkpoint_path=str(Path(checkpoint_path).expanduser()) if checkpoint_path else None,
            load_from_HF=not bool(checkpoint_path),
            device=device,
            eval_mode=True,
            enable_segmentation=True,
        )
        self.model = model
        self.processor = Sam3Processor(
            model,
            resolution=resolution,
            device=device,
            confidence_threshold=confidence_threshold,
        )
        self.config = key
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _log(f"[sam3 resident] ready elapsed_ms={elapsed_ms:.1f}")
        return {"loaded": True, "config": key, "elapsed_ms": elapsed_ms}

    def warmup(self, payload: dict) -> dict:
        info = self._load_once(dict(payload.get("config") or payload))
        image_path = payload.get("rgb_path")
        if image_path:
            image = Image.open(Path(image_path).expanduser()).convert("RGB")
        else:
            image = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8) + 16)
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if str(self.config.get("device", "")).startswith("cuda")
            else contextlib.nullcontext()
        )
        t0 = time.perf_counter()
        with amp_ctx:
            _ = self.processor.set_image(image)
            if str(self.config.get("device", "")).startswith("cuda"):
                torch.cuda.synchronize()
        info["image_warmup_ms"] = (time.perf_counter() - t0) * 1000.0
        return info

    def segment(self, payload: dict) -> dict:
        self._load_once(dict(payload.get("config") or payload))
        rgb_path = Path(payload["rgb_path"]).expanduser()
        output_dir = Path(payload["output_dir"]).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        items = payload.get("items")
        if items is None and payload.get("items_json"):
            items = json.loads(Path(payload["items_json"]).expanduser().read_text())
        if not isinstance(items, list):
            raise ValueError("segment requires items or items_json")

        image = Image.open(rgb_path).convert("RGB")
        rgb = np.asarray(image, dtype=np.uint8)
        device = str(self.config.get("device") or "cuda")
        args = SimpleNamespace(
            mode=str(payload.get("mode") or "text"),
            prompt=None,
            box=None,
            select_box=None,
            min_mask_area=int(payload.get("min_mask_area") or 64),
            morph_kernel=int(payload.get("morph_kernel") or 3),
            sam3_max_masks_per_item=int(payload.get("sam3_max_masks_per_item") or 1),
            device=device,
        )
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.startswith("cuda")
            else contextlib.nullcontext()
        )
        t0 = time.perf_counter()
        with amp_ctx:
            base_state = self.processor.set_image(image)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
        image_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        results = []
        for idx, item in enumerate(items):
            safe = sam3_provider._safe_name(str(item.get("id") or item.get("object_name") or f"item_{idx:02d}"))
            try:
                candidates = sam3_provider._run_one_item(
                    self.processor,
                    base_state,
                    rgb,
                    args,
                    item,
                    output_dir / "items" / safe,
                )
                if isinstance(candidates, list):
                    results.extend(candidates)
                else:
                    results.append(candidates)
            except Exception as exc:
                results.append(
                    {
                        "ok": False,
                        "id": item.get("id"),
                        "object_name": item.get("object_name"),
                        "mode": item.get("mode", args.mode),
                        "prompt": item.get("prompt"),
                        "error": repr(exc),
                    }
                )
        result = {
            "ok": True,
            "rgb_path": str(rgb_path),
            "item_count": len(results),
            "results": results,
            "image_elapsed_ms": float(image_elapsed_ms),
            "total_prompt_infer_elapsed_ms": float(sum(float(item.get("infer_elapsed_ms", 0.0)) for item in results)),
        }
        result_path = output_dir / "sam3_batch_result.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["result_path"] = str(result_path)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Resident SAM3 JSON-lines worker.")
    parser.add_argument("--checkpoint-path", type=str, default=sam3_provider.DEFAULT_SAM3_CHECKPOINT_PATH)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    args = parser.parse_args()
    worker = SAM3ResidentWorker()
    _write({"ok": True, "event": "started", "worker": "sam3"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            cmd = str(payload.get("cmd") or "")
            request_id = payload.get("request_id")
            payload.setdefault("checkpoint_path", args.checkpoint_path)
            payload.setdefault("device", args.device)
            payload.setdefault("resolution", args.resolution)
            payload.setdefault("confidence_threshold", args.confidence_threshold)
            if cmd == "warmup":
                result = worker.warmup(payload)
            elif cmd == "segment":
                result = worker.segment(payload)
            elif cmd == "shutdown":
                _write(_response({"ok": True, "cmd": cmd}, request_id))
                return
            else:
                raise ValueError(f"unknown cmd: {cmd}")
            _write(_response({"ok": True, "cmd": cmd, "result": result}, request_id))
        except Exception as exc:
            _log(traceback.format_exc())
            _write(_response({"ok": False, "error": repr(exc)}, locals().get("request_id")))


if __name__ == "__main__":
    main()
