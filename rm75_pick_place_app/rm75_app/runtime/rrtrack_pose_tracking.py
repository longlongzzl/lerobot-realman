#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh

from rm75_app.assets.object_specs import get_object_spec, normalize_object_name, resolve_object_spec_scales
from rm75_app.paths import APP_ROOT, RUNTIME_DIR
from rm75_app.perception.rrtrack import FrameObservation, RRTrackConfig, RRTracker
from rm75_app.perception.rrtrack.agreement import TriangleMeshMaskRenderer
from rm75_app.perception.rrtrack.banks import DinoV2Descriptor, FeaturePoseBank, build_sam6d_template_bank
from rm75_app.perception.rrtrack.cutie_adapter import CutieMaskTracker
from rm75_app.perception.rrtrack.foundationpose_adapter import FoundationPoseRefiner
from rm75_app.perception.rrtrack.scene_registry import build_scene_registry, resolve_active_result
from rm75_app.perception.rrtrack.sam3_relocalizer import SAM3TextRelocalizer


DEFAULT_FOUNDATIONPOSE_ROOT = "/home/zhangzhao/PycharmProjects/FoundationPose"
DEFAULT_CUTIE_ROOT = str(RUNTIME_DIR / "third_party" / "Cutie")
DEFAULT_DINOV2_ROOT = "/home/zhangzhao/.cache/torch/hub/facebookresearch_dinov2_main"
DEFAULT_CAM_POSES = (
    "/home/zhangzhao/PycharmProjects/SAM-6D/SAM-6D/"
    "Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy"
)
DEFAULT_SAM3_PYTHON = "/home/zhangzhao/anaconda3/envs/sam3/bin/python"
DEFAULT_SAM3_CHECKPOINT = "/home/zhangzhao/Downloads/sam3.pt"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _load_camera(path: Path) -> np.ndarray:
    payload = _load_json(path)
    values = payload.get("cam_K", payload.get("K"))
    if values is None:
        raise ValueError(f"camera JSON has no cam_K/K: {path}")
    return np.asarray(values, dtype=np.float32).reshape(3, 3)


def _load_depth(path: Path, depth_scale: float) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path)
        return np.asarray(depth, dtype=np.float32) * float(depth_scale)
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(path)
    return np.asarray(depth, dtype=np.float32) * float(depth_scale)


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_initial_bundle(
    result_path: Path, object_name: str, instance_index: int = 0
) -> tuple[FrameObservation, np.ndarray, np.ndarray, dict]:
    result = resolve_active_result(_load_json(result_path), object_name, instance_index)
    run_dir = Path(result.get("run_dir", result_path.parent)).expanduser()
    frame_dir = run_dir
    if not (frame_dir / "rgb.png").exists() and (frame_dir.parent / "rgb.png").exists():
        # Same-category batch initialization keeps one shared frame in the
        # batch parent and one mask/result directory per physical instance.
        frame_dir = frame_dir.parent
    rgb_path = frame_dir / "rgb.png"
    depth_path = frame_dir / "depth.png"
    camera_path = frame_dir / "camera.json"
    mask_path = run_dir / "sam6d_binary_mask.png"
    for path in (rgb_path, depth_path, camera_path, mask_path):
        if not path.exists():
            raise FileNotFoundError(f"missing SAM6D initialization artifact: {path}")
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        raise FileNotFoundError(mask_path)
    frame = FrameObservation(
        rgb=_load_rgb(rgb_path),
        depth_m=_load_depth(depth_path, 0.001),
        K=_load_camera(camera_path),
        frame_index=0,
        timestamp_s=time.time(),
    )
    pose = np.asarray(result["T_cam_obj"], dtype=np.float64).reshape(4, 4)
    return frame, mask_image > 0, pose, result


def _run_sam6d_initialization(args: argparse.Namespace) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "rm75_app.perception.sam6d_pose_provider",
        *( ["--object-names", *args.scene_object_names] if args.scene_object_names else ["--object-name", args.object_name] ),
        "--mask-mode",
        "sam3_text",
        "--output-root",
        str(Path(args.output_root).expanduser() / "initialization"),
        "--no-sam3-full-scene-mask-confirm",
        "--no-sam3-show-full-scene-mask-window",
        "--sam3-checkpoint-path",
        str(Path(args.sam3_checkpoint).expanduser()) if Path(args.sam3_checkpoint).expanduser().is_file() else "",
    ]
    if args.camera_serial:
        cmd += ["--camera-serial", args.camera_serial]
    if args.scene_object_names:
        normalized = [normalize_object_name(name) for name in args.scene_object_names]
        if len(set(normalized)) < len(normalized):
            cmd.append("--sam3-full-scene-keep-multi-instances")
    print("[rrtrack] initializing with SAM3 + SAM6D:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(APP_ROOT), text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.returncode != 0:
        raise RuntimeError(f"SAM6D initialization failed:\n{proc.stderr[-4000:]}")
    match = re.findall(
        r"\[sam6d-gdino\] (?:result|full scene result):\s*(.+(?:sam6d_pose_result|full_scene_pose_results)\.json)",
        proc.stdout or "",
    )
    if not match:
        raise RuntimeError("SAM6D initialization returned no result JSON")
    result = Path(match[-1].strip()).expanduser()
    if not result.exists():
        raise FileNotFoundError(result)
    return result


def _load_mesh(args: argparse.Namespace) -> trimesh.Trimesh:
    spec = get_object_spec(args.object_name)
    mesh_file = args.mesh_file
    mesh_scale = args.mesh_scale
    if spec is not None:
        spec_scale, _ = resolve_object_spec_scales(spec)
        mesh_file = mesh_file or spec.mesh_file
        mesh_scale = spec_scale if mesh_scale is None else mesh_scale
    if not mesh_file:
        raise ValueError("provide --mesh-file for an object without an ObjectSpec")
    path = Path(mesh_file).expanduser()
    if not path.is_absolute():
        path = APP_ROOT / path
    loaded = trimesh.load(path, force="scene")
    mesh = loaded.copy() if isinstance(loaded, trimesh.Trimesh) else loaded.dump(concatenate=True)
    if mesh is None:
        raise ValueError(f"no geometry in {path}")
    mesh.apply_scale(float(1.0 if mesh_scale is None else mesh_scale))
    return mesh


def _prepare_offline_bank(
    args: argparse.Namespace,
    descriptor: DinoV2Descriptor | None,
    initialization_result: dict,
) -> tuple[FeaturePoseBank, Path | None]:
    if args.offline_bank:
        path = Path(args.offline_bank).expanduser()
        return FeaturePoseBank.load_npz(path), path
    if descriptor is None:
        return FeaturePoseBank("offline"), None

    safe_name = normalize_object_name(args.object_name) or "object"
    cache_path = Path(args.offline_bank_cache_root).expanduser() / f"{safe_name}_{args.dino_model_name}.npz"
    if cache_path.exists():
        print(f"[rrtrack] reused offline recovery bank: {cache_path}")
        return FeaturePoseBank.load_npz(cache_path), cache_path

    initialization_dir = Path(str(initialization_result.get("run_dir", ""))).expanduser()
    templates_dir = initialization_dir / "templates"
    if not templates_dir.exists() and (initialization_dir.parent / "templates").exists():
        templates_dir = initialization_dir.parent / "templates"
    cam_poses = Path(args.cam_poses).expanduser()
    if templates_dir.exists() and cam_poses.exists():
        print(f"[rrtrack] building offline recovery bank from {templates_dir}")
        bank = build_sam6d_template_bank(
            templates_dir,
            cam_poses,
            descriptor,
            base_views=args.offline_bank_base_views,
            progress=lambda done, total, index, entries: print(
                f"[rrtrack] offline bank {done}/{total} template={index} entries={entries}"
            ),
        )
        bank.save_npz(cache_path)
        print(f"[rrtrack] cached offline recovery bank: {cache_path}")
        return bank, cache_path

    if args.allow_online_bank_only:
        print("[rrtrack] warning: no templates/camera poses; recovery starts with online bank only")
        return FeaturePoseBank("offline"), None
    raise FileNotFoundError(
        "RRTrack offline recovery bank is missing and could not be built; "
        f"expected templates at {templates_dir} and camera poses at {cam_poses}. "
        "Provide --offline-bank or explicitly use --allow-online-bank-only."
    )


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _live_frames(args: argparse.Namespace, start_index: int):
    foundationpose_root = Path(args.foundationpose_root).expanduser().resolve()
    if str(foundationpose_root) not in sys.path:
        sys.path.insert(0, str(foundationpose_root))
    module = _load_module("rm75_rrtrack_realsense", foundationpose_root / "run_realtime_demo.py")
    reader = module.RealSenseRGBDReader(width=args.camera_width, height=args.camera_height, fps=args.camera_fps)
    if args.camera_serial:
        reader.config.enable_device(args.camera_serial)
    reader.start()
    try:
        for _ in range(max(0, args.warmup_frames)):
            reader.get_frame()
        index = int(start_index)
        while args.max_frames <= 0 or index <= args.max_frames:
            item = reader.get_frame()
            if item is None:
                continue
            yield FrameObservation(item["color"], item["depth"], item["K"], index, time.time())
            index += 1
    finally:
        reader.stop()


def _sequence_frames(args: argparse.Namespace, start_index: int):
    root = Path(args.sequence_dir).expanduser()
    camera_path = Path(args.camera_json).expanduser() if args.camera_json else root / "camera.json"
    K = _load_camera(camera_path)
    rgb_files = sorted((root / "rgb").glob("*"))
    if not rgb_files:
        rgb_files = sorted(root.glob("rgb_*.*"))
    if not rgb_files:
        raise FileNotFoundError(f"no RGB frames under {root}")
    for offset, rgb_path in enumerate(rgb_files):
        if args.max_frames > 0 and offset >= args.max_frames:
            break
        candidates = [
            root / "depth" / f"{rgb_path.stem}.npy",
            root / "depth" / f"{rgb_path.stem}.png",
            root / rgb_path.name.replace("rgb_", "depth_").replace(rgb_path.suffix, ".npy"),
            root / rgb_path.name.replace("rgb_", "depth_").replace(rgb_path.suffix, ".png"),
        ]
        depth_path = next((path for path in candidates if path.exists()), None)
        if depth_path is None:
            raise FileNotFoundError(f"no depth frame corresponding to {rgb_path}")
        yield FrameObservation(
            _load_rgb(rgb_path),
            _load_depth(depth_path, args.sequence_depth_scale),
            K,
            start_index + offset,
            None,
        )


def _preview(output, rgb: np.ndarray) -> bool:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.asarray(output.mask, dtype=bool)
    rendered = None if output.rendered_mask is None else np.asarray(output.rendered_mask, dtype=bool)
    canvas[mask] = (0.55 * canvas[mask] + 0.45 * np.asarray([0, 255, 0])).astype(np.uint8)
    if rendered is not None:
        contours, _ = cv2.findContours(rendered.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 0, 255), 2)
    label = (
        f"{output.state.value} {output.event} "
        f"P={output.agreement.precision:.2f} S={output.agreement.support:.2f} H={output.agreement.entropy:.2f}"
    )
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("RM75 RRTrack", canvas)
    return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RRTrack-inspired recoverable 2D--6D perception entry")
    parser.add_argument("--object-name", required=True)
    parser.add_argument(
        "--scene-object-names",
        nargs="+",
        help="Initialize all listed table objects once, then hand the selected --object-name instance to RRTrack",
    )
    parser.add_argument("--active-instance-index", type=int, default=0)
    parser.add_argument("--mesh-file")
    parser.add_argument("--mesh-scale", type=float)
    parser.add_argument("--init-result-json", help="Existing SAM6D result; omitted means live SAM3+SAM6D initialization")
    parser.add_argument("--sequence-dir", help="Offline rgb/depth sequence; omitted means live RealSense")
    parser.add_argument("--camera-json")
    parser.add_argument("--sequence-depth-scale", type=float, default=1.0, help="Use 0.001 for uint16 millimeter NPY/PNG sequences")
    parser.add_argument("--foundationpose-root", default=DEFAULT_FOUNDATIONPOSE_ROOT)
    parser.add_argument("--cutie-root", default=DEFAULT_CUTIE_ROOT)
    parser.add_argument("--cutie-weights", help="Path to cutie-base-mega.pth; defaults to CUTIE_ROOT/weights")
    parser.add_argument("--dinov2-root", default=DEFAULT_DINOV2_ROOT)
    parser.add_argument("--cutie-device", default="cuda")
    parser.add_argument(
        "--cutie-max-internal-size",
        type=int,
        default=120,
        help="CUTIE shortest-side resolution; 120 is 1/4 of the 640x480 RM75 stream",
    )
    parser.add_argument("--disable-dino-recovery", action="store_true")
    parser.add_argument("--dino-model-name", default="dinov2_vits14")
    parser.add_argument("--offline-bank")
    parser.add_argument("--offline-bank-cache-root", default=str(RUNTIME_DIR / "rrtrack_banks"))
    parser.add_argument("--cam-poses", default=DEFAULT_CAM_POSES)
    parser.add_argument("--offline-bank-base-views", type=int, default=128)
    parser.add_argument("--allow-online-bank-only", action="store_true")
    parser.add_argument("--disable-sam3-recovery", action="store_true")
    parser.add_argument("--sam3-python", default=DEFAULT_SAM3_PYTHON)
    parser.add_argument("--sam3-checkpoint", default=DEFAULT_SAM3_CHECKPOINT)
    parser.add_argument("--sam3-recovery-device", default="cuda")
    parser.add_argument("--sam3-recovery-confidence", type=float, default=0.30)
    parser.add_argument("--sam3-recovery-resolution", type=int, default=640)
    parser.add_argument("--sam3-recovery-max-candidates", type=int, default=3)
    parser.add_argument("--sam3-recovery-timeout", type=float, default=180.0)
    parser.add_argument("--rrtrack-config", help="JSON overrides for RRTrackConfig thresholds and state policy")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-serial")
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=0, help="0 tracks until q/Ctrl-C")
    parser.add_argument("--refine-iterations", type=int, default=2)
    parser.add_argument("--register-iterations", type=int, default=5)
    parser.add_argument("--output-root", default=str(RUNTIME_DIR / "rrtrack_runs"))
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sam3_checkpoint and not Path(args.sam3_checkpoint).expanduser().is_file():
        print(f"[rrtrack] SAM3 checkpoint not found; falling back to Hugging Face loading: {args.sam3_checkpoint}")
        args.sam3_checkpoint = ""
    if args.active_instance_index < 0:
        raise ValueError("--active-instance-index must be non-negative")
    if args.scene_object_names:
        wanted = normalize_object_name(args.object_name)
        if not any(normalize_object_name(name) == wanted for name in args.scene_object_names):
            args.scene_object_names.append(args.object_name)
    config_overrides = _load_json(Path(args.rrtrack_config).expanduser()) if args.rrtrack_config else {}
    config = RRTrackConfig(**config_overrides)
    result_path = Path(args.init_result_json).expanduser() if args.init_result_json else _run_sam6d_initialization(args)
    initialization_payload = _load_json(result_path)
    initial_frame, initial_mask, initial_pose, initialization_result = _load_initial_bundle(
        result_path, args.object_name, args.active_instance_index
    )
    mesh = _load_mesh(args)

    segmenter = CutieMaskTracker(
        cutie_root=args.cutie_root,
        weights_path=args.cutie_weights,
        device=args.cutie_device,
        max_internal_size=args.cutie_max_internal_size,
        max_long_anchors=config.max_long_anchors,
    )
    pose_refiner = FoundationPoseRefiner(
        mesh,
        foundationpose_root=args.foundationpose_root,
        refine_iterations=args.refine_iterations,
        register_iterations=args.register_iterations,
    )
    descriptor = None
    if not args.disable_dino_recovery:
        dino_root = Path(args.dinov2_root).expanduser()
        descriptor = DinoV2Descriptor(
            model_name=args.dino_model_name,
            repo_or_dir=str(dino_root) if dino_root.exists() else "facebookresearch/dinov2",
            source="local" if dino_root.exists() else "github",
            input_size=224,
            context_padding=0.15,
        )
    offline_bank, offline_bank_path = _prepare_offline_bank(args, descriptor, initialization_result)
    relocalizer = None
    if not args.disable_sam3_recovery:
        spec = get_object_spec(args.object_name)
        prompt = spec.grounding_prompt if spec is not None else args.object_name
        relocalizer = SAM3TextRelocalizer(
            prompt=prompt,
            output_root=Path(args.output_root).expanduser() / "sam3_recovery" / time.strftime("%Y%m%d_%H%M%S"),
            python_executable=args.sam3_python,
            provider_script=APP_ROOT / "rm75_app" / "perception" / "sam3_mask_provider.py",
            checkpoint_path=args.sam3_checkpoint,
            device=args.sam3_recovery_device,
            confidence=args.sam3_recovery_confidence,
            resolution=args.sam3_recovery_resolution,
            max_candidates=args.sam3_recovery_max_candidates,
            timeout_s=args.sam3_recovery_timeout,
        )
    tracker = RRTracker(
        segmenter=segmenter,
        pose_refiner=pose_refiner,
        renderer=TriangleMeshMaskRenderer(mesh.vertices, mesh.faces),
        config=config,
        descriptor=descriptor,
        offline_bank=offline_bank,
        relocalize_mask=relocalizer,
    )

    run_dir = Path(args.output_root).expanduser() / time.strftime("%Y%m%d_%H%M%S_rrtrack")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "trajectory.jsonl"
    latest_path = run_dir / "latest_pose.json"
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "rrtrack": dataclasses.asdict(config),
                "initial_result_json": str(result_path),
                "offline_bank": str(offline_bank_path) if offline_bank_path else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    scene_registry = build_scene_registry(initialization_payload, args.object_name, args.active_instance_index)
    (run_dir / "scene_registry.json").write_text(
        json.dumps(scene_registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[rrtrack] output={run_dir}")

    first = tracker.initialize(initial_frame, initial_mask, initial_pose)
    with log_path.open("w", encoding="utf-8") as log:
        first_payload = first.jsonable()
        log.write(json.dumps(first_payload, ensure_ascii=False) + "\n")
        latest_path.write_text(json.dumps(first_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        frames = _sequence_frames(args, 1) if args.sequence_dir else _live_frames(args, 1)
        try:
            for frame in frames:
                output = tracker.step(frame)
                payload = output.jsonable()
                log.write(json.dumps(payload, ensure_ascii=False) + "\n")
                log.flush()
                latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                print(
                    f"[rrtrack] frame={output.frame_index} state={output.state.value} event={output.event} "
                    f"P={output.agreement.precision:.3f} S={output.agreement.support:.3f} "
                    f"memory={','.join(output.memory_updates) or '-'}"
                )
                if not args.no_preview and not _preview(output, frame.rgb):
                    break
        except KeyboardInterrupt:
            print("[rrtrack] interrupted")
        finally:
            cv2.destroyAllWindows()
    print(f"[rrtrack] trajectory={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
