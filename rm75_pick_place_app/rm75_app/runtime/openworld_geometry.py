from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rm75_app.perception.openworld_geometry import DynamicGeometryConfig, DynamicGeometrySession, GeometryFrame
from rm75_app.perception.openworld_geometry.providers import ObservedSurfaceProvider, RaySt3RProvider


def _first_existing(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"none of {names} exists under {root}")


def _load_transform(value: Any, *, base_dir: Path) -> np.ndarray:
    if value is None:
        return np.eye(4, dtype=np.float64)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float64).reshape(4, 4)
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float64).reshape(4, 4)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("T_base_camera", payload.get("transform", payload.get("matrix")))
    return np.asarray(payload, dtype=np.float64).reshape(4, 4)


def _camera_info(frame_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    path = frame_dir / "camera.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    K = payload.get("cam_K", payload.get("K", payload.get("intrinsic")))
    if K is None:
        raise ValueError(f"camera JSON has no cam_K/K/intrinsic: {path}")
    return np.asarray(K, dtype=np.float64).reshape(3, 3), payload


def _load_depth(path: Path, camera: dict[str, Any], explicit_scale: float | None) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        raw = np.load(path)
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    raw_is_float = np.issubdtype(np.asarray(raw).dtype, np.floating)
    raw = np.asarray(raw, dtype=np.float32)
    if explicit_scale is not None:
        return raw * float(explicit_scale)
    positive = raw[np.isfinite(raw) & (raw > 0)]
    if raw_is_float and positive.size and float(np.nanmedian(positive)) < 20.0:
        return raw
    scale = float(camera.get("depth_scale", 1.0))
    return raw / scale if scale > 100.0 else raw * scale / 1000.0


def load_frame(frame_dir: str | Path, *, index: int, depth_scale: float | None = None) -> GeometryFrame:
    root = Path(frame_dir).expanduser().resolve()
    rgb_path = _first_existing(root, ("rgb.png", "color.png", "rgb.jpg", "color.jpg"))
    depth_path = _first_existing(root, ("depth.npy", "depth.png", "depth.tiff"))
    mask_path = _first_existing(root, ("mask.png", "sam6d_binary_mask.png", "object_mask.png"))
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if rgb_bgr is None or mask is None:
        raise FileNotFoundError(f"failed to load RGB/mask under {root}")
    K, camera = _camera_info(root)
    return GeometryFrame(
        rgb=cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
        depth_m=_load_depth(depth_path, camera, depth_scale),
        mask=mask > 0,
        K=K,
        frame_index=index,
        source=str(root),
    )


def _load_config(path: str | None) -> DynamicGeometryConfig:
    if not path:
        return DynamicGeometryConfig()
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    known = set(asdict(DynamicGeometryConfig()))
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"unknown dynamic geometry config keys: {unknown}")
    return DynamicGeometryConfig(**payload)


def _updates(args: argparse.Namespace) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if args.updates_manifest:
        manifest_path = Path(args.updates_manifest).expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_entries = payload.get("frames", payload) if isinstance(payload, dict) else payload
        for item in raw_entries:
            entry = dict(item) if isinstance(item, dict) else {"frame_dir": item}
            entry["_base_dir"] = manifest_path.parent
            entries.append(entry)
    if args.updates_dir:
        root = Path(args.updates_dir).expanduser().resolve()
        entries.extend({"frame_dir": path, "_base_dir": root} for path in sorted(root.iterdir()) if path.is_dir())
    return entries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and incrementally update an unseen rigid object's visual/collision geometry"
    )
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--initial-frame-dir", required=True)
    parser.add_argument("--initial-T-base-camera", help="4x4 .npy/.json; identity when omitted")
    parser.add_argument("--updates-manifest", help="JSON frame list with frame_dir and T_base_camera")
    parser.add_argument("--updates-dir", help="Sorted update frame folders; each may contain T_base_camera.npy")
    parser.add_argument("--depth-scale", type=float, default=None, help="Raw depth to metres multiplier")
    parser.add_argument("--provider", choices=("rayst3r", "observed"), default="rayst3r")
    parser.add_argument("--rayst3r-root", default="runtime_data/third_party/RaySt3R")
    parser.add_argument("--rayst3r-python", default=sys.executable)
    parser.add_argument("--rayst3r-confidence", type=float, default=5.0)
    parser.add_argument("--rayst3r-views-per-axis", type=int, default=3)
    parser.add_argument("--output-root", default="runtime_data/openworld_geometry_runs")
    parser.add_argument("--config", help="DynamicGeometryConfig JSON overrides")
    parser.add_argument("--trust-camera-poses", action="store_true", help="Skip ICP; intended for calibrated replay/debug")
    parser.add_argument("--force-remesh-every-update", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    provider = (
        RaySt3RProvider(
            root=args.rayst3r_root,
            python_executable=args.rayst3r_python,
            confidence_threshold=args.rayst3r_confidence,
            predicted_views_per_axis=args.rayst3r_views_per_axis,
            voxel_size_m=config.voxel_size_m,
        )
        if args.provider == "rayst3r"
        else ObservedSurfaceProvider(config.voxel_size_m)
    )
    initial_root = Path(args.initial_frame_dir).expanduser().resolve()
    session = DynamicGeometrySession(
        instance_id=args.instance_id,
        output_root=args.output_root,
        provider=provider,
        config=config,
    )
    initial_pose = _load_transform(args.initial_T_base_camera, base_dir=initial_root)
    snapshot = session.initialize(
        load_frame(initial_root, index=0, depth_scale=args.depth_scale),
        T_base_camera=initial_pose,
    )
    print(f"[openworld] initialized v{snapshot.version}: {snapshot.collision_mesh_path}")

    for index, entry in enumerate(_updates(args), start=1):
        base_dir = Path(entry.pop("_base_dir"))
        frame_dir = Path(str(entry["frame_dir"])).expanduser()
        if not frame_dir.is_absolute():
            frame_dir = base_dir / frame_dir
        transform_value = entry.get("T_base_camera")
        if transform_value is None:
            local_npy = frame_dir / "T_base_camera.npy"
            local_json = frame_dir / "T_base_camera.json"
            transform_value = local_npy if local_npy.exists() else local_json if local_json.exists() else None
        if transform_value is None:
            raise ValueError(f"update frame requires T_base_camera: {frame_dir}")
        result = session.update(
            load_frame(frame_dir, index=index, depth_scale=args.depth_scale),
            T_base_camera=_load_transform(transform_value, base_dir=base_dir),
            predicted_T_base_model=None
            if entry.get("predicted_T_base_model") is None
            else _load_transform(entry["predicted_T_base_model"], base_dir=base_dir),
            trust_camera_pose=bool(entry.get("trust_camera_pose", args.trust_camera_poses)),
            force_remesh=bool(entry.get("force_remesh", args.force_remesh_every_update)),
        )
        print(
            f"[openworld] frame={index} accepted={result.accepted} remeshed={result.remeshed} "
            f"version={result.snapshot.version if result.snapshot else '-'} new_voxels={result.new_observed_voxels} "
            f"reason={result.reason}"
        )
    latest = session.latest_snapshot
    assert latest is not None
    print(json.dumps(latest.curobo_mesh_obstacle(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
