#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


BETA_DIR = Path(__file__).resolve().parent
REPO_ROOT = BETA_DIR.parent
DEFAULT_CAMERA_EXTRINSIC = str(
    REPO_ROOT / "rm75_pick_place_app" / "assets" / "tingzi_calibration" / "camera_extrinsic_opencv.npy"
)
DEFAULT_OUT = BETA_DIR / "fixed_scenes" / "jimu_9objects_default" / "full_scene_pose_results.json"
DEFAULT_EXTENTS_M = np.asarray([0.100058, 0.013194, 0.088722], dtype=np.float32)
JIMU_PROVIDER_OBJECT_NAME = "red_bricks_cube"
ROLE_ORDER = (
    "floor",
    "right_wall",
    "back_wall",
    "left_wall",
    "front_wall",
    "right_second_wall",
    "back_second_wall",
    "left_second_wall",
    "front_second_wall",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        raise ValueError(f"cannot normalize degenerate vector: {arr}")
    return (arr / norm).astype(np.float32)


def _rotation_from_axes(x_axis: tuple[float, float, float], y_axis: tuple[float, float, float], z_axis: tuple[float, float, float]) -> np.ndarray:
    rot = np.column_stack([
        _normalize(np.asarray(x_axis, dtype=np.float32)),
        _normalize(np.asarray(y_axis, dtype=np.float32)),
        _normalize(np.asarray(z_axis, dtype=np.float32)),
    ]).astype(np.float32)
    if float(np.linalg.det(rot)) < 0.0:
        rot[:, 0] *= -1.0
    return rot


def _pose_matrix(position: tuple[float, float, float], rotation: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    mat[:3, 3] = np.asarray(position, dtype=np.float32).reshape(3)
    return mat


def _base_poses(args: argparse.Namespace) -> dict[str, np.ndarray]:
    extents = DEFAULT_EXTENTS_M
    half_thick = float(extents[1] * 0.5)
    half_height = float(extents[2] * 0.5)

    floor_rot = _rotation_from_axes(
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, -1.0, 0.0),
    )
    wall_rot = _rotation_from_axes(
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    poses: dict[str, np.ndarray] = {
        "floor": _pose_matrix(
            (float(args.floor_x), float(args.floor_y), float(args.floor_z) + half_thick),
            floor_rot,
        )
    }

    stage_x0 = float(args.stage_x_start)
    stage_dx = float(args.stage_x_step)
    z = float(args.stage_z) + half_height
    first_y = float(args.first_row_y)
    second_y = float(args.second_row_y)
    for row_index, role in enumerate(ROLE_ORDER[1:5]):
        poses[role] = _pose_matrix((stage_x0 + row_index * stage_dx, first_y, z), wall_rot)
    for row_index, role in enumerate(ROLE_ORDER[5:]):
        poses[role] = _pose_matrix((stage_x0 + row_index * stage_dx, second_y, z), wall_rot)
    return poses


def _make_result_item(role: str, index: int, T_cam_obj: np.ndarray, T_base_obj: np.ndarray) -> dict[str, Any]:
    return {
        "object_name": JIMU_PROVIDER_OBJECT_NAME,
        "prompt": "small square plastic building block.",
        "run_dir": f"synthetic_default_scene/{role}",
        "sam3_instance_index": int(index),
        "score": 1.0,
        "mask_source": "synthetic_default_scene",
        "mask_pixels": 0,
        "depth_repair": {"applied": False, "reason": "synthetic"},
        "detection_box_xyxy": [0.0, 0.0, 0.0, 0.0],
        "detection_ism": None,
        "ok": True,
        "mask_elapsed_ms": 0.0,
        "pem_elapsed_ms": 0.0,
        "pem_batch_instance_count": len(ROLE_ORDER),
        "T_cam_obj_raw_pem": T_cam_obj,
        "translation_m_raw_pem": T_cam_obj[:3, 3],
        "T_cam_obj": T_cam_obj,
        "translation_m": T_cam_obj[:3, 3],
        "pem_refine": {"applied": False, "reason": "synthetic"},
        "pem_detection": {"synthetic_role": role, "T_base_obj": T_base_obj},
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    T_base_cam = np.load(Path(args.camera_extrinsic_opencv_path).expanduser()).astype(np.float32).reshape(4, 4)
    base_poses = _base_poses(args)
    results = []
    for index, role in enumerate(ROLE_ORDER):
        T_base_obj = base_poses[role]
        T_cam_obj = (T_base_cam @ T_base_obj).astype(np.float32)
        results.append(_make_result_item(role, index, T_cam_obj, T_base_obj))
    return {
        "scene_dir": str(Path(args.out).expanduser().parent),
        "object_count": len(results),
        "ok_count": len(results),
        "results": results,
        "full_scene_pem_visualization": {
            "rendered": len(results),
            "synthetic": True,
            "role_order": list(ROLE_ORDER),
        },
        "synthetic_fixed_scene": {
            "type": "jimu_9objects_default",
            "camera_extrinsic_opencv_path": str(Path(args.camera_extrinsic_opencv_path).expanduser()),
            "role_order": list(ROLE_ORDER),
            "base_poses": {role: mat for role, mat in base_poses.items()},
            "stage_rows": {
                "first_row_y": float(args.first_row_y),
                "second_row_y": float(args.second_row_y),
                "stage_x_start": float(args.stage_x_start),
                "stage_x_step": float(args.stage_x_step),
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic 9-object Jimu SAM6D fixed-scene JSON.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--camera-extrinsic-opencv-path", default=DEFAULT_CAMERA_EXTRINSIC)
    parser.add_argument("--floor-x", type=float, default=-0.20)
    parser.add_argument("--floor-y", type=float, default=0.10)
    parser.add_argument("--floor-z", type=float, default=0.0)
    parser.add_argument("--stage-x-start", type=float, default=0.18)
    parser.add_argument("--stage-x-step", type=float, default=-0.12)
    parser.add_argument("--first-row-y", type=float, default=-0.30)
    parser.add_argument("--second-row-y", type=float, default=-0.17)
    parser.add_argument("--stage-z", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "object_count": payload["object_count"], "roles": list(ROLE_ORDER)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
