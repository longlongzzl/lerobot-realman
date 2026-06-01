#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CAMERA_EXTRINSIC = (
    REPO_ROOT / "rm75_pick_place_app" / "assets" / "tingzi_calibration" / "camera_extrinsic_opencv.npy"
)
DEFAULT_OUT_DIR = SCRIPT_DIR / "jimu_portable_repro" / "scenes" / "random"


def _base_plate_rotation() -> np.ndarray:
    # Jimu plate convention: local Y is the thin axis. For a floor plate,
    # local Y points up and local X/Z span the table plane.
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


def _tray_rotation() -> np.ndarray:
    return np.eye(3, dtype=np.float32)


def _transform(rotation: np.ndarray, translation: tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    T[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(3)
    return T


def _result_item(name: str, prompt: str, role: str, index: int, T_base_cam: np.ndarray, T_base_obj: np.ndarray) -> dict[str, Any]:
    T_cam_obj = (T_base_cam @ T_base_obj).astype(np.float32)
    return {
        "object_name": name,
        "prompt": prompt,
        "run_dir": f"synthetic_random_assembly_anchors/{role}",
        "sam3_instance_index": int(index),
        "score": 1.0,
        "mask_source": "synthetic_random_assembly_anchors",
        "mask_pixels": 0,
        "detection_box_xyxy": [0.0, 0.0, 0.0, 0.0],
        "ok": True,
        "mask_elapsed_ms": 0.0,
        "pem_elapsed_ms": 0.0,
        "T_cam_obj_raw_pem": T_cam_obj.astype(float).tolist(),
        "translation_m_raw_pem": T_cam_obj[:3, 3].astype(float).tolist(),
        "T_cam_obj": T_cam_obj.astype(float).tolist(),
        "translation_m": T_cam_obj[:3, 3].astype(float).tolist(),
        "pem_detection": {
            "synthetic_anchor": role,
            "T_base_obj": T_base_obj.astype(float).tolist(),
        },
    }


def _sample_range(rng: random.Random, values: list[float]) -> float:
    if len(values) != 2:
        raise ValueError("range values must contain exactly two floats")
    lo, hi = float(values[0]), float(values[1])
    return rng.uniform(min(lo, hi), max(lo, hi))


def build_scene(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(int(args.seed))
    T_base_cam = np.load(Path(args.camera_extrinsic_opencv_path).expanduser()).astype(np.float32)

    base_xy = (
        _sample_range(rng, args.base_x_range),
        _sample_range(rng, args.base_y_range),
    )
    tray_xy = (
        _sample_range(rng, args.tray_x_range),
        _sample_range(rng, args.tray_y_range),
    )

    T_base_assembly = _transform(
        _base_plate_rotation(),
        (base_xy[0], base_xy[1], float(args.base_z)),
    )
    T_base_tray = _transform(
        _tray_rotation(),
        (tray_xy[0], tray_xy[1], float(args.tray_z)),
    )

    results = [
        _result_item(
            "jimu_base_assembly",
            "red cross shaped base assembly made of five square plastic building plates.",
            "base_assembly",
            0,
            T_base_cam,
            T_base_assembly,
        ),
        _result_item(
            "jimu_liaoban",
            "gray plastic tray with slots for red building plates.",
            "tray",
            0,
            T_base_cam,
            T_base_tray,
        ),
    ]
    return {
        "scene_dir": "synthetic_random_assembly_anchors",
        "object_count": len(results),
        "ok_count": len(results),
        "results": results,
        "synthetic_fixed_scene": True,
        "jimu_assembly_anchor_scene": True,
        "random_seed": int(args.seed),
        "random_initial_pose": {
            "base_xy": [float(base_xy[0]), float(base_xy[1])],
            "tray_xy": [float(tray_xy[0]), float(tray_xy[1])],
            "base_z": float(args.base_z),
            "tray_z": float(args.tray_z),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a random two-anchor Jimu fixed scene.")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--camera-extrinsic-opencv-path", type=str, default=str(DEFAULT_CAMERA_EXTRINSIC))
    parser.add_argument("--base-x-range", type=float, nargs=2, default=[-0.245, -0.150])
    parser.add_argument("--base-y-range", type=float, nargs=2, default=[0.035, 0.145])
    # The tray slots are expanded along base X, which maps to the positive-Y
    # reach direction in the Jimu simulation convention.  Keeping the anchor in
    # this band leaves all four first-layer pickup slots inside the RM75 IK
    # envelope instead of letting the fourth slot drift just out of reach.
    parser.add_argument("--tray-x-range", type=float, nargs=2, default=[0.065, 0.085])
    parser.add_argument("--tray-y-range", type=float, nargs=2, default=[-0.405, -0.315])
    parser.add_argument("--base-z", type=float, default=0.0066)
    parser.add_argument("--tray-z", type=float, default=0.0050)
    args = parser.parse_args()
    if args.seed is None:
        args.seed = int(datetime.now().strftime("%H%M%S"))
    if not args.out:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        args.out = str(DEFAULT_OUT_DIR / f"jimu_assembly_random_seed{int(args.seed)}.json")
    return args


def main() -> None:
    args = parse_args()
    scene = build_scene(args)
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(scene, f, indent=2)
    print(out_path)
    print(json.dumps(scene["random_initial_pose"], indent=2))


if __name__ == "__main__":
    main()
