#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

import rm75_jiaobang_pick_real_with_foundationpose as base
from object_specs import list_object_spec_names, normalize_object_name


def _matrix_to_json(matrix: np.ndarray | None):
    if matrix is None:
        raise ValueError("Cannot serialize None matrix")
    arr = np.asarray(matrix, dtype=np.float32).reshape(4, 4)
    return arr.tolist()


def _build_scene_payload(demo, args, obstacle_names: list[str]) -> dict:
    target_name = normalize_object_name(getattr(args, "object_name", None))
    if target_name is None:
        raise ValueError("target object-name is missing")

    objects: dict[str, dict] = {}

    target_pose = base.get_current_active_object_world_pose_matrix(demo)
    if target_pose is None:
        raise RuntimeError("Failed to read target object pose after registration")
    objects[target_name] = {
        "label": str(getattr(args, "target_object_name", target_name) or target_name),
        "score": 1.0,
        "placed": False,
        "T_world_obj": _matrix_to_json(target_pose),
    }

    for item in list(getattr(demo, "scene_obstacles", []) or []):
        object_name = normalize_object_name(item.get("object_name"))
        if object_name is None:
            continue
        if object_name == target_name:
            continue
        if object_name.startswith("virtual_"):
            continue
        pose = item.get("T_world_obj")
        if pose is None:
            continue
        objects[object_name] = {
            "label": str(item.get("label", object_name)),
            "score": float(item.get("score", 0.0)),
            "placed": bool(item.get("placed", False)),
            "T_world_obj": _matrix_to_json(np.asarray(pose, dtype=np.float32)),
        }

    missing = [name for name in obstacle_names if name not in objects]
    if missing:
        print(f"[warn] skipped uncaptured obstacles: {missing}")

    return {
        "version": 1,
        "scene_name": str(getattr(args, "scene_name", "foundationpose_capture")),
        "source": "foundationpose_capture",
        "objects": objects,
    }


def parse_args() -> argparse.Namespace:
    parser = base.build_arg_parser()
    parser.add_argument(
        "--output-scene-json",
        type=Path,
        required=True,
        help="Path to write captured fixed-scene json.",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default="foundationpose_capture",
        help="Scene name stored in json payload.",
    )
    parser.add_argument(
        "--scene-obstacle-names",
        type=str,
        nargs="*",
        default=None,
        help="Optional obstacle names to capture. If omitted, capture all known object_specs except target.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_name = normalize_object_name(args.object_name)
    if target_name is None:
        raise SystemExit("--object-name is required")

    if args.scene_obstacle_names is None:
        obstacle_names = [
            normalize_object_name(name)
            for name in list_object_spec_names()
            if normalize_object_name(name) != target_name
        ]
    else:
        obstacle_names = [
            name
            for name in base.resolve_object_spec_name_list(
                args.scene_obstacle_names,
                available_names=list_object_spec_names(),
            )
            if name != target_name
        ]

    args.selected_obstacle_object_names = obstacle_names
    args.scene_name = args.scene_name

    base_args = argparse.Namespace(**vars(args).copy())
    base_args.object_name = target_name
    cycle_args, _spec = base.make_cycle_args(base_args, target_name)

    bridge_mod = base.load_module_from_path("jiaobang_fp_bridge", args.bridge_script_path)
    planner_mod = base.load_module_from_path("jiaobang_planner_impl", args.pick_script_path)

    env = None
    demo = None
    try:
        env, demo = base.create_demo(cycle_args, bridge_mod, planner_mod, scene_capture_cache=None)
        payload = _build_scene_payload(demo, cycle_args, obstacle_names)
        output = args.output_scene_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"[done] scene saved -> {output}")
        print(f"[done] captured objects: {sorted(payload['objects'].keys())}")
    finally:
        if demo is not None:
            try:
                base.close_env_quietly(env)
            except Exception:
                pass
        gc.collect()


if __name__ == "__main__":
    main()
