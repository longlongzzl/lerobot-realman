#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import rm75_jiaobang_pick_real_with_foundationpose as base


DEFAULT_CAPTURE_OBJECTS = [
    "gluestick",
    "hongshupian",
    "shuazi",
    "lvmukuai",
    "tennis",
    "carriot",
    "bi",
    "desk",
    "bitong",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = base.build_arg_parser()
    parser.description = (
        "Capture the current tabletop with FoundationPose and export a fixed-scene JSON "
        "usable with --skip-foundationpose --fixed-scene-pose-file."
    )
    parser.set_defaults(render_mode="none")
    parser.add_argument(
        "--objects",
        nargs="*",
        default=list(DEFAULT_CAPTURE_OBJECTS),
        help=(
            "Object spec keys to capture. The first object, or --capture-target if set, "
            "is registered as the active target; all others are registered as scene obstacles."
        ),
    )
    parser.add_argument(
        "--capture-target",
        type=str,
        default=None,
        help="Object spec key to use as the active FoundationPose target. Defaults to the first --objects entry.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=THIS_DIR / "test_scenes" / "fixed_scene_capture.json",
        help="Output fixed-scene JSON path.",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default=None,
        help="Scene name written to the JSON. Defaults to the output file stem.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write the JSON even if some non-target requested objects were not captured.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser


def _matrix_to_list(T: np.ndarray) -> list[list[float]]:
    return np.asarray(T, dtype=np.float32).reshape(4, 4).tolist()


def _jsonable_float(value, default: float = 1.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _collect_fixed_scene_objects(demo, args, object_names: list[str]) -> tuple[dict, list[str]]:
    requested = list(dict.fromkeys(object_names))
    requested_set = set(requested)
    objects: dict[str, dict] = {}

    target_name = base.normalize_object_name(getattr(args, "object_name", None))
    if target_name in requested_set:
        T_world_obj = base.get_current_active_object_world_pose_matrix(demo)
        if T_world_obj is not None:
            objects[target_name] = {
                "label": str(getattr(args, "target_object_name", "") or target_name),
                "score": 1.0,
                "placed": False,
                "T_world_obj": _matrix_to_list(T_world_obj),
            }

    for item in list(getattr(demo, "scene_obstacles", []) or []):
        object_name = base.normalize_object_name(item.get("object_name"))
        if object_name is None or object_name not in requested_set:
            continue
        T_world_obj = item.get("T_world_obj")
        if T_world_obj is None:
            continue
        objects[object_name] = {
            "label": str(item.get("label", object_name)),
            "score": _jsonable_float(item.get("score", 1.0)),
            "placed": bool(item.get("placed", False)),
            "T_world_obj": _matrix_to_list(T_world_obj),
        }

    ordered_objects = {name: objects[name] for name in requested if name in objects}
    missing = [name for name in requested if name not in ordered_objects]
    return ordered_objects, missing


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def main() -> None:
    args = parse_args()
    base.maybe_print_and_exit_object_specs(args)

    object_names = base.resolve_object_spec_name_list(args.objects or [])
    if not object_names:
        raise ValueError("--objects must include at least one object spec key")

    capture_target = base.resolve_object_spec_name(args.capture_target) if args.capture_target else object_names[0]
    if capture_target not in object_names:
        object_names = [capture_target] + object_names

    output = Path(args.output).expanduser().resolve()
    if output.exists() and not bool(args.overwrite):
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")

    base_args = argparse.Namespace(**vars(args).copy())
    base_args.object_name = capture_target
    base_args.skip_foundationpose = False
    base_args.execute_real = False
    base_args.render_mode = str(getattr(args, "render_mode", "none") or "none")
    base_args.selected_obstacle_object_names = [name for name in object_names if name != capture_target]
    base_args.required_scene_object_names = [] if bool(args.allow_missing) else list(base_args.selected_obstacle_object_names)

    cycle_args, spec = base.make_cycle_args(base_args, capture_target)
    cycle_args.selected_obstacle_object_names = list(base_args.selected_obstacle_object_names)
    cycle_args.required_scene_object_names = list(base_args.required_scene_object_names)

    bridge_mod = base.load_module_from_path("jiaobang_fp_bridge_capture_fixed_scene", cycle_args.bridge_script_path)
    planner_mod = base.load_module_from_path("jiaobang_planner_impl_capture_fixed_scene", cycle_args.pick_script_path)

    print(f"[capture] target object: {capture_target}")
    print(f"[capture] target prompt: {cycle_args.target_object_name}")
    print(f"[capture] obstacle objects: {cycle_args.selected_obstacle_object_names}")
    print(f"[capture] output: {output}")
    print(f"[capture] render_mode: {cycle_args.render_mode}")

    env = None
    scene_capture_cache: dict = {}
    try:
        env, demo = base.create_demo(
            cycle_args,
            bridge_mod,
            planner_mod,
            scene_capture_cache=scene_capture_cache,
        )
        objects, missing = _collect_fixed_scene_objects(demo, cycle_args, object_names)
        if missing and not bool(args.allow_missing):
            raise RuntimeError(
                "Missing requested object pose(s): "
                + ", ".join(missing)
                + ". Re-run and fix the GroundingDINO boxes, or pass --allow-missing."
            )

        payload = {
            "version": 1,
            "scene_name": str(args.scene_name or output.stem),
            "source": "foundationpose_capture",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "objects": objects,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"[capture] wrote {len(objects)} object pose(s) to {output}")
        if missing:
            print(f"[capture] missing object pose(s): {missing}")
    finally:
        base.close_env_quietly(env)
        gc.collect()


if __name__ == "__main__":
    main()
