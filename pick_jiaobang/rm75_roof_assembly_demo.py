#!/usr/bin/env python3
"""
Template-based magnetic-roof assembly demo for RM75 + cuRobo.

Goal
----
Assume a red cube is already assembled and fixed in the scene. Four red triangle
panels start upright on the table. For each triangle object, the robot:

  1) grasps the top tip of the triangle,
  2) transports it to a roof-side target pose on top of the cube,
  3) performs a short final contact approach,
  4) opens the gripper and optionally snaps the simulated triangle to the exact
     target pose to approximate magnetic attachment.

After all four panels are placed, run the optional close-only mode to use the
opened/closing gripper as a simple clamp primitive for opposite roof panels.

Important assumptions
---------------------
This is a template assembly wrapper around the existing pick-place script. It
uses the existing cuRobo helpers for two-step grasp, attached payload transport,
short final contact, and execution. The mesh-local frames of your red_triangle.glb
may differ, so the triangle local calibration arguments usually need tuning:

  --roof-triangle-tip-local x y z
  --roof-triangle-base-center-local x y z
  --roof-triangle-local-edge-axis x y z
  --roof-triangle-local-up-axis x y z

Default convention assumes triangle local +X runs along the base edge, +Y goes
from base edge to tip in the panel plane, and +Z is the panel normal.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np

import rm75_jiaobang_pick_place_targeted as targeted
import rm75_jiaobang_pick_place_targeted_curobo as curobo_wrapper
import rm75_jiaobang_pick_place_targeted_curobo_direct_pre_place as direct


# ----------------------------- small geometry utils -----------------------------

def _np3(v, default=None) -> np.ndarray:
    if v is None:
        if default is None:
            default = [0.0, 0.0, 0.0]
        v = default
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"expected 3 values, got {arr}")
    return arr.astype(np.float32)


def _normalize(v: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32).reshape(3)
    n = float(np.linalg.norm(arr))
    if n <= 1e-8:
        raise ValueError(f"zero vector cannot be normalized: {arr}")
    return (arr / n).astype(np.float32)


def _orthonormalize(x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    x = _normalize(x_axis)
    y_raw = np.asarray(y_axis, dtype=np.float32).reshape(3)
    y = y_raw - float(np.dot(y_raw, x)) * x
    y = _normalize(y)
    z = _normalize(np.cross(x, y))
    # Recompute y so R is exactly right-handed.
    y = _normalize(np.cross(z, x))
    return np.column_stack([x, y, z]).astype(np.float32)


def _pose_to_matrix(pose) -> np.ndarray:
    return targeted.base.pose_to_matrix(
        targeted.base.flatten_np(pose.p)[:3],
        targeted.base.flatten_np(pose.q)[:4],
    ).astype(np.float32)


def _matrix_to_pose(T: np.ndarray):
    T = np.asarray(T, dtype=np.float32).reshape(4, 4)
    q = targeted.base.bridge_mod_mat2quat(T[:3, :3]).astype(np.float32)
    return targeted.Pose.create_from_pq(p=T[:3, 3].astype(np.float32), q=q)


def _pose_from_Rp(R: np.ndarray, p: np.ndarray):
    q = targeted.base.bridge_mod_mat2quat(np.asarray(R, dtype=np.float32).reshape(3, 3)).astype(np.float32)
    return targeted.Pose.create_from_pq(p=np.asarray(p, dtype=np.float32).reshape(3), q=q)


def _make_pose_offset_world(pose, offset_xyz: np.ndarray):
    p = targeted.base.flatten_np(pose.p)[:3].astype(np.float32) + np.asarray(offset_xyz, dtype=np.float32).reshape(3)
    return targeted.base.make_pose_with_position(pose, p.astype(np.float32))


def _parse_side_from_name(name: str) -> str:
    text = str(name or "").lower()
    aliases = {
        "front": ("front", "qian", "前"),
        "back": ("back", "hou", "后"),
        "left": ("left", "zuo", "左"),
        "right": ("right", "you", "右"),
    }
    for side, keys in aliases.items():
        if any(k in text for k in keys):
            return side
    raise ValueError(
        f"cannot infer roof side from object_name={name!r}. "
        "Use names containing front/back/left/right, or pass --roof-side."
    )


# ----------------------------- argument parsing -----------------------------

def build_arg_parser():
    parser = direct.build_arg_parser()
    parser.description = "RM75 cuRobo magnetic roof assembly wrapper."

    # Main task switch. If off, this script behaves like the original direct-pre-place script.
    parser.add_argument("--roof-assembly", action="store_true", default=True,
                        help="Run template roof assembly instead of the normal targeted pick-place episode.")
    parser.add_argument("--roof-close-only", action="store_true", default=False,
                        help="Skip triangle picking and run the final roof-closing clamp primitive only.")
    parser.add_argument("--roof-side", choices=["auto", "front", "back", "left", "right"], default="auto",
                        help="Roof side for the current triangle; auto infers from object_name.")

    # Mesh/object names. These are mainly used for scene filtering and debug labels.
    parser.add_argument("--roof-cube-name", type=str, default="red_bricks_cube",
                        help="Scene obstacle name of the pre-assembled cube.")
    parser.add_argument("--roof-triangle-mesh", type=str, default="red_triangle.glb",
                        help="Triangle mesh path, used only for documentation/debug; the active object still comes from normal object spec.")
    parser.add_argument("--roof-cube-mesh", type=str, default="red_bricks_cube.glb",
                        help="Cube mesh path, used only for documentation/debug.")

    # Cube / roof geometry. If --roof-cube-center is omitted, we try to read the cube scene pose.
    parser.add_argument("--roof-cube-center", type=float, nargs=3, default=None,
                        help="World xyz center of the already assembled cube. If omitted, read from scene obstacle named --roof-cube-name.")
    parser.add_argument("--roof-cube-size-m", type=float, default=0.105,
                        help="Outer cube side length in meters.")
    parser.add_argument("--roof-cube-top-z", type=float, default=None,
                        help="Override cube top z. Default: cube_center_z + cube_size/2.")
    parser.add_argument("--roof-height-m", type=float, default=0.080,
                        help="Height from cube top plane to roof apex.")
    parser.add_argument("--roof-overhang-m", type=float, default=0.000,
                        help="Outward base-edge offset for each roof triangle.")

    # Triangle local calibration. Defaults: local +X = base edge, +Y = base-to-apex, +Z = panel normal.
    parser.add_argument("--roof-triangle-tip-local", type=float, nargs=3, default=[0.0, 0.060, 0.0],
                        help="Triangle tip point in triangle local frame; grasp TCP is placed near this point.")
    parser.add_argument("--roof-triangle-base-center-local", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="Triangle base-edge center in triangle local frame; used to align to cube top edge.")
    parser.add_argument("--roof-triangle-local-edge-axis", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                        help="Triangle local axis along the base edge.")
    parser.add_argument("--roof-triangle-local-up-axis", type=float, nargs=3, default=[0.0, 1.0, 0.0],
                        help="Triangle local axis from base edge toward tip/apex in the panel plane.")

    # Grasp at triangle tip.
    parser.add_argument("--roof-tip-grasp-world-offset", type=float, nargs=3, default=[0.0, 0.0, 0.010],
                        help="World offset added to triangle tip to create the grasp TCP. Use small +z to grasp above the tip.")
    parser.add_argument("--roof-tip-grasp-roll-deg", type=float, nargs="*", default=[0.0, 90.0, 180.0, 270.0],
                        help="Roll/yaw variants around the TCP approach axis for tip grasp.")
    parser.add_argument("--roof-grasp-pregrasp-max-winners", type=int, default=2,
                        help="Number of pregrasp winners kept for triangle-tip grasp.")

    # Place/capture distances.
    parser.add_argument("--roof-preplace-outward-offset-m", type=float, default=0.055,
                        help="Move the held triangle this far outward from final pose before final approach.")
    parser.add_argument("--roof-hover-extra-z-m", type=float, default=0.030,
                        help="Extra world-z clearance for the transport hover pose.")
    parser.add_argument("--roof-release-outward-offset-m", type=float, default=0.006,
                        help="Final release/capture pose is slightly outward from exact magnetic target by this distance.")
    parser.add_argument("--roof-snap-after-release", action="store_true", default=True,
                        help="After gripper opens, snap the simulated triangle to exact target pose to approximate magnet attachment.")
    parser.add_argument("--no-roof-snap-after-release", dest="roof_snap_after_release", action="store_false",
                        help="Do not snap the simulated active triangle after release.")

    # Final roof-closing clamp.
    parser.add_argument("--roof-close-pairs", type=str, nargs="*", default=["front_back", "left_right"],
                        help="Opposite pairs to clamp after all triangles are placed: front_back and/or left_right.")
    parser.add_argument("--roof-close-pre-z-m", type=float, default=0.060,
                        help="Clamp pre-pose height above the roof apex.")
    parser.add_argument("--roof-close-z-m", type=float, default=0.020,
                        help="Clamp pose height below pre-pose. Tune so gripper contacts upper triangle areas.")
    parser.add_argument("--roof-close-gripper-open", type=float, default=None,
                        help="Optional gripper command before clamping. Default uses args.real_gripper_open.")
    parser.add_argument("--roof-close-gripper-close", type=float, default=None,
                        help="Optional gripper command during clamping. Default uses args.real_gripper_close.")

    # Bake in the roof-assembly defaults so the script is self-contained.
    _demo_dir = Path(__file__).resolve().parent
    parser.set_defaults(
        skip_foundationpose=True,
        fixed_scene_pose_file=_demo_dir / "test_scenes" / "rm75_roof_assembly_scene.json",
        fixed_scene_strict=True,
        cycle_object_names=["red_triangle_front", "red_triangle_back",
                            "red_triangle_left", "red_triangle_right"],
        tracked_scene_object_names=["red_bricks_cube"],
        render_mode="human",
        trajectory_preview_sleep=0.08,
        skip_return_to_cycle_start=True,
        skip_post_place_clearance=True,
    )
    return parser


def parse_args():
    args = build_arg_parser().parse_args()
    # Reuse the current direct script's torch extension setup if present.
    if hasattr(direct, "_configure_curobo_torch_extensions"):
        direct._configure_curobo_torch_extensions(args)
    return args


# ----------------------------- scene / target geometry -----------------------------

def _get_scene_object_pose_from_cache_or_demo(demo, scene_capture_cache, name: str) -> np.ndarray | None:
    # 1) current targeted scene helper if available
    try:
        T = targeted._get_scene_object_world_transform(demo, None, scene_capture_cache, name)
        if T is not None:
            return np.asarray(T, dtype=np.float32).reshape(4, 4)
    except Exception:
        pass

    # 2) common cache formats
    if isinstance(scene_capture_cache, dict):
        objects = scene_capture_cache.get("objects") or scene_capture_cache.get("scene_objects") or {}
        for key, value in list(objects.items()):
            if curobo_wrapper.normalize_object_name(key) != curobo_wrapper.normalize_object_name(name):
                continue
            if isinstance(value, dict):
                for k in ("T_world_obj", "T_world_object", "T_world"):
                    if k in value and value[k] is not None:
                        return np.asarray(value[k], dtype=np.float32).reshape(4, 4)
                if "pose" in value:
                    pose = np.asarray(value["pose"], dtype=np.float32).reshape(-1)
                    if pose.size >= 7:
                        return targeted.base.pose_to_matrix(pose[:3], pose[3:7]).astype(np.float32)
    return None


def _cube_center_and_top(demo, scene_capture_cache, args) -> tuple[np.ndarray, float]:
    if getattr(args, "roof_cube_center", None) is not None:
        center = _np3(args.roof_cube_center)
    else:
        T_cube = _get_scene_object_pose_from_cache_or_demo(demo, scene_capture_cache, args.roof_cube_name)
        if T_cube is None:
            raise RuntimeError(
                f"Cannot locate cube {args.roof_cube_name!r}. Pass --roof-cube-center x y z "
                "or include the cube in your fixed scene JSON."
            )
        center = T_cube[:3, 3].astype(np.float32)
    top_z = float(args.roof_cube_top_z) if args.roof_cube_top_z is not None else float(center[2] + 0.5 * float(args.roof_cube_size_m))
    return center.astype(np.float32), top_z


def _roof_side_axes(side: str) -> tuple[np.ndarray, np.ndarray]:
    # edge axis follows the top cube edge; inward points from the side toward cube center.
    if side == "front":
        return _np3([1.0, 0.0, 0.0]), _np3([0.0, 1.0, 0.0])
    if side == "back":
        return _np3([-1.0, 0.0, 0.0]), _np3([0.0, -1.0, 0.0])
    if side == "left":
        return _np3([0.0, -1.0, 0.0]), _np3([1.0, 0.0, 0.0])
    if side == "right":
        return _np3([0.0, 1.0, 0.0]), _np3([-1.0, 0.0, 0.0])
    raise ValueError(side)


def _roof_target_triangle_pose(demo, scene_capture_cache, args, side: str):
    cube_center, top_z = _cube_center_and_top(demo, scene_capture_cache, args)
    cube_size = float(args.roof_cube_size_m)
    overhang = float(args.roof_overhang_m)
    roof_h = float(args.roof_height_m)

    edge_axis_world, inward_world = _roof_side_axes(side)
    edge_axis_world = _normalize(edge_axis_world)
    inward_world = _normalize(inward_world)
    outward_world = (-inward_world).astype(np.float32)

    base_edge_center = cube_center.copy()
    base_edge_center[:2] += outward_world[:2] * (0.5 * cube_size + overhang)
    base_edge_center[2] = top_z

    # Direction in the triangle plane from base edge toward roof apex.
    panel_up_world = _normalize(inward_world * (0.5 * cube_size + overhang) + np.array([0.0, 0.0, roof_h], dtype=np.float32))
    R_world_panel = _orthonormalize(edge_axis_world, panel_up_world)

    local_edge = _normalize(_np3(args.roof_triangle_local_edge_axis))
    local_up = _normalize(_np3(args.roof_triangle_local_up_axis))
    R_obj_panel = _orthonormalize(local_edge, local_up)

    # R_world_obj maps object local axes to world. We need object local edge/up to align with panel edge/up.
    R_world_obj = (R_world_panel @ R_obj_panel.T).astype(np.float32)
    base_local = _np3(args.roof_triangle_base_center_local)
    obj_p = (base_edge_center - R_world_obj @ base_local).astype(np.float32)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_world_obj
    T[:3, 3] = obj_p
    target_pose = _matrix_to_pose(T)

    apex = cube_center.copy()
    apex[2] = top_z + roof_h
    return {
        "side": side,
        "target_T_world_obj": T,
        "target_pose": target_pose,
        "base_edge_center": base_edge_center.astype(np.float32),
        "apex": apex.astype(np.float32),
        "edge_axis_world": edge_axis_world.astype(np.float32),
        "inward_world": inward_world.astype(np.float32),
        "outward_world": outward_world.astype(np.float32),
        "panel_up_world": panel_up_world.astype(np.float32),
    }


def _release_pose_from_target_and_grasp(target_T_world_obj: np.ndarray, T_tcp_obj: np.ndarray):
    # T_tcp_obj maps object coordinates into TCP frame? Existing direct code stores inv(T_world_tcp) @ T_world_obj.
    # Therefore T_world_tcp = T_world_obj @ inv(T_tcp_obj).
    T_world_tcp = np.asarray(target_T_world_obj, dtype=np.float32).reshape(4, 4) @ np.linalg.inv(
        np.asarray(T_tcp_obj, dtype=np.float32).reshape(4, 4)
    )
    return _matrix_to_pose(T_world_tcp)


# ----------------------------- grasp / place primitives -----------------------------

def _build_roof_tip_grasp_candidates(demo, args):
    obj_p, obj_q = demo.get_obj_pose()
    T_world_obj = targeted.base.pose_to_matrix(obj_p, obj_q).astype(np.float32)
    tip_local = _np3(args.roof_triangle_tip_local)
    tip_world = (T_world_obj @ np.array([tip_local[0], tip_local[1], tip_local[2], 1.0], dtype=np.float32))[:3]
    grasp_p = (tip_world + _np3(args.roof_tip_grasp_world_offset)).astype(np.float32)

    base_pose = demo.build_topdown_grasp_pose()
    base_pose = targeted.base.make_pose_with_position(base_pose, grasp_p)

    candidates = []
    seen = set()
    for roll_deg in list(getattr(args, "roof_tip_grasp_roll_deg", [0.0]) or [0.0]):
        pose = base_pose
        if abs(float(roll_deg)) > 1e-6 and hasattr(direct, "_roll_pose_about_tcp_approach"):
            pose = direct._roll_pose_about_tcp_approach(base_pose, float(roll_deg))
        pre = demo.build_pregrasp_pose(pose)
        try:
            pose, pre, _ = targeted.base.enforce_topdown_grasp_insertion_limit(demo, args, pose, pre)
            pose, _, _ = targeted.base.enforce_min_grasp_tcp_z(pose, pre, args.min_grasp_tcp_z)
        except Exception:
            pass
        key = (
            tuple(np.round(targeted.base.flatten_np(pose.p)[:3], 5).tolist()),
            tuple(np.round(targeted.base.flatten_np(pose.q)[:4], 5).tolist()),
        )
        if key in seen:
            continue
        seen.add(key)
        label = "roof_tip_grasp" if abs(float(roll_deg)) <= 1e-6 else f"roof_tip_grasp_roll_{int(round(float(roll_deg)))}deg"
        T_world_tcp = _pose_to_matrix(pose)
        candidates.append(
            {
                "label": label,
                "pose": pose,
                "pregrasp_pose": pre,
                "T_tcp_obj": np.linalg.inv(T_world_tcp) @ T_world_obj,
                "roof_tip_world": tip_world.astype(np.float32),
                "grasp_approach_roll_deg": float(roll_deg),
            }
        )
    print(f"[roof] built {len(candidates)} triangle-tip grasp candidate(s)")
    return candidates


def _select_and_execute_roof_grasp(demo, bridge_mod, real_exec, args, planner):
    print("\n[roof] planning triangle-tip grasp")
    with direct._profile_stage(args, "roof_build_tip_grasp_candidates") as prof:
        candidates = _build_roof_tip_grasp_candidates(demo, args)
        prof["candidate_count"] = len(candidates)
        prof["success"] = bool(candidates)
        prof["status"] = "Success" if candidates else "NO_CANDIDATES"
    if not candidates:
        return None

    disabled = direct._direct_grasp_target_contact_only_disabled_links(planner) if hasattr(direct, "_direct_grasp_target_contact_only_disabled_links") else []
    max_winners = int(max(getattr(args, "roof_grasp_pregrasp_max_winners", 2), 1))
    pre_successes = direct._evaluate_two_step_grasp_candidates(
        planner,
        demo,
        args,
        np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7],
        candidates,
        label="roof_tip_two_step_grasp",
        max_winners=max_winners,
        include_active_object=True,
        disabled_world_collision_links=disabled,
    )
    if not pre_successes:
        print("[roof][FAIL] no roof tip pregrasp candidate succeeded")
        return None

    lookup = {str(x.get("label", "")): x for x in pre_successes}
    pre_successes.sort(key=direct._candidate_sort_key)
    grasp_choice = pre_successes[0]
    if hasattr(direct, "_apply_deferred_two_step_final_approach"):
        grasp_choice = direct._apply_deferred_two_step_final_approach(planner, demo, args, grasp_choice, lookup)
    if not grasp_choice.get("q_path"):
        print("[roof][FAIL] final approach did not produce q_path")
        return None

    q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in grasp_choice["q_path"]]
    ok, _ = targeted.base.execute_pose_path_stage(
        demo,
        bridge_mod,
        real_exec,
        str(grasp_choice.get("label", "roof_tip_grasp")),
        grasp_choice["pose"],
        q_path,
        args.real_gripper_open,
        args,
    )
    if not ok:
        return None

    print("\n[roof] close gripper at triangle tip")
    if not targeted.base.confirm_simple_action("close the real gripper on triangle tip", args, bridge_mod=bridge_mod, env=demo.env, repeats=6):
        print("[roof][abort] user cancelled before closing gripper")
        return None
    if real_exec is not None:
        real_exec.set_gripper(args.real_gripper_close)
    targeted.base.sync_demo_gripper_state(demo, closed=True, steps=4)
    try:
        targeted.base.set_pregrasp_object_freeze(demo, False)
    except Exception:
        pass

    direct._stabilize_post_grasp_attached_state(demo, args, grasp_choice)
    targeted.base.lift_active_object_above_table_if_needed(
        demo,
        args,
        min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
    )
    if bool(getattr(args, "curobo_attach_object", True)):
        direct._attach_transport_payload_to_curobo(planner, demo, args, label="roof_transport")
    return grasp_choice


def _plan_and_execute_roof_place(demo, bridge_mod, real_exec, args, scene_capture_cache, planner, grasp_choice, side: str) -> bool:
    target = _roof_target_triangle_pose(demo, scene_capture_cache, args, side)
    target_T = target["target_T_world_obj"]
    T_tcp_obj = grasp_choice.get("T_tcp_obj")
    if T_tcp_obj is None:
        print("[roof][FAIL] grasp choice has no T_tcp_obj; cannot derive release TCP pose")
        return False

    exact_release_pose = _release_pose_from_target_and_grasp(target_T, T_tcp_obj)
    # Capture/release is slightly outward so the real magnets can pull in without the gripper forcing penetration.
    release_pose = _make_pose_offset_world(exact_release_pose, target["outward_world"] * float(args.roof_release_outward_offset_m))
    hover_pose = _make_pose_offset_world(
        release_pose,
        target["outward_world"] * float(args.roof_preplace_outward_offset_m)
        + np.array([0.0, 0.0, float(args.roof_hover_extra_z_m)], dtype=np.float32),
    )

    # Minimal place_plan object so existing sorting/debug helpers do not crash.
    rule_stub = SimpleNamespace(
        primitive="roof_assembly",
        target_object_name=args.roof_cube_name,
        tabletop_place_tcp_verticality_target=None,
        tabletop_place_tcp_axis_vertical=None,
    )
    place_plan = SimpleNamespace(
        rule=rule_stub,
        target_name=args.roof_cube_name,
        slot_name=side,
        variant_label=f"roof_{side}",
        T_world_obj_desired=target_T,
    )
    candidate = {
        "label": f"roof_{side}_transport_hover",
        "pose": hover_pose,
        "hover_pose": hover_pose,
        "pre_place_pose": hover_pose,
        "place_pose": release_pose,
        "release_pose": release_pose,
        "raw_release_pose": exact_release_pose,
        "retreat_pose": hover_pose,
        "target_name": args.roof_cube_name,
        "slot_name": side,
        "variant_label": f"roof_{side}",
        "tcp_verticality": 0.0,
        "pad_tilt": 0.0,
        "place_plan": place_plan,
        "place_mode": "vertical_place",
        "object_category": "roof_triangle",
        "direct_place_mode": False,
        "transport_to_hover": True,
        "release_lift_m": 0.0,
        "hover_extra_height_m": float(args.roof_hover_extra_z_m),
    }

    include_table = bool(getattr(args, "curobo_table_collision", True))
    # During transport, exclude the active source triangle only; keep cube/desk as world obstacles.
    exclude_names = direct._attached_source_exclude_names(args, rule_stub) if hasattr(direct, "_attached_source_exclude_names") else {args.object_name}
    disabled_links = direct._direct_place_contact_tolerant_disabled_links(planner) if hasattr(direct, "_direct_place_contact_tolerant_disabled_links") else []

    print("\n[roof] plan transport to roof hover")
    transport = direct.plan_transport_to_hover(
        planner,
        demo,
        args,
        np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7],
        [candidate],
        include_table=include_table,
        exclude_object_names=exclude_names,
        disabled_world_collision_links=disabled_links,
    )
    if not transport:
        print("[roof][FAIL] transport to roof hover failed")
        return False

    print("\n[roof] plan final contact/capture approach")
    place_choice = direct.plan_final_contact_approach(
        planner,
        demo,
        args,
        np.asarray(transport[0]["q_path"][-1], dtype=np.float32).reshape(-1)[:7],
        transport[0],
        disabled_world_collision_links=disabled_links,
    )
    if place_choice is None:
        print("[roof][FAIL] final contact to roof capture pose failed")
        return False

    q_pre = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in place_choice.get("q_pre_place_path", place_choice["q_path"])]
    q_place = [
        np.asarray(q, dtype=np.float32).reshape(-1)[:7]
        for q in place_choice.get("q_place_path", [np.asarray(q_pre[-1], dtype=np.float32).reshape(-1)[:7]])
    ]

    print(
        f"[roof] side={side}, hover p={np.round(targeted.base.flatten_np(hover_pose.p)[:3], 5).tolist()}, "
        f"release p={np.round(targeted.base.flatten_np(release_pose.p)[:3], 5).tolist()}"
    )
    ok, _ = targeted.base.execute_pose_path_stage(
        demo,
        bridge_mod,
        real_exec,
        f"roof_{side}_hover",
        place_choice["pre_place_pose"],
        q_pre,
        args.real_gripper_close,
        args,
        use_attach=True,
        skip_confirmation=True,
    )
    if ok and q_place:
        ok, _ = targeted.base.execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            f"roof_{side}_capture",
            place_choice["place_pose"],
            q_place,
            args.real_gripper_close,
            args,
            use_attach=True,
            skip_confirmation=True,
        )
    if not ok:
        print("[roof][FAIL] executing roof place path failed")
        return False

    print("\n[roof] open gripper and snap triangle")
    if not targeted.base.confirm_simple_action("open gripper at roof capture pose", args, bridge_mod=bridge_mod, env=demo.env, repeats=6):
        return False
    if real_exec is not None:
        real_exec.set_gripper(args.real_gripper_open)
    targeted.base.sync_demo_gripper_state(demo, closed=False, steps=4)
    if getattr(planner, "attached_object_active", False):
        planner.detach_object_from_robot()

    # Magnetic approximation: in sim, snap to exact roof target pose after release.
    if bool(getattr(args, "roof_snap_after_release", True)):
        q_target = targeted.base.bridge_mod_mat2quat(target_T[:3, :3]).astype(np.float32)
        if hasattr(direct, "_set_active_object_pose_quiet"):
            direct._set_active_object_pose_quiet(demo, target_T[:3, 3].astype(np.float32), q_target)
        else:
            obj = getattr(getattr(demo, "base_env", None), "obj", None)
            if obj is not None:
                obj.set_pose(targeted.Pose.create_from_pq(p=target_T[:3, 3].astype(np.float32), q=q_target))
        print(f"[roof] snapped active triangle to exact magnetic target for side={side}")

    try:
        targeted.base.settle_released_active_object_for_scene_cache(demo, args)
    except Exception:
        pass

    if bool(getattr(args, "skip_post_place_clearance", False)):
        print("[roof] skip post-place clearance by request")
        return True

    # Reuse normal post-place clearance if desired; many assembly tests skip it for speed.
    return True


# ----------------------------- final clamp primitive -----------------------------

def _clamp_pair_pose(demo, scene_capture_cache, args, pair: str, *, pre: bool):
    cube_center, top_z = _cube_center_and_top(demo, scene_capture_cache, args)
    roof_h = float(args.roof_height_m)
    apex = cube_center.copy()
    apex[2] = top_z + roof_h
    z_offset = float(args.roof_close_pre_z_m if pre else args.roof_close_z_m)
    p = apex + np.array([0.0, 0.0, z_offset], dtype=np.float32)

    # Use current topdown orientation and roll it so the gripper opening axis roughly spans the pair.
    pose = demo.build_topdown_grasp_pose()
    pose = targeted.base.make_pose_with_position(pose, p.astype(np.float32))
    if pair == "front_back":
        roll = 90.0
    elif pair == "left_right":
        roll = 0.0
    else:
        roll = 0.0
    if hasattr(direct, "_roll_pose_about_tcp_approach"):
        pose = direct._roll_pose_about_tcp_approach(pose, roll)
    return pose


def _run_roof_closure(demo, bridge_mod, real_exec, args, scene_capture_cache) -> bool:
    planner = curobo_wrapper._get_or_create_curobo_planner(args)
    pairs = list(getattr(args, "roof_close_pairs", []) or [])
    if not pairs:
        print("[roof_close] no close pairs requested")
        return True
    for pair in pairs:
        print(f"\n[roof_close] clamp pair={pair}")
        pre_pose = _clamp_pair_pose(demo, scene_capture_cache, args, pair, pre=True)
        clamp_pose = _clamp_pair_pose(demo, scene_capture_cache, args, pair, pre=False)
        start_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
        direct._refresh_curobo_world(
            planner,
            demo,
            args,
            label=f"roof_close_{pair}",
            include_active_object=False,
            include_table=bool(getattr(args, "curobo_table_collision", True)),
        )
        pre_cand = [{"label": f"roof_close_{pair}_pre", "pose": pre_pose}]
        pre_path = direct._evaluate_curobo_pose_candidates_goalset(
            planner,
            demo,
            args,
            start_q,
            pre_cand,
            label=f"roof_close_{pair}_pre",
            use_attach=False,
            max_winners=1,
            include_table=bool(getattr(args, "curobo_table_collision", True)),
        )
        if not pre_path:
            print(f"[roof_close][FAIL] cannot reach pre clamp pose for {pair}")
            return False
        q_pre = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in pre_path[0]["q_path"]]
        ok, _ = targeted.base.execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            f"roof_close_{pair}_pre",
            pre_pose,
            q_pre,
            args.real_gripper_open,
            args,
            use_attach=False,
            skip_confirmation=True,
        )
        if not ok:
            return False

        q_clamp = direct._plan_constrained_linear_segment(
            planner,
            demo,
            args,
            np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7],
            pre_pose,
            clamp_pose,
            label=f"roof_close_{pair}_descend",
            validation_pos_tol_m=float(max(getattr(args, "strict_short_linear_waypoint_pos_tol_m", 0.010), 0.0)),
        ) if hasattr(direct, "_plan_constrained_linear_segment") else None
        if not q_clamp:
            print(f"[roof_close][FAIL] clamp descent failed for {pair}")
            return False
        ok, _ = targeted.base.execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            f"roof_close_{pair}",
            clamp_pose,
            q_clamp,
            args.real_gripper_open,
            args,
            use_attach=False,
            skip_confirmation=True,
        )
        if not ok:
            return False

        # Close the gripper to squeeze the opposite panels inward. Tune commands cautiously on real hardware.
        close_cmd = args.roof_close_gripper_close if args.roof_close_gripper_close is not None else args.real_gripper_close
        open_cmd = args.roof_close_gripper_open if args.roof_close_gripper_open is not None else args.real_gripper_open
        if not targeted.base.confirm_simple_action(f"close gripper to clamp roof pair {pair}", args, bridge_mod=bridge_mod, env=demo.env, repeats=6):
            return False
        if real_exec is not None:
            real_exec.set_gripper(close_cmd)
        targeted.base.sync_demo_gripper_state(demo, closed=True, steps=4)
        time_sleep = float(getattr(args, "roof_close_hold_s", 0.5) if hasattr(args, "roof_close_hold_s") else 0.5)
        if time_sleep > 0:
            import time
            time.sleep(time_sleep)
        if real_exec is not None:
            real_exec.set_gripper(open_cmd)
        targeted.base.sync_demo_gripper_state(demo, closed=False, steps=4)
        print(f"[roof_close] clamped pair={pair}")
    return True


# ----------------------------- episode / main -----------------------------

def run_roof_assembly_episode(demo, bridge_mod, real_exec, args, scene_capture_cache, place_state_cache) -> bool:
    print("\n[roof] template magnetic roof assembly episode")
    args._skip_remaining_step_confirms_in_object = False

    planner = curobo_wrapper._get_or_create_curobo_planner(args)
    if getattr(planner, "attached_object_active", False):
        planner.detach_object_from_robot()
    try:
        demo._attached_box_visual_visible = False
        demo._attached_object_visual_active = False
        targeted.base.update_attached_box_visual(demo, visible=False)
    except Exception:
        pass

    start_q = demo.current_arm_qpos()
    if real_exec is not None:
        print("\n[roof] real robot setup")
        if args.render_mode == "human":
            bridge_mod.render_preview(demo.env, repeats=5)
        ok, q_sent = targeted.base.align_real_robot_to_sim_start(demo, bridge_mod, real_exec, start_q, args)
        if not ok:
            print("[roof][abort] failed to align real robot to sim start")
            return False
        targeted.base.sync_demo_arm_qpos(demo, q_sent if q_sent is not None else start_q)
    else:
        print("\n[roof] dry-run: --execute-real not provided")

    if bool(getattr(args, "roof_close_only", False)):
        return _run_roof_closure(demo, bridge_mod, real_exec, args, scene_capture_cache)

    side = args.roof_side if args.roof_side != "auto" else _parse_side_from_name(getattr(args, "object_name", ""))
    print(f"[roof] current triangle side={side}, object={getattr(args, 'object_name', None)}")

    grasp_choice = _select_and_execute_roof_grasp(demo, bridge_mod, real_exec, args, planner)
    if grasp_choice is None:
        return False

    post_lift_ok = direct._skip_post_grasp_escape(demo, bridge_mod, real_exec, args, "roof_post_grasp_lift", use_attach=True)
    if not post_lift_ok:
        print("[roof] post-grasp lift failed; continuing only if current pose remains valid")

    ok = _plan_and_execute_roof_place(demo, bridge_mod, real_exec, args, scene_capture_cache, planner, grasp_choice, side)
    if not ok:
        return False

    if bool(getattr(args, "skip_return_to_cycle_start", False)):
        print("[roof] placed triangle; skipping return_to_cycle_start by request")
        return True
    return targeted.base.plan_and_execute_return_to_start_if_available(demo, bridge_mod, real_exec, args) if hasattr(targeted.base, "plan_and_execute_return_to_start_if_available") else True


def main():
    original_create_demo = targeted.base.create_demo

    def _profiled_create_demo(args, bridge_mod, planner_mod, scene_capture_cache=None):
        with direct._profile_stage(args, "scene_capture") as prof:
            env, demo = original_create_demo(args, bridge_mod, planner_mod, scene_capture_cache=scene_capture_cache)
            cached_objects = []
            if isinstance(scene_capture_cache, dict):
                cached_objects = list((scene_capture_cache.get("objects") or {}).keys())
            prof["success"] = True
            prof["status"] = "Success"
            prof["candidate_count"] = len(cached_objects)
            return env, demo

    targeted.base.create_demo = _profiled_create_demo
    targeted.parse_args = parse_args

    def _dispatch_episode(demo, bridge_mod, real_exec, args, scene_capture_cache, place_state_cache):
        if bool(getattr(args, "roof_assembly", False)) or bool(getattr(args, "roof_close_only", False)):
            return run_roof_assembly_episode(demo, bridge_mod, real_exec, args, scene_capture_cache, place_state_cache)
        return direct.run_targeted_place_episode_curobo_direct(demo, bridge_mod, real_exec, args, scene_capture_cache, place_state_cache)

    targeted.run_targeted_place_episode = _dispatch_episode
    targeted.main()


if __name__ == "__main__":
    main()
