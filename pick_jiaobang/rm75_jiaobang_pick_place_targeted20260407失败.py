#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2mat

import rm75_jiaobang_pick_real_with_foundationpose as base
from object_specs import normalize_object_name
from place_rules import (
    LocalPoseSpec,
    PlaceRule,
    describe_place_rules,
    get_place_rule,
    get_required_place_target_names,
    get_runtime_slot_specs,
    list_place_rule_sources,
)


@dataclass
class TargetedPlacePlan:
    rule: PlaceRule
    target_name: str
    slot_name: str | None
    variant_label: str | None
    staging_pose: Pose | None
    pre_place_pose: Pose
    place_pose: Pose
    retreat_pose: Pose
    tcp_verticality: float = 0.0
    pre_place_distance: float = 0.0
    approach_label: str | None = None
    ranking_score: float = float("inf")
    pre_place_ik_distance: float = float("inf")
    direct_place_ik_distance: float = float("inf")
    direct_place_reachable: bool = False


def build_arg_parser():
    parser = base.build_arg_parser()
    parser.description = "FoundationPose -> grasp -> targeted place pipeline with rule-based destination objects."
    parser.add_argument("--list-place-rules", action="store_true", help="List configured targeted-place rules and exit.")
    parser.add_argument(
        "--tracked-scene-object-names",
        type=str,
        nargs="*",
        default=None,
        help="Optional extra object spec keys to capture into the cached tabletop scene on cycle 1 so later cycles can reuse them without a fresh FoundationPose pass.",
    )
    parser.add_argument(
        "--place-insert-target-collision-scale",
        type=float,
        default=0.25,
        help="During the final descent/retreat of insert-style place primitives, temporarily scale the destination object's planner collision box by this factor to avoid coarse-box false positives.",
    )
    parser.add_argument(
        "--insert-vertical-axial-spin-deg",
        type=float,
        nargs="*",
        default=[0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
        help="For insert-style place rules, also try these equivalent rotations around the source object's long axis when solving pre_place/place IK.",
    )
    parser.add_argument(
        "--targeted-place-staging",
        dest="targeted_place_staging",
        action="store_true",
        default=True,
        help="Before moving to the destination hover pose, first reorient the grasped object at a safer staging pose away from the destination. Enabled by default.",
    )
    parser.add_argument(
        "--no-targeted-place-staging",
        dest="targeted_place_staging",
        action="store_false",
        help="Disable the extra staging pose before the destination hover pose.",
    )
    parser.add_argument(
        "--targeted-place-staging-z-margin",
        type=float,
        default=0.08,
        help="Extra Z clearance above both the current TCP height and the destination hover pose when building the targeted-place staging pose.",
    )
    parser.add_argument(
        "--targeted-place-direct-first",
        dest="targeted_place_direct_first",
        action="store_true",
        default=False,
        help="Try planning directly from the post-grasp pose to the final place pose before post_grasp_escape/pre_place. Disabled by default.",
    )
    parser.add_argument(
        "--no-targeted-place-direct-first",
        dest="targeted_place_direct_first",
        action="store_false",
        help="Disable direct-place-first probing from the current post-grasp pose.",
    )
    parser.add_argument(
        "--tabletop-place-tilt-toward-robot-deg",
        type=float,
        nargs="*",
        default=[0.0, 15.0, 25.0, 35.0, 50.0],
        help="For place_on_slots rules, also try these tabletop place tilt angles toward the robot. 0 keeps the original upright/rule pose. Nonzero angles automatically compensate the object height to keep it above the tabletop.",
    )
    parser.add_argument(
        "--tabletop-place-yaw-variant-deg",
        type=float,
        nargs="*",
        default=[0.0, -15.0, 15.0, -30.0, 30.0, -45.0, 45.0, -60.0, 60.0, -75.0, 75.0, -90.0, 90.0, 180.0],
        help="For place_on_slots rules, also try these extra in-plane yaw rotations around the destination tabletop normal. This is useful when the object may be placed slightly skewed and a different xy heading makes pre_place IK reachable.",
    )
    parser.add_argument(
        "--tabletop-place-vertical-yaw-variant-deg",
        type=float,
        nargs="*",
        default=[],
        help="Optional extra in-plane yaw set for tabletop objects that must remain vertically placed. Empty by default to avoid expanding the candidate set unless explicitly requested.",
    )
    parser.add_argument(
        "--tabletop-place-axial-spin-deg",
        type=float,
        nargs="*",
        default=[0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
        help="For elongated place_on_slots objects, also try these equivalent rotations around the object's own longest axis before solving pre_place/place IK. This helps keep the gripper from scraping the tabletop while preserving the same final tabletop placement.",
    )
    parser.add_argument(
        "--tabletop-place-min-tcp-verticality",
        type=float,
        default=0.55,
        help="For tabletop place_on_slots rules, when any candidate keeps the TCP approach axis at least this aligned with the tabletop normal, reject flatter/horizontal TCP candidates and try the more vertical ones first.",
    )
    parser.add_argument(
        "--tabletop-place-pre-distance-m",
        type=float,
        nargs="*",
        default=[0.04, 0.06, 0.08, 0.10],
        help="For place_on_slots rules, try these retreat distances along the selected approach axis when building pre_place poses.",
    )
    parser.add_argument(
        "--insert-place-pre-distance-m",
        type=float,
        nargs="*",
        default=[0.05, 0.07, 0.09, 0.12],
        help="For insert_vertical rules, try these retreat distances along the container axis when building pre_place poses.",
    )
    parser.add_argument(
        "--targeted-place-local-offset-m",
        type=float,
        nargs="*",
        default=[0.0, 0.015],
        help="Small local x/z offsets to try around the nominal target place pose before solving pre_place/place IK. 0 keeps the original slot/mouth center.",
    )
    parser.add_argument(
        "--targeted-place-vertical-local-offset-m",
        type=float,
        nargs="*",
        default=[],
        help="Optional extra local x/z offsets for vertically placed tabletop objects. Empty by default to avoid expanding the candidate set unless explicitly requested.",
    )
    parser.add_argument(
        "--targeted-place-rank-probe-count",
        type=int,
        default=6,
        help="How many generated targeted-place candidates to lightly probe with IK before truncating to the final full-planning shortlist.",
    )
    parser.add_argument(
        "--targeted-place-full-plan-top-k",
        type=int,
        default=4,
        help="After lightweight candidate ranking, keep at most this many targeted-place candidates for full path planning.",
    )
    parser.add_argument(
        "--grasp-placeability-lookahead",
        dest="grasp_placeability_lookahead",
        action="store_true",
        default=False,
        help="Probe whether a grasp is likely to lead to a reachable place plan before executing it. Disabled by default to keep grasp selection stable.",
    )
    parser.add_argument(
        "--no-grasp-placeability-lookahead",
        dest="grasp_placeability_lookahead",
        action="store_false",
        help="Disable the grasp-before-placeability lookahead.",
    )
    parser.add_argument(
        "--grasp-placeability-probe-top-k",
        type=int,
        default=2,
        help="Before executing a grasp, probe this many top-ranked targeted-place candidates from the predicted post-grasp state.",
    )
    parser.add_argument(
        "--grasp-placeability-probe-planning-time",
        type=float,
        default=2.0,
        help="Per-candidate planning_time for the lightweight pre-grasp placeability probe.",
    )
    parser.add_argument(
        "--grasp-placeability-probe-attempt-count",
        type=int,
        default=1,
        help="RRT attempt count for the lightweight pre-grasp placeability probe.",
    )
    return parser


def parse_args():
    return build_arg_parser().parse_args()


def maybe_print_and_exit_place_rules(args):
    if not getattr(args, "list_place_rules", False):
        return
    print("Configured targeted-place rules:")
    print(describe_place_rules() or "(none)")
    raise SystemExit(0)


def _dedupe_names(names) -> list[str]:
    deduped = []
    for name in list(names or []):
        normalized = normalize_object_name(name)
        if normalized is None or normalized in deduped:
            continue
        deduped.append(normalized)
    return deduped


def _merge_unique_float_values(*groups) -> list[float]:
    values: list[float] = []
    seen = set()
    for group in groups:
        for raw_v in list(group or []):
            try:
                value = float(raw_v)
            except Exception:
                continue
            if not np.isfinite(value):
                continue
            key = round(value, 6)
            if key in seen:
                continue
            seen.add(key)
            values.append(value)
    return values


def _validate_cycle_sources_have_place_rules(source_names) -> None:
    missing = [name for name in _dedupe_names(source_names) if get_place_rule(name) is None]
    if missing:
        raise ValueError(
            "No targeted-place rule is configured for: "
            + ", ".join(missing)
            + ". Add entries in place_rules.py first."
        )


def _local_pose_spec_to_matrix(spec: LocalPoseSpec) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    rpy_rad = np.deg2rad(np.asarray(spec.rpy_deg, dtype=np.float32).reshape(3))
    T[:3, :3] = euler2mat(float(rpy_rad[0]), float(rpy_rad[1]), float(rpy_rad[2]), axes="sxyz").astype(np.float32)
    T[:3, 3] = np.asarray(spec.position, dtype=np.float32).reshape(3)
    return T


def _normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray | None:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        return None
    return (vec / norm).astype(np.float32)


def _axis_angle_to_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize(axis)
    if axis is None or abs(float(angle_rad)) <= 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z = [float(v) for v in axis]
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    C = 1.0 - c
    return np.asarray(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float32,
    )


def _pose_from_matrix(T: np.ndarray) -> Pose:
    T = np.asarray(T, dtype=np.float32).reshape(4, 4)
    return Pose.create_from_pq(
        p=T[:3, 3].astype(np.float32),
        q=base.bridge_mod_mat2quat(T[:3, :3]).astype(np.float32),
    )


def _find_scene_object_entry(demo, object_name: str | None):
    normalized = normalize_object_name(object_name)
    if normalized is None:
        return None
    for item in list(getattr(demo, "scene_obstacles", []) or []):
        if normalize_object_name(item.get("object_name")) == normalized:
            return item
    return None


def _ensure_target_registered_for_place(demo, target_name: str | None) -> None:
    normalized = normalize_object_name(target_name)
    scene_entry = _find_scene_object_entry(demo, normalized)
    if scene_entry is None:
        raise RuntimeError(
            f"Targeted-place target {normalized!r} is not registered as a scene obstacle in the planner. "
            "Its pose may exist in the cache, but the planner would not avoid it."
        )
    if not bool(scene_entry.get("planner_collision", False)):
        raise RuntimeError(
            f"Targeted-place target {normalized!r} is present visually but has no planner collision object. "
            "Refusing to plan a place path that could pass through the target container."
        )


def _get_scene_object_world_transform(demo, bridge_mod, scene_capture_cache, object_name: str | None) -> np.ndarray | None:
    normalized = normalize_object_name(object_name)
    if normalized is None:
        return None
    scene_entry = _find_scene_object_entry(demo, normalized)
    if scene_entry is not None and scene_entry.get("T_world_obj") is not None:
        return np.asarray(scene_entry["T_world_obj"], dtype=np.float32).reshape(4, 4)
    if normalize_object_name(getattr(getattr(demo, "args", None), "object_name", None)) == normalized:
        obj_p, obj_q = demo.get_obj_pose()
        return base.pose_to_matrix(obj_p, obj_q)
    if not isinstance(scene_capture_cache, dict):
        return None
    objects = scene_capture_cache.get("objects")
    if not isinstance(objects, dict):
        return None
    item = objects.get(normalized)
    if item is None:
        return None
    T_cam_obj = item.get("T_cam_obj")
    T_base_cam = scene_capture_cache.get("T_base_cam")
    object_args = item.get("object_args", getattr(demo, "args", None))
    if T_cam_obj is None or T_base_cam is None or object_args is None:
        return None
    try:
        return bridge_mod.map_camera_pose_to_pick_world(T_cam_obj, T_base_cam, demo.env, object_args)
    except Exception:
        return None


def _current_tcp_to_object_transform(demo) -> np.ndarray:
    obj_p, obj_q = demo.get_obj_pose()
    tcp_pose = demo.tcp.pose
    T_world_obj = base.pose_to_matrix(obj_p, obj_q)
    T_world_tcp = base.pose_to_matrix(base.flatten_np(tcp_pose.p)[:3], base.flatten_np(tcp_pose.q)[:4])
    return np.linalg.inv(T_world_tcp) @ T_world_obj


def _tcp_to_object_transform_from_pose(demo, tcp_pose) -> np.ndarray:
    obj_p, obj_q = demo.get_obj_pose()
    T_world_obj = base.pose_to_matrix(obj_p, obj_q)
    T_world_tcp = base.pose_to_matrix(base.flatten_np(tcp_pose.p)[:3], base.flatten_np(tcp_pose.q)[:4])
    return np.linalg.inv(T_world_tcp) @ T_world_obj


def _ordered_rule_slots(rule: PlaceRule, T_world_target: np.ndarray, bridge_mod, demo) -> list:
    slots = list(rule.slots or [])
    if not slots:
        return []
    target_name = normalize_object_name(getattr(rule, "target_object_name", None))
    if target_name == "desk":
        robot_base_p = base._get_robot_base_world_position(demo)
        runtime_slots = get_runtime_slot_specs(target_name, slots, T_world_target, robot_base_p)
        preferred_order = {f"slot_{idx}": idx - 1 for idx in range(1, 7)}
        if all(str(getattr(slot, "name", "")) in preferred_order for slot in runtime_slots):
            return sorted(
                runtime_slots,
                key=lambda slot: (
                    preferred_order.get(str(getattr(slot, "name", "")), 999),
                    str(getattr(slot, "name", "")),
                ),
            )
        return runtime_slots
    T_world_target = np.asarray(T_world_target, dtype=np.float32).reshape(4, 4)
    robot_base_T = bridge_mod.get_robot_base_transform(demo.env)
    robot_xy = None if robot_base_T is None else np.asarray(robot_base_T[:2, 3], dtype=np.float32).reshape(2)
    annotated = []
    for idx, slot in enumerate(slots):
        local_p = np.asarray(slot.object_pose_local.position, dtype=np.float32).reshape(3)
        world_p = (T_world_target[:3, :3] @ local_p) + T_world_target[:3, 3]
        distance_xy = 0.0 if robot_xy is None else float(np.linalg.norm(world_p[:2] - robot_xy))
        annotated.append((idx, slot, world_p, distance_xy))
    annotated.sort(
        key=lambda item: (
            -item[3],  # farther from the robot first, so near-row slots do not block the far row
            float(item[2][0]),
            float(item[2][1]),
            item[0],
        )
    )
    return [slot for _, slot, _, _ in annotated]


def _mark_place_rule_success(rule: PlaceRule, place_state_cache, slot_name: str | None = None) -> None:
    if rule.primitive != "place_on_slots":
        return
    target_key = normalize_object_name(rule.target_object_name) or str(rule.target_object_name)
    used_slots_by_target = place_state_cache.setdefault("used_slots_by_target", {})
    used_slots = used_slots_by_target.setdefault(target_key, [])
    if slot_name is not None and slot_name not in used_slots:
        used_slots.append(str(slot_name))


def _make_target_local_translation_variants(
    rule: PlaceRule,
    args,
) -> list[tuple[str | None, np.ndarray]]:
    prefix = "slot" if rule.primitive == "place_on_slots" else "mouth"
    offsets = _merge_unique_float_values(getattr(args, "targeted_place_local_offset_m", []) or [])
    if bool(getattr(rule, "preserve_long_axis_vertical", False)):
        offsets = _merge_unique_float_values(offsets, getattr(args, "targeted_place_vertical_local_offset_m", []) or [])
    variants: list[tuple[str | None, np.ndarray]] = [(None, np.zeros(3, dtype=np.float32))]
    seen = {tuple(np.round(variants[0][1], 6).tolist())}
    for offset_m in offsets:
        if abs(offset_m) <= 1e-6:
            continue
        for dx_m, dz_m in [
            (offset_m, 0.0),
            (-offset_m, 0.0),
            (0.0, offset_m),
            (0.0, -offset_m),
        ]:
            delta = np.asarray([dx_m, 0.0, dz_m], dtype=np.float32)
            key = tuple(np.round(delta, 6).tolist())
            if key in seen:
                continue
            seen.add(key)
            label = f"{prefix}_dx{int(round(dx_m * 1000.0)):+d}mm_dz{int(round(dz_m * 1000.0)):+d}mm"
            variants.append((label, delta))
    return variants


def _make_insert_vertical_local_pose_variants(
    rule: PlaceRule,
    object_pose_local: LocalPoseSpec,
    spin_degs,
) -> list[tuple[str | None, np.ndarray]]:
    T_base = _local_pose_spec_to_matrix(object_pose_local)
    variants: list[tuple[str | None, np.ndarray]] = [(None, T_base)]
    if rule.primitive != "insert_vertical":
        return variants
    spin_degs = [float(v) for v in list(spin_degs or []) if np.isfinite(float(v))]
    if not spin_degs:
        return variants
    seen = {tuple(np.round(T_base.reshape(-1), 6).tolist())}
    for spin_deg in spin_degs:
        if abs(spin_deg) <= 1e-6:
            continue
        T_spin = np.eye(4, dtype=np.float32)
        T_spin[:3, :3] = euler2mat(0.0, np.deg2rad(float(spin_deg)), 0.0, axes="sxyz").astype(np.float32)
        T_variant = T_base.copy()
        T_variant[:3, :3] = (T_base[:3, :3] @ T_spin[:3, :3]).astype(np.float32)
        key = tuple(np.round(T_variant.reshape(-1), 6).tolist())
        if key in seen:
            continue
        seen.add(key)
        variants.append((f"spin_{int(round(spin_deg))}deg", T_variant))
    return variants


def _make_tabletop_axial_spin_local_pose_variants(
    args,
    rule: PlaceRule,
    object_pose_local: LocalPoseSpec,
) -> list[tuple[str | None, np.ndarray]]:
    T_base = _local_pose_spec_to_matrix(object_pose_local)
    variants: list[tuple[str | None, np.ndarray]] = [(None, T_base)]
    if rule.primitive != "place_on_slots":
        return variants

    spin_degs = [float(v) for v in list(getattr(args, "tabletop_place_axial_spin_deg", []) or []) if np.isfinite(float(v))]
    if not spin_degs:
        return variants

    try:
        extents = np.asarray(base.get_asset_box_size(args.sim_asset_file, args.sim_asset_scale), dtype=np.float32).reshape(3)
    except Exception:
        return variants
    if extents.shape[0] != 3:
        return variants
    sorted_extents = np.sort(extents)
    if float(sorted_extents[-1]) < 1.8 * float(max(sorted_extents[1], 1e-6)):
        return variants

    axis_idx = int(np.argmax(extents))
    axis_local = np.zeros(3, dtype=np.float32)
    axis_local[axis_idx] = 1.0
    target_up_local = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    axis_after_rule = T_base[:3, :3] @ axis_local
    if abs(float(np.dot(axis_after_rule, target_up_local))) >= 0.75:
        return variants

    seen = {tuple(np.round(T_base.reshape(-1), 6).tolist())}
    for spin_deg in spin_degs:
        if abs(spin_deg) <= 1e-6:
            continue
        R_spin = _axis_angle_to_matrix(axis_local, np.deg2rad(float(spin_deg)))
        T_variant = T_base.copy()
        T_variant[:3, :3] = (T_base[:3, :3] @ R_spin).astype(np.float32)
        key = tuple(np.round(T_variant.reshape(-1), 6).tolist())
        if key in seen:
            continue
        seen.add(key)
        variants.append((f"axial_spin_{int(round(spin_deg))}deg", T_variant))
    return variants


def _make_orientation_invariant_local_pose_variants(
    rule: PlaceRule,
    object_pose_local: LocalPoseSpec,
) -> list[tuple[str | None, np.ndarray]]:
    T_base = _local_pose_spec_to_matrix(object_pose_local)
    variants: list[tuple[str | None, np.ndarray]] = [(None, T_base)]
    if not bool(getattr(rule, "orientation_invariant", False)):
        return variants

    seen = {tuple(np.round(T_base.reshape(-1), 6).tolist())}
    angle_degs = [0.0, 90.0, 180.0, 270.0]
    for rx_deg in angle_degs:
        for ry_deg in angle_degs:
            for rz_deg in angle_degs:
                if abs(rx_deg) <= 1e-6 and abs(ry_deg) <= 1e-6 and abs(rz_deg) <= 1e-6:
                    continue
                R_extra = euler2mat(
                    np.deg2rad(float(rx_deg)),
                    np.deg2rad(float(ry_deg)),
                    np.deg2rad(float(rz_deg)),
                    axes="sxyz",
                ).astype(np.float32)
                T_variant = T_base.copy()
                T_variant[:3, :3] = (T_base[:3, :3] @ R_extra).astype(np.float32)
                key = tuple(np.round(T_variant.reshape(-1), 6).tolist())
                if key in seen:
                    continue
                seen.add(key)
                variants.append(
                    (
                        f"free_orientation_rx{int(round(rx_deg))}_ry{int(round(ry_deg))}_rz{int(round(rz_deg))}",
                        T_variant,
                    )
                )
    return variants


def _build_targeted_place_staging_pose(demo, pre_place_pose, args) -> Pose:
    z_margin = float(max(getattr(args, "targeted_place_staging_z_margin", 0.04), 0.0))
    try:
        obj_p, obj_q = demo.get_obj_pose()
        T_world_obj_current = base.pose_to_matrix(obj_p, obj_q)
        T_tcp_obj = _current_tcp_to_object_transform(demo)
        T_world_tcp_pre = base.pose_to_matrix(base.flatten_np(pre_place_pose.p)[:3], base.flatten_np(pre_place_pose.q)[:4])
        T_world_obj_pre = T_world_tcp_pre @ T_tcp_obj

        T_world_obj_stage = np.asarray(T_world_obj_current, dtype=np.float32).reshape(4, 4).copy()
        T_world_obj_stage[:3, :3] = np.asarray(T_world_obj_pre[:3, :3], dtype=np.float32)
        stage_obj_p = np.asarray(T_world_obj_current[:3, 3], dtype=np.float32).reshape(3).copy()
        pre_obj_p = np.asarray(T_world_obj_pre[:3, 3], dtype=np.float32).reshape(3)
        stage_obj_p[2] = max(float(stage_obj_p[2]), float(pre_obj_p[2])) + z_margin
        T_world_obj_stage[:3, 3] = stage_obj_p

        T_world_tcp_stage = T_world_obj_stage @ np.linalg.inv(T_tcp_obj)
        return _pose_from_matrix(T_world_tcp_stage)
    except Exception:
        tcp_pose = demo.tcp.pose
        tcp_p = base.flatten_np(tcp_pose.p)[:3].copy()
        pre_place_p = base.flatten_np(pre_place_pose.p)[:3].copy()
        tcp_p[2] = max(float(tcp_p[2]), float(pre_place_p[2])) + z_margin
        return base.make_pose_with_position(pre_place_pose, tcp_p)


def _current_tcp_approach_axis(demo) -> np.ndarray | None:
    tcp_pose = demo.tcp.pose
    T_world_tcp = base.pose_to_matrix(base.flatten_np(tcp_pose.p)[:3], base.flatten_np(tcp_pose.q)[:4])
    return _normalize(T_world_tcp[:3, 2])


def _build_pre_place_approach_variants(
    demo,
    rule: PlaceRule,
    T_world_target: np.ndarray,
    T_world_tcp_place: np.ndarray,
    args,
) -> list[tuple[str | None, np.ndarray, float]]:
    target_up_axis = _normalize(np.asarray(T_world_target[:3, 1], dtype=np.float32).reshape(3))
    if target_up_axis is None:
        target_up_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    distance_values = (
        list(getattr(args, "insert_place_pre_distance_m", []) or [])
        if rule.primitive == "insert_vertical"
        else list(getattr(args, "tabletop_place_pre_distance_m", []) or [])
    )
    distance_values = [float(v) for v in distance_values if np.isfinite(float(v)) and float(v) > 1e-6]
    if not distance_values:
        distance_values = [float(rule.hover_height)]

    direction_variants: list[tuple[str | None, np.ndarray]] = [("target_axis", target_up_axis.astype(np.float32))]
    if rule.primitive == "place_on_slots":
        tcp_approach_axis = _current_tcp_approach_axis(demo)
        if tcp_approach_axis is not None:
            blended = _normalize(target_up_axis + tcp_approach_axis)
            if blended is not None and float(np.dot(blended, target_up_axis)) >= 0.35:
                direction_variants.append(("blended_axis", blended.astype(np.float32)))

    variants: list[tuple[str | None, np.ndarray, float]] = []
    seen = set()
    for direction_label, direction in direction_variants:
        for distance_m in distance_values:
            key = (
                tuple(np.round(np.asarray(direction, dtype=np.float32).reshape(3), 4).tolist()),
                round(float(distance_m), 4),
            )
            if key in seen:
                continue
            seen.add(key)
            label = None
            if direction_label is not None or abs(float(distance_m)) > 1e-6:
                label = f"{direction_label or 'approach'}_pre{int(round(float(distance_m) * 1000.0))}mm"
            variants.append((label, np.asarray(direction, dtype=np.float32).reshape(3), float(distance_m)))
    return variants


def _probe_place_pose_ik_hint(demo, pose: Pose, q_current: np.ndarray, args, *, label: str, phase: str):
    q_candidates = base._solve_pose_goal_q_candidates(
        demo,
        pose,
        q_current,
        variant_name=args.variant,
        label=label,
        max_candidates=1,
        phase=phase,
        verbose=False,
    )
    if not q_candidates:
        return None
    q_best = np.asarray(q_candidates[0], dtype=np.float32).reshape(-1)[:7]
    dq = q_best - q_current
    return {
        "q": q_best,
        "distance": float(np.linalg.norm(dq)),
        "joint7_delta": abs(float(dq[6])) if dq.shape[0] >= 7 else 0.0,
        "wrist_delta": float(np.linalg.norm(dq[3:7])) if dq.shape[0] >= 7 else float(np.linalg.norm(dq)),
    }


def _is_center_no_offset_candidate(plan: TargetedPlacePlan) -> bool:
    variant_label = "" if plan.variant_label is None else str(plan.variant_label)
    return ("slot_dx" not in variant_label) and ("mouth_dx" not in variant_label)


def rank_and_filter_place_candidates(demo, plans: list[TargetedPlacePlan], args) -> list[TargetedPlacePlan]:
    if not plans:
        return []

    q_current = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    probe_limit = int(max(getattr(args, "targeted_place_rank_probe_count", 8), 1))
    top_k = int(max(getattr(args, "targeted_place_full_plan_top_k", 3), 1))

    def _geometric_key(plan: TargetedPlacePlan):
        verticality_penalty = 1.0 - float(plan.tcp_verticality)
        return (
            verticality_penalty,
            float(plan.pre_place_distance),
            "" if plan.approach_label is None else str(plan.approach_label),
            "" if plan.variant_label is None else str(plan.variant_label),
            "" if plan.slot_name is None else str(plan.slot_name),
        )

    plans_sorted = sorted(plans, key=_geometric_key)
    probed = plans_sorted[: min(len(plans_sorted), probe_limit)]
    fallback_tail = plans_sorted[len(probed) :]
    forced_center_candidate = next((plan for plan in plans_sorted if _is_center_no_offset_candidate(plan)), None)
    if forced_center_candidate is not None and not any(plan is forced_center_candidate for plan in probed):
        probed.append(forced_center_candidate)
        fallback_tail = [plan for plan in fallback_tail if plan is not forced_center_candidate]

    scored: list[TargetedPlacePlan] = []
    for idx, plan in enumerate(probed, start=1):
        pre_hint = _probe_place_pose_ik_hint(
            demo,
            plan.pre_place_pose,
            q_current,
            args,
            label=f"place_rank_pre_{idx}",
            phase="pre_place",
        )
        direct_hint = _probe_place_pose_ik_hint(
            demo,
            plan.place_pose,
            q_current,
            args,
            label=f"place_rank_direct_{idx}",
            phase="insert_vertical" if plan.rule.primitive == "insert_vertical" else "place_final",
        )

        plan.pre_place_ik_distance = float(pre_hint["distance"]) if pre_hint is not None else float("inf")
        plan.direct_place_ik_distance = float(direct_hint["distance"]) if direct_hint is not None else float("inf")
        plan.direct_place_reachable = bool(direct_hint is not None)

        pre_cost = (
            float(pre_hint["distance"]) + 0.18 * float(pre_hint["joint7_delta"]) + 0.10 * float(pre_hint["wrist_delta"])
            if pre_hint is not None
            else 25.0
        )
        direct_cost = (
            float(direct_hint["distance"]) + 0.18 * float(direct_hint["joint7_delta"]) + 0.10 * float(direct_hint["wrist_delta"])
            if direct_hint is not None
            else 25.0
        )
        if plan.rule.primitive == "insert_vertical":
            verticality_penalty = 1.2 * (1.0 - float(plan.tcp_verticality))
        elif bool(getattr(plan.rule, "preserve_long_axis_vertical", False)):
            verticality_penalty = 0.35 * (1.0 - float(plan.tcp_verticality))
        else:
            verticality_penalty = 0.05 * (1.0 - float(plan.tcp_verticality))
        reachability_penalty = 4.0 if (pre_hint is None and direct_hint is None) else 0.0
        direct_bonus = 0.0 if direct_hint is not None else 0.8

        plan.ranking_score = float(
            min(pre_cost, direct_cost)
            + 0.40 * float(plan.pre_place_distance)
            + verticality_penalty
            + direct_bonus
            + reachability_penalty
        )
        scored.append(plan)

    scored.sort(
        key=lambda plan: (
            float(plan.ranking_score),
            0 if plan.direct_place_reachable else 1,
            float(plan.pre_place_ik_distance),
            float(plan.pre_place_distance),
            "" if plan.variant_label is None else str(plan.variant_label),
            "" if plan.slot_name is None else str(plan.slot_name),
        )
    )

    selected = list(scored[:top_k])
    if forced_center_candidate is not None and not any(plan is forced_center_candidate for plan in selected):
        if len(selected) < top_k:
            selected.append(forced_center_candidate)
        elif selected:
            selected[-1] = forced_center_candidate
    if plans and plans[0].rule.primitive == "place_on_slots" and not bool(getattr(plans[0].rule, "preserve_long_axis_vertical", False)):
        has_tilted = any(
            ("tilt_toward_robot" in str(getattr(plan, "variant_label", "") or "")) or float(getattr(plan, "tcp_verticality", 1.0)) < 0.92
            for plan in selected
        )
        if not has_tilted:
            for plan in scored[top_k:]:
                if ("tilt_toward_robot" in str(getattr(plan, "variant_label", "") or "")) or float(getattr(plan, "tcp_verticality", 1.0)) < 0.92:
                    if selected:
                        selected[-1] = plan
                    else:
                        selected.append(plan)
                    break
    if len(selected) < top_k:
        selected.extend(fallback_tail[: top_k - len(selected)])

    deduped_selected: list[TargetedPlacePlan] = []
    for plan in selected:
        if any(existing is plan for existing in deduped_selected):
            continue
        deduped_selected.append(plan)
    selected = deduped_selected[:top_k]

    if selected:
        for rank_idx, plan in enumerate(selected, start=1):
            print(
                f"[place rank] #{rank_idx}: target={plan.target_name}"
                f"{'' if plan.slot_name is None else f', slot={plan.slot_name}'}"
                f"{'' if plan.variant_label is None else f', variant={plan.variant_label}'}"
                f"{'' if plan.approach_label is None else f', approach={plan.approach_label}'}"
                f", score={plan.ranking_score:.3f}, pre_d={plan.pre_place_distance:.3f}, "
                f"pre_ik={'yes' if np.isfinite(plan.pre_place_ik_distance) else 'no'}, "
                f"direct_ik={'yes' if plan.direct_place_reachable else 'no'}, "
                f"tcp_verticality={plan.tcp_verticality:.3f}"
            )
    return selected


def _make_tabletop_place_world_pose_variants(
    demo,
    bridge_mod,
    args,
    rule: PlaceRule,
    T_world_target: np.ndarray,
    T_world_obj_desired: np.ndarray,
) -> list[tuple[str | None, np.ndarray]]:
    if rule.primitive != "place_on_slots":
        return [(None, np.asarray(T_world_obj_desired, dtype=np.float32).reshape(4, 4))]
    if bool(getattr(rule, "orientation_invariant", False)):
        return [(None, np.asarray(T_world_obj_desired, dtype=np.float32).reshape(4, 4))]

    T_world_target = np.asarray(T_world_target, dtype=np.float32).reshape(4, 4)
    T_world_obj_desired = np.asarray(T_world_obj_desired, dtype=np.float32).reshape(4, 4)
    tilt_degs = _merge_unique_float_values(getattr(args, "tabletop_place_tilt_toward_robot_deg", []) or [])
    yaw_degs = _merge_unique_float_values(getattr(args, "tabletop_place_yaw_variant_deg", []) or [])
    if bool(getattr(rule, "preserve_long_axis_vertical", False)):
        tilt_degs = [0.0]
        yaw_degs = _merge_unique_float_values(
            yaw_degs,
            getattr(args, "tabletop_place_vertical_yaw_variant_deg", []) or [],
        )
    if not tilt_degs:
        tilt_degs = [0.0]
    if not yaw_degs:
        yaw_degs = [0.0]

    up_axis = _normalize(T_world_target[:3, 1])
    robot_base_T = bridge_mod.get_robot_base_transform(demo.env)
    robot_pos = None if robot_base_T is None else np.asarray(robot_base_T[:3, 3], dtype=np.float32).reshape(3)
    if up_axis is None or robot_pos is None:
        return [(None, T_world_obj_desired)]

    object_pos = np.asarray(T_world_obj_desired[:3, 3], dtype=np.float32).reshape(3)
    toward_robot = robot_pos - object_pos
    toward_robot = toward_robot - float(np.dot(toward_robot, up_axis)) * up_axis
    toward_robot = _normalize(toward_robot)
    if toward_robot is None:
        return [(None, T_world_obj_desired)]

    tilt_axis = _normalize(np.cross(up_axis, toward_robot))
    if tilt_axis is None:
        return [(None, T_world_obj_desired)]

    local_points = base.get_asset_local_points(args.sim_asset_file, args.sim_asset_scale)
    plane_origin = np.asarray(T_world_target[:3, 3], dtype=np.float32).reshape(3)
    variants: list[tuple[str | None, np.ndarray]] = []
    seen = set()
    for yaw_deg in yaw_degs:
        yaw_label = None if abs(yaw_deg) <= 1e-6 else f"yaw_{int(round(yaw_deg))}deg"
        T_yaw = T_world_obj_desired.copy()
        if abs(yaw_deg) > 1e-6:
            R_yaw = _axis_angle_to_matrix(up_axis, np.deg2rad(float(yaw_deg)))
            T_yaw[:3, :3] = (R_yaw @ T_world_obj_desired[:3, :3]).astype(np.float32)
        yaw_world_points = (T_yaw[:3, :3] @ local_points.T).T + T_yaw[:3, 3]
        yaw_bottom_along_up = float(np.min((yaw_world_points - plane_origin) @ up_axis))

        for tilt_deg in tilt_degs:
            tilt_label = None if abs(tilt_deg) <= 1e-6 else f"tilt_toward_robot_{int(round(tilt_deg))}deg"
            labels = [label for label in (yaw_label, tilt_label) if label]
            label = "+".join(labels) if labels else None
            T_variant = T_yaw.copy()
            if abs(tilt_deg) > 1e-6:
                R_tilt = _axis_angle_to_matrix(tilt_axis, np.deg2rad(float(tilt_deg)))
                T_variant[:3, :3] = (R_tilt @ T_yaw[:3, :3]).astype(np.float32)
            variant_world_points = (T_variant[:3, :3] @ local_points.T).T + T_variant[:3, 3]
            variant_bottom_along_up = float(np.min((variant_world_points - plane_origin) @ up_axis))
            height_compensation = yaw_bottom_along_up - variant_bottom_along_up
            T_variant[:3, 3] = (T_variant[:3, 3] + up_axis * float(height_compensation)).astype(np.float32)
            key = tuple(np.round(T_variant.reshape(-1), 6).tolist())
            if key in seen:
                continue
            seen.add(key)
            variants.append((label, T_variant))
    return variants or [(None, T_world_obj_desired)]


def generate_targeted_place_plan_variants(
    demo,
    bridge_mod,
    scene_capture_cache,
    rule: PlaceRule,
    place_state_cache,
    args,
    *,
    T_tcp_obj_override: np.ndarray | None = None,
) -> list[TargetedPlacePlan]:
    target_name = normalize_object_name(rule.target_object_name)
    if target_name is None:
        raise RuntimeError(f"Invalid target object name in place rule: {rule.target_object_name!r}")
    T_world_target = _get_scene_object_world_transform(demo, bridge_mod, scene_capture_cache, target_name)
    if T_world_target is None:
        raise RuntimeError(
            f"Failed to resolve the world pose of destination object {target_name}. "
            "Make sure it is captured into the scene cache."
        )
    target_up_axis = _normalize(np.asarray(T_world_target[:3, 1], dtype=np.float32).reshape(3))

    T_tcp_obj = (
        np.asarray(T_tcp_obj_override, dtype=np.float32).reshape(4, 4)
        if T_tcp_obj_override is not None
        else _current_tcp_to_object_transform(demo)
    )
    if rule.primitive == "place_on_slots":
        ordered_slots = _ordered_rule_slots(rule, T_world_target, bridge_mod, demo)
        if not ordered_slots:
            raise RuntimeError(f"Rule for {rule.source_object_name} uses place_on_slots but defines no slots")
        target_key = normalize_object_name(rule.target_object_name) or str(rule.target_object_name)
        used_slots_by_target = place_state_cache.setdefault("used_slots_by_target", {})
        used_slot_names = {
            str(name)
            for name in list(used_slots_by_target.get(target_key, []) or [])
            if name is not None
        }
        remaining_slots = [slot for slot in ordered_slots if str(slot.name) not in used_slot_names]
        if not remaining_slots:
            raise RuntimeError(
                f"All configured slots for target {rule.target_object_name} have been consumed "
                f"({len(ordered_slots)} slots)"
            )
        next_slot = remaining_slots[0]
        slot_specs = [(next_slot.object_pose_local, str(next_slot.name))]
    else:
        if rule.object_pose_local is None:
            raise RuntimeError(f"Rule for {rule.source_object_name} does not define object_pose_local")
        slot_specs = [(rule.object_pose_local, None)]

    plans: list[TargetedPlacePlan] = []
    local_translation_variants = _make_target_local_translation_variants(rule, args)
    for object_pose_local, slot_name in slot_specs:
        local_variants = _make_insert_vertical_local_pose_variants(
            rule,
            object_pose_local,
            getattr(args, "insert_vertical_axial_spin_deg", None),
        )
        if rule.primitive == "place_on_slots":
            local_variants = _make_orientation_invariant_local_pose_variants(rule, object_pose_local)
            if len(local_variants) <= 1:
                local_variants = _make_tabletop_axial_spin_local_pose_variants(args, rule, object_pose_local)

        for local_variant_label, T_target_obj_desired_base in local_variants:
            for offset_label, local_offset in local_translation_variants:
                T_target_obj_desired = np.asarray(T_target_obj_desired_base, dtype=np.float32).reshape(4, 4).copy()
                T_target_obj_desired[:3, 3] = (T_target_obj_desired[:3, 3] + np.asarray(local_offset, dtype=np.float32).reshape(3)).astype(np.float32)
                T_world_obj_desired_base = T_world_target @ T_target_obj_desired
                for tabletop_variant_label, T_world_obj_desired in _make_tabletop_place_world_pose_variants(
                    demo,
                    bridge_mod,
                    args,
                    rule,
                    T_world_target,
                    T_world_obj_desired_base,
                ):
                    T_world_tcp_place = T_world_obj_desired @ np.linalg.inv(T_tcp_obj)
                    tcp_verticality = 0.0
                    if target_up_axis is not None:
                        tcp_approach_axis = _normalize(np.asarray(T_world_tcp_place[:3, 2], dtype=np.float32).reshape(3))
                        if tcp_approach_axis is not None:
                            tcp_verticality = abs(float(np.dot(tcp_approach_axis, target_up_axis)))
                    place_pose = _pose_from_matrix(T_world_tcp_place)
                    p_place = base.flatten_np(place_pose.p)[:3].copy()
                    approach_variants = _build_pre_place_approach_variants(
                        demo,
                        rule,
                        T_world_target,
                        T_world_tcp_place,
                        args,
                    )
                    for approach_label, approach_dir, pre_place_distance in approach_variants:
                        pre_place_pose = base.make_pose_with_position(
                            place_pose,
                            p_place + np.asarray(approach_dir, dtype=np.float32).reshape(3) * float(pre_place_distance),
                        )
                        staging_pose = None
                        if bool(getattr(args, "targeted_place_staging", True)):
                            staging_pose = _build_targeted_place_staging_pose(demo, pre_place_pose, args)
                        retreat_pose = base.make_pose_with_position(
                            place_pose,
                            p_place + np.asarray(target_up_axis, dtype=np.float32).reshape(3) * float(rule.release_retreat_height),
                        )
                        labels = [label for label in (offset_label, local_variant_label, tabletop_variant_label) if label]
                        variant_label = "+".join(labels) if labels else None
                        plans.append(
                            TargetedPlacePlan(
                                rule=rule,
                                target_name=target_name,
                                slot_name=slot_name,
                                variant_label=variant_label,
                                staging_pose=staging_pose,
                                pre_place_pose=pre_place_pose,
                                place_pose=place_pose,
                                retreat_pose=retreat_pose,
                                tcp_verticality=float(tcp_verticality),
                                pre_place_distance=float(pre_place_distance),
                                approach_label=approach_label,
                            )
                        )
    if rule.primitive == "place_on_slots" and plans:
        min_verticality = float(max(getattr(args, "tabletop_place_min_tcp_verticality", 0.0), 0.0))
        enforce_verticality = bool(getattr(rule, "preserve_long_axis_vertical", False))
        if enforce_verticality and any(plan.tcp_verticality >= min_verticality for plan in plans):
            plans = [plan for plan in plans if plan.tcp_verticality >= min_verticality]
    return plans


def build_targeted_place_plan_variants(
    demo,
    bridge_mod,
    scene_capture_cache,
    rule: PlaceRule,
    place_state_cache,
    args,
    *,
    T_tcp_obj_override: np.ndarray | None = None,
) -> list[TargetedPlacePlan]:
    plans = generate_targeted_place_plan_variants(
        demo,
        bridge_mod,
        scene_capture_cache,
        rule,
        place_state_cache,
        args,
        T_tcp_obj_override=T_tcp_obj_override,
    )
    return rank_and_filter_place_candidates(demo, plans, args)


def _set_scene_obstacle_planner_box_scale(demo, object_name: str, scale: float) -> bool:
    scene_entry = _find_scene_object_entry(demo, object_name)
    if scene_entry is None:
        return False
    if not bool(scene_entry.get("planner_collision", False)):
        return False
    planner_box_size = scene_entry.get("planner_box_size")
    T_world_obj = scene_entry.get("T_world_obj")
    actor_name = str(scene_entry.get("actor_name", "") or "")
    if planner_box_size is None or T_world_obj is None or not actor_name:
        return False
    from mplib import collision_detection as mplib_cd

    planner_box_size = np.asarray(planner_box_size, dtype=np.float32).reshape(3)
    scale = float(max(scale, 1e-3))
    scaled_box_size = np.maximum(planner_box_size * scale, 1e-4).astype(np.float32)
    T_world_obj = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4)
    pos = T_world_obj[:3, 3].astype(np.float32)
    quat = base.bridge_mod_mat2quat(T_world_obj[:3, :3]).astype(np.float32)
    collision_object = mplib_cd.fcl.CollisionObject(
        mplib_cd.fcl.Box(scaled_box_size.tolist()),
        pos.tolist(),
        quat.tolist(),
    )
    demo.planner.set_normal_object(actor_name, collision_object)
    print(
        f"[place] updated planner collision box for {object_name}: "
        f"scale={scale:.3f}, box_size={np.round(scaled_box_size, 6)}"
    )
    return True


def _register_transport_attached_box(demo, args) -> np.ndarray:
    target_box_size = base.get_asset_box_size(args.sim_asset_file, args.sim_asset_scale)
    attached_box_scale = float(np.clip(args.transport_attached_box_scale, 0.5, 2.0))
    attached_box_size = np.maximum(target_box_size * attached_box_scale, 1e-4).astype(np.float32)
    print(
        "[planner] targeted_place raw target box size:",
        np.round(target_box_size, 6),
        f"(asset={Path(str(args.sim_asset_file)).name}, scale={float(args.sim_asset_scale):.6f})",
    )
    print(
        "[planner] targeted_place attached box size:",
        np.round(attached_box_size, 6),
        f"(scale={attached_box_scale:.3f})",
    )
    attach_pose_local = base.make_attached_box_pose(demo, attached_box_size)
    try:
        demo.planner.update_attached_box(attached_box_size.tolist(), attach_pose_local.tolist())
        base.setup_attached_box_visual(demo, demo.env, attached_box_size, attach_pose_local)
    except Exception as exc:
        print(f"[warn] failed to register attached target box for targeted place planning: {exc}")
    return attached_box_size


def _register_transport_attached_box_for_probe(demo, args, T_tcp_obj_override: np.ndarray) -> np.ndarray:
    target_box_size = base.get_asset_box_size(args.sim_asset_file, args.sim_asset_scale)
    attached_box_scale = float(np.clip(args.transport_attached_box_scale, 0.5, 2.0))
    attached_box_size = np.maximum(target_box_size * attached_box_scale, 1e-4).astype(np.float32)
    T_tcp_obj_override = np.asarray(T_tcp_obj_override, dtype=np.float32).reshape(4, 4)
    attach_pose_local = np.concatenate(
        [
            T_tcp_obj_override[:3, 3].astype(np.float32),
            base.bridge_mod_mat2quat(T_tcp_obj_override[:3, :3]).astype(np.float32),
        ]
    ).astype(np.float32)
    try:
        demo.planner.update_attached_box(attached_box_size.tolist(), attach_pose_local.tolist())
    except Exception as exc:
        print(f"[warn] failed to register probe attached box for targeted place planning: {exc}")
    demo.attached_box_size = attached_box_size
    demo.attached_box_pose_tcp = attach_pose_local
    demo._attached_box_visual_visible = False
    demo._attached_object_visual_active = False
    base.update_attached_box_visual(demo, visible=False)
    return attached_box_size


def _run_post_grasp_escape_for_place(demo, bridge_mod, real_exec, args, label: str, *, use_attach: bool) -> bool:
    lift_height = float(max(getattr(args, "pre_transport_lift_height", 0.0), 0.0))
    retreat_distance = float(max(getattr(args, "post_grasp_retreat_distance", 0.0), 0.0))
    if lift_height <= 1e-6:
        return True
    print(f"\n[{label}]")
    lift_pose = base.make_lifted_tcp_pose(demo, lift_height, retreat_distance=retreat_distance)
    lift_path = base.plan_lift_path(
        demo,
        lift_pose,
        variant_name=args.variant,
        use_attach=use_attach,
        label=label,
        planning_time=min(float(args.fixed_goal_planning_time), 3.0),
        rrt_range=float(args.fixed_goal_rrt_range),
        start_q=np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7],
        allow_pose_rrt_fallback=False,
        max_segment_joint_delta=0.35,
        max_segment_joint7_delta=0.80,
        max_segment_norm_delta=0.60,
    )
    if lift_path is None:
        print(f"[warn] {label} planning failed")
        return False
    ok, _ = base.execute_pose_path_stage(
        demo,
        bridge_mod,
        real_exec,
        label,
        lift_pose,
        lift_path,
        args.real_gripper_close,
        args,
        use_attach=use_attach,
    )
    if not ok:
        return False
    base.lift_active_object_above_table_if_needed(
        demo,
        args,
        min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
    )
    return True


def _plan_pose_with_ranked_ik_joint_paths(
    demo,
    pose,
    args,
    *,
    label: str,
    phase: str,
    use_attach: bool,
    start_q=None,
    planning_time: float,
    rrt_range: float,
    allow_linear_fallback: bool = True,
    allow_reverse_rrt_fallback: bool = False,
    prefer_reverse_rrt: bool = False,
    allow_shortcut: bool = True,
    allow_start_in_collision: bool = False,
    attempt_count_override: int | None = None,
):
    q_start = np.asarray(demo.current_arm_qpos() if start_q is None else start_q, dtype=np.float32).reshape(-1)[:7]
    print(f"[planner] current arm q for {label}:", np.round(q_start, 6))
    print(f"[planner] target pose p for {label}:", np.round(base.flatten_np(pose.p)[:3], 6))
    print(f"[planner] target pose q for {label}:", np.round(base.flatten_np(pose.q)[:4], 6))

    q_goal_candidates = base._solve_pose_goal_q_candidates(
        demo,
        pose,
        q_start,
        variant_name=args.variant,
        label=label,
        phase=phase,
        max_candidates=int(max(getattr(args, "pose_ik_candidate_top_k", 4), 1)),
        verbose=True,
    )
    if q_goal_candidates:
        success_candidates = []
        for ik_idx, q_goal in enumerate(q_goal_candidates, start=1):
            q_path = base.plan_joint_path(
                demo,
                q_goal,
                use_attach=use_attach,
                label=f"{label}_ik{ik_idx}",
                planning_time=planning_time,
                rrt_range=rrt_range,
                start_q=q_start,
                allow_linear_fallback=allow_linear_fallback,
                allow_reverse_rrt_fallback=allow_reverse_rrt_fallback,
                prefer_reverse_rrt=prefer_reverse_rrt,
                allow_shortcut=allow_shortcut,
                allow_start_in_collision=allow_start_in_collision,
                attempt_count_override=attempt_count_override,
            )
            if q_path is None:
                continue
            success_candidates.append(
                {
                    "attempt_label": f"ik{ik_idx}",
                    "path": q_path,
                }
            )
        if success_candidates:
            best_candidate = base._select_best_rrt_success_candidate(label, q_start, success_candidates)
            return best_candidate["path"]
        print(f"[planner] {label} explicit IK-ranked joint planning failed; falling back to direct pose planning")
    else:
        print(f"[planner] {label} explicit IK candidate search failed; falling back to direct pose planning")

    return base._plan_pose_path_rrt_only(
        demo,
        pose,
        variant_name=args.variant,
        use_attach=use_attach,
        label=label,
        planning_time=planning_time,
        rrt_range=rrt_range,
        start_q=q_start,
        allow_start_in_collision=allow_start_in_collision,
        attempt_count_override=attempt_count_override,
    )


def _plan_direct_place_candidate(
    demo,
    candidate: TargetedPlacePlan,
    args,
    *,
    label_prefix: str,
    start_q=None,
    planning_time_override: float | None = None,
    attempt_count_override: int | None = None,
    probe_only: bool = False,
):
    q_eval_saved = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    try:
        label = label_prefix if candidate.variant_label is None else f"{label_prefix}_{candidate.variant_label}"
        q_place_candidate = _plan_pose_with_ranked_ik_joint_paths(
            demo,
            candidate.place_pose,
            args,
            label=label,
            phase="insert_vertical" if candidate.rule.primitive == "insert_vertical" else "place_final",
            use_attach=True,
            planning_time=float(planning_time_override if planning_time_override is not None else args.fixed_goal_planning_time),
            rrt_range=args.fixed_goal_rrt_range,
            start_q=start_q,
            attempt_count_override=attempt_count_override,
        )
        if q_place_candidate is None:
            return None, None

        if probe_only:
            return q_place_candidate, None

        q_retreat_candidate = None
        try:
            base.sync_demo_arm_qpos(demo, np.asarray(q_place_candidate[-1], dtype=np.float32).reshape(-1)[:7])
            q_retreat_candidate = base.plan_lift_path(
                demo,
                candidate.retreat_pose,
                variant_name=args.variant,
                use_attach=False,
                label="post_place_retreat" if candidate.variant_label is None else f"post_place_retreat_{candidate.variant_label}",
                planning_time=min(float(args.fixed_goal_planning_time), 3.0),
                rrt_range=float(args.fixed_goal_rrt_range),
                start_q=np.asarray(q_place_candidate[-1], dtype=np.float32).reshape(-1)[:7],
                allow_pose_rrt_fallback=False,
                max_segment_joint_delta=0.45,
                max_segment_joint7_delta=0.45,
                max_segment_norm_delta=0.75,
            )
        finally:
            base.sync_demo_arm_qpos(demo, q_eval_saved)

        return q_place_candidate, q_retreat_candidate
    finally:
        base.sync_demo_arm_qpos(demo, q_eval_saved)


def _probe_grasp_variant_placeability(
    demo,
    bridge_mod,
    scene_capture_cache,
    rule: PlaceRule,
    place_state_cache,
    args,
    *,
    grasp_pose,
    q_grasp: np.ndarray,
    variant_label: str,
) -> bool:
    q_saved = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    prev_attached_box_size = None if getattr(demo, "attached_box_size", None) is None else np.asarray(demo.attached_box_size, dtype=np.float32).copy()
    prev_attached_box_pose_tcp = None if getattr(demo, "attached_box_pose_tcp", None) is None else np.asarray(demo.attached_box_pose_tcp, dtype=np.float32).copy()
    prev_attached_box_visible = bool(getattr(demo, "_attached_box_visual_visible", False))
    prev_attached_object_visual_active = bool(getattr(demo, "_attached_object_visual_active", False))
    try:
        T_tcp_obj_override = _tcp_to_object_transform_from_pose(demo, grasp_pose)
        _register_transport_attached_box_for_probe(demo, args, T_tcp_obj_override)
        base.sync_demo_arm_qpos(demo, q_grasp)
        place_plan_candidates = build_targeted_place_plan_variants(
            demo,
            bridge_mod,
            scene_capture_cache,
            rule,
            place_state_cache,
            args,
            T_tcp_obj_override=T_tcp_obj_override,
        )
        if not place_plan_candidates:
            print(f"[grasp lookahead] {variant_label}: no targeted-place candidate could be built")
            return False

        probe_top_k = int(max(getattr(args, "grasp_placeability_probe_top_k", 2), 1))
        probe_planning_time = float(max(getattr(args, "grasp_placeability_probe_planning_time", 2.0), 0.25))
        probe_attempt_count = int(max(getattr(args, "grasp_placeability_probe_attempt_count", 1), 1))
        for candidate in place_plan_candidates[:probe_top_k]:
            slot_suffix = f", slot={candidate.slot_name}" if candidate.slot_name else ""
            variant_suffix = f", variant={candidate.variant_label}" if candidate.variant_label else ""
            print(
                f"[grasp lookahead] probing {variant_label}: "
                f"target={candidate.target_name}{slot_suffix}{variant_suffix}, "
                f"planning_time={probe_planning_time:.2f}, attempts={probe_attempt_count}"
            )
            q_place_candidate, _ = _plan_direct_place_candidate(
                demo,
                candidate,
                args,
                label_prefix=f"grasp_place_probe_{variant_label}",
                start_q=np.asarray(q_grasp, dtype=np.float32).reshape(-1)[:7],
                planning_time_override=probe_planning_time,
                attempt_count_override=probe_attempt_count,
                probe_only=True,
            )
            if q_place_candidate is not None:
                print(f"[grasp lookahead] {variant_label}: found a reachable direct-place candidate")
                return True
        print(f"[grasp lookahead] {variant_label}: no reachable direct-place candidate from the predicted grasp state")
        return False
    finally:
        demo.attached_box_size = prev_attached_box_size
        demo.attached_box_pose_tcp = prev_attached_box_pose_tcp
        demo._attached_box_visual_visible = prev_attached_box_visible
        demo._attached_object_visual_active = prev_attached_object_visual_active
        base.update_attached_box_visual(demo, visible=prev_attached_box_visible)
        base.sync_demo_arm_qpos(demo, q_saved)


def run_targeted_place_episode(demo, bridge_mod, real_exec, args, scene_capture_cache, place_state_cache) -> bool:
    print("\n[episode] planning from FoundationPose-initialized object pose")
    args._skip_remaining_step_confirms_in_object = False

    start_q = demo.current_arm_qpos()
    if real_exec is not None and bool(getattr(args, "single_confirm_per_object", False)):
        if not base.begin_single_confirm_window_for_object(demo, bridge_mod, args):
            print("[abort] user cancelled before executing this object's pick-place sequence")
            return False
    if real_exec is not None:
        print("\n[real robot setup]")
        if args.render_mode == "human":
            bridge_mod.render_preview(demo.env, repeats=5)
        ok, q_sent = base.align_real_robot_to_sim_start(demo, bridge_mod, real_exec, start_q, args)
        if not ok:
            print("[abort] failed to align the real robot to the simulation start pose")
            return False
        base.sync_demo_arm_qpos(demo, q_sent if q_sent is not None else start_q)
    else:
        print("\n[dry-run] --execute-real was not provided, so motions will only be planned and previewed")

    base.lift_active_object_above_table_if_needed(
        demo,
        args,
        min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
    )
    raw_grasp_pose = demo.build_topdown_grasp_pose()
    grasp_pose_variants = base.build_grasp_pose_variants(demo, raw_grasp_pose, args)
    use_grasp_placeability_lookahead = bool(getattr(args, "grasp_placeability_lookahead", False))
    selected_grasp_variant_label = None
    grasp_pose = None
    pregrasp_pose = None
    fallback_grasp_variant_label = None
    fallback_grasp_pose = None
    fallback_pregrasp_pose = None
    for grasp_variant_label, grasp_variant_pose in grasp_pose_variants:
        current_grasp_pose = grasp_variant_pose
        current_pregrasp_pose = demo.build_pregrasp_pose(current_grasp_pose)
        current_grasp_pose, current_pregrasp_pose, geometry_grasp_raise = base.enforce_topdown_grasp_insertion_limit(
            demo,
            args,
            current_grasp_pose,
            current_pregrasp_pose,
        )
        if geometry_grasp_raise > 0:
            print(
                f"[safety] raised grasp/pregrasp TCP z by {geometry_grasp_raise:.4f} m "
                f"to satisfy topdown_grasp_max_insertion_depth={args.topdown_grasp_max_insertion_depth:.4f}"
            )
        current_grasp_pose, current_pregrasp_pose, grasp_tcp_raise = base.enforce_min_grasp_tcp_z(
            current_grasp_pose,
            current_pregrasp_pose,
            args.min_grasp_tcp_z,
        )
        if grasp_tcp_raise > 0:
            print(
                f"[safety] raised grasp/pregrasp TCP z by {grasp_tcp_raise:.4f} m "
                f"to satisfy min_grasp_tcp_z={args.min_grasp_tcp_z:.4f}"
            )

        if use_grasp_placeability_lookahead and grasp_variant_label != "grasp":
            print(f"[planner] trying grasp variant with placeability lookahead: {grasp_variant_label}")

        q_pre_preview_path = base.plan_pose_path(
            demo,
            current_pregrasp_pose,
            variant_name=args.variant,
            label="pregrasp_probe" if grasp_variant_label == "grasp" else f"pregrasp_probe_{grasp_variant_label}",
        )
        q_pre_preview = None
        if q_pre_preview_path is not None and len(q_pre_preview_path) > 0:
            q_pre_preview = np.asarray(q_pre_preview_path[-1], dtype=np.float32).reshape(-1)[:7]
        else:
            q_pre_preview = demo.plan_terminal_q(current_pregrasp_pose, variant_name=args.variant)
            if q_pre_preview is not None:
                q_pre_preview = np.asarray(q_pre_preview, dtype=np.float32).reshape(-1)[:7]
        if q_pre_preview is None:
            print(
                f"[grasp lookahead] {grasp_variant_label}: pregrasp is unreachable, "
                "trying the next grasp candidate"
            )
            continue

        q_saved = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
        try:
            base.sync_demo_arm_qpos(demo, q_pre_preview)
            q_grasp_preview = demo.plan_terminal_q(current_grasp_pose, variant_name=args.variant)
        finally:
            base.sync_demo_arm_qpos(demo, q_saved)
        if q_grasp_preview is None:
            print(
                f"[grasp lookahead] {grasp_variant_label}: grasp IK is unreachable from pregrasp, "
                "trying the next grasp candidate"
            )
            continue
        q_grasp_preview = np.asarray(q_grasp_preview, dtype=np.float32).reshape(-1)[:7]
        if fallback_grasp_pose is None:
            fallback_grasp_variant_label = grasp_variant_label
            fallback_grasp_pose = current_grasp_pose
            fallback_pregrasp_pose = current_pregrasp_pose

        if use_grasp_placeability_lookahead:
            if not _probe_grasp_variant_placeability(
                demo,
                bridge_mod,
                scene_capture_cache,
                get_place_rule(args.object_name),
                place_state_cache,
                args,
                grasp_pose=current_grasp_pose,
                q_grasp=q_grasp_preview,
                variant_label=grasp_variant_label,
            ):
                continue

        selected_grasp_variant_label = grasp_variant_label
        grasp_pose = current_grasp_pose
        pregrasp_pose = current_pregrasp_pose
        break

    if grasp_pose is None or pregrasp_pose is None:
        if fallback_grasp_pose is not None and fallback_pregrasp_pose is not None:
            selected_grasp_variant_label = fallback_grasp_variant_label
            grasp_pose = fallback_grasp_pose
            pregrasp_pose = fallback_pregrasp_pose
            if use_grasp_placeability_lookahead:
                print(
                    "[warn] no grasp candidate passed the pre-placeability lookahead; "
                    "falling back to the first reachable grasp candidate"
                )
        else:
            if use_grasp_placeability_lookahead:
                print("[FAIL] no grasp candidate was reachable even before placeability lookahead")
            else:
                print("[FAIL] no grasp candidate was reachable")
            base.inspect_failed_pose(demo, bridge_mod, "grasp_lookahead", args, pose=raw_grasp_pose, gripper_closed=False)
            return False

    print("\n[poses]")
    print("object p:", np.round(demo.get_obj_pose()[0], 6), "object q:", np.round(demo.get_obj_pose()[1], 6))
    print("tcp grasp p:", np.round(base.flatten_np(grasp_pose.p)[:3], 6), "q:", np.round(base.flatten_np(grasp_pose.q)[:4], 6))
    print("tcp pregrasp p:", np.round(base.flatten_np(pregrasp_pose.p)[:3], 6), "q:", np.round(base.flatten_np(pregrasp_pose.q)[:4], 6))

    print("\n[move to pregrasp]")
    demo.preview_target_pose(pregrasp_pose)
    if args.render_mode == "human":
        bridge_mod.render_preview(demo.env, repeats=10)
    q_pre_path = base.plan_pose_path(
        demo,
        pregrasp_pose,
        variant_name=args.variant,
        label="pregrasp",
    )
    if q_pre_path is not None:
        ok, _ = base.execute_pose_path_stage(demo, bridge_mod, real_exec, "pregrasp", pregrasp_pose, q_pre_path, args.real_gripper_open, args)
    else:
        print("[planner] pregrasp pose-path planning failed; falling back to terminal joint target")
        q_pre = demo.plan_terminal_q(pregrasp_pose, variant_name=args.variant)
        if q_pre is None:
            print("[FAIL] pregrasp planning failed")
            base.inspect_failed_pose(demo, bridge_mod, "pregrasp", args, pose=pregrasp_pose, gripper_closed=False)
            return False
        ok, _ = base.execute_stage(demo, bridge_mod, real_exec, "pregrasp", pregrasp_pose, q_pre, args.real_gripper_open, args)
    if not ok:
        ok, _ = base.retry_pregrasp_after_escape(
            demo,
            bridge_mod,
            real_exec,
            "pregrasp",
            pregrasp_pose,
            args.real_gripper_open,
            args,
        )
        if not ok:
            return False

    print("\n[move to grasp]")
    if selected_grasp_variant_label not in (None, "grasp"):
        if use_grasp_placeability_lookahead:
            print(f"[planner] selected grasp variant after placeability lookahead: {selected_grasp_variant_label}")
        else:
            print(f"[planner] selected grasp variant: {selected_grasp_variant_label}")
    grasp_plan = base.plan_grasp_pose_with_fallbacks(demo, bridge_mod, grasp_pose, args)
    if grasp_plan is None:
        print("[FAIL] grasp planning failed")
        base.inspect_failed_pose(demo, bridge_mod, "grasp", args, pose=grasp_pose, gripper_closed=False)
        return False
    grasp_label = grasp_plan.label
    selected_grasp_pose = grasp_plan.final_grasp_pose
    if grasp_label != "grasp":
        print(f"[planner] selected grasp fallback approach: {grasp_label}")

    if bool(getattr(grasp_plan, "require_runtime_true_grasp_planning", False)):
        approach_path = base.plan_pose_path(
            demo,
            grasp_plan.approach_pose,
            variant_name=args.variant,
            label=grasp_label or "grasp",
        )
        if approach_path is not None:
            ok, _ = base.execute_pose_path_stage(
                demo,
                bridge_mod,
                real_exec,
                grasp_label or "grasp",
                grasp_plan.approach_pose,
                approach_path,
                args.real_gripper_open,
                args,
            )
        else:
            print(f"[planner] {grasp_label or 'grasp'} pose-path planning failed; falling back to terminal joint target")
            ok, _ = base.execute_stage(
                demo,
                bridge_mod,
                real_exec,
                grasp_label or "grasp",
                grasp_plan.approach_pose,
                grasp_plan.approach_q,
                args.real_gripper_open,
                args,
            )
        if not ok:
            return False

        q_grasp_path = base.plan_pose_path(
            demo,
            selected_grasp_pose,
            variant_name=args.variant,
            label=f"{grasp_label}_to_true_grasp",
        )
        if q_grasp_path is not None:
            ok, _ = base.execute_pose_path_stage(
                demo,
                bridge_mod,
                real_exec,
                f"{grasp_label}_to_true_grasp",
                selected_grasp_pose,
                q_grasp_path,
                args.real_gripper_open,
                args,
            )
        else:
            print(f"[planner] {grasp_label}_to_true_grasp pose-path planning failed after executing the fallback approach; trying direct terminal joint target")
            q_grasp = demo.plan_terminal_q(selected_grasp_pose, variant_name=args.variant)
            if q_grasp is None:
                print(f"[FAIL] {grasp_label}_to_true_grasp planning failed from the executed fallback approach")
                base.inspect_failed_pose(demo, bridge_mod, "grasp", args, pose=selected_grasp_pose, gripper_closed=False)
                return False
            ok, _ = base.execute_stage(
                demo,
                bridge_mod,
                real_exec,
                f"{grasp_label}_to_true_grasp",
                selected_grasp_pose,
                q_grasp,
                args.real_gripper_open,
                args,
            )
    elif grasp_plan.final_grasp_path is not None:
        approach_path = base.plan_pose_path(
            demo,
            grasp_plan.approach_pose,
            variant_name=args.variant,
            label=grasp_label or "grasp",
        )
        if approach_path is not None:
            ok, _ = base.execute_pose_path_stage(
                demo,
                bridge_mod,
                real_exec,
                grasp_label or "grasp",
                grasp_plan.approach_pose,
                approach_path,
                args.real_gripper_open,
                args,
            )
        else:
            print(f"[planner] {grasp_label or 'grasp'} pose-path planning failed; falling back to terminal joint target")
            ok, _ = base.execute_stage(
                demo,
                bridge_mod,
                real_exec,
                grasp_label or "grasp",
                grasp_plan.approach_pose,
                grasp_plan.approach_q,
                args.real_gripper_open,
                args,
            )
        if not ok:
            return False
        ok, _ = base.execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            f"{grasp_label}_to_true_grasp",
            selected_grasp_pose,
            grasp_plan.final_grasp_path,
            args.real_gripper_open,
            args,
        )
    else:
        q_grasp_path = base.plan_pose_path(
            demo,
            selected_grasp_pose,
            variant_name=args.variant,
            label=grasp_label or "grasp",
        )
        if q_grasp_path is not None:
            ok, _ = base.execute_pose_path_stage(
                demo,
                bridge_mod,
                real_exec,
                grasp_label or "grasp",
                selected_grasp_pose,
                q_grasp_path,
                args.real_gripper_open,
                args,
            )
        else:
            print(f"[planner] {grasp_label or 'grasp'} pose-path planning failed; falling back to terminal joint target")
            ok, _ = base.execute_stage(
                demo,
                bridge_mod,
                real_exec,
                grasp_label or "grasp",
                selected_grasp_pose,
                grasp_plan.approach_q,
                args.real_gripper_open,
                args,
            )
    if not ok:
        return False

    print("\n[close gripper]")
    if not base.confirm_simple_action("close the real gripper", args, bridge_mod=bridge_mod, env=demo.env, repeats=6):
        print("[abort] user cancelled before closing the real gripper")
        return False
    if real_exec is not None:
        real_exec.set_gripper(args.real_gripper_close)
        base.sync_demo_gripper_state(demo, closed=True, steps=4)
        base.set_pregrasp_object_freeze(demo, False)
        gripper_pos, blocked = base.real_gripper_blocked_after_close(
            real_exec,
            close_cmd=args.real_gripper_close,
            blocked_margin=args.real_gripper_blocked_margin,
        )
        if gripper_pos is not None and blocked is not None:
            print(
                f"[real] gripper.pos after close: {gripper_pos:.4f} "
                f"blocked_before_full_close={blocked}"
            )
    else:
        print("[dry-run] skipped real gripper close")
        base.sync_demo_gripper_state(demo, closed=True, steps=4)
        base.set_pregrasp_object_freeze(demo, False)
    base.lift_active_object_above_table_if_needed(
        demo,
        args,
        min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
    )

    if args.skip_goal_motion:
        print("[done] skipped place motion as requested")
        return True

    rule = get_place_rule(args.object_name)
    if rule is None:
        print(f"[FAIL] no targeted-place rule is configured for source object {args.object_name}")
        return False

    print("\n[move to targeted place]")
    try:
        _ensure_target_registered_for_place(demo, rule.target_object_name)
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return False
    _register_transport_attached_box(demo, args)

    try:
        place_plan_candidates = build_targeted_place_plan_variants(
            demo,
            bridge_mod,
            scene_capture_cache,
            rule,
            place_state_cache,
            args,
        )
    except Exception as exc:
        print(f"[FAIL] failed to build the targeted place plan: {exc}")
        return False
    if not place_plan_candidates:
        print("[FAIL] no targeted place candidate could be built")
        return False

    place_plan = None
    q_staging_path = None
    q_pre_place_path = None
    q_place_path = None
    q_retreat_path = None
    direct_place_without_pre_place = False
    direct_place_from_current_pose = False
    place_execution_label = "place"
    last_staging_pose = None
    last_pre_place_pose = None
    last_place_pose = None
    if bool(getattr(args, "targeted_place_direct_first", False)):
        for candidate in place_plan_candidates:
            slot_suffix = f", slot={candidate.slot_name}" if candidate.slot_name else ""
            variant_suffix = f", variant={candidate.variant_label}" if candidate.variant_label else ""
            print(
                f"[place] trying direct-place-first from the current post-grasp pose: "
                f"target={candidate.target_name}{slot_suffix}{variant_suffix}, "
                f"tcp_verticality={candidate.tcp_verticality:.3f}"
            )
            print("[place] place p:", np.round(base.flatten_np(candidate.place_pose.p)[:3], 6), "q:", np.round(base.flatten_np(candidate.place_pose.q)[:4], 6))
            last_place_pose = candidate.place_pose
            q_place_candidate, q_retreat_candidate = _plan_direct_place_candidate(
                demo,
                candidate,
                args,
                label_prefix="place_direct_first",
            )
            if q_place_candidate is None:
                continue
            place_plan = candidate
            q_staging_path = None
            q_pre_place_path = None
            q_place_path = q_place_candidate
            q_retreat_path = q_retreat_candidate
            direct_place_without_pre_place = True
            direct_place_from_current_pose = True
            place_execution_label = "place_direct_first" if candidate.variant_label is None else f"place_direct_first_{candidate.variant_label}"
            print("[place] direct-place-first probe succeeded from the current post-grasp pose; skipping post_grasp_escape and pre_place")
            break

    if place_plan is None:
        if not _run_post_grasp_escape_for_place(demo, bridge_mod, real_exec, args, "post_grasp_escape", use_attach=True):
            print("[warn] post-grasp escape planning failed; continuing directly to the targeted place approach")
        try:
            place_plan_candidates = build_targeted_place_plan_variants(
                demo,
                bridge_mod,
                scene_capture_cache,
                rule,
                place_state_cache,
                args,
            )
        except Exception as exc:
            print(f"[FAIL] failed to rebuild the targeted place plan after post_grasp_escape: {exc}")
            return False
        if not place_plan_candidates:
            print("[FAIL] no targeted place candidate could be rebuilt after post_grasp_escape")
            return False
    for candidate in ([] if place_plan is not None else place_plan_candidates):
        slot_suffix = f", slot={candidate.slot_name}" if candidate.slot_name else ""
        variant_suffix = f", variant={candidate.variant_label}" if candidate.variant_label else ""
        print(
            f"[place] source={args.object_name}, primitive={rule.primitive}, "
            f"target={candidate.target_name}{slot_suffix}{variant_suffix}, "
            f"tcp_verticality={candidate.tcp_verticality:.3f}"
        )
        if candidate.staging_pose is not None:
            print("[place] staging p:", np.round(base.flatten_np(candidate.staging_pose.p)[:3], 6), "q:", np.round(base.flatten_np(candidate.staging_pose.q)[:4], 6))
        print("[place] pre_place p:", np.round(base.flatten_np(candidate.pre_place_pose.p)[:3], 6), "q:", np.round(base.flatten_np(candidate.pre_place_pose.q)[:4], 6))
        print("[place] place p:", np.round(base.flatten_np(candidate.place_pose.p)[:3], 6), "q:", np.round(base.flatten_np(candidate.place_pose.q)[:4], 6))
        q_staging_path = None
        stage_start_q = None
        if candidate.staging_pose is not None:
            last_staging_pose = candidate.staging_pose
            q_staging_path = _plan_pose_with_ranked_ik_joint_paths(
                demo,
                candidate.staging_pose,
                args,
                label="pre_place_staging" if candidate.variant_label is None else f"pre_place_staging_{candidate.variant_label}",
                phase="pre_place_staging",
                use_attach=True,
                planning_time=args.fixed_goal_planning_time,
                rrt_range=args.fixed_goal_rrt_range,
            )
            if q_staging_path is None:
                print(
                    f"[place] staging planning failed for variant={candidate.variant_label}; "
                    "trying direct pre_place planning for the same place target"
                )
            else:
                stage_start_q = np.asarray(q_staging_path[-1], dtype=np.float32).reshape(-1)[:7]
        last_pre_place_pose = candidate.pre_place_pose
        q_pre_place_path = _plan_pose_with_ranked_ik_joint_paths(
            demo,
            candidate.pre_place_pose,
            args,
            label="pre_place" if candidate.variant_label is None else f"pre_place_{candidate.variant_label}",
            phase="pre_place",
            use_attach=True,
            planning_time=args.fixed_goal_planning_time,
            rrt_range=args.fixed_goal_rrt_range,
            start_q=stage_start_q,
        )
        if q_pre_place_path is not None:
            q_eval_saved = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
            try:
                base.sync_demo_arm_qpos(demo, np.asarray(q_pre_place_path[-1], dtype=np.float32).reshape(-1)[:7])
                last_place_pose = candidate.place_pose
                q_place_candidate = base.plan_lift_path(
                    demo,
                    candidate.place_pose,
                    variant_name=args.variant,
                    use_attach=True,
                    label="place" if candidate.variant_label is None else f"place_{candidate.variant_label}",
                    planning_time=min(float(args.fixed_goal_planning_time), 3.0),
                    rrt_range=float(args.fixed_goal_rrt_range),
                    start_q=np.asarray(q_pre_place_path[-1], dtype=np.float32).reshape(-1)[:7],
                    allow_pose_rrt_fallback=True,
                    max_segment_joint_delta=0.35,
                    max_segment_joint7_delta=0.80,
                    max_segment_norm_delta=0.60,
                )
                if q_place_candidate is None:
                    print(
                        f"[place] place descent planning failed for variant={candidate.variant_label}; "
                        "trying the next place candidate before executing anything"
                    )
                    continue

                q_retreat_candidate = None
                try:
                    base.sync_demo_arm_qpos(demo, np.asarray(q_place_candidate[-1], dtype=np.float32).reshape(-1)[:7])
                    q_retreat_candidate = base.plan_lift_path(
                        demo,
                        candidate.retreat_pose,
                        variant_name=args.variant,
                        use_attach=False,
                        label="post_place_retreat" if candidate.variant_label is None else f"post_place_retreat_{candidate.variant_label}",
                        planning_time=min(float(args.fixed_goal_planning_time), 3.0),
                        rrt_range=float(args.fixed_goal_rrt_range),
                        start_q=np.asarray(q_place_candidate[-1], dtype=np.float32).reshape(-1)[:7],
                        allow_pose_rrt_fallback=False,
                        max_segment_joint_delta=0.45,
                        max_segment_joint7_delta=0.45,
                        max_segment_norm_delta=0.75,
                    )
                finally:
                    base.sync_demo_arm_qpos(demo, q_eval_saved)

                place_plan = candidate
                q_place_path = q_place_candidate
                q_retreat_path = q_retreat_candidate
                direct_place_without_pre_place = False
                place_execution_label = "place" if candidate.variant_label is None else f"place_{candidate.variant_label}"
                break
            finally:
                base.sync_demo_arm_qpos(demo, q_eval_saved)
        else:
            print(
                f"[place] pre_place planning failed for variant={candidate.variant_label}; "
                "trying a direct place plan to the final pose"
            )
            last_place_pose = candidate.place_pose
            q_place_candidate, q_retreat_candidate = _plan_direct_place_candidate(
                demo,
                candidate,
                args,
                label_prefix="place_direct",
                start_q=stage_start_q,
            )
            if q_place_candidate is None:
                if candidate.variant_label is not None:
                    print(f"[place] direct place planning also failed for variant={candidate.variant_label}; trying the next equivalent insertion orientation")
                continue

            place_plan = candidate
            q_pre_place_path = None
            q_place_path = q_place_candidate
            q_retreat_path = q_retreat_candidate
            direct_place_without_pre_place = True
            direct_place_from_current_pose = False
            place_execution_label = "place_direct" if candidate.variant_label is None else f"place_direct_{candidate.variant_label}"
            break
        if candidate.variant_label is not None:
            print(f"[place] pre_place planning failed for variant={candidate.variant_label}; trying the next equivalent insertion orientation")

    if place_plan is None or q_place_path is None:
        print("[FAIL] pre_place planning failed")
        if last_staging_pose is not None:
            print("[place] last staging p:", np.round(base.flatten_np(last_staging_pose.p)[:3], 6), "q:", np.round(base.flatten_np(last_staging_pose.q)[:4], 6))
        base.inspect_failed_pose(
            demo,
            bridge_mod,
            "place" if last_place_pose is not None else "pre_place",
            args,
            pose=last_place_pose if last_place_pose is not None else last_pre_place_pose,
            gripper_closed=True,
            use_attach=True,
        )
        return False
    if place_plan.staging_pose is not None and q_staging_path:
        ok, _ = base.execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            "pre_place_staging" if place_plan.variant_label is None else f"pre_place_staging_{place_plan.variant_label}",
            place_plan.staging_pose,
            q_staging_path,
            args.real_gripper_close,
            args,
            use_attach=True,
        )
        if not ok:
            return False
    if q_pre_place_path is not None:
        ok, _ = base.execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            "pre_place" if place_plan.variant_label is None else f"pre_place_{place_plan.variant_label}",
            place_plan.pre_place_pose,
            q_pre_place_path,
            args.real_gripper_close,
            args,
            use_attach=True,
        )
        if not ok:
            return False
    elif direct_place_without_pre_place and direct_place_from_current_pose:
        print("[place] direct place from the current post-grasp pose succeeded; skipped post_grasp_escape and pre_place")
    elif direct_place_without_pre_place:
        print("[place] pre_place was unreachable; using direct place fallback to the final pose")

    relaxed_target_collision = False
    if rule.primitive == "insert_vertical":
        relaxed_target_collision = _set_scene_obstacle_planner_box_scale(
            demo,
            place_plan.target_name,
            float(args.place_insert_target_collision_scale),
        )

    ok, _ = base.execute_pose_path_stage(
        demo,
        bridge_mod,
        real_exec,
        place_execution_label,
        place_plan.place_pose,
        q_place_path,
        args.real_gripper_close,
        args,
        use_attach=True,
    )
    if not ok:
        if relaxed_target_collision:
            _set_scene_obstacle_planner_box_scale(demo, place_plan.target_name, 1.0)
        return False

    print("\n[open gripper at place]")
    if not base.confirm_simple_action("open the real gripper at the targeted place", args, bridge_mod=bridge_mod, env=demo.env, repeats=6):
        print("[abort] user cancelled before opening the real gripper at the targeted place")
        if relaxed_target_collision:
            _set_scene_obstacle_planner_box_scale(demo, place_plan.target_name, 1.0)
        return False
    if real_exec is not None:
        real_exec.set_gripper(args.real_gripper_open)
        base.sync_demo_gripper_state(demo, closed=False, steps=4)
    else:
        print("[dry-run] skipped real gripper open at the targeted place")
    demo._attached_box_visual_visible = False
    demo._attached_object_visual_active = False
    base.update_attached_box_visual(demo, visible=False)
    base.settle_released_object_after_open(demo, args, label="targeted_place_release")
    _mark_place_rule_success(rule, place_state_cache, place_plan.slot_name)

    if q_retreat_path is None:
        if relaxed_target_collision:
            _set_scene_obstacle_planner_box_scale(demo, place_plan.target_name, 1.0)
        print("[warn] post-place retreat planning failed after the object was already released; keeping the placement result and ending this cycle without retreat")
        print("[done] completed one targeted place motion; post-place retreat was skipped")
        return True
    ok, _ = base.execute_pose_path_stage(
        demo,
        bridge_mod,
        real_exec,
        "post_place_retreat",
        place_plan.retreat_pose,
        q_retreat_path,
        args.real_gripper_open,
        args,
        use_attach=False,
    )
    if relaxed_target_collision:
        _set_scene_obstacle_planner_box_scale(demo, place_plan.target_name, 1.0)
    if not ok:
        print("[warn] post-place retreat execution failed after the object was already released; keeping the placement result and ending this cycle without retreat")
        print("[done] completed one targeted place motion; post-place retreat was skipped")
        return True

    print("[done] completed one targeted place motion and retreated after release")
    return True


def _derive_cycle_obstacle_names(base_args, cycle_idx: int, selected_name: str, cycle_object_sequence, cached_scene_names) -> list[str]:
    selected_name = normalize_object_name(selected_name)
    if cycle_idx > 1 and cached_scene_names:
        return [name for name in cached_scene_names if name != selected_name]

    future_cycle_sources = [name for name in list(cycle_object_sequence or []) if normalize_object_name(name) != selected_name]
    tracked_scene_names = list(getattr(base_args, "tracked_scene_object_names", []) or [])
    explicit_names = list(getattr(base_args, "selected_obstacle_object_names", []) or [])
    current_rule_target_names = get_required_place_target_names([selected_name])
    future_rule_target_names = get_required_place_target_names(future_cycle_sources)
    return _dedupe_names(explicit_names + tracked_scene_names + future_cycle_sources + current_rule_target_names + future_rule_target_names)


def _list_cached_unplaced_rule_names(scene_capture_cache) -> list[str]:
    if not isinstance(scene_capture_cache, dict):
        return []
    objects = scene_capture_cache.get("objects")
    if not isinstance(objects, dict):
        return []
    names = []
    for raw_name, entry in objects.items():
        name = normalize_object_name(raw_name)
        if name is None:
            continue
        if get_place_rule(name) is None:
            continue
        if isinstance(entry, dict) and bool(entry.get("placed", False)):
            continue
        names.append(name)
    return sorted(dict.fromkeys(names))


def main():
    args = parse_args()
    maybe_print_and_exit_place_rules(args)
    base.maybe_print_and_exit_object_specs(args)
    if int(args.repeat_count) < 1:
        raise ValueError("--repeat-count must be >= 1")

    base_args = argparse.Namespace(**vars(args).copy())
    base_args.object_name = base.resolve_object_spec_name(base_args.object_name) if base_args.object_name else None
    if base_args.selected_obstacle_object_names is not None:
        base_args.selected_obstacle_object_names = base.resolve_object_spec_name_list(base_args.selected_obstacle_object_names)
    if base_args.tracked_scene_object_names is not None:
        base_args.tracked_scene_object_names = base.resolve_object_spec_name_list(base_args.tracked_scene_object_names)

    cycle_object_sequence = base.resolve_object_spec_name_list(base_args.cycle_object_names) if base_args.cycle_object_names else []
    if cycle_object_sequence:
        _validate_cycle_sources_have_place_rules(cycle_object_sequence)
    if base_args.object_name is not None:
        _validate_cycle_sources_have_place_rules([base_args.object_name])
    if cycle_object_sequence and not base_args.repeat_forever:
        base_args.repeat_count = max(int(base_args.repeat_count), len(cycle_object_sequence))

    bridge_mod = base.load_module_from_path("jiaobang_fp_bridge_targeted", args.bridge_script_path)
    planner_mod = base.load_module_from_path("jiaobang_planner_impl_targeted", args.pick_script_path)

    print(f"Using bridge script: {Path(args.bridge_script_path).resolve()}")
    print(f"Using planner script: {Path(args.pick_script_path).resolve()}")
    print(f"Using camera extrinsic from: {args.camera_extrinsic_opencv_path}")
    if args.repeat_forever:
        print("Repeat mode: forever")
    else:
        print(f"Repeat mode: {base_args.repeat_count} cycle(s)")
    if cycle_object_sequence:
        print(f"Planned cycle sequence: {cycle_object_sequence}")
    elif base_args.object_name is not None:
        print(f"Initial target object: {base_args.object_name}")
    print("Targeted place rules:")
    print(describe_place_rules() or "(none)")

    real_exec = None
    final_ok = True
    cycle_idx = 0
    scene_capture_cache: dict | None = {} if bool(getattr(args, "reuse_foundationpose_scene_across_cycles", True)) else None
    place_state_cache: dict = {"used_slots_by_target": {}}
    try:
        if args.execute_real:
            real_exec = base.RealmanJointExecutor(args)
            if args.reset_real_before_start:
                print("\n[real robot pre-reset]")
                if not base.confirm_simple_action("reset the real robot to its hardware home pose before FoundationPose initialization", args):
                    print("[abort] user cancelled before the pre-FoundationPose real robot reset")
                    return
                real_exec.reset_robot(gripper_pos=args.real_gripper_open)

        while True:
            cycle_idx += 1
            env = None
            ok = False
            cached_scene_names = base.list_cached_scene_object_names(scene_capture_cache)
            available_rule_names = _list_cached_unplaced_rule_names(scene_capture_cache)
            if cycle_idx <= len(cycle_object_sequence):
                selected_name = cycle_object_sequence[cycle_idx - 1]
                print(f"\n[cycle {cycle_idx}] using CLI target object: {selected_name}")
            elif cycle_idx == 1 and base_args.object_name is not None:
                selected_name = base_args.object_name
                print(f"\n[cycle {cycle_idx}] using CLI target object: {selected_name}")
            elif cycle_idx > 1 and available_rule_names:
                print(f"\n[cycle {cycle_idx}] reusing the cached tabletop scene; choose the next target from the remaining rule-enabled objects")
                selected_name = base.prompt_cycle_object_name(
                    base_args,
                    cycle_idx,
                    available_names=available_rule_names,
                    default_name=available_rule_names[0],
                )
            else:
                selected_name = base.prompt_cycle_object_name(
                    base_args,
                    cycle_idx,
                    available_names=list_place_rule_sources(),
                    default_name=(base_args.object_name or (list_place_rule_sources()[0] if list_place_rule_sources() else None)),
                )
            if selected_name is None:
                final_ok = False
                print(f"[abort] user cancelled object selection for cycle {cycle_idx}")
                break
            rule = get_place_rule(selected_name)
            if rule is None:
                final_ok = False
                print(f"[abort] no targeted-place rule is configured for {selected_name}")
                break

            cycle_args, spec = base.make_cycle_args(base_args, selected_name)
            selected_obstacles = _derive_cycle_obstacle_names(base_args, cycle_idx, selected_name, cycle_object_sequence, cached_scene_names)
            cycle_args.selected_obstacle_object_names = list(selected_obstacles)
            cycle_args.required_scene_object_names = list(selected_obstacles)
            print(f"\n[cycle {cycle_idx}] using object spec: {spec.name}")
            print(f"[cycle {cycle_idx}] mesh file: {cycle_args.mesh_file}")
            print(f"[cycle {cycle_idx}] simulation asset file: {cycle_args.sim_asset_file}")
            print(f"[cycle {cycle_idx}] simulation asset scale: {cycle_args.sim_asset_scale}")
            print(f"[cycle {cycle_idx}] GroundingDINO target: {cycle_args.target_object_name}")
            print(f"[cycle {cycle_idx}] targeted place: {rule.primitive} -> {rule.target_object_name}")
            print(f"[cycle {cycle_idx}] selected obstacles: {cycle_args.selected_obstacle_object_names}")
            print(f"\n================ cycle {cycle_idx} ================")
            try:
                env, demo = base.create_demo(cycle_args, bridge_mod, planner_mod, scene_capture_cache=scene_capture_cache)
                base.register_place_slot_markers(demo, rule, cycle_args)
                ok = run_targeted_place_episode(demo, bridge_mod, real_exec, cycle_args, scene_capture_cache, place_state_cache)
            finally:
                base.close_env_quietly(env)
                gc.collect()
            print(f"\ncycle {cycle_idx} success = {ok}")
            if not ok:
                final_ok = False
                break
            base.cache_successfully_placed_object_world_pose(demo, cycle_args.object_name, cycle_args)
            if not base_args.repeat_forever and cycle_idx >= int(base_args.repeat_count):
                break
            if real_exec is not None:
                real_exec.set_gripper(args.real_gripper_open)
                print(f"\n[cycle {cycle_idx}] keeping the current real robot pose; the next cycle will continue planning from the current arm configuration")
            print(f"[cycle {cycle_idx}] ready for the next cycle")

        print("\nfinal success =", final_ok)
    finally:
        if real_exec is not None:
            real_exec.close()


if __name__ == "__main__":
    main()
