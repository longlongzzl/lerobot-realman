#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import random
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
    T_world_obj_desired: np.ndarray | None
    staging_pose: Pose | None
    pre_place_pose: Pose
    place_pose: Pose
    retreat_pose: Pose
    tcp_verticality: float = 0.0


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
        default=[0.0],
        help="For insert-style place rules, also try these equivalent rotations around the source object's long axis when solving pre_place/place IK. Default keeps the final object pose fixed.",
    )
    parser.add_argument(
        "--targeted-place-allow-insert-axis-flip",
        dest="targeted_place_allow_insert_axis_flip",
        action="store_true",
        default=False,
        help="Also try the 180-degree long-axis flip family for insert-style place rules. Disabled by default to keep the final object pose fixed.",
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
        default=0.04,
        help="Extra Z clearance above both the current TCP height and the destination hover pose when building the targeted-place staging pose.",
    )
    parser.add_argument(
        "--tabletop-place-tilt-toward-robot-deg",
        type=float,
        nargs="*",
        default=[0.0],
        help="For place_on_slots rules, also try these tabletop place tilt angles. Default keeps the final object pose fixed.",
    )
    parser.add_argument(
        "--tabletop-place-yaw-variant-deg",
        type=float,
        nargs="*",
        default=[0.0],
        help="For place_on_slots rules, also try these extra in-plane yaw rotations around the destination tabletop normal. Default keeps the final object pose fixed.",
    )
    parser.add_argument(
        "--carriot-tabletop-place-yaw-variant-deg",
        type=float,
        nargs="*",
        default=None,
        help=(
            "For carriot place_on_slots, try these relative in-plane yaw offsets around the destination tabletop normal. "
            "When omitted, carriot keeps the canonical place-rule yaw."
        ),
    )
    parser.add_argument(
        "--tabletop-place-axial-spin-deg",
        type=float,
        nargs="*",
        default=[0.0],
        help="For elongated place_on_slots objects, also try these equivalent rotations around the object's own longest axis before solving pre_place/place IK. Default keeps the final object pose fixed.",
    )
    parser.add_argument(
        "--targeted-place-expand-orientation-invariant",
        dest="targeted_place_expand_orientation_invariant",
        action="store_true",
        default=False,
        help="Expand orientation-invariant rules into multiple equivalent final object orientations. Disabled by default to keep the final object pose fixed.",
    )
    parser.add_argument(
        "--tabletop-place-min-tcp-verticality",
        type=float,
        default=0.55,
        help="For tabletop place_on_slots rules, when any candidate keeps the TCP approach axis at least this aligned with the tabletop normal, reject flatter/horizontal TCP candidates and try the more vertical ones first.",
    )
    parser.add_argument(
        "--targeted-place-slot-order",
        type=str,
        nargs="*",
        default=None,
        help=(
            "For place_on_slots rules, prefer these slot numbers or names first, for example "
            "'3' or 'slot_3 slot_4'. Any unspecified slots remain available after the preferred order."
        ),
    )
    parser.add_argument(
        "--targeted-place-source-slot-map",
        type=str,
        nargs="*",
        default=None,
        help=(
            "For place_on_slots rules, bind source objects to exact tabletop slots, for example "
            "'gluestick:3 tennis:slot_4'. Insert-style rules such as bi->bitong ignore this."
        ),
    )
    parser.add_argument(
        "--target-selection-order",
        choices=("cycle", "random", "risk_aware"),
        default="risk_aware",
        help=(
            "How to choose among rule-enabled targets. cycle follows --cycle-object-names order; "
            "risk_aware keeps randomness inside priority groups but tries small/easily-blocked objects "
            "before bulky placed obstacles."
        ),
    )
    parser.add_argument(
        "--targeted-place-hover-extra-height-m",
        type=float,
        nargs="*",
        default=[0.0, 0.01],
        help="Additional world-z hover heights to try on top of each rule's base hover_height when building pre_place candidates.",
    )
    parser.add_argument(
        "--carry-sim-arm-across-cycles",
        dest="carry_sim_arm_across_cycles",
        action="store_true",
        default=True,
        help="When repeating cycles in simulation, carry the final simulated arm joint state into the next recreated env so planning continues from the previous cycle's end posture. Enabled by default.",
    )
    parser.add_argument(
        "--no-carry-sim-arm-across-cycles",
        dest="carry_sim_arm_across_cycles",
        action="store_false",
        help="Reset the simulated arm to the new env's default start state on every cycle.",
    )
    parser.add_argument(
        "--post-place-retreat-after-release",
        dest="post_place_retreat_after_release",
        action="store_true",
        default=False,
        help="After releasing the object, execute the planned retreat trajectory before ending the cycle.",
    )
    parser.add_argument(
        "--no-post-place-retreat-after-release",
        dest="post_place_retreat_after_release",
        action="store_false",
        help="Do not execute post-place retreat; keep the arm at the release pose so the next cycle plans from there. Enabled by default.",
    )
    return parser


def parse_args():
    return build_arg_parser().parse_args()


def _need_hidden_post_place_clearance(args) -> bool:
    return False


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


def _ordered_rule_slots(rule: PlaceRule, T_world_target: np.ndarray, bridge_mod, demo) -> list:
    slots = list(rule.slots or [])
    if not slots:
        return []
    T_world_target = np.asarray(T_world_target, dtype=np.float32).reshape(4, 4)
    robot_base_T = bridge_mod.get_robot_base_transform(demo.env)
    robot_base_p = None if robot_base_T is None else np.asarray(robot_base_T[:3, 3], dtype=np.float32).reshape(3)
    runtime_slots = get_runtime_slot_specs(
        rule.target_object_name,
        slots,
        T_world_target,
        robot_base_p,
    )
    if normalize_object_name(rule.target_object_name) == "desk" and runtime_slots:
        def _slot_index(slot: PlaceSlotSpec) -> int:
            try:
                return int(str(slot.name).split("_")[-1])
            except Exception:
                return 999

        return sorted(list(runtime_slots), key=_slot_index)

    robot_xy = None if robot_base_p is None else np.asarray(robot_base_p[:2], dtype=np.float32).reshape(2)
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


def _slot_name_from_user_token(token: object) -> str | None:
    text = str(token).strip()
    if not text:
        return None
    if text.lower().startswith("slot_"):
        return f"slot_{text.split('_')[-1]}"
    try:
        return f"slot_{int(text)}"
    except Exception:
        return text


def _apply_user_slot_order(slots: list, args) -> list:
    raw_order = list(getattr(args, "targeted_place_slot_order", None) or [])
    preferred_names = []
    for token in raw_order:
        name = _slot_name_from_user_token(token)
        if name is not None and name not in preferred_names:
            preferred_names.append(name)
    if not preferred_names:
        return slots

    by_name = {str(getattr(slot, "name", "")): slot for slot in slots}
    ordered = [by_name[name] for name in preferred_names if name in by_name]
    ordered_names = {str(getattr(slot, "name", "")) for slot in ordered}
    ordered.extend(slot for slot in slots if str(getattr(slot, "name", "")) not in ordered_names)
    return ordered


def _parse_user_source_slot_map(args) -> dict[str, str]:
    raw_items = list(getattr(args, "targeted_place_source_slot_map", None) or [])
    out: dict[str, str] = {}
    for raw in raw_items:
        text = str(raw).strip()
        if not text:
            continue
        if ":" in text:
            source_token, slot_token = text.split(":", 1)
        elif "=" in text:
            source_token, slot_token = text.split("=", 1)
        else:
            continue
        source_name = normalize_object_name(source_token.strip())
        slot_name = _slot_name_from_user_token(slot_token)
        if source_name is None or slot_name is None:
            continue
        out[source_name] = slot_name
    return out


def _apply_rule_slot_order(slots: list, args, rule: PlaceRule) -> list:
    source_name = normalize_object_name(rule.source_object_name) or str(rule.source_object_name)
    source_slot_map = _parse_user_source_slot_map(args)
    mapped_slot_name = source_slot_map.get(source_name)
    if mapped_slot_name:
        by_name = {str(getattr(slot, "name", "")): slot for slot in slots}
        mapped_slot = by_name.get(mapped_slot_name)
        return [mapped_slot] if mapped_slot is not None else []
    return _apply_user_slot_order(slots, args)


def _mark_place_rule_success(rule: PlaceRule, place_state_cache, slot_name: str | None = None) -> None:
    if rule.primitive != "place_on_slots":
        return
    target_key = normalize_object_name(rule.target_object_name) or str(rule.target_object_name)
    used_slots_by_target = place_state_cache.setdefault("used_slots_by_target", {})
    used_slots = used_slots_by_target.setdefault(target_key, [])
    if slot_name is not None and slot_name not in used_slots:
        used_slots.append(str(slot_name))


def _make_insert_vertical_local_pose_variants(
    rule: PlaceRule,
    object_pose_local: LocalPoseSpec,
    spin_degs,
    *,
    allow_axis_flip: bool = False,
) -> list[tuple[str | None, np.ndarray]]:
    T_base = _local_pose_spec_to_matrix(object_pose_local)
    variants: list[tuple[str | None, np.ndarray]] = [(None, T_base)]
    if rule.primitive != "insert_vertical":
        return variants
    spin_degs = [float(v) for v in list(spin_degs or []) if np.isfinite(float(v))]
    base_families: list[tuple[str | None, np.ndarray]] = [(None, T_base)]
    if bool(getattr(rule, "allow_long_axis_flip", False)) and bool(allow_axis_flip):
        T_flip = np.eye(4, dtype=np.float32)
        T_flip[:3, :3] = euler2mat(np.deg2rad(180.0), 0.0, 0.0, axes="sxyz").astype(np.float32)
        T_flipped = T_base.copy()
        T_flipped[:3, :3] = (T_base[:3, :3] @ T_flip[:3, :3]).astype(np.float32)
        base_families.append(("flip_180deg", T_flipped))

    seen = set()
    for family_label, T_family in base_families:
        key = tuple(np.round(T_family.reshape(-1), 6).tolist())
        if key not in seen:
            seen.add(key)
            variants.append((family_label, T_family) if family_label is not None else (None, T_family))
        for spin_deg in spin_degs:
            if abs(spin_deg) <= 1e-6:
                continue
            T_spin = np.eye(4, dtype=np.float32)
            T_spin[:3, :3] = euler2mat(0.0, np.deg2rad(float(spin_deg)), 0.0, axes="sxyz").astype(np.float32)
            T_variant = T_family.copy()
            T_variant[:3, :3] = (T_family[:3, :3] @ T_spin[:3, :3]).astype(np.float32)
            key = tuple(np.round(T_variant.reshape(-1), 6).tolist())
            if key in seen:
                continue
            seen.add(key)
            if family_label is None:
                label = f"spin_{int(round(spin_deg))}deg"
            else:
                label = f"{family_label}+spin_{int(round(spin_deg))}deg"
            variants.append((label, T_variant))
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

    rule_spin_degs = tuple(getattr(rule, "tabletop_axial_spin_deg", ()) or ())
    raw_spin_degs = rule_spin_degs if rule_spin_degs else getattr(args, "tabletop_place_axial_spin_deg", [])
    spin_degs = [float(v) for v in list(raw_spin_degs or []) if np.isfinite(float(v))]
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
    args,
    rule: PlaceRule,
    object_pose_local: LocalPoseSpec,
) -> list[tuple[str | None, np.ndarray]]:
    T_base = _local_pose_spec_to_matrix(object_pose_local)
    variants: list[tuple[str | None, np.ndarray]] = [(None, T_base)]
    if not bool(getattr(rule, "orientation_invariant", False)):
        return variants
    if not bool(getattr(args, "targeted_place_expand_orientation_invariant", False)):
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
    tcp_pose = demo.tcp.pose
    tcp_p = base.flatten_np(tcp_pose.p)[:3].copy()
    pre_place_p = base.flatten_np(pre_place_pose.p)[:3].copy()
    z_margin = float(max(getattr(args, "targeted_place_staging_z_margin", 0.04), 0.0))
    tcp_p[2] = max(float(tcp_p[2]), float(pre_place_p[2])) + z_margin
    return base.make_pose_with_position(pre_place_pose, tcp_p)


def _target_place_up_axis(rule: PlaceRule, T_world_target: np.ndarray) -> np.ndarray | None:
    if rule.primitive == "place_on_slots" and normalize_object_name(rule.target_object_name) == "desk":
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return _normalize(np.asarray(T_world_target[:3, 1], dtype=np.float32).reshape(3))


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

    source_name = normalize_object_name(getattr(rule, "source_object_name", None))
    carriot_relative_yaw_values = None
    if source_name == "carriot":
        raw_carriot_yaws = getattr(args, "carriot_tabletop_place_yaw_variant_deg", None)
        if raw_carriot_yaws is not None:
            carriot_relative_yaw_values = [
                float(v) for v in list(raw_carriot_yaws or []) if np.isfinite(float(v))
            ]
    use_carriot_relative_yaw = carriot_relative_yaw_values is not None

    if (
        bool(getattr(rule, "orientation_invariant", False))
        and not bool(getattr(rule, "allow_tabletop_yaw_variants", False))
        and not use_carriot_relative_yaw
    ):
        return [(None, np.asarray(T_world_obj_desired, dtype=np.float32).reshape(4, 4))]

    T_world_target = np.asarray(T_world_target, dtype=np.float32).reshape(4, 4)
    T_world_obj_desired = np.asarray(T_world_obj_desired, dtype=np.float32).reshape(4, 4)
    tilt_degs = [float(v) for v in list(getattr(args, "tabletop_place_tilt_toward_robot_deg", []) or []) if np.isfinite(float(v))]
    if use_carriot_relative_yaw:
        yaw_degs = list(carriot_relative_yaw_values or [])
    elif bool(getattr(rule, "allow_tabletop_yaw_variants", False)):
        yaw_degs = [float(v) for v in list(getattr(args, "tabletop_place_yaw_variant_deg", []) or []) if np.isfinite(float(v))]
    else:
        yaw_degs = [0.0]
    if bool(getattr(rule, "preserve_long_axis_vertical", False)):
        tilt_degs = [0.0]
    if not tilt_degs:
        tilt_degs = [0.0]
    if not yaw_degs:
        yaw_degs = [0.0]
    yaw_degs = sorted(yaw_degs, key=lambda d: (abs(float(d)), float(d)))

    up_axis = _target_place_up_axis(rule, T_world_target)
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

    T_world_obj_base = T_world_obj_desired.copy()
    if bool(getattr(rule, "allow_tabletop_yaw_variants", False)) and not use_carriot_relative_yaw:
        ref_axis = None
        face_axis_local = getattr(rule, "face_robot_axis_local", None)
        if face_axis_local is not None:
            face_axis_local = np.asarray(face_axis_local, dtype=np.float32).reshape(3)
            ref_axis = (T_world_obj_desired[:3, :3] @ face_axis_local).astype(np.float32)
            ref_axis = ref_axis - float(np.dot(ref_axis, up_axis)) * up_axis
            ref_axis = _normalize(ref_axis)
        else:
            best_dot = -float("inf")
            for axis_idx in (0, 2):
                axis_world = T_world_obj_desired[:3, axis_idx].astype(np.float32)
                axis_world = axis_world - float(np.dot(axis_world, up_axis)) * up_axis
                axis_world = _normalize(axis_world)
                if axis_world is None:
                    continue
                for sign in (1.0, -1.0):
                    signed_axis = (axis_world * float(sign)).astype(np.float32)
                    dot_score = float(np.dot(signed_axis, toward_robot))
                    if dot_score > best_dot:
                        best_dot = dot_score
                        ref_axis = signed_axis
        if ref_axis is not None:
            cross_val = float(np.dot(up_axis, np.cross(ref_axis, toward_robot)))
            dot_val = float(np.clip(np.dot(ref_axis, toward_robot), -1.0, 1.0))
            align_yaw_rad = float(np.arctan2(cross_val, dot_val))
            R_align = _axis_angle_to_matrix(up_axis, align_yaw_rad)
            T_world_obj_base[:3, :3] = (R_align @ T_world_obj_desired[:3, :3]).astype(np.float32)

    tilt_axis = _normalize(np.cross(up_axis, toward_robot))
    if tilt_axis is None:
        return [(None, T_world_obj_base)]

    local_points = base.get_asset_local_points(args.sim_asset_file, args.sim_asset_scale)
    plane_origin = np.asarray(T_world_target[:3, 3], dtype=np.float32).reshape(3)
    variants: list[tuple[str | None, np.ndarray]] = []
    seen = set()
    for yaw_deg in yaw_degs:
        if use_carriot_relative_yaw:
            yaw_label = None if abs(yaw_deg) <= 1e-6 else f"target_yaw_{int(round(yaw_deg))}deg"
        else:
            yaw_label = "face_robot" if abs(yaw_deg) <= 1e-6 else f"face_robot_yaw_{int(round(yaw_deg))}deg"
        T_yaw = T_world_obj_base.copy()
        if abs(yaw_deg) > 1e-6:
            R_yaw = _axis_angle_to_matrix(up_axis, np.deg2rad(float(yaw_deg)))
            T_yaw[:3, :3] = (R_yaw @ T_world_obj_base[:3, :3]).astype(np.float32)
        yaw_world_points = (T_yaw[:3, :3] @ local_points.T).T + T_yaw[:3, 3]
        yaw_bottom_along_up = float(np.min((yaw_world_points - plane_origin) @ up_axis))

        for tilt_deg in tilt_degs:
            if abs(tilt_deg) <= 1e-6:
                tilt_label = None
            elif tilt_deg > 0.0:
                tilt_label = f"tilt_toward_robot_{int(round(abs(tilt_deg)))}deg"
            else:
                tilt_label = f"tilt_away_robot_{int(round(abs(tilt_deg)))}deg"
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
    return variants or [("face_robot", T_world_obj_base)]


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
    target_name = normalize_object_name(rule.target_object_name)
    if target_name is None:
        raise RuntimeError(f"Invalid target object name in place rule: {rule.target_object_name!r}")
    T_world_target = _get_scene_object_world_transform(demo, bridge_mod, scene_capture_cache, target_name)
    if T_world_target is None:
        raise RuntimeError(
            f"Failed to resolve the world pose of destination object {target_name}. "
            "Make sure it is captured into the scene cache."
        )
    target_up_axis = _target_place_up_axis(rule, T_world_target)

    if T_tcp_obj_override is None:
        T_tcp_obj = _current_tcp_to_object_transform(demo)
    else:
        T_tcp_obj = np.asarray(T_tcp_obj_override, dtype=np.float32).reshape(4, 4)
    if rule.primitive == "place_on_slots":
        ordered_slots = _apply_rule_slot_order(_ordered_rule_slots(rule, T_world_target, bridge_mod, demo), args, rule)
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

    if rule.primitive == "insert_vertical":
        hover_extra_values = [0.0]
    else:
        hover_extra_values = [
            float(v)
            for v in list(getattr(args, "targeted_place_hover_extra_height_m", []) or [])
            if np.isfinite(float(v)) and float(v) >= -1e-6
        ]
        if not hover_extra_values:
            hover_extra_values = [0.0]

    plans: list[TargetedPlacePlan] = []
    for object_pose_local, slot_name in slot_specs:
        local_variants = _make_insert_vertical_local_pose_variants(
            rule,
            object_pose_local,
            getattr(args, "insert_vertical_axial_spin_deg", None),
            allow_axis_flip=bool(getattr(args, "targeted_place_allow_insert_axis_flip", False)),
        )
        if rule.primitive == "place_on_slots":
            local_variants = _make_orientation_invariant_local_pose_variants(args, rule, object_pose_local)
            if len(local_variants) <= 1:
                local_variants = _make_tabletop_axial_spin_local_pose_variants(args, rule, object_pose_local)

        for local_variant_label, T_target_obj_desired in local_variants:
            T_world_obj_desired_base = T_world_target @ T_target_obj_desired
            for tabletop_variant_label, T_world_obj_desired in _make_tabletop_place_world_pose_variants(
                demo,
                bridge_mod,
                args,
                rule,
                T_world_target,
                T_world_obj_desired_base,
            ):
                labels = [label for label in (local_variant_label, tabletop_variant_label) if label]
                variant_label = "+".join(labels) if labels else None
                T_world_tcp_place = T_world_obj_desired @ np.linalg.inv(T_tcp_obj)
                tcp_verticality = 0.0
                if target_up_axis is not None:
                    tcp_approach_axis = _normalize(np.asarray(T_world_tcp_place[:3, 2], dtype=np.float32).reshape(3))
                    if tcp_approach_axis is not None:
                        tcp_verticality = abs(float(np.dot(tcp_approach_axis, target_up_axis)))
                place_pose = _pose_from_matrix(T_world_tcp_place)
                p_place = base.flatten_np(place_pose.p)[:3].copy()
                retreat_pose = base.make_pose_with_position(
                    place_pose,
                    p_place + np.array([0.0, 0.0, float(rule.release_retreat_height)], dtype=np.float32),
                )
                for hover_extra in hover_extra_values:
                    hover_height = float(rule.hover_height) + float(hover_extra)
                    pre_place_pose = base.make_pose_with_position(
                        place_pose,
                        p_place + np.array([0.0, 0.0, hover_height], dtype=np.float32),
                    )
                    staging_pose = None
                    if bool(getattr(args, "targeted_place_staging", True)):
                        staging_pose = _build_targeted_place_staging_pose(demo, pre_place_pose, args)
                    hover_label = None if abs(float(hover_extra)) <= 1e-6 else f"hover_plus_{int(round(float(hover_extra) * 1000.0))}mm"
                    labels = [label for label in (variant_label, hover_label) if label]
                    plan_variant_label = "+".join(labels) if labels else None
                    plans.append(
                        TargetedPlacePlan(
                            rule=rule,
                            target_name=target_name,
                            slot_name=slot_name,
                            variant_label=plan_variant_label,
                            T_world_obj_desired=np.asarray(T_world_obj_desired, dtype=np.float32).reshape(4, 4),
                            staging_pose=staging_pose,
                            pre_place_pose=pre_place_pose,
                            place_pose=place_pose,
                            retreat_pose=retreat_pose,
                            tcp_verticality=float(tcp_verticality),
                        )
                    )
    if rule.primitive == "place_on_slots" and plans:
        verticality_target = getattr(rule, "tabletop_place_tcp_verticality_target", None)
        if verticality_target is not None:
            target_value = float(np.clip(float(verticality_target), 0.0, 1.0))
            plans.sort(
                key=lambda plan: (
                    abs(float(plan.tcp_verticality) - target_value),
                    "" if plan.variant_label is None else str(plan.variant_label),
                    "" if plan.slot_name is None else str(plan.slot_name),
                )
            )
        else:
            min_verticality = float(max(getattr(args, "tabletop_place_min_tcp_verticality", 0.0), 0.0))
            if any(plan.tcp_verticality >= min_verticality for plan in plans):
                plans = [plan for plan in plans if plan.tcp_verticality >= min_verticality]
            plans.sort(
                key=lambda plan: (
                    -float(plan.tcp_verticality),
                    "" if plan.variant_label is None else str(plan.variant_label),
                    "" if plan.slot_name is None else str(plan.slot_name),
                )
            )
    return plans


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


def _register_transport_attached_box(
    demo,
    args,
    *,
    show_visual: bool = True,
    activate_payload_visual: bool | None = None,
    T_tcp_obj_override: np.ndarray | None = None,
) -> np.ndarray:
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
    attach_pose_local = base.make_attached_box_pose(
        demo,
        attached_box_size,
        T_tcp_obj_override=T_tcp_obj_override,
    )
    try:
        demo.planner.update_attached_box(attached_box_size.tolist(), attach_pose_local.tolist())
        if show_visual:
            base.setup_attached_box_visual(demo, demo.env, attached_box_size, attach_pose_local)
        else:
            demo.attached_box_size = np.asarray(attached_box_size, dtype=np.float32).reshape(3)
            demo.attached_box_pose_tcp = np.asarray(attach_pose_local, dtype=np.float32).reshape(7)
        if activate_payload_visual is None:
            payload_visual_active = bool(show_visual) and bool(
                getattr(getattr(demo, "args", None), "teleport_attached_object_during_transport", False)
            )
        else:
            payload_visual_active = bool(activate_payload_visual)
        demo._attached_object_visual_active = payload_visual_active
        demo._attached_box_visual_visible = bool(show_visual)
        base.update_attached_box_visual(demo, visible=bool(show_visual))
    except Exception as exc:
        print(f"[warn] failed to register attached target box for targeted place planning: {exc}")
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
        allow_pose_rrt_fallback=True,
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
    grasp_pose = demo.build_topdown_grasp_pose()
    pregrasp_pose = demo.build_pregrasp_pose(grasp_pose)
    grasp_pose, pregrasp_pose, geometry_grasp_raise = base.enforce_topdown_grasp_insertion_limit(
        demo,
        args,
        grasp_pose,
        pregrasp_pose,
    )
    if geometry_grasp_raise > 0:
        print(
            f"[safety] raised grasp/pregrasp TCP z by {geometry_grasp_raise:.4f} m "
            f"to satisfy topdown_grasp_max_insertion_depth={args.topdown_grasp_max_insertion_depth:.4f}"
        )
    grasp_pose, pregrasp_pose, grasp_tcp_raise = base.enforce_min_grasp_tcp_z(
        grasp_pose,
        pregrasp_pose,
        args.min_grasp_tcp_z,
    )
    if grasp_tcp_raise > 0:
        print(
            f"[safety] raised grasp/pregrasp TCP z by {grasp_tcp_raise:.4f} m "
            f"to satisfy min_grasp_tcp_z={args.min_grasp_tcp_z:.4f}"
        )

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
    place_execution_label = "place"
    last_staging_pose = None
    last_pre_place_pose = None
    last_place_pose = None
    for candidate in place_plan_candidates:
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
            q_staging_path = base.plan_pose_path(
                demo,
                candidate.staging_pose,
                variant_name=args.variant,
                use_attach=True,
                label="pre_place_staging" if candidate.variant_label is None else f"pre_place_staging_{candidate.variant_label}",
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
        q_pre_place_path = base.plan_pose_path(
            demo,
            candidate.pre_place_pose,
            variant_name=args.variant,
            use_attach=True,
            label="pre_place" if candidate.variant_label is None else f"pre_place_{candidate.variant_label}",
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
                if _need_hidden_post_place_clearance(args):
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
                place_execution_label = "place" if candidate.variant_label is None else f"place_{candidate.variant_label}"
                break
            finally:
                base.sync_demo_arm_qpos(demo, q_eval_saved)
        else:
            print(
                f"[place] pre_place planning failed for variant={candidate.variant_label}; "
                "skipping direct place fallback and trying the next equivalent insertion orientation"
            )
            continue
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
    clearance_pose, q_clearance_path = base.plan_post_place_clearance_path(
        demo,
        retreat_distance=0.05,
        label="post_place_clearance",
    )
    clearance_executed = False
    if q_clearance_path:
        ok, _ = base.execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            "post_place_clearance",
            clearance_pose,
            q_clearance_path,
            args.real_gripper_open,
            args,
            use_attach=False,
            skip_confirmation=True,
        )
        if not ok:
            print("[warn] post-place clearance execution failed after release; continuing to settle the object in place")
        else:
            clearance_executed = True
            print(
                "[place] post_place_clearance final q:",
                np.round(np.asarray(q_clearance_path[-1], dtype=np.float32).reshape(-1)[:7], 5).tolist(),
            )
    else:
        print("[warn] post-place clearance planning failed after release; settling the object without moving the arm away first")

    base.settle_released_active_object_for_scene_cache(demo, args)
    _mark_place_rule_success(rule, place_state_cache, place_plan.slot_name)

    if relaxed_target_collision:
        _set_scene_obstacle_planner_box_scale(demo, place_plan.target_name, 1.0)
    if clearance_executed:
        print("[done] completed one targeted place motion and cleared 5cm along the gripper axis before settling the released object")
    else:
        print("[done] completed one targeted place motion; post-place clearance did not execute, so the next cycle will start from the release pose")
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


def _is_cached_scene_object_placed(scene_capture_cache, object_name: str | None) -> bool:
    name = normalize_object_name(object_name)
    if name is None or not isinstance(scene_capture_cache, dict):
        return False
    objects = scene_capture_cache.get("objects")
    if not isinstance(objects, dict):
        return False
    entry = objects.get(name)
    return isinstance(entry, dict) and bool(entry.get("placed", False))


def _unique_rule_names(names) -> list[str]:
    result = []
    seen = set()
    for raw_name in list(names or []):
        name = normalize_object_name(raw_name)
        if name is None or name in seen or get_place_rule(name) is None:
            continue
        seen.add(name)
        result.append(name)
    return result


def _random_target_pool_for_cycle(
    base_args,
    cycle_object_sequence,
    scene_capture_cache,
    available_rule_names,
    cycle_idx: int,
) -> list[str]:
    order = str(getattr(base_args, "target_selection_order", "random"))
    if available_rule_names:
        if cycle_object_sequence:
            available = set(_unique_rule_names(available_rule_names))
            cycle_pool = [name for name in _unique_rule_names(cycle_object_sequence) if name in available]
            if order == "cycle":
                pool = cycle_pool
            else:
                allowed = set(cycle_pool)
                pool = [name for name in _unique_rule_names(available_rule_names) if name in allowed]
        else:
            pool = _unique_rule_names(available_rule_names)
    elif cycle_object_sequence:
        pool = _unique_rule_names(cycle_object_sequence)
    elif cycle_idx == 1 and base_args.object_name is not None:
        pool = _unique_rule_names([base_args.object_name])
    else:
        pool = _unique_rule_names(list_place_rule_sources())
    return [name for name in pool if not _is_cached_scene_object_placed(scene_capture_cache, name)]


def _select_random_cycle_target(
    base_args,
    cycle_object_sequence,
    scene_capture_cache,
    available_rule_names,
    failed_targets_this_cycle: set[str],
    deferred_failed_targets: set[str] | None,
    cycle_idx: int,
) -> tuple[str | None, list[str], list[str]]:
    pool = _random_target_pool_for_cycle(
        base_args,
        cycle_object_sequence,
        scene_capture_cache,
        available_rule_names,
        cycle_idx,
    )
    deferred = set(deferred_failed_targets or set())
    candidates = [
        name for name in pool
        if name not in failed_targets_this_cycle and name not in deferred
    ]
    if not candidates:
        candidates = [name for name in pool if name not in failed_targets_this_cycle]
    if not candidates:
        return None, pool, []
    order = str(getattr(base_args, "target_selection_order", "random"))
    if order == "cycle":
        return candidates[0], pool, candidates
    if order == "risk_aware":
        priority = {
            # Remove the pen first so it no longer blocks nearby small long-axis
            # sources.  Place lvmukuai early: it needs an upright gripper relation
            # and becomes much harder once bulky tabletop objects consume slots.
            "bi": 0,
            "lvmukuai": 1,
            # Brush transport is sensitive to nearby tall tabletop objects;
            # place it before the glue stick so its target path is not blocked
            # by an already-placed vertical cylinder.
            "shuazi": 2,
            "gluestick": 3,
            "hongshupian": 4,
            "carriot": 4,
            "tennis": 5,
        }
        best_priority = min(priority.get(name, 10) for name in candidates)
        priority_candidates = [name for name in candidates if priority.get(name, 10) == best_priority]
        return random.choice(priority_candidates), pool, candidates
    return random.choice(candidates), pool, candidates


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
    previous_cycle_final_q: np.ndarray | None = None
    failed_targets_this_cycle: set[str] = set()
    deferred_failed_targets: set[str] = set()
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
            selected_name, target_pool, target_candidates = _select_random_cycle_target(
                base_args,
                cycle_object_sequence,
                scene_capture_cache,
                available_rule_names,
                failed_targets_this_cycle,
                deferred_failed_targets,
                cycle_idx,
            )
            if selected_name is None:
                final_ok = False
                if target_pool and failed_targets_this_cycle:
                    print(
                        f"[abort] cycle {cycle_idx}: all selectable targets failed in this cycle: "
                        f"{sorted(failed_targets_this_cycle & set(target_pool))}"
                    )
                else:
                    print(f"[abort] cycle {cycle_idx}: no selectable target object remains")
                break
            if failed_targets_this_cycle:
                print(
                    f"\n[cycle {cycle_idx}] target pool after failures: {target_candidates}; "
                    f"failed_this_cycle={sorted(failed_targets_this_cycle)}, "
                    f"deferred_failed={sorted(deferred_failed_targets)}"
                )
            elif deferred_failed_targets:
                print(
                    f"\n[cycle {cycle_idx}] target pool: {target_candidates}; "
                    f"deferred_failed={sorted(deferred_failed_targets)}"
                )
            else:
                print(f"\n[cycle {cycle_idx}] target pool: {target_candidates}")
            print(f"[cycle {cycle_idx}] selected target object: {selected_name}")
            rule = get_place_rule(selected_name)
            if rule is None:
                final_ok = False
                print(f"[abort] no targeted-place rule is configured for {selected_name}")
                break

            cycle_args, spec = base.make_cycle_args(base_args, selected_name)
            cycle_args._targeted_place_state_cache = place_state_cache
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
            final_sim_arm_q = None
            try:
                env, demo = base.create_demo(cycle_args, bridge_mod, planner_mod, scene_capture_cache=scene_capture_cache)
                if (
                    previous_cycle_final_q is not None
                    and bool(getattr(base_args, "carry_sim_arm_across_cycles", True))
                ):
                    base.sync_demo_arm_qpos(demo, previous_cycle_final_q)
                    print(
                        f"[cycle {cycle_idx}] seeded sim arm from previous cycle final q: "
                        f"{np.round(previous_cycle_final_q, 5).tolist()}"
                    )
                ok = run_targeted_place_episode(demo, bridge_mod, real_exec, cycle_args, scene_capture_cache, place_state_cache)
                if ok:
                    final_sim_arm_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
                    print(
                        f"[cycle {cycle_idx}] final sim arm q after place cycle: "
                        f"{np.round(final_sim_arm_q, 5).tolist()}"
                    )
            finally:
                base.close_env_quietly(env)
                gc.collect()
            print(f"\ncycle {cycle_idx} success = {ok}")
            if not ok:
                if bool(getattr(base_args, "reselect_target_on_planning_failure", True)):
                    failed_targets_this_cycle.add(selected_name)
                    deferred_failed_targets.add(selected_name)
                    print(
                        f"[cycle {cycle_idx}] planning/execution failed; "
                        "keeping the current cached scene and trying a different target."
                    )
                    cycle_idx -= 1
                    continue
                final_ok = False
                break
            failed_targets_this_cycle.clear()
            deferred_failed_targets.discard(selected_name)
            if final_sim_arm_q is not None:
                previous_cycle_final_q = final_sim_arm_q
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
