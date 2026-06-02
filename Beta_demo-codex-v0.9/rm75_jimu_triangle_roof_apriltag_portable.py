#!/usr/bin/env python3
"""Demo_Triangle-style triangle-roof entrypoint on top of the local portable path.

This file intentionally leaves rm75_jimu_four_wall_portable.py unchanged.  It
imports that entrypoint, patches only this process, and then delegates to the
same AprilTag/SAM6D/Realman execution path already used by the portable runner.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

import rm75_jimu_four_wall_portable as portable


# Put the front roof first. Its default tray slot is the least reachable; when
# it is attempted before the other roof panels, the same-family source retry can
# borrow a reachable triangle slot instead of failing after all triangles are used.
JIMU_ROOF_TRIANGLE_ROLES = ("front_roof_triangle", "right_roof_triangle", "back_roof_triangle", "left_roof_triangle")
JIMU_SECOND_LAYER_PICK_ORDER = ("front_second_wall", "left_second_wall", "right_second_wall", "back_second_wall")
TRIANGLE_ROLE_SPECS = {
    "front_roof_triangle": "red_triangle_front",
    "back_roof_triangle": "red_triangle_back",
    "left_roof_triangle": "red_triangle_left",
    "right_roof_triangle": "red_triangle_right",
}
TRIANGLE_PARENT_ROLES = {
    "right_roof_triangle": "right_second_wall",
    "back_roof_triangle": "back_second_wall",
    "left_roof_triangle": "left_second_wall",
    "front_roof_triangle": "front_second_wall",
}
DEFAULT_TRIANGLE_EXTENTS_M = np.asarray([0.074, 0.0065, 0.135], dtype=np.float32)
DEFAULT_RELATION_SLOTS = 16
DEFAULT_FIXED_BATCH_SIZE = 16
DEFAULT_FAST_TOP_PAIRS = 8
DEFAULT_TRIANGLE_TOPDOWN_GRASP_MAX_INSERTION_DEPTH = 0.07
DEFAULT_ROOF_MAX_HOVER_CANDIDATES_PER_GRASP = 16
DEFAULT_ROOF_RELATION_SLOTS = 64
DEFAULT_ROOF_FIXED_BATCH_SIZE = 16
DEFAULT_SECOND_LAYER_RELATION_SLOTS = 64
DEFAULT_SECOND_LAYER_FIXED_BATCH_SIZE = 16
DEFAULT_PREGRASP_EXTRA_WORLD_Z_M = 0.18
DEFAULT_PREGRASP_FALLBACK_WORLD_Z_M = 0.12
DEFAULT_PREGRASP_EMERGENCY_WORLD_Z_M = 0.10
DEFAULT_PREGRASP_LEGACY_LOW_WORLD_Z_M = 0.08
DEFAULT_ROOF_PREGRASP_EXTRA_WORLD_Z_M = 0.12
DEFAULT_ROOF_PREGRASP_FALLBACK_WORLD_Z_M = 0.0
DEFAULT_ROOF_PREGRASP_EMERGENCY_WORLD_Z_M = 0.0
DEFAULT_ROOF_PREGRASP_LEGACY_LOW_WORLD_Z_M = 0.0
DEFAULT_ROOF_PREGRASP_SAFETY_LOW_WORLD_Z_M = 0.0
DEFAULT_POST_GRASP_START_LIFT_M = 0.10
DEFAULT_INDEPENDENT_POST_GRASP_LIFT_M = 0.10
DEFAULT_ROOF_POST_PLACE_RETREAT_UP_RATIO = 1.0
DEFAULT_ROOF_POST_PLACE_FOLLOWUP_UP_M = 0.0
DEFAULT_ROOF_POST_PLACE_FOLLOWUP_SIDE_M = 0.02
DEFAULT_ROOF_SCENE_OBSTACLE_BOX_SCALE = 0.62
DEFAULT_ROOF_CUROBO_MESH_OBSTACLES = True
DEFAULT_ROOF_UNIFORM_PREPLACE_HEIGHT_M = 0.03
DEFAULT_DRY_RUN_RETURN_LINEAR_FALLBACK = False
# Keep the triangle panel thickness axis aligned with the tray slot narrow axis.
# The tip-up local rotation already fixes the mesh's vertical direction; an
# extra yaw here rotates the panel flat across the slot and makes the tray view
# look wrong.
DEFAULT_TRIANGLE_TRAY_SLOT_YAW_OFFSET_DEG = 0.0
TRIANGLE_TIP_UP_LOCAL_RPY_DEG = (0.0, 180.0, 0.0)
DEMO_TRIANGLE_MESH = Path(__file__).resolve().parents[1] / "Demo_Triangle" / "red_triangle_74x135x6p5.glb"

_ORIGINAL_BUILD_ARG_PARSER = portable.build_arg_parser
_ORIGINAL_PARSE_ARGS = portable.parse_args
_ORIGINAL_INSTALL_JIMU_OBJECT_SPECS = portable.install_jimu_object_specs
_ORIGINAL_INSTALL_JIMU_PLACE_RULES = portable.install_jimu_place_rules
_ORIGINAL_SECOND_LAYER_LOCAL_POSE_SPECS = portable._jimu_second_layer_local_pose_specs
_ORIGINAL_SECOND_LAYER_TARGET_POSE = portable._jimu_second_layer_target_pose_from_floor
_ORIGINAL_LAYER_FILTERED_TARGET_POOL = portable._jimu_layer_filtered_target_pool
_ORIGINAL_FLOOR_ANCHOR_SECOND_LAYER_PLANS = portable._jimu_floor_anchor_second_layer_plans
_ORIGINAL_TRAY_SLOT_LOCAL_POSES = portable._jimu_tray_slot_local_poses
_ORIGINAL_VALIDATE_LINEAR_JOINT_PATH = portable._jimu_validate_linear_joint_path
_ORIGINAL_MAKE_JIMU_PARALLEL_GRASP_PLACE_CANDIDATE = portable._make_jimu_parallel_grasp_place_candidate
_ORIGINAL_SELECT_JIMU_PARALLEL_PLACE_SOURCE_CANDIDATES = portable._select_jimu_parallel_place_source_candidates
_ORIGINAL_BUILD_DIRECT_GRASP_CANDIDATES = portable.direct._build_direct_grasp_candidates
_ORIGINAL_CHOOSE_NEXT_TRAY_SOURCE_ROLE = portable._jimu_choose_next_tray_source_role
_ORIGINAL_FAST_CHAIN_PRESELECT_GRASP_PLACE_PAIR = portable.direct._fast_chain_preselect_grasp_place_pair


def _argv_has_option(option_name: str) -> bool:
    prefix = f"{option_name}="
    return any(arg == option_name or str(arg).startswith(prefix) for arg in sys.argv[1:])


def _should_pre_enable_apriltag() -> bool:
    return (
        not _argv_has_option("--no-jimu-demo-triangle-profile")
        and not _argv_has_option("--no-jimu-demo-triangle-apriltag")
        and not _argv_has_option("--sam6d-fixed-scene-result-file")
        and not _argv_has_option("--jimu-apriltag-anchor-localization")
    )


def _has_explicit_fixed_scene_arg() -> bool:
    return _argv_has_option("--sam6d-fixed-scene-result-file")


def _add_arg_if_missing(parser: argparse.ArgumentParser, *option_strings: str, **kwargs: Any) -> None:
    if any(option in parser._option_string_actions for option in option_strings):
        return
    parser.add_argument(*option_strings, **kwargs)


def _triangle_profile_enabled(args: argparse.Namespace | None) -> bool:
    if args is None:
        return False
    return bool(
        getattr(
            args,
            "jimu_roof_triangle_profile",
            getattr(args, "jimu_second_layer_triangle_profile", False),
        )
    )


def _current_cycle_role(args: argparse.Namespace | None) -> str | None:
    try:
        return portable.direct.curobo_wrapper.normalize_object_name(getattr(args, "object_name", None))
    except Exception:
        return str(getattr(args, "object_name", "") or "") or None


def _current_cycle_is_roof(args: argparse.Namespace | None) -> bool:
    return _current_cycle_role(args) in set(JIMU_ROOF_TRIANGLE_ROLES)


def _current_cycle_is_second_layer_square(args: argparse.Namespace | None) -> bool:
    return _current_cycle_role(args) in set(portable.JIMU_SECOND_LAYER_ROLES)


def _is_roof_triangle_role(role: str | None) -> bool:
    normalized = portable.direct.curobo_wrapper.normalize_object_name(role)
    return normalized in set(JIMU_ROOF_TRIANGLE_ROLES)


def _choose_next_tray_source_role_triangle(
    args: argparse.Namespace,
    scene_capture_cache: dict,
    target_role: str,
) -> tuple[str | None, int | None, int | None]:
    objects = scene_capture_cache.get("objects")
    if not isinstance(objects, dict):
        return None, None, None
    role_order = portable._jimu_tray_slot_role_order(args)
    target_role = portable.direct.curobo_wrapper.normalize_object_name(target_role) or str(target_role)
    target_entry = objects.get(target_role)
    current_slot = portable._jimu_entry_slot_index(target_role, target_entry, role_order)
    if current_slot is None:
        return None, None, None

    retry_meta = scene_capture_cache.setdefault("_jimu_source_retry_meta", {})
    if not isinstance(retry_meta, dict):
        retry_meta = {}
        scene_capture_cache["_jimu_source_retry_meta"] = retry_meta
    attempted_by_target = retry_meta.setdefault("attempted_slot_indices_by_target", {})
    if not isinstance(attempted_by_target, dict):
        attempted_by_target = {}
        retry_meta["attempted_slot_indices_by_target"] = attempted_by_target
    attempted = attempted_by_target.setdefault(target_role, [])
    attempted_set = {
        int(v)
        for v in list(attempted or [])
        if isinstance(v, (int, np.integer)) or str(v).lstrip("-").isdigit()
    }
    attempted_set.add(int(current_slot))
    attempted_by_target[target_role] = sorted(attempted_set)

    target_is_roof = _is_roof_triangle_role(target_role)
    total_slots = max(len(role_order), 1)
    candidates: list[tuple[int, int, str]] = []
    skipped_family = 0
    for role in role_order:
        role_name = portable.direct.curobo_wrapper.normalize_object_name(role)
        if role_name is None or role_name == target_role:
            continue
        if _is_roof_triangle_role(role_name) != target_is_roof:
            skipped_family += 1
            continue
        entry = objects.get(role_name)
        if not isinstance(entry, dict) or bool(entry.get("placed", False)):
            continue
        if entry.get("T_world_obj") is None and entry.get("T_cam_obj") is None and entry.get("jimu_T_base_obj") is None:
            continue
        slot_idx = portable._jimu_entry_slot_index(role_name, entry, role_order)
        if slot_idx is None or int(slot_idx) in attempted_set:
            continue
        distance = (int(slot_idx) - int(current_slot)) % total_slots
        if distance <= 0:
            distance += total_slots
        candidates.append((distance, int(slot_idx), role_name))

    if not candidates:
        if skipped_family:
            print(
                "[triangle-roof] source retry found no same-family tray source for "
                f"{target_role} ({'roof' if target_is_roof else 'square'}); "
                f"skipped_other_family={skipped_family}"
            )
        return None, current_slot, None
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, donor_slot, donor_role = candidates[0]
    return donor_role, current_slot, donor_slot


def _effective_pregrasp_extra_world_z_m(args: argparse.Namespace | None) -> float:
    if args is None:
        return 0.0
    if _current_cycle_is_roof(args):
        return float(
            getattr(
                args,
                "jimu_roof_pregrasp_extra_world_z_m",
                getattr(args, "jimu_pregrasp_extra_world_z_m", 0.0),
            )
            or 0.0
        )
    return float(getattr(args, "jimu_pregrasp_extra_world_z_m", 0.0) or 0.0)


def _raise_pregrasp_candidates_world_z(candidates: list[dict], args) -> list[dict]:
    extra_z = _effective_pregrasp_extra_world_z_m(args)
    raised = _raise_pregrasp_candidates_by_world_z(candidates, args, extra_z, variant_label="primary")
    if _current_cycle_is_roof(args):
        return raised
    fallback_z = float(getattr(args, "jimu_pregrasp_fallback_world_z_m", DEFAULT_PREGRASP_FALLBACK_WORLD_Z_M) or 0.0)
    emergency_z = float(getattr(args, "jimu_pregrasp_emergency_world_z_m", DEFAULT_PREGRASP_EMERGENCY_WORLD_Z_M) or 0.0)
    legacy_low_z = float(getattr(args, "jimu_pregrasp_legacy_low_world_z_m", DEFAULT_PREGRASP_LEGACY_LOW_WORLD_Z_M) or 0.0)
    variants: list[tuple[str, float]] = [
        ("primary", extra_z),
        ("fallback", fallback_z),
        ("emergency", emergency_z),
        ("legacy_low", legacy_low_z),
    ]
    raised_groups: list[tuple[str, float, list[dict]]] = [("primary", extra_z, raised)]
    seen_z = {int(round(float(extra_z) * 1000000.0))}
    for label, value in variants[1:]:
        if value <= 1e-6:
            continue
        key = int(round(float(value) * 1000000.0))
        if key in seen_z:
            continue
        if extra_z <= value + 1e-6:
            continue
        seen_z.add(key)
        raised_groups.append((label, value, _raise_pregrasp_candidates_by_world_z(candidates, args, value, variant_label=label)))
    if len(raised_groups) == 1:
        return raised
    print(
        "[triangle-roof] added square pregrasp height fallback candidates: "
        + ", ".join(f"{label}={value * 1000.0:.1f}mm" for label, value, _ in raised_groups)
        + f", candidate_count={'+'.join(str(len(group)) for _, _, group in raised_groups)}"
    )
    return _interleave_candidate_groups([group for _, _, group in raised_groups])


def _interleave_candidate_groups(groups: list[list[dict]]) -> list[dict]:
    if not groups:
        return []
    max_len = max((len(group) for group in groups), default=0)
    interleaved: list[dict] = []
    for idx in range(max_len):
        for group in groups:
            if idx < len(group):
                interleaved.append(group[idx])
    return interleaved


def _raise_pregrasp_candidates_by_world_z(
    candidates: list[dict],
    args,
    extra_z: float,
    *,
    variant_label: str = "",
) -> list[dict]:
    if abs(extra_z) <= 1e-6:
        return candidates

    raised: list[dict] = []
    changed = 0
    for candidate in candidates:
        item = dict(candidate)
        pregrasp_pose = item.get("pregrasp_pose")
        if pregrasp_pose is not None:
            try:
                # The original direct pregrasp may already include a retreat along
                # the gripper approach axis.  Jimu plates need a vertical
                # pregrasp/grasp segment, so rebuild pregrasp from the actual
                # grasp pose and add only world-Z clearance.
                grasp_pose = item.get("pose") or item.get("grasp_pose") or pregrasp_pose
                p = portable.direct.targeted.base.flatten_np(grasp_pose.p)[:3].astype(np.float32)
                p = p + np.asarray([0.0, 0.0, extra_z], dtype=np.float32)
                item["pregrasp_pose"] = portable.direct.targeted.base.make_pose_with_position(
                    grasp_pose,
                    p.astype(np.float32),
                )
                item["jimu_pregrasp_extra_world_z_m"] = extra_z
                if _current_cycle_is_roof(args):
                    item["jimu_roof_pregrasp_extra_world_z_m"] = extra_z
                    if variant_label:
                        item["jimu_roof_pregrasp_height_variant"] = str(variant_label)
                        base_label = str(item.get("label", "") or "")
                        variant_order = {
                            "primary": 0,
                            "fallback": 1,
                            "emergency": 2,
                            "legacy_low": 3,
                            "safety_low": 4,
                        }.get(str(variant_label), 9)
                        height_mm = int(round(float(extra_z) * 1000.0))
                        item["jimu_roof_base_grasp_label"] = base_label
                        item["label"] = f"{base_label}_prez_{variant_order:02d}_{height_mm:03d}mm"
                changed += 1
            except Exception as exc:
                item["jimu_pregrasp_extra_world_z_error"] = str(exc)
        raised.append(item)

    if changed:
        suffix = f" variant={variant_label}" if variant_label else ""
        print(f"[jimu grasp] raised pregrasp candidate height by {extra_z * 1000.0:.1f}mm ({changed}/{len(candidates)}){suffix}")
    return raised


def _raise_roof_pregrasp_candidates_with_fallback(candidates: list[dict], args) -> list[dict]:
    extra_z = _effective_pregrasp_extra_world_z_m(args)
    fallback_z = float(getattr(args, "jimu_roof_pregrasp_fallback_world_z_m", DEFAULT_ROOF_PREGRASP_FALLBACK_WORLD_Z_M) or 0.0)
    emergency_z = float(getattr(args, "jimu_roof_pregrasp_emergency_world_z_m", DEFAULT_ROOF_PREGRASP_EMERGENCY_WORLD_Z_M) or 0.0)
    legacy_low_z = float(getattr(args, "jimu_roof_pregrasp_legacy_low_world_z_m", DEFAULT_ROOF_PREGRASP_LEGACY_LOW_WORLD_Z_M) or 0.0)
    safety_low_z = float(getattr(args, "jimu_roof_pregrasp_safety_low_world_z_m", DEFAULT_ROOF_PREGRASP_SAFETY_LOW_WORLD_Z_M) or 0.0)
    variants: list[tuple[str, float]] = [("primary", extra_z)]
    if _current_cycle_is_roof(args):
        variants.extend(
            [
                ("fallback", fallback_z),
                ("emergency", emergency_z),
                ("legacy_low", legacy_low_z),
                ("safety_low", safety_low_z),
            ]
        )

    raised_groups: list[list[dict]] = []
    seen_z: set[int] = set()
    for label, value in variants:
        if value <= 1e-6:
            continue
        key = int(round(float(value) * 1000000.0))
        if key in seen_z:
            continue
        seen_z.add(key)
        raised_groups.append(_raise_pregrasp_candidates_by_world_z(candidates, args, value, variant_label=label))

    if not raised_groups:
        return candidates
    if len(raised_groups) == 1:
        return raised_groups[0]
    print(
        "[triangle-roof] added roof pregrasp height fallback candidates: "
        + ", ".join(f"{name}={value * 1000.0:.1f}mm" for name, value in variants if value > 1e-6)
        + f", candidate_count={'+'.join(str(len(group)) for group in raised_groups)}"
    )
    return _interleave_candidate_groups(raised_groups)


def _roof_keep_tilt_only_grasp_candidates(candidates: list[dict], args) -> list[dict]:
    if not _current_cycle_is_roof(args):
        return candidates

    kept: list[dict] = []
    dropped: list[str] = []
    for candidate in list(candidates or []):
        label = str(candidate.get("label", "") or "")
        label_lower = label.lower()
        roll_deg = abs(float(candidate.get("grasp_approach_roll_deg", 0.0) or 0.0))
        axis_shift = abs(float(candidate.get("grasp_axis_shift_m", 0.0) or 0.0))
        z_lift = abs(float(candidate.get("grasp_z_lift_m", 0.0) or 0.0))
        forbidden = (
            "roll" in label_lower
            or "yaw" in label_lower
            or "panel_normal" in label_lower
            or "shift" in label_lower
            or "axis_" in label_lower
            or "lift_" in label_lower
            or roll_deg > 1e-6
            or axis_shift > 1e-6
            or z_lift > 1e-6
        )
        if forbidden:
            dropped.append(label or "<unnamed>")
            continue
        item = dict(candidate)
        item["grasp_approach_roll_deg"] = 0.0
        item["grasp_axis_shift_m"] = 0.0
        item["grasp_z_lift_m"] = 0.0
        item["jimu_roof_tilt_only_grasp"] = True
        kept.append(item)

    if dropped:
        preview = dropped[:8]
        suffix = "" if len(dropped) <= len(preview) else f", ... +{len(dropped) - len(preview)}"
        print(
            "[triangle-roof] dropped non-tilt roof grasp candidates: "
            f"{len(dropped)} removed ({preview}{suffix}); kept={len(kept)}"
        )
    if not kept and candidates:
        raise RuntimeError("roof tilt-only grasp filter removed every candidate; refusing non-tilt roof grasp")
    return kept


def _roof_pose_tcp_y_axis(candidate: dict) -> np.ndarray | None:
    pose = candidate.get("pose") if isinstance(candidate, dict) else None
    if pose is None:
        return None
    try:
        T_world_tcp = portable.direct.targeted.base.pose_to_matrix(
            portable.direct.targeted.base.flatten_np(pose.p)[:3],
            portable.direct.targeted.base.flatten_np(pose.q)[:4],
        )
        axis = np.asarray(T_world_tcp[:3, 1], dtype=np.float32)
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-8:
            return None
        return axis / norm
    except Exception:
        return None


def _roof_align_grasp_roll_to_panel_normal(demo, args, candidates: list[dict]) -> list[dict]:
    if not _current_cycle_is_roof(args) or not candidates:
        return candidates
    try:
        obj_p, obj_q = demo.get_obj_pose()
        T_world_obj = portable.direct.targeted.base.pose_to_matrix(obj_p, obj_q).astype(np.float32)
    except Exception as exc:
        print(f"[triangle-roof] failed to read active roof object pose for roll alignment: {exc}")
        return candidates

    extents = _triangle_extents()
    thin_axis_idx = int(np.argmin(np.asarray(extents, dtype=np.float32).reshape(3)))
    panel_normal = _roof_normalize_vec(T_world_obj[:3, thin_axis_idx])
    if panel_normal is None:
        return candidates

    aligned: list[dict] = []
    changed = 0
    max_before_deg = 0.0
    max_after_deg = 0.0
    for candidate in candidates:
        pose = candidate.get("pose") if isinstance(candidate, dict) else None
        if pose is None:
            aligned.append(candidate)
            continue
        try:
            T_world_tcp = portable.direct.targeted.base.pose_to_matrix(
                portable.direct.targeted.base.flatten_np(pose.p)[:3],
                portable.direct.targeted.base.flatten_np(pose.q)[:4],
            ).astype(np.float32)
        except Exception:
            aligned.append(candidate)
            continue

        approach = _roof_normalize_vec(T_world_tcp[:3, 2])
        current_y = _roof_normalize_vec(T_world_tcp[:3, 1])
        if approach is None or current_y is None:
            aligned.append(candidate)
            continue

        desired_y = panel_normal.copy()
        if float(np.dot(desired_y, current_y)) < float(np.dot(-desired_y, current_y)):
            desired_y = -desired_y

        desired_z = approach - desired_y * float(np.dot(approach, desired_y))
        desired_z = _roof_normalize_vec(desired_z)
        if desired_z is None:
            current_x = _roof_normalize_vec(T_world_tcp[:3, 0])
            if current_x is not None:
                desired_z = _roof_normalize_vec(np.cross(current_x, desired_y))
        if desired_z is None:
            aligned.append(candidate)
            continue

        tcp_x = _roof_normalize_vec(np.cross(desired_y, desired_z))
        if tcp_x is None:
            aligned.append(candidate)
            continue
        tcp_y = _roof_normalize_vec(np.cross(desired_z, tcp_x))
        if tcp_y is None:
            aligned.append(candidate)
            continue
        tcp_z = _roof_normalize_vec(np.cross(tcp_x, tcp_y))
        if tcp_z is None:
            aligned.append(candidate)
            continue
        R_new = np.stack([tcp_x, tcp_y, tcp_z], axis=1).astype(np.float32)
        if float(np.linalg.det(R_new)) < 0.0:
            tcp_x = -tcp_x
            R_new = np.stack([tcp_x, tcp_y, tcp_z], axis=1).astype(np.float32)

        before_dot = float(np.clip(abs(np.dot(current_y, panel_normal)), -1.0, 1.0))
        after_dot = float(np.clip(abs(np.dot(R_new[:, 1], panel_normal)), -1.0, 1.0))
        before_deg = float(np.degrees(np.arccos(before_dot)))
        after_deg = float(np.degrees(np.arccos(after_dot)))
        max_before_deg = max(max_before_deg, before_deg)
        max_after_deg = max(max_after_deg, after_deg)

        new_pose = portable.direct.targeted.Pose.create_from_pq(
            p=portable.direct.targeted.base.flatten_np(pose.p)[:3].astype(np.float32),
            q=portable.direct.targeted.base.bridge_mod_mat2quat(R_new),
        )
        new_pregrasp_pose = demo.build_pregrasp_pose(new_pose)
        new_pose, new_pregrasp_pose, _ = portable.direct.targeted.base.enforce_topdown_grasp_insertion_limit(
            demo,
            args,
            new_pose,
            new_pregrasp_pose,
        )
        new_pose, new_pregrasp_pose, _ = portable.direct.targeted.base.enforce_min_grasp_tcp_z(
            new_pose,
            new_pregrasp_pose,
            args.min_grasp_tcp_z,
        )
        T_world_tcp_new = portable.direct.targeted.base.pose_to_matrix(
            portable.direct.targeted.base.flatten_np(new_pose.p)[:3],
            portable.direct.targeted.base.flatten_np(new_pose.q)[:4],
        ).astype(np.float32)

        item = dict(candidate)
        item["pose"] = new_pose
        item["pregrasp_pose"] = new_pregrasp_pose
        item["T_tcp_obj"] = (np.linalg.inv(T_world_tcp_new).astype(np.float32) @ T_world_obj).astype(np.float32)
        item["jimu_roof_panel_normal_roll_aligned"] = True
        item["jimu_roof_panel_normal_axis_idx"] = int(thin_axis_idx)
        item["jimu_roof_tcp_y_panel_normal_error_before_deg"] = before_deg
        item["jimu_roof_tcp_y_panel_normal_error_after_deg"] = after_deg
        aligned.append(item)
        if after_deg + 1e-3 < before_deg:
            changed += 1

    print(
        "[triangle-roof] aligned roof grasp TCP-Y to panel normal: "
        f"changed={changed}/{len(candidates)}, "
        f"max_tcp_y_normal_error {max_before_deg:.2f}->{max_after_deg:.2f}deg"
    )
    return aligned


def _roof_reject_pad_axis_roll_drift(candidates: list[dict], max_deg: float = 1.0) -> list[dict]:
    """Roof tilt is allowed only about TCP Y; TCP Y itself must not roll."""
    if not candidates:
        return candidates
    direct = next(
        (
            item
            for item in candidates
            if str(item.get("label", "") or "").strip().lower() in {"grasp_direct", "direct", "grasp"}
        ),
        candidates[0],
    )
    ref_axis = _roof_pose_tcp_y_axis(direct)
    if ref_axis is None:
        return candidates

    kept: list[dict] = []
    dropped: list[str] = []
    max_delta = 0.0
    limit = float(max(max_deg, 0.0))
    for item in candidates:
        axis = _roof_pose_tcp_y_axis(item)
        if axis is None:
            kept.append(item)
            continue
        dot = float(np.clip(abs(np.dot(axis, ref_axis)), -1.0, 1.0))
        delta = float(np.degrees(np.arccos(dot)))
        max_delta = max(max_delta, delta)
        if delta > limit:
            dropped.append(f"{item.get('label', '<unnamed>')}:{delta:.2f}deg")
            continue
        checked = dict(item)
        checked["jimu_roof_pad_axis_delta_deg"] = delta
        kept.append(checked)
    if dropped:
        preview = dropped[:8]
        suffix = "" if len(dropped) <= len(preview) else f", ... +{len(dropped) - len(preview)}"
        print(
            "[triangle-roof] dropped roof grasp candidates with non-tilt pad-axis drift: "
            f"{len(dropped)} removed ({preview}{suffix}); kept={len(kept)}"
        )
    print(f"[triangle-roof] roof grasp pad-axis drift audit: max={max_delta:.3f}deg, limit={limit:.3f}deg")
    if not kept:
        raise RuntimeError("roof pad-axis roll-drift audit removed every candidate; refusing non-tilt roof grasp")
    return kept


def _build_direct_grasp_candidates_triangle(demo, args, **kwargs):
    if not _current_cycle_is_roof(args):
        candidates = _ORIGINAL_BUILD_DIRECT_GRASP_CANDIDATES(demo, args, **kwargs)
        return _raise_pregrasp_candidates_world_z(candidates, args)

    roof_args = argparse.Namespace(**vars(args))
    roof_args.topdown_tilt_toward_robot_deg = []
    roof_args.topdown_tilt_toward_robot_shift_m = [0.0]
    tilt_degs = list(getattr(args, "direct_grasp_tilt_toward_robot_deg", []) or [])
    if not tilt_degs:
        tilt_degs = [
            4.0,
            8.0,
            12.0,
            16.0,
            20.0,
            24.0,
            28.0,
            32.0,
            36.0,
            40.0,
            45.0,
            50.0,
            55.0,
            60.0,
            65.0,
        ]
    # Preserve caller customizations but force a signed sweep for roof candidates.
    # Previous versions used a one-sided list for compatibility; that can miss
    # feasible grasps when a single-side tilt has poor clearance.
    signed_tilts: list[float] = []
    seen_tilts: set[float] = set()
    for value in tilt_degs:
        try:
            magnitude = abs(float(value))
        except Exception:
            continue
        if magnitude <= 1e-9:
            continue
        for signed in (magnitude, -magnitude):
            key = round(float(signed), 6)
            if key in seen_tilts:
                continue
            seen_tilts.add(key)
            signed_tilts.append(signed)
    roof_args.direct_grasp_tilt_toward_robot_deg = signed_tilts
    roof_args.direct_grasp_tilt_toward_robot_shift_m = [0.0]
    roof_args.direct_grasp_object_axis_shifts_m = [0.0]
    roof_args.direct_grasp_z_lifts_m = [0.0]
    print(
        "[triangle-roof] roof grasp uses Jimu signed tilt batch "
        f"(direct + {len(signed_tilts)} tilt), "
        "shifts/extra-roll/yaw disabled, panel-normal roll alignment enabled"
    )
    candidates = _ORIGINAL_BUILD_DIRECT_GRASP_CANDIDATES(demo, roof_args, **kwargs)
    candidates = _roof_align_grasp_roll_to_panel_normal(demo, roof_args, candidates)
    candidates = _roof_keep_tilt_only_grasp_candidates(candidates, roof_args)
    candidates = _roof_reject_pad_axis_roll_drift(candidates, max_deg=1.0)
    return _raise_roof_pregrasp_candidates_with_fallback(candidates, roof_args)


def _fast_chain_preselect_grasp_place_pair_triangle(*call_args, **call_kwargs):
    args = call_kwargs.get("args")
    if args is None and len(call_args) >= 4:
        args = call_args[3]
    if args is not None and _current_cycle_is_roof(args):
        roof_args = argparse.Namespace(**vars(args).copy())
        roof_slots = max(1, int(getattr(args, "jimu_roof_relation_slots", DEFAULT_ROOF_RELATION_SLOTS) or DEFAULT_ROOF_RELATION_SLOTS))
        roof_fixed = 16
        roof_args.fast_chain_relation_ik_slots = roof_slots
        roof_args.fast_chain_cuda_graph_ik_fixed_batch_size = roof_fixed
        roof_args.fast_chain_cuda_graph_ik_max_batch_size = roof_fixed
        roof_args.strict_final_contact_waypoint_rot_tol_deg = max(
            float(getattr(args, "strict_final_contact_waypoint_rot_tol_deg", 8.0) or 8.0),
            10.0,
        )
        if (
            int(getattr(args, "fast_chain_relation_ik_slots", roof_slots) or roof_slots) != roof_slots
            or int(getattr(args, "fast_chain_cuda_graph_ik_fixed_batch_size", roof_fixed) or roof_fixed) != roof_fixed
        ):
            print(
                "[triangle-roof] roof fast-chain IK screen uses local batch: "
                f"relation_slots={roof_slots}, fixed_batch={roof_fixed}, "
                f"chunks={(roof_slots + roof_fixed - 1) // roof_fixed}"
            )
        if args is call_kwargs.get("args"):
            call_kwargs["args"] = roof_args
        else:
            call_args = list(call_args)
            call_args[3] = roof_args
            call_args = tuple(call_args)
    elif args is not None and _current_cycle_is_second_layer_square(args):
        second_args = argparse.Namespace(**vars(args).copy())
        second_slots = max(
            1,
            int(
                getattr(
                    args,
                    "jimu_second_layer_relation_slots",
                    DEFAULT_SECOND_LAYER_RELATION_SLOTS,
                )
                or DEFAULT_SECOND_LAYER_RELATION_SLOTS
            ),
        )
        second_fixed = 16
        second_args.fast_chain_relation_ik_slots = second_slots
        second_args.fast_chain_cuda_graph_ik_fixed_batch_size = second_fixed
        second_args.fast_chain_cuda_graph_ik_max_batch_size = second_fixed
        if (
            int(getattr(args, "fast_chain_relation_ik_slots", second_slots) or second_slots) != second_slots
            or int(getattr(args, "fast_chain_cuda_graph_ik_fixed_batch_size", second_fixed) or second_fixed) != second_fixed
        ):
            print(
                "[triangle-roof] second-layer square fast-chain IK screen uses local batch: "
                f"relation_slots={second_slots}, fixed_batch={second_fixed}, "
                f"chunks={(second_slots + second_fixed - 1) // second_fixed}"
            )
        if args is call_kwargs.get("args"):
            call_kwargs["args"] = second_args
        else:
            call_args = list(call_args)
            call_args[3] = second_args
            call_args = tuple(call_args)
    return _ORIGINAL_FAST_CHAIN_PRESELECT_GRASP_PLACE_PAIR(*call_args, **call_kwargs)


def _install_roof_role_constants() -> None:
    tray_roles = (
        *portable.JIMU_FIRST_LAYER_ROLES,
        "right_roof_triangle",
        "back_roof_triangle",
        "left_roof_triangle",
        portable.JIMU_SPARE_TRAY_SLOT_ROLES[0],
        *portable.JIMU_SECOND_LAYER_ROLES,
        "front_roof_triangle",
        portable.JIMU_SPARE_TRAY_SLOT_ROLES[1],
    )
    portable.JIMU_ROOF_TRIANGLE_ROLES = JIMU_ROOF_TRIANGLE_ROLES
    portable.JIMU_PICK_ROLES = (*portable.JIMU_FIRST_LAYER_ROLES, *JIMU_SECOND_LAYER_PICK_ORDER, *JIMU_ROOF_TRIANGLE_ROLES)
    portable.JIMU_TRAY_SLOT_ROLES = tray_roles
    portable.JIMU_LEGACY_SCENE_ROLES = (portable.JIMU_FLOOR_ROLE, *portable.JIMU_PICK_ROLES)
    portable.JIMU_SCENE_ROLES = (*portable.JIMU_BASE_ROLES, *portable.JIMU_TRAY_SLOT_ROLES)
    portable.JIMU_DERIVED_ROLE_SET = set((*portable.JIMU_BASE_ROLES, *portable.JIMU_TRAY_SLOT_ROLES))


def _triangle_extents() -> np.ndarray:
    try:
        spec = portable.object_specs.get_object_spec("red_triangle_front")
        if spec is not None:
            _, sim_scale = portable.object_specs.resolve_object_spec_scales(spec)
            mesh_path = Path(spec.sim_asset_file or spec.mesh_file).expanduser()
            if mesh_path.exists():
                loaded = trimesh.load(mesh_path, force="scene")
                bounds = np.asarray(loaded.bounds, dtype=np.float32)
                extents = (bounds[1] - bounds[0]) * float(sim_scale)
                if extents.shape == (3,) and np.all(np.isfinite(extents)) and float(np.min(extents)) > 1e-6:
                    return extents.astype(np.float32)
    except Exception as exc:
        print(f"[triangle-roof] warning: failed to resolve triangle extents, using fallback: {exc}")
    return DEFAULT_TRIANGLE_EXTENTS_M.copy()


def _triangle_tip_needs_local_y_flip() -> bool:
    try:
        spec = portable.object_specs.get_object_spec("red_triangle_front")
        if spec is None:
            return False
        mesh_path = Path(spec.sim_asset_file or spec.mesh_file).expanduser()
        if not mesh_path.exists():
            return False
        loaded = trimesh.load(mesh_path, force="scene")
        vertices = [
            np.asarray(geom.vertices, dtype=np.float32)
            for geom in loaded.geometry.values()
            if hasattr(geom, "vertices") and len(getattr(geom, "vertices", []))
        ]
        if not vertices:
            return False
        points = np.concatenate(vertices, axis=0)
        z_min = float(np.min(points[:, 2]))
        z_max = float(np.max(points[:, 2]))
        min_slice = points[np.isclose(points[:, 2], z_min, atol=max(1e-4, abs(z_max - z_min) * 1e-4))]
        max_slice = points[np.isclose(points[:, 2], z_max, atol=max(1e-4, abs(z_max - z_min) * 1e-4))]
        if len(min_slice) == 0 or len(max_slice) == 0:
            return False
        min_width = float(np.ptp(min_slice[:, 0]))
        max_width = float(np.ptp(max_slice[:, 0]))
        return min_width < max_width
    except Exception as exc:
        print(f"[triangle-roof] warning: failed to detect triangle tip direction, keeping mesh frame: {exc}")
        return False


def _triangle_tip_up_local_rotation() -> np.ndarray:
    rotation = np.eye(3, dtype=np.float32)
    if _triangle_tip_needs_local_y_flip():
        rotation[0, 0] = -1.0
        rotation[2, 2] = -1.0
    return rotation


def _rot_z_deg(deg: float) -> np.ndarray:
    rad = np.deg2rad(float(deg))
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _triangle_tip_up_local_rpy_deg() -> tuple[float, float, float]:
    if _triangle_tip_needs_local_y_flip():
        return TRIANGLE_TIP_UP_LOCAL_RPY_DEG
    return (0.0, 0.0, 0.0)


def _demo_triangle_spec(name: str):
    triangle_spec = portable.object_specs.get_object_spec(name)
    if triangle_spec is None:
        return None
    if DEMO_TRIANGLE_MESH.exists():
        return replace(
            triangle_spec,
            mesh_file=str(DEMO_TRIANGLE_MESH),
            mesh_scale=1.0,
            sim_asset_file=str(DEMO_TRIANGLE_MESH),
            sim_asset_scale=1.0,
            real_longest_axis_m=None,
        )
    return triangle_spec


def _roof_scene_obstacle_box_scale(args: argparse.Namespace | None) -> float:
    value = getattr(args, "jimu_roof_scene_obstacle_box_scale", DEFAULT_ROOF_SCENE_OBSTACLE_BOX_SCALE)
    try:
        return float(np.clip(float(value), 0.2, 1.0))
    except Exception:
        return DEFAULT_ROOF_SCENE_OBSTACLE_BOX_SCALE


def _roof_curobo_mesh_obstacles_enabled(args: argparse.Namespace | None) -> bool:
    return bool(getattr(args, "jimu_roof_curobo_mesh_obstacles", DEFAULT_ROOF_CUROBO_MESH_OBSTACLES))


def _enable_roof_curobo_mesh_obstacles(args: argparse.Namespace) -> None:
    existing = [
        str(name)
        for name in list(getattr(args, "curobo_world_mesh_object_names", []) or [])
        if str(name).strip()
    ]
    merged = list(existing)
    existing_norm = {portable.direct.curobo_wrapper.normalize_object_name(name) for name in existing}
    for role in JIMU_ROOF_TRIANGLE_ROLES:
        if portable.direct.curobo_wrapper.normalize_object_name(role) not in existing_norm:
            merged.append(role)
    args.curobo_world_mesh_object_names = merged


def install_jimu_object_specs_triangle(args: argparse.Namespace | None = None) -> None:
    _ORIGINAL_INSTALL_JIMU_OBJECT_SPECS(args)
    if not _triangle_profile_enabled(args):
        return

    for triangle_name in set(TRIANGLE_ROLE_SPECS.values()):
        triangle_spec = _demo_triangle_spec(triangle_name)
        if triangle_spec is not None:
            portable.object_specs.OBJECT_SPECS[triangle_name] = replace(
                triangle_spec,
                scene_obstacle_box_scale=_roof_scene_obstacle_box_scale(args),
            )

    local_rotation_offset = portable._jimu_cad_to_sim_rpy_deg(args)
    for role, triangle_name in TRIANGLE_ROLE_SPECS.items():
        triangle_spec = portable.object_specs.get_object_spec(triangle_name)
        if triangle_spec is None:
            raise RuntimeError(f"Missing triangle object spec: {triangle_name}")
        portable.object_specs.OBJECT_SPECS[role] = replace(
            triangle_spec,
            name=role,
            grounding_prompt="red isosceles triangle building panel.",
            foundationpose_local_rotation_offset_deg=local_rotation_offset,
            scene_obstacle_box_scale=_roof_scene_obstacle_box_scale(args),
        )
        portable.object_specs.OBJECT_NAME_ALIASES[role] = role


def _q7_or_none(value) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < 7 or not np.all(np.isfinite(arr[:7])):
        return None
    return arr[:7].copy()


def _jimu_second_layer_local_pose_specs_triangle(args: argparse.Namespace | None = None):
    return _ORIGINAL_SECOND_LAYER_LOCAL_POSE_SPECS(args)


def _jimu_roof_triangle_local_pose_specs(args: argparse.Namespace | None = None):
    if not _triangle_profile_enabled(args):
        return {}
    wall_extents = portable._load_scaled_jimu_extents(args)
    wall_height = float(wall_extents[2])
    child_height = float(_triangle_extents()[2])
    z_extra = float(getattr(args, "jimu_roof_layer_z_extra", 0.0) if args is not None else 0.0)
    center_offset_z = 0.5 * (wall_height + child_height) + z_extra
    return {
        role: portable.place_rules.LocalPoseSpec(
            position=(0.0, 0.0, center_offset_z),
            rpy_deg=_triangle_tip_up_local_rpy_deg(),
        )
        for role in JIMU_ROOF_TRIANGLE_ROLES
    }


def _jimu_second_layer_target_pose_from_floor_triangle(
    demo,
    bridge_mod,
    scene_capture_cache,
    source_name: str,
    args,
) -> np.ndarray | None:
    return _ORIGINAL_SECOND_LAYER_TARGET_POSE(demo, bridge_mod, scene_capture_cache, source_name, args)


def _jimu_roof_triangle_target_pose_from_floor(
    demo,
    bridge_mod,
    scene_capture_cache,
    source_name: str,
    args,
) -> np.ndarray | None:
    if not _triangle_profile_enabled(args) or source_name not in set(JIMU_ROOF_TRIANGLE_ROLES):
        return None
    parent_name = TRIANGLE_PARENT_ROLES.get(source_name)
    if parent_name is None:
        return None
    parent_target = portable._jimu_second_layer_target_pose_from_floor(
        demo,
        bridge_mod,
        scene_capture_cache,
        parent_name,
        args,
    )
    if parent_target is None:
        return None
    wall_height = float(portable._load_scaled_jimu_extents(args)[2])
    child_height = float(_triangle_extents()[2])
    z_extra = float(getattr(args, "jimu_second_layer_z_extra", 0.0) or 0.0)
    out = np.asarray(parent_target, dtype=np.float32).reshape(4, 4).copy()
    out[:3, :3] = (parent_target[:3, :3] @ _triangle_tip_up_local_rotation()).astype(np.float32)
    out[:3, 3] = (
        parent_target[:3, 3]
        + np.asarray([0.0, 0.0, 0.5 * (wall_height + child_height) + z_extra], dtype=np.float32)
    ).astype(np.float32)
    return out


def _jimu_floor_anchor_second_layer_and_roof_plans(plans, demo, bridge_mod, scene_capture_cache, source_name: str | None, args) -> list:
    if source_name in set(JIMU_ROOF_TRIANGLE_ROLES):
        target = _jimu_roof_triangle_target_pose_from_floor(demo, bridge_mod, scene_capture_cache, source_name, args)
        if target is None:
            return list(plans or [])
        anchored = [portable._jimu_rebuild_place_plan_for_target(plan, target) for plan in list(plans or [])]
        print(
            f"[triangle-roof] {source_name}: roof target anchored from second-layer goal, "
            f"target_xyz={np.round(target[:3, 3], 6).tolist()}"
        )
        return anchored
    return _ORIGINAL_FLOOR_ANCHOR_SECOND_LAYER_PLANS(plans, demo, bridge_mod, scene_capture_cache, source_name, args)


def install_jimu_place_rules_triangle(args: argparse.Namespace | None = None) -> None:
    _ORIGINAL_INSTALL_JIMU_PLACE_RULES(args)
    if not _triangle_profile_enabled(args):
        return
    hover_height = float(
        getattr(
            args,
            "jimu_roof_hover_height",
            getattr(args, "jimu_second_layer_hover_height", 0.08),
        )
        if args is not None
        else 0.08
    )
    release_retreat_height = float(
        getattr(
            args,
            "jimu_roof_release_retreat_height",
            getattr(args, "jimu_second_layer_release_retreat_height", 0.08),
        )
        if args is not None
        else 0.08
    )
    for role, local_pose in _jimu_roof_triangle_local_pose_specs(args).items():
        portable.place_rules.PLACE_RULES[role] = portable.place_rules.PlaceRule(
            source_object_name=role,
            target_object_name=TRIANGLE_PARENT_ROLES[role],
            primitive="jimu_relative_pose",
            hover_height=hover_height,
            release_retreat_height=release_retreat_height,
            preserve_long_axis_vertical=True,
            object_pose_local=local_pose,
        )


def _jimu_layer_filtered_target_pool_triangle(
    base_args: argparse.Namespace,
    pool: list[str],
    scene_capture_cache: dict | None,
) -> tuple[list[str], list[str]]:
    if not bool(getattr(base_args, "jimu_enforce_layer_order", True)):
        return pool, []
    normalized_pool = [
        portable.direct.curobo_wrapper.normalize_object_name(item)
        for item in list(pool or [])
    ]
    normalized_pool = [item for item in normalized_pool if item is not None]
    if not any(role in set(portable.JIMU_PICK_ROLES) for role in normalized_pool):
        return pool, []
    pool_set = set(normalized_pool)

    for layer_name, roles in (
        ("first square layer", portable.JIMU_FIRST_LAYER_ROLES),
        ("second square layer", portable.JIMU_SECOND_LAYER_ROLES),
        ("triangle roof layer", JIMU_ROOF_TRIANGLE_ROLES),
    ):
        pending = [
            role
            for role in roles
            if role in pool_set and not portable._is_jimu_role_placed(scene_capture_cache, role)
        ]
        if pending:
            allowed = set(pending)
            gated = [role for role in normalized_pool if role in allowed]
            if layer_name != "first square layer":
                print(f"[triangle-roof] holding later targets until {layer_name} is complete: pending={pending}")
            return gated, pending
    return pool, []


def _jimu_tray_slot_local_poses_triangle(args: argparse.Namespace | None = None) -> dict[str, np.ndarray]:
    poses = _ORIGINAL_TRAY_SLOT_LOCAL_POSES(args)
    if not _triangle_profile_enabled(args):
        return poses

    tip_rotation = _triangle_tip_up_local_rotation()
    yaw_offset_deg = float(
        getattr(args, "jimu_triangle_tray_slot_yaw_offset_deg", DEFAULT_TRIANGLE_TRAY_SLOT_YAW_OFFSET_DEG)
        if args is not None
        else DEFAULT_TRIANGLE_TRAY_SLOT_YAW_OFFSET_DEG
    )
    yaw_rotation = _rot_z_deg(yaw_offset_deg)
    square_height = float(portable._load_scaled_jimu_extents(args)[2])
    triangle_height = float(_triangle_extents()[2])
    tray_z_delta = 0.5 * (triangle_height - square_height)
    for role in TRIANGLE_ROLE_SPECS:
        if role not in poses:
            continue
        T = np.asarray(poses[role], dtype=np.float32).reshape(4, 4).copy()
        T[:3, :3] = (T[:3, :3] @ yaw_rotation @ tip_rotation).astype(np.float32)
        T[:3, 3] = (T[:3, 3] + np.asarray([0.0, 0.0, tray_z_delta], dtype=np.float32)).astype(np.float32)
        poses[role] = T
    return poses


def _jimu_validate_linear_joint_path_triangle(planner, q_path: np.ndarray, args) -> bool:
    if _current_cycle_is_roof(args) and bool(getattr(args, "jimu_roof_skip_linear_transport_start_check", True)):
        return True
    return _ORIGINAL_VALIDATE_LINEAR_JOINT_PATH(planner, q_path, args)


def _roof_world_z_hover_height(args, rule) -> float:
    hover_height = float(
        max(
            getattr(
                args,
                "jimu_roof_parallel_hover_height",
                getattr(args, "jimu_roof_hover_height", getattr(rule, "hover_height", 0.03)),
            ),
            0.0,
        )
    )
    if hover_height <= 1e-8:
        hover_height = float(
            max(
                getattr(
                    args,
                    "jimu_roof_release_retreat_height",
                    getattr(rule, "release_retreat_height", 0.05),
                ),
                0.0,
            )
        )
    return hover_height


def _roof_world_z_hover_from_release(release_pose, args, rule):
    hover_height = _roof_world_z_hover_height(args, rule)
    if hover_height <= 1e-8:
        return release_pose
    p = portable.direct.targeted.base.flatten_np(release_pose.p)[:3].astype(np.float32)
    return portable.direct.targeted.base.make_pose_with_position(
        release_pose,
        (p + np.asarray([0.0, 0.0, hover_height], dtype=np.float32)).astype(np.float32),
    )


def _roof_uniform_preplace_height_enabled(args) -> bool:
    return bool(getattr(args, "jimu_roof_uniform_preplace_height", True))


def _roof_uniform_preplace_height_m(args, rule) -> float:
    try:
        height = float(
            getattr(
                args,
                "jimu_roof_uniform_preplace_height_m",
                DEFAULT_ROOF_UNIFORM_PREPLACE_HEIGHT_M,
            )
        )
    except Exception:
        height = DEFAULT_ROOF_UNIFORM_PREPLACE_HEIGHT_M
    if height < 0.0:
        return 0.0
    return height


def _roof_pose_position(pose) -> np.ndarray:
    return portable.direct.targeted.base.flatten_np(pose.p)[:3].astype(np.float32)


def _roof_make_pose_with_position(pose, position: np.ndarray):
    return portable.direct.targeted.base.make_pose_with_position(
        pose,
        np.asarray(position, dtype=np.float32).reshape(3),
    )


def _roof_force_uniform_preplace_height(hover_pose, release_pose, args, rule):
    if not _roof_uniform_preplace_height_enabled(args):
        return hover_pose
    hover_height = _roof_uniform_preplace_height_m(args, rule)
    if hover_height <= 1e-8:
        return hover_pose
    p = _roof_pose_position(hover_pose)
    release_p = _roof_pose_position(release_pose)
    p[2] = np.float32(release_p[2] + hover_height)
    return _roof_make_pose_with_position(hover_pose, p)


def _roof_normalize_vec(vec) -> np.ndarray | None:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    arr = arr[:3]
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return None
    return (arr / norm).astype(np.float32)


def _roof_hover_variant_specs(args) -> list[dict]:
    try:
        max_variants = max(1, int(getattr(args, "jimu_roof_hover_variants_per_source", 6) or 6))
    except Exception:
        max_variants = 6
    low_height = float(max(getattr(args, "jimu_roof_hover_low_height", 0.03), 0.0))
    outward_distance = float(max(getattr(args, "jimu_roof_hover_outward_distance", 0.035), 0.0))
    outward_up = float(max(getattr(args, "jimu_roof_hover_outward_up_m", 0.02), 0.0))
    original_extra = float(max(getattr(args, "jimu_roof_hover_original_extra_m", 0.03), 0.0))
    if _roof_uniform_preplace_height_enabled(args):
        return [{"kind": "world_z", "label": "roof_world_z"}]
    specs = [
        {"kind": "world_z", "label": "roof_world_z"},
        {"kind": "release_direct", "label": "roof_release_direct"},
        {"kind": "original", "label": "roof_tcp_axis"},
    ]
    if low_height > 1e-6:
        specs.append({"kind": "world_z_height", "height": low_height, "label": f"roof_world_z_{int(round(low_height * 1000.0))}mm"})
    if original_extra > 1e-6:
        specs.append(
            {
                "kind": "original_extend",
                "extra": original_extra,
                "label": f"roof_tcp_axis_plus_{int(round(original_extra * 1000.0))}mm",
            }
        )
    if outward_distance > 1e-6:
        specs.append(
            {
                "kind": "object_y",
                "sign": 1.0,
                "distance": outward_distance,
                "up": outward_up,
                "label": f"roof_obj_y_plus_{int(round(outward_distance * 1000.0))}mm",
            }
        )
        specs.append(
            {
                "kind": "object_y",
                "sign": -1.0,
                "distance": outward_distance,
                "up": outward_up,
                "label": f"roof_obj_y_minus_{int(round(outward_distance * 1000.0))}mm",
            }
        )
    return specs[:max_variants]


def _roof_hover_pose_for_variant(
    item: dict,
    release_pose,
    original_hover_pose,
    args,
    rule,
    variant: dict | None,
):
    variant = dict(variant or {"kind": "world_z", "label": "roof_world_z"})
    kind = str(variant.get("kind", "world_z") or "world_z")
    release_p = _roof_pose_position(release_pose)
    if kind == "release_direct":
        if _roof_uniform_preplace_height_enabled(args):
            return _roof_force_uniform_preplace_height(release_pose, release_pose, args, rule)
        return release_pose
    if kind == "world_z":
        return _roof_force_uniform_preplace_height(
            _roof_world_z_hover_from_release(release_pose, args, rule),
            release_pose,
            args,
            rule,
        )
    if kind == "world_z_height":
        height = float(max(variant.get("height", 0.0) or 0.0, 0.0))
        if height <= 1e-8:
            return release_pose
        return _roof_force_uniform_preplace_height(
            _roof_make_pose_with_position(release_pose, release_p + np.asarray([0.0, 0.0, height], dtype=np.float32)),
            release_pose,
            args,
            rule,
        )
    if kind == "original" and original_hover_pose is not None:
        return _roof_force_uniform_preplace_height(original_hover_pose, release_pose, args, rule)
    if kind == "original_extend" and original_hover_pose is not None:
        try:
            hover_pose = portable.direct._extend_hover_pose_along_release_approach(
                release_pose,
                original_hover_pose,
                float(max(variant.get("extra", 0.0) or 0.0, 0.0)),
            )
            return _roof_force_uniform_preplace_height(hover_pose, release_pose, args, rule)
        except Exception:
            return _roof_force_uniform_preplace_height(original_hover_pose, release_pose, args, rule)
    if kind == "object_y":
        try:
            T_world_obj = np.asarray(item.get("T_world_obj_desired"), dtype=np.float32).reshape(4, 4)
            axis = _roof_normalize_vec(T_world_obj[:3, 1])
        except Exception:
            axis = None
        if axis is not None:
            sign = 1.0 if float(variant.get("sign", 1.0) or 1.0) >= 0.0 else -1.0
            distance = float(max(variant.get("distance", 0.0) or 0.0, 0.0))
            up = float(max(variant.get("up", 0.0) or 0.0, 0.0))
            return _roof_force_uniform_preplace_height(
                _roof_make_pose_with_position(
                    release_pose,
                    release_p + axis * sign * distance + np.asarray([0.0, 0.0, up], dtype=np.float32),
                ),
                release_pose,
                args,
                rule,
            )
    return _roof_force_uniform_preplace_height(
        _roof_world_z_hover_from_release(release_pose, args, rule),
        release_pose,
        args,
        rule,
    )


def _roof_post_place_retreat_m(args) -> float:
    if (
        _argv_has_option("--jimu-roof-post-place-tilt-retreat-m")
        and not _argv_has_option("--jimu-roof-post-place-retreat-m")
    ):
        value = getattr(args, "jimu_roof_post_place_tilt_retreat_m", 0.03)
    else:
        value = getattr(
            args,
            "jimu_roof_post_place_retreat_m",
            getattr(args, "jimu_roof_post_place_tilt_retreat_m", 0.03),
        )
    return float(
        max(
            value,
            0.0,
        )
    )


def _roof_tilt_axis_retreat_from_release(release_pose, args):
    retreat_m = _roof_post_place_retreat_m(args)
    if retreat_m <= 1e-8:
        return release_pose
    release_p = _roof_pose_position(release_pose)
    try:
        T_world_tcp = portable.direct._pose_to_matrix_from_pose_obj(release_pose).astype(np.float32)
        axes_world = T_world_tcp[:3, :3]
        z_dots = axes_world.T @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        axis_idx = int(np.argmax(np.abs(z_dots)))
        sign = 1.0 if float(z_dots[axis_idx]) >= 0.0 else -1.0
        retreat_dir = _roof_normalize_vec(axes_world[:, axis_idx] * sign)
    except Exception:
        retreat_dir = None
    if retreat_dir is None:
        retreat_dir = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return _roof_make_pose_with_position(release_pose, release_p + retreat_dir * retreat_m)


def _roof_post_place_plane_basis(item: dict, release_pose, hover_pose):
    release_p = _roof_pose_position(release_pose)
    plane_normal = None
    plane_x = None
    plane_z = None
    try:
        T_world_obj = np.asarray(item.get("T_world_obj_desired"), dtype=np.float32).reshape(4, 4)
        R_world_obj = T_world_obj[:3, :3].astype(np.float32)
        # Jimu plates and triangle panels use local Y as the thin axis, so the
        # broad placement face is the local X/Z plane.
        plane_x = _roof_normalize_vec(R_world_obj[:, 0])
        plane_normal = _roof_normalize_vec(R_world_obj[:, 1])
        plane_z = _roof_normalize_vec(R_world_obj[:, 2])
    except Exception:
        plane_x = None
        plane_normal = None
        plane_z = None

    main_dir = None
    if hover_pose is not None:
        raw_dir = _roof_normalize_vec(_roof_pose_position(hover_pose) - release_p)
        if raw_dir is not None and plane_normal is not None:
            projected = raw_dir - plane_normal * float(np.dot(raw_dir, plane_normal))
            main_dir = _roof_normalize_vec(projected)
        elif raw_dir is not None:
            main_dir = raw_dir
    if main_dir is None and plane_z is not None:
        z_dot = float(np.dot(plane_z, np.asarray([0.0, 0.0, 1.0], dtype=np.float32)))
        main_dir = _roof_normalize_vec(plane_z * (1.0 if z_dot >= 0.0 else -1.0))
    if main_dir is None and plane_x is not None:
        main_dir = plane_x
    if main_dir is None:
        main_dir = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

    if plane_normal is not None:
        perp_dir = _roof_normalize_vec(np.cross(plane_normal, main_dir))
    else:
        perp_dir = _roof_normalize_vec(np.cross(main_dir, np.asarray([0.0, 0.0, 1.0], dtype=np.float32)))
    if perp_dir is None and plane_x is not None:
        perp_dir = plane_x
    if perp_dir is None:
        perp_dir = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    # Re-orthogonalize main inside the panel plane after choosing perp.
    if plane_normal is not None:
        reorthogonalized_main = _roof_normalize_vec(np.cross(perp_dir, plane_normal))
        if reorthogonalized_main is not None:
            main_dir = reorthogonalized_main
        if hover_pose is not None:
            hover_dir = _roof_normalize_vec(_roof_pose_position(hover_pose) - release_p)
            if hover_dir is not None and float(np.dot(main_dir, hover_dir)) < 0.0:
                main_dir = -main_dir
                perp_dir = -perp_dir
    return main_dir.astype(np.float32), perp_dir.astype(np.float32), plane_normal


def _roof_post_place_translation_retreat_candidates(item: dict, release_pose, hover_pose, args) -> list[dict]:
    retreat_m = _roof_post_place_retreat_m(args)
    if retreat_m <= 1e-8:
        return [{"label": "retreat_zero", "pose": release_pose}]
    release_p = _roof_pose_position(release_pose)
    main_dir, perp_dir, plane_normal = _roof_post_place_plane_basis(item, release_pose, hover_pose)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

    lateral = float(max(getattr(args, "jimu_roof_post_place_retreat_lateral_step_m", 0.006), 0.0))
    forward_extra = float(max(getattr(args, "jimu_roof_post_place_retreat_forward_extra_m", 0.010), 0.0))
    up_ratio = float(max(getattr(args, "jimu_roof_post_place_retreat_up_ratio", DEFAULT_ROOF_POST_PLACE_RETREAT_UP_RATIO), 0.0))
    up_m = float(min(retreat_m * up_ratio, 0.025))
    followup_up_m = float(max(getattr(args, "jimu_roof_post_place_followup_up_m", DEFAULT_ROOF_POST_PLACE_FOLLOWUP_UP_M), 0.0))
    followup_side_m = float(
        max(getattr(args, "jimu_roof_post_place_followup_side_m", DEFAULT_ROOF_POST_PLACE_FOLLOWUP_SIDE_M), 0.0)
    )
    try:
        max_count = max(1, int(getattr(args, "jimu_roof_post_place_retreat_candidate_count", 16) or 16))
    except Exception:
        max_count = 16
    allow_free_motiongen = bool(getattr(args, "jimu_roof_post_place_free_motiongen_fallback", False))

    far_m = float(retreat_m + max(forward_extra, 4.0 * lateral, 0.025))
    specs = []
    for level_label, dist in (("near", retreat_m), ("far", far_m)):
        diag = float(dist / max(2.0 ** 0.5, 1e-6))
        specs.extend(
            [
                (f"plane_main_p_{level_label}_up", dist, 0.0, up_m),
                (f"plane_main_m_{level_label}_up", -dist, 0.0, up_m),
                (f"plane_perp_p_{level_label}_up", 0.0, dist, up_m),
                (f"plane_perp_m_{level_label}_up", 0.0, -dist, up_m),
                (f"plane_diag_pp_{level_label}_up", diag, diag, up_m),
                (f"plane_diag_pm_{level_label}_up", diag, -diag, up_m),
                (f"plane_diag_mp_{level_label}_up", -diag, diag, up_m),
                (f"plane_diag_mm_{level_label}_up", -diag, -diag, up_m),
            ]
        )
    candidates = []
    for label, main_offset, perp_offset, up_offset in specs[:max_count]:
        delta = (
            main_dir * float(main_offset)
            + perp_dir * float(perp_offset)
            + world_up * float(up_offset)
        )
        candidate_pose = _roof_make_pose_with_position(release_pose, release_p + delta)
        plane_normal_error = 0.0
        if plane_normal is not None:
            plane_normal_error = float(np.dot(delta, plane_normal))
        candidate = {
            "label": f"roof_post_{label}",
            "pose": candidate_pose,
            "plane_main_m": float(main_offset),
            "plane_perp_m": float(perp_offset),
            "world_z_m": float(up_offset),
            "retreat_up_ratio": float(up_ratio),
            "retreat_delta_norm_m": float(np.linalg.norm(delta)),
            "plane_normal_error_m": plane_normal_error,
            "allow_post_place_free_motiongen": allow_free_motiongen,
            "post_place_endpoint_ik_first": True,
        }
        if followup_up_m > 1e-6:
            candidate_p = release_p + delta
            side_sign = 1.0 if float(perp_offset) >= 0.0 else -1.0
            followup_specs = [
                ("follow_world_z", 0.0, 0.0, followup_up_m),
                ("follow_world_z_short", 0.0, 0.0, 0.75 * followup_up_m),
                ("follow_same_side_up", 0.0, side_sign * followup_side_m, followup_up_m),
                ("follow_opposite_side_up", 0.0, -side_sign * followup_side_m, followup_up_m),
                ("follow_world_z_long", 0.0, 0.0, 1.25 * followup_up_m),
            ]
            followups = []
            for follow_label, follow_main, follow_perp, follow_up in followup_specs:
                follow_delta = (
                    main_dir * float(follow_main)
                    + perp_dir * float(follow_perp)
                    + world_up * float(follow_up)
                )
                followups.append(
                    {
                        "label": f"roof_post_{label}_{follow_label}",
                        "pose": _roof_make_pose_with_position(release_pose, candidate_p + follow_delta),
                        "plane_main_m": float(main_offset + follow_main),
                        "plane_perp_m": float(perp_offset + follow_perp),
                        "world_z_m": float(up_offset + follow_up),
                        "followup_plane_main_m": float(follow_main),
                        "followup_plane_perp_m": float(follow_perp),
                        "followup_world_z_m": float(follow_up),
                        "followup_delta_norm_m": float(np.linalg.norm(follow_delta)),
                        "allow_post_place_free_motiongen": allow_free_motiongen,
                        "post_place_endpoint_ik_first": True,
                    }
                )
            candidate["followup_retreat_pose_candidates"] = followups
            candidate["require_followup_retreat"] = False
        candidates.append(candidate)
    return candidates


def _make_jimu_parallel_grasp_place_candidate_triangle(
    grasp_candidate: dict,
    place_candidate: dict,
    args,
    rule,
    **kwargs,
) -> dict | None:
    item = _ORIGINAL_MAKE_JIMU_PARALLEL_GRASP_PLACE_CANDIDATE(grasp_candidate, place_candidate, args, rule, **kwargs)
    if item is None or not _current_cycle_is_roof(args):
        return item

    release_pose = item.get("release_pose", item.get("place_pose"))
    if release_pose is None:
        return item
    original_hover_pose = item.get("hover_pose", item.get("pose"))
    hover_variant = dict(place_candidate.get("_jimu_roof_hover_variant") or item.get("_jimu_roof_hover_variant") or {})
    hover_pose = _roof_hover_pose_for_variant(item, release_pose, original_hover_pose, args, rule, hover_variant)
    item["pose"] = hover_pose
    item["hover_pose"] = hover_pose
    item["pre_place_pose"] = hover_pose
    item["place_pose"] = release_pose
    item["release_pose"] = release_pose
    retreat_candidates = _roof_post_place_translation_retreat_candidates(item, release_pose, hover_pose, args)
    item["retreat_pose_candidates"] = retreat_candidates
    item["retreat_pose"] = retreat_candidates[0]["pose"] if retreat_candidates else _roof_tilt_axis_retreat_from_release(release_pose, args)
    if str(hover_variant.get("kind", "")) == "release_direct":
        item["place_mode"] = "drop_place"
        item["jimu_roof_direct_release_hover"] = True
    elif str(item.get("place_mode", "vertical_place") or "vertical_place") == "drop_place":
        item["place_mode"] = "vertical_place"
    item["jimu_roof_exact_release_drop"] = False
    item["force_replan_post_place_clearance"] = True
    item["jimu_roof_post_place_retreat_mode"] = "preplace_diagonal_world_z_candidates"
    item["jimu_roof_post_place_retreat_m"] = _roof_post_place_retreat_m(args)
    item["jimu_roof_post_place_retreat_up_ratio"] = float(
        max(getattr(args, "jimu_roof_post_place_retreat_up_ratio", DEFAULT_ROOF_POST_PLACE_RETREAT_UP_RATIO), 0.0)
    )
    item["jimu_roof_post_place_retreat_candidate_count"] = len(retreat_candidates)
    item["jimu_roof_hover_variant"] = dict(hover_variant or {"kind": "world_z", "label": "roof_world_z"})
    item["jimu_roof_hover_variant_label"] = str(item["jimu_roof_hover_variant"].get("label", "roof_world_z") or "roof_world_z")
    item["jimu_roof_world_z_hover"] = str(item["jimu_roof_hover_variant"].get("kind", "world_z")) in {"world_z", "world_z_height"}
    item["jimu_roof_world_z_hover_height_m"] = _roof_world_z_hover_height(args, rule)
    base_label = str(item.get("label", "transport_hover") or "transport_hover")
    label_suffix = item["jimu_roof_hover_variant_label"]
    if label_suffix not in base_label:
        item["label"] = f"{base_label}_{label_suffix}"
    return item


def _roof_target_pose_variants_for_place_candidate(base: dict, args) -> list[tuple[str, dict]]:
    variants: list[tuple[str, dict]] = [("", dict(base))]
    T_world_obj = base.get("T_world_obj_desired")
    if T_world_obj is None:
        T_world_obj = getattr(base.get("place_plan", None), "T_world_obj_desired", None)
    if T_world_obj is None:
        return variants
    try:
        T_world_obj_arr = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4)
    except Exception:
        return variants

    # The triangle panel is a thin 2-sided part.  Rotating around local Z keeps
    # the tip-up silhouette and target center, but swaps which face points out.
    # This changes the place TCP orientation without adding a non-tilt grasp.
    R_local_z_180 = np.eye(4, dtype=np.float32)
    R_local_z_180[:3, :3] = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    flipped = dict(base)
    flipped["T_world_obj_desired"] = (T_world_obj_arr @ R_local_z_180).astype(np.float32)
    base_variant = str(flipped.get("variant_label", "") or "")
    suffix = "roof_face_z180"
    flipped["variant_label"] = f"{base_variant}+{suffix}" if base_variant else suffix
    flipped["jimu_roof_target_face_variant"] = "local_z_180"
    variants.append((suffix, flipped))
    return variants


def _select_jimu_parallel_place_source_candidates_triangle(place_candidates, args) -> list[dict]:
    if not _current_cycle_is_roof(args):
        return _ORIGINAL_SELECT_JIMU_PARALLEL_PLACE_SOURCE_CANDIDATES(place_candidates, args)

    ordered = sorted([dict(item) for item in list(place_candidates or [])], key=portable.direct._pre_place_screen_sort_key)
    if not ordered:
        return []
    try:
        limit = max(1, int(getattr(args, "jimu_roof_parallel_sources_per_grasp", 4) or 4))
    except Exception:
        limit = 4
    selected_bases: list[dict] = []
    seen: set[tuple] = set()
    for item in ordered:
        key = (
            portable.direct._pose_dedupe_key(item.get("pose")),
            portable.direct._pose_dedupe_key(item.get("release_pose", item.get("place_pose"))),
        )
        if key in seen:
            continue
        seen.add(key)
        selected_bases.append(item)
        if len(selected_bases) >= limit:
            break
    try:
        max_expanded = max(
            1,
            int(
                getattr(
                    args,
                    "jimu_roof_max_hover_candidates_per_grasp",
                    DEFAULT_ROOF_MAX_HOVER_CANDIDATES_PER_GRASP,
                )
                or DEFAULT_ROOF_MAX_HOVER_CANDIDATES_PER_GRASP
            ),
        )
    except Exception:
        max_expanded = DEFAULT_ROOF_MAX_HOVER_CANDIDATES_PER_GRASP
    variants = _roof_hover_variant_specs(args)
    expanded: list[dict] = []
    seen_variant: set[tuple] = set()
    for base in selected_bases:
        for target_suffix, target_base in _roof_target_pose_variants_for_place_candidate(base, args):
            base_label = str(target_base.get("label", "transport_hover") or "transport_hover")
            for variant in variants:
                label = str(variant.get("label", "roof_world_z") or "roof_world_z")
                full_label = label if not target_suffix else f"{target_suffix}_{label}"
                item = dict(target_base)
                item["_jimu_roof_hover_variant"] = dict(variant)
                item["_jimu_roof_hover_variant_label"] = full_label
                item["label"] = f"{base_label}_{full_label}" if full_label not in base_label else base_label
                key = (
                    portable.direct._pose_dedupe_key(base.get("pose")),
                    portable.direct._pose_dedupe_key(base.get("release_pose", base.get("place_pose"))),
                    target_suffix,
                    label,
                )
                if key in seen_variant:
                    continue
                seen_variant.add(key)
                expanded.append(item)
                if len(expanded) >= max_expanded:
                    return expanded
    return expanded


def build_arg_parser_triangle() -> argparse.ArgumentParser:
    parser = _ORIGINAL_BUILD_ARG_PARSER()
    _add_arg_if_missing(
        parser,
        "--jimu-demo-triangle-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Demo_Triangle-style full four-wall + triangle-roof defaults in this wrapper.",
    )
    _add_arg_if_missing(
        parser,
        "--jimu-roof-triangle-profile",
        "--jimu-second-layer-triangle-profile",
        dest="jimu_roof_triangle_profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use red triangle panels for the four roof roles above the second square layer.",
    )
    _add_arg_if_missing(
        parser,
        "--jimu-demo-triangle-apriltag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force this wrapper to use the local AprilTag assembly-anchor localization path.",
    )
    _add_arg_if_missing(parser, "--jimu-demo-triangle-relation-slots", type=int, default=DEFAULT_RELATION_SLOTS)
    _add_arg_if_missing(parser, "--jimu-demo-triangle-fixed-batch-size", type=int, default=DEFAULT_FIXED_BATCH_SIZE)
    _add_arg_if_missing(parser, "--jimu-demo-triangle-fast-top-pairs", type=int, default=DEFAULT_FAST_TOP_PAIRS)
    _add_arg_if_missing(
        parser,
        "--jimu-second-layer-relation-slots",
        type=int,
        default=DEFAULT_SECOND_LAYER_RELATION_SLOTS,
    )
    _add_arg_if_missing(
        parser,
        "--jimu-second-layer-fixed-batch-size",
        type=int,
        default=DEFAULT_SECOND_LAYER_FIXED_BATCH_SIZE,
    )
    _add_arg_if_missing(parser, "--jimu-roof-relation-slots", type=int, default=DEFAULT_ROOF_RELATION_SLOTS)
    _add_arg_if_missing(parser, "--jimu-roof-fixed-batch-size", type=int, default=DEFAULT_ROOF_FIXED_BATCH_SIZE)
    _add_arg_if_missing(parser, "--jimu-roof-hover-height", type=float, default=0.0)
    _add_arg_if_missing(parser, "--jimu-roof-parallel-hover-height", type=float, default=0.0)
    _add_arg_if_missing(parser, "--jimu-roof-release-retreat-height", type=float, default=0.05)
    _add_arg_if_missing(parser, "--jimu-roof-parallel-sources-per-grasp", type=int, default=4)
    _add_arg_if_missing(parser, "--jimu-roof-hover-variants-per-source", type=int, default=6)
    _add_arg_if_missing(
        parser,
        "--jimu-roof-uniform-preplace-height",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep every roof pre-place/hover candidate at the same world-Z height above release.",
    )
    _add_arg_if_missing(
        parser,
        "--jimu-roof-uniform-preplace-height-m",
        type=float,
        default=DEFAULT_ROOF_UNIFORM_PREPLACE_HEIGHT_M,
        help="World-Z height above the roof release pose when uniform roof pre-place height is enabled.",
    )
    _add_arg_if_missing(
        parser,
        "--jimu-roof-max-hover-candidates-per-grasp",
        type=int,
        default=DEFAULT_ROOF_MAX_HOVER_CANDIDATES_PER_GRASP,
    )
    _add_arg_if_missing(parser, "--jimu-roof-hover-low-height", type=float, default=0.03)
    _add_arg_if_missing(parser, "--jimu-roof-hover-original-extra-m", type=float, default=0.03)
    _add_arg_if_missing(parser, "--jimu-roof-hover-outward-distance", type=float, default=0.035)
    _add_arg_if_missing(parser, "--jimu-roof-hover-outward-up-m", type=float, default=0.02)
    _add_arg_if_missing(parser, "--jimu-roof-post-place-retreat-m", type=float, default=0.05)
    _add_arg_if_missing(parser, "--jimu-roof-post-place-tilt-retreat-m", type=float, default=0.05)
    _add_arg_if_missing(parser, "--jimu-roof-post-place-retreat-lateral-step-m", type=float, default=0.006)
    _add_arg_if_missing(parser, "--jimu-roof-post-place-retreat-forward-extra-m", type=float, default=0.010)
    _add_arg_if_missing(parser, "--jimu-roof-post-place-retreat-candidate-count", type=int, default=16)
    _add_arg_if_missing(
        parser,
        "--jimu-roof-scene-obstacle-box-scale",
        type=float,
        default=DEFAULT_ROOF_SCENE_OBSTACLE_BOX_SCALE,
        help=(
            "Object-specific planner/sim collision-box scale for placed roof triangle panels. "
            "This reduces false collisions from the triangle mesh's rectangular bounding box."
        ),
    )
    _add_arg_if_missing(
        parser,
        "--jimu-roof-curobo-mesh-obstacles",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ROOF_CUROBO_MESH_OBSTACLES,
        help=(
            "Use the actual triangle GLB mesh as cuRobo world obstacles for placed roof panels. "
            "Other Jimu parts still use their normal cuboid obstacles."
        ),
    )
    _add_arg_if_missing(
        parser,
        "--jimu-dry-run-return-linear-fallback",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DRY_RUN_RETURN_LINEAR_FALLBACK,
        help=(
            "Allow a dry-run-only linear joint interpolation fallback for roof return_to_cycle_start "
            "after cuRobo/MPLib fail. Disabled by default so simulation exposes real planner failures."
        ),
    )
    _add_arg_if_missing(
        parser,
        "--jimu-validate-post-place-return",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before accepting a roof post-place clearance candidate, also plan the "
            "subsequent return_to_cycle_start path from that candidate endpoint."
        ),
    )
    _add_arg_if_missing(
        parser,
        "--jimu-roof-post-place-retreat-up-ratio",
        type=float,
        default=DEFAULT_ROOF_POST_PLACE_RETREAT_UP_RATIO,
        help="Add world-Z lift to roof post-place retreat candidates as ratio * retreat_m; 1.0 gives a 45-degree diagonal.",
    )
    _add_arg_if_missing(
        parser,
        "--jimu-roof-post-place-followup-up-m",
        type=float,
        default=DEFAULT_ROOF_POST_PLACE_FOLLOWUP_UP_M,
        help="After the first roof post-place retreat, add a second world-Z clearance segment before returning.",
    )
    _add_arg_if_missing(
        parser,
        "--jimu-roof-post-place-followup-side-m",
        type=float,
        default=DEFAULT_ROOF_POST_PLACE_FOLLOWUP_SIDE_M,
        help="Optional same-plane side component for the second roof post-place clearance segment.",
    )
    _add_arg_if_missing(
        parser,
        "--jimu-roof-post-place-free-motiongen-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow ordinary unconstrained MotionGen if a roof post-place retreat candidate cannot be "
            "planned with cuRobo PoseCostMetric. Disabled by default so retreat stays constrained."
        ),
    )
    _add_arg_if_missing(
        parser,
        "--jimu-triangle-tray-slot-yaw-offset-deg",
        type=float,
        default=DEFAULT_TRIANGLE_TRAY_SLOT_YAW_OFFSET_DEG,
        help="Rotate only the triangle panel source poses inside the tray slots; square plates and roof targets are unchanged.",
    )
    _add_arg_if_missing(
        parser,
        "--jimu-pregrasp-extra-world-z-m",
        type=float,
        default=DEFAULT_PREGRASP_EXTRA_WORLD_Z_M,
        help="Raise non-roof Jimu pre_grasp poses by this world-Z offset before IK preselection.",
    )
    _add_arg_if_missing(
        parser,
        "--jimu-pregrasp-fallback-world-z-m",
        type=float,
        default=DEFAULT_PREGRASP_FALLBACK_WORLD_Z_M,
        help="Also include non-roof Jimu pre_grasp candidates at this lower world-Z offset for reachability.",
    )
    _add_arg_if_missing(
        parser,
        "--jimu-pregrasp-emergency-world-z-m",
        type=float,
        default=DEFAULT_PREGRASP_EMERGENCY_WORLD_Z_M,
        help="Final non-roof Jimu pre_grasp height fallback kept for tray slots that lose IK at the raised heights.",
    )
    _add_arg_if_missing(
        parser,
        "--jimu-pregrasp-legacy-low-world-z-m",
        type=float,
        default=DEFAULT_PREGRASP_LEGACY_LOW_WORLD_Z_M,
        help="Keep the old low non-roof pre_grasp fallback while preferring the raised pre_grasp heights.",
    )
    _add_arg_if_missing(
        parser,
        "--jimu-roof-pregrasp-extra-world-z-m",
        type=float,
        default=DEFAULT_ROOF_PREGRASP_EXTRA_WORLD_Z_M,
        help="Raise roof-triangle Jimu pre_grasp poses by this world-Z offset before IK preselection.",
    )
    _add_arg_if_missing(
        parser,
        "--jimu-roof-pregrasp-fallback-world-z-m",
        type=float,
        default=DEFAULT_ROOF_PREGRASP_FALLBACK_WORLD_Z_M,
        help="Also include roof-triangle pre_grasp candidates at this lower world-Z offset when the primary offset is higher.",
    )
    _add_arg_if_missing(
        parser,
        "--jimu-roof-pregrasp-emergency-world-z-m",
        type=float,
        default=DEFAULT_ROOF_PREGRASP_EMERGENCY_WORLD_Z_M,
        help="Final roof-triangle pre_grasp height fallback kept for reachability when the raised pre_grasp is outside IK.",
    )
    _add_arg_if_missing(
        parser,
        "--jimu-roof-pregrasp-legacy-low-world-z-m",
        type=float,
        default=DEFAULT_ROOF_PREGRASP_LEGACY_LOW_WORLD_Z_M,
        help="Keep the old low roof pre_grasp fallback while preferring the raised roof pre_grasp heights.",
    )
    _add_arg_if_missing(
        parser,
        "--jimu-roof-pregrasp-safety-low-world-z-m",
        type=float,
        default=DEFAULT_ROOF_PREGRASP_SAFETY_LOW_WORLD_Z_M,
        help="Keep the original lowest roof pre_grasp fallback so all four tray triangle slots stay reachable.",
    )
    _add_arg_if_missing(
        parser,
        "--jimu-roof-align-grasp-opening-to-panel-normal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,
    )
    _add_arg_if_missing(
        parser,
        "--jimu-roof-skip-linear-transport-start-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For roof panels only, allow the linear transport fallback to ignore the current start-state contact check.",
    )
    _add_arg_if_missing(
        parser,
        "--jimu-skip-return-to-cycle-start-after-roof-place",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For roof panels, stop at the solved post-place clearance pose instead of forcing the empty gripper "
            "back to the original cycle_start_q. Disabled by default so roof cycles behave like wall cycles."
        ),
    )
    _add_arg_if_missing(parser, "--jimu-roof-layer-z-extra", type=float, default=0.0)
    _add_arg_if_missing(
        parser,
        "--jimu-return-to-cycle-start-after-place",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Return the empty gripper to the recorded cycle_start_q after every placed Jimu part. "
            "Disable only for fast headless regression tests where the next pick can safely start "
            "from the previous clearance pose."
        ),
    )
    parser.description = (
        "Jimu Demo_Triangle bridge: triangle-roof profile using the local portable "
        "AprilTag/SAM6D and Realman execution path."
    )
    return parser


def _apply_demo_triangle_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not bool(getattr(args, "jimu_demo_triangle_profile", True)):
        return args

    args.jimu_build_layers = "two"
    args.jimu_roof_triangle_profile = True
    args.jimu_second_layer_triangle_profile = False
    args.jimu_localization_mode = "assembly"
    explicit_fixed_scene = _has_explicit_fixed_scene_arg()
    explicit_live_apriltag = _argv_has_option("--jimu-apriltag-anchor-localization")
    if explicit_fixed_scene and not explicit_live_apriltag:
        fixed_scene_file = str(getattr(args, "sam6d_fixed_scene_result_file", "") or "").strip()
        args.sam6d_fixed_scene_result_file = fixed_scene_file
        args.jimu_demo_triangle_apriltag = False
        args.jimu_apriltag_anchor_localization = False
        args.jimu_tabletop_anchor_localization = False
        print(
            "[triangle-roof] using fixed SAM6D scene result; skipping apriltag/anchor live localization "
            f"and forcing jimu_demo_triangle_apriltag=False: {args.sam6d_fixed_scene_result_file}"
        )
    elif bool(getattr(args, "jimu_demo_triangle_apriltag", True)) and (
        explicit_live_apriltag or not explicit_fixed_scene
    ):
        args.jimu_apriltag_anchor_localization = True
        args.sam6d_fixed_scene_result_file = ""
        if not (
            _argv_has_option("--jimu-canonical-snap-cardinal")
            or _argv_has_option("--no-jimu-canonical-snap-cardinal")
        ):
            args.jimu_canonical_snap_cardinal = False

    if not _argv_has_option("--cycle-object-names"):
        args.cycle_object_names = list(portable.JIMU_PICK_ROLES)
    if not _argv_has_option("--jimu-scene-roles"):
        args.jimu_scene_roles = [portable.JIMU_FLOOR_ROLE, *portable.JIMU_TRAY_SLOT_ROLES]
        if bool(getattr(args, "jimu_base_support_obstacles", True)):
            args.jimu_scene_roles.extend(role for role in portable.JIMU_BASE_SUPPORT_ROLES if role not in args.jimu_scene_roles)
    if not _argv_has_option("--repeat-count"):
        args.repeat_count = len(list(args.cycle_object_names))
    args.sam3_full_scene_keep_multi_instances = True
    args.sam3_max_masks_per_item = max(int(getattr(args, "sam3_max_masks_per_item", 1) or 1), len(args.jimu_scene_roles))

    relation_slots = max(1, int(getattr(args, "jimu_demo_triangle_relation_slots", DEFAULT_RELATION_SLOTS) or DEFAULT_RELATION_SLOTS))
    fixed_batch_size = 16
    fast_top_pairs = max(
        1,
        int(getattr(args, "jimu_demo_triangle_fast_top_pairs", DEFAULT_FAST_TOP_PAIRS) or DEFAULT_FAST_TOP_PAIRS),
    )
    args.fast_chain_screening = True
    args.fast_chain_relation_ik_slots = relation_slots
    args.fast_chain_ik_seeds = 32
    args.fast_chain_cuda_graph_ik = True
    args.fast_chain_cuda_graph_ik_fixed_batch_size = fixed_batch_size
    args.fast_chain_cuda_graph_ik_max_batch_size = fixed_batch_size
    args.jimu_demo_triangle_fixed_batch_size = fixed_batch_size
    args.jimu_second_layer_fixed_batch_size = 16
    args.jimu_roof_fixed_batch_size = 16
    args.fast_chain_top_pairs = fast_top_pairs
    args.fast_chain_place_rank_grasp_limit = min(relation_slots, 16)
    args.fixed_tabletop_fast_chain_place_rank_grasp_limit = min(relation_slots, 16)
    args.fast_chain_allow_legacy_fallback = False
    if not (
        _argv_has_option("--jimu-linear-joint-transport-fallback")
        or _argv_has_option("--no-jimu-linear-joint-transport-fallback")
    ):
        args.jimu_linear_joint_transport_fallback = False
    args.jimu_parallel_grasp_place = True
    if not (
        _argv_has_option("--jimu-parallel-grasp-place-snap-yaw-90")
        or _argv_has_option("--no-jimu-parallel-grasp-place-snap-yaw-90")
    ):
        args.jimu_parallel_grasp_place_snap_yaw_90 = False
    args.jimu_parallel_grasp_place_max_sources_per_grasp = max(
        1,
        int(getattr(args, "jimu_parallel_grasp_place_max_sources_per_grasp", 1) or 1),
    )
    if not _argv_has_option("--direct-grasp-object-axis-shifts-m"):
        args.direct_grasp_object_axis_shifts_m = [0.0]
    if not _argv_has_option("--direct-grasp-z-lifts-m"):
        args.direct_grasp_z_lifts_m = [0.0]
    if not _argv_has_option("--direct-grasp-max-axis-shift-ratio"):
        args.direct_grasp_max_axis_shift_ratio = float(getattr(args, "direct_grasp_max_axis_shift_ratio", 0.0) or 0.0)
    if (
        hasattr(args, "joint_search_start_collision_lift_m")
        and not _argv_has_option("--joint-search-start-collision-lift-m")
    ):
        args.joint_search_start_collision_lift_m = DEFAULT_POST_GRASP_START_LIFT_M
    if hasattr(args, "post_grasp_lift_height_m") and not _argv_has_option("--post-grasp-lift-height-m"):
        args.post_grasp_lift_height_m = max(
            float(getattr(args, "post_grasp_lift_height_m", 0.0) or 0.0),
            DEFAULT_INDEPENDENT_POST_GRASP_LIFT_M,
        )
    args.jimu_place_symmetry_enabled = True
    args.jimu_place_symmetry_deg = [0.0, 180.0]
    if hasattr(args, "jimu_roof_align_grasp_opening_to_panel_normal"):
        if bool(getattr(args, "jimu_roof_align_grasp_opening_to_panel_normal", False)):
            print("[triangle-roof] ignoring roof panel-normal grasp roll; roof grasp is tilt-only")
        args.jimu_roof_align_grasp_opening_to_panel_normal = False
    if _roof_curobo_mesh_obstacles_enabled(args):
        _enable_roof_curobo_mesh_obstacles(args)
    args.transport_use_prefilter_q_goal = True
    args.transport_prefilter_q_goal_max_trials = 1
    args.transport_prefilter_q_goal_timeout = 2.0
    args.transport_prefilter_q_goal_num_trajopt_seeds = 1
    if hasattr(args, "transport_hover_extra_heights_m") and not _argv_has_option("--transport-hover-extra-heights-m"):
        args.transport_hover_extra_heights_m = [0.0]
    if (
        hasattr(args, "jimu_partial_open_before_grasp")
        and not _argv_has_option("--jimu-partial-open-before-grasp")
        and not _argv_has_option("--no-jimu-partial-open-before-grasp")
    ):
        args.jimu_partial_open_before_grasp = True
    if (
        hasattr(args, "jimu_release_partial_open_fraction")
        and not _argv_has_option("--jimu-release-partial-open-fraction")
    ):
        args.jimu_release_partial_open_fraction = float(getattr(args, "jimu_pregrasp_open_fraction", 0.79))
    if (
        hasattr(args, "jimu_full_open_after_post_place_clearance")
        and not _argv_has_option("--jimu-full-open-after-post-place-clearance")
        and not _argv_has_option("--no-jimu-full-open-after-post-place-clearance")
    ):
        args.jimu_full_open_after_post_place_clearance = False
    if (
        hasattr(args, "jimu_pair_first_pregrasp_motiongen")
        and not _argv_has_option("--jimu-pair-first-pregrasp-motiongen")
    ):
        args.jimu_pair_first_pregrasp_motiongen = False
    if (
        hasattr(args, "fuse_grasp_approach_stages")
        and not _argv_has_option("--fuse-grasp-approach-stages")
        and not _argv_has_option("--no-fuse-grasp-approach-stages")
    ):
        args.fuse_grasp_approach_stages = False
    if hasattr(args, "jimu_pregrasp_extra_world_z_m") and not _argv_has_option("--jimu-pregrasp-extra-world-z-m"):
        args.jimu_pregrasp_extra_world_z_m = max(
            float(getattr(args, "jimu_pregrasp_extra_world_z_m", 0.0) or 0.0),
            DEFAULT_PREGRASP_EXTRA_WORLD_Z_M,
        )
    if hasattr(args, "jimu_pregrasp_fallback_world_z_m") and not _argv_has_option("--jimu-pregrasp-fallback-world-z-m"):
        args.jimu_pregrasp_fallback_world_z_m = max(
            float(getattr(args, "jimu_pregrasp_fallback_world_z_m", 0.0) or 0.0),
            DEFAULT_PREGRASP_FALLBACK_WORLD_Z_M,
        )
    if hasattr(args, "jimu_pregrasp_emergency_world_z_m") and not _argv_has_option("--jimu-pregrasp-emergency-world-z-m"):
        args.jimu_pregrasp_emergency_world_z_m = max(
            float(getattr(args, "jimu_pregrasp_emergency_world_z_m", 0.0) or 0.0),
            DEFAULT_PREGRASP_EMERGENCY_WORLD_Z_M,
        )
    if hasattr(args, "jimu_pregrasp_legacy_low_world_z_m") and not _argv_has_option("--jimu-pregrasp-legacy-low-world-z-m"):
        args.jimu_pregrasp_legacy_low_world_z_m = max(
            float(getattr(args, "jimu_pregrasp_legacy_low_world_z_m", 0.0) or 0.0),
            DEFAULT_PREGRASP_LEGACY_LOW_WORLD_Z_M,
        )
    if hasattr(args, "jimu_roof_pregrasp_extra_world_z_m") and not _argv_has_option("--jimu-roof-pregrasp-extra-world-z-m"):
        args.jimu_roof_pregrasp_extra_world_z_m = max(
            float(getattr(args, "jimu_roof_pregrasp_extra_world_z_m", 0.0) or 0.0),
            DEFAULT_ROOF_PREGRASP_EXTRA_WORLD_Z_M,
        )
    if (
        hasattr(args, "jimu_roof_pregrasp_fallback_world_z_m")
        and not _argv_has_option("--jimu-roof-pregrasp-fallback-world-z-m")
    ):
        args.jimu_roof_pregrasp_fallback_world_z_m = max(
            float(getattr(args, "jimu_roof_pregrasp_fallback_world_z_m", 0.0) or 0.0),
            DEFAULT_ROOF_PREGRASP_FALLBACK_WORLD_Z_M,
        )
    if (
        hasattr(args, "jimu_roof_pregrasp_emergency_world_z_m")
        and not _argv_has_option("--jimu-roof-pregrasp-emergency-world-z-m")
    ):
        args.jimu_roof_pregrasp_emergency_world_z_m = max(
            float(getattr(args, "jimu_roof_pregrasp_emergency_world_z_m", 0.0) or 0.0),
            DEFAULT_ROOF_PREGRASP_EMERGENCY_WORLD_Z_M,
        )
    if (
        hasattr(args, "jimu_roof_pregrasp_legacy_low_world_z_m")
        and not _argv_has_option("--jimu-roof-pregrasp-legacy-low-world-z-m")
    ):
        args.jimu_roof_pregrasp_legacy_low_world_z_m = max(
            float(getattr(args, "jimu_roof_pregrasp_legacy_low_world_z_m", 0.0) or 0.0),
            DEFAULT_ROOF_PREGRASP_LEGACY_LOW_WORLD_Z_M,
        )
    if (
        hasattr(args, "jimu_roof_pregrasp_safety_low_world_z_m")
        and not _argv_has_option("--jimu-roof-pregrasp-safety-low-world-z-m")
    ):
        args.jimu_roof_pregrasp_safety_low_world_z_m = max(
            float(getattr(args, "jimu_roof_pregrasp_safety_low_world_z_m", 0.0) or 0.0),
            DEFAULT_ROOF_PREGRASP_SAFETY_LOW_WORLD_Z_M,
        )
    if (
        hasattr(args, "strict_short_linear_waypoint_pos_tol_m")
        and not _argv_has_option("--strict-short-linear-waypoint-pos-tol-m")
    ):
        args.strict_short_linear_waypoint_pos_tol_m = 0.008
    if (
        hasattr(args, "strict_final_contact_waypoint_pos_tol_m")
        and not _argv_has_option("--strict-final-contact-waypoint-pos-tol-m")
    ):
        args.strict_final_contact_waypoint_pos_tol_m = 0.008
    if (
        hasattr(args, "curobo_approach_metric_locked_axis_tol_m")
        and not _argv_has_option("--curobo-approach-metric-locked-axis-tol-m")
    ):
        args.curobo_approach_metric_locked_axis_tol_m = 0.008
    if (
        hasattr(args, "short_linear_endpoint_ik_first")
        and not _argv_has_option("--short-linear-endpoint-ik-first")
        and not _argv_has_option("--no-short-linear-endpoint-ik-first")
    ):
        args.short_linear_endpoint_ik_first = False
    if not _argv_has_option("--skip-post-place-clearance"):
        # Wall panels should retreat along the validated final-contact path
        # instead of replanning a new clearance motion that can rotate the wrist.
        # Roof candidates still set force_replan_post_place_clearance per item
        # because they need the roof-specific diagonal retreat set.
        args.force_replan_post_place_clearance = False
    if hasattr(args, "real_control_hz") and not _argv_has_option("--real-control-hz"):
        args.real_control_hz = 30.0
    if hasattr(args, "real_max_delta_per_step") and not _argv_has_option("--real-max-delta-per-step"):
        args.real_max_delta_per_step = 0.1
    if (
        hasattr(args, "dry_run_motion_window_scale")
        and not bool(getattr(args, "execute_real", False))
        and str(getattr(args, "render_mode", "") or "") == "human"
        and not _argv_has_option("--dry-run-motion-window-scale")
    ):
        args.dry_run_motion_window_scale = 1.0
    if (
        not bool(getattr(args, "execute_real", False))
        and str(getattr(args, "render_mode", "") or "") == "human"
    ):
        if hasattr(args, "dry_run_motion_window_min_s") and not _argv_has_option("--dry-run-motion-window-min-s"):
            args.dry_run_motion_window_min_s = max(
                float(getattr(args, "dry_run_motion_window_min_s", 0.0) or 0.0),
                0.6,
            )
        if hasattr(args, "dry_run_motion_window_max_s") and not _argv_has_option("--dry-run-motion-window-max-s"):
            args.dry_run_motion_window_max_s = 2.5
        if hasattr(args, "jimu_render_motion_scale") and not _argv_has_option("--jimu-render-motion-scale"):
            args.jimu_render_motion_scale = max(
                float(getattr(args, "jimu_render_motion_scale", 0.25) or 0.25),
                1.0,
            )
        if hasattr(args, "jimu_render_motion_min_s") and not _argv_has_option("--jimu-render-motion-min-s"):
            args.jimu_render_motion_min_s = max(
                float(getattr(args, "jimu_render_motion_min_s", 0.25) or 0.25),
                0.6,
            )
        if hasattr(args, "jimu_render_motion_max_s") and not _argv_has_option("--jimu-render-motion-max-s"):
            args.jimu_render_motion_max_s = max(
                float(getattr(args, "jimu_render_motion_max_s", 1.5) or 1.5),
                2.5,
            )
    if not _argv_has_option("--topdown-grasp-max-insertion-depth"):
        args.topdown_grasp_max_insertion_depth = DEFAULT_TRIANGLE_TOPDOWN_GRASP_MAX_INSERTION_DEPTH
    args.empty_grasp_check_after_lift = False
    args.empty_grasp_relocalize_target = False
    args.empty_grasp_max_relocalize_retries = 0
    args.jimu_enforce_layer_order = True
    args.joint_search_validate_final_contact = True
    args.validate_post_place_clearance_return_to_start = bool(
        getattr(args, "jimu_validate_post_place_return", True)
    )
    if hasattr(args, "joint_search_primary_fallback_after_fast_ik_fail"):
        args.joint_search_primary_fallback_after_fast_ik_fail = False
    if hasattr(args, "release_pose_error_safe_retries"):
        args.release_pose_error_safe_retries = 0
    if (
        hasattr(args, "reselect_target_on_planning_failure")
        and not _argv_has_option("--reselect-target-on-planning-failure")
        and not _argv_has_option("--no-reselect-target-on-planning-failure")
    ):
        args.reselect_target_on_planning_failure = True
    if (
        hasattr(args, "jimu_retry_next_tray_source_on_grasp_failure")
        and not _argv_has_option("--jimu-retry-next-tray-source-on-grasp-failure")
        and not _argv_has_option("--no-jimu-retry-next-tray-source-on-grasp-failure")
    ):
        args.jimu_retry_next_tray_source_on_grasp_failure = True
    if (
        hasattr(args, "jimu_planning_failure_source_retry_max")
        and not _argv_has_option("--jimu-planning-failure-source-retry-max")
    ):
        # Planning-only failures are usually deterministic for a target pose.  Keep
        # source swaps useful for a bad tray slot, but avoid looping over every
        # unused plate when the place-chain generator itself is failing.
        args.jimu_planning_failure_source_retry_max = 2
    if (
        hasattr(args, "strict_return_to_cycle_start")
        and not _argv_has_option("--strict-return-to-cycle-start")
        and not _argv_has_option("--no-strict-return-to-cycle-start")
    ):
        args.strict_return_to_cycle_start = True
    if (
        hasattr(args, "return_to_start_self_collision_audit")
        and not bool(getattr(args, "execute_real", False))
        and not _argv_has_option("--return-to-start-self-collision-audit")
        and not _argv_has_option("--no-return-to-start-self-collision-audit")
    ):
        # Keep dry-run viewer tests from rejecting an otherwise valid cuRobo
        # return due to the extra post-place visual/audit pass. Real execution
        # keeps the audit unless the caller explicitly overrides it.
        args.return_to_start_self_collision_audit = False
    if (
        hasattr(args, "jimu_dry_run_return_linear_fallback")
        and not _argv_has_option("--jimu-dry-run-return-linear-fallback")
        and not _argv_has_option("--no-jimu-dry-run-return-linear-fallback")
    ):
        args.jimu_dry_run_return_linear_fallback = DEFAULT_DRY_RUN_RETURN_LINEAR_FALLBACK
    if (
        hasattr(args, "jimu_skip_return_to_cycle_start_after_roof_place")
        and not _argv_has_option("--jimu-skip-return-to-cycle-start-after-roof-place")
        and not _argv_has_option("--no-jimu-skip-return-to-cycle-start-after-roof-place")
    ):
        args.jimu_skip_return_to_cycle_start_after_roof_place = False
    if (
        hasattr(args, "skip_return_to_cycle_start_after_final_place")
        and not _argv_has_option("--skip-return-to-cycle-start-after-final-place")
        and not _argv_has_option("--no-skip-return-to-cycle-start-after-final-place")
    ):
        args.skip_return_to_cycle_start_after_final_place = False
    if (
        hasattr(args, "skip_return_to_cycle_start")
        and not _argv_has_option("--skip-return-to-cycle-start")
    ):
        if bool(getattr(args, "jimu_return_to_cycle_start_after_place", True)):
            args.skip_return_to_cycle_start = False
        else:
            args.skip_return_to_cycle_start = True
            args.return_to_start_preplan = False

    install_jimu_object_specs_triangle(args)
    portable.install_jimu_place_rules(args)
    print(
        "[triangle-roof] post-place gripper override: "
        f"release_fraction={float(getattr(args, 'jimu_release_partial_open_fraction', 0.0)):.2f}, "
        f"release_value={portable._jimu_partial_release_gripper_value(args):.3f}, "
        f"same_as_pregrasp={portable._jimu_pregrasp_partial_open_value(args):.3f}, "
        f"full_open_after_clearance={bool(getattr(args, 'jimu_full_open_after_post_place_clearance', False))}"
    )
    print(
        "[triangle-roof] Demo_Triangle bridge enabled: "
        f"apriltag={bool(getattr(args, 'jimu_apriltag_anchor_localization', False))}, "
        f"execute_real={bool(getattr(args, 'execute_real', False))}, "
        f"roles={list(args.cycle_object_names)}, "
        f"relation_slots={relation_slots}, fixed_batch={fixed_batch_size}, top_pairs={fast_top_pairs}, "
        f"second_relation_slots={int(getattr(args, 'jimu_second_layer_relation_slots', DEFAULT_SECOND_LAYER_RELATION_SLOTS) or DEFAULT_SECOND_LAYER_RELATION_SLOTS)}, "
        f"second_fixed_batch={int(getattr(args, 'jimu_second_layer_fixed_batch_size', DEFAULT_SECOND_LAYER_FIXED_BATCH_SIZE) or DEFAULT_SECOND_LAYER_FIXED_BATCH_SIZE)}, "
        f"roof_relation_slots={int(getattr(args, 'jimu_roof_relation_slots', DEFAULT_ROOF_RELATION_SLOTS) or DEFAULT_ROOF_RELATION_SLOTS)}, "
        f"roof_fixed_batch={int(getattr(args, 'jimu_roof_fixed_batch_size', DEFAULT_ROOF_FIXED_BATCH_SIZE) or DEFAULT_ROOF_FIXED_BATCH_SIZE)}, "
        f"roof_hover_max={int(getattr(args, 'jimu_roof_max_hover_candidates_per_grasp', DEFAULT_ROOF_MAX_HOVER_CANDIDATES_PER_GRASP))}, "
        f"roof_uniform_preplace_z={_roof_uniform_preplace_height_enabled(args)}"
        f"/{_roof_uniform_preplace_height_m(args, None):.3f}m, "
        f"roof_post_retreat={_roof_post_place_retreat_m(args):.3f}m/"
        f"{int(getattr(args, 'jimu_roof_post_place_retreat_candidate_count', 16) or 16)}pts, "
        f"roof_box_scale={_roof_scene_obstacle_box_scale(args):.2f}, "
        f"roof_mesh_obstacles={_roof_curobo_mesh_obstacles_enabled(args)}, "
        f"roof_post_up_ratio={float(getattr(args, 'jimu_roof_post_place_retreat_up_ratio', DEFAULT_ROOF_POST_PLACE_RETREAT_UP_RATIO) or 0.0):.2f}, "
        f"roof_post_followup={float(getattr(args, 'jimu_roof_post_place_followup_up_m', DEFAULT_ROOF_POST_PLACE_FOLLOWUP_UP_M) or 0.0):.3f}m, "
        f"roof_post_free_motiongen_fallback={bool(getattr(args, 'jimu_roof_post_place_free_motiongen_fallback', False))}, "
        f"roof_skip_return={bool(getattr(args, 'jimu_skip_return_to_cycle_start_after_roof_place', False))}, "
        f"validate_post_return={bool(getattr(args, 'validate_post_place_clearance_return_to_start', False))}, "
        "roof_grasp_tilt_only=True, "
        f"triangle_tray_yaw_offset={float(getattr(args, 'jimu_triangle_tray_slot_yaw_offset_deg', DEFAULT_TRIANGLE_TRAY_SLOT_YAW_OFFSET_DEG) or 0.0):.1f}deg, "
        f"pregrasp_motiongen={bool(getattr(args, 'jimu_pair_first_pregrasp_motiongen', True))}, "
        f"fuse_grasp_approach={bool(getattr(args, 'fuse_grasp_approach_stages', True))}, "
        f"transport_hover_extra={list(getattr(args, 'transport_hover_extra_heights_m', []))}, "
        f"pregrasp_extra_z={float(getattr(args, 'jimu_pregrasp_extra_world_z_m', 0.0) or 0.0):.3f}m/"
        f"fallback={float(getattr(args, 'jimu_pregrasp_fallback_world_z_m', 0.0) or 0.0):.3f}m/"
        f"emergency={float(getattr(args, 'jimu_pregrasp_emergency_world_z_m', 0.0) or 0.0):.3f}m/"
        f"legacy_low={float(getattr(args, 'jimu_pregrasp_legacy_low_world_z_m', DEFAULT_PREGRASP_LEGACY_LOW_WORLD_Z_M) or 0.0):.3f}m/"
        f"roof={float(getattr(args, 'jimu_roof_pregrasp_extra_world_z_m', 0.0) or 0.0):.3f}m"
        f"+fallback={float(getattr(args, 'jimu_roof_pregrasp_fallback_world_z_m', 0.0) or 0.0):.3f}m"
        f"+emergency={float(getattr(args, 'jimu_roof_pregrasp_emergency_world_z_m', 0.0) or 0.0):.3f}m"
        f"+legacy_low={float(getattr(args, 'jimu_roof_pregrasp_legacy_low_world_z_m', DEFAULT_ROOF_PREGRASP_LEGACY_LOW_WORLD_Z_M) or 0.0):.3f}m"
        f"+safety_low={float(getattr(args, 'jimu_roof_pregrasp_safety_low_world_z_m', DEFAULT_ROOF_PREGRASP_SAFETY_LOW_WORLD_Z_M) or 0.0):.3f}m, "
        f"post_grasp_lift={float(getattr(args, 'joint_search_start_collision_lift_m', 0.0) or 0.0):.3f}m, "
        f"dry_motion_scale={float(getattr(args, 'dry_run_motion_window_scale', 0.0) or 0.0):.2f}, "
        f"render_min={float(getattr(args, 'jimu_render_motion_min_s', 0.0) or 0.0):.2f}s/"
        f"restore_min={float(getattr(args, 'dry_run_motion_window_min_s', 0.0) or 0.0):.2f}s, "
        f"stream={float(getattr(args, 'real_control_hz', 0.0) or 0.0):.1f}Hz/"
        f"{float(getattr(args, 'real_max_delta_per_step', 0.0) or 0.0):.3f}rad, "
        f"skip_return={bool(getattr(args, 'skip_return_to_cycle_start', False))}, "
        f"strict_return={bool(getattr(args, 'strict_return_to_cycle_start', True))}, "
        f"dry_return_linear_fallback={bool(getattr(args, 'jimu_dry_run_return_linear_fallback', False))}, "
        f"reselect_on_fail={bool(getattr(args, 'reselect_target_on_planning_failure', True))}, "
        f"source_retry={bool(getattr(args, 'jimu_retry_next_tray_source_on_grasp_failure', True))}, "
        f"return_audit={bool(getattr(args, 'return_to_start_self_collision_audit', True))}, "
        f"topdown_grasp_max_insertion_depth={float(args.topdown_grasp_max_insertion_depth):.3f}"
    )
    print(
        "[triangle-roof] first/second layers use square plates; roof roles use red_triangle specs; "
        "roof grasp is tilt-only by default, roof pre-place uses a single world-Z hover, "
        "and post-place retreat keeps release orientation while trying side+world-Z diagonal endpoints; "
        "planner/execution still delegates to rm75_jimu_four_wall_portable.py"
    )
    return args


def parse_args_triangle() -> argparse.Namespace:
    if not _should_pre_enable_apriltag():
        return _apply_demo_triangle_defaults(_ORIGINAL_PARSE_ARGS())

    original_argv = list(sys.argv)
    sys.argv = [*original_argv, "--jimu-apriltag-anchor-localization"]
    try:
        args = _ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = original_argv
    return _apply_demo_triangle_defaults(args)


def install_patches() -> None:
    _install_roof_role_constants()
    portable.build_arg_parser = build_arg_parser_triangle
    portable.parse_args = parse_args_triangle
    portable.install_jimu_object_specs = install_jimu_object_specs_triangle
    portable.install_jimu_place_rules = install_jimu_place_rules_triangle
    portable._jimu_second_layer_local_pose_specs = _jimu_second_layer_local_pose_specs_triangle
    portable._jimu_second_layer_target_pose_from_floor = _jimu_second_layer_target_pose_from_floor_triangle
    portable._jimu_floor_anchor_second_layer_plans = _jimu_floor_anchor_second_layer_and_roof_plans
    portable._jimu_layer_filtered_target_pool = _jimu_layer_filtered_target_pool_triangle
    portable._jimu_tray_slot_local_poses = _jimu_tray_slot_local_poses_triangle
    portable._jimu_validate_linear_joint_path = _jimu_validate_linear_joint_path_triangle
    portable._jimu_choose_next_tray_source_role = _choose_next_tray_source_role_triangle
    portable._make_jimu_parallel_grasp_place_candidate = _make_jimu_parallel_grasp_place_candidate_triangle
    portable._select_jimu_parallel_place_source_candidates = _select_jimu_parallel_place_source_candidates_triangle
    portable.direct._build_direct_grasp_candidates = _build_direct_grasp_candidates_triangle
    portable.direct._fast_chain_preselect_grasp_place_pair = _fast_chain_preselect_grasp_place_pair_triangle


def main() -> None:
    install_patches()
    portable.main()


if __name__ == "__main__":
    main()
