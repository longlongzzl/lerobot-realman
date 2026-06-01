#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import random
import re
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from transforms3d.euler import euler2mat, mat2euler


BETA_DIR = Path(__file__).resolve().parent
REPO_ROOT = BETA_DIR.parent
PICK_JIAOBANG_DIR = REPO_ROOT / "pick_jiaobang"
if str(PICK_JIAOBANG_DIR) not in sys.path:
    sys.path.insert(0, str(PICK_JIAOBANG_DIR))

import object_specs  # noqa: E402
import place_rules  # noqa: E402
import rm75_jiaobang_pick_place_targeted_curobo_direct_pre_place as direct  # noqa: E402
import rm75_jiaobang_pick_place_targeted_curobo_direct_pre_place_sam6d as direct_sam6d  # noqa: E402
import sam6d_to_assembly_state as assembly_sam6d  # noqa: E402


JIMU_PROVIDER_OBJECT_NAME = "red_bricks_cube"
JIMU_FLOOR_ROLE = "floor"
JIMU_FIRST_LAYER_ROLES = ("right_wall", "back_wall", "left_wall", "front_wall")
JIMU_SECOND_LAYER_ROLES = ("right_second_wall", "back_second_wall", "left_second_wall", "front_second_wall")
JIMU_PICK_ROLES = (*JIMU_FIRST_LAYER_ROLES, *JIMU_SECOND_LAYER_ROLES)
JIMU_SCENE_ROLES = (JIMU_FLOOR_ROLE, *JIMU_PICK_ROLES)
JIMU_DEFAULT_SIM_ASSET_FILE = BETA_DIR / "jimu_portable_repro" / "assets" / "red_jimu_plate_74x6x74.glb"
JIMU_PLATE_SIZE_M = 0.074
JIMU_PLATE_THICKNESS_M = 0.006
DEFAULT_JIMU_PHYSICAL_EXTENTS_M = np.asarray(
    [JIMU_PLATE_SIZE_M, JIMU_PLATE_THICKNESS_M, JIMU_PLATE_SIZE_M],
    dtype=np.float32,
)
DEFAULT_JIMU_MESH_EXTENTS_M = DEFAULT_JIMU_PHYSICAL_EXTENTS_M.copy()
DEFAULT_JIMU_CAD_TO_SIM_RPY_DEG = (90.0, 0.0, 0.0)
_JIMU_SECOND_LAYER_PARENT = {
    "right_second_wall": "right_wall",
    "back_second_wall": "back_wall",
    "left_second_wall": "left_wall",
    "front_second_wall": "front_wall",
}
_JIMU_RUNTIME_CONTEXT = threading.local()
_ORIGINAL_TARGETED_BUILD_PLACE_PLAN_VARIANTS = None
_ORIGINAL_FAST_CHAIN_RANK_PAIRED_RELATION_CANDIDATES = None
_ORIGINAL_FAST_CHAIN_RELATION_MATCH_KEY = None
_ORIGINAL_EVALUATE_CUROBO_POSE_CANDIDATES_MULTI_START = None
_ORIGINAL_COPY_LAST_CANDIDATE_COUNTS_TO_PROFILE = None
_ORIGINAL_PROFILE_PLAN_TO_JOINT_STATE = None
_ORIGINAL_PROFILE_PLAN_GOALSET_TO_POSES = None
_ORIGINAL_PROFILE_PLAN_BATCH_START_GOAL_PAIRS = None
_ORIGINAL_ATTACH_TRANSPORT_PAYLOAD_TO_CUROBO = None
_ORIGINAL_PROFILE_STAGE = None
_ORIGINAL_EXECUTE_POSE_PATH_STAGE = None
_ORIGINAL_REALMAN_SET_GRIPPER = None
_ORIGINAL_SYNC_DEMO_GRIPPER_STATE = None
_ORIGINAL_SELECT_RANDOM_CYCLE_TARGET = None


def _split_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [value]
    else:
        raw_items = list(value)
    result: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        for part in str(item).replace(",", " ").split():
            name = direct.curobo_wrapper.normalize_object_name(part)
            if name is None or name in seen:
                continue
            seen.add(name)
            result.append(name)
    return result


def _argv_has_option(option_name: str) -> bool:
    prefix = f"{option_name}="
    return any(arg == option_name or str(arg).startswith(prefix) for arg in sys.argv[1:])


def _default_pick_roles_for_layers(layers: str | None) -> list[str]:
    mode = str(layers or "two").strip().lower()
    if mode in {"1", "first", "first_layer", "one"}:
        return list(JIMU_FIRST_LAYER_ROLES)
    if mode in {"2", "two", "second", "two_layers", "full"}:
        return list(JIMU_PICK_ROLES)
    raise ValueError(f"invalid Jimu layer mode: {layers!r}")


def _default_scene_roles_for_layers(layers: str | None) -> list[str]:
    return [JIMU_FLOOR_ROLE, *_default_pick_roles_for_layers(layers)]


def _is_jimu_role_placed(scene_capture_cache: dict | None, role: str) -> bool:
    try:
        return bool(direct.targeted._is_cached_scene_object_placed(scene_capture_cache, role))
    except Exception:
        objects = scene_capture_cache.get("objects") if isinstance(scene_capture_cache, dict) else None
        entry = objects.get(role) if isinstance(objects, dict) else None
        return isinstance(entry, dict) and bool(entry.get("placed", False))


def _jimu_layer_filtered_target_pool(
    base_args: argparse.Namespace,
    pool: list[str],
    scene_capture_cache: dict | None,
) -> tuple[list[str], list[str]]:
    if not bool(getattr(base_args, "jimu_enforce_layer_order", True)):
        return pool, []
    normalized_pool = [
        direct.curobo_wrapper.normalize_object_name(item)
        for item in list(pool or [])
    ]
    normalized_pool = [item for item in normalized_pool if item is not None]
    if not any(role in set(JIMU_PICK_ROLES) for role in normalized_pool):
        return pool, []

    pool_set = set(normalized_pool)
    pending_first_layer = [
        role
        for role in JIMU_FIRST_LAYER_ROLES
        if role in pool_set and not _is_jimu_role_placed(scene_capture_cache, role)
    ]
    if not pending_first_layer:
        return pool, []
    allowed = set(pending_first_layer)
    return [role for role in normalized_pool if role in allowed], pending_first_layer


def select_random_cycle_target_jimu_layered(
    base_args,
    cycle_object_sequence,
    scene_capture_cache,
    available_rule_names,
    failed_targets_this_cycle: set[str],
    deferred_failed_targets: set[str] | None,
    cycle_idx: int,
) -> tuple[str | None, list[str], list[str]]:
    original = _ORIGINAL_SELECT_RANDOM_CYCLE_TARGET
    if original is None:
        return None, [], []
    if not bool(getattr(base_args, "jimu_enforce_layer_order", True)):
        return original(
            base_args,
            cycle_object_sequence,
            scene_capture_cache,
            available_rule_names,
            failed_targets_this_cycle,
            deferred_failed_targets,
            cycle_idx,
        )
    pool = direct.targeted._random_target_pool_for_cycle(
        base_args,
        cycle_object_sequence,
        scene_capture_cache,
        available_rule_names,
        cycle_idx,
    )
    gated_pool, pending_first_layer = _jimu_layer_filtered_target_pool(base_args, pool, scene_capture_cache)
    if not pending_first_layer:
        return original(
            base_args,
            cycle_object_sequence,
            scene_capture_cache,
            available_rule_names,
            failed_targets_this_cycle,
            deferred_failed_targets,
            cycle_idx,
        )

    failed = set(failed_targets_this_cycle or set())
    deferred = set(deferred_failed_targets or set())
    candidates = [name for name in gated_pool if name not in failed and name not in deferred]
    if not candidates:
        candidates = [name for name in gated_pool if name not in failed]
    if not candidates:
        print(
            "[jimu-order] first layer is not complete, but all first-layer candidates failed; "
            f"pending_first_layer={pending_first_layer}"
        )
        return None, gated_pool, []

    order = str(getattr(base_args, "target_selection_order", "random") or "random")
    if order == "cycle":
        selected = candidates[0]
    else:
        selected = random.choice(candidates)
    print(
        "[jimu-order] holding second layer until first layer is complete: "
        f"pending_first_layer={pending_first_layer}, candidates={candidates}, selected={selected}"
    )
    return selected, gated_pool, candidates


def _parse_role_instance_map(value: Any) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for item in _split_mapping_items(value):
        if ":" not in item and "=" not in item:
            raise ValueError(f"Invalid --jimu-role-instance-map item: {item!r}; use role:index")
        sep = ":" if ":" in item else "="
        role_raw, index_raw = item.split(sep, 1)
        role = direct.curobo_wrapper.normalize_object_name(role_raw)
        if role is None:
            raise ValueError(f"Invalid role in --jimu-role-instance-map: {role_raw!r}")
        mapping[role] = int(index_raw)
    return mapping


def _split_mapping_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [value]
    else:
        raw_items = list(value)
    items: list[str] = []
    for item in raw_items:
        for part in str(item).replace(",", " ").split():
            if part:
                items.append(part)
    return items


def _jimu_physical_extents(args: argparse.Namespace | None = None) -> np.ndarray:
    size = float(getattr(args, "jimu_plate_size_m", JIMU_PLATE_SIZE_M) if args is not None else JIMU_PLATE_SIZE_M)
    thickness = float(
        getattr(args, "jimu_plate_thickness_m", JIMU_PLATE_THICKNESS_M)
        if args is not None
        else JIMU_PLATE_THICKNESS_M
    )
    if not np.isfinite(size) or size <= 1e-6:
        size = JIMU_PLATE_SIZE_M
    if not np.isfinite(thickness) or thickness <= 1e-6:
        thickness = JIMU_PLATE_THICKNESS_M
    return np.asarray([size, thickness, size], dtype=np.float32)


def _load_scaled_jimu_extents(args: argparse.Namespace | None = None) -> np.ndarray:
    if not bool(getattr(args, "jimu_use_mesh_extents", False) if args is not None else False):
        return _jimu_physical_extents(args)
    try:
        spec = object_specs.get_object_spec(JIMU_PROVIDER_OBJECT_NAME)
        if spec is None:
            return _jimu_physical_extents(args)
        _, sim_scale = object_specs.resolve_object_spec_scales(spec)
        mesh_path = Path(spec.sim_asset_file or spec.mesh_file).expanduser()
        loaded = trimesh.load(mesh_path, force="scene")
        bounds = np.asarray(loaded.bounds, dtype=np.float32)
        extents = (bounds[1] - bounds[0]) * float(sim_scale)
        if extents.shape == (3,) and np.all(np.isfinite(extents)) and float(np.min(extents)) > 1e-6:
            return extents.astype(np.float32)
    except Exception as exc:
        print(f"[jimu config] warning: failed to read red_jimu_cube extents, using fallback: {exc}")
    return _jimu_physical_extents(args)


def _jimu_sim_asset_file_override(args: argparse.Namespace | None = None) -> str | None:
    raw = str(getattr(args, "jimu_sim_asset_file", "") if args is not None else "").strip()
    if raw:
        return str(Path(raw).expanduser())
    if JIMU_DEFAULT_SIM_ASSET_FILE.exists():
        return str(JIMU_DEFAULT_SIM_ASSET_FILE)
    return None


def _rpy_deg_from_matrix(rotation: np.ndarray) -> tuple[float, float, float]:
    roll, pitch, yaw = mat2euler(np.asarray(rotation, dtype=np.float64).reshape(3, 3), axes="sxyz")
    values = np.rad2deg([roll, pitch, yaw])
    return tuple(float(v) for v in values)


def _normalize_vec(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray | None:
    arr = np.asarray(vec, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= eps:
        return None
    return (arr / norm).astype(np.float32)


def _horizontal_axis(vec: np.ndarray, *, snap_cardinal: bool) -> np.ndarray | None:
    arr = np.asarray(vec, dtype=np.float32).reshape(3).copy()
    arr[2] = 0.0
    axis = _normalize_vec(arr)
    if axis is None:
        return None
    if not snap_cardinal:
        return axis
    cardinals = (
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, -1.0, 0.0], dtype=np.float32),
    )
    return max(cardinals, key=lambda candidate: float(np.dot(axis, candidate))).copy()


def _rotation_from_xy_z(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    x = _normalize_vec(x_axis)
    y = _normalize_vec(y_axis)
    z = _normalize_vec(z_axis)
    if x is None or y is None or z is None:
        raise ValueError("cannot construct Jimu canonical frame from degenerate axis")
    R = np.column_stack([x, y, z]).astype(np.float32)
    if float(np.linalg.det(R)) < 0.0:
        x = -x
        R = np.column_stack([x, y, z]).astype(np.float32)
    return R


def _raw_camera_pose_from_base_pose(args: argparse.Namespace, T_base_cam: np.ndarray, T_base_obj: np.ndarray) -> np.ndarray:
    T_base_cam = np.asarray(T_base_cam, dtype=np.float32).reshape(4, 4)
    T_base_obj = np.asarray(T_base_obj, dtype=np.float32).reshape(4, 4)
    inv_fix = np.linalg.inv(_jimu_cad_to_sim_local_fix(args)).astype(np.float32)
    if bool(getattr(args, "use_direct_camera_extrinsic", False)):
        T_cam_obj_fixed = np.linalg.inv(T_base_cam).astype(np.float32) @ T_base_obj
    else:
        T_cam_obj_fixed = T_base_cam @ T_base_obj
    return (T_cam_obj_fixed @ inv_fix).astype(np.float32)


def _canonical_jimu_base_pose(
    role: str,
    T_base_obj: np.ndarray,
    *,
    floor_T_base_obj: np.ndarray | None,
    args: argparse.Namespace,
) -> np.ndarray:
    T_base_obj = np.asarray(T_base_obj, dtype=np.float32).reshape(4, 4)
    out = np.eye(4, dtype=np.float32)
    out[:3, 3] = T_base_obj[:3, 3]
    snap_cardinal = bool(getattr(args, "jimu_canonical_snap_cardinal", True))
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    if role == JIMU_FLOOR_ROLE:
        x_axis = _horizontal_axis(T_base_obj[:3, 0], snap_cardinal=snap_cardinal)
        if x_axis is None:
            x_axis = _horizontal_axis(T_base_obj[:3, 2], snap_cardinal=snap_cardinal)
        if x_axis is None:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        y_axis = world_up
        z_axis = _normalize_vec(np.cross(x_axis, y_axis))
        if z_axis is None:
            z_axis = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        out[:3, :3] = _rotation_from_xy_z(x_axis, y_axis, z_axis)
        return out

    y_axis = _horizontal_axis(T_base_obj[:3, 1], snap_cardinal=snap_cardinal)
    if y_axis is None:
        best_axis = None
        best_score = -float("inf")
        for axis_idx in (0, 1, 2):
            axis = _horizontal_axis(T_base_obj[:3, axis_idx], snap_cardinal=snap_cardinal)
            if axis is None:
                continue
            score = 1.0 - abs(float(T_base_obj[2, axis_idx]))
            if score > best_score:
                best_score = score
                best_axis = axis
        y_axis = best_axis
    if y_axis is None and floor_T_base_obj is not None:
        floor_p = np.asarray(floor_T_base_obj, dtype=np.float32).reshape(4, 4)[:3, 3]
        y_axis = _horizontal_axis(T_base_obj[:3, 3] - floor_p, snap_cardinal=snap_cardinal)
    if y_axis is None:
        y_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    z_axis = world_up
    x_axis = _normalize_vec(np.cross(y_axis, z_axis))
    if x_axis is None:
        x_axis = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    y_axis = _normalize_vec(np.cross(z_axis, x_axis))
    if y_axis is None:
        y_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    out[:3, :3] = _rotation_from_xy_z(x_axis, y_axis, z_axis)
    return out


def _canonicalize_jimu_result_item(
    item: dict,
    role: str,
    args: argparse.Namespace,
    T_base_cam: np.ndarray,
    *,
    floor_T_base_obj: np.ndarray | None,
) -> tuple[dict, np.ndarray]:
    corrected = copy.deepcopy(item)
    T_cam_obj = np.asarray(corrected["T_cam_obj"], dtype=np.float32).reshape(4, 4)
    T_base_before = _map_cam_pose_to_base_for_assignment(args, T_base_cam, T_cam_obj)
    if not bool(getattr(args, "jimu_canonicalize_local_frames", True)):
        return corrected, T_base_before
    T_base_after = _canonical_jimu_base_pose(
        role,
        T_base_before,
        floor_T_base_obj=floor_T_base_obj,
        args=args,
    )
    corrected["T_cam_obj"] = _raw_camera_pose_from_base_pose(args, T_base_cam, T_base_after)
    corrected["jimu_canonical_frame_applied"] = True
    corrected["jimu_canonical_axis_dot_base_z_before"] = T_base_before[2, :3].astype(float).tolist()
    corrected["jimu_canonical_axis_dot_base_z_after"] = T_base_after[2, :3].astype(float).tolist()
    corrected["jimu_canonical_base_translation"] = T_base_after[:3, 3].astype(float).tolist()
    print(
        f"[jimu-sam6d] canonicalized {role} local frame: "
        f"axis_z_before={np.round(T_base_before[2, :3], 3).tolist()} "
        f"axis_z_after={np.round(T_base_after[2, :3], 3).tolist()}"
    )
    return corrected, T_base_after


def _jimu_symmetry_degrees(args: argparse.Namespace) -> list[float]:
    raw_values = list(getattr(args, "jimu_place_symmetry_deg", [0.0, 90.0, 180.0, 270.0]) or [])
    values: list[float] = []
    seen: set[int] = set()
    for value in raw_values:
        deg = ((float(value) + 180.0) % 360.0) - 180.0
        if abs(deg + 180.0) <= 1e-6:
            deg = 180.0
        bucket = int(round(deg))
        if bucket in seen:
            continue
        seen.add(bucket)
        values.append(float(deg))
    return values or [0.0]


def _jimu_wall_local_pose_specs(args: argparse.Namespace | None = None) -> dict[str, place_rules.LocalPoseSpec]:
    # The red_jimu_cube mesh uses local Y as the thin axis. In the floor frame,
    # local Y is the floor normal, local X/Z span the plate.
    extents = _load_scaled_jimu_extents(args)
    half_x = float(extents[0] * 0.5)
    half_thick = float(extents[1] * 0.5)
    half_z = float(extents[2] * 0.5)

    wall_center_y = half_z
    x_offset = half_x + half_thick
    z_offset = half_z + half_thick

    rotations = {
        "right_wall": np.column_stack(
            [
                [0.0, 0.0, 1.0],  # object local X along floor +Z
                [1.0, 0.0, 0.0],  # object local Y/thickness outward
                [0.0, 1.0, 0.0],  # object local Z vertical
            ]
        ),
        "left_wall": np.column_stack(
            [
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        "back_wall": np.column_stack(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        "front_wall": np.column_stack(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    }
    positions = {
        "right_wall": (x_offset, wall_center_y, 0.0),
        "left_wall": (-x_offset, wall_center_y, 0.0),
        "back_wall": (0.0, wall_center_y, z_offset),
        "front_wall": (0.0, wall_center_y, -z_offset),
    }
    return {
        role: place_rules.LocalPoseSpec(
            position=tuple(float(v) for v in positions[role]),
            rpy_deg=_rpy_deg_from_matrix(rotations[role]),
        )
        for role in JIMU_FIRST_LAYER_ROLES
    }


def _jimu_second_layer_local_pose_specs(args: argparse.Namespace | None = None) -> dict[str, place_rules.LocalPoseSpec]:
    extents = _load_scaled_jimu_extents(args)
    wall_height = float(extents[2])
    z_extra = float(getattr(args, "jimu_second_layer_z_extra", 0.0) if args is not None else 0.0)
    center_offset_z = wall_height + z_extra
    return {
        role: place_rules.LocalPoseSpec(
            position=(0.0, 0.0, center_offset_z),
            rpy_deg=(0.0, 0.0, 0.0),
        )
        for role in JIMU_SECOND_LAYER_ROLES
    }


def _jimu_cad_to_sim_rpy_deg(args: argparse.Namespace | None = None) -> tuple[float, float, float]:
    mode = str(getattr(args, "jimu_cad_to_sim_correction", "none") if args is not None else "none").strip().lower()
    if mode in {"none", "auto", "off", "identity"}:
        return (0.0, 0.0, 0.0)
    raw = str(getattr(args, "jimu_cad_to_sim_local_rpy_deg", "") if args is not None else "").strip()
    if raw:
        values = []
        for part in raw.replace(",", " ").split():
            values.append(float(part))
        if len(values) != 3:
            raise ValueError("--jimu-cad-to-sim-local-rpy-deg must contain exactly 3 values")
        return tuple(float(v) for v in values)
    if mode in {"red_jimu_cube", "red_bricks_cube"}:
        return DEFAULT_JIMU_CAD_TO_SIM_RPY_DEG
    return (0.0, 0.0, 0.0)


def _jimu_cad_to_sim_local_fix(args: argparse.Namespace | None = None) -> np.ndarray:
    rpy = np.asarray(_jimu_cad_to_sim_rpy_deg(args), dtype=np.float32)
    fix = np.eye(4, dtype=np.float32)
    if np.any(np.abs(rpy) > 1e-6):
        fix[:3, :3] = euler2mat(*np.deg2rad(rpy), axes="sxyz").astype(np.float32)
    return fix


def install_jimu_object_specs(args: argparse.Namespace | None = None) -> None:
    base_spec = object_specs.get_object_spec(JIMU_PROVIDER_OBJECT_NAME)
    if base_spec is None:
        raise RuntimeError(f"Missing base object spec: {JIMU_PROVIDER_OBJECT_NAME}")
    role_base_spec = base_spec
    sim_asset_file = _jimu_sim_asset_file_override(args)
    if sim_asset_file:
        role_base_spec = replace(
            role_base_spec,
            sim_asset_file=sim_asset_file,
            sim_asset_scale=1.0,
        )
    local_rotation_offset = _jimu_cad_to_sim_rpy_deg(args)
    for role in JIMU_SCENE_ROLES:
        object_specs.OBJECT_SPECS[role] = replace(
            role_base_spec,
            name=role,
            grounding_prompt="small square plastic building block.",
            foundationpose_local_rotation_offset_deg=local_rotation_offset,
        )
        object_specs.OBJECT_NAME_ALIASES[role] = role


def install_jimu_place_rules(args: argparse.Namespace | None = None) -> None:
    hover_height = float(getattr(args, "jimu_wall_hover_height", 0.08) if args is not None else 0.08)
    release_retreat_height = float(
        getattr(args, "jimu_wall_release_retreat_height", 0.08) if args is not None else 0.08
    )
    for role, local_pose in _jimu_wall_local_pose_specs(args).items():
        place_rules.PLACE_RULES[role] = place_rules.PlaceRule(
            source_object_name=role,
            target_object_name=JIMU_FLOOR_ROLE,
            primitive="jimu_relative_pose",
            hover_height=hover_height,
            release_retreat_height=release_retreat_height,
            preserve_long_axis_vertical=True,
            object_pose_local=local_pose,
        )
    second_hover_height = float(getattr(args, "jimu_second_layer_hover_height", hover_height) if args is not None else hover_height)
    second_release_retreat_height = float(
        getattr(args, "jimu_second_layer_release_retreat_height", release_retreat_height)
        if args is not None
        else release_retreat_height
    )
    for role, local_pose in _jimu_second_layer_local_pose_specs(args).items():
        place_rules.PLACE_RULES[role] = place_rules.PlaceRule(
            source_object_name=role,
            target_object_name=_JIMU_SECOND_LAYER_PARENT[role],
            primitive="jimu_relative_pose",
            hover_height=second_hover_height,
            release_retreat_height=second_release_retreat_height,
            preserve_long_axis_vertical=True,
            object_pose_local=local_pose,
        )


def install_jimu_runtime_config(args: argparse.Namespace | None = None) -> None:
    install_jimu_object_specs(args)
    install_jimu_place_rules(args)


def _map_cam_pose_to_base_for_assignment(args: argparse.Namespace, T_base_cam: np.ndarray, T_cam_obj: np.ndarray) -> np.ndarray:
    T_base_cam = np.asarray(T_base_cam, dtype=np.float32).reshape(4, 4)
    T_cam_obj = np.asarray(T_cam_obj, dtype=np.float32).reshape(4, 4)
    T_cam_obj = (T_cam_obj @ _jimu_cad_to_sim_local_fix(args)).astype(np.float32)
    if bool(getattr(args, "use_direct_camera_extrinsic", False)):
        return (T_base_cam @ T_cam_obj).astype(np.float32)
    return (np.linalg.inv(T_base_cam).astype(np.float32) @ T_cam_obj).astype(np.float32)


def _candidate_entries(summary: dict, args: argparse.Namespace, T_base_cam: np.ndarray, provider_name: str) -> list[dict]:
    provider_name = direct.curobo_wrapper.normalize_object_name(provider_name) or provider_name
    entries: list[dict] = []
    for fallback_index, item in enumerate(list(summary.get("results") or [])):
        if not isinstance(item, dict) or not bool(item.get("ok", True)) or item.get("T_cam_obj") is None:
            continue
        item_name = direct.curobo_wrapper.normalize_object_name(item.get("object_name"))
        if item_name != provider_name:
            continue
        T_cam_obj = np.asarray(item["T_cam_obj"], dtype=np.float32).reshape(4, 4)
        T_base_obj = _map_cam_pose_to_base_for_assignment(args, T_base_cam, T_cam_obj)
        try:
            sam3_instance_index = int(item.get("sam3_instance_index", fallback_index))
        except Exception:
            sam3_instance_index = int(fallback_index)
        entries.append(
            {
                "fallback_index": int(fallback_index),
                "sam3_instance_index": int(sam3_instance_index),
                "item": item,
                "T_base_obj": T_base_obj,
            }
        )
    return entries


def _find_candidate_by_instance(entries: list[dict], requested_index: int, used_ids: set[int]) -> dict:
    for entry in entries:
        if id(entry) in used_ids:
            continue
        if int(entry["sam3_instance_index"]) == int(requested_index):
            return entry
    for entry in entries:
        if id(entry) in used_ids:
            continue
        if int(entry["fallback_index"]) == int(requested_index):
            return entry
    raise RuntimeError(f"SAM6D result instance {requested_index} is not available")


def _select_floor_candidate(entries: list[dict], args: argparse.Namespace, used_ids: set[int]) -> tuple[dict, dict]:
    available = [entry for entry in entries if id(entry) not in used_ids]
    if not available:
        raise RuntimeError("No SAM6D candidates remain for floor assignment")
    threshold = float(getattr(args, "jimu_floor_normal_threshold", 0.65))
    annotated = []
    for entry in available:
        T_base_obj = np.asarray(entry["T_base_obj"], dtype=np.float32).reshape(4, 4)
        local_y_dot_z = abs(float(T_base_obj[2, 1]))
        annotated.append(
            {
                "entry": entry,
                "local_y_abs_dot_base_z": local_y_dot_z,
                "base_z": float(T_base_obj[2, 3]),
                "score": float(entry["item"].get("score", 0.0)),
            }
        )
    flat = [item for item in annotated if item["local_y_abs_dot_base_z"] >= threshold]
    if flat:
        chosen_info = max(flat, key=lambda item: (item["local_y_abs_dot_base_z"], item["score"]))
        method = "local_y_normal_z"
    else:
        chosen_info = min(annotated, key=lambda item: item["base_z"])
        method = "lowest_center_z"
    debug = {
        "method": method,
        "normal_threshold": threshold,
        "candidates": [
            {
                "fallback_index": int(item["entry"]["fallback_index"]),
                "sam3_instance_index": int(item["entry"]["sam3_instance_index"]),
                "local_y_abs_dot_base_z": float(item["local_y_abs_dot_base_z"]),
                "base_z": float(item["base_z"]),
                "score": float(item["score"]),
            }
            for item in annotated
        ],
    }
    return chosen_info["entry"], debug


def _assign_jimu_roles(
    args: argparse.Namespace,
    entries: list[dict],
    role_names: list[str],
) -> tuple[dict[str, dict], dict]:
    if len(entries) < len(role_names):
        raise RuntimeError(
            f"SAM6D produced only {len(entries)} Jimu block pose(s), but {len(role_names)} role(s) are required: {role_names}"
        )

    role_set = set(role_names)
    explicit_map = _parse_role_instance_map(getattr(args, "jimu_role_instance_map", None))
    unknown_roles = sorted(set(explicit_map) - role_set)
    if unknown_roles:
        raise ValueError(f"--jimu-role-instance-map contains role(s) not in this scene: {unknown_roles}")

    used_ids: set[int] = set()
    assigned: dict[str, dict] = {}
    for role, requested_index in explicit_map.items():
        entry = _find_candidate_by_instance(entries, requested_index, used_ids)
        assigned[role] = entry
        used_ids.add(id(entry))

    if JIMU_FLOOR_ROLE in role_set and JIMU_FLOOR_ROLE not in assigned:
        floor_entry, floor_debug = _select_floor_candidate(entries, args, used_ids)
        assigned[JIMU_FLOOR_ROLE] = floor_entry
        used_ids.add(id(floor_entry))
    else:
        floor_debug = {"method": "explicit_instance_map"}

    for role in role_names:
        if role in assigned:
            continue
        for entry in entries:
            if id(entry) in used_ids:
                continue
            assigned[role] = entry
            used_ids.add(id(entry))
            break
        if role not in assigned:
            raise RuntimeError(f"No SAM6D candidate remains for role {role}")

    debug = {
        "floor_selection": floor_debug,
        "explicit_instance_map": explicit_map,
        "assignments": {
            role: {
                "fallback_index": int(entry["fallback_index"]),
                "sam3_instance_index": int(entry["sam3_instance_index"]),
                "score": float(entry["item"].get("score", 0.0)),
                "base_translation": np.asarray(entry["T_base_obj"], dtype=np.float32)[:3, 3].astype(float).tolist(),
                "axis_dot_base_z": np.asarray(entry["T_base_obj"], dtype=np.float32)[2, :3].astype(float).tolist(),
                "abs_axis_dot_base_z": np.abs(np.asarray(entry["T_base_obj"], dtype=np.float32)[2, :3]).astype(float).tolist(),
            }
            for role, entry in assigned.items()
        },
    }
    return assigned, debug


def _ensure_floor_local_y_points_up(item: dict, args: argparse.Namespace, T_base_cam: np.ndarray) -> dict:
    corrected = copy.deepcopy(item)
    T_cam_obj = np.asarray(corrected["T_cam_obj"], dtype=np.float32).reshape(4, 4)
    T_base_obj = _map_cam_pose_to_base_for_assignment(args, T_base_cam, T_cam_obj)
    local_y_dot_up = float(T_base_obj[2, 1])
    corrected["jimu_floor_local_y_dot_base_z_before"] = local_y_dot_up
    if local_y_dot_up < 0.0:
        flip = np.eye(4, dtype=np.float32)
        flip[1, 1] = -1.0
        flip[2, 2] = -1.0
        T_cam_obj = (T_cam_obj @ flip).astype(np.float32)
        corrected["T_cam_obj"] = T_cam_obj
        corrected["jimu_floor_local_y_flip_applied"] = True
        T_base_obj_after = _map_cam_pose_to_base_for_assignment(args, T_base_cam, T_cam_obj)
        corrected["jimu_floor_local_y_dot_base_z_after"] = float(T_base_obj_after[2, 1])
        print(
            "[jimu-sam6d] flipped floor local frame by 180deg around local X "
            f"so floor +Y points upward ({local_y_dot_up:.3f} -> {float(T_base_obj_after[2, 1]):.3f})"
        )
    else:
        corrected["jimu_floor_local_y_flip_applied"] = False
        corrected["jimu_floor_local_y_dot_base_z_after"] = local_y_dot_up
    return corrected


def _print_assignment(debug: dict) -> None:
    print("[jimu-sam6d] role assignment:")
    for role in list((debug.get("assignments") or {}).keys()):
        item = (debug.get("assignments") or {}).get(role)
        if not item:
            continue
        print(
            "  "
            f"{role} <- result_index={item['fallback_index']} "
            f"sam3_instance={item['sam3_instance_index']} "
            f"score={item['score']:.3f} "
            f"base_t={np.round(np.asarray(item['base_translation'], dtype=np.float32), 4).tolist()} "
            f"axis_dot_base_z(x,y,z)="
            f"{np.round(np.asarray(item['axis_dot_base_z'], dtype=np.float32), 3).tolist()}"
        )


def _jimu_cache_key(args: argparse.Namespace, role_names: list[str], provider_names: list[str]) -> tuple:
    return (
        "jimu_layered_wall_sam6d",
        tuple(role_names),
        tuple(provider_names),
        direct_sam6d._sam6d_cache_key(args, provider_names),
        str(getattr(args, "jimu_assembly_sam6d_provider_script", "") or assembly_sam6d.DEFAULT_SAM6D_PROVIDER_SCRIPT),
    )


def _run_jimu_assembly_sam6d_provider(args: argparse.Namespace, provider_names: list[str]) -> tuple[dict[str, Any], Path]:
    provider_script = str(getattr(args, "jimu_assembly_sam6d_provider_script", "") or assembly_sam6d.DEFAULT_SAM6D_PROVIDER_SCRIPT)
    run_args = argparse.Namespace(
        sam6d_python=sys.executable,
        sam6d_provider_script=str(Path(provider_script).expanduser()),
        sam6d_output_root=str(Path(getattr(args, "sam6d_output_root", BETA_DIR / "sam6d_jimu_direct_runs")).expanduser()),
        sam6d_mask_mode=str(getattr(args, "sam6d_mask_mode", "sam3_text")),
        sam6d_root=str(Path(getattr(args, "sam6d_root", assembly_sam6d.DEFAULT_SAM6D_ROOT)).expanduser()),
        sam6d_camera_width=int(getattr(args, "camera_width", 640)),
        sam6d_camera_height=int(getattr(args, "camera_height", 480)),
        sam6d_camera_fps=int(getattr(args, "camera_fps", 30)),
        sam6d_warmup_frames=int(getattr(args, "warmup_frames", 30)),
        sam6d_sam3_full_scene_keep_multi_instances=True,
        sam3_max_masks_per_item=max(int(getattr(args, "sam3_max_masks_per_item", 1) or 1), len(provider_names)),
        sam3_confidence_threshold=float(getattr(args, "sam3_confidence_threshold", 0.20)),
        sam6d_frame_dir=str(getattr(args, "sam6d_frame_dir", "") or ""),
        sam3_instance_index=int(getattr(args, "sam3_instance_index", 0)),
        sam6d_confirm_full_scene_masks=bool(getattr(args, "sam6d_confirm_segmentation", True)),
        sam6d_require_full_scene_masks=bool(getattr(args, "sam6d_require_full_scene_masks", True)),
        sam6d_show_full_scene_mask_window=bool(getattr(args, "sam6d_show_segmentation_window", True)),
        sam6d_skip_pem=False,
        sam6d_sam3_full_scene_result_json=str(getattr(args, "sam3_full_scene_result_json", "") or ""),
        sam6d_pem_warmup_during_sam3=False,
        sam6d_provider_timeout_s=float(getattr(args, "sam6d_provider_timeout_s", 240.0) or 240.0),
    )
    print("[jimu-sam6d] using Beta assembly SAM3/SAM6D subprocess call")
    return assembly_sam6d._run_sam6d_provider(run_args, provider_names)


def capture_or_reuse_jimu_sam6d_scene(args, bridge_mod, scene_capture_cache=None):
    target_name = direct.curobo_wrapper.normalize_object_name(getattr(args, "object_name", None))
    if target_name is None:
        raise ValueError("--object-name is required for Jimu SAM6D scene capture")

    role_names = _split_names(getattr(args, "jimu_scene_roles", None)) or list(JIMU_SCENE_ROLES)
    if target_name not in role_names:
        role_names.append(target_name)
    role_names = [name for name in role_names if name in set(JIMU_SCENE_ROLES)]
    if JIMU_FLOOR_ROLE not in role_names:
        role_names.insert(0, JIMU_FLOOR_ROLE)
    build_roles = _split_names(getattr(args, "cycle_object_names", None)) or _default_pick_roles_for_layers(
        getattr(args, "jimu_build_layers", "two")
    )
    for role in build_roles:
        if role not in role_names:
            role_names.append(role)

    selected_obstacles = direct_sam6d._selected_obstacle_names(args, target_name)
    required_names = list(role_names)
    required_obstacles = [name for name in required_names if name != target_name]
    provider_name = direct.curobo_wrapper.normalize_object_name(
        getattr(args, "jimu_provider_object_name", JIMU_PROVIDER_OBJECT_NAME)
    ) or JIMU_PROVIDER_OBJECT_NAME
    provider_names = [provider_name] * len(role_names)
    cache_key = _jimu_cache_key(args, role_names, provider_names)

    if (
        bool(getattr(args, "sam6d_reuse_scene_across_cycles", True))
        and isinstance(scene_capture_cache, dict)
        and scene_capture_cache.get("key") == cache_key
    ):
        cached_objects = dict(scene_capture_cache.get("objects", {}) or {})
        if target_name in cached_objects and all(name in cached_objects for name in required_obstacles):
            print("[jimu-sam6d] reusing cached Jimu SAM6D scene")
            return (
                None,
                np.asarray(scene_capture_cache["T_base_cam"], dtype=np.float32),
                np.asarray(cached_objects[target_name]["T_cam_obj"], dtype=np.float32).reshape(4, 4),
                direct_sam6d._scene_obstacle_entries_from_cache(args, cached_objects, selected_obstacles),
            )

    if (
        bool(getattr(args, "_planning_prefetch_capture_only", False))
        and not bool(getattr(args, "sam6d_allow_prefetch_scene_recapture", False))
    ):
        cached_objects = {}
        if isinstance(scene_capture_cache, dict):
            cached_objects = dict(scene_capture_cache.get("objects", {}) or {})
        missing_required = [name for name in required_names if name not in cached_objects]
        raise RuntimeError(
            "Jimu SAM6D prefetch scene cache miss; refusing to recapture in the background. "
            f"target={target_name}, missing_required={missing_required}, cached={sorted(cached_objects.keys())}"
        )

    summary, result_path = direct_sam6d._load_fixed_sam6d_summary(args)
    if summary is None:
        with direct._CUROBO_GPU_LOCK:
            summary, result_path = _run_jimu_assembly_sam6d_provider(args, provider_names)

    T_base_cam = bridge_mod.load_matrix(args.camera_extrinsic_opencv_path).astype(np.float32)
    entries = _candidate_entries(summary, args, T_base_cam, provider_name)
    assigned, assignment_debug = _assign_jimu_roles(args, entries, role_names)
    if bool(getattr(args, "jimu_print_role_assignment", True)):
        _print_assignment(assignment_debug)

    cached_objects = {}
    canonical_base_poses: dict[str, np.ndarray] = {}
    for role in role_names:
        entry = assigned[role]
        item = copy.deepcopy(entry["item"])
        floor_T = canonical_base_poses.get(JIMU_FLOOR_ROLE)
        item, canonical_T = _canonicalize_jimu_result_item(
            item,
            role,
            args,
            T_base_cam,
            floor_T_base_obj=floor_T,
        )
        canonical_base_poses[role] = canonical_T
        item["jimu_role"] = role
        item["jimu_provider_object_name"] = provider_name
        item["jimu_provider_result_index"] = int(entry["fallback_index"])
        item["sam3_instance_index"] = int(entry["sam3_instance_index"])
        cached_objects[role] = direct_sam6d._cache_entry_from_sam6d_result(args, bridge_mod, role, item)
        cached_objects[role]["jimu_provider_object_name"] = provider_name
        cached_objects[role]["jimu_provider_result_index"] = int(entry["fallback_index"])

    missing_required = [name for name in required_names if name not in cached_objects]
    if missing_required and bool(getattr(args, "sam6d_strict_scene", True)):
        raise RuntimeError(f"Jimu SAM6D strict scene is enabled and required roles are missing: {missing_required}")

    if isinstance(scene_capture_cache, dict):
        scene_capture_cache.clear()
        scene_capture_cache.update(
            {
                "key": cache_key,
                "fp_rt": None,
                "T_base_cam": T_base_cam.copy(),
                "objects": cached_objects,
                "sam6d_summary_path": str(result_path),
                "sam6d_scene_dir": str(summary.get("scene_dir", Path(result_path).parent)),
                "jimu_assignment": assignment_debug,
            }
        )

    print(
        f"[jimu-sam6d] using target {target_name}: "
        f"camera translation={np.round(cached_objects[target_name]['T_cam_obj'][:3, 3], 6).tolist()}"
    )
    return (
        None,
        T_base_cam,
        cached_objects[target_name]["T_cam_obj"],
        direct_sam6d._scene_obstacle_entries_from_cache(args, cached_objects, selected_obstacles),
    )


def relocalize_active_target_after_empty_grasp_jimu(demo, bridge_mod, args, scene_capture_cache=None) -> bool:
    target_name = direct.curobo_wrapper.normalize_object_name(getattr(args, "object_name", None))
    if target_name is None:
        return False
    relocalize_args = copy.copy(args)
    relocalize_args.sam6d_reuse_scene_across_cycles = False
    try:
        _, T_base_cam, T_cam_obj, _ = capture_or_reuse_jimu_sam6d_scene(
            relocalize_args,
            bridge_mod,
            scene_capture_cache=scene_capture_cache,
        )
        T_world_obj = bridge_mod.map_camera_pose_to_pick_world(T_cam_obj, T_base_cam, demo.env, args)
        bridge_mod.apply_pose_to_pick_object(demo.env, T_world_obj)
        direct._update_relocalized_target_scene_cache(demo, scene_capture_cache, target_name, T_cam_obj, T_world_obj, args)
        print(f"[jimu-sam6d] relocalized active target after empty grasp: {target_name}")
        return True
    except Exception as exc:
        print(f"[jimu-sam6d] relocalize after empty grasp failed for {target_name}: {exc}")
        return False


def _jimu_second_layer_target_pose_from_floor(demo, bridge_mod, scene_capture_cache, source_name: str, args) -> np.ndarray | None:
    parent_name = _JIMU_SECOND_LAYER_PARENT.get(source_name)
    if parent_name is None:
        return None
    T_world_floor = direct.targeted._get_scene_object_world_transform(
        demo,
        bridge_mod,
        scene_capture_cache,
        JIMU_FLOOR_ROLE,
    )
    if T_world_floor is None:
        return None
    parent_specs = _jimu_wall_local_pose_specs(args)
    parent_spec = parent_specs.get(parent_name)
    if parent_spec is None:
        return None
    T_floor_parent_target = direct.targeted._local_pose_spec_to_matrix(parent_spec)
    T_world_parent_target = (
        np.asarray(T_world_floor, dtype=np.float32).reshape(4, 4)
        @ np.asarray(T_floor_parent_target, dtype=np.float32).reshape(4, 4)
    ).astype(np.float32)
    extents = _load_scaled_jimu_extents(args)
    z_extra = float(getattr(args, "jimu_second_layer_z_extra", 0.0) or 0.0)
    T_world_second_target = T_world_parent_target.copy()
    T_world_second_target[:3, 3] = (
        T_world_parent_target[:3, 3] + np.asarray([0.0, 0.0, float(extents[2]) + z_extra], dtype=np.float32)
    ).astype(np.float32)
    return T_world_second_target


def _jimu_rebuild_place_plan_for_target(plan, T_world_obj_target: np.ndarray):
    T_world_obj_target = np.asarray(T_world_obj_target, dtype=np.float32).reshape(4, 4)
    T_world_obj_old = getattr(plan, "T_world_obj_desired", None)
    if T_world_obj_old is None:
        return plan
    try:
        T_world_obj_old = np.asarray(T_world_obj_old, dtype=np.float32).reshape(4, 4)
        T_world_tcp_old = direct._pose_to_matrix_from_pose_obj(plan.place_pose).astype(np.float32)
        T_tcp_obj = (np.linalg.inv(T_world_tcp_old).astype(np.float32) @ T_world_obj_old).astype(np.float32)
        T_world_tcp_new = (T_world_obj_target @ np.linalg.inv(T_tcp_obj).astype(np.float32)).astype(np.float32)
        place_pose = direct._pose_from_world_matrix(T_world_tcp_new)
        old_place_p = direct.targeted.base.flatten_np(plan.place_pose.p)[:3].astype(np.float32)
        new_place_p = T_world_tcp_new[:3, 3].astype(np.float32)

        def shifted_pose(old_pose):
            if old_pose is None:
                return None
            old_p = direct.targeted.base.flatten_np(old_pose.p)[:3].astype(np.float32)
            return direct.targeted.base.make_pose_with_position(place_pose, new_place_p + (old_p - old_place_p))

        return direct.targeted.TargetedPlacePlan(
            rule=plan.rule,
            target_name=plan.target_name,
            slot_name=plan.slot_name,
            variant_label=plan.variant_label,
            T_world_obj_desired=T_world_obj_target,
            staging_pose=shifted_pose(plan.staging_pose),
            pre_place_pose=shifted_pose(plan.pre_place_pose),
            place_pose=place_pose,
            retreat_pose=shifted_pose(plan.retreat_pose),
            tcp_verticality=float(getattr(plan, "tcp_verticality", 0.0)),
        )
    except Exception as exc:
        print(f"[jimu place] failed to rebuild floor-anchored second-layer plan: {exc}")
        return plan


def _jimu_floor_anchor_second_layer_plans(plans, demo, bridge_mod, scene_capture_cache, source_name: str | None, args) -> list:
    if source_name not in set(JIMU_SECOND_LAYER_ROLES):
        return list(plans or [])
    T_target = _jimu_second_layer_target_pose_from_floor(demo, bridge_mod, scene_capture_cache, source_name, args)
    if T_target is None:
        return list(plans or [])
    anchored = [_jimu_rebuild_place_plan_for_target(plan, T_target) for plan in list(plans or [])]
    print(
        f"[jimu place] {source_name}: second-layer target anchored from floor/first-layer goal, "
        f"target_xyz={np.round(T_target[:3, 3], 6).tolist()}"
    )
    return anchored


def build_targeted_place_plan_variants_jimu(
    demo,
    bridge_mod,
    scene_capture_cache,
    rule,
    place_state_cache,
    args,
    *,
    T_tcp_obj_override: np.ndarray | None = None,
):
    original = _ORIGINAL_TARGETED_BUILD_PLACE_PLAN_VARIANTS
    if original is None:
        return []
    plans = list(
        original(
            demo,
            bridge_mod,
            scene_capture_cache,
            rule,
            place_state_cache,
            args,
            T_tcp_obj_override=T_tcp_obj_override,
        )
        or []
    )
    source_name = direct.curobo_wrapper.normalize_object_name(getattr(rule, "source_object_name", None))
    plans = _jimu_floor_anchor_second_layer_plans(plans, demo, bridge_mod, scene_capture_cache, source_name, args)
    if (
        source_name not in set(JIMU_PICK_ROLES)
        or bool(getattr(args, "jimu_parallel_grasp_place", True))
        or not bool(getattr(args, "jimu_place_symmetry_enabled", False))
    ):
        return plans
    degrees = [deg for deg in _jimu_symmetry_degrees(args) if abs(float(deg)) > 1e-6]
    if not degrees:
        return plans

    augmented = list(plans)
    for plan in plans:
        T_world_obj = getattr(plan, "T_world_obj_desired", None)
        if T_world_obj is None:
            continue
        T_world_obj = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4)
        T_world_tcp = direct._pose_to_matrix_from_pose_obj(plan.place_pose).astype(np.float32)
        T_tcp_obj = (np.linalg.inv(T_world_tcp).astype(np.float32) @ T_world_obj).astype(np.float32)
        inv_T_tcp_obj = np.linalg.inv(T_tcp_obj).astype(np.float32)
        place_p = direct.targeted.base.flatten_np(plan.place_pose.p)[:3].astype(np.float32)
        pre_p = direct.targeted.base.flatten_np(plan.pre_place_pose.p)[:3].astype(np.float32)
        retreat_p = direct.targeted.base.flatten_np(plan.retreat_pose.p)[:3].astype(np.float32)
        pre_delta = pre_p - place_p
        retreat_delta = retreat_p - place_p
        base_label = str(plan.variant_label or "")
        for deg in degrees:
            R_local = np.eye(4, dtype=np.float32)
            R_local[:3, :3] = euler2mat(0.0, np.deg2rad(float(deg)), 0.0, axes="sxyz").astype(np.float32)
            T_world_obj_sym = (T_world_obj @ R_local).astype(np.float32)
            T_world_tcp_sym = (T_world_obj_sym @ inv_T_tcp_obj).astype(np.float32)
            place_pose = direct._pose_from_world_matrix(T_world_tcp_sym)
            sym_place_p = T_world_tcp_sym[:3, 3].astype(np.float32)
            pre_place_pose = direct.targeted.base.make_pose_with_position(place_pose, sym_place_p + pre_delta)
            retreat_pose = direct.targeted.base.make_pose_with_position(place_pose, sym_place_p + retreat_delta)
            sym_label = f"yaw_{int(round(float(deg)))}deg"
            labels = [label for label in (base_label, sym_label) if label]
            augmented.append(
                direct.targeted.TargetedPlacePlan(
                    rule=plan.rule,
                    target_name=plan.target_name,
                    slot_name=plan.slot_name,
                    variant_label="+".join(labels) if labels else sym_label,
                    T_world_obj_desired=T_world_obj_sym,
                    staging_pose=None,
                    pre_place_pose=pre_place_pose,
                    place_pose=place_pose,
                    retreat_pose=retreat_pose,
                    tcp_verticality=float(getattr(plan, "tcp_verticality", 0.0)),
                )
            )
    if len(augmented) != len(plans):
        print(
            f"[jimu place] added {len(augmented) - len(plans)} symmetry place plan(s) "
            f"for {source_name}: degrees={np.round(np.asarray(_jimu_symmetry_degrees(args), dtype=np.float32), 1).tolist()}"
        )
    return augmented


def _select_jimu_symmetry_place_candidates(place_candidates, args) -> list[dict]:
    candidates = sorted([dict(item) for item in list(place_candidates or [])], key=direct._pre_place_screen_sort_key)
    if not candidates:
        return []
    target_degs = _jimu_symmetry_degrees(args)
    max_per_grasp = int(getattr(args, "jimu_fast_chain_symmetry_expand_max_per_grasp", 12) or 0)
    selected: list[dict] = []
    seen_keys: set[tuple] = set()
    for target_deg in target_degs:
        if max_per_grasp > 0 and len(selected) >= max_per_grasp:
            break
        matches = [
            item
            for item in candidates
            if direct._yaw_distance_deg(
                direct._variant_yaw_deg_from_labels(item.get("variant_label"), item.get("label")),
                float(target_deg),
            )
            <= 1.0
        ]
        for item in matches:
            key = (
                direct._pose_dedupe_key(item.get("pose")),
                direct._pose_dedupe_key(item.get("release_pose", item.get("place_pose"))),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out = dict(item)
            out["_fast_chain_yaw_expansion"] = True
            out["_fast_chain_yaw_expansion_token"] = direct._fast_chain_yaw_expansion_token(out, "gluestick")
            selected.append(out)
            break
    return selected


def _pose_position_or_none(pose) -> np.ndarray | None:
    if pose is None:
        return None
    try:
        return direct.targeted.base.flatten_np(pose.p)[:3].astype(np.float32)
    except Exception:
        return None


def _jimu_release_object_center_from_candidate(place_candidate: dict, T_tcp_obj: np.ndarray) -> np.ndarray | None:
    T_world_obj = _jimu_target_object_pose_from_candidate(place_candidate, T_tcp_obj)
    if T_world_obj is not None:
        return T_world_obj[:3, 3].astype(np.float32)
    return None


def _jimu_target_object_pose_from_candidate(place_candidate: dict, T_tcp_obj: np.ndarray) -> np.ndarray | None:
    T_world_obj = place_candidate.get("T_world_obj_desired")
    if T_world_obj is None:
        T_world_obj = getattr(place_candidate.get("place_plan", None), "T_world_obj_desired", None)
    if T_world_obj is not None:
        try:
            return np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4).astype(np.float32)
        except Exception:
            return None
    release_src = place_candidate.get("release_pose", place_candidate.get("place_pose", place_candidate.get("pose")))
    if release_src is not None:
        try:
            T_world_tcp = direct._pose_to_matrix_from_pose_obj(release_src).astype(np.float32)
            return (T_world_tcp @ T_tcp_obj).astype(np.float32)
        except Exception:
            pass
    return None


def _world_z_rotation(deg: float) -> np.ndarray:
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


def _jimu_world_z_yaw_delta_deg(T_world_obj_from: np.ndarray, T_world_obj_to: np.ndarray) -> float:
    R_from = np.asarray(T_world_obj_from, dtype=np.float32).reshape(4, 4)[:3, :3]
    R_to = np.asarray(T_world_obj_to, dtype=np.float32).reshape(4, 4)[:3, :3]
    dot_sum = 0.0
    cross_sum = 0.0
    for axis_idx in range(3):
        a = R_from[:2, axis_idx].astype(np.float64)
        b = R_to[:2, axis_idx].astype(np.float64)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 1e-5 or nb <= 1e-5:
            continue
        a /= na
        b /= nb
        weight = na * nb
        dot_sum += weight * float(a[0] * b[0] + a[1] * b[1])
        cross_sum += weight * float(a[0] * b[1] - a[1] * b[0])
    if abs(dot_sum) <= 1e-9 and abs(cross_sum) <= 1e-9:
        return 0.0
    yaw = float(np.rad2deg(np.arctan2(cross_sum, dot_sum)))
    return ((yaw + 180.0) % 360.0) - 180.0


def _snap_yaw_deg(yaw_deg: float, step_deg: float) -> float:
    step = float(step_deg)
    if step <= 1e-6:
        return float(yaw_deg)
    snapped = round(float(yaw_deg) / step) * step
    return ((snapped + 180.0) % 360.0) - 180.0


def _make_jimu_parallel_grasp_place_candidate(
    grasp_candidate: dict,
    place_candidate: dict,
    args,
    rule,
) -> dict | None:
    """Keep TCP->object fixed, then rotate the carried object about world Z into the target yaw."""
    try:
        T_tcp_obj = np.asarray(grasp_candidate.get("T_tcp_obj"), dtype=np.float32).reshape(4, 4)
    except Exception:
        return None
    grasp_pose = grasp_candidate.get("pose", grasp_candidate.get("grasp_pose", grasp_candidate.get("pregrasp_pose")))
    if grasp_pose is None:
        return None
    try:
        T_world_tcp_grasp = direct._pose_to_matrix_from_pose_obj(grasp_pose).astype(np.float32)
    except Exception:
        return None
    target_center = _jimu_release_object_center_from_candidate(place_candidate, T_tcp_obj)
    if target_center is None or not np.all(np.isfinite(target_center)):
        return None
    T_world_obj_grasp = (T_world_tcp_grasp @ T_tcp_obj).astype(np.float32)
    T_world_obj_target = _jimu_target_object_pose_from_candidate(place_candidate, T_tcp_obj)
    if T_world_obj_target is None:
        T_world_obj_target = T_world_obj_grasp.copy()
        T_world_obj_target[:3, 3] = target_center.astype(np.float32)
    yaw_delta_deg = _jimu_world_z_yaw_delta_deg(T_world_obj_grasp, T_world_obj_target)
    if bool(getattr(args, "jimu_parallel_grasp_place_snap_yaw_90", True)):
        yaw_delta_deg = _snap_yaw_deg(yaw_delta_deg, 90.0)
    R_z = _world_z_rotation(yaw_delta_deg)

    T_world_tcp_release = np.eye(4, dtype=np.float32)
    T_world_tcp_release[:3, :3] = (R_z @ T_world_tcp_grasp[:3, :3]).astype(np.float32)
    T_world_tcp_release[:3, 3] = (
        target_center - T_world_tcp_release[:3, :3] @ T_tcp_obj[:3, 3].astype(np.float32)
    ).astype(np.float32)
    release_pose = direct._pose_from_world_matrix(T_world_tcp_release)

    place_mode = str(place_candidate.get("place_mode", "vertical_place") or "vertical_place")
    try:
        hover_pose = direct.build_hover_pose(
            release_pose,
            place_mode,
            args,
            rule,
            candidate_pre_place_pose=None,
        )
    except Exception:
        hover_height = float(max(getattr(args, "vertical_place_hover_height_m", 0.08), 0.0))
        hover_pose = direct.targeted.base.make_pose_with_position(
            release_pose,
            (T_world_tcp_release[:3, 3] + np.asarray([0.0, 0.0, hover_height], dtype=np.float32)).astype(np.float32),
        )
    hover_extra = float(max(place_candidate.get("hover_extra_height_m", 0.0) or 0.0, 0.0))
    if hover_extra > 1e-6:
        try:
            hover_pose = direct._extend_hover_pose_along_release_approach(release_pose, hover_pose, hover_extra)
        except Exception:
            pass

    predicted_obj = (T_world_tcp_release @ T_tcp_obj).astype(np.float32)
    item = dict(place_candidate)
    base_label = str(item.get("label", "transport_hover") or "transport_hover")
    base_variant = str(item.get("variant_label", "") or "")
    yaw_label = f"z_yaw_{float(yaw_delta_deg):+.1f}"
    item["label"] = f"{base_label}_parallel_grasp_{yaw_label}"
    item["pose"] = hover_pose
    item["hover_pose"] = hover_pose
    item["pre_place_pose"] = hover_pose
    item["place_pose"] = release_pose
    item["release_pose"] = release_pose
    item["raw_release_pose"] = release_pose
    item["retreat_pose"] = hover_pose
    item["variant_label"] = f"{base_variant}+parallel_grasp+{yaw_label}" if base_variant else f"parallel_grasp+{yaw_label}"
    item["T_world_obj_desired"] = predicted_obj
    item["jimu_parallel_grasp_place"] = True
    item["jimu_parallel_world_z_yaw_deg"] = float(yaw_delta_deg)
    item["jimu_parallel_target_center"] = target_center.astype(float).tolist()
    item["jimu_parallel_release_tcp_position"] = T_world_tcp_release[:3, 3].astype(float).tolist()
    item["jimu_parallel_source_place_label"] = base_label
    item["_fast_chain_yaw_expansion"] = True
    item["_fast_chain_yaw_expansion_token"] = "parallel_grasp"
    try:
        item["pad_tilt"] = float(direct._tcp_pad_tilt_z(release_pose))
    except Exception:
        pass
    for stale_key in (
        "result",
        "q_path",
        "q_goal",
        "q_hover",
        "q_release",
        "fast_chain_hover_q",
        "fast_chain_release_q",
        "fast_chain_score",
        "pair_score",
        "ik_pos_error",
        "ik_rot_error",
        "ik_score",
    ):
        item.pop(stale_key, None)
    return item


def _select_jimu_parallel_place_source_candidates(place_candidates, args) -> list[dict]:
    ordered = sorted([dict(item) for item in list(place_candidates or [])], key=direct._pre_place_screen_sort_key)
    if not ordered:
        return []
    try:
        limit = max(1, int(getattr(args, "jimu_parallel_grasp_place_max_sources_per_grasp", 1) or 1))
    except Exception:
        limit = 1
    selected: list[dict] = []
    seen: set[tuple] = set()
    for item in ordered:
        key = (
            direct._pose_dedupe_key(item.get("pose")),
            direct._pose_dedupe_key(item.get("release_pose", item.get("place_pose"))),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _jimu_tilt_abs_relation_label(label: str) -> str:
    text = str(label or "").lower()
    # Jimu plates are symmetric for the paired-relation screen: a +tilt and a
    # -tilt with the same magnitude should share the same hover/release bucket.
    text = re.sub(
        r"tilt_(?:toward|away)_robot_([+-]?\d+(?:\.\d+)?)deg",
        lambda m: f"tilt_abs_{abs(float(m.group(1))):.1f}deg",
        text,
    )
    text = re.sub(
        r"tilt_([+-]?\d+(?:\.\d+)?)deg",
        lambda m: f"tilt_abs_{abs(float(m.group(1))):.1f}deg",
        text,
    )
    text = re.sub(r"tilt_([+-]?\d+(?:\.\d+)?)(?=_(?:x|y|shift|lift|yaw|$))", lambda m: f"tilt_abs_{abs(float(m.group(1))):.1f}", text)
    return text


def fast_chain_relation_match_key_jimu(item: dict | None, source_name: str | None) -> tuple:
    source = direct.curobo_wrapper.normalize_object_name(source_name)
    if source not in set(JIMU_PICK_ROLES):
        original = _ORIGINAL_FAST_CHAIN_RELATION_MATCH_KEY
        if original is None:
            return ("label", "" if item is None else str(item.get("label", "") or ""))
        return original(item, source_name)
    label = "" if item is None else str(item.get("label", "") or "")
    return ("jimu_tilt_abs_label", _jimu_tilt_abs_relation_label(label))


def fast_chain_rank_paired_relation_candidates_jimu(
    planner,
    demo,
    bridge_mod,
    args,
    scene_capture_cache,
    place_state_cache,
    rule,
    grasp_relation_candidates,
    start_q,
    *,
    source_name: str | None,
    label: str,
    disabled_world_collision_links: list[str] | None,
):
    original = _ORIGINAL_FAST_CHAIN_RANK_PAIRED_RELATION_CANDIDATES
    source = direct.curobo_wrapper.normalize_object_name(source_name)
    if (
        source not in set(JIMU_PICK_ROLES)
        or not bool(getattr(args, "jimu_parallel_grasp_place", True))
    ):
        if original is None:
            return [], {"status": "NO_ORIGINAL_FAST_CHAIN_RANKER"}
        return original(
            planner,
            demo,
            bridge_mod,
            args,
            scene_capture_cache,
            place_state_cache,
            rule,
            grasp_relation_candidates,
            start_q,
            source_name=source_name,
            label=label,
            disabled_world_collision_links=disabled_world_collision_links,
        )

    relation_slots = direct._fast_chain_relation_ik_slots(args)
    records: list[dict] = []
    missing_place_count = 0
    missing_relation_count = 0
    skipped_grasp_chain_count = 0
    for grasp_candidate in list(grasp_relation_candidates or [])[:relation_slots]:
        if grasp_candidate.get("T_tcp_obj") is None:
            missing_relation_count += 1
            continue
        if not (
            bool(grasp_candidate.get("_winner_preselect_pregrasp_success", False))
            and bool(grasp_candidate.get("_winner_preselect_grasp_success", False))
            and bool(grasp_candidate.get("_winner_preselect_grasp_approach_q_path"))
        ):
            skipped_grasp_chain_count += 1
            continue
        place_candidates = direct._build_direct_place_candidates(
            demo,
            bridge_mod,
            scene_capture_cache,
            rule,
            place_state_cache,
            args,
            T_tcp_obj_override=grasp_candidate.get("T_tcp_obj"),
        )
        source_place_candidates = _select_jimu_parallel_place_source_candidates(place_candidates, args)
        if not source_place_candidates:
            missing_place_count += 1
            continue
        parallel_candidates = []
        for place_candidate in source_place_candidates:
            parallel_candidate = _make_jimu_parallel_grasp_place_candidate(
                grasp_candidate,
                place_candidate,
                args,
                rule,
            )
            if parallel_candidate is not None:
                parallel_candidates.append(parallel_candidate)
        if not parallel_candidates:
            missing_place_count += 1
            continue
        for place_candidate in parallel_candidates:
            records.append(
                {
                    "grasp_candidate": grasp_candidate,
                    "place_candidate": dict(place_candidate),
                    "place_candidates": [dict(item) for item in parallel_candidates],
                    "place_candidate_count": len(parallel_candidates),
                    "relation_token_suffix": "parallel_grasp",
                    "yaw_expansion": True,
                    "place_expansion_kind": "jimu_parallel_grasp",
                }
            )

    debug = {
        "status": "NO_PLACE_CANDIDATES",
        "relation_slots": int(relation_slots),
        "missing_place_count": int(missing_place_count),
        "missing_relation_count": int(missing_relation_count),
        "skipped_grasp_chain_count": int(skipped_grasp_chain_count),
        "paired_jimu_parallel_grasp_place": True,
        "paired_jimu_parallel_candidate_count": int(len(records)),
        "paired_place_expansion_kind": "jimu_parallel_grasp",
    }
    if not records:
        return [], debug

    print(
        f"[winner_chain] {source} jimu parallel-grasp place IK: "
        f"{len(records)} relation-place candidate(s); keeping grasp TCP rotation and translating to target center"
    )
    pair_records, paired_debug = direct._fast_chain_evaluate_paired_relation_records(
        planner,
        demo,
        args,
        records,
        start_q,
        source_name=source_name,
        label=f"{label}_jimu_parallel_grasp_place",
        disabled_world_collision_links=disabled_world_collision_links,
    )
    paired_debug = dict(paired_debug or {})
    paired_debug["relation_slots"] = int(relation_slots)
    paired_debug["missing_place_count"] = int(missing_place_count)
    paired_debug["missing_relation_count"] = int(missing_relation_count)
    paired_debug["skipped_grasp_chain_count"] = int(skipped_grasp_chain_count)
    paired_debug["paired_jimu_parallel_grasp_place"] = True
    paired_debug["paired_jimu_parallel_candidate_count"] = int(len(records))
    paired_debug["paired_place_expansion_kind"] = "jimu_parallel_grasp"
    return pair_records, paired_debug


def _jimu_motiongen_success(result) -> bool:
    return bool(getattr(result, "success", False)) and getattr(result, "joint_path", None) is not None


def _record_jimu_motiongen_result(planner, kind: str, result) -> None:
    if planner is None or result is None:
        return
    status = str(getattr(result, "status", "UNKNOWN") or "UNKNOWN")
    success = _jimu_motiongen_success(result)
    counts = getattr(planner, "_jimu_motiongen_status_counts", None)
    failure_counts = getattr(planner, "_jimu_motiongen_failure_status_counts", None)
    success_counts = getattr(planner, "_jimu_motiongen_success_status_counts", None)
    events = getattr(planner, "_jimu_motiongen_status_events", None)
    if not isinstance(counts, dict):
        counts = {}
        planner._jimu_motiongen_status_counts = counts
    if not isinstance(failure_counts, dict):
        failure_counts = {}
        planner._jimu_motiongen_failure_status_counts = failure_counts
    if not isinstance(success_counts, dict):
        success_counts = {}
        planner._jimu_motiongen_success_status_counts = success_counts
    if not isinstance(events, list):
        events = []
        planner._jimu_motiongen_status_events = events
    counts[status] = int(counts.get(status, 0)) + 1
    if success:
        success_counts[status] = int(success_counts.get(status, 0)) + 1
    else:
        failure_counts[status] = int(failure_counts.get(status, 0)) + 1
    if len(events) < 32:
        events.append(
            {
                "kind": str(kind),
                "status": status,
                "success": bool(success),
                "has_path": bool(getattr(result, "joint_path", None) is not None),
            }
        )


def _record_jimu_motiongen_results(planner, kind: str, results) -> None:
    if isinstance(results, (list, tuple)):
        for result in results:
            _record_jimu_motiongen_result(planner, kind, result)
    else:
        _record_jimu_motiongen_result(planner, kind, results)


def _jimu_start_collision_diagnosis(planner, args_ns, candidates, label: str, disabled_world_collision_links):
    if planner is None:
        return None
    candidate_list = list(candidates or [])
    if not candidate_list:
        return None
    first = candidate_list[0]
    if not isinstance(first, dict) or first.get("start_q") is None:
        return None
    start_q = np.asarray(first["start_q"], dtype=np.float32).reshape(-1)[:7]
    disabled = []
    try:
        disabled = direct._set_world_collision_for_links(
            planner,
            disabled_world_collision_links,
            enabled=False,
            label=f"{label}_jimu_start_collision_diag",
        )
        diag = planner.diagnose_start_state_world_collision(start_q)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            direct._set_world_collision_for_links(
                planner,
                disabled,
                enabled=True,
                label=f"{label}_jimu_start_collision_diag",
            )
        except Exception:
            pass
    ablation = []
    valid_after_removing = []
    for item in list((diag or {}).get("ablation", []) or []):
        entry = {
            "removed": str(item.get("removed", "") or ""),
            "valid": bool(item.get("valid", False)),
            "status": str(item.get("status", "") or ""),
        }
        ablation.append(entry)
        if entry["valid"]:
            valid_after_removing.append(entry["removed"])
    group_ablation = []
    obstacle_names = [str(x) for x in list((diag or {}).get("world_obstacle_names", []) or [])]
    group_specs = {
        "virtual_walls": [name for name in obstacle_names if "virtual_side_wall" in name or "virtual_top_wall" in name],
        "assembly_blocks": [
            name
            for name in obstacle_names
            if name.startswith("scene_obstacle_") and "virtual_" not in name
        ],
        "virtual_table": [name for name in obstacle_names if name == "virtual_table_plane"],
        "virtual_walls_and_table": [
            name
            for name in obstacle_names
            if "virtual_side_wall" in name or "virtual_top_wall" in name or name == "virtual_table_plane"
        ],
        "all_scene_obstacles": list(obstacle_names),
    }
    if hasattr(planner, "set_world_obstacles_enabled"):
        try:
            for group_name, removed_names in group_specs.items():
                removed_names = [name for name in removed_names if name]
                if not removed_names:
                    continue
                disabled_group = []
                try:
                    disabled_group = planner.set_world_obstacles_enabled(removed_names, enabled=False)
                    group_valid, group_status = planner.check_start_state(start_q)
                    group_ablation.append(
                        {
                            "removed_group": group_name,
                            "removed": removed_names[:32],
                            "disabled": list(disabled_group[:32]),
                            "valid": bool(group_valid),
                            "status": str(group_status),
                        }
                    )
                finally:
                    if disabled_group:
                        planner.set_world_obstacles_enabled(disabled_group, enabled=True)
        except Exception as exc:
            group_ablation.append({"error": f"{type(exc).__name__}: {exc}"})
    attached_disabled_diag = None
    try:
        disabled_attached = direct._set_world_collision_for_links(
            planner,
            ["attached_object"],
            enabled=False,
            label=f"{label}_jimu_start_collision_diag_attached_disabled",
        )
        attached_valid, attached_status = planner.check_start_state(start_q)
        attached_disabled_diag = {
            "valid": bool(attached_valid),
            "status": str(attached_status),
            "disabled_links": list(disabled_attached or []),
        }
    except Exception as exc:
        attached_disabled_diag = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            direct._set_world_collision_for_links(
                planner,
                ["attached_object"],
                enabled=True,
                label=f"{label}_jimu_start_collision_diag_attached_disabled",
            )
        except Exception:
            pass
    bottom_z = None
    try:
        bottom_z = direct._attached_sphere_bottom_z(planner, start_q)
    except Exception:
        bottom_z = None
    result = {
        "valid": bool((diag or {}).get("valid", False)),
        "status": str((diag or {}).get("status", "") or ""),
        "world_obstacle_names": [str(x) for x in list((diag or {}).get("world_obstacle_names", []) or [])[:32]],
        "ablation": ablation[:32],
        "group_ablation": group_ablation[:16],
        "attached_disabled_diag": attached_disabled_diag,
        "valid_after_removing": valid_after_removing[:16],
        "attached_sphere_bottom_z": None if bottom_z is None else float(bottom_z),
        "virtual_table_top_z": float(getattr(args_ns, "curobo_table_z_offset", -0.01)),
        "candidate_label": str(first.get("label", "") or ""),
    }
    return result


def profile_plan_to_joint_state_jimu(planner, *args, **kwargs):
    original = _ORIGINAL_PROFILE_PLAN_TO_JOINT_STATE
    if original is None:
        return planner.plan_to_joint_state(*args, **kwargs)
    result = original(planner, *args, **kwargs)
    _record_jimu_motiongen_result(planner, "plan_to_joint_state", result)
    return result


def profile_plan_goalset_to_poses_jimu(planner, *args, **kwargs):
    original = _ORIGINAL_PROFILE_PLAN_GOALSET_TO_POSES
    if original is None:
        return planner.plan_goalset_to_poses(*args, **kwargs)
    result = original(planner, *args, **kwargs)
    _record_jimu_motiongen_result(planner, "plan_goalset_to_poses", result)
    return result


def profile_plan_batch_start_goal_pairs_jimu(planner, *args, **kwargs):
    original = _ORIGINAL_PROFILE_PLAN_BATCH_START_GOAL_PAIRS
    if original is None:
        return planner.plan_batch_start_goal_pairs(*args, **kwargs)
    results = original(planner, *args, **kwargs)
    _record_jimu_motiongen_results(planner, "plan_batch_start_goal_pairs", results)
    return results


def evaluate_curobo_pose_candidates_multi_start_jimu(planner, *args, **kwargs):
    original = _ORIGINAL_EVALUATE_CUROBO_POSE_CANDIDATES_MULTI_START
    if original is None:
        raise RuntimeError("original _evaluate_curobo_pose_candidates_multi_start is not installed")
    label = str(kwargs.get("label", "") or "")
    planner._jimu_motiongen_capture_active = True
    planner._jimu_motiongen_capture_label = label
    planner._jimu_motiongen_status_counts = {}
    planner._jimu_motiongen_failure_status_counts = {}
    planner._jimu_motiongen_success_status_counts = {}
    planner._jimu_motiongen_status_events = []
    planner._jimu_start_collision_diag = None
    winners = original(planner, *args, **kwargs)
    failure_counts = dict(getattr(planner, "_jimu_motiongen_failure_status_counts", {}) or {})
    success_counts = dict(getattr(planner, "_jimu_motiongen_success_status_counts", {}) or {})
    if "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION" in failure_counts:
        candidates = args[2] if len(args) >= 3 else kwargs.get("candidates", [])
        args_ns = args[1] if len(args) >= 2 else None
        start_diag = _jimu_start_collision_diagnosis(
            planner,
            args_ns,
            candidates,
            label,
            kwargs.get("disabled_world_collision_links"),
        )
        planner._jimu_start_collision_diag = start_diag
        if isinstance(start_diag, dict):
            print(
                f"[jimu-curobo-diag] {label or 'unknown'} start collision diag: "
                f"status={start_diag.get('status')} "
                f"attached_bottom_z={start_diag.get('attached_sphere_bottom_z')} "
                f"valid_if_remove={start_diag.get('valid_after_removing')} "
                f"obstacles={start_diag.get('world_obstacle_names')}"
            )
    if failure_counts or success_counts:
        print(
            f"[jimu-curobo-diag] {label or 'unknown'} MotionGen statuses: "
            f"success={success_counts} failure={failure_counts}"
        )
    return winners


def copy_last_candidate_counts_to_profile_jimu(prof: dict, planner) -> None:
    original = _ORIGINAL_COPY_LAST_CANDIDATE_COUNTS_TO_PROFILE
    if original is not None:
        original(prof, planner)
    if planner is None or not bool(getattr(planner, "_jimu_motiongen_capture_active", False)):
        return
    counts = dict(getattr(planner, "_jimu_motiongen_status_counts", {}) or {})
    failure_counts = dict(getattr(planner, "_jimu_motiongen_failure_status_counts", {}) or {})
    success_counts = dict(getattr(planner, "_jimu_motiongen_success_status_counts", {}) or {})
    events = list(getattr(planner, "_jimu_motiongen_status_events", []) or [])
    if counts:
        prof["motiongen_status_counts"] = counts
        prof["motiongen_failure_status_counts"] = failure_counts
        prof["motiongen_success_status_counts"] = success_counts
        prof["motiongen_status_events"] = events[:16]
        prof["motiongen_diag_label"] = str(getattr(planner, "_jimu_motiongen_capture_label", "") or "")
    start_diag = getattr(planner, "_jimu_start_collision_diag", None)
    if isinstance(start_diag, dict):
        prof["start_collision_diag"] = start_diag
    planner._jimu_motiongen_capture_active = False


def _jimu_grid_axis_positions(dim: float, count: int, radius: float, span_scale: float) -> np.ndarray:
    count = max(1, int(count))
    usable_half = max(0.0, 0.5 * float(dim) * float(span_scale) - 0.5 * float(radius))
    if count <= 1 or usable_half <= 1e-6:
        return np.zeros((1,), dtype=np.float32)
    return np.linspace(-usable_half, usable_half, count, dtype=np.float32)


def _attach_jimu_planar_spheres_to_robot(
    planner,
    q,
    *,
    object_pose_base: np.ndarray,
    dims: np.ndarray,
    radius: float,
    long_count: int,
    wide_count: int,
    span_scale: float,
    world_z_offset: float,
    link_name: str = "attached_object",
) -> bool:
    if not getattr(planner, "collision_enabled", False):
        return False
    planner._invalidate_cuda_graph_batch_ik_solvers()
    torch = planner.mods["torch"]
    q_np = planner._normalize_q(q)
    dims = np.asarray(dims, dtype=np.float32).reshape(3)
    object_pose_base = np.asarray(object_pose_base, dtype=np.float32).reshape(4, 4)
    thin_axis = int(np.argmin(dims))
    plane_axes = sorted([idx for idx in range(3) if idx != thin_axis], key=lambda idx: float(dims[idx]), reverse=True)
    long_axis, wide_axis = int(plane_axes[0]), int(plane_axes[1])
    long_positions = _jimu_grid_axis_positions(float(dims[long_axis]), int(long_count), radius, span_scale)
    wide_positions = _jimu_grid_axis_positions(float(dims[wide_axis]), int(wide_count), radius, span_scale)
    local_centers = []
    for long_pos in long_positions:
        for wide_pos in wide_positions:
            center = np.zeros(3, dtype=np.float32)
            center[long_axis] = float(long_pos)
            center[wide_axis] = float(wide_pos)
            local_centers.append(center)
    local_centers = np.asarray(local_centers, dtype=np.float32).reshape(-1, 3)

    joint_state = planner._make_start_state(q_np)
    kin_state = planner.motion_gen.compute_kinematics(joint_state)
    ee_p = planner._to_numpy(kin_state.ee_pose.position).reshape(-1, 3)[0].astype(np.float32)
    ee_q = planner._to_numpy(kin_state.ee_pose.quaternion).reshape(-1, 4)[0].astype(np.float32)
    ee_R = planner._quat_wxyz_to_rotmat(ee_q)
    obj_R = object_pose_base[:3, :3].astype(np.float32)
    obj_p = object_pose_base[:3, 3].astype(np.float32).copy()
    obj_p[2] += float(world_z_offset)
    base_centers = (obj_R @ local_centers.T).T + obj_p.reshape(1, 3)
    ee_centers = ((ee_R.T) @ (base_centers - ee_p.reshape(1, 3)).T).T

    max_spheres = int(planner.motion_gen.robot_cfg.kinematics.kinematics_config.get_number_of_spheres(str(link_name)))
    sphere_tensor = np.zeros((max_spheres, 4), dtype=np.float32)
    sphere_tensor[:, 3] = -10.0
    fill_count = min(int(len(ee_centers)), max_spheres)
    sphere_tensor[:fill_count, :3] = ee_centers[:fill_count]
    sphere_tensor[:fill_count, 3] = float(radius)
    sphere_tensor_t = torch.as_tensor(sphere_tensor, device=planner.tensor_args.device, dtype=planner.tensor_args.dtype)
    planner.motion_gen.attach_spheres_to_robot(
        sphere_radius=0.0,
        sphere_tensor=sphere_tensor_t,
        link_name=str(link_name),
    )
    try:
        planner.ik_solver.attach_object_to_robot(
            sphere_radius=0.0,
            sphere_tensor=sphere_tensor_t.clone(),
            link_name=str(link_name),
        )
    except Exception as exc:
        print(f"[jimu-curobo] failed to mirror Jimu attached sphere grid to ik_solver: {exc}")
    planner._attached_object_active = True
    enabled_spheres = planner.get_attached_sphere_count(link_name=str(link_name))
    print(
        f"[jimu-curobo] attached planar Jimu payload grid: "
        f"dims={np.round(dims * 1000.0, 1).tolist()}mm, "
        f"thin_axis={thin_axis}, plane_axes={[long_axis, wide_axis]}, "
        f"spheres={fill_count}/{max_spheres}, enabled={enabled_spheres}, "
        f"radius={float(radius) * 1000.0:.1f}mm, "
        f"span_scale={float(span_scale):.2f}, z_offset={float(world_z_offset) * 1000.0:.1f}mm"
    )
    return bool(enabled_spheres > 0)


def attach_transport_payload_to_curobo_jimu(planner, demo, args, *, label: str) -> bool:
    original = _ORIGINAL_ATTACH_TRANSPORT_PAYLOAD_TO_CUROBO
    source = direct.curobo_wrapper.normalize_object_name(getattr(args, "object_name", None))
    if source not in set(JIMU_PICK_ROLES):
        if original is None:
            return False
        return original(planner, demo, args, label=label)

    start_t = time.perf_counter()
    counter_start = direct._snapshot_profile_counters()
    profile_success = False
    profile_status = "disabled"
    if not bool(getattr(args, "curobo_attach_object", True)):
        direct._record_profile(
            args,
            "transport_attach",
            success=False,
            status=profile_status,
            elapsed_ms=round((time.perf_counter() - start_t) * 1000.0, 3),
            **direct._profile_counter_delta(counter_start),
        )
        return False
    try:
        if getattr(planner, "attached_object_active", False):
            planner.detach_object_from_robot()
        raw_dims = _load_scaled_jimu_extents(args)
        dim_scale = float(np.clip(getattr(args, "jimu_attached_sphere_dim_scale", 0.94), 0.6, 1.05))
        dims = np.maximum(raw_dims * dim_scale, 1e-4).astype(np.float32)
        radius = float(np.clip(getattr(args, "jimu_attached_sphere_radius_m", 0.008), 0.003, 0.030))
        long_count = int(max(getattr(args, "jimu_attached_sphere_long_count", 3), 1))
        wide_count = int(max(getattr(args, "jimu_attached_sphere_wide_count", 3), 1))
        span_scale = float(np.clip(getattr(args, "jimu_attached_sphere_span_scale", 0.88), 0.4, 1.0))
        current_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
        obj_p_attach, obj_q_attach = demo.get_obj_pose()
        T_world_obj_attach = direct.targeted.base.pose_to_matrix(
            np.asarray(obj_p_attach, dtype=np.float32).reshape(3),
            np.asarray(obj_q_attach, dtype=np.float32).reshape(4),
        )
        T_base_obj_attach = np.linalg.inv(direct._get_robot_base_world_transform(demo)) @ T_world_obj_attach
        base_z_offset = float(getattr(args, "curobo_attach_world_z_offset_m", 0.002))
        direct._bump_profile_counter("attach_object_count")
        ok = _attach_jimu_planar_spheres_to_robot(
            planner,
            current_q,
            object_pose_base=T_base_obj_attach,
            dims=dims,
            radius=radius,
            long_count=long_count,
            wide_count=wide_count,
            span_scale=span_scale,
            world_z_offset=base_z_offset,
        )
        if ok and getattr(planner, "attached_object_active", False):
            bottom_z = direct._attached_sphere_bottom_z(planner, current_q)
            min_clearance = float(max(getattr(args, "curobo_attach_min_start_clearance_m", 0.003), 0.0))
            max_auto_offset = float(max(getattr(args, "curobo_attach_max_auto_z_offset_m", 0.030), base_z_offset))
            if bottom_z is not None:
                print(
                    f"[jimu-curobo] {label} attached grid bottom after attach: "
                    f"z={bottom_z:.4f}m, min_clearance={min_clearance:.4f}m"
                )
            if bottom_z is not None and bottom_z < min_clearance - 1e-5 and base_z_offset < max_auto_offset:
                adjusted_offset = min(max_auto_offset, base_z_offset + (min_clearance - bottom_z))
                planner.detach_object_from_robot()
                direct._bump_profile_counter("attach_object_count")
                ok = _attach_jimu_planar_spheres_to_robot(
                    planner,
                    current_q,
                    object_pose_base=T_base_obj_attach,
                    dims=dims,
                    radius=radius,
                    long_count=long_count,
                    wide_count=wide_count,
                    span_scale=span_scale,
                    world_z_offset=adjusted_offset,
                )
                adjusted_bottom_z = direct._attached_sphere_bottom_z(planner, current_q)
                print(
                    f"[jimu-curobo] {label} reattached grid with z_offset={adjusted_offset:.4f}m, "
                    f"bottom_z={adjusted_bottom_z}"
                )
        if ok and getattr(planner, "attached_object_active", False):
            profile_success = True
            profile_status = "Success"
            return True
        profile_status = str(ok)
        return False
    except Exception as exc:
        print(f"[jimu-curobo] failed to attach planar Jimu payload for {label}: {exc}")
        profile_status = type(exc).__name__
        return False
    finally:
        direct._record_profile(
            args,
            "transport_attach",
            success=profile_success,
            status=profile_status,
            elapsed_ms=round((time.perf_counter() - start_t) * 1000.0, 3),
            **direct._profile_counter_delta(counter_start),
        )


def _jimu_partial_release_gripper_value(args: argparse.Namespace) -> float:
    open_value = float(getattr(args, "real_gripper_open", 0.0))
    close_value = float(getattr(args, "real_gripper_close", 0.91))
    fraction_closed = float(np.clip(getattr(args, "jimu_release_partial_open_fraction", 0.70), 0.0, 1.0))
    return float(open_value + (close_value - open_value) * fraction_closed)


def _jimu_partial_release_sim_gripper_value(args: argparse.Namespace | None) -> float:
    open_value = float(getattr(args, "gripper_open", -1.0) if args is not None else -1.0)
    close_value = float(getattr(args, "gripper_close", 1.0) if args is not None else 1.0)
    fraction_closed = float(np.clip(getattr(args, "jimu_release_partial_open_fraction", 0.70), 0.0, 1.0))
    return float(open_value + (close_value - open_value) * fraction_closed)


def _jimu_partial_release_enabled(args: argparse.Namespace | None) -> bool:
    return bool(args is not None and getattr(args, "jimu_partial_open_during_post_place_clearance", True))


@contextmanager
def profile_stage_jimu(args, stage_name: str, **fields):
    if _ORIGINAL_PROFILE_STAGE is None:
        raise RuntimeError("Jimu profile stage wrapper was installed before original _profile_stage was captured")
    prev_stage = getattr(_JIMU_RUNTIME_CONTEXT, "profile_stage", None)
    prev_args = getattr(_JIMU_RUNTIME_CONTEXT, "profile_args", None)
    _JIMU_RUNTIME_CONTEXT.profile_stage = str(stage_name)
    _JIMU_RUNTIME_CONTEXT.profile_args = args
    try:
        with _ORIGINAL_PROFILE_STAGE(args, stage_name, **fields) as prof:
            yield prof
    finally:
        if prev_stage is None:
            try:
                delattr(_JIMU_RUNTIME_CONTEXT, "profile_stage")
            except AttributeError:
                pass
        else:
            _JIMU_RUNTIME_CONTEXT.profile_stage = prev_stage
        if prev_args is None:
            try:
                delattr(_JIMU_RUNTIME_CONTEXT, "profile_args")
            except AttributeError:
                pass
        else:
            _JIMU_RUNTIME_CONTEXT.profile_args = prev_args


def realman_set_gripper_jimu(self, gripper_pos: float, repeats: int | None = None, hz: float | None = None):
    if _ORIGINAL_REALMAN_SET_GRIPPER is None:
        raise RuntimeError("Jimu Realman set_gripper wrapper was installed before original method was captured")
    stage = str(getattr(_JIMU_RUNTIME_CONTEXT, "profile_stage", "") or "")
    args = getattr(_JIMU_RUNTIME_CONTEXT, "profile_args", None)
    if stage == "place_open_gripper" and _jimu_partial_release_enabled(args):
        full_open = float(getattr(args, "real_gripper_open", 0.0))
        if abs(float(gripper_pos) - full_open) <= 1e-6:
            partial = _jimu_partial_release_gripper_value(args)
            print(
                "[jimu gripper] place release uses partial open before clearance: "
                f"{full_open:.3f} -> {partial:.3f}"
            )
            setattr(args, "_jimu_release_partial_open_used", True)
            return _ORIGINAL_REALMAN_SET_GRIPPER(self, partial, repeats=repeats, hz=hz)
    return _ORIGINAL_REALMAN_SET_GRIPPER(self, gripper_pos, repeats=repeats, hz=hz)


def sync_demo_gripper_state_jimu(demo, closed: bool, steps: int = 3):
    if _ORIGINAL_SYNC_DEMO_GRIPPER_STATE is None:
        raise RuntimeError("Jimu sync_demo_gripper_state wrapper was installed before original function was captured")
    stage = str(getattr(_JIMU_RUNTIME_CONTEXT, "profile_stage", "") or "")
    args = getattr(_JIMU_RUNTIME_CONTEXT, "profile_args", None) or getattr(demo, "args", None)
    if stage != "place_open_gripper" or bool(closed) or not _jimu_partial_release_enabled(args):
        return _ORIGINAL_SYNC_DEMO_GRIPPER_STATE(demo, closed, steps=steps)

    sim_value = _jimu_partial_release_sim_gripper_value(args)
    min_steps = int(max(getattr(args, "sim_gripper_sync_min_steps", 20), 0))
    total_steps = max(int(steps), min_steps)
    try:
        demo.hold_current_and_set_gripper(sim_value, steps=total_steps)
        direct.targeted.base.refresh_frozen_active_object_pose(demo)
        demo.refresh_runtime_handles(rebuild_visual=False)
        direct.targeted.base.sync_planner_qpos_from_demo(demo)
        direct.targeted.base.update_attached_box_visual(demo)
        gap = direct.targeted.base.get_sim_gripper_pad_gap(demo)
        gap_text = "unknown" if gap is None else f"{gap:.4f} m"
        print(
            "[jimu gripper] simulated partial release open before clearance: "
            f"value={sim_value:.3f}, steps={total_steps}, pad_gap={gap_text}"
        )
    except Exception as exc:
        print(f"[warn] failed to sync simulated partial release gripper state: {exc}")


def execute_pose_path_stage_jimu(
    demo,
    bridge_mod,
    real_exec,
    label: str,
    pose,
    q_path,
    gripper_pos: float,
    args,
    *,
    use_attach: bool = False,
    allow_start_in_collision: bool = False,
    skip_confirmation: bool = False,
):
    if _ORIGINAL_EXECUTE_POSE_PATH_STAGE is None:
        raise RuntimeError("Jimu execute_pose_path_stage wrapper was installed before original function was captured")
    label_text = str(label or "")
    if label_text != "post_place_clearance" or not _jimu_partial_release_enabled(args):
        return _ORIGINAL_EXECUTE_POSE_PATH_STAGE(
            demo,
            bridge_mod,
            real_exec,
            label,
            pose,
            q_path,
            gripper_pos,
            args,
            use_attach=use_attach,
            allow_start_in_collision=allow_start_in_collision,
            skip_confirmation=skip_confirmation,
        )

    partial = _jimu_partial_release_gripper_value(args)
    full_open = float(getattr(args, "real_gripper_open", gripper_pos))
    print(
        "[jimu gripper] post-place clearance keeps partial open during lift: "
        f"{partial:.3f}; full open after clearance: {full_open:.3f}"
    )
    ok, q_sent = _ORIGINAL_EXECUTE_POSE_PATH_STAGE(
        demo,
        bridge_mod,
        real_exec,
        label,
        pose,
        q_path,
        partial,
        args,
        use_attach=use_attach,
        allow_start_in_collision=allow_start_in_collision,
        skip_confirmation=skip_confirmation,
    )
    if ok and bool(getattr(args, "jimu_full_open_after_post_place_clearance", True)):
        try:
            if real_exec is not None:
                if _ORIGINAL_REALMAN_SET_GRIPPER is None:
                    real_exec.set_gripper(full_open)
                else:
                    _ORIGINAL_REALMAN_SET_GRIPPER(
                        real_exec,
                        full_open,
                        repeats=int(getattr(args, "real_gripper_command_repeats", 2)),
                        hz=float(getattr(args, "real_gripper_command_hz", 10.0)),
                    )
            else:
                sync_fn = getattr(direct.targeted.base, "sync_demo_gripper_state", None)
                if callable(sync_fn):
                    sync_fn(demo, closed=False, steps=max(int(getattr(args, "sim_gripper_sync_min_steps", 8)), 1))
            direct._record_profile(
                args,
                "jimu_full_open_after_post_place_clearance",
                success=True,
                status="Success",
                partial_gripper_pos=partial,
                full_open_gripper_pos=full_open,
            )
            print("[jimu gripper] post-place clearance finished; gripper opened to maximum")
        except Exception as exc:
            direct._record_profile(
                args,
                "jimu_full_open_after_post_place_clearance",
                success=False,
                status=type(exc).__name__,
                partial_gripper_pos=partial,
                full_open_gripper_pos=full_open,
                error=str(exc),
            )
            raise
    return ok, q_sent


def build_arg_parser():
    install_jimu_object_specs()
    parser = direct_sam6d.build_arg_parser()
    parser.description = (
        "Jimu multi-block SAM6D/SAM3 scene capture -> direct cuRobo grasp/place. "
        "This entrypoint keeps pick_jiaobang direct/sam6d logic unchanged and only registers "
        "floor + first/second-layer wall roles at runtime."
    )
    parser.set_defaults(
        object_name="right_wall",
        cycle_object_names=list(JIMU_PICK_ROLES),
        tracked_scene_object_names=[JIMU_FLOOR_ROLE],
        target_selection_order="cycle",
        repeat_count=len(JIMU_PICK_ROLES),
        next_cycle_plan_prefetch=False,
        sam6d_provider_script=assembly_sam6d.DEFAULT_SAM6D_PROVIDER_SCRIPT,
        sam6d_output_root=str(BETA_DIR / "sam6d_jimu_direct_runs"),
        sam6d_reuse_scene_across_cycles=True,
        sam6d_no_pem_warmup_during_sam3=True,
        sam3_full_scene_keep_multi_instances=True,
        sam3_max_masks_per_item=len(JIMU_SCENE_ROLES),
        sam3_confidence_threshold=assembly_sam6d.DEFAULT_SAM3_CONFIDENCE_THRESHOLD,
        sam6d_confirm_segmentation=True,
        sam6d_require_full_scene_masks=True,
        empty_grasp_check_after_lift=False,
        empty_grasp_relocalize_target=False,
        empty_grasp_max_relocalize_retries=0,
        snap_low_profile_objects_flat_on_table=False,
        fast_chain_screening=True,
        fast_chain_relation_ik_slots=16,
        fast_chain_ik_seeds=32,
        fast_chain_cuda_graph_ik=True,
        fast_chain_cuda_graph_ik_fixed_batch_size=16,
        fast_chain_cuda_graph_ik_max_batch_size=128,
        fast_chain_top_pairs=16,
        fast_chain_place_rank_grasp_limit=16,
        fixed_tabletop_fast_chain_place_rank_grasp_limit=16,
        joint_search_max_grasp_candidates=16,
        joint_search_start_collision_lift_m=0.13,
        fast_chain_allow_legacy_fallback=False,
        transport_use_prefilter_q_goal=True,
        transport_prefilter_q_goal_max_trials=1,
        transport_prefilter_q_goal_timeout=2.0,
        transport_prefilter_q_goal_num_trajopt_seeds=1,
        vertical_place_hover_height_m=0.08,
        final_contact_clearance_m=0.0,
        planner_virtual_top_wall_z=1.5,
    )
    parser.add_argument(
        "--jimu-build-layers",
        choices=["first", "two"],
        default="two",
        help="Default build set: first builds four walls; two also builds four second-layer walls.",
    )
    parser.add_argument(
        "--jimu-enforce-layer-order",
        dest="jimu_enforce_layer_order",
        action="store_true",
        default=True,
        help="Only allow second-layer Jimu roles after every selectable first-layer role has been placed.",
    )
    parser.add_argument(
        "--no-jimu-enforce-layer-order",
        dest="jimu_enforce_layer_order",
        action="store_false",
        help="Allow first-layer and second-layer Jimu roles to be selected from the same target pool.",
    )
    parser.add_argument(
        "--jimu-wait-on-failure-before-close",
        dest="jimu_wait_on_failure_before_close",
        action="store_true",
        default=True,
        help="When the final Jimu run fails in human render mode, keep rendering the scene until Enter is pressed.",
    )
    parser.add_argument(
        "--no-jimu-wait-on-failure-before-close",
        dest="jimu_wait_on_failure_before_close",
        action="store_false",
    )
    parser.add_argument(
        "--jimu-scene-roles",
        type=str,
        nargs="*",
        default=list(JIMU_SCENE_ROLES),
        help="Role names represented by the same-object SAM6D detections.",
    )
    parser.add_argument(
        "--jimu-provider-object-name",
        type=str,
        default=JIMU_PROVIDER_OBJECT_NAME,
        help="Object spec sent to SAM6D for every Jimu instance.",
    )
    parser.add_argument(
        "--jimu-sim-asset-file",
        type=str,
        default="",
        help="Override the Jimu GLB used by ManiSkill visual/collision. Defaults to a bundled 74x6x74mm red box.",
    )
    parser.add_argument(
        "--jimu-plate-size-m",
        type=float,
        default=JIMU_PLATE_SIZE_M,
        help="Logical Jimu square plate side length used for placement, stacking, and attached payload collision.",
    )
    parser.add_argument(
        "--jimu-plate-thickness-m",
        type=float,
        default=JIMU_PLATE_THICKNESS_M,
        help="Logical Jimu plate thickness used for placement, stacking, and attached payload collision.",
    )
    parser.add_argument(
        "--jimu-use-mesh-extents",
        dest="jimu_use_mesh_extents",
        action="store_true",
        default=False,
        help="Use the GLB scaled bounds instead of the logical 74mm square plate dimensions.",
    )
    parser.add_argument(
        "--jimu-assembly-sam6d-provider-script",
        type=str,
        default=assembly_sam6d.DEFAULT_SAM6D_PROVIDER_SCRIPT,
        help="SAM6D provider script used by the Beta assembly-style subprocess call.",
    )
    parser.add_argument(
        "--jimu-role-instance-map",
        type=str,
        nargs="*",
        default=None,
        help="Optional role:index mapping after checking the SAM3/SAM6D overlay, e.g. floor:3 right_wall:0.",
    )
    parser.add_argument(
        "--jimu-floor-normal-threshold",
        type=float,
        default=0.65,
        help="Minimum abs(dot(block local Y, base Z)) used to auto-select the flat floor plate.",
    )
    parser.add_argument("--jimu-wall-hover-height", type=float, default=0.08)
    parser.add_argument("--jimu-wall-release-retreat-height", type=float, default=0.08)
    parser.add_argument("--jimu-second-layer-hover-height", type=float, default=0.08)
    parser.add_argument("--jimu-second-layer-release-retreat-height", type=float, default=0.08)
    parser.add_argument(
        "--jimu-second-layer-z-extra",
        type=float,
        default=0.0,
        help="Extra offset along the parent wall local Z when stacking a second-layer wall.",
    )
    parser.add_argument(
        "--jimu-place-symmetry-deg",
        type=float,
        nargs="*",
        default=[0.0, 90.0, 180.0, 270.0],
        help="Equivalent local-Y in-plane rotations for Jimu wall release poses.",
    )
    parser.add_argument(
        "--jimu-fast-chain-symmetry-expand-max-per-grasp",
        type=int,
        default=12,
        help="Maximum Jimu symmetry place candidates evaluated per grasp relation in the fast-chain expansion pass.",
    )
    parser.add_argument("--jimu-place-symmetry-enabled", dest="jimu_place_symmetry_enabled", action="store_true", default=False)
    parser.add_argument("--no-jimu-place-symmetry", dest="jimu_place_symmetry_enabled", action="store_false")
    parser.add_argument(
        "--jimu-parallel-grasp-place",
        dest="jimu_parallel_grasp_place",
        action="store_true",
        default=True,
        help="For Jimu walls, keep the solved grasp TCP relation and apply only the required world-Z yaw plus translation to the target center.",
    )
    parser.add_argument("--no-jimu-parallel-grasp-place", dest="jimu_parallel_grasp_place", action="store_false")
    parser.add_argument(
        "--jimu-parallel-grasp-place-max-sources-per-grasp",
        type=int,
        default=1,
        help="Number of target-center source place candidates converted to parallel-grasp place per grasp relation.",
    )
    parser.add_argument(
        "--jimu-parallel-grasp-place-snap-yaw-90",
        dest="jimu_parallel_grasp_place_snap_yaw_90",
        action="store_true",
        default=True,
        help="Snap the world-Z release yaw correction to 90-degree increments for Jimu walls.",
    )
    parser.add_argument(
        "--no-jimu-parallel-grasp-place-snap-yaw-90",
        dest="jimu_parallel_grasp_place_snap_yaw_90",
        action="store_false",
    )
    parser.add_argument(
        "--jimu-canonicalize-local-frames",
        dest="jimu_canonicalize_local_frames",
        action="store_true",
        default=True,
        help="Rebuild Jimu local frames before loading SAM6D poses: floor local Y is world Z; wall local Z is world Z.",
    )
    parser.add_argument("--no-jimu-canonicalize-local-frames", dest="jimu_canonicalize_local_frames", action="store_false")
    parser.add_argument(
        "--jimu-canonical-snap-cardinal",
        dest="jimu_canonical_snap_cardinal",
        action="store_true",
        default=True,
        help="Snap canonical Jimu horizontal axes to the nearest base-frame cardinal direction.",
    )
    parser.add_argument("--no-jimu-canonical-snap-cardinal", dest="jimu_canonical_snap_cardinal", action="store_false")
    parser.add_argument(
        "--jimu-cad-to-sim-correction",
        choices=["auto", "none", "red_jimu_cube", "red_bricks_cube"],
        default="none",
        help=(
            "Local-frame correction before loading SAM6D poses into the direct GLB scene. "
            "Default none because this entrypoint uses the same red_jimu_cube.glb mesh for SAM6D and simulation."
        ),
    )
    parser.add_argument(
        "--jimu-cad-to-sim-local-rpy-deg",
        type=str,
        default="",
        help="Override the Jimu CAD-to-sim local rotation as 'roll,pitch,yaw' degrees.",
    )
    parser.add_argument(
        "--jimu-attached-sphere-radius-m",
        type=float,
        default=0.008,
        help="Radius for the Jimu planar attached-payload collision sphere grid used by cuRobo transport.",
    )
    parser.add_argument("--jimu-attached-sphere-long-count", type=int, default=3)
    parser.add_argument("--jimu-attached-sphere-wide-count", type=int, default=3)
    parser.add_argument("--jimu-attached-sphere-span-scale", type=float, default=0.88)
    parser.add_argument("--jimu-attached-sphere-dim-scale", type=float, default=0.94)
    parser.add_argument(
        "--jimu-partial-open-during-post-place-clearance",
        dest="jimu_partial_open_during_post_place_clearance",
        action="store_true",
        default=True,
        help="Release with a partial gripper opening, lift away with that opening, then fully open after clearance.",
    )
    parser.add_argument(
        "--no-jimu-partial-open-during-post-place-clearance",
        dest="jimu_partial_open_during_post_place_clearance",
        action="store_false",
    )
    parser.add_argument(
        "--jimu-release-partial-open-fraction",
        type=float,
        default=0.70,
        help="Partial release opening as a fraction from full-open to closed; 0=open, 1=closed.",
    )
    parser.add_argument(
        "--jimu-full-open-after-post-place-clearance",
        dest="jimu_full_open_after_post_place_clearance",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-jimu-full-open-after-post-place-clearance",
        dest="jimu_full_open_after_post_place_clearance",
        action="store_false",
    )
    parser.add_argument(
        "--jimu-print-role-assignment",
        dest="jimu_print_role_assignment",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no-jimu-print-role-assignment", dest="jimu_print_role_assignment", action="store_false")
    return parser


def parse_args():
    args = build_arg_parser().parse_args()
    default_pick_roles = _default_pick_roles_for_layers(getattr(args, "jimu_build_layers", "two"))
    default_scene_roles = [JIMU_FLOOR_ROLE, *default_pick_roles]
    if _argv_has_option("--cycle-object-names"):
        args.cycle_object_names = _split_names(getattr(args, "cycle_object_names", None)) or default_pick_roles
    else:
        args.cycle_object_names = default_pick_roles
    if _argv_has_option("--jimu-scene-roles"):
        args.jimu_scene_roles = _split_names(getattr(args, "jimu_scene_roles", None)) or default_scene_roles
    else:
        args.jimu_scene_roles = default_scene_roles
    args.tracked_scene_object_names = _split_names(getattr(args, "tracked_scene_object_names", None))
    if JIMU_FLOOR_ROLE not in args.tracked_scene_object_names:
        args.tracked_scene_object_names.insert(0, JIMU_FLOOR_ROLE)
    if _argv_has_option("--repeat-count"):
        args.repeat_count = max(int(getattr(args, "repeat_count", len(default_pick_roles))), len(args.cycle_object_names))
    else:
        args.repeat_count = len(args.cycle_object_names)
    args.sam3_full_scene_keep_multi_instances = True
    args.sam3_max_masks_per_item = max(int(getattr(args, "sam3_max_masks_per_item", 1) or 1), len(args.jimu_scene_roles))
    if bool(getattr(args, "skip_foundationpose", False)):
        print("[jimu-sam6d] ignoring --skip-foundationpose; this entrypoint uses SAM6D camera poses")
        args.skip_foundationpose = False
    install_jimu_runtime_config(args)
    sim_asset_file = _jimu_sim_asset_file_override(args)
    print(f"[jimu config] sim_asset_file={sim_asset_file or 'object_specs default'}")
    jimu_extents = _load_scaled_jimu_extents(args)
    print(
        "[jimu config] logical_plate_extents_mm="
        f"{np.round(jimu_extents * 1000.0, 2).tolist()} "
        f"(use_mesh_extents={bool(getattr(args, 'jimu_use_mesh_extents', False))})"
    )
    print(f"[jimu config] cad_to_sim_local_rpy_deg={list(_jimu_cad_to_sim_rpy_deg(args))}")
    print(f"[jimu config] snap_low_profile_objects_flat_on_table={bool(args.snap_low_profile_objects_flat_on_table)}")
    print(
        "[jimu config] build_layers="
        f"{str(args.jimu_build_layers)}, scene_roles={list(args.jimu_scene_roles)}, "
        f"cycle_roles={list(args.cycle_object_names)}, repeat_count={int(args.repeat_count)}, "
        f"enforce_layer_order={bool(args.jimu_enforce_layer_order)}"
    )
    print(
        "[jimu config] fast_chain_flow=ik_batch_intersection, "
        f"slots={int(args.fast_chain_relation_ik_slots)}, "
        f"seeds={int(args.fast_chain_ik_seeds)}, "
        f"cuda_graph={bool(args.fast_chain_cuda_graph_ik)}, "
        f"fixed_batch={int(args.fast_chain_cuda_graph_ik_fixed_batch_size)}, "
        f"top_pairs={int(args.fast_chain_top_pairs)}, "
        f"legacy_fallback={bool(args.fast_chain_allow_legacy_fallback)}"
    )
    print(
        "[jimu config] transport_q_goal_prefilter="
        f"{bool(args.transport_use_prefilter_q_goal)}, "
        f"trials={int(args.transport_prefilter_q_goal_max_trials)}, "
        f"timeout={float(args.transport_prefilter_q_goal_timeout):.2f}, "
        f"trajopt_seeds={int(args.transport_prefilter_q_goal_num_trajopt_seeds)}"
    )
    print(f"[jimu config] empty_grasp_check_after_lift={bool(args.empty_grasp_check_after_lift)}")
    print(
        "[jimu config] jimu place rules use vertical_place auto-classification, "
        f"vertical_place_hover_height_m={float(args.vertical_place_hover_height_m):.3f}, "
        f"final_contact_clearance_m={float(args.final_contact_clearance_m):.3f}, "
        f"start_collision_lift_m={float(args.joint_search_start_collision_lift_m):.3f}, "
        f"virtual_top_wall_z={float(args.planner_virtual_top_wall_z):.3f}"
    )
    print(
        "[jimu config] canonical_frames="
        f"{bool(args.jimu_canonicalize_local_frames)}, "
        f"snap_cardinal={bool(args.jimu_canonical_snap_cardinal)}, "
        f"place_symmetry_enabled={bool(args.jimu_place_symmetry_enabled)}, "
        f"place_symmetry_deg={np.round(np.asarray(_jimu_symmetry_degrees(args), dtype=np.float32), 1).tolist()}, "
        f"parallel_grasp_place={bool(args.jimu_parallel_grasp_place)}, "
        f"parallel_sources_per_grasp={int(args.jimu_parallel_grasp_place_max_sources_per_grasp)}, "
        f"parallel_snap_yaw_90={bool(args.jimu_parallel_grasp_place_snap_yaw_90)}"
    )
    print(
        "[jimu config] attached_payload_grid="
        f"{int(args.jimu_attached_sphere_long_count)}x{int(args.jimu_attached_sphere_wide_count)}, "
        f"radius={float(args.jimu_attached_sphere_radius_m) * 1000.0:.1f}mm, "
        f"span_scale={float(args.jimu_attached_sphere_span_scale):.2f}, "
        f"dim_scale={float(args.jimu_attached_sphere_dim_scale):.2f}"
    )
    print(
        "[jimu config] post_place_gripper="
        f"partial_enabled={bool(args.jimu_partial_open_during_post_place_clearance)}, "
        f"partial_fraction={float(args.jimu_release_partial_open_fraction):.2f}, "
        f"partial_value={_jimu_partial_release_gripper_value(args):.3f}, "
        f"full_open_after_clearance={bool(args.jimu_full_open_after_post_place_clearance)}"
    )
    direct._configure_curobo_torch_extensions(args)
    return args


def main():
    global _ORIGINAL_TARGETED_BUILD_PLACE_PLAN_VARIANTS
    global _ORIGINAL_FAST_CHAIN_RANK_PAIRED_RELATION_CANDIDATES
    global _ORIGINAL_FAST_CHAIN_RELATION_MATCH_KEY
    global _ORIGINAL_EVALUATE_CUROBO_POSE_CANDIDATES_MULTI_START
    global _ORIGINAL_COPY_LAST_CANDIDATE_COUNTS_TO_PROFILE
    global _ORIGINAL_PROFILE_PLAN_TO_JOINT_STATE
    global _ORIGINAL_PROFILE_PLAN_GOALSET_TO_POSES
    global _ORIGINAL_PROFILE_PLAN_BATCH_START_GOAL_PAIRS
    global _ORIGINAL_ATTACH_TRANSPORT_PAYLOAD_TO_CUROBO
    global _ORIGINAL_PROFILE_STAGE
    global _ORIGINAL_EXECUTE_POSE_PATH_STAGE
    global _ORIGINAL_REALMAN_SET_GRIPPER
    global _ORIGINAL_SYNC_DEMO_GRIPPER_STATE
    global _ORIGINAL_SELECT_RANDOM_CYCLE_TARGET
    install_jimu_runtime_config()
    original_capture = direct.targeted.base.capture_or_reuse_foundationpose_scene
    original_relocalize = direct._relocalize_active_target_after_empty_grasp
    original_run_episode = direct.run_targeted_place_episode_curobo_direct
    original_parse_args = direct.parse_args
    original_build_place_variants = direct.targeted.build_targeted_place_plan_variants
    original_rank_paired = direct._fast_chain_rank_paired_relation_candidates
    original_relation_match_key = direct._fast_chain_relation_match_key
    original_evaluate_multi_start = direct._evaluate_curobo_pose_candidates_multi_start
    original_copy_candidate_counts = direct._copy_last_candidate_counts_to_profile
    original_profile_plan_joint = direct._profile_plan_to_joint_state
    original_profile_plan_goalset = direct._profile_plan_goalset_to_poses
    original_profile_plan_batch_pairs = direct._profile_plan_batch_start_goal_pairs
    original_attach_transport_payload = direct._attach_transport_payload_to_curobo
    original_profile_stage = direct._profile_stage
    original_execute_pose_path_stage = direct.targeted.base.execute_pose_path_stage
    original_realman_set_gripper = direct.targeted.base.RealmanJointExecutor.set_gripper
    original_sync_demo_gripper_state = direct.targeted.base.sync_demo_gripper_state
    original_select_random_cycle_target = direct.targeted._select_random_cycle_target
    _ORIGINAL_TARGETED_BUILD_PLACE_PLAN_VARIANTS = original_build_place_variants
    _ORIGINAL_FAST_CHAIN_RANK_PAIRED_RELATION_CANDIDATES = original_rank_paired
    _ORIGINAL_FAST_CHAIN_RELATION_MATCH_KEY = original_relation_match_key
    _ORIGINAL_EVALUATE_CUROBO_POSE_CANDIDATES_MULTI_START = original_evaluate_multi_start
    _ORIGINAL_COPY_LAST_CANDIDATE_COUNTS_TO_PROFILE = original_copy_candidate_counts
    _ORIGINAL_PROFILE_PLAN_TO_JOINT_STATE = original_profile_plan_joint
    _ORIGINAL_PROFILE_PLAN_GOALSET_TO_POSES = original_profile_plan_goalset
    _ORIGINAL_PROFILE_PLAN_BATCH_START_GOAL_PAIRS = original_profile_plan_batch_pairs
    _ORIGINAL_ATTACH_TRANSPORT_PAYLOAD_TO_CUROBO = original_attach_transport_payload
    _ORIGINAL_PROFILE_STAGE = original_profile_stage
    _ORIGINAL_EXECUTE_POSE_PATH_STAGE = original_execute_pose_path_stage
    _ORIGINAL_REALMAN_SET_GRIPPER = original_realman_set_gripper
    _ORIGINAL_SYNC_DEMO_GRIPPER_STATE = original_sync_demo_gripper_state
    _ORIGINAL_SELECT_RANDOM_CYCLE_TARGET = original_select_random_cycle_target
    try:
        direct.targeted.base.capture_or_reuse_foundationpose_scene = capture_or_reuse_jimu_sam6d_scene
        direct._relocalize_active_target_after_empty_grasp = relocalize_active_target_after_empty_grasp_jimu
        direct.run_targeted_place_episode_curobo_direct = direct_sam6d._wrap_run_targeted_place_episode_for_sam6d_prefetch_fail_fast(
            original_run_episode
        )
        direct.parse_args = parse_args
        direct.targeted.build_targeted_place_plan_variants = build_targeted_place_plan_variants_jimu
        direct._fast_chain_rank_paired_relation_candidates = fast_chain_rank_paired_relation_candidates_jimu
        direct._fast_chain_relation_match_key = fast_chain_relation_match_key_jimu
        direct._evaluate_curobo_pose_candidates_multi_start = evaluate_curobo_pose_candidates_multi_start_jimu
        direct._copy_last_candidate_counts_to_profile = copy_last_candidate_counts_to_profile_jimu
        direct._profile_plan_to_joint_state = profile_plan_to_joint_state_jimu
        direct._profile_plan_goalset_to_poses = profile_plan_goalset_to_poses_jimu
        direct._profile_plan_batch_start_goal_pairs = profile_plan_batch_start_goal_pairs_jimu
        direct._attach_transport_payload_to_curobo = attach_transport_payload_to_curobo_jimu
        direct._profile_stage = profile_stage_jimu
        direct.targeted.base.execute_pose_path_stage = execute_pose_path_stage_jimu
        direct.targeted.base.RealmanJointExecutor.set_gripper = realman_set_gripper_jimu
        direct.targeted.base.sync_demo_gripper_state = sync_demo_gripper_state_jimu
        direct.targeted._select_random_cycle_target = select_random_cycle_target_jimu_layered
        direct.main()
    finally:
        direct.targeted.base.capture_or_reuse_foundationpose_scene = original_capture
        direct._relocalize_active_target_after_empty_grasp = original_relocalize
        direct.run_targeted_place_episode_curobo_direct = original_run_episode
        direct.parse_args = original_parse_args
        direct.targeted.build_targeted_place_plan_variants = original_build_place_variants
        direct._fast_chain_rank_paired_relation_candidates = original_rank_paired
        direct._fast_chain_relation_match_key = original_relation_match_key
        direct._evaluate_curobo_pose_candidates_multi_start = original_evaluate_multi_start
        direct._copy_last_candidate_counts_to_profile = original_copy_candidate_counts
        direct._profile_plan_to_joint_state = original_profile_plan_joint
        direct._profile_plan_goalset_to_poses = original_profile_plan_goalset
        direct._profile_plan_batch_start_goal_pairs = original_profile_plan_batch_pairs
        direct._attach_transport_payload_to_curobo = original_attach_transport_payload
        direct._profile_stage = original_profile_stage
        direct.targeted.base.execute_pose_path_stage = original_execute_pose_path_stage
        direct.targeted.base.RealmanJointExecutor.set_gripper = original_realman_set_gripper
        direct.targeted.base.sync_demo_gripper_state = original_sync_demo_gripper_state
        direct.targeted._select_random_cycle_target = original_select_random_cycle_target
        _ORIGINAL_TARGETED_BUILD_PLACE_PLAN_VARIANTS = None
        _ORIGINAL_FAST_CHAIN_RANK_PAIRED_RELATION_CANDIDATES = None
        _ORIGINAL_FAST_CHAIN_RELATION_MATCH_KEY = None
        _ORIGINAL_EVALUATE_CUROBO_POSE_CANDIDATES_MULTI_START = None
        _ORIGINAL_COPY_LAST_CANDIDATE_COUNTS_TO_PROFILE = None
        _ORIGINAL_PROFILE_PLAN_TO_JOINT_STATE = None
        _ORIGINAL_PROFILE_PLAN_GOALSET_TO_POSES = None
        _ORIGINAL_PROFILE_PLAN_BATCH_START_GOAL_PAIRS = None
        _ORIGINAL_ATTACH_TRANSPORT_PAYLOAD_TO_CUROBO = None
        _ORIGINAL_PROFILE_STAGE = None
        _ORIGINAL_EXECUTE_POSE_PATH_STAGE = None
        _ORIGINAL_REALMAN_SET_GRIPPER = None
        _ORIGINAL_SYNC_DEMO_GRIPPER_STATE = None
        _ORIGINAL_SELECT_RANDOM_CYCLE_TARGET = None


if __name__ == "__main__":
    main()
