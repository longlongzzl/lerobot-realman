#!/usr/bin/env python3
"""Portable Jimu four-wall entrypoint.

This file intentionally does not import Beta_demo-codex modules.  It only
patches the existing pick_jiaobang direct/SAM6D runtime at process startup,
then delegates execution back to that runtime.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import os
import random
import re
import sys
import threading
import time
import types
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from transforms3d.euler import euler2mat, mat2euler


SCRIPT_DIR = Path(__file__).resolve().parent


def _preparse_lerobot_root() -> Path | None:
    argv = list(sys.argv[1:])
    for idx, item in enumerate(argv):
        if item == "--lerobot-root" and idx + 1 < len(argv):
            return Path(argv[idx + 1]).expanduser()
        if item.startswith("--lerobot-root="):
            return Path(item.split("=", 1)[1]).expanduser()
    env_root = os.environ.get("LEROBOT_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return None


def _find_lerobot_root() -> Path:
    explicit = _preparse_lerobot_root()
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend([SCRIPT_DIR, SCRIPT_DIR.parent, Path.cwd(), Path.cwd().parent])
    candidates.extend(SCRIPT_DIR.parents)
    candidates.extend(Path.cwd().parents)
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "pick_jiaobang").is_dir():
            return candidate
    checked = ", ".join(str(path) for path in list(seen)[:8])
    raise FileNotFoundError(
        "Cannot find lerobot root containing pick_jiaobang. "
        "Put this file under the lerobot tree, pass --lerobot-root /path/to/lerobot, "
        f"or set LEROBOT_ROOT. checked={checked}"
    )


REPO_ROOT = _find_lerobot_root()
PICK_JIAOBANG_DIR = REPO_ROOT / "pick_jiaobang"
if str(PICK_JIAOBANG_DIR) not in sys.path:
    sys.path.insert(0, str(PICK_JIAOBANG_DIR))

import object_specs  # noqa: E402
import place_rules  # noqa: E402
import rm75_jiaobang_pick_place_targeted_curobo_direct_pre_place as direct  # noqa: E402
import rm75_jiaobang_pick_place_targeted_curobo_direct_pre_place_sam6d as direct_sam6d  # noqa: E402


JIMU_PROVIDER_OBJECT_NAME = "red_bricks_cube"
JIMU_TRAY_OBJECT_NAME = "jimu_liaoban"
JIMU_BASE_ASSEMBLY_OBJECT_NAME = "jimu_base_assembly"
JIMU_FLOOR_ROLE = "floor"
JIMU_BASE_SUPPORT_ROLES = (
    "floor_right_support",
    "floor_back_support",
    "floor_left_support",
    "floor_front_support",
)
JIMU_BASE_ROLES = (JIMU_FLOOR_ROLE, *JIMU_BASE_SUPPORT_ROLES)
JIMU_FIRST_LAYER_ROLES = ("right_wall", "back_wall", "left_wall", "front_wall")
JIMU_SECOND_LAYER_ROLES = ("right_second_wall", "back_second_wall", "left_second_wall", "front_second_wall")
JIMU_PICK_ROLES = (*JIMU_FIRST_LAYER_ROLES, *JIMU_SECOND_LAYER_ROLES)
JIMU_SPARE_TRAY_SLOT_ROLES = tuple(f"jimu_spare_slot_{idx:02d}" for idx in range(9, 15))
# Keep the second-layer pick plates away from the leftmost tray edge by default.
# The latest IK traces showed the row-1/col-0 plate consistently had zero
# pregrasp IK solutions while row-1/col-1..3 were reachable with the same
# tilt-only candidate batch.
JIMU_TRAY_SLOT_ROLES = (
    *JIMU_FIRST_LAYER_ROLES,
    *JIMU_SPARE_TRAY_SLOT_ROLES[:3],
    JIMU_SPARE_TRAY_SLOT_ROLES[3],
    *JIMU_SECOND_LAYER_ROLES,
    *JIMU_SPARE_TRAY_SLOT_ROLES[4:],
)
JIMU_LEGACY_SCENE_ROLES = (JIMU_FLOOR_ROLE, *JIMU_PICK_ROLES)
JIMU_SCENE_ROLES = (*JIMU_BASE_ROLES, *JIMU_PICK_ROLES)
JIMU_DERIVED_ROLE_SET = set((*JIMU_BASE_ROLES, *JIMU_TRAY_SLOT_ROLES))
PORTABLE_REPRO_DIR = SCRIPT_DIR / "jimu_portable_repro"
PORTABLE_DEFAULT_ASSEMBLY_SCENE_JSON = PORTABLE_REPRO_DIR / "scenes" / "jimu_assembly_anchors_default_sam6d.json"
PORTABLE_DEFAULT_SCENE_JSON = PORTABLE_DEFAULT_ASSEMBLY_SCENE_JSON
PORTABLE_LEGACY_DEFAULT_SCENE_JSON = PORTABLE_REPRO_DIR / "scenes" / "jimu_9objects_default_sam6d.json"
PORTABLE_DEFAULT_MESH_FILE = PORTABLE_REPRO_DIR / "assets" / "red_jimu_cube.glb"
PORTABLE_DEFAULT_SIM_ASSET_FILE = PORTABLE_REPRO_DIR / "assets" / "red_jimu_plate_74x6x74.glb"
PORTABLE_DEFAULT_TRAY_MESH_FILE = PORTABLE_REPRO_DIR / "assets" / "jimu_liaoban_new" / "jimu_liaoban_new.obj"
PORTABLE_DEFAULT_LOADED_TRAY_MESH_FILE = PORTABLE_REPRO_DIR / "assets" / "jimu_liaoban_new" / "jimu_liaoban_new_loaded_14plates.glb"
PORTABLE_DEFAULT_BASE_ASSEMBLY_MESH_FILE = PORTABLE_REPRO_DIR / "assets" / "jimu_base_assembly_5plates.glb"
PORTABLE_DEFAULT_TRAY_APRILTAG_CENTER_OFFSET_X_M = 0.0
PORTABLE_DEFAULT_TRAY_APRILTAG_CENTER_OFFSET_Y_M = 0.0
PORTABLE_DEFAULT_TRAY_SLOT_X_OFFSET_M = 0.006
PORTABLE_DEFAULT_SAM6D_PROVIDER_SCRIPT = SCRIPT_DIR / "jimu_sam6d_pose_provider.py"
PORTABLE_LEROBOT_SIM2REAL_ROOT = REPO_ROOT / "lerobot-sim2real"
PORTABLE_CAMERA_EXTRINSIC = REPO_ROOT / "rm75_pick_place_app" / "assets" / "calibration" / "camera_extrinsic_opencv.npy"
PORTABLE_MANISKILL_ROOT = PORTABLE_REPRO_DIR / "maniskill_env" / "mani_skill"
PORTABLE_MANISKILL_ROBOTS_DIR = PORTABLE_MANISKILL_ROOT / "agents" / "robots"
PORTABLE_MANISKILL_DT_DIR = PORTABLE_MANISKILL_ROOT / "envs" / "tasks" / "digital_twins"
PORTABLE_MANISKILL_PICK_TASK = (
    PORTABLE_MANISKILL_DT_DIR / "so101_arm_with_two_cameras" / "pick_jiaobang.py"
)
PORTABLE_MANISKILL_RM75_URDF = (
    PORTABLE_MANISKILL_ROOT / "assets" / "robots" / "RM75_gripper" / "RM75-B" / "urdf" / "RM75-B.urdf"
)
PORTABLE_MANISKILL_RM75_PLANNING_URDF = PORTABLE_MANISKILL_RM75_URDF.with_name("RM75-B.planning.tiny.urdf")
PORTABLE_MANISKILL_RM75_SRDF = PORTABLE_MANISKILL_RM75_URDF.with_name("RM75-B.permissive.srdf")
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
_ORIGINAL_PLAN_RETURN_TO_START_JOINT_CUROBO = None
_ORIGINAL_PROFILE_PLAN_GOALSET_TO_POSES = None
_ORIGINAL_PROFILE_PLAN_BATCH_START_GOAL_PAIRS = None
_ORIGINAL_ATTACH_TRANSPORT_PAYLOAD_TO_CUROBO = None
_ORIGINAL_PLAN_SHORT_TCP_UP_AXIS_LIFT_IK = None
_ORIGINAL_BUILD_HOVER_POSE = None
_ORIGINAL_PROFILE_STAGE = None
_ORIGINAL_EXECUTE_POSE_PATH_STAGE = None
_ORIGINAL_EXECUTE_JOINT_PATH_STAGE = None
_ORIGINAL_PLAN_JOINT_PATH = None
_ORIGINAL_REALMAN_SET_GRIPPER = None
_ORIGINAL_SYNC_DEMO_GRIPPER_STATE = None
_ORIGINAL_SETTLE_RELEASED_ACTIVE_OBJECT = None
_ORIGINAL_SELECT_RANDOM_CYCLE_TARGET = None
_ORIGINAL_MAKE_IK_PRESELECTED_GRASP_SUCCESS = None
_ORIGINAL_BUILD_DIRECT_GRASP_CANDIDATES = None
_ORIGINAL_CREATE_DEMO = None
_ORIGINAL_MPLIB_COLLISION_DETECTION = None
_ORIGINAL_CUROBO_SOLVE_IK = None
_ORIGINAL_CUROBO_SOLVE_BATCH_START_GOAL_IK = None
_ORIGINAL_CUROBO_SOLVE_BATCH_START_GOAL_IK_CUDA_GRAPH = None


def _jimu_to_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return np.asarray(value.numpy())
        return np.asarray(value)
    except Exception:
        return None


def _jimu_result_debug_float(result, key: str) -> float | None:
    debug = getattr(result, "debug", None)
    if isinstance(debug, dict):
        value = debug.get(key)
        try:
            if value is not None:
                return float(value)
        except Exception:
            pass
    raw = getattr(result, "raw_result", None)
    raw_value = getattr(raw, key, None)
    arr = _jimu_to_numpy(raw_value)
    if arr is None or arr.size <= 0:
        return None
    try:
        return float(arr.reshape(-1)[0])
    except Exception:
        return None


def _jimu_extract_near_ik_solution(raw_result, start_q, *, batch_index: int | None = None) -> np.ndarray | None:
    solution = getattr(raw_result, "solution", None)
    arr = _jimu_to_numpy(solution)
    if arr is None or arr.size <= 0:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[-1] < 7:
        return None
    if batch_index is not None and arr.ndim >= 3 and int(batch_index) < arr.shape[0]:
        arr = arr[int(batch_index)]
    elif batch_index is not None and arr.ndim == 2 and arr.shape[0] > int(batch_index):
        arr = arr[int(batch_index) : int(batch_index) + 1]
    flat = arr.reshape(-1, arr.shape[-1])[:, :7]
    finite = np.all(np.isfinite(flat), axis=1)
    flat = flat[finite]
    if flat.size <= 0:
        return None
    try:
        ref = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
        if ref.size == 7 and np.all(np.isfinite(ref)):
            idx = int(np.argmin(np.linalg.norm(flat - ref.reshape(1, 7), axis=1)))
            return flat[idx].astype(np.float32)
    except Exception:
        pass
    return flat[0].astype(np.float32)


def _jimu_maybe_accept_near_ik_result(result, start_q, args, *, batch_index: int | None = None):
    if result is None or bool(getattr(result, "success", False)):
        return result
    if not bool(getattr(args, "jimu_near_ik_fallback", True)):
        return result
    pos_err = _jimu_result_debug_float(result, "position_error")
    rot_err = _jimu_result_debug_float(result, "rotation_error")
    pos_thr = float(getattr(args, "jimu_near_ik_position_threshold", 1.0e-4) or 0.0)
    rot_thr = float(getattr(args, "jimu_near_ik_rotation_threshold", 1.0e-3) or 0.0)
    if pos_err is None or rot_err is None:
        return result
    if not (np.isfinite(pos_err) and np.isfinite(rot_err)):
        return result
    if pos_err > pos_thr or rot_err > rot_thr:
        return result
    q = _jimu_extract_near_ik_solution(getattr(result, "raw_result", None), start_q, batch_index=batch_index)
    if q is None:
        return result
    result.success = True
    result.status = "SuccessNearIK"
    result.goal_joint = np.asarray(q, dtype=np.float32).reshape(-1)[:7]
    debug = getattr(result, "debug", None)
    if not isinstance(debug, dict):
        debug = {}
        result.debug = debug
    debug["jimu_near_ik_fallback"] = True
    debug["jimu_near_ik_position_threshold"] = float(pos_thr)
    debug["jimu_near_ik_rotation_threshold"] = float(rot_thr)
    return result


def _jimu_motiongen_capture_args(planner):
    return getattr(planner, "_jimu_motiongen_capture_args", None)


def _jimu_motiongen_capture_label(planner) -> str:
    return str(getattr(planner, "_jimu_motiongen_capture_label", "") or "")


def _jimu_linear_transport_fallback_allowed(planner) -> bool:
    if not bool(getattr(planner, "_jimu_linear_transport_eval_active", False)):
        return False
    args = _jimu_motiongen_capture_args(planner)
    if not bool(getattr(args, "jimu_linear_joint_transport_fallback", True) if args is not None else True):
        return False
    label = _jimu_motiongen_capture_label(planner)
    return "joint_transport_hover_pairs" in label


def _jimu_linear_joint_path(start_q, goal_q, *, step_rad: float) -> np.ndarray | None:
    try:
        start = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
        goal = np.asarray(goal_q, dtype=np.float32).reshape(-1)[:7]
    except Exception:
        return None
    if start.size != 7 or goal.size != 7 or not (np.all(np.isfinite(start)) and np.all(np.isfinite(goal))):
        return None
    max_delta = float(np.max(np.abs(goal - start)))
    step = float(max(step_rad, 1.0e-4))
    count = max(2, int(np.ceil(max_delta / step)) + 1)
    return np.linspace(start, goal, count, dtype=np.float32)


def _jimu_validate_linear_joint_path(planner, q_path: np.ndarray, args) -> bool:
    if not bool(getattr(args, "jimu_linear_joint_transport_check_start_state", True) if args is not None else True):
        return True
    for idx, q in enumerate(np.asarray(q_path, dtype=np.float32).reshape(-1, 7)):
        try:
            ok, status = planner.check_start_state(q)
        except Exception as exc:
            print(f"[jimu-curobo] linear transport fallback start-state check failed at waypoint {idx}: {exc}")
            return False
        if not bool(ok):
            print(
                "[jimu-curobo] linear transport fallback rejected by cuRobo start-state check: "
                f"waypoint={idx}, status={status}"
            )
            return False
    return True


def _jimu_make_linear_transport_result(planner, failed_result, start_q, goal_q):
    args = _jimu_motiongen_capture_args(planner)
    step_rad = float(getattr(args, "jimu_linear_joint_transport_step_rad", 0.035) if args is not None else 0.035)
    q_path = _jimu_linear_joint_path(start_q, goal_q, step_rad=step_rad)
    if q_path is None:
        return failed_result
    if not _jimu_validate_linear_joint_path(planner, q_path, args):
        return failed_result
    result_cls = failed_result.__class__
    debug = dict(getattr(failed_result, "debug", {}) or {})
    source_status = str(getattr(failed_result, "status", None))
    if "INVALID_START_STATE_WORLD_COLLISION" in source_status:
        label = _jimu_motiongen_capture_label(planner)
        diag = _jimu_start_collision_diagnosis(
            planner,
            args,
            [{"start_q": start_q, "label": label}],
            label,
            getattr(planner, "_jimu_motiongen_capture_disabled_links", None),
        )
        if isinstance(diag, dict):
            planner._jimu_start_collision_diag = diag
            debug["source_start_collision_diag"] = diag
            print(
                f"[jimu-curobo-diag] {label or 'unknown'} fallback source start collision: "
                f"status={diag.get('status')} "
                f"attached_bottom_z={diag.get('attached_sphere_bottom_z')} "
                f"valid_if_remove={diag.get('valid_after_removing')} "
                f"obstacles={diag.get('world_obstacle_names')}"
            )
    debug.update(
        {
            "jimu_linear_joint_transport_fallback": True,
            "source_status": source_status,
            "waypoint_count": int(q_path.shape[0]),
            "max_joint_delta": float(np.max(np.abs(q_path[-1] - q_path[0]))),
            "step_rad": float(step_rad),
        }
    )
    print(
        "[jimu-curobo] accepted linear joint transport fallback: "
        f"label={_jimu_motiongen_capture_label(planner)}, "
        f"source_status={source_status}, "
        f"waypoints={int(q_path.shape[0])}, "
        f"max_delta={float(np.max(np.abs(q_path[-1] - q_path[0]))):.3f}rad"
    )
    return result_cls(
        success=True,
        status="JimuLinearJointTransportFallback",
        goal_joint=np.asarray(q_path[-1], dtype=np.float32),
        joint_path=np.asarray(q_path, dtype=np.float32),
        solve_time=float(getattr(failed_result, "solve_time", 0.0) or 0.0),
        ik_time=float(getattr(failed_result, "ik_time", 0.0) or 0.0),
        trajopt_time=float(getattr(failed_result, "trajopt_time", 0.0) or 0.0),
        raw_result=getattr(failed_result, "raw_result", None),
        debug=debug,
    )


def _install_jimu_near_ik_fallback(args: argparse.Namespace | None = None) -> None:
    global _ORIGINAL_CUROBO_SOLVE_IK
    global _ORIGINAL_CUROBO_SOLVE_BATCH_START_GOAL_IK
    global _ORIGINAL_CUROBO_SOLVE_BATCH_START_GOAL_IK_CUDA_GRAPH
    if not bool(getattr(args, "jimu_near_ik_fallback", True) if args is not None else True):
        return
    planner_cls = getattr(direct.curobo_wrapper, "RM75CuRoboPlanner", None)
    if planner_cls is None:
        return
    if _ORIGINAL_CUROBO_SOLVE_IK is not None:
        return
    _ORIGINAL_CUROBO_SOLVE_IK = planner_cls.solve_ik
    _ORIGINAL_CUROBO_SOLVE_BATCH_START_GOAL_IK = planner_cls.solve_batch_start_goal_ik
    _ORIGINAL_CUROBO_SOLVE_BATCH_START_GOAL_IK_CUDA_GRAPH = planner_cls._solve_batch_start_goal_ik_cuda_graph_once

    def solve_ik_jimu(self, start_q, goal_pose, *call_args, **kwargs):
        result = _ORIGINAL_CUROBO_SOLVE_IK(self, start_q, goal_pose, *call_args, **kwargs)
        return _jimu_maybe_accept_near_ik_result(result, start_q, args)

    def solve_batch_start_goal_ik_jimu(self, start_qs, goal_poses, *call_args, **kwargs):
        results = _ORIGINAL_CUROBO_SOLVE_BATCH_START_GOAL_IK(self, start_qs, goal_poses, *call_args, **kwargs)
        patched = []
        start_q_list = list(start_qs) if start_qs is not None else []
        for idx, result in enumerate(list(results or [])):
            start_q = start_q_list[idx] if idx < len(start_q_list) else None
            batch_index = None
            debug = getattr(result, "debug", None)
            if isinstance(debug, dict):
                try:
                    batch_index = int(debug.get("batch_index"))
                except Exception:
                    batch_index = idx
            else:
                batch_index = idx
            patched.append(_jimu_maybe_accept_near_ik_result(result, start_q, args, batch_index=batch_index))
        return patched

    def solve_batch_start_goal_ik_cuda_graph_once_jimu(self, start_qs, goal_poses, *call_args, **kwargs):
        results = _ORIGINAL_CUROBO_SOLVE_BATCH_START_GOAL_IK_CUDA_GRAPH(self, start_qs, goal_poses, *call_args, **kwargs)
        patched = []
        start_q_list = list(start_qs) if start_qs is not None else []
        for idx, result in enumerate(list(results or [])):
            start_q = start_q_list[idx] if idx < len(start_q_list) else None
            batch_index = None
            debug = getattr(result, "debug", None)
            if isinstance(debug, dict):
                try:
                    batch_index = int(debug.get("batch_index"))
                except Exception:
                    batch_index = idx
            else:
                batch_index = idx
            patched.append(_jimu_maybe_accept_near_ik_result(result, start_q, args, batch_index=batch_index))
        return patched

    planner_cls.solve_ik = solve_ik_jimu
    planner_cls.solve_batch_start_goal_ik = solve_batch_start_goal_ik_jimu
    planner_cls._solve_batch_start_goal_ik_cuda_graph_once = solve_batch_start_goal_ik_cuda_graph_once_jimu
    print(
        "[jimu config] cuRobo near-IK fallback enabled: "
        f"pos<={float(getattr(args, 'jimu_near_ik_position_threshold', 1.0e-4) or 0.0):.6g}m, "
        f"rot<={float(getattr(args, 'jimu_near_ik_rotation_threshold', 1.0e-3) or 0.0):.6g}rad"
    )


def _restore_jimu_near_ik_fallback() -> None:
    global _ORIGINAL_CUROBO_SOLVE_IK
    global _ORIGINAL_CUROBO_SOLVE_BATCH_START_GOAL_IK
    global _ORIGINAL_CUROBO_SOLVE_BATCH_START_GOAL_IK_CUDA_GRAPH
    planner_cls = getattr(direct.curobo_wrapper, "RM75CuRoboPlanner", None)
    if planner_cls is not None and _ORIGINAL_CUROBO_SOLVE_IK is not None:
        planner_cls.solve_ik = _ORIGINAL_CUROBO_SOLVE_IK
        planner_cls.solve_batch_start_goal_ik = _ORIGINAL_CUROBO_SOLVE_BATCH_START_GOAL_IK
        planner_cls._solve_batch_start_goal_ik_cuda_graph_once = _ORIGINAL_CUROBO_SOLVE_BATCH_START_GOAL_IK_CUDA_GRAPH
    _ORIGINAL_CUROBO_SOLVE_IK = None
    _ORIGINAL_CUROBO_SOLVE_BATCH_START_GOAL_IK = None
    _ORIGINAL_CUROBO_SOLVE_BATCH_START_GOAL_IK_CUDA_GRAPH = None


class _JimuNoopMplibRobot:
    def __init__(self, demo=None):
        self.demo = demo
        self._qpos = np.zeros(7, dtype=np.float32)

    def get_qpos(self):
        if self.demo is not None:
            try:
                return np.asarray(self.demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
            except Exception:
                pass
        return self._qpos.copy()

    def set_qpos(self, qpos, *_args, **_kwargs):
        self._qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)[:7].copy()


class _JimuNoopMplibPlanner:
    """Small compatibility shim for legacy demo-planner hooks.

    Jimu portable uses cuRobo for planning.  The old demo wrapper still expects
    a `demo.planner` object for obstacle bookkeeping and diagnostics, but
    constructing `mplib.Planner` is no longer needed and can segfault in this
    environment.
    """

    def __init__(self, demo=None):
        self.demo = demo
        self.robot = _JimuNoopMplibRobot(demo)
        self.normal_objects: dict[str, object] = {}
        self.joint_types = []
        self.joint_limits = np.zeros((0, 2), dtype=np.float32)

    def set_base_pose(self, *_args, **_kwargs):
        return None

    def set_normal_object(self, name, obj):
        self.normal_objects[str(name)] = obj
        return None

    def remove_normal_object(self, name):
        self.normal_objects.pop(str(name), None)
        return None

    def check_for_self_collision(self, *args, **kwargs):
        return []

    def check_for_env_collision(self, *args, **kwargs):
        return []

    def update_attached_box(self, *args, **kwargs):
        return None

    def plan_qpos_to_pose(self, *args, **kwargs):
        return {"status": "JimuNoMplibDisabled", "position": []}

    def plan_qpos_to_qpos(self, *args, **kwargs):
        return {"status": "JimuNoMplibDisabled", "position": []}

    def IK(self, *args, **kwargs):
        return "JimuNoMplibDisabled", []


class _JimuNoopFclBox:
    def __init__(self, size):
        self.size = np.asarray(size, dtype=np.float32).reshape(-1).copy()


class _JimuNoopFclCollisionObject:
    def __init__(self, geometry, position=None, quaternion=None):
        self.geometry = geometry
        self.position = np.asarray(position if position is not None else [0.0, 0.0, 0.0], dtype=np.float32).copy()
        self.quaternion = np.asarray(quaternion if quaternion is not None else [1.0, 0.0, 0.0, 0.0], dtype=np.float32).copy()


class _JimuNoopFcl:
    Box = _JimuNoopFclBox
    CollisionObject = _JimuNoopFclCollisionObject


def _install_jimu_no_mplib_collision_detection(args: argparse.Namespace | None = None) -> None:
    global _ORIGINAL_MPLIB_COLLISION_DETECTION
    if not bool(getattr(args, "jimu_disable_mplib_demo_planner", True) if args is not None else True):
        return
    try:
        import mplib
    except Exception:
        return
    if getattr(mplib, "_jimu_no_mplib_collision_detection_installed", False):
        return
    _ORIGINAL_MPLIB_COLLISION_DETECTION = getattr(mplib, "collision_detection", None)
    fake_collision_detection = types.SimpleNamespace(fcl=_JimuNoopFcl)
    mplib.collision_detection = fake_collision_detection
    sys.modules["mplib.collision_detection"] = fake_collision_detection
    mplib._jimu_no_mplib_collision_detection_installed = True
    print("[jimu config] disabled legacy mplib FCL collision objects for scene obstacles")


def _restore_jimu_no_mplib_collision_detection() -> None:
    global _ORIGINAL_MPLIB_COLLISION_DETECTION
    try:
        import mplib
    except Exception:
        _ORIGINAL_MPLIB_COLLISION_DETECTION = None
        return
    if _ORIGINAL_MPLIB_COLLISION_DETECTION is not None:
        mplib.collision_detection = _ORIGINAL_MPLIB_COLLISION_DETECTION
        sys.modules["mplib.collision_detection"] = _ORIGINAL_MPLIB_COLLISION_DETECTION
    else:
        try:
            delattr(mplib, "collision_detection")
        except Exception:
            pass
        sys.modules.pop("mplib.collision_detection", None)
    try:
        delattr(mplib, "_jimu_no_mplib_collision_detection_installed")
    except Exception:
        pass
    _ORIGINAL_MPLIB_COLLISION_DETECTION = None


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


def _prepend_package_path(package: Any, path: Path) -> None:
    if not path.exists() or not hasattr(package, "__path__"):
        return
    path_str = str(path)
    existing = [str(item) for item in list(package.__path__)]
    existing = [item for item in existing if item != path_str]
    new_paths = [path_str, *existing]
    try:
        package.__path__[:] = new_paths
    except Exception:
        package.__path__ = new_paths


def install_portable_maniskill_env(enabled: bool = True) -> bool:
    if not enabled:
        return False
    if not PORTABLE_MANISKILL_PICK_TASK.exists():
        print(f"[jimu config] portable ManiSkill env not found: {PORTABLE_MANISKILL_PICK_TASK}")
        return False

    try:
        import gymnasium as gym
        import mani_skill.agents.robots as robots_pkg
        import mani_skill.envs.tasks.digital_twins as dt_pkg
    except Exception as exc:
        print(f"[jimu config] failed to import ManiSkill packages before portable env install: {exc}")
        return False

    _prepend_package_path(robots_pkg, PORTABLE_MANISKILL_ROBOTS_DIR)
    _prepend_package_path(dt_pkg, PORTABLE_MANISKILL_DT_DIR)

    for module_name in (
        "mani_skill.agents.robots.realman.realman_with_gripper",
        "mani_skill.agents.robots.realman",
    ):
        module = sys.modules.get(module_name)
        module_file = Path(str(getattr(module, "__file__", "") or "")).expanduser()
        if module is not None and PORTABLE_MANISKILL_ROOT not in module_file.parents:
            sys.modules.pop(module_name, None)
    if hasattr(robots_pkg, "realman"):
        try:
            delattr(robots_pkg, "realman")
        except Exception:
            pass

    importlib.import_module("mani_skill.agents.robots.realman")

    module_name = "jimu_portable_pick_jiaobang_env"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, PORTABLE_MANISKILL_PICK_TASK)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load portable ManiSkill env spec: {PORTABLE_MANISKILL_PICK_TASK}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if "Two_finger_PickJiaobang-v1" not in gym.envs.registry:
        raise RuntimeError("portable ManiSkill task did not register Two_finger_PickJiaobang-v1")
    print(f"[jimu config] portable ManiSkill task registered: {PORTABLE_MANISKILL_PICK_TASK}")
    if PORTABLE_MANISKILL_RM75_URDF.exists():
        print(f"[jimu config] portable RM75 asset: {PORTABLE_MANISKILL_RM75_URDF}")
    else:
        print("[jimu config] portable RM75 asset missing; falling back to ManiSkill ASSET_DIR")
    return True


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
    pending_retry = None
    if isinstance(scene_capture_cache, dict):
        pending_retry = scene_capture_cache.pop("_jimu_retry_target_after_source_swap", None)
    if isinstance(pending_retry, dict):
        retry_target = direct.curobo_wrapper.normalize_object_name(pending_retry.get("target_name"))
        if retry_target is not None and not _is_jimu_role_placed(scene_capture_cache, retry_target):
            try:
                failed_targets_this_cycle.discard(retry_target)
            except Exception:
                pass
            try:
                if deferred_failed_targets is not None:
                    deferred_failed_targets.discard(retry_target)
            except Exception:
                pass
            pool = direct.targeted._random_target_pool_for_cycle(
                base_args,
                cycle_object_sequence,
                scene_capture_cache,
                available_rule_names,
                cycle_idx,
            )
            if retry_target not in pool:
                pool = [retry_target, *list(pool or [])]
            print(
                "[jimu-source-retry] retrying same logical target after tray-source swap: "
                f"target={retry_target}, donor={pending_retry.get('donor_role')}, "
                f"slot={pending_retry.get('donor_slot_index')}"
            )
            return retry_target, pool, [retry_target]
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


def _jimu_mesh_file_override(args: argparse.Namespace | None = None) -> str | None:
    raw = str(getattr(args, "jimu_mesh_file", "") if args is not None else "").strip()
    if raw:
        return str(Path(raw).expanduser())
    if PORTABLE_DEFAULT_MESH_FILE.exists():
        return str(PORTABLE_DEFAULT_MESH_FILE)
    return None


def _jimu_sim_asset_file_override(args: argparse.Namespace | None = None) -> str | None:
    raw = str(getattr(args, "jimu_sim_asset_file", "") if args is not None else "").strip()
    if raw:
        return str(Path(raw).expanduser())
    if PORTABLE_DEFAULT_SIM_ASSET_FILE.exists():
        return str(PORTABLE_DEFAULT_SIM_ASSET_FILE)
    return None


def _jimu_tray_mesh_file_override(args: argparse.Namespace | None = None) -> str | None:
    raw = str(getattr(args, "jimu_tray_mesh_file", "") if args is not None else "").strip()
    if raw:
        return str(Path(raw).expanduser())
    if PORTABLE_DEFAULT_TRAY_MESH_FILE.exists():
        return str(PORTABLE_DEFAULT_TRAY_MESH_FILE)
    return None


def _jimu_base_assembly_mesh_file_override(args: argparse.Namespace | None = None) -> str | None:
    raw = str(getattr(args, "jimu_base_assembly_mesh_file", "") if args is not None else "").strip()
    if raw:
        return str(Path(raw).expanduser())
    if PORTABLE_DEFAULT_BASE_ASSEMBLY_MESH_FILE.exists():
        return str(PORTABLE_DEFAULT_BASE_ASSEMBLY_MESH_FILE)
    return None


def _jimu_tray_mesh_scale(args: argparse.Namespace | None = None) -> float:
    scale = float(getattr(args, "jimu_tray_mesh_scale", 0.01) if args is not None else 0.01)
    if not np.isfinite(scale) or scale <= 1e-9:
        return 0.01
    return scale


def _jimu_base_assembly_mesh_scale(args: argparse.Namespace | None = None) -> float:
    scale = float(getattr(args, "jimu_base_assembly_mesh_scale", 1.0) if args is not None else 1.0)
    if not np.isfinite(scale) or scale <= 1e-9:
        return 1.0
    return scale


def _jimu_tray_bounds_scaled(args: argparse.Namespace | None = None) -> np.ndarray:
    mesh_file = _jimu_tray_mesh_file_override(args)
    scale = _jimu_tray_mesh_scale(args)
    if mesh_file:
        try:
            loaded = trimesh.load(Path(mesh_file).expanduser(), force="scene")
            bounds = np.asarray(loaded.bounds, dtype=np.float32)
            if bounds.shape == (2, 3) and np.all(np.isfinite(bounds)):
                return (bounds * float(scale)).astype(np.float32)
        except Exception as exc:
            print(f"[jimu config] warning: failed to read tray bounds, using fallback: {exc}")
    return np.asarray([[0.0, 0.0, -0.005], [0.2425, 0.2000, 0.0200]], dtype=np.float32)


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


def _horizontal_axis_score(vec: np.ndarray) -> float:
    arr = np.asarray(vec, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return -float("inf")
    horizontal = arr.copy()
    horizontal[2] = 0.0
    return float(np.linalg.norm(horizontal) / norm)


def _preferred_horizontal_axis(
    vectors: list[np.ndarray],
    *,
    snap_cardinal: bool,
    min_preferred_score: float = 0.35,
) -> np.ndarray | None:
    best_axis = None
    best_score = -float("inf")
    for vec in vectors:
        axis = _horizontal_axis(vec, snap_cardinal=snap_cardinal)
        score = _horizontal_axis_score(vec)
        if axis is None:
            continue
        if score >= min_preferred_score:
            return axis
        if score > best_score:
            best_score = score
            best_axis = axis
    return best_axis


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
        x_axis = _preferred_horizontal_axis(
            [T_base_obj[:3, 0], T_base_obj[:3, 2]],
            snap_cardinal=snap_cardinal,
        )
        if x_axis is None:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        y_axis = world_up
        z_axis = _normalize_vec(np.cross(x_axis, y_axis))
        if z_axis is None:
            z_axis = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        out[:3, :3] = _rotation_from_xy_z(x_axis, y_axis, z_axis)
        return out

    y_axis = None
    y_score = _horizontal_axis_score(T_base_obj[:3, 1])
    if y_score >= 0.35:
        y_axis = _horizontal_axis(T_base_obj[:3, 1], snap_cardinal=snap_cardinal)
    if y_axis is None:
        best_axis = None
        best_score = -float("inf")
        for axis_idx in (0, 1, 2):
            axis = _horizontal_axis(T_base_obj[:3, axis_idx], snap_cardinal=snap_cardinal)
            if axis is None:
                continue
            score = _horizontal_axis_score(T_base_obj[:3, axis_idx])
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
    mesh_file = _jimu_mesh_file_override(args)
    sim_asset_file = _jimu_sim_asset_file_override(args)
    if mesh_file:
        base_spec = replace(
            base_spec,
            mesh_file=mesh_file,
        )
        object_specs.OBJECT_SPECS[JIMU_PROVIDER_OBJECT_NAME] = base_spec
    role_base_spec = base_spec
    if sim_asset_file:
        role_base_spec = replace(
            role_base_spec,
            sim_asset_file=sim_asset_file,
            sim_asset_scale=1.0,
        )
    local_rotation_offset = _jimu_cad_to_sim_rpy_deg(args)
    for role in JIMU_DERIVED_ROLE_SET:
        object_specs.OBJECT_SPECS[role] = replace(
            role_base_spec,
            name=role,
            grounding_prompt="small square plastic building block.",
            foundationpose_local_rotation_offset_deg=local_rotation_offset,
        )
        object_specs.OBJECT_NAME_ALIASES[role] = role

    tray_mesh_file = _jimu_tray_mesh_file_override(args)
    if tray_mesh_file:
        object_specs.OBJECT_SPECS[JIMU_TRAY_OBJECT_NAME] = object_specs.ObjectSpec(
            name=JIMU_TRAY_OBJECT_NAME,
            grounding_prompt="gray plastic tray with slots for red building plates.",
            mesh_file=tray_mesh_file,
            mesh_scale=_jimu_tray_mesh_scale(args),
            sim_asset_file=tray_mesh_file,
            sim_asset_scale=_jimu_tray_mesh_scale(args),
            real_longest_axis_m=None,
            fixed_goal_joints_deg=object_specs.DEFAULT_FIXED_GOAL_JOINTS_DEG,
            grasp_mode="topdown_long_axis",
            scene_obstacle_box_scale=1.0,
        )
        object_specs.OBJECT_NAME_ALIASES[JIMU_TRAY_OBJECT_NAME] = JIMU_TRAY_OBJECT_NAME
        object_specs.OBJECT_NAME_ALIASES["jimu_tray"] = JIMU_TRAY_OBJECT_NAME
        object_specs.OBJECT_NAME_ALIASES["liaoban"] = JIMU_TRAY_OBJECT_NAME

    base_assembly_mesh_file = _jimu_base_assembly_mesh_file_override(args)
    if base_assembly_mesh_file:
        object_specs.OBJECT_SPECS[JIMU_BASE_ASSEMBLY_OBJECT_NAME] = object_specs.ObjectSpec(
            name=JIMU_BASE_ASSEMBLY_OBJECT_NAME,
            grounding_prompt="red cross shaped base assembly made of five square plastic building plates.",
            mesh_file=base_assembly_mesh_file,
            mesh_scale=_jimu_base_assembly_mesh_scale(args),
            sim_asset_file=base_assembly_mesh_file,
            sim_asset_scale=_jimu_base_assembly_mesh_scale(args),
            real_longest_axis_m=None,
            fixed_goal_joints_deg=object_specs.DEFAULT_FIXED_GOAL_JOINTS_DEG,
            grasp_mode="topdown_long_axis",
            scene_obstacle_box_scale=1.0,
        )
        object_specs.OBJECT_NAME_ALIASES[JIMU_BASE_ASSEMBLY_OBJECT_NAME] = JIMU_BASE_ASSEMBLY_OBJECT_NAME
        object_specs.OBJECT_NAME_ALIASES["jimu_base"] = JIMU_BASE_ASSEMBLY_OBJECT_NAME
        object_specs.OBJECT_NAME_ALIASES["base_assembly"] = JIMU_BASE_ASSEMBLY_OBJECT_NAME


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


def _install_jimu_no_mplib_demo_planner(planner_mod, args: argparse.Namespace | None = None) -> None:
    if not bool(getattr(args, "jimu_disable_mplib_demo_planner", True) if args is not None else True):
        return
    if bool(getattr(planner_mod, "_jimu_no_mplib_demo_planner_installed", False)):
        return
    original_cls = getattr(planner_mod, "RM75JiaobangPickMove", None)
    if original_cls is None:
        return

    class RM75JiaobangPickMoveNoMplib(original_cls):
        def __init__(self, env, urdf_path: str, srdf_path: str, args):
            self.env = env
            self.base_env = env.unwrapped
            self.robot = self.base_env.agent.robot
            self.tcp = self.base_env.agent.tcp
            self.tip_link_name = "gripper_tcp"
            self.action_dim = int(np.prod(env.action_space.shape))
            self.control_timestep = 1.0 / 20.0
            self.args = args

            self.link_names = planner_mod.parse_link_names(urdf_path)
            self.active_joint_names = [j.get_name() for j in self.robot.get_active_joints()]
            self.arm_indices = [self.active_joint_names.index(n) for n in planner_mod.ARM_JOINT_NAMES]

            self.planner = _JimuNoopMplibPlanner(self)
            self.grasp_pose_visual = None
            self.step_counter = 0
            self.left_finger_link = None
            self.right_finger_link = None
            self.last_obs = None
            self.last_reward = None
            self.last_terminated = None
            self.last_truncated = None
            self.last_info = {}
            self.refresh_runtime_handles(rebuild_visual=True)

        def step_and_render(self, action, tag=""):
            try:
                torch = importlib.import_module("torch")
                device = getattr(self.base_env, "device", None)
                if not torch.is_tensor(action):
                    action = torch.as_tensor(action, device=device, dtype=torch.float32)
                elif device is not None:
                    action = action.to(device=device, dtype=torch.float32)
            except Exception:
                pass
            return super().step_and_render(action, tag=tag)

    RM75JiaobangPickMoveNoMplib.__name__ = "RM75JiaobangPickMoveNoMplib"
    RM75JiaobangPickMoveNoMplib.__qualname__ = "RM75JiaobangPickMoveNoMplib"
    planner_mod.RM75JiaobangPickMove = RM75JiaobangPickMoveNoMplib
    planner_mod._jimu_no_mplib_demo_planner_installed = True
    planner_mod._jimu_original_RM75JiaobangPickMove = original_cls
    print("[jimu config] disabled legacy mplib demo planner; cuRobo remains the motion planner")


def create_demo_jimu_no_mplib(args, bridge_mod, planner_mod, scene_capture_cache=None):
    if _ORIGINAL_CREATE_DEMO is None:
        raise RuntimeError("Jimu create_demo wrapper was installed before original create_demo was captured")
    _install_jimu_no_mplib_demo_planner(planner_mod, args)
    _install_jimu_no_mplib_collision_detection(args)
    env, demo = _ORIGINAL_CREATE_DEMO(args, bridge_mod, planner_mod, scene_capture_cache=scene_capture_cache)
    _apply_jimu_sim_start_qpos(demo, args)
    _render_jimu_tray_visuals(env, demo, bridge_mod, args, scene_capture_cache)
    return env, demo


def _apply_jimu_sim_start_qpos(demo, args: argparse.Namespace | None) -> None:
    raw = getattr(args, "jimu_sim_start_joints_deg", None) if args is not None else None
    if raw is None:
        return
    q_deg = np.asarray(raw, dtype=np.float32).reshape(-1)
    if q_deg.size != 7:
        raise ValueError("--jimu-sim-start-joints-deg must contain exactly 7 joint angles")
    if not np.all(np.isfinite(q_deg)):
        raise ValueError("--jimu-sim-start-joints-deg contains non-finite values")
    q_rad = np.deg2rad(q_deg).astype(np.float32)
    direct.targeted.base.sync_demo_arm_qpos(demo, q_rad)
    setattr(demo, "_jimu_sim_start_qpos", q_rad.copy())
    if args is not None:
        args._jimu_sim_start_joints_deg_applied = q_deg.astype(float).tolist()
    print(f"[jimu config] applied sim start joints deg={np.round(q_deg, 3).tolist()}")
    if _jimu_pregrasp_partial_open_enabled(args):
        sim_partial = _jimu_pregrasp_partial_open_sim_value(args)
        _jimu_set_sim_gripper_visual(demo, sim_partial, args, label="initial_pregrasp_partial")
        print(
            "[jimu gripper] initialized simulated gripper to pregrasp partial open: "
            f"sim={sim_partial:.3f}, real_equiv={_jimu_pregrasp_partial_open_value(args):.3f}"
        )


def plan_joint_path_jimu_no_mplib(demo, q_goal, *args, **kwargs):
    demo_args = getattr(demo, "args", None)
    role = direct.curobo_wrapper.normalize_object_name(getattr(demo_args, "object_name", None))
    is_roof_triangle = "roof_triangle" in str(role or "")
    if bool(getattr(demo_args, "jimu_disable_mplib_demo_planner", True)) and is_roof_triangle:
        label = str(kwargs.get("label", "joint_goal"))
        print(f"[jimu config] blocked legacy plan_joint_path fallback for {label}; cuRobo failure is not bypassed")
        return None
    if _ORIGINAL_PLAN_JOINT_PATH is None:
        return None
    return _ORIGINAL_PLAN_JOINT_PATH(demo, q_goal, *args, **kwargs)


def _jimu_cached_visual_world_pose(demo, bridge_mod, args, cache_entry: dict, T_base_cam: np.ndarray) -> np.ndarray | None:
    try:
        # Jimu assembly anchors and tray slot visuals are already defined in
        # robot-base coordinates.  Mapping them through the generic
        # FoundationPose helper would apply the tabletop safety clamp using
        # the tray mesh bounds, which incorrectly lifts the whole tray.
        unclamped = _jimu_base_pose_to_pick_world_no_table_clamp(demo, bridge_mod, args, cache_entry)
        if unclamped is not None:
            return unclamped
        if cache_entry.get("T_world_obj") is not None:
            return np.asarray(cache_entry["T_world_obj"], dtype=np.float32).reshape(4, 4)
        T_cam_obj = np.asarray(cache_entry["T_cam_obj"], dtype=np.float32).reshape(4, 4)
        object_args = cache_entry.get("object_args")
        if object_args is None:
            object_args = args
        return bridge_mod.map_camera_pose_to_pick_world(T_cam_obj, T_base_cam, demo.env, object_args)
    except Exception as exc:
        print(f"[jimu visual] failed to map visual pose for {cache_entry.get('object_name', 'object')}: {exc}")
        return None


def _upsert_jimu_visual_actor(env, bridge_mod, cache_entry: dict, actor_name: str, T_world_obj: np.ndarray, *, color=None):
    object_args = cache_entry.get("object_args")
    if object_args is None:
        return None
    asset_file = str(Path(str(getattr(object_args, "sim_asset_file", "") or getattr(object_args, "mesh_file", ""))).expanduser())
    if not asset_file:
        return None
    asset_scale = float(getattr(object_args, "sim_asset_scale", 0.0) or getattr(object_args, "mesh_scale", 1.0) or 1.0)
    actor = None
    find_actor = getattr(direct.targeted.base, "find_named_scene_actor", None)
    if callable(find_actor):
        try:
            actor = find_actor(env, actor_name)
        except Exception:
            actor = None
    if actor is None:
        actor = direct.targeted.base.build_visual_obstacle_actor(
            env,
            asset_file,
            asset_scale,
            actor_name,
            box_size=None,
            color=color,
        )
    pos = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4)[:3, 3].astype(np.float32)
    quat = bridge_mod.mat2quat(np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4)[:3, :3]).astype(np.float32)
    actor.set_pose(direct.targeted.base.Pose.create_from_pq(p=pos, q=quat))
    return actor


def _hide_jimu_visual_actor(actor) -> None:
    if actor is None:
        return
    try:
        hide_pose = getattr(direct, "_hidden_visual_pose", None)
        if callable(hide_pose):
            actor.set_pose(hide_pose())
            return
    except Exception:
        pass
    try:
        actor.set_pose(
            direct.targeted.base.Pose.create_from_pq(
                p=np.asarray([0.0, 0.0, -10.0], dtype=np.float32),
                q=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
        )
    except Exception:
        pass


def _refresh_jimu_visual_render(env, demo) -> None:
    scene = None
    try:
        scene = getattr(getattr(demo, "base_env", None), "scene", None)
        update_render = getattr(scene, "update_render", None)
        if callable(update_render):
            update_render(update_sensors=False, update_human_render_cameras=True)
        elif scene is not None:
            scene.update_render()
    except TypeError:
        try:
            scene.update_render()
        except Exception:
            pass
    except Exception:
        pass
    if str(getattr(getattr(demo, "args", None), "render_mode", "") or "") == "human":
        try:
            env.render()
        except Exception:
            pass


def _sync_jimu_tray_slot_visuals_for_cycle(
    env,
    demo,
    args,
    scene_capture_cache=None,
    *,
    active_name: str | None = None,
) -> None:
    env_unwrapped = getattr(env, "unwrapped", None)
    if env_unwrapped is None:
        return
    slot_actors = getattr(env_unwrapped, "_jimu_tray_visual_slot_actors", None)
    if not isinstance(slot_actors, dict) or not slot_actors:
        return
    hidden_roles: set[str] = set()
    active = direct.curobo_wrapper.normalize_object_name(active_name or getattr(args, "object_name", None))
    if active:
        hidden_roles.add(active)
    objects = {}
    if isinstance(scene_capture_cache, dict):
        objects = scene_capture_cache.get("objects") or {}
    if active and isinstance(objects, dict):
        active_entry = objects.get(active)
        if isinstance(active_entry, dict):
            physical_source = direct.curobo_wrapper.normalize_object_name(active_entry.get("jimu_physical_source_role"))
            if physical_source:
                hidden_roles.add(physical_source)
    if isinstance(objects, dict):
        for role, entry in objects.items():
            role_name = direct.curobo_wrapper.normalize_object_name(role)
            if role_name and isinstance(entry, dict) and bool(entry.get("placed", False)):
                hidden_roles.add(role_name)
    changed = []
    already_hidden = getattr(env_unwrapped, "_jimu_tray_visual_hidden_roles", set())
    if not isinstance(already_hidden, set):
        already_hidden = set(already_hidden or [])
    for role in sorted(hidden_roles):
        actor = slot_actors.get(role)
        if actor is None:
            continue
        _hide_jimu_visual_actor(actor)
        if role not in already_hidden:
            changed.append(role)
    if not changed:
        return
    env_unwrapped._jimu_tray_visual_hidden_roles = already_hidden | set(changed)
    print(f"[jimu visual] hid tray slot visual actor(s) for picked/active role(s): {changed}")
    _refresh_jimu_visual_render(env, demo)


def _render_jimu_tray_visuals(env, demo, bridge_mod, args, scene_capture_cache=None) -> None:
    if not isinstance(scene_capture_cache, dict):
        return
    if not bool(getattr(args, "jimu_render_tray_visual", True)):
        return
    try:
        T_base_cam = np.asarray(scene_capture_cache["T_base_cam"], dtype=np.float32).reshape(4, 4)
    except Exception:
        return
    actors = []
    preview_points: list[np.ndarray] = []
    visual_anchors = dict(scene_capture_cache.get("jimu_visual_anchors", {}) or {})
    tray_name = direct.curobo_wrapper.normalize_object_name(
        getattr(args, "jimu_tray_object_name", JIMU_TRAY_OBJECT_NAME)
    ) or JIMU_TRAY_OBJECT_NAME
    tray_entry = visual_anchors.get(tray_name)
    if isinstance(tray_entry, dict):
        T_world_tray = _jimu_cached_visual_world_pose(demo, bridge_mod, args, tray_entry, T_base_cam)
        if T_world_tray is not None:
            preview_points.append(np.asarray(T_world_tray[:3, 3], dtype=np.float32).reshape(3))
            actor = _upsert_jimu_visual_actor(
                env,
                bridge_mod,
                tray_entry,
                "jimu_visual_liaoban_tray",
                T_world_tray,
            )
            if actor is not None:
                actors.append(actor)
                print(
                    "[jimu visual] rendered tray visual-only actor: "
                    f"p={np.round(T_world_tray[:3, 3], 4).tolist()}"
                )

    if bool(getattr(args, "jimu_render_tray_slot_visuals", True)):
        active_name = direct.curobo_wrapper.normalize_object_name(getattr(args, "object_name", None))
        visual_slots = dict(scene_capture_cache.get("jimu_visual_slots", {}) or {})
        slot_actors = {}
        rendered_slots = 0
        for role, entry in visual_slots.items():
            role_name = direct.curobo_wrapper.normalize_object_name(role)
            if role_name is None or role_name == active_name or not isinstance(entry, dict):
                continue
            T_world_slot = _jimu_cached_visual_world_pose(demo, bridge_mod, args, entry, T_base_cam)
            if T_world_slot is None:
                continue
            preview_points.append(np.asarray(T_world_slot[:3, 3], dtype=np.float32).reshape(3))
            actor = _upsert_jimu_visual_actor(
                env,
                bridge_mod,
                entry,
                f"jimu_visual_tray_slot_{role_name}",
                T_world_slot,
                color=(1.0, 0.02, 0.02, 0.35),
            )
            if actor is not None:
                actors.append(actor)
                slot_actors[role_name] = actor
                rendered_slots += 1
        if rendered_slots:
            print(f"[jimu visual] rendered tray slot plate visual-only actors: {rendered_slots}")
    else:
        slot_actors = {}

    env_unwrapped = getattr(env, "unwrapped", None)
    if env_unwrapped is not None:
        env_unwrapped._jimu_tray_visual_actors = actors
        env_unwrapped._jimu_tray_visual_slot_actors = slot_actors
        env_unwrapped._jimu_tray_visual_hidden_roles = set()
    _sync_jimu_tray_slot_visuals_for_cycle(
        env,
        demo,
        args,
        scene_capture_cache,
        active_name=getattr(args, "object_name", None),
    )
    try:
        scene = getattr(getattr(demo, "base_env", None), "scene", None)
        update_render = getattr(scene, "update_render", None)
        if callable(update_render):
            update_render(update_sensors=False, update_human_render_cameras=True)
    except TypeError:
        try:
            scene.update_render()
        except Exception:
            pass
    except Exception:
        pass
    if str(getattr(args, "render_mode", "") or "") == "human":
        try:
            env.render()
        except Exception:
            pass
    if bool(getattr(args, "jimu_save_scene_preview", True)):
        try:
            path = direct.targeted.base.save_failure_render_image(demo, args, "jimu_scene_preview")
            if path is not None:
                print(f"[jimu visual] saved scene preview image: {path}")
        except Exception as exc:
            print(f"[jimu visual] failed to save scene preview image: {exc}")
        try:
            for item in list(getattr(demo, "scene_obstacles", []) or []):
                T_world_obj = item.get("T_world_obj")
                if T_world_obj is None:
                    continue
                preview_points.append(np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4)[:3, 3])
            active_pose = getattr(getattr(demo, "base_env", None), "obj", None)
            if active_pose is not None:
                pose = active_pose.pose
                preview_points.append(direct.targeted.base.flatten_np(pose.p)[:3].astype(np.float32))
            _save_jimu_scene_preview_views(demo, args, preview_points)
        except Exception as exc:
            print(f"[jimu visual] failed to save multi-view scene preview: {exc}")


def _save_jimu_scene_preview_views(demo, args, points: list[np.ndarray]) -> None:
    valid_points = [
        np.asarray(point, dtype=np.float32).reshape(3)
        for point in list(points or [])
        if np.asarray(point, dtype=np.float32).size == 3 and np.all(np.isfinite(np.asarray(point, dtype=np.float32)))
    ]
    if not valid_points:
        return
    camera_pose_fn = getattr(direct.targeted.base, "_failure_place_camera_poses", None)
    tile_fn = getattr(direct.targeted.base, "_tile_failure_render_images", None)
    output_path_fn = getattr(direct.targeted.base, "_failure_render_output_path", None)
    capture_fn = getattr(direct.targeted.base, "capture_failure_render_image", None)
    if not (callable(camera_pose_fn) and callable(tile_fn) and callable(output_path_fn) and callable(capture_fn)):
        return
    labeled_images = []
    for view_name, camera_pose in list(camera_pose_fn(valid_points) or []):
        if len(labeled_images) >= 2:
            break
        image = capture_fn(demo, args, camera_pose=camera_pose)
        if image is not None:
            labeled_images.append((f"jimu_{view_name}", image))
    tiled = tile_fn(labeled_images)
    if tiled is None:
        return
    output_path = output_path_fn(args, "jimu_scene_preview", suffix="views")
    tiled.save(output_path)
    print(f"[jimu visual] saved multi-view scene preview image: {output_path}")


def _map_cam_pose_to_base_for_assignment(
    args: argparse.Namespace,
    T_base_cam: np.ndarray,
    T_cam_obj: np.ndarray,
    *,
    apply_jimu_local_fix: bool = True,
) -> np.ndarray:
    T_base_cam = np.asarray(T_base_cam, dtype=np.float32).reshape(4, 4)
    T_cam_obj = np.asarray(T_cam_obj, dtype=np.float32).reshape(4, 4)
    if apply_jimu_local_fix:
        T_cam_obj = (T_cam_obj @ _jimu_cad_to_sim_local_fix(args)).astype(np.float32)
    if bool(getattr(args, "use_direct_camera_extrinsic", False)):
        return (T_base_cam @ T_cam_obj).astype(np.float32)
    return (np.linalg.inv(T_base_cam).astype(np.float32) @ T_cam_obj).astype(np.float32)


def _candidate_entries(
    summary: dict,
    args: argparse.Namespace,
    T_base_cam: np.ndarray,
    provider_name: str,
    *,
    apply_jimu_local_fix: bool = True,
) -> list[dict]:
    provider_name = direct.curobo_wrapper.normalize_object_name(provider_name) or provider_name
    entries: list[dict] = []
    for fallback_index, item in enumerate(list(summary.get("results") or [])):
        if not isinstance(item, dict) or not bool(item.get("ok", True)) or item.get("T_cam_obj") is None:
            continue
        item_name = direct.curobo_wrapper.normalize_object_name(item.get("object_name"))
        if item_name != provider_name:
            continue
        T_cam_obj = np.asarray(item["T_cam_obj"], dtype=np.float32).reshape(4, 4)
        T_base_obj = _map_cam_pose_to_base_for_assignment(
            args,
            T_base_cam,
            T_cam_obj,
            apply_jimu_local_fix=apply_jimu_local_fix,
        )
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


def _best_entry_for_anchor(entries: list[dict], args: argparse.Namespace, index_attr: str) -> dict:
    if not entries:
        raise RuntimeError(f"no SAM6D candidate available for anchor {index_attr}")
    requested = getattr(args, index_attr, None)
    if requested is not None and int(requested) >= 0:
        return _find_candidate_by_instance(entries, int(requested), set())
    return max(entries, key=lambda entry: float(entry["item"].get("score", 0.0)))


def _canonical_tray_base_pose(T_base_tray: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    T_base_tray = np.asarray(T_base_tray, dtype=np.float32).reshape(4, 4)
    out = np.eye(4, dtype=np.float32)
    out[:3, 3] = T_base_tray[:3, 3]
    snap_cardinal = bool(getattr(args, "jimu_canonical_snap_cardinal", True))
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    x_axis = _preferred_horizontal_axis(
        [T_base_tray[:3, 0], T_base_tray[:3, 1]],
        snap_cardinal=snap_cardinal,
    )
    if x_axis is None:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    y_axis = _normalize_vec(np.cross(world_up, x_axis))
    if y_axis is None:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = _normalize_vec(np.cross(y_axis, world_up))
    if x_axis is None:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    out[:3, :3] = _rotation_from_xy_z(x_axis, y_axis, world_up)
    return out


def _jimu_base_support_local_poses(args: argparse.Namespace | None = None) -> dict[str, np.ndarray]:
    extents = _load_scaled_jimu_extents(args)
    offset_x = float(extents[0])
    offset_z = float(extents[2])
    positions = {
        "floor_right_support": (offset_x, 0.0, 0.0),
        "floor_left_support": (-offset_x, 0.0, 0.0),
        "floor_back_support": (0.0, 0.0, offset_z),
        "floor_front_support": (0.0, 0.0, -offset_z),
    }
    out: dict[str, np.ndarray] = {}
    for role, position in positions.items():
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = np.asarray(position, dtype=np.float32)
        out[role] = T
    return out


def _jimu_tray_slot_role_order(args: argparse.Namespace | None = None) -> list[str]:
    explicit = _split_names(getattr(args, "jimu_tray_slot_role_order", None) if args is not None else None)
    roles: list[str] = []
    seen: set[str] = set()
    for role in [*explicit, *JIMU_TRAY_SLOT_ROLES]:
        if role in seen:
            continue
        seen.add(role)
        roles.append(role)
    rows = max(1, int(getattr(args, "jimu_tray_slot_rows", 2) if args is not None else 2))
    cols = max(1, int(getattr(args, "jimu_tray_slot_columns", 7) if args is not None else 7))
    return roles[: rows * cols]


def _jimu_tray_slot_local_poses(args: argparse.Namespace | None = None) -> dict[str, np.ndarray]:
    bounds = _jimu_tray_bounds_scaled(args)
    min_v = bounds[0]
    max_v = bounds[1]
    extents = _load_scaled_jimu_extents(args)
    plate_size = float(extents[0])
    cols = max(1, int(getattr(args, "jimu_tray_slot_columns", 7) if args is not None else 7))
    rows = max(1, int(getattr(args, "jimu_tray_slot_rows", 2) if args is not None else 2))
    x_margin = float(
        getattr(args, "jimu_tray_slot_x_margin_m", 0.00875)
        if args is not None
        else 0.00875
    )
    x_offset = float(
        getattr(args, "jimu_tray_slot_x_offset_m", PORTABLE_DEFAULT_TRAY_SLOT_X_OFFSET_M)
        if args is not None
        else PORTABLE_DEFAULT_TRAY_SLOT_X_OFFSET_M
    )
    # The tray OBJ already contains the actual slot apertures.  Its 14 slots
    # are not centered on an inset rectangular grid: in the CAD, the slot
    # centers sit close to the two long side edges at X ~= 8.75mm and
    # X ~= width-8.75mm.  The two row centers are 40mm from the long edges in
    # the CAD, not at 20%/80% of the 230mm tray span (46/184mm).
    # Keep the margin as an explicit override, but make the default match the
    # modeled slot centers instead of the old visual-only layout.
    x_min = float(min_v[0] + x_margin)
    x_max = float(max_v[0] - x_margin)
    if x_max < x_min:
        center_x = float((min_v[0] + max_v[0]) * 0.5)
        x_min = x_max = center_x
    if cols == 1:
        x_values = [float((x_min + x_max) * 0.5)]
    else:
        x_values = [float(v) for v in np.linspace(x_min, x_max, cols)]
    if abs(x_offset) > 1e-9:
        x_values = [float(v + x_offset) for v in x_values]

    y_min = float(min_v[1])
    y_max = float(max_v[1])
    if rows == 1:
        y_values = [float((y_min + y_max) * 0.5)]
    elif rows == 2:
        y_margin = float(
            getattr(args, "jimu_tray_slot_y_margin_m", 0.040)
            if args is not None
            else 0.040
        )
        y_values = [
            float(y_min + y_margin),
            float(y_max - y_margin),
        ]
    else:
        y_margin = max(float(plate_size * 0.35), 0.010)
        y_values = [float(v) for v in np.linspace(y_min + y_margin, y_max - y_margin, rows)]

    insertion_depth = float(
        getattr(args, "jimu_tray_slot_insertion_depth_m", 0.012)
        if args is not None
        else 0.012
    )
    z_center = float(max_v[2] + 0.5 * plate_size - insertion_depth)

    # Plate convention: local Y is the thin axis, local Z is the long vertical axis.
    # Tray convention from the CAD: local Z is up, local X spans the slot sequence.
    R_tray_plate = np.column_stack(
        [
            [0.0, -1.0, 0.0],  # plate local X along tray -Y
            [1.0, 0.0, 0.0],   # plate local Y/thickness along tray +X
            [0.0, 0.0, 1.0],   # plate local Z upward
        ]
    ).astype(np.float32)

    out: dict[str, np.ndarray] = {}
    role_order = _jimu_tray_slot_role_order(args)
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= len(role_order):
                break
            role = role_order[idx]
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R_tray_plate
            T[:3, 3] = np.asarray([x_values[col], y_values[row], z_center], dtype=np.float32)
            out[role] = T
            idx += 1
    return out


def _make_synthetic_jimu_result_item(
    args: argparse.Namespace,
    T_base_cam: np.ndarray,
    role: str,
    T_base_obj: np.ndarray,
    *,
    source_entry: dict,
    source_name: str,
    slot_index: int | None = None,
) -> dict:
    source_item = copy.deepcopy(source_entry["item"])
    T_cam_obj = _raw_camera_pose_from_base_pose(args, T_base_cam, T_base_obj)
    source_item.update(
        {
            "object_name": role,
            "prompt": "small square plastic building block.",
            "T_cam_obj": T_cam_obj,
            "translation_m": T_cam_obj[:3, 3].astype(float).tolist(),
            "ok": True,
            "jimu_role": role,
            "jimu_derived_from_anchor": source_name,
            "jimu_anchor_result_index": int(source_entry["fallback_index"]),
            "jimu_anchor_sam3_instance_index": int(source_entry["sam3_instance_index"]),
        }
    )
    if slot_index is not None:
        source_item["jimu_tray_slot_index"] = int(slot_index)
        source_item["sam3_instance_index"] = int(slot_index)
    return source_item


def _cache_jimu_derived_role(
    args: argparse.Namespace,
    bridge_mod,
    T_base_cam: np.ndarray,
    role: str,
    T_base_obj: np.ndarray,
    *,
    source_entry: dict,
    source_name: str,
    slot_index: int | None = None,
) -> tuple[dict, dict]:
    item = _make_synthetic_jimu_result_item(
        args,
        T_base_cam,
        role,
        T_base_obj,
        source_entry=source_entry,
        source_name=source_name,
        slot_index=slot_index,
    )
    cached = direct_sam6d._cache_entry_from_sam6d_result(args, bridge_mod, role, item)
    cached["jimu_provider_object_name"] = source_name
    cached["jimu_provider_result_index"] = int(source_entry["fallback_index"])
    cached["jimu_derived_from_anchor"] = source_name
    cached["jimu_T_base_obj"] = np.asarray(T_base_obj, dtype=np.float32).reshape(4, 4).copy()
    if slot_index is not None:
        cached["jimu_tray_slot_index"] = int(slot_index)
    debug = {
        "fallback_index": int(source_entry["fallback_index"]),
        "sam3_instance_index": int(source_entry["sam3_instance_index"]),
        "score": float(source_entry["item"].get("score", 0.0)),
        "base_translation": np.asarray(T_base_obj, dtype=np.float32)[:3, 3].astype(float).tolist(),
        "axis_dot_base_z": np.asarray(T_base_obj, dtype=np.float32)[2, :3].astype(float).tolist(),
        "abs_axis_dot_base_z": np.abs(np.asarray(T_base_obj, dtype=np.float32)[2, :3]).astype(float).tolist(),
        "derived_from_anchor": source_name,
    }
    if slot_index is not None:
        debug["tray_slot_index"] = int(slot_index)
    return cached, debug


def _cache_jimu_anchor_visual(
    args: argparse.Namespace,
    bridge_mod,
    T_base_cam: np.ndarray,
    object_name: str,
    T_base_obj: np.ndarray,
    *,
    source_entry: dict,
) -> dict:
    item = copy.deepcopy(source_entry["item"])
    T_cam_obj = _raw_camera_pose_from_base_pose(args, T_base_cam, T_base_obj)
    item.update(
        {
            "object_name": object_name,
            "prompt": str(item.get("prompt", object_name)),
            "T_cam_obj": T_cam_obj,
            "translation_m": T_cam_obj[:3, 3].astype(float).tolist(),
            "ok": True,
            "jimu_visual_anchor": True,
            "jimu_anchor_result_index": int(source_entry["fallback_index"]),
            "jimu_anchor_sam3_instance_index": int(source_entry["sam3_instance_index"]),
        }
    )
    cached = direct_sam6d._cache_entry_from_sam6d_result(args, bridge_mod, object_name, item)
    cached["jimu_visual_anchor"] = True
    cached["jimu_T_base_obj"] = np.asarray(T_base_obj, dtype=np.float32).reshape(4, 4).copy()
    return cached


def _jimu_base_pose_to_pick_world_no_table_clamp(demo, bridge_mod, args, cache_entry: dict) -> np.ndarray | None:
    try:
        T_base_obj = np.asarray(cache_entry["jimu_T_base_obj"], dtype=np.float32).reshape(4, 4)
    except Exception:
        return None
    adjusted = T_base_obj.copy()
    object_args = cache_entry.get("object_args")
    map_args = object_args if object_args is not None else args
    if not bool(getattr(map_args, "no_map_foundationpose_through_robot_base", False)):
        get_robot_base_transform = getattr(bridge_mod, "get_robot_base_transform", None)
        if callable(get_robot_base_transform):
            robot_base_T = get_robot_base_transform(demo.env)
            if robot_base_T is not None:
                adjusted = np.asarray(robot_base_T, dtype=np.float32).reshape(4, 4) @ adjusted
    offset = np.asarray(
        getattr(map_args, "foundationpose_position_offset", getattr(args, "foundationpose_position_offset", [0.0, 0.0, 0.0])),
        dtype=np.float32,
    ).reshape(-1)
    if offset.size >= 3:
        adjusted[:3, 3] += offset[:3]
    return adjusted.astype(np.float32)


def _derive_jimu_roles_from_assembly_anchors(
    args: argparse.Namespace,
    bridge_mod,
    summary: dict,
    T_base_cam: np.ndarray,
    role_names: list[str],
) -> tuple[dict[str, dict], dict] | None:
    base_name = direct.curobo_wrapper.normalize_object_name(
        getattr(args, "jimu_base_assembly_object_name", JIMU_BASE_ASSEMBLY_OBJECT_NAME)
    ) or JIMU_BASE_ASSEMBLY_OBJECT_NAME
    tray_name = direct.curobo_wrapper.normalize_object_name(
        getattr(args, "jimu_tray_object_name", JIMU_TRAY_OBJECT_NAME)
    ) or JIMU_TRAY_OBJECT_NAME
    base_entries = _candidate_entries(
        summary,
        args,
        T_base_cam,
        base_name,
        apply_jimu_local_fix=False,
    )
    tray_entries = _candidate_entries(
        summary,
        args,
        T_base_cam,
        tray_name,
        apply_jimu_local_fix=False,
    )
    if not base_entries or not tray_entries:
        return None

    base_entry = _best_entry_for_anchor(base_entries, args, "jimu_base_assembly_instance_index")
    tray_entry = _best_entry_for_anchor(tray_entries, args, "jimu_tray_instance_index")
    T_base_assembly_raw = np.asarray(base_entry["T_base_obj"], dtype=np.float32).reshape(4, 4)
    T_base_tray_raw = np.asarray(tray_entry["T_base_obj"], dtype=np.float32).reshape(4, 4)
    T_base_floor = _canonical_jimu_base_pose(JIMU_FLOOR_ROLE, T_base_assembly_raw, floor_T_base_obj=None, args=args)
    T_base_tray = _canonical_tray_base_pose(T_base_tray_raw, args)
    raw_floor_z = float(T_base_floor[2, 3])
    raw_tray_z = float(T_base_tray[2, 3])
    T_base_floor[2, 3] = float(_jimu_floor_center_z_on_table(args))
    T_base_tray[2, 3] = float(_jimu_tray_origin_z_on_table(args))
    print(
        "[jimu-sam6d] table-locked assembly anchor Z: "
        f"floor {raw_floor_z:.4f}->{float(T_base_floor[2, 3]):.4f}, "
        f"tray {raw_tray_z:.4f}->{float(T_base_tray[2, 3]):.4f}"
    )
    visual_anchors = {
        tray_name: _cache_jimu_anchor_visual(
            args,
            bridge_mod,
            T_base_cam,
            tray_name,
            T_base_tray,
            source_entry=tray_entry,
        ),
        base_name: _cache_jimu_anchor_visual(
            args,
            bridge_mod,
            T_base_cam,
            base_name,
            T_base_floor,
            source_entry=base_entry,
        ),
    }

    role_set = set(role_names)
    cached_objects: dict[str, dict] = {}
    assignments: dict[str, dict] = {}
    if JIMU_FLOOR_ROLE in role_set:
        cached, debug = _cache_jimu_derived_role(
            args,
            bridge_mod,
            T_base_cam,
            JIMU_FLOOR_ROLE,
            T_base_floor,
            source_entry=base_entry,
            source_name=base_name,
        )
        cached_objects[JIMU_FLOOR_ROLE] = cached
        assignments[JIMU_FLOOR_ROLE] = debug

    for role, T_floor_role in _jimu_base_support_local_poses(args).items():
        if role not in role_set:
            continue
        T_base_role = (T_base_floor @ T_floor_role).astype(np.float32)
        cached, debug = _cache_jimu_derived_role(
            args,
            bridge_mod,
            T_base_cam,
            role,
            T_base_role,
            source_entry=base_entry,
            source_name=base_name,
        )
        cached_objects[role] = cached
        assignments[role] = debug

    slot_poses = _jimu_tray_slot_local_poses(args)
    slot_index_by_role = {role: idx for idx, role in enumerate(_jimu_tray_slot_role_order(args))}
    visual_slots: dict[str, dict] = {}
    for role, T_tray_slot in slot_poses.items():
        T_base_role = (T_base_tray @ T_tray_slot).astype(np.float32)
        slot_index = slot_index_by_role.get(role)
        cached, _debug = _cache_jimu_derived_role(
            args,
            bridge_mod,
            T_base_cam,
            role,
            T_base_role,
            source_entry=tray_entry,
            source_name=tray_name,
            slot_index=slot_index,
        )
        cached["jimu_visual_slot"] = True
        visual_slots[role] = cached

    # Cache every physical tray slot, not just the logical build roles.  The
    # build loop can then remap a logical wall target to the next tray plate
    # when one physical slot has no grasp IK.
    slot_cache_roles = list(dict.fromkeys([*list(role_names or []), *_jimu_tray_slot_role_order(args)]))
    for role in slot_cache_roles:
        if role in cached_objects or role not in slot_poses:
            continue
        T_base_role = (T_base_tray @ slot_poses[role]).astype(np.float32)
        slot_index = slot_index_by_role.get(role)
        cached, debug = _cache_jimu_derived_role(
            args,
            bridge_mod,
            T_base_cam,
            role,
            T_base_role,
            source_entry=tray_entry,
            source_name=tray_name,
            slot_index=slot_index,
        )
        cached_objects[role] = cached
        assignments[role] = debug

    debug = {
        "method": "assembly_anchor_derivation",
        "base_anchor": {
            "object_name": base_name,
            "fallback_index": int(base_entry["fallback_index"]),
            "sam3_instance_index": int(base_entry["sam3_instance_index"]),
            "raw_translation": T_base_assembly_raw[:3, 3].astype(float).tolist(),
            "canonical_floor_translation": T_base_floor[:3, 3].astype(float).tolist(),
        },
        "tray_anchor": {
            "object_name": tray_name,
            "fallback_index": int(tray_entry["fallback_index"]),
            "sam3_instance_index": int(tray_entry["sam3_instance_index"]),
            "raw_translation": T_base_tray_raw[:3, 3].astype(float).tolist(),
            "canonical_translation": T_base_tray[:3, 3].astype(float).tolist(),
        },
        "slot_roles": _jimu_tray_slot_role_order(args),
        "visual_anchors": visual_anchors,
        "visual_slots": visual_slots,
        "assignments": assignments,
    }
    print(
        "[jimu-sam6d] derived Jimu scene from assembly anchors: "
        f"base={base_name} tray={tray_name} roles={sorted(cached_objects.keys())}"
    )
    return cached_objects, debug


def _jimu_cache_key(args: argparse.Namespace, role_names: list[str], provider_names: list[str]) -> tuple:
    return (
        "jimu_layered_wall_sam6d",
        tuple(role_names),
        tuple(provider_names),
        direct_sam6d._sam6d_cache_key(args, provider_names),
        str(
            getattr(args, "jimu_sam6d_provider_script", "")
            or getattr(args, "sam6d_provider_script", direct_sam6d.DEFAULT_SAM6D_PROVIDER_SCRIPT)
        ),
    )


def _run_jimu_sam6d_provider_batch(
    args: argparse.Namespace,
    provider_names: list[str],
    *,
    sam3_result_json: str = "",
    frame_dir: str = "",
    same_object_instance_start_index: int = 0,
) -> tuple[dict[str, Any], Path]:
    provider_script = str(
        getattr(args, "jimu_sam6d_provider_script", "")
        or getattr(args, "sam6d_provider_script", direct_sam6d.DEFAULT_SAM6D_PROVIDER_SCRIPT)
    )
    run_args = argparse.Namespace(
        sam6d_provider_script=str(Path(provider_script).expanduser()),
        sam6d_output_root=str(Path(getattr(args, "sam6d_output_root", SCRIPT_DIR / "sam6d_jimu_direct_runs")).expanduser()),
        sam6d_mask_mode=str(getattr(args, "sam6d_mask_mode", "sam3_text")),
        sam6d_root=str(Path(getattr(args, "sam6d_root", direct_sam6d.sam6d_provider.DEFAULT_SAM6D_ROOT)).expanduser()),
        camera_width=int(getattr(args, "camera_width", 640)),
        camera_height=int(getattr(args, "camera_height", 480)),
        camera_fps=int(getattr(args, "camera_fps", 30)),
        warmup_frames=int(getattr(args, "warmup_frames", 30)),
        camera_serial=str(getattr(args, "camera_serial", "") or ""),
        camera_extrinsic_opencv_path=str(
            Path(
                getattr(
                    args,
                    "camera_extrinsic_opencv_path",
                    direct_sam6d.sam6d_provider.DEFAULT_CAMERA_EXTRINSIC_OPENCV_PATH,
                )
            ).expanduser()
        ),
        use_direct_camera_extrinsic=bool(getattr(args, "use_direct_camera_extrinsic", False)),
        sam3_python=str(Path(getattr(args, "sam3_python", direct_sam6d.sam6d_provider.DEFAULT_SAM3_PYTHON)).expanduser()),
        sam3_provider_script=str(
            Path(getattr(args, "sam3_provider_script", direct_sam6d.sam6d_provider.DEFAULT_SAM3_PROVIDER_SCRIPT)).expanduser()
        ),
        sam3_checkpoint_path=str(
            Path(getattr(args, "sam3_checkpoint_path", direct_sam6d.sam6d_provider.DEFAULT_SAM3_CHECKPOINT_PATH)).expanduser()
        ),
        sam3_resolution=int(getattr(args, "sam3_resolution", 1008)),
        sam3_device=str(getattr(args, "sam3_device", "") or ""),
        sam3_full_scene_keep_multi_instances=True,
        sam3_max_masks_per_item=max(int(getattr(args, "sam3_max_masks_per_item", 1) or 1), len(provider_names)),
        sam3_confidence_threshold=float(getattr(args, "sam3_confidence_threshold", 0.20)),
        sam3_morph_kernel=int(getattr(args, "sam3_morph_kernel", 3)),
        sam6d_frame_dir=str(frame_dir or getattr(args, "sam6d_frame_dir", "") or ""),
        sam3_full_scene_result_json=str(sam3_result_json or getattr(args, "sam3_full_scene_result_json", "") or ""),
        sam3_instance_index=int(getattr(args, "sam3_instance_index", 0)),
        same_object_instance_start_index=int(max(0, same_object_instance_start_index)),
        sam6d_confirm_segmentation=bool(getattr(args, "sam6d_confirm_segmentation", True)),
        sam6d_require_full_scene_masks=bool(getattr(args, "sam6d_require_full_scene_masks", True)),
        sam6d_show_segmentation_window=bool(getattr(args, "sam6d_show_segmentation_window", True)),
        sam6d_no_pem_warmup_during_sam3=bool(getattr(args, "sam6d_no_pem_warmup_during_sam3", True)),
        sam6d_no_post_pem_mask_refine=bool(getattr(args, "sam6d_no_post_pem_mask_refine", True)),
        sam6d_no_full_scene_pem_visualization=bool(getattr(args, "sam6d_no_full_scene_pem_visualization", False)),
        sam6d_no_pem_save_visualization=bool(getattr(args, "sam6d_no_pem_save_visualization", False)),
        sam6d_pem_feature_cache_root=str(
            Path(
                getattr(
                    args,
                    "sam6d_pem_feature_cache_root",
                    PICK_JIAOBANG_DIR / "sam6d_pem_feature_cache",
                )
            ).expanduser()
        ),
        sam6d_post_pem_mask_refine_objects=str(getattr(args, "sam6d_post_pem_mask_refine_objects", "")),
        sam6d_post_pem_mask_refine_trigger_px=float(getattr(args, "sam6d_post_pem_mask_refine_trigger_px", 6.0)),
        sam6d_provider_timeout_s=float(getattr(args, "sam6d_provider_timeout_s", 240.0) or 240.0),
        grounding_dino_model_id=str(getattr(args, "grounding_dino_model_id", "IDEA-Research/grounding-dino-base")),
        grounding_dino_local_files_only=bool(getattr(args, "grounding_dino_local_files_only", True)),
        grounding_dino_box_threshold=float(getattr(args, "grounding_dino_box_threshold", 0.25)),
        grounding_dino_text_threshold=float(getattr(args, "grounding_dino_text_threshold", 0.20)),
        jimu_manual_sam6d_bboxes=bool(getattr(args, "jimu_manual_sam6d_bboxes", False)),
        jimu_tabletop_anchor_localization=bool(getattr(args, "jimu_tabletop_anchor_localization", False)),
        jimu_apriltag_anchor_localization=bool(getattr(args, "jimu_apriltag_anchor_localization", False)),
        jimu_apriltag_base_id=int(getattr(args, "jimu_apriltag_base_id", 1)),
        jimu_apriltag_tray_id=int(getattr(args, "jimu_apriltag_tray_id", 0)),
        jimu_apriltag_base_size_m=float(getattr(args, "jimu_apriltag_base_size_m", 0.052)),
        jimu_apriltag_tray_size_m=float(getattr(args, "jimu_apriltag_tray_size_m", 0.06)),
        jimu_apriltag_base_yaw_deg=float(getattr(args, "jimu_apriltag_base_yaw_deg", 0.0)),
        jimu_apriltag_tray_yaw_deg=float(getattr(args, "jimu_apriltag_tray_yaw_deg", 90.0)),
    )
    print("[jimu-sam6d] using portable same-object SAM3/SAM6D subprocess call")
    return direct_sam6d._run_sam6d_provider(run_args, provider_names)


def _sam3_result_json_from_summary(summary: dict, result_path: Path) -> Path | None:
    scene_dir = summary.get("scene_dir")
    candidates: list[Path] = []
    if scene_dir:
        candidates.append(Path(scene_dir) / "sam3_full_scene_text" / "sam3_batch_result.json")
    candidates.append(Path(result_path).parent / "sam3_full_scene_text" / "sam3_batch_result.json")
    candidates.append(Path(result_path).parent.parent / "sam3_full_scene_text" / "sam3_batch_result.json")
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate.exists():
            return candidate
    return None


def _shared_frame_dir_from_summary(summary: dict, result_path: Path) -> Path | None:
    scene_dir = summary.get("scene_dir")
    candidates: list[Path] = []
    if scene_dir:
        candidates.append(Path(scene_dir) / "shared_frame")
    candidates.append(Path(result_path).parent / "shared_frame")
    candidates.append(Path(result_path).parent.parent / "shared_frame")
    for candidate in candidates:
        candidate = candidate.expanduser()
        if (candidate / "rgb.png").exists() and (candidate / "depth.png").exists() and (candidate / "camera.json").exists():
            return candidate
    return None


def _write_merged_sam6d_summary(chunks: list[tuple[dict, Path]], object_count: int, batch_size: int) -> Path:
    first_summary, first_path = chunks[0]
    scene_dir = Path(first_summary.get("scene_dir") or Path(first_path).parent).expanduser()
    merged_path = scene_dir / f"full_scene_pose_results_merged_batch{batch_size}.json"
    results: list[dict] = []
    for summary, _ in chunks:
        results.extend(list(summary.get("results") or []))
    merged = {
        "scene_dir": str(scene_dir),
        "object_count": int(object_count),
        "ok_count": sum(1 for item in results if bool(item.get("ok", True))),
        "results": results,
        "chunked_same_object_pem": True,
        "chunk_count": len(chunks),
        "max_pem_batch_size": int(batch_size),
        "chunk_result_paths": [str(path) for _, path in chunks],
    }
    with open(merged_path, "w") as f:
        import json

        json.dump(merged, f, indent=2)
    return merged_path


def _jimu_provider_names_for_localization(args: argparse.Namespace, role_names: list[str]) -> list[str]:
    localization_mode = str(getattr(args, "jimu_localization_mode", "assembly") or "assembly").strip().lower()
    if localization_mode == "assembly":
        return [
            direct.curobo_wrapper.normalize_object_name(
                getattr(args, "jimu_base_assembly_object_name", JIMU_BASE_ASSEMBLY_OBJECT_NAME)
            )
            or JIMU_BASE_ASSEMBLY_OBJECT_NAME,
            direct.curobo_wrapper.normalize_object_name(getattr(args, "jimu_tray_object_name", JIMU_TRAY_OBJECT_NAME))
            or JIMU_TRAY_OBJECT_NAME,
        ]
    provider_name = direct.curobo_wrapper.normalize_object_name(
        getattr(args, "jimu_provider_object_name", JIMU_PROVIDER_OBJECT_NAME)
    ) or JIMU_PROVIDER_OBJECT_NAME
    return [provider_name] * len(role_names)


def _jimu_default_export_scene_path() -> Path:
    return SCRIPT_DIR / "jimu_exported_scenes" / f"jimu_scene_{time.strftime('%Y%m%d_%H%M%S')}.json"


def _fixed_scene_uses_measured_anchor_orientation(path_like: str | Path | None) -> bool:
    if not path_like:
        return False
    try:
        path = Path(path_like).expanduser()
        if not path.exists():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    if isinstance(payload.get("jimu_apriltag_anchor_localization"), dict):
        return True
    if isinstance(payload.get("jimu_tabletop_anchor_localization"), dict):
        return True
    expanded = payload.get("jimu_expanded_scene")
    if isinstance(expanded, dict):
        anchor_results = expanded.get("anchor_results")
        if isinstance(anchor_results, list):
            for item in anchor_results:
                if isinstance(item, dict) and (
                    item.get("jimu_apriltag_anchor") is not None
                    or item.get("jimu_tabletop_anchor") is not None
                ):
                    return True
    return False


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _load_matrix_for_jimu_export(path: str | Path) -> np.ndarray:
    matrix_path = Path(path).expanduser()
    if matrix_path.suffix.lower() == ".npy":
        return np.load(matrix_path).astype(np.float32)
    return np.loadtxt(matrix_path).astype(np.float32)


def _jimu_export_pose_entry(
    args: argparse.Namespace,
    T_base_cam: np.ndarray,
    object_name: str,
    T_base_obj: np.ndarray,
    *,
    kind: str,
    source: str,
    slot_index: int | None = None,
) -> dict:
    T_base_obj = np.asarray(T_base_obj, dtype=np.float32).reshape(4, 4)
    T_cam_obj = _raw_camera_pose_from_base_pose(args, T_base_cam, T_base_obj)
    spec = object_specs.get_object_spec(object_name)
    entry = {
        "object_name": object_name,
        "ok": True,
        "kind": kind,
        "source": source,
        "T_base_obj": T_base_obj.astype(float).tolist(),
        "T_cam_obj": T_cam_obj.astype(float).tolist(),
        "translation_m": T_cam_obj[:3, 3].astype(float).tolist(),
    }
    if spec is not None:
        mesh_file = str(spec.mesh_file or "")
        sim_asset_file = str(spec.sim_asset_file or "")
        entry["object_spec_name"] = str(spec.name)
        entry["grounding_prompt"] = str(spec.grounding_prompt)
        entry["mesh_file"] = mesh_file
        entry["mesh_scale"] = float(spec.mesh_scale if spec.mesh_scale is not None else 1.0)
        entry["sim_asset_file"] = sim_asset_file
        entry["sim_asset_scale"] = float(spec.sim_asset_scale if spec.sim_asset_scale is not None else 1.0)
        entry["grasp_mode"] = str(spec.grasp_mode or "")
        asset_name = Path(sim_asset_file or mesh_file).name.lower()
        if "triangle" in asset_name:
            entry["jimu_shape"] = "red_triangle"
        elif "jimu" in asset_name or "plate" in asset_name or "cube" in asset_name:
            entry["jimu_shape"] = "red_square_plate"
    if slot_index is not None:
        entry["jimu_tray_slot_index"] = int(slot_index)
    return entry


def _build_jimu_expanded_scene_export(
    args: argparse.Namespace,
    summary: dict,
    T_base_cam: np.ndarray,
) -> dict | None:
    if str(getattr(args, "jimu_localization_mode", "assembly") or "assembly").strip().lower() != "assembly":
        return None
    base_name = direct.curobo_wrapper.normalize_object_name(
        getattr(args, "jimu_base_assembly_object_name", JIMU_BASE_ASSEMBLY_OBJECT_NAME)
    ) or JIMU_BASE_ASSEMBLY_OBJECT_NAME
    tray_name = direct.curobo_wrapper.normalize_object_name(getattr(args, "jimu_tray_object_name", JIMU_TRAY_OBJECT_NAME)) or JIMU_TRAY_OBJECT_NAME
    base_entries = _candidate_entries(summary, args, T_base_cam, base_name, apply_jimu_local_fix=False)
    tray_entries = _candidate_entries(summary, args, T_base_cam, tray_name, apply_jimu_local_fix=False)
    if not base_entries or not tray_entries:
        return None

    base_entry = _best_entry_for_anchor(base_entries, args, "jimu_base_assembly_instance_index")
    tray_entry = _best_entry_for_anchor(tray_entries, args, "jimu_tray_instance_index")
    T_base_assembly_raw = np.asarray(base_entry["T_base_obj"], dtype=np.float32).reshape(4, 4)
    T_base_tray_raw = np.asarray(tray_entry["T_base_obj"], dtype=np.float32).reshape(4, 4)
    T_base_floor = _canonical_jimu_base_pose(JIMU_FLOOR_ROLE, T_base_assembly_raw, floor_T_base_obj=None, args=args)
    T_base_tray = _canonical_tray_base_pose(T_base_tray_raw, args)
    T_base_floor[2, 3] = float(_jimu_floor_center_z_on_table(args))
    T_base_tray[2, 3] = float(_jimu_tray_origin_z_on_table(args))

    expanded_results: list[dict] = [
        _jimu_export_pose_entry(
            args,
            T_base_cam,
            base_name,
            T_base_floor,
            kind="anchor_base_assembly",
            source=base_name,
        ),
        _jimu_export_pose_entry(
            args,
            T_base_cam,
            tray_name,
            T_base_tray,
            kind="anchor_tray",
            source=tray_name,
        ),
    ]

    role_results: dict[str, dict] = {}
    role_results[JIMU_FLOOR_ROLE] = _jimu_export_pose_entry(
        args,
        T_base_cam,
        JIMU_FLOOR_ROLE,
        T_base_floor,
        kind="base_plate",
        source=base_name,
    )
    for role, T_floor_role in _jimu_base_support_local_poses(args).items():
        role_results[role] = _jimu_export_pose_entry(
            args,
            T_base_cam,
            role,
            (T_base_floor @ T_floor_role).astype(np.float32),
            kind="base_support_plate",
            source=base_name,
        )

    slot_poses = _jimu_tray_slot_local_poses(args)
    slot_index_by_role = {role: idx for idx, role in enumerate(_jimu_tray_slot_role_order(args))}
    for role in _jimu_tray_slot_role_order(args):
        T_tray_slot = slot_poses.get(role)
        if T_tray_slot is None:
            continue
        role_results[role] = _jimu_export_pose_entry(
            args,
            T_base_cam,
            role,
            (T_base_tray @ T_tray_slot).astype(np.float32),
            kind="tray_slot_plate",
            source=tray_name,
            slot_index=slot_index_by_role.get(role),
        )

    expanded_results.extend(role_results.values())
    return {
        "schema": "jimu_expanded_scene_v1",
        "description": "Derived base/tray/slot poses from the two assembly anchors. Root results remain the reusable provider anchors.",
        "base_anchor_name": base_name,
        "tray_anchor_name": tray_name,
        "table_top_z_m": float(_jimu_table_top_z(args)),
        "plate_extents_m": _load_scaled_jimu_extents(args).astype(float).tolist(),
        "tray_slot_roles": _jimu_tray_slot_role_order(args),
        "base_roles": list(JIMU_BASE_ROLES),
        "pick_roles": list(JIMU_PICK_ROLES),
        "anchor_results": expanded_results[:2],
        "role_results": role_results,
        "results": expanded_results,
    }


def _export_jimu_scene_json(
    args: argparse.Namespace,
    summary: dict,
    result_path: Path | None,
    *,
    force: bool = False,
    expanded_scene: dict | None = None,
) -> Path | None:
    requested = str(getattr(args, "jimu_export_scene_json", "") or "").strip()
    export_only = bool(getattr(args, "jimu_export_scene_only", False))
    if not force and not requested and not export_only:
        return None

    out_path = Path(requested).expanduser() if requested else _jimu_default_export_scene_path()
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source_path = Path(result_path).expanduser() if result_path is not None else None
    if source_path is not None and source_path.exists():
        try:
            if source_path.resolve() == out_path.resolve():
                print(f"[jimu export] scene json already at: {out_path}")
                setattr(args, "_jimu_exported_scene_json_path", str(out_path))
                return out_path
        except Exception:
            pass
        content = source_path.read_text()
        if expanded_scene is not None:
            payload = json.loads(content)
            payload["jimu_expanded_scene"] = _json_safe(expanded_scene)
            content = json.dumps(_json_safe(payload), indent=2)
    else:
        payload = _json_safe(summary)
        if expanded_scene is not None and isinstance(payload, dict):
            payload["jimu_expanded_scene"] = _json_safe(expanded_scene)
        content = json.dumps(payload, indent=2)

    out_path.write_text(content)
    print(f"[jimu export] scene json: {out_path}")
    if source_path is not None:
        print(f"[jimu export] source result: {source_path}")
    setattr(args, "_jimu_exported_scene_json_path", str(out_path))
    return out_path


def _run_jimu_sam6d_provider(args: argparse.Namespace, provider_names: list[str]) -> tuple[dict[str, Any], Path]:
    max_batch = max(1, int(getattr(args, "jimu_sam6d_max_pem_batch_size", 5) or 5))
    normalized_names = [direct.curobo_wrapper.normalize_object_name(name) or str(name) for name in provider_names]
    same_object = len(set(normalized_names)) == 1
    if len(provider_names) <= max_batch or not same_object:
        return _run_jimu_sam6d_provider_batch(args, provider_names)

    print(
        "[jimu-sam6d] splitting same-object PEM to avoid GPU OOM: "
        f"instances={len(provider_names)}, max_batch={max_batch}"
    )
    chunks: list[tuple[dict, Path]] = []
    first_names = provider_names[:max_batch]
    first_summary, first_path = _run_jimu_sam6d_provider_batch(args, first_names, same_object_instance_start_index=0)
    chunks.append((first_summary, first_path))
    sam3_result_json = _sam3_result_json_from_summary(first_summary, first_path)
    shared_frame_dir = _shared_frame_dir_from_summary(first_summary, first_path)
    if sam3_result_json is None or shared_frame_dir is None:
        raise RuntimeError(
            "Jimu SAM6D chunking could not find reusable SAM3/shared-frame outputs from the first batch; "
            f"result={first_path}"
        )

    for start in range(max_batch, len(provider_names), max_batch):
        chunk_names = provider_names[start : start + max_batch]
        print(
            "[jimu-sam6d] running same-object PEM chunk "
            f"start={start}, count={len(chunk_names)}, sam3={sam3_result_json}"
        )
        summary, result_path = _run_jimu_sam6d_provider_batch(
            args,
            chunk_names,
            sam3_result_json=str(sam3_result_json),
            frame_dir=str(shared_frame_dir),
            same_object_instance_start_index=start,
        )
        chunks.append((summary, result_path))

    merged_path = _write_merged_sam6d_summary(chunks, len(provider_names), max_batch)
    merged_summary = direct_sam6d._load_provider_summary(merged_path)
    merged_summary["provider_result_path"] = str(merged_path)
    print(
        "[jimu-sam6d] merged chunked SAM6D result: "
        f"ok_count={merged_summary.get('ok_count')}/{merged_summary.get('object_count')} result={merged_path}"
    )
    return merged_summary, merged_path


def _run_jimu_scene_export_only(args: argparse.Namespace) -> Path | None:
    role_names = _split_names(getattr(args, "jimu_scene_roles", None)) or list(JIMU_SCENE_ROLES)
    role_names = [name for name in role_names if name in JIMU_DERIVED_ROLE_SET]
    if JIMU_FLOOR_ROLE not in role_names:
        role_names.insert(0, JIMU_FLOOR_ROLE)
    build_roles = _split_names(getattr(args, "cycle_object_names", None)) or _default_pick_roles_for_layers(
        getattr(args, "jimu_build_layers", "two")
    )
    for role in build_roles:
        if role not in role_names:
            role_names.append(role)
    provider_names = _jimu_provider_names_for_localization(args, role_names)

    summary, result_path = direct_sam6d._load_fixed_sam6d_summary(args)
    if summary is None:
        with direct._CUROBO_GPU_LOCK:
            summary, result_path = _run_jimu_sam6d_provider(args, provider_names)
    _raise_if_forced_assembly_anchor_missing(args, summary, result_path, provider_names)
    expanded_scene = None
    try:
        T_base_cam = _load_matrix_for_jimu_export(args.camera_extrinsic_opencv_path).astype(np.float32)
        expanded_scene = _build_jimu_expanded_scene_export(args, summary, T_base_cam)
    except Exception as exc:
        print(f"[jimu export] warning: failed to build expanded scene export: {exc}")
    out_path = _export_jimu_scene_json(args, summary, result_path, force=True, expanded_scene=expanded_scene)
    if expanded_scene is not None:
        print(
            "[jimu export] expanded scene includes: "
            f"anchors={len(expanded_scene.get('anchor_results') or [])}, "
            f"roles={len(expanded_scene.get('role_results') or {})}"
        )
    print(
        "[jimu export] done: "
        f"ok_count={summary.get('ok_count')}/{summary.get('object_count')} "
        f"objects={[item.get('object_name') for item in list(summary.get('results') or [])]}"
    )
    return out_path


def _raise_if_forced_assembly_anchor_missing(args: argparse.Namespace, summary: dict, result_path: Path, provider_names: list[str]) -> None:
    forced_anchor = (
        bool(getattr(args, "jimu_apriltag_anchor_localization", False))
        or bool(getattr(args, "jimu_tabletop_anchor_localization", False))
    )
    if not forced_anchor:
        return
    ok_names = {
        direct.curobo_wrapper.normalize_object_name(item.get("object_name")) or str(item.get("object_name"))
        for item in list(summary.get("results") or [])
        if isinstance(item, dict) and bool(item.get("ok", True)) and item.get("T_cam_obj") is not None
    }
    expected = [direct.curobo_wrapper.normalize_object_name(name) or str(name) for name in provider_names]
    missing = [name for name in expected if name not in ok_names]
    if not missing:
        return
    details = []
    apriltag_info = summary.get("jimu_apriltag_anchor_localization")
    if isinstance(apriltag_info, dict):
        details.append(f"detected_tag_ids={apriltag_info.get('detected_tag_ids')}")
        if apriltag_info.get("overlay_path"):
            details.append(f"overlay={apriltag_info.get('overlay_path')}")
    for item in list(summary.get("results") or []):
        if isinstance(item, dict) and not bool(item.get("ok", True)):
            details.append(f"{item.get('object_name')}: {item.get('error')}")
    detail_text = "; ".join(str(v) for v in details if v)
    raise RuntimeError(
        "Jimu assembly anchor localization did not produce all required anchors; "
        f"missing={missing}, result={result_path}"
        + (f"; {detail_text}" if detail_text else "")
    )


def capture_or_reuse_jimu_sam6d_scene(args, bridge_mod, scene_capture_cache=None):
    target_name = direct.curobo_wrapper.normalize_object_name(getattr(args, "object_name", None))
    if target_name is None:
        raise ValueError("--object-name is required for Jimu SAM6D scene capture")

    role_names = _split_names(getattr(args, "jimu_scene_roles", None)) or list(JIMU_SCENE_ROLES)
    if target_name not in role_names:
        role_names.append(target_name)
    role_names = [name for name in role_names if name in JIMU_DERIVED_ROLE_SET]
    if JIMU_FLOOR_ROLE not in role_names:
        role_names.insert(0, JIMU_FLOOR_ROLE)
    build_roles = _split_names(getattr(args, "cycle_object_names", None)) or _default_pick_roles_for_layers(
        getattr(args, "jimu_build_layers", "two")
    )
    for role in build_roles:
        if role not in role_names:
            role_names.append(role)
    for role in _split_names(getattr(args, "tracked_scene_object_names", None)):
        if role in JIMU_DERIVED_ROLE_SET and role not in role_names:
            role_names.append(role)

    selected_obstacles = direct_sam6d._selected_obstacle_names(args, target_name)
    required_names = list(role_names)
    required_obstacles = [name for name in required_names if name != target_name]
    localization_mode = str(getattr(args, "jimu_localization_mode", "assembly") or "assembly").strip().lower()
    provider_name = direct.curobo_wrapper.normalize_object_name(
        getattr(args, "jimu_provider_object_name", JIMU_PROVIDER_OBJECT_NAME)
    ) or JIMU_PROVIDER_OBJECT_NAME
    provider_names = _jimu_provider_names_for_localization(args, role_names)
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
            summary, result_path = _run_jimu_sam6d_provider(args, provider_names)

    T_base_cam = bridge_mod.load_matrix(args.camera_extrinsic_opencv_path).astype(np.float32)
    if str(getattr(args, "jimu_export_scene_json", "") or "").strip():
        expanded_scene = None
        try:
            expanded_scene = _build_jimu_expanded_scene_export(args, summary, T_base_cam)
        except Exception as exc:
            print(f"[jimu export] warning: failed to build expanded scene export: {exc}")
        _export_jimu_scene_json(args, summary, result_path, expanded_scene=expanded_scene)
    cached_objects: dict[str, dict]
    assignment_debug: dict
    derived = None
    if localization_mode == "assembly":
        derived = _derive_jimu_roles_from_assembly_anchors(args, bridge_mod, summary, T_base_cam, role_names)
        if derived is None:
            _raise_if_forced_assembly_anchor_missing(args, summary, result_path, provider_names)
    if derived is not None:
        cached_objects, assignment_debug = derived
    else:
        if localization_mode == "assembly":
            print(
                "[jimu-sam6d] assembly anchors not found in summary; falling back to legacy per-block assignment "
                "(fixed legacy json is still supported)"
            )
            role_names = [role for role in role_names if role in set(JIMU_LEGACY_SCENE_ROLES)]
            required_names = list(role_names)
            required_obstacles = [name for name in required_names if name != target_name]
        entries = _candidate_entries(summary, args, T_base_cam, provider_name)
        assigned, assignment_debug = _assign_jimu_roles(args, entries, role_names)
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
    if bool(getattr(args, "jimu_print_role_assignment", True)):
        _print_assignment(assignment_debug)

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
                "jimu_visual_anchors": dict(assignment_debug.get("visual_anchors", {}) or {}),
                "jimu_visual_slots": dict(assignment_debug.get("visual_slots", {}) or {}),
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
    T_world_parent_target = _jimu_first_layer_target_pose_from_floor(
        demo,
        bridge_mod,
        scene_capture_cache,
        parent_name,
        args,
    )
    if T_world_parent_target is None:
        return None
    extents = _load_scaled_jimu_extents(args)
    z_extra = float(getattr(args, "jimu_second_layer_z_extra", 0.0) or 0.0)
    T_world_second_target = T_world_parent_target.copy()
    T_world_second_target[:3, 3] = (
        T_world_parent_target[:3, 3] + np.asarray([0.0, 0.0, float(extents[2]) + z_extra], dtype=np.float32)
    ).astype(np.float32)
    return T_world_second_target


def _jimu_table_top_z(args) -> float:
    try:
        return float(getattr(args, "curobo_table_z_offset", -0.01)) + 0.5 * float(
            getattr(args, "curobo_table_thickness", 0.02)
        )
    except Exception:
        return 0.0


def _jimu_floor_center_z_on_table(args) -> float:
    extents = _load_scaled_jimu_extents(args)
    return float(_jimu_table_top_z(args) + 0.5 * float(extents[1]))


def _jimu_first_layer_bottom_z(args) -> float:
    extents = _load_scaled_jimu_extents(args)
    table_top = float(_jimu_table_top_z(args))
    support_top = table_top + float(extents[1]) if bool(getattr(args, "jimu_base_support_obstacles", True)) else table_top
    clearance = float(getattr(args, "jimu_first_layer_bottom_clearance_m", 0.006) or 0.0)
    return float(support_top + clearance)


def _jimu_tray_origin_z_on_table(args) -> float:
    try:
        bounds = _jimu_tray_bounds_scaled(args)
        min_z = float(np.asarray(bounds[0], dtype=np.float32).reshape(3)[2])
    except Exception:
        min_z = 0.0
    return float(_jimu_table_top_z(args) - min_z)


def _matrix_axis_yaws_deg(T: np.ndarray) -> list[float | None]:
    R = np.asarray(T, dtype=np.float32).reshape(4, 4)[:3, :3]
    out: list[float | None] = []
    for axis_idx in range(3):
        xy = R[:2, axis_idx].astype(np.float64)
        if float(np.linalg.norm(xy)) <= 1e-6:
            out.append(None)
        else:
            yaw = float(np.rad2deg(np.arctan2(xy[1], xy[0])))
            out.append(float(((yaw + 180.0) % 360.0) - 180.0))
    return out


def _format_axis_yaws_deg(T: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for value in _matrix_axis_yaws_deg(T):
        out.append(None if value is None else round(float(value), 2))
    return out


def _jimu_first_layer_target_pose_from_floor(demo, bridge_mod, scene_capture_cache, source_name: str, args) -> np.ndarray | None:
    if source_name not in set(JIMU_FIRST_LAYER_ROLES):
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
    parent_spec = parent_specs.get(source_name)
    if parent_spec is None:
        return None
    T_floor_parent_target = direct.targeted._local_pose_spec_to_matrix(parent_spec)
    T_world_parent_target = (
        np.asarray(T_world_floor, dtype=np.float32).reshape(4, 4)
        @ np.asarray(T_floor_parent_target, dtype=np.float32).reshape(4, 4)
    ).astype(np.float32)
    extents = _load_scaled_jimu_extents(args)
    T_world_parent_target[:3, 3][2] = float(_jimu_first_layer_bottom_z(args) + 0.5 * float(extents[2]))
    return T_world_parent_target


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
        f"target_xyz={np.round(T_target[:3, 3], 6).tolist()}, "
        f"axis_yaw_deg={_format_axis_yaws_deg(T_target)}"
    )
    return anchored


def _jimu_floor_anchor_first_layer_plans(plans, demo, bridge_mod, scene_capture_cache, source_name: str | None, args) -> list:
    if source_name not in set(JIMU_FIRST_LAYER_ROLES):
        return list(plans or [])
    T_target = _jimu_first_layer_target_pose_from_floor(demo, bridge_mod, scene_capture_cache, source_name, args)
    if T_target is None:
        return list(plans or [])
    anchored = [_jimu_rebuild_place_plan_for_target(plan, T_target) for plan in list(plans or [])]
    print(
        f"[jimu place] {source_name}: first-layer target anchored to table-level bottom, "
        f"target_xyz={np.round(T_target[:3, 3], 6).tolist()}, "
        f"axis_yaw_deg={_format_axis_yaws_deg(T_target)}"
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
    plans = _jimu_floor_anchor_first_layer_plans(plans, demo, bridge_mod, scene_capture_cache, source_name, args)
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


def _local_z_rotation4(deg: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = euler2mat(0.0, 0.0, np.deg2rad(float(deg)), axes="sxyz").astype(np.float32)
    return T


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
    *,
    target_symmetry_deg: float = 0.0,
) -> dict | None:
    """Keep the grasp TCP->object relation fixed, then solve the TCP pose from the target object pose."""
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
    T_world_obj_grasp = (T_world_tcp_grasp @ T_tcp_obj).astype(np.float32)
    T_world_obj_target = _jimu_target_object_pose_from_candidate(place_candidate, T_tcp_obj)
    release_mode = "target_object_pose"
    if T_world_obj_target is None:
        target_center = _jimu_release_object_center_from_candidate(place_candidate, T_tcp_obj)
        if target_center is None or not np.all(np.isfinite(target_center)):
            return None
        T_world_obj_target = T_world_obj_grasp.copy()
        T_world_obj_target[:3, 3] = target_center.astype(np.float32)
        release_mode = "center_only_fallback"
    else:
        target_center = T_world_obj_target[:3, 3].astype(np.float32)

    target_symmetry_deg = float(target_symmetry_deg)
    if abs(target_symmetry_deg) > 1e-6:
        T_world_obj_target = (T_world_obj_target @ _local_z_rotation4(target_symmetry_deg)).astype(np.float32)
        target_center = T_world_obj_target[:3, 3].astype(np.float32)

    yaw_delta_deg = _jimu_world_z_yaw_delta_deg(T_world_obj_grasp, T_world_obj_target)
    T_world_tcp_release = (T_world_obj_target @ np.linalg.inv(T_tcp_obj).astype(np.float32)).astype(np.float32)
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
    sym_label = "" if abs(target_symmetry_deg) <= 1e-6 else f"_target_sym_yaw_{float(target_symmetry_deg):+.0f}"
    item["label"] = f"{base_label}_parallel_grasp_{yaw_label}{sym_label}"
    item["pose"] = hover_pose
    item["hover_pose"] = hover_pose
    item["pre_place_pose"] = hover_pose
    item["place_pose"] = release_pose
    item["release_pose"] = release_pose
    item["raw_release_pose"] = release_pose
    item["retreat_pose"] = hover_pose
    variant_parts = [part for part in (base_variant, "parallel_grasp", yaw_label, sym_label.strip("_")) if part]
    item["variant_label"] = "+".join(variant_parts)
    item["T_world_obj_desired"] = predicted_obj
    item["jimu_parallel_grasp_place"] = True
    item["jimu_parallel_world_z_yaw_deg"] = float(yaw_delta_deg)
    item["jimu_parallel_release_mode"] = release_mode
    item["jimu_parallel_target_symmetry_deg"] = float(target_symmetry_deg)
    item["jimu_parallel_target_center"] = target_center.astype(float).tolist()
    item["jimu_parallel_release_tcp_position"] = T_world_tcp_release[:3, 3].astype(float).tolist()
    item["jimu_parallel_source_place_label"] = base_label
    item["jimu_parallel_source_grasp_label"] = str(grasp_candidate.get("label", ""))
    item["jimu_parallel_grasp_tcp_position"] = T_world_tcp_grasp[:3, 3].astype(float).tolist()
    item["jimu_parallel_grasp_obj_center"] = T_world_obj_grasp[:3, 3].astype(float).tolist()
    item["grasp_z_lift_m"] = float(grasp_candidate.get("grasp_z_lift_m", 0.0) or 0.0)
    item["grasp_axis_shift_m"] = float(grasp_candidate.get("grasp_axis_shift_m", 0.0) or 0.0)
    item["grasp_approach_roll_deg"] = float(grasp_candidate.get("grasp_approach_roll_deg", 0.0) or 0.0)
    item["_fast_chain_yaw_expansion"] = True
    item["_fast_chain_yaw_expansion_token"] = f"parallel_grasp{sym_label}"
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


def _jimu_parallel_place_symmetry_degrees(args) -> list[float]:
    # This is intentionally independent from the legacy place-symmetry switch.
    # In Jimu parallel grasp/place, 180-deg square-plate symmetry is needed to
    # flip the release-side tilt sign while keeping the visible object target.
    values = _jimu_symmetry_degrees(args)
    if not any(abs(float(v)) <= 1e-6 for v in values):
        values.insert(0, 0.0)
    # Keep the parallel place expansion small: for square Jimu plates, 180 deg
    # around the plate's local vertical axis is the only symmetry we need to
    # flip the effective release-side tilt without changing the visible target.
    out: list[float] = []
    seen: set[int] = set()
    for value in values:
        deg = float(value)
        if abs(deg) > 1e-6 and abs(abs(deg) - 180.0) > 1.0:
            continue
        bucket = int(round(deg))
        if bucket in seen:
            continue
        seen.add(bucket)
        out.append(deg)
    return out or [0.0]


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


def _jimu_float_list_from_args(args, name: str, default, *, min_value: float | None = None) -> list[float]:
    try:
        return list(direct._unique_finite_float_list(getattr(args, name, default), min_value=min_value))
    except TypeError:
        return list(direct._unique_finite_float_list(getattr(args, name, default)))
    except Exception:
        values = getattr(args, name, default)
        if values is None:
            return []
        if not isinstance(values, (list, tuple)):
            values = [values]
        out: list[float] = []
        seen: set[float] = set()
        for value in values:
            try:
                number = float(value)
            except Exception:
                continue
            if not np.isfinite(number):
                continue
            if min_value is not None and number < float(min_value):
                continue
            key = round(number, 9)
            if key in seen:
                continue
            seen.add(key)
            out.append(number)
        return out


def _jimu_signed_tilt_label_queues(args, label_counts: dict[int, int] | None = None) -> dict[int, list[float]]:
    tilt_values = _jimu_float_list_from_args(
        args,
        "direct_grasp_tilt_toward_robot_deg",
        [12.0, 20.0, 30.0, 45.0],
    )
    shift_values = _jimu_float_list_from_args(
        args,
        "direct_grasp_tilt_toward_robot_shift_m",
        [0.0, 0.02],
        min_value=0.0,
    )
    axis_values = _jimu_float_list_from_args(
        args,
        "direct_grasp_object_axis_shifts_m",
        [0.0],
    )
    lift_values = _jimu_float_list_from_args(
        args,
        "direct_grasp_z_lifts_m",
        [0.0],
        min_value=0.0,
    )
    repeat_count = max(1, len(shift_values or [0.0]) * len(axis_values or [0.0]) * len(lift_values or [0.0]))
    signed_by_bucket: dict[int, list[float]] = {}
    for value in tilt_values:
        signed = float(value)
        if abs(signed) <= 1e-6:
            continue
        bucket = int(round(abs(signed)))
        signed_by_bucket.setdefault(bucket, []).append(signed)
    queues: dict[int, list[float]] = {}
    for bucket, signed_values in signed_by_bucket.items():
        count = int((label_counts or {}).get(bucket, 0) or 0)
        per_sign_repeat = repeat_count
        if count > 0:
            per_sign_repeat = max(1, (count + len(signed_values) - 1) // len(signed_values))
        queue: list[float] = []
        for signed in signed_values:
            queue.extend([signed] * per_sign_repeat)
        queues[bucket] = queue
    return queues


def _jimu_relabel_signed_tilt_grasp_candidates(candidates, args, source_name: str | None):
    source = direct.curobo_wrapper.normalize_object_name(source_name)
    if source not in set(JIMU_PICK_ROLES):
        return candidates
    label_counts: dict[int, int] = {}
    for candidate in list(candidates or []):
        label = str((candidate or {}).get("label", "") or "")
        lower_label = label.lower()
        if "tilt_toward_robot_" in lower_label or "tilt_away_robot_" in lower_label:
            continue
        match = re.search(r"tilt_([0-9]+(?:\.\d+)?)deg", label)
        if match is None:
            continue
        try:
            magnitude = int(round(float(match.group(1))))
        except Exception:
            continue
        label_counts[magnitude] = label_counts.get(magnitude, 0) + 1
    queues = _jimu_signed_tilt_label_queues(args, label_counts)
    if not queues:
        return candidates
    changed = 0
    relabeled: list[dict] = []
    for candidate in list(candidates or []):
        item = dict(candidate)
        label = str(item.get("label", "") or "")
        lower_label = label.lower()
        if "tilt_toward_robot_" in lower_label or "tilt_away_robot_" in lower_label:
            relabeled.append(item)
            continue
        match = re.search(r"tilt_([0-9]+(?:\.\d+)?)deg", label)
        if match is None:
            relabeled.append(item)
            continue
        try:
            magnitude = int(round(float(match.group(1))))
        except Exception:
            relabeled.append(item)
            continue
        queue = queues.get(magnitude) or []
        signed_tilt = float(queue.pop(0)) if queue else float(magnitude)
        direction = "toward_robot" if signed_tilt >= 0.0 else "away_robot"
        magnitude_text = f"{abs(signed_tilt):.1f}".rstrip("0").rstrip(".")
        signed_token = f"tilt_{direction}_{magnitude_text}deg"
        item["label"] = f"{label[:match.start()]}{signed_token}{label[match.end():]}"
        item["jimu_grasp_tilt_deg"] = signed_tilt
        item["jimu_grasp_tilt_direction"] = direction
        relabeled.append(item)
        changed += 1
    if changed:
        preview = [str(item.get("label", "")) for item in relabeled if "tilt_" in str(item.get("label", "")).lower()][:8]
        print(f"[jimu tilt] preserved signed tilt labels for {source}: changed={changed}, preview={preview}")
    return relabeled


def build_direct_grasp_candidates_jimu(demo, args, **kwargs):
    original = _ORIGINAL_BUILD_DIRECT_GRASP_CANDIDATES
    if original is None:
        return []
    candidates = original(demo, args, **kwargs)
    source = kwargs.get("source_name", None)
    if source is None:
        source = getattr(args, "object_name", None)
    return _jimu_relabel_signed_tilt_grasp_candidates(candidates, args, source)


def fast_chain_relation_match_key_jimu(item: dict | None, source_name: str | None) -> tuple:
    source = direct.curobo_wrapper.normalize_object_name(source_name)
    if source not in set(JIMU_PICK_ROLES):
        original = _ORIGINAL_FAST_CHAIN_RELATION_MATCH_KEY
        if original is None:
            return ("label", "" if item is None else str(item.get("label", "") or ""))
        return original(item, source_name)
    label = "" if item is None else str(item.get("label", "") or "")
    return ("jimu_tilt_abs_label", _jimu_tilt_abs_relation_label(label))


def _jimu_return_to_start_pair_score_penalty(rec: dict, start_q, args) -> float:
    weight = float(max(getattr(args, "jimu_return_to_start_pair_score_weight", 1.0) or 0.0, 0.0))
    if weight <= 1.0e-9:
        return 0.0
    place_candidate = rec.get("place_candidate", {}) if isinstance(rec, dict) else {}
    q_release = _jimu_q7_or_none(
        place_candidate.get("q_release", place_candidate.get("fast_chain_release_q"))
    )
    q_start = _jimu_q7_or_none(start_q)
    if q_release is None or q_start is None:
        return 0.0
    delta = np.abs(q_release - q_start)
    norm = float(np.linalg.norm(delta))
    max_delta = float(np.max(delta))
    base_delta = float(delta[0])
    wrist_delta = float(delta[6])
    penalty = weight * (norm + 0.55 * max_delta + 0.45 * base_delta + 0.25 * wrist_delta)
    place_candidate["jimu_return_to_start_score_penalty"] = float(penalty)
    place_candidate["jimu_return_to_start_release_delta_norm"] = float(norm)
    place_candidate["jimu_return_to_start_release_delta_max"] = float(max_delta)
    return float(penalty)


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
        grasp_approach_ready = (
            bool(grasp_candidate.get("_winner_preselect_grasp_approach_q_path"))
            or not bool(getattr(args, "short_linear_endpoint_ik_first", True))
        )
        if not (
            bool(grasp_candidate.get("_winner_preselect_pregrasp_success", False))
            and bool(grasp_candidate.get("_winner_preselect_grasp_success", False))
            and grasp_approach_ready
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
        symmetry_degrees = _jimu_parallel_place_symmetry_degrees(args)
        for place_candidate in source_place_candidates:
            for symmetry_deg in symmetry_degrees:
                parallel_candidate = _make_jimu_parallel_grasp_place_candidate(
                    grasp_candidate,
                    place_candidate,
                    args,
                    rule,
                    target_symmetry_deg=float(symmetry_deg),
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
                    "relation_token_suffix": str(place_candidate.get("_fast_chain_yaw_expansion_token", "parallel_grasp") or "parallel_grasp"),
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
        f"{len(records)} relation-place candidate(s); target object pose with 0/180 symmetry"
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
    tilted_symmetry_bias = 0
    return_start_penalty_count = 0
    max_return_start_penalty = 0.0
    symmetry_bias = float(getattr(args, "jimu_tilted_symmetry_score_bias", 0.75) or 0.0)
    for rec in pair_records:
        place_candidate = rec.get("place_candidate", {})
        grasp_label = str(
            place_candidate.get("jimu_parallel_source_grasp_label")
            or rec.get("grasp_candidate", {}).get("label", "")
        ).lower()
        sym_deg = abs(float(place_candidate.get("jimu_parallel_target_symmetry_deg", 0.0) or 0.0))
        if "grasp_tilt" in grasp_label and abs(sym_deg - 180.0) <= 1.0:
            # A 180-deg square-plate symmetry flips which side the tilted TCP
            # releases from without changing the visible block pose.  Prefer it
            # for tilted grasps, but keep the bias small so return-to-start
            # reachability can still dominate when a symmetric release wraps the
            # arm around the base.
            rec["score"] = float(rec.get("score", 0.0)) - symmetry_bias
            place_candidate["jimu_parallel_tilt_symmetry_preferred"] = True
            tilted_symmetry_bias += 1
        return_penalty = _jimu_return_to_start_pair_score_penalty(rec, start_q, args)
        if return_penalty > 0.0:
            rec["score"] = float(rec.get("score", 0.0)) + float(return_penalty)
            return_start_penalty_count += 1
            max_return_start_penalty = max(max_return_start_penalty, float(return_penalty))
    paired_debug["paired_jimu_parallel_tilt_symmetry_bias_count"] = int(tilted_symmetry_bias)
    paired_debug["paired_jimu_parallel_tilt_symmetry_bias"] = float(symmetry_bias)
    paired_debug["paired_jimu_return_to_start_penalty_count"] = int(return_start_penalty_count)
    paired_debug["paired_jimu_return_to_start_penalty_max"] = float(max_return_start_penalty)
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
        event = {
            "kind": str(kind),
            "status": status,
            "success": bool(success),
            "has_path": bool(getattr(result, "joint_path", None) is not None),
        }
        debug = getattr(result, "debug", None)
        if isinstance(debug, dict) and debug.get("jimu_linear_joint_transport_fallback"):
            event["fallback_source_status"] = str(debug.get("source_status", "") or "")
            diag = debug.get("source_start_collision_diag")
            if isinstance(diag, dict):
                event["fallback_valid_after_removing"] = list(diag.get("valid_after_removing") or [])[:8]
                event["fallback_world_obstacle_names"] = list(diag.get("world_obstacle_names") or [])[:16]
        events.append(event)


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
    empty_world_diag = None
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
    original_world = getattr(planner, "_world", None)
    if original_world is not None and hasattr(planner, "_world_without_obstacles"):
        try:
            for group_name, removed_names in group_specs.items():
                removed_names = [name for name in removed_names if name]
                if not removed_names:
                    continue
                ablated_world = planner._world_without_obstacles(original_world, excluded_names=set(removed_names))
                if hasattr(planner, "_world_obstacle_count") and planner._world_obstacle_count(ablated_world) <= 0:
                    planner.clear_world()
                else:
                    planner.motion_gen.update_world(ablated_world)
                    planner.ik_solver.update_world(ablated_world)
                group_valid, group_status = planner.check_start_state(start_q)
                group_ablation.append(
                    {
                        "removed_group": group_name,
                        "removed": removed_names[:32],
                        "valid": bool(group_valid),
                        "status": str(group_status),
                    }
                )
        except Exception as exc:
            group_ablation.append({"error": f"{type(exc).__name__}: {exc}"})
        finally:
            try:
                planner.motion_gen.update_world(original_world)
                planner.ik_solver.update_world(original_world)
                planner._world = original_world
                if hasattr(planner, "_update_cuda_graph_batch_ik_world"):
                    planner._update_cuda_graph_batch_ik_world(original_world)
            except Exception:
                pass
    original_world = getattr(planner, "_world", None)
    try:
        if hasattr(planner, "clear_world"):
            planner.clear_world()
            empty_valid, empty_status = planner.check_start_state(start_q)
            empty_world_diag = {
                "valid": bool(empty_valid),
                "status": str(empty_status),
            }
            disabled_empty_attached = []
            try:
                direct._cache_attached_spheres_for_contact(planner)
                disabled_empty_attached = direct._set_world_collision_for_links(
                    planner,
                    ["attached_object"],
                    enabled=False,
                    label=f"{label}_jimu_start_collision_diag_empty_attached_disabled",
                )
                empty_attached_valid, empty_attached_status = planner.check_start_state(start_q)
                empty_world_diag["attached_disabled"] = {
                    "valid": bool(empty_attached_valid),
                    "status": str(empty_attached_status),
                    "disabled_links": list(disabled_empty_attached or []),
                }
            finally:
                try:
                    direct._set_world_collision_for_links(
                        planner,
                        disabled_empty_attached,
                        enabled=True,
                        label=f"{label}_jimu_start_collision_diag_empty_attached_disabled",
                    )
                except Exception:
                    pass
                try:
                    direct._restore_attached_spheres_after_contact(planner)
                except Exception:
                    pass
            if original_world is not None:
                planner.motion_gen.update_world(original_world)
                planner.ik_solver.update_world(original_world)
                planner._world = original_world
                if hasattr(planner, "_update_cuda_graph_batch_ik_world"):
                    planner._update_cuda_graph_batch_ik_world(original_world)
    except Exception as exc:
        empty_world_diag = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            if original_world is not None:
                planner.motion_gen.update_world(original_world)
                planner.ik_solver.update_world(original_world)
                planner._world = original_world
                if hasattr(planner, "_update_cuda_graph_batch_ik_world"):
                    planner._update_cuda_graph_batch_ik_world(original_world)
        except Exception:
            pass
    attached_disabled_diag = None
    try:
        direct._cache_attached_spheres_for_contact(planner)
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
        try:
            direct._restore_attached_spheres_after_contact(planner)
        except Exception:
            pass
    link_ablation = []
    try:
        configured_links = [str(x) for x in list(getattr(planner, "configured_collision_links", []) or []) if str(x)]
        configured_links = [name for name in configured_links if name != "attached_object"]
        # Keep the caller's already-disabled contact-tolerant links disabled while
        # probing one additional robot link at a time.  If disabling one link makes
        # the start state valid, the attached payload is colliding with that link.
        base_disabled_links = [str(x) for x in list(disabled_world_collision_links or []) if str(x)]
        for link_name in configured_links:
            disabled_probe = []
            try:
                disabled_probe = direct._set_world_collision_for_links(
                    planner,
                    [*base_disabled_links, link_name],
                    enabled=False,
                    label=f"{label}_jimu_start_collision_diag_link_{link_name}",
                )
                link_valid, link_status = planner.check_start_state(start_q)
                entry = {
                    "disabled_link": str(link_name),
                    "valid": bool(link_valid),
                    "status": str(link_status),
                }
                if bool(link_valid) or "gripper" in str(link_name).lower() or "pad" in str(link_name).lower():
                    link_ablation.append(entry)
            except Exception as exc:
                link_ablation.append({"disabled_link": str(link_name), "error": f"{type(exc).__name__}: {exc}"})
            finally:
                try:
                    direct._set_world_collision_for_links(
                        planner,
                        disabled_probe,
                        enabled=True,
                        label=f"{label}_jimu_start_collision_diag_link_{link_name}",
                    )
                except Exception:
                    pass
    except Exception as exc:
        link_ablation = [{"error": f"{type(exc).__name__}: {exc}"}]
    bottom_z = None
    try:
        bottom_z = direct._attached_sphere_bottom_z(planner, start_q)
    except Exception:
        bottom_z = None
    attached_sphere_summary = {}
    attached_robot_sphere_contacts = []
    try:
        attached_spheres = list(planner.get_attached_spheres_world(start_q) or [])
        attached_sphere_summary = {
            "active": bool(getattr(planner, "attached_object_active", False)),
            "count": int(len(attached_spheres)),
            "enabled_count": int(planner.get_attached_sphere_count(link_name="attached_object")),
            "spheres": [
                {
                    "center": [float(v) for v in list(item.get("center", []))[:3]],
                    "radius": float(item.get("radius", 0.0) or 0.0),
                }
                for item in attached_spheres[:16]
            ],
        }
        robot_spheres = planner._compute_world_link_spheres(start_q)
        robot_link_names = list(planner._collision_sphere_link_names() or [])
        attached_indices = [
            idx for idx, name in enumerate(robot_link_names)
            if str(name) == "attached_object"
        ]
        if attached_indices and not attached_spheres:
            robot_arr = np.asarray(robot_spheres, dtype=np.float32).reshape(-1, 4)
            extracted = []
            for idx in attached_indices:
                if idx >= robot_arr.shape[0]:
                    continue
                row = robot_arr[idx]
                radius = float(row[3])
                if radius <= 0:
                    continue
                extracted.append({"center": [float(x) for x in row[:3].tolist()], "radius": radius})
            if extracted:
                attached_spheres = extracted
                attached_sphere_summary["count_from_link_spheres"] = int(len(extracted))
                attached_sphere_summary["spheres_from_link_spheres"] = extracted[:16]
        disabled_link_set = {str(x) for x in list(disabled_world_collision_links or []) if str(x)}
        attached_link_name = "attached_object"
        pairs = []
        for attached_idx, attached_sphere in enumerate(attached_spheres):
            attached_center = np.asarray(attached_sphere.get("center", [np.nan, np.nan, np.nan]), dtype=np.float32).reshape(3)
            attached_radius = float(attached_sphere.get("radius", 0.0) or 0.0)
            if not np.all(np.isfinite(attached_center)) or attached_radius <= 0:
                continue
            for sphere_idx, robot_sphere in enumerate(np.asarray(robot_spheres, dtype=np.float32).reshape(-1, 4)):
                link_name = str(robot_link_names[sphere_idx]) if sphere_idx < len(robot_link_names) else f"sphere_{sphere_idx}"
                if link_name == attached_link_name or link_name in disabled_link_set:
                    continue
                robot_radius = float(robot_sphere[3])
                if robot_radius <= 0:
                    continue
                robot_center = np.asarray(robot_sphere[:3], dtype=np.float32).reshape(3)
                center_dist = float(np.linalg.norm(attached_center - robot_center))
                clearance = center_dist - float(attached_radius + robot_radius)
                pairs.append(
                    {
                        "attached_sphere": int(attached_idx),
                        "robot_sphere": int(sphere_idx),
                        "robot_link": link_name,
                        "clearance_m": float(clearance),
                        "center_distance_m": float(center_dist),
                        "attached_radius_m": float(attached_radius),
                        "robot_radius_m": float(robot_radius),
                        "attached_center": [float(x) for x in attached_center.tolist()],
                        "robot_center": [float(x) for x in robot_center.tolist()],
                    }
                )
        pairs.sort(key=lambda item: float(item["clearance_m"]))
        attached_robot_sphere_contacts = pairs[:16]
    except Exception as exc:
        attached_robot_sphere_contacts = [{"error": f"{type(exc).__name__}: {exc}"}]
        attached_sphere_summary = {"error": f"{type(exc).__name__}: {exc}"}
    result = {
        "valid": bool((diag or {}).get("valid", False)),
        "status": str((diag or {}).get("status", "") or ""),
        "world_obstacle_names": [str(x) for x in list((diag or {}).get("world_obstacle_names", []) or [])[:32]],
        "ablation": ablation[:32],
        "group_ablation": group_ablation[:16],
        "empty_world_diag": empty_world_diag,
        "attached_disabled_diag": attached_disabled_diag,
        "link_ablation": link_ablation[:32],
        "attached_sphere_summary": attached_sphere_summary,
        "attached_robot_sphere_contacts": attached_robot_sphere_contacts,
        "valid_after_removing": valid_after_removing[:16],
        "attached_sphere_bottom_z": None if bottom_z is None else float(bottom_z),
        "virtual_table_top_z": float(getattr(args_ns, "curobo_table_z_offset", -0.01)),
        "candidate_label": str(first.get("label", "") or ""),
    }
    return result


def profile_plan_to_joint_state_jimu(planner, *args, **kwargs):
    original = _ORIGINAL_PROFILE_PLAN_TO_JOINT_STATE
    if original is None:
        result = planner.plan_to_joint_state(*args, **kwargs)
    else:
        result = original(planner, *args, **kwargs)
    if (
        not bool(getattr(result, "success", False))
        and _jimu_linear_transport_fallback_allowed(planner)
        and len(args) >= 2
    ):
        result = _jimu_make_linear_transport_result(planner, result, args[0], args[1])
    _record_jimu_motiongen_result(planner, "plan_to_joint_state", result)
    return result


def plan_return_to_start_joint_curobo_jimu(
    planner,
    demo,
    args,
    start_q,
    goal_q,
    *,
    label: str,
    extra_scene_obstacles: list[dict] | None = None,
) -> dict:
    original = _ORIGINAL_PLAN_RETURN_TO_START_JOINT_CUROBO
    if original is None:
        original = getattr(direct, "_plan_return_to_start_joint_curobo", None)
    if not callable(original):
        return {"success": False, "status": "NO_RETURN_PLANNER", "q_path": None}

    payload = original(
        planner,
        demo,
        args,
        start_q,
        goal_q,
        label=label,
        extra_scene_obstacles=extra_scene_obstacles,
    )
    if bool(payload.get("success", False)) or not bool(getattr(args, "jimu_return_to_start_curobo_retry", False)):
        return payload

    base_attempts = int(getattr(args, "curobo_max_attempts", 2))
    base_timeout = float(getattr(args, "curobo_timeout", 5.0))
    base_trajopt = int(getattr(args, "curobo_num_trajopt_seeds", 1))
    base_graph = int(getattr(args, "curobo_num_graph_seeds", 1))
    retry_attempts = max(base_attempts, int(getattr(args, "jimu_return_to_start_curobo_retry_max_attempts", 4) or 0))
    retry_timeout = max(base_timeout, float(getattr(args, "jimu_return_to_start_curobo_retry_timeout", 8.0) or 0.0))
    retry_trajopt = max(base_trajopt, int(getattr(args, "jimu_return_to_start_curobo_retry_trajopt_seeds", 4) or 0))
    retry_graph = max(base_graph, int(getattr(args, "jimu_return_to_start_curobo_retry_graph_seeds", 4) or 0))
    retry_enable_graph = bool(getattr(args, "jimu_return_to_start_curobo_retry_enable_graph", False))
    if (
        retry_attempts <= base_attempts
        and retry_timeout <= base_timeout + 1.0e-6
        and retry_trajopt <= base_trajopt
        and retry_graph <= base_graph
        and bool(getattr(args, "curobo_enable_graph", False)) == retry_enable_graph
    ):
        return payload

    status = str(payload.get("status", "FAILED"))
    print(
        f"[jimu return] {label}: cuRobo primary failed status={status}; retrying with "
        f"attempts={retry_attempts}, trajopt_seeds={retry_trajopt}, graph_seeds={retry_graph}, "
        f"timeout={retry_timeout:.1f}s, graph={retry_enable_graph}"
    )
    old_values = {
        "curobo_enable_graph": getattr(args, "curobo_enable_graph", None),
        "curobo_max_attempts": getattr(args, "curobo_max_attempts", None),
        "curobo_timeout": getattr(args, "curobo_timeout", None),
        "curobo_num_trajopt_seeds": getattr(args, "curobo_num_trajopt_seeds", None),
        "curobo_num_graph_seeds": getattr(args, "curobo_num_graph_seeds", None),
    }
    with direct._profile_stage(
        args,
        "jimu_return_to_start_curobo_retry",
        label=str(label),
        primary_status=status,
        retry_max_attempts=retry_attempts,
        retry_timeout=retry_timeout,
        retry_trajopt_seeds=retry_trajopt,
        retry_graph_seeds=retry_graph,
        retry_enable_graph=retry_enable_graph,
    ) as prof:
        try:
            args.curobo_enable_graph = retry_enable_graph
            args.curobo_max_attempts = retry_attempts
            args.curobo_timeout = retry_timeout
            args.curobo_num_trajopt_seeds = retry_trajopt
            args.curobo_num_graph_seeds = retry_graph
            retry_payload = original(
                planner,
                demo,
                args,
                start_q,
                goal_q,
                label=f"{label}_jimu_retry",
                extra_scene_obstacles=extra_scene_obstacles,
            )
        finally:
            for key, value in old_values.items():
                if value is None:
                    try:
                        delattr(args, key)
                    except AttributeError:
                        pass
                else:
                    setattr(args, key, value)

        prof["success"] = bool(retry_payload.get("success", False))
        prof["status"] = str(retry_payload.get("status", "FAILED"))
        prof["path_waypoints"] = int(retry_payload.get("path_waypoints", 0) or 0)
        prof["solve_time"] = retry_payload.get("solve_time")
        prof["trajopt_time"] = retry_payload.get("trajopt_time")
        prof["max_joint_error"] = retry_payload.get("max_joint_error")

    if bool(retry_payload.get("success", False)):
        retry_payload = dict(retry_payload)
        retry_payload["jimu_return_retry"] = True
        retry_payload["primary_status"] = status
        return retry_payload
    return payload


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
        results = planner.plan_batch_start_goal_pairs(*args, **kwargs)
    else:
        results = original(planner, *args, **kwargs)
    if _jimu_linear_transport_fallback_allowed(planner) and len(args) >= 2:
        start_qs = list(args[0] or [])
        goal_poses = list(args[1] or [])
        if start_qs and goal_poses:
            try:
                ik_results = planner.solve_batch_start_goal_ik(
                    start_qs,
                    goal_poses,
                    num_seeds=kwargs.get("num_ik_seeds"),
                    use_cuda_graph_batch=False,
                )
            except Exception as exc:
                print(f"[jimu-curobo] linear transport fallback could not solve hover IK batch: {exc}")
                ik_results = []
            patched = []
            for idx, result in enumerate(list(results or [])):
                if bool(getattr(result, "success", False)) or idx >= len(ik_results):
                    patched.append(result)
                    continue
                ik_result = ik_results[idx]
                if not bool(getattr(ik_result, "success", False)) or getattr(ik_result, "goal_joint", None) is None:
                    patched.append(result)
                    continue
                patched.append(_jimu_make_linear_transport_result(planner, result, start_qs[idx], ik_result.goal_joint))
            results = patched
    _record_jimu_motiongen_results(planner, "plan_batch_start_goal_pairs", results)
    return results


def evaluate_curobo_pose_candidates_multi_start_jimu(planner, *args, **kwargs):
    original = _ORIGINAL_EVALUATE_CUROBO_POSE_CANDIDATES_MULTI_START
    if original is None:
        raise RuntimeError("original _evaluate_curobo_pose_candidates_multi_start is not installed")
    label = str(kwargs.get("label", "") or "")
    planner._jimu_motiongen_capture_active = True
    planner._jimu_motiongen_capture_label = label
    planner._jimu_motiongen_capture_args = args[1] if len(args) >= 2 else kwargs.get("args", None)
    planner._jimu_motiongen_status_counts = {}
    planner._jimu_motiongen_failure_status_counts = {}
    planner._jimu_motiongen_success_status_counts = {}
    planner._jimu_motiongen_status_events = []
    planner._jimu_start_collision_diag = None
    planner._jimu_linear_transport_eval_active = True
    planner._jimu_motiongen_capture_disabled_links = kwargs.get("disabled_world_collision_links")
    try:
        winners = original(planner, *args, **kwargs)
    finally:
        planner._jimu_linear_transport_eval_active = False
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
    planner._jimu_motiongen_capture_args = None
    planner._jimu_motiongen_capture_disabled_links = None
    planner._jimu_linear_transport_eval_active = False


def _jimu_grid_axis_positions(dim: float, count: int, radius: float, span_scale: float) -> np.ndarray:
    count = max(1, int(count))
    usable_half = max(0.0, 0.5 * float(dim) * float(span_scale) - 0.5 * float(radius))
    if count <= 1 or usable_half <= 1e-6:
        return np.zeros((1,), dtype=np.float32)
    return np.linspace(-usable_half, usable_half, count, dtype=np.float32)


def _jimu_shape_is_triangle(shape_hint: str | None, dims: np.ndarray) -> bool:
    hint = str(shape_hint or "").strip().lower()
    if "triangle" in hint or "roof" in hint:
        return True
    dims = np.asarray(dims, dtype=np.float32).reshape(3)
    if dims.size != 3 or np.min(dims) <= 0:
        return False
    sorted_dims = np.sort(dims)
    # Jimu roof panels are roughly 135x74x6.5mm; square plates are 74x74x6mm.
    return bool(sorted_dims[-1] > 1.35 * sorted_dims[-2])


def _jimu_local_planar_payload_centers(
    *,
    dims: np.ndarray,
    radius: float,
    long_count: int,
    wide_count: int,
    span_scale: float,
    max_spheres: int,
    shape_hint: str | None,
) -> tuple[np.ndarray, int, int, int, str]:
    dims = np.asarray(dims, dtype=np.float32).reshape(3)
    thin_axis = int(np.argmin(dims))
    plane_axes = sorted([idx for idx in range(3) if idx != thin_axis], key=lambda idx: float(dims[idx]), reverse=True)
    long_axis, wide_axis = int(plane_axes[0]), int(plane_axes[1])
    max_spheres = max(0, int(max_spheres))
    if max_spheres <= 0:
        return np.zeros((0, 3), dtype=np.float32), thin_axis, long_axis, wide_axis, "none"

    long_half = float(_jimu_grid_axis_positions(float(dims[long_axis]), 2, radius, span_scale)[-1])
    wide_half = float(_jimu_grid_axis_positions(float(dims[wide_axis]), 2, radius, span_scale)[-1])
    if _jimu_shape_is_triangle(shape_hint, dims):
        raw_long_half = max(0.0, 0.5 * float(dims[long_axis]) * float(span_scale))
        raw_wide_half = max(0.0, 0.5 * float(dims[wide_axis]) * float(span_scale))
        apex = np.array([raw_long_half, 0.0], dtype=np.float32)
        base_left = np.array([-raw_long_half, -raw_wide_half], dtype=np.float32)
        base_right = np.array([-raw_long_half, raw_wide_half], dtype=np.float32)
        centroid = (apex + base_left + base_right) / 3.0

        def _inset_towards_inside(point: np.ndarray, distance: float) -> tuple[float, float]:
            direction = centroid - np.asarray(point, dtype=np.float32)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-8:
                return float(point[0]), float(point[1])
            inset = min(float(distance), max(0.0, norm * 0.9))
            center_2d = np.asarray(point, dtype=np.float32) + direction / norm * inset
            return float(center_2d[0]), float(center_2d[1])

        def _inset_edge_midpoint(point_a: np.ndarray, point_b: np.ndarray, distance: float) -> tuple[float, float]:
            midpoint = 0.5 * (np.asarray(point_a, dtype=np.float32) + np.asarray(point_b, dtype=np.float32))
            edge = np.asarray(point_b, dtype=np.float32) - np.asarray(point_a, dtype=np.float32)
            normal_a = np.array([-edge[1], edge[0]], dtype=np.float32)
            normal_b = -normal_a
            normal = normal_a if float(np.dot(normal_a, centroid - midpoint)) >= float(np.dot(normal_b, centroid - midpoint)) else normal_b
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-8:
                return _inset_towards_inside(midpoint, distance)
            center_2d = midpoint + normal / norm * float(distance)
            return float(center_2d[0]), float(center_2d[1])

        # Six available attached-object spheres fit a triangle well: three
        # inset vertices plus three inset edge midpoints.  The sphere boundary
        # reaches the triangle boundary while the centers stay inside the panel.
        tri_2d = [
            _inset_towards_inside(apex, radius),
            _inset_towards_inside(base_left, radius),
            _inset_towards_inside(base_right, radius),
            _inset_edge_midpoint(apex, base_left, radius),
            _inset_edge_midpoint(base_left, base_right, radius),
            _inset_edge_midpoint(base_right, apex, radius),
            (0.0, 0.0),
        ]
        layout = "triangle_inset_vertices_edges"
        chosen_2d = tri_2d[:max_spheres]
    else:
        if max_spheres <= 6:
            # Balanced six-point plate footprint: four corners and the two
            # wide-edge midpoints.  Do not truncate a larger row-major grid,
            # because that biases all spheres to one side of the held plate.
            plate_2d = [
                (-long_half, -wide_half),
                (-long_half, wide_half),
                (long_half, -wide_half),
                (long_half, wide_half),
                (0.0, -wide_half),
                (0.0, wide_half),
                (0.0, 0.0),
            ]
            layout = "plate_corners_edges"
            chosen_2d = plate_2d[:max_spheres]
        else:
            long_positions = _jimu_grid_axis_positions(float(dims[long_axis]), int(long_count), radius, span_scale)
            wide_positions = _jimu_grid_axis_positions(float(dims[wide_axis]), int(wide_count), radius, span_scale)
            grid_2d = [(float(long_pos), float(wide_pos)) for long_pos in long_positions for wide_pos in wide_positions]
            if len(grid_2d) > max_spheres:
                # Prefer extremes and the center before filling remaining slots.
                preferred = [
                    (-long_half, -wide_half),
                    (-long_half, wide_half),
                    (long_half, -wide_half),
                    (long_half, wide_half),
                    (0.0, -wide_half),
                    (0.0, wide_half),
                    (-long_half, 0.0),
                    (long_half, 0.0),
                    (0.0, 0.0),
                ]
                chosen_2d = []
                for target in preferred:
                    nearest = min(
                        grid_2d,
                        key=lambda point: (point[0] - target[0]) ** 2 + (point[1] - target[1]) ** 2,
                    )
                    if nearest not in chosen_2d:
                        chosen_2d.append(nearest)
                    if len(chosen_2d) >= max_spheres:
                        break
                for point in grid_2d:
                    if point not in chosen_2d:
                        chosen_2d.append(point)
                    if len(chosen_2d) >= max_spheres:
                        break
            else:
                chosen_2d = grid_2d
            layout = "plate_grid"

    local_centers = []
    for long_pos, wide_pos in chosen_2d[:max_spheres]:
        center = np.zeros(3, dtype=np.float32)
        center[long_axis] = float(long_pos)
        center[wide_axis] = float(wide_pos)
        local_centers.append(center)
    return np.asarray(local_centers, dtype=np.float32).reshape(-1, 3), thin_axis, long_axis, wide_axis, layout


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
    shape_hint: str | None = None,
    link_name: str = "attached_object",
) -> bool:
    if not getattr(planner, "collision_enabled", False):
        return False
    planner._invalidate_cuda_graph_batch_ik_solvers()
    torch = planner.mods["torch"]
    q_np = planner._normalize_q(q)
    dims = np.asarray(dims, dtype=np.float32).reshape(3)
    object_pose_base = np.asarray(object_pose_base, dtype=np.float32).reshape(4, 4)
    max_spheres = int(planner.motion_gen.robot_cfg.kinematics.kinematics_config.get_number_of_spheres(str(link_name)))
    local_centers, thin_axis, long_axis, wide_axis, layout = _jimu_local_planar_payload_centers(
        dims=dims,
        radius=radius,
        long_count=long_count,
        wide_count=wide_count,
        span_scale=span_scale,
        max_spheres=max_spheres,
        shape_hint=shape_hint,
    )

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
        f"layout={layout}, shape={str(shape_hint or 'plate')}, "
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
        try:
            asset_file = getattr(args, "sim_asset_file", None) or getattr(args, "mesh_file", None)
            asset_scale = float(getattr(args, "sim_asset_scale", None) or getattr(args, "mesh_scale", 1.0) or 1.0)
            raw_dims = np.asarray(
                direct.targeted.base.get_asset_box_size(str(Path(str(asset_file)).expanduser()), asset_scale),
                dtype=np.float32,
            ).reshape(3)
        except Exception:
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
            shape_hint=source,
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
                    shape_hint=source,
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


def plan_short_tcp_up_axis_lift_ik_jimu(
    planner,
    demo,
    args,
    start_q,
    grasp_choice,
    *,
    lift_m: float,
    label: str,
    include_table: bool = True,
    exclude_object_names: set[str] | None = None,
    disabled_world_collision_links: list[str] | None = None,
):
    original = _ORIGINAL_PLAN_SHORT_TCP_UP_AXIS_LIFT_IK
    source = direct.curobo_wrapper.normalize_object_name(getattr(args, "object_name", None))
    if source in set(JIMU_PICK_ROLES) and bool(getattr(args, "jimu_lift_world_z_only", True)):
        print(f"[jimu-lift] {label}: using world-Z lift instead of TCP-axis lift")
        return direct._plan_short_world_z_lift_ik(
            planner,
            demo,
            args,
            start_q,
            lift_m=float(lift_m),
            label=f"{label}_world_z",
            include_table=include_table,
            exclude_object_names=exclude_object_names,
            disabled_world_collision_links=disabled_world_collision_links,
        )
    if callable(original):
        return original(
            planner,
            demo,
            args,
            start_q,
            grasp_choice,
            lift_m=lift_m,
            label=label,
            include_table=include_table,
            exclude_object_names=exclude_object_names,
            disabled_world_collision_links=disabled_world_collision_links,
        )
    return None


def build_hover_pose_jimu(release_pose, place_mode: str, args, rule, *, candidate_pre_place_pose=None):
    original = _ORIGINAL_BUILD_HOVER_POSE
    source = direct.curobo_wrapper.normalize_object_name(getattr(args, "object_name", None))
    if source in set(JIMU_PICK_ROLES) and bool(getattr(args, "jimu_lift_world_z_only", True)):
        place_mode_text = str(place_mode or "")
        if place_mode_text == "drop_place":
            return release_pose
        if place_mode_text == "vertical_place":
            hover_height = float(max(getattr(args, "vertical_place_hover_height_m", 0.040), 0.0))
        elif place_mode_text == "surface_place":
            hover_height = float(max(getattr(args, "surface_place_hover_height_m", 0.050), 0.0))
        else:
            hover_height = float(max(getattr(rule, "hover_height", 0.05), 0.0))
        if hover_height <= 1e-8:
            return release_pose
        print(f"[jimu-lift] build_hover_pose: using world-Z hover for {source} ({hover_height:.3f}m)")
        return direct._lift_pose_world_z(release_pose, hover_height)
    if callable(original):
        return original(
            release_pose,
            place_mode,
            args,
            rule,
            candidate_pre_place_pose=candidate_pre_place_pose,
        )
    return release_pose


_JIMU_SOURCE_RETRY_POSE_FIELDS = (
    "T_cam_obj",
    "T_world_obj",
    "jimu_T_base_obj",
    "translation_m",
    "box",
    "jimu_tray_slot_index",
    "sam3_instance_index",
)


def _jimu_entry_slot_index(role: str, entry: dict | None, role_order: list[str]) -> int | None:
    if isinstance(entry, dict):
        try:
            return int(entry.get("jimu_tray_slot_index"))
        except Exception:
            pass
    try:
        return int(role_order.index(role))
    except Exception:
        return None


def _jimu_copy_cache_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _jimu_swap_cache_pose_fields(objects: dict, role_a: str, role_b: str) -> None:
    entry_a = objects.get(role_a)
    entry_b = objects.get(role_b)
    if not isinstance(entry_a, dict) or not isinstance(entry_b, dict):
        return
    source_a = direct.curobo_wrapper.normalize_object_name(entry_a.get("jimu_physical_source_role")) or role_a
    source_b = direct.curobo_wrapper.normalize_object_name(entry_b.get("jimu_physical_source_role")) or role_b
    for field in _JIMU_SOURCE_RETRY_POSE_FIELDS:
        has_a = field in entry_a
        has_b = field in entry_b
        value_a = _jimu_copy_cache_value(entry_a.get(field)) if has_a else None
        value_b = _jimu_copy_cache_value(entry_b.get(field)) if has_b else None
        if has_b:
            entry_a[field] = value_b
        elif has_a:
            entry_a.pop(field, None)
        if has_a:
            entry_b[field] = value_a
        elif has_b:
            entry_b.pop(field, None)
    entry_a["jimu_physical_source_role"] = source_b
    entry_b["jimu_physical_source_role"] = source_a
    entry_a["jimu_source_swapped_from_role"] = role_b
    entry_b["jimu_source_swapped_from_role"] = role_a


def _jimu_apply_cached_pose_to_registry_actor(demo, role: str, entry: dict | None) -> None:
    if not isinstance(entry, dict) or entry.get("T_world_obj") is None:
        return
    registry_func = getattr(direct, "_single_scene_registry", None)
    registry = registry_func(demo) if callable(registry_func) else {}
    actor = (registry.get(role) or {}).get("actor") if isinstance(registry, dict) else None
    if actor is None:
        return
    try:
        T_world_obj = np.asarray(entry["T_world_obj"], dtype=np.float32).reshape(4, 4)
        actor.set_pose(direct._pose_from_world_matrix(T_world_obj))
        zero_fn = getattr(direct, "_single_scene_zero_actor_velocity", None)
        if callable(zero_fn):
            zero_fn(actor)
    except Exception as exc:
        print(f"[jimu-source-retry] warning: failed to move actor for {role}: {exc}")


def _jimu_ensure_cached_world_pose_from_base(demo, args: argparse.Namespace, entry: dict | None) -> bool:
    if not isinstance(entry, dict):
        return False
    if entry.get("T_world_obj") is not None:
        return True
    if entry.get("jimu_T_base_obj") is None:
        return False
    try:
        T_base_obj = np.asarray(entry["jimu_T_base_obj"], dtype=np.float32).reshape(4, 4)
        T_world_obj = T_base_obj.copy()
        if not bool(getattr(args, "no_map_foundationpose_through_robot_base", False)):
            T_world_obj = (direct._get_robot_base_world_transform(demo) @ T_world_obj).astype(np.float32)
        offset = np.asarray(
            getattr(args, "foundationpose_position_offset", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        ).reshape(-1)
        if offset.size >= 3:
            T_world_obj[:3, 3] += offset[:3]
        entry["T_world_obj"] = T_world_obj.astype(np.float32)
        return True
    except Exception as exc:
        print(f"[jimu-source-retry] warning: failed to derive world pose for {entry.get('object_name', 'object')}: {exc}")
        return False


def _jimu_choose_next_tray_source_role(
    args: argparse.Namespace,
    scene_capture_cache: dict,
    target_role: str,
) -> tuple[str | None, int | None, int | None]:
    objects = scene_capture_cache.get("objects")
    if not isinstance(objects, dict):
        return None, None, None
    role_order = _jimu_tray_slot_role_order(args)
    target_entry = objects.get(target_role)
    current_slot = _jimu_entry_slot_index(target_role, target_entry, role_order)
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
    attempted_set = {int(v) for v in list(attempted or []) if isinstance(v, (int, np.integer)) or str(v).lstrip("-").isdigit()}
    attempted_set.add(int(current_slot))
    attempted_by_target[target_role] = sorted(attempted_set)

    total_slots = max(len(role_order), 1)
    candidates: list[tuple[int, int, str]] = []
    for role in role_order:
        role_name = direct.curobo_wrapper.normalize_object_name(role)
        if role_name is None or role_name == target_role:
            continue
        entry = objects.get(role_name)
        if not isinstance(entry, dict) or bool(entry.get("placed", False)):
            continue
        if entry.get("T_world_obj") is None and entry.get("T_cam_obj") is None and entry.get("jimu_T_base_obj") is None:
            continue
        slot_idx = _jimu_entry_slot_index(role_name, entry, role_order)
        if slot_idx is None or int(slot_idx) in attempted_set:
            continue
        distance = (int(slot_idx) - int(current_slot)) % total_slots
        if distance <= 0:
            distance += total_slots
        candidates.append((distance, int(slot_idx), role_name))
    if not candidates:
        return None, current_slot, None
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, donor_slot, donor_role = candidates[0]
    return donor_role, current_slot, donor_slot


def _jimu_should_retry_next_source_after_failure(args: argparse.Namespace) -> bool:
    source = direct.curobo_wrapper.normalize_object_name(getattr(args, "object_name", None))
    if source not in set(JIMU_PICK_ROLES):
        return False
    if not bool(getattr(args, "jimu_retry_next_tray_source_on_grasp_failure", True)):
        return False
    if bool(getattr(args, "_episode_object_grasped", False)):
        return False
    if bool(getattr(args, "_episode_real_motion_started", False)):
        return False
    failure_kind = str(getattr(args, "_episode_failure_kind", "") or "")
    failure_phase = str(getattr(args, "_episode_failure_phase", "") or "")
    if failure_kind == "empty_grasp_after_lift":
        return False
    return True


def _jimu_prepare_next_tray_source_retry(
    demo,
    args: argparse.Namespace,
    scene_capture_cache,
    selected_name: str,
) -> bool:
    target_role = direct.curobo_wrapper.normalize_object_name(selected_name)
    if target_role is None or not isinstance(scene_capture_cache, dict):
        return False
    objects = scene_capture_cache.get("objects")
    if not isinstance(objects, dict) or target_role not in objects:
        return False
    failure_kind = str(getattr(args, "_episode_failure_kind", "") or "")
    failure_phase = str(getattr(args, "_episode_failure_phase", "") or "")
    is_planning_failure = (
        failure_kind
        in {
            "",
            "planning_no_complete_chain",
            "transport_hover_no_winners",
            "post_place_clearance_plan_fail",
            "release_plan_fail",
        }
        or any(token in failure_phase for token in ("transport", "pre_place", "release", "place", "return"))
    )
    if is_planning_failure:
        max_planning_retries = int(getattr(args, "jimu_planning_failure_source_retry_max", -1))
        retry_meta = scene_capture_cache.setdefault("_jimu_source_retry_meta", {})
        planning_counts = retry_meta.setdefault("planning_failure_retry_count_by_target", {})
        used_count = int(planning_counts.get(target_role, 0) or 0)
        if max_planning_retries >= 0 and used_count >= max_planning_retries:
            print(
                "[jimu-source-retry] planning failure source retry limit reached; "
                f"target={target_role}, used={used_count}, max={max_planning_retries}, "
                f"kind={failure_kind or 'unset'}, phase={failure_phase or 'unset'}"
            )
            try:
                direct._record_profile(
                    args,
                    "jimu_source_retry_swap",
                    success=False,
                    status="PLANNING_RETRY_LIMIT_REACHED",
                    target_name=target_role,
                    planning_retry_count=used_count,
                    planning_retry_max=max_planning_retries,
                    failure_kind=failure_kind,
                    failure_phase=failure_phase,
                )
            except Exception:
                pass
            return False
        planning_counts[target_role] = used_count + 1
    donor_role, current_slot, donor_slot = _jimu_choose_next_tray_source_role(args, scene_capture_cache, target_role)
    if donor_role is None:
        print(
            f"[jimu-source-retry] {target_role}: no unused tray source left after failed slot {current_slot}; "
            "falling back to normal target selection"
        )
        try:
            direct._record_profile(
                args,
                "jimu_source_retry_swap",
                success=False,
                status="NO_UNUSED_TRAY_SOURCE",
                target_name=target_role,
                from_slot_index=current_slot,
            )
        except Exception:
            pass
        return False

    _jimu_swap_cache_pose_fields(objects, target_role, donor_role)
    _jimu_ensure_cached_world_pose_from_base(demo, args, objects.get(target_role))
    _jimu_ensure_cached_world_pose_from_base(demo, args, objects.get(donor_role))
    _jimu_apply_cached_pose_to_registry_actor(demo, target_role, objects.get(target_role))
    _jimu_apply_cached_pose_to_registry_actor(demo, donor_role, objects.get(donor_role))
    refresh_fn = getattr(direct, "_single_scene_refresh_after_pose_change", None)
    if callable(refresh_fn):
        try:
            refresh_fn(demo)
        except Exception:
            pass

    scene_capture_cache["_jimu_retry_target_after_source_swap"] = {
        "target_name": target_role,
        "donor_role": donor_role,
        "from_slot_index": current_slot,
        "donor_slot_index": donor_slot,
    }
    print(
        "[jimu-source-retry] grasp planning failed before motion; remapped logical target "
        f"{target_role} from tray slot {current_slot} to donor {donor_role} slot {donor_slot}"
    )
    try:
        direct._record_profile(
            args,
            "jimu_source_retry_swap",
            success=True,
            status="SCHEDULED_RETRY",
            target_name=target_role,
            donor_role=donor_role,
            from_slot_index=current_slot,
            donor_slot_index=donor_slot,
        )
    except Exception:
        pass
    return True


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


def _jimu_pregrasp_partial_open_enabled(args: argparse.Namespace | None) -> bool:
    return bool(args is not None and getattr(args, "jimu_partial_open_before_grasp", True))


def _jimu_pregrasp_partial_open_value(args: argparse.Namespace) -> float:
    open_value = float(getattr(args, "real_gripper_open", 0.0))
    close_value = float(getattr(args, "real_gripper_close", 0.91))
    fraction_closed = float(np.clip(getattr(args, "jimu_pregrasp_open_fraction", 0.79), 0.0, 1.0))
    return float(open_value + (close_value - open_value) * fraction_closed)


def _jimu_pregrasp_partial_open_sim_value(args: argparse.Namespace) -> float:
    open_value = float(getattr(args, "gripper_open", -1.0))
    close_value = float(getattr(args, "gripper_close", 1.0))
    fraction_closed = float(np.clip(getattr(args, "jimu_pregrasp_open_fraction", 0.79), 0.0, 1.0))
    return float(open_value + (close_value - open_value) * fraction_closed)


def _jimu_should_use_partial_open_for_grasp(label: str, gripper_pos: float, args: argparse.Namespace | None) -> bool:
    if not _jimu_pregrasp_partial_open_enabled(args):
        return False
    if args is None:
        return False
    label_l = str(label or "").lower()
    if "grasp" not in label_l:
        return False
    if "post_grasp" in label_l or label_l.startswith("post_") or "post_grasp_lift" in label_l:
        return False
    full_open = float(getattr(args, "real_gripper_open", 0.0))
    return abs(float(gripper_pos) - full_open) <= 1e-6


def _jimu_refresh_after_no_step_gripper_sync(demo) -> None:
    try:
        direct.targeted.base.refresh_frozen_active_object_pose(demo)
    except Exception:
        pass
    try:
        demo.refresh_runtime_handles(rebuild_visual=False)
    except Exception:
        pass
    try:
        direct.targeted.base.sync_planner_qpos_from_demo(demo)
    except Exception:
        pass
    try:
        direct.targeted.base.update_attached_box_visual(demo)
    except Exception:
        pass


def _jimu_skip_physics_gripper_sync(args: argparse.Namespace | None) -> bool:
    return bool(getattr(args, "jimu_skip_physics_gripper_sync", True) if args is not None else True)


def _jimu_set_sim_gripper_visual(
    demo,
    sim_value: float,
    args: argparse.Namespace | None,
    *,
    label: str,
) -> None:
    open_sim = float(getattr(args, "gripper_open", -1.0) if args is not None else -1.0)
    close_sim = float(getattr(args, "gripper_close", 1.0) if args is not None else 1.0)
    denom = close_sim - open_sim
    if abs(denom) <= 1e-9:
        fraction_closed = 0.0
    else:
        fraction_closed = float(np.clip((float(sim_value) - open_sim) / denom, 0.0, 1.0))
    close_q = float(getattr(args, "real_gripper_close", 0.91) if args is not None else 0.91)
    open_q = 0.0
    gripper_q = float(open_q + (close_q - open_q) * fraction_closed)
    try:
        full_q = direct.targeted.base.flatten_np(demo.robot.get_qpos()).astype(np.float32).copy()
        active_names = list(getattr(demo, "active_joint_names", []) or [])
        if not active_names:
            try:
                active_names = [joint.get_name() for joint in demo.robot.get_active_joints()]
            except Exception:
                active_names = []
        arm_indices = set(int(i) for i in list(getattr(demo, "arm_indices", []) or []))
        changed = 0
        for idx, name in enumerate(active_names):
            if idx >= full_q.size or idx in arm_indices:
                continue
            if "gripper" not in str(name).lower():
                continue
            full_q[idx] = np.float32(gripper_q)
            changed += 1
        if changed == 0 and full_q.size > 7:
            full_q[7:] = np.float32(gripper_q)
            changed = int(full_q.size - 7)
        if changed == 0:
            raise RuntimeError("no gripper qpos entries found")
        demo.robot.set_qpos(full_q)
        set_qvel = getattr(demo.robot, "set_qvel", None)
        if callable(set_qvel):
            try:
                set_qvel(np.zeros_like(full_q, dtype=np.float32))
            except Exception:
                pass
        _jimu_refresh_after_no_step_gripper_sync(demo)
        gap = direct.targeted.base.get_sim_gripper_pad_gap(demo)
        gap_text = "unknown" if gap is None else f"{gap:.4f} m"
        print(
            "[jimu gripper] synced simulated gripper visual: "
            f"label={label}, action_value={float(sim_value):.3f}, qpos={gripper_q:.3f}, "
            f"joints={changed}, pad_gap={gap_text}"
        )
    except Exception as exc:
        print(f"[warn] failed to sync simulated Jimu gripper visual for {label}: {exc}")
        _jimu_refresh_after_no_step_gripper_sync(demo)


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
    if _jimu_should_use_partial_open_for_grasp(stage, gripper_pos, args):
        partial = _jimu_pregrasp_partial_open_value(args)
        print(
            "[jimu gripper] grasp approach uses partial open to avoid tray/neighbor collision: "
            f"{float(gripper_pos):.3f} -> {partial:.3f}"
        )
        return _ORIGINAL_REALMAN_SET_GRIPPER(self, partial, repeats=repeats, hz=hz)
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
    if _jimu_skip_physics_gripper_sync(args):
        if stage == "place_open_gripper" and not bool(closed) and _jimu_partial_release_enabled(args):
            sim_value = _jimu_partial_release_sim_gripper_value(args)
            label = "place_partial_open"
        elif (
            not bool(closed)
            and bool(getattr(_JIMU_RUNTIME_CONTEXT, "failed_attempt_restore", False))
            and not bool(getattr(args, "_episode_object_grasped", False))
            and _jimu_pregrasp_partial_open_enabled(args)
        ):
            sim_value = _jimu_pregrasp_partial_open_sim_value(args)
            label = "restore_pregrasp_partial"
        else:
            sim_value = float(getattr(args, "gripper_close", 1.0) if bool(closed) else getattr(args, "gripper_open", -1.0))
            label = "closed" if bool(closed) else "opened"
        _jimu_set_sim_gripper_visual(demo, sim_value, args, label=f"{stage or 'gripper_sync'}_{label}")
        _jimu_refresh_after_no_step_gripper_sync(demo)
        if label == "place_partial_open":
            state = "partially opened for place release"
        elif label == "restore_pregrasp_partial":
            state = "kept at pregrasp partial open during failed-attempt restore"
        else:
            state = "closed" if bool(closed) else "opened"
        print(f"[jimu gripper] lightweight sim gripper sync; marked sim gripper {state} for planning refresh")
        return
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


def settle_released_active_object_for_scene_cache_jimu(demo, args) -> np.ndarray | None:
    demo._freeze_active_object_before_grasp = False
    demo._frozen_active_object_pose = None
    try:
        demo.refresh_runtime_handles(rebuild_visual=False)
    except Exception:
        pass
    try:
        direct.targeted.base.sync_planner_qpos_from_demo(demo)
    except Exception:
        pass
    try:
        direct.targeted.base.update_attached_box_visual(demo, visible=False)
    except Exception:
        pass
    T_world_obj = direct.targeted.base.remember_current_active_object_world_pose_for_scene_cache(demo)
    if T_world_obj is not None:
        print(
            "[jimu settle] remembered released object pose without extra physics settle: "
            f"{np.round(T_world_obj[:3, 3], 6).tolist()}"
        )
    return T_world_obj


def _jimu_render_dry_run_motion_enabled(args, real_exec) -> bool:
    if real_exec is not None:
        return False
    if bool(getattr(args, "execute_real", False)):
        return False
    if bool(getattr(args, "_planning_prefetch_capture_only", False)):
        return False
    if not bool(getattr(args, "jimu_render_dry_run_motion", True)):
        return False
    return str(getattr(args, "render_mode", "") or "") == "human"


def _jimu_estimate_render_motion_duration_s(q_start, q_path, args) -> float:
    points = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in list(q_path or [])]
    if not points:
        return 0.0
    estimate_fn = getattr(direct, "_estimate_real_waypoint_stream_duration_s", None)
    if callable(estimate_fn):
        try:
            base_s = float(estimate_fn(q_start, points, args))
        except Exception:
            base_s = 0.0
    else:
        base_s = 0.0
    if base_s <= 1e-6:
        hz = float(max(getattr(args, "real_control_hz", 10.0), 1e-3))
        max_delta = float(max(getattr(args, "real_max_delta_per_step", 0.03), 1e-6))
        prev = np.asarray(q_start, dtype=np.float32).reshape(-1)[:7]
        steps = 0
        for q in points:
            steps += max(1, int(np.ceil(float(np.max(np.abs(q - prev))) / max_delta)))
            prev = q
        base_s = float(max(steps, 1)) / hz
    scale = float(max(getattr(args, "jimu_render_motion_scale", 0.25), 0.0))
    duration_s = base_s * scale
    if points:
        duration_s = max(duration_s, float(max(getattr(args, "jimu_render_motion_min_s", 0.25), 0.0)))
    max_s = float(getattr(args, "jimu_render_motion_max_s", 1.5) or 0.0)
    if max_s > 0.0:
        duration_s = min(duration_s, max_s)
    return float(max(duration_s, 0.0))


def _jimu_render_motion_frame(demo, bridge_mod, args, q: np.ndarray, *, use_attach: bool) -> None:
    direct.targeted.base.sync_demo_arm_qpos(demo, np.asarray(q, dtype=np.float32).reshape(-1)[:7])
    if use_attach:
        try:
            direct.targeted.base.force_active_object_to_attached_pose(demo)
        except Exception:
            pass
        try:
            direct.targeted.base.update_attached_box_visual(demo)
        except Exception:
            pass
    try:
        direct.targeted.base.update_robot_collision_sphere_visuals(demo, args)
    except Exception:
        pass
    try:
        direct.targeted.base.update_attached_object_collision_sphere_visuals(demo, args, visible=bool(use_attach))
    except Exception:
        pass
    try:
        bridge_mod.render_preview(demo.env, repeats=1)
    except Exception:
        try:
            demo.env.render()
        except Exception:
            pass


def _jimu_play_rendered_dry_run_motion(demo, bridge_mod, label: str, q_start, q_path, args, *, use_attach: bool) -> None:
    q_points = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in list(q_path or [])]
    if not q_points:
        return
    q0 = np.asarray(q_start, dtype=np.float32).reshape(-1)[:7]
    points = [q0.copy()]
    for q in q_points:
        if not np.allclose(points[-1], q, atol=1e-6, rtol=0.0):
            points.append(q.copy())
    if len(points) == 1:
        _jimu_render_motion_frame(demo, bridge_mod, args, points[-1], use_attach=use_attach)
        return
    duration_s = _jimu_estimate_render_motion_duration_s(q0, q_points, args)
    fps = float(np.clip(float(getattr(args, "jimu_render_motion_fps", 18.0) or 18.0), 5.0, 30.0))
    if duration_s <= 1e-6:
        frame_count = min(max(len(points), 2), 24)
    else:
        frame_count = max(len(points), int(np.ceil(duration_s * fps)) + 1)
        frame_count = min(frame_count, int(max(getattr(args, "jimu_render_motion_max_frames", 48), 2)))
    seg_len = [float(np.max(np.abs(points[i + 1] - points[i]))) for i in range(len(points) - 1)]
    total = float(sum(seg_len))
    if total <= 1e-9:
        _jimu_render_motion_frame(demo, bridge_mod, args, points[-1], use_attach=use_attach)
        return
    cumulative = np.cumsum([0.0] + seg_len)
    print(
        f"[jimu render] playing dry-run motion for {label}: "
        f"{duration_s:.2f}s, frames={frame_count}, waypoints={len(q_points)}, attach={bool(use_attach)}"
    )
    start_t = time.perf_counter()
    for frame_idx in range(frame_count):
        progress = total * float(frame_idx) / float(max(frame_count - 1, 1))
        seg_idx = int(np.searchsorted(cumulative, progress, side="right") - 1)
        seg_idx = max(0, min(seg_idx, len(points) - 2))
        denom = max(float(cumulative[seg_idx + 1] - cumulative[seg_idx]), 1e-9)
        alpha = float((progress - cumulative[seg_idx]) / denom)
        q = (1.0 - alpha) * points[seg_idx] + alpha * points[seg_idx + 1]
        _jimu_render_motion_frame(demo, bridge_mod, args, q.astype(np.float32), use_attach=use_attach)
        if duration_s > 1e-6:
            next_t = start_t + duration_s * float(frame_idx + 1) / float(max(frame_count - 1, 1))
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
    _jimu_render_motion_frame(demo, bridge_mod, args, points[-1], use_attach=use_attach)


def _jimu_execute_pose_path_stage_base(
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
    q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in list(q_path or [])]
    if _jimu_render_dry_run_motion_enabled(args, real_exec):
        if not q_path:
            q_current = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
            print(f"[planner] {label} is a zero-length pose path; skipping execution")
            return True, q_current
        if not skip_confirmation and not direct.targeted.base.confirm_planned_motion_or_skip(
            demo,
            bridge_mod,
            label,
            pose,
            q_path[-1],
            args,
            q_preview_path=q_path,
        ):
            print(f"[abort] user cancelled before executing {label}")
            return False, None
        if skip_confirmation:
            print(f"[planner] auto-executing {label} without separate preview/confirmation")
        q_start = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
        _jimu_play_rendered_dry_run_motion(
            demo,
            bridge_mod,
            str(label),
            q_start,
            q_path,
            args,
            use_attach=use_attach,
        )
        return True, q_path[-1]
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
    if _jimu_should_use_partial_open_for_grasp(label_text, gripper_pos, args):
        partial = _jimu_pregrasp_partial_open_value(args)
        sim_partial = _jimu_pregrasp_partial_open_sim_value(args)
        print(
            "[jimu gripper] grasp motion keeps gripper partially open: "
            f"real={partial:.3f}, sim={sim_partial:.3f}, label={label_text}"
        )
        gripper_pos = partial
        if _jimu_skip_physics_gripper_sync(args):
            _jimu_set_sim_gripper_visual(
                demo,
                sim_partial,
                args,
                label=f"{label_text}_pregrasp_partial",
            )
            print("[jimu gripper] lightweight pregrasp partial-open sync for portable dry-run")
        else:
            try:
                demo.hold_current_and_set_gripper(sim_partial, steps=max(int(getattr(args, "sim_gripper_sync_min_steps", 8)), 1))
            except Exception as exc:
                print(f"[warn] failed to set simulated pregrasp partial gripper: {exc}")
        if real_exec is not None:
            try:
                if _ORIGINAL_REALMAN_SET_GRIPPER is not None:
                    _ORIGINAL_REALMAN_SET_GRIPPER(
                        real_exec,
                        partial,
                        repeats=int(getattr(args, "real_gripper_command_repeats", 2)),
                        hz=float(getattr(args, "real_gripper_command_hz", 10.0)),
                    )
                else:
                    real_exec.set_gripper(partial)
                print("[jimu gripper] real gripper pre-set to partial open before grasp motion")
            except Exception as exc:
                print(f"[warn] failed to pre-set real gripper partial open before grasp motion: {exc}")
    if label_text != "post_place_clearance" or not _jimu_partial_release_enabled(args):
        return _jimu_execute_pose_path_stage_base(
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
    full_open_after_clearance = bool(getattr(args, "jimu_full_open_after_post_place_clearance", True))
    full_open_text = f"full open after clearance: {full_open:.3f}" if full_open_after_clearance else "no full open after clearance"
    print(
        "[jimu gripper] post-place clearance keeps partial open during lift: "
        f"{partial:.3f}; {full_open_text}"
    )
    ok, q_sent = _jimu_execute_pose_path_stage_base(
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
    if ok and full_open_after_clearance:
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


def execute_joint_path_stage_jimu(
    demo,
    bridge_mod,
    real_exec,
    label: str,
    q_path,
    gripper_pos: float,
    args,
    *,
    use_attach: bool = False,
    allow_start_in_collision: bool = False,
):
    if _ORIGINAL_EXECUTE_JOINT_PATH_STAGE is None:
        raise RuntimeError("Jimu execute_joint_path_stage wrapper was installed before original function was captured")
    q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in list(q_path or [])]
    if _jimu_render_dry_run_motion_enabled(args, real_exec):
        if not q_path:
            q_current = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
            print(f"[planner] {label} is a zero-length joint path; skipping execution")
            return True, q_current
        if not direct.targeted.base.confirm_joint_path_motion(demo, bridge_mod, label, q_path, args):
            print(f"[abort] user cancelled before executing {label}")
            return False, None
        q_start = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
        _jimu_play_rendered_dry_run_motion(
            demo,
            bridge_mod,
            str(label),
            q_start,
            q_path,
            args,
            use_attach=use_attach,
        )
        return True, q_path[-1]
    return _ORIGINAL_EXECUTE_JOINT_PATH_STAGE(
        demo,
        bridge_mod,
        real_exec,
        label,
        q_path,
        gripper_pos,
        args,
        use_attach=use_attach,
        allow_start_in_collision=allow_start_in_collision,
    )


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
        bridge_script_path=str(PICK_JIAOBANG_DIR / "rm75_jiaobang_pick_move_with_foundationpose.py"),
        pick_script_path=str(PICK_JIAOBANG_DIR / "rm75_jiaobang_pick_move_v10_perpendicular_to_object.py"),
        target_selection_order="cycle",
        repeat_count=len(JIMU_PICK_ROLES),
        skip_return_to_cycle_start_after_final_place=False,
        next_cycle_plan_prefetch=False,
        lerobot_sim2real_root=str(PORTABLE_LEROBOT_SIM2REAL_ROOT),
        urdf_path=str(PORTABLE_MANISKILL_RM75_URDF),
        srdf_path=str(PORTABLE_MANISKILL_RM75_SRDF),
        camera_extrinsic_opencv_path=str(PORTABLE_CAMERA_EXTRINSIC),
        curobo_rm75_robot_cfg=PICK_JIAOBANG_DIR / "curobo_rm75_config" / "rm75.yml",
        curobo_rm75_urdf=PORTABLE_MANISKILL_RM75_PLANNING_URDF,
        sam6d_provider_script=str(PORTABLE_DEFAULT_SAM6D_PROVIDER_SCRIPT)
        if PORTABLE_DEFAULT_SAM6D_PROVIDER_SCRIPT.exists()
        else direct_sam6d.DEFAULT_SAM6D_PROVIDER_SCRIPT,
        sam6d_output_root=str(SCRIPT_DIR / "sam6d_jimu_direct_runs"),
        sam6d_fixed_scene_result_file=str(PORTABLE_DEFAULT_SCENE_JSON)
        if PORTABLE_DEFAULT_SCENE_JSON.exists()
        else (str(PORTABLE_LEGACY_DEFAULT_SCENE_JSON) if PORTABLE_LEGACY_DEFAULT_SCENE_JSON.exists() else ""),
        sam6d_reuse_scene_across_cycles=True,
        sam6d_no_pem_warmup_during_sam3=True,
        sam3_full_scene_keep_multi_instances=True,
        sam3_max_masks_per_item=len(JIMU_SCENE_ROLES),
        sam3_confidence_threshold=0.20,
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
        joint_search_validate_final_contact=True,
        joint_search_start_collision_lift_m=0.10,
        short_linear_endpoint_ik_first=False,
        strict_short_linear_waypoint_pos_tol_m=0.008,
        strict_final_contact_waypoint_pos_tol_m=0.006,
        curobo_approach_metric_locked_axis_tol_m=0.006,
        force_replan_post_place_clearance=True,
        # Jimu blocks are thin and densely staged in the tray.  Use a sparse
        # signed tilt set: positive angles lean toward the robot, negative
        # angles lean away from the robot.  This keeps the first layer in one
        # fixed 16-slot batch while letting the second layer try both sides.
        direct_grasp_tilt_toward_robot_deg=[
            -60.0,
            -45.0,
            -32.0,
            -24.0,
            -16.0,
            -8.0,
            -4.0,
            4.0,
            8.0,
            12.0,
            16.0,
            24.0,
            32.0,
            45.0,
            60.0,
        ],
        direct_grasp_tilt_toward_robot_shift_m=[0.0],
        direct_grasp_object_axis_shifts_m=[0.0],
        direct_grasp_z_lifts_m=[0.0],
        direct_grasp_z_lift_penalty_per_cm=3.0,
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
        "--jimu-live-sam6d",
        action="store_true",
        default=False,
        help="Disable the bundled fixed scene and run live SAM3/SAM6D camera capture.",
    )
    parser.add_argument(
        "--no-jimu-portable-maniskill-env",
        dest="jimu_portable_maniskill_env",
        action="store_false",
        default=True,
        help="Do not force-register the bundled ManiSkill PickJiaobang/RM75 environment.",
    )
    parser.add_argument(
        "--jimu-mesh-file",
        type=str,
        default="",
        help="Override the Jimu GLB mesh used for SAM6D/template geometry. Defaults to bundled red_jimu_cube.glb when present.",
    )
    parser.add_argument(
        "--jimu-sim-asset-file",
        type=str,
        default="",
        help="Override the Jimu GLB used by ManiSkill visual/collision. Defaults to a bundled 74x6x74mm red box.",
    )
    parser.add_argument(
        "--jimu-localization-mode",
        choices=["assembly", "per_block"],
        default="assembly",
        help="assembly locates the 5-plate base and 14-slot tray as two anchors; per_block keeps the old every-plate SAM6D flow.",
    )
    parser.add_argument(
        "--jimu-tray-object-name",
        type=str,
        default=JIMU_TRAY_OBJECT_NAME,
        help="Object spec name used by SAM6D for the 14-slot Jimu tray anchor.",
    )
    parser.add_argument(
        "--jimu-base-assembly-object-name",
        type=str,
        default=JIMU_BASE_ASSEMBLY_OBJECT_NAME,
        help="Object spec name used by SAM6D for the five-plate base anchor.",
    )
    parser.add_argument(
        "--jimu-tray-mesh-file",
        type=str,
        default="",
        help="Override the tray OBJ/GLB used by SAM6D for whole-tray localization.",
    )
    parser.add_argument("--jimu-tray-mesh-scale", type=float, default=0.01)
    parser.add_argument(
        "--jimu-base-assembly-mesh-file",
        type=str,
        default="",
        help="Override the five-plate base assembly GLB used by SAM6D.",
    )
    parser.add_argument("--jimu-base-assembly-mesh-scale", type=float, default=1.0)
    parser.add_argument("--jimu-base-assembly-instance-index", type=int, default=-1)
    parser.add_argument("--jimu-tray-instance-index", type=int, default=-1)
    parser.add_argument("--jimu-tray-slot-columns", type=int, default=7)
    parser.add_argument("--jimu-tray-slot-rows", type=int, default=2)
    parser.add_argument("--jimu-tray-slot-x-margin-m", type=float, default=0.00875)
    parser.add_argument(
        "--jimu-tray-slot-x-offset-m",
        type=float,
        default=PORTABLE_DEFAULT_TRAY_SLOT_X_OFFSET_M,
        help="Calibration offset applied to every tray slot center along tray local X. Use this for residual slot/pick X bias without moving the tray anchor itself.",
    )
    parser.add_argument("--jimu-tray-slot-y-margin-m", type=float, default=0.040)
    parser.add_argument("--jimu-tray-slot-insertion-depth-m", type=float, default=0.012)
    parser.add_argument("--jimu-render-tray-visual", dest="jimu_render_tray_visual", action="store_true", default=True)
    parser.add_argument("--no-jimu-render-tray-visual", dest="jimu_render_tray_visual", action="store_false")
    parser.add_argument("--jimu-save-scene-preview", dest="jimu_save_scene_preview", action="store_true", default=True)
    parser.add_argument("--no-jimu-save-scene-preview", dest="jimu_save_scene_preview", action="store_false")
    parser.add_argument(
        "--jimu-render-dry-run-motion",
        dest="jimu_render_dry_run_motion",
        action="store_true",
        default=True,
        help="In human dry-run mode, animate planned arm paths in the simulator instead of teleporting to the final q.",
    )
    parser.add_argument("--no-jimu-render-dry-run-motion", dest="jimu_render_dry_run_motion", action="store_false")
    parser.add_argument(
        "--jimu-render-motion-scale",
        type=float,
        default=0.25,
        help="Playback duration scale for human dry-run rendered motion. Increase toward 1.0 for real-time-like playback.",
    )
    parser.add_argument("--jimu-render-motion-min-s", type=float, default=0.25)
    parser.add_argument("--jimu-render-motion-max-s", type=float, default=1.5)
    parser.add_argument("--jimu-render-motion-fps", type=float, default=18.0)
    parser.add_argument("--jimu-render-motion-max-frames", type=int, default=48)
    parser.add_argument(
        "--jimu-render-tray-slot-visuals",
        dest="jimu_render_tray_slot_visuals",
        action="store_true",
        default=True,
        help="Render visual-only red plates in all non-active tray slots for layout inspection.",
    )
    parser.add_argument("--no-jimu-render-tray-slot-visuals", dest="jimu_render_tray_slot_visuals", action="store_false")
    parser.add_argument(
        "--jimu-tray-slot-role-order",
        type=str,
        nargs="*",
        default=None,
        help="Roles assigned to tray slots in row-major order. Defaults to first-layer roles in row 0 and second-layer roles in row 1.",
    )
    parser.add_argument("--jimu-base-support-obstacles", dest="jimu_base_support_obstacles", action="store_true", default=True)
    parser.add_argument("--no-jimu-base-support-obstacles", dest="jimu_base_support_obstacles", action="store_false")
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
        "--jimu-sam6d-provider-script",
        "--jimu-assembly-sam6d-provider-script",
        dest="jimu_sam6d_provider_script",
        type=str,
        default="",
        help="SAM6D provider script used by the portable same-object subprocess call.",
    )
    parser.add_argument(
        "--jimu-sam6d-max-pem-batch-size",
        type=int,
        default=5,
        help="Maximum same-object Jimu instances sent to one SAM6D PEM batch; larger scenes are split.",
    )
    parser.add_argument(
        "--jimu-manual-sam6d-bboxes",
        action="store_true",
        default=False,
        help="Use one live RGB-D frame, draw one 2D bbox per Jimu assembly anchor, then run SAM6D inside those boxes.",
    )
    parser.add_argument(
        "--jimu-tabletop-anchor-localization",
        action="store_true",
        default=False,
        help="Use manual boxes plus RGB-D tabletop geometry to localize the base/tray anchors without SAM3 or SAM6D PEM.",
    )
    parser.add_argument(
        "--jimu-apriltag-anchor-localization",
        action="store_true",
        default=False,
        help="Use tag25h9 AprilTags to localize the base/tray anchors without SAM3, SAM6D, or manual boxes.",
    )
    parser.add_argument(
        "--jimu-export-scene-json",
        type=str,
        default="",
        help="Write the current Jimu localization result to this fixed-scene JSON path for later replay.",
    )
    parser.add_argument(
        "--jimu-export-scene-only",
        action="store_true",
        default=False,
        help="Only capture/localize and export the Jimu scene JSON; do not create the simulator or plan motions.",
    )
    parser.add_argument("--jimu-apriltag-base-id", type=int, default=1)
    parser.add_argument("--jimu-apriltag-tray-id", type=int, default=0)
    parser.add_argument(
        "--jimu-apriltag-base-size-m",
        type=float,
        default=0.052,
        help="Physical side length of the detected tag black outer square for base tag id=1, excluding extra white paper margin.",
    )
    parser.add_argument(
        "--jimu-apriltag-tray-size-m",
        type=float,
        default=0.06,
        help="Physical side length of the detected tag black outer square for tray tag id=0, excluding extra white paper margin.",
    )
    parser.add_argument(
        "--jimu-apriltag-base-yaw-deg",
        type=float,
        default=0.0,
        help="Rotation of the printed base tag frame inside the base anchor plane. Use this if the tag is pasted rotated.",
    )
    parser.add_argument(
        "--jimu-apriltag-tray-yaw-deg",
        type=float,
        default=90.0,
        help="Rotation of the printed tray tag frame inside the tray anchor plane. Use this if the tag is pasted rotated.",
    )
    parser.add_argument(
        "--jimu-apriltag-tray-center-offset-x-m",
        type=float,
        default=PORTABLE_DEFAULT_TRAY_APRILTAG_CENTER_OFFSET_X_M,
        help="Calibration offset added to the tray tag center along tray local X after reading the CAD recess center.",
    )
    parser.add_argument(
        "--jimu-apriltag-tray-center-offset-y-m",
        type=float,
        default=PORTABLE_DEFAULT_TRAY_APRILTAG_CENTER_OFFSET_Y_M,
        help="Calibration offset added to the tray tag center along tray local Y after reading the CAD recess center.",
    )
    parser.add_argument(
        "--grounding-dino-local-files-only",
        dest="grounding_dino_local_files_only",
        action="store_true",
        default=True,
        help="Load the GroundingDINO model from the local HuggingFace cache/path only.",
    )
    parser.add_argument(
        "--no-grounding-dino-local-files-only",
        dest="grounding_dino_local_files_only",
        action="store_false",
        help="Allow HuggingFace to download the GroundingDINO model if it is not cached.",
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
        "--jimu-first-layer-bottom-clearance-m",
        type=float,
        default=0.006,
        help="Bottom clearance above the base/support top surface for first-layer vertical wall targets; second layer inherits this height through its parent target.",
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
    parser.add_argument(
        "--jimu-tilted-symmetry-score-bias",
        type=float,
        default=0.75,
        help="Small score bonus for 180-degree symmetric Jimu release poses paired with tilted grasps.",
    )
    parser.add_argument(
        "--jimu-return-to-start-pair-score-weight",
        type=float,
        default=1.0,
        help="Penalty weight applied during Jimu fast-chain ranking for q_release distance from the cycle-start joint pose.",
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
        default=False,
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
    parser.add_argument("--jimu-attached-sphere-long-count", type=int, default=5)
    parser.add_argument("--jimu-attached-sphere-wide-count", type=int, default=5)
    parser.add_argument("--jimu-attached-sphere-span-scale", type=float, default=1.0)
    parser.add_argument("--jimu-attached-sphere-dim-scale", type=float, default=1.0)
    parser.add_argument(
        "--jimu-lift-world-z-only",
        dest="jimu_lift_world_z_only",
        action="store_true",
        default=True,
        help="Use world-Z straight lifts for Jimu post-grasp/start-lift clearance instead of TCP-axis lifts.",
    )
    parser.add_argument(
        "--no-jimu-lift-world-z-only",
        dest="jimu_lift_world_z_only",
        action="store_false",
        help="Allow the inherited TCP-axis lift candidates for Jimu.",
    )
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
        "--jimu-partial-open-before-grasp",
        dest="jimu_partial_open_before_grasp",
        action="store_true",
        default=True,
        help="Move to Jimu grasp/pregrasp with the gripper partially closed instead of fully open.",
    )
    parser.add_argument("--no-jimu-partial-open-before-grasp", dest="jimu_partial_open_before_grasp", action="store_false")
    parser.add_argument(
        "--jimu-pregrasp-open-fraction",
        type=float,
        default=0.79,
        help="Pre-grasp opening as a fraction from full-open to closed; 0=open, 1=closed. Default is about 30mm pad gap.",
    )
    parser.add_argument(
        "--jimu-sim-start-joints-deg",
        type=float,
        nargs=7,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help=(
            "Override the Jimu simulation start arm qpos after ManiSkill reset. "
            "This is the pose recorded as cycle_start_q and used by return_to_cycle_start."
        ),
    )
    parser.add_argument(
        "--jimu-disable-mplib-demo-planner",
        dest="jimu_disable_mplib_demo_planner",
        action="store_true",
        default=True,
        help="Do not construct the legacy mplib demo planner; this Jimu entrypoint uses cuRobo for motion planning.",
    )
    parser.add_argument(
        "--no-jimu-disable-mplib-demo-planner",
        dest="jimu_disable_mplib_demo_planner",
        action="store_false",
        help="Restore the old mplib demo planner initialization for debugging.",
    )
    parser.add_argument(
        "--jimu-dry-run-return-linear-fallback",
        dest="jimu_dry_run_return_linear_fallback",
        action="store_true",
        default=True,
        help="In dry-run roof debugging, render a joint-linear return path if all planned return-to-start solvers fail.",
    )
    parser.add_argument(
        "--no-jimu-dry-run-return-linear-fallback",
        dest="jimu_dry_run_return_linear_fallback",
        action="store_false",
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
    parser.add_argument("--jimu-return-to-start-linear-fallback-step-rad", type=float, default=0.035)
    parser.add_argument(
        "--jimu-return-to-start-curobo-retry",
        dest="jimu_return_to_start_curobo_retry",
        action="store_true",
        default=False,
        help="Retry Jimu return_to_cycle_start cuRobo joint planning with a larger seed/timeout budget after the fast pass fails.",
    )
    parser.add_argument(
        "--no-jimu-return-to-start-curobo-retry",
        dest="jimu_return_to_start_curobo_retry",
        action="store_false",
    )
    parser.add_argument("--jimu-return-to-start-curobo-retry-max-attempts", type=int, default=4)
    parser.add_argument("--jimu-return-to-start-curobo-retry-timeout", type=float, default=8.0)
    parser.add_argument("--jimu-return-to-start-curobo-retry-trajopt-seeds", type=int, default=4)
    parser.add_argument("--jimu-return-to-start-curobo-retry-graph-seeds", type=int, default=4)
    parser.add_argument(
        "--jimu-return-to-start-curobo-retry-enable-graph",
        dest="jimu_return_to_start_curobo_retry_enable_graph",
        action="store_true",
        default=False,
        help="Enable cuRobo graph search on Jimu return retry. The fast primary pass still keeps the global setting.",
    )
    parser.add_argument(
        "--no-jimu-return-to-start-curobo-retry-enable-graph",
        dest="jimu_return_to_start_curobo_retry_enable_graph",
        action="store_false",
    )
    parser.add_argument(
        "--jimu-return-to-start-staged-fallback-for-real",
        dest="jimu_return_to_start_staged_fallback_for_real",
        action="store_true",
        default=True,
        help="Allow audited staged joint fallback for real Jimu return_to_cycle_start if cuRobo and legacy fallback fail.",
    )
    parser.add_argument(
        "--no-jimu-return-to-start-staged-fallback-for-real",
        dest="jimu_return_to_start_staged_fallback_for_real",
        action="store_false",
    )
    parser.add_argument(
        "--jimu-near-ik-fallback",
        dest="jimu_near_ik_fallback",
        action="store_true",
        default=True,
        help="Accept cuRobo IK solutions whose reported pose error is already below the Jimu thresholds even if the raw success flag is false.",
    )
    parser.add_argument("--no-jimu-near-ik-fallback", dest="jimu_near_ik_fallback", action="store_false")
    parser.add_argument("--jimu-near-ik-position-threshold", type=float, default=1.0e-4)
    parser.add_argument("--jimu-near-ik-rotation-threshold", type=float, default=1.0e-3)
    parser.add_argument(
        "--jimu-linear-joint-transport-fallback",
        dest="jimu_linear_joint_transport_fallback",
        action="store_true",
        default=True,
        help="For Jimu transport-hover only, fall back to a small-step joint interpolation when MotionGen rejects an already IK-solved hover q.",
    )
    parser.add_argument(
        "--no-jimu-linear-joint-transport-fallback",
        dest="jimu_linear_joint_transport_fallback",
        action="store_false",
    )
    parser.add_argument("--jimu-linear-joint-transport-step-rad", type=float, default=0.035)
    parser.add_argument(
        "--jimu-linear-joint-transport-check-start-state",
        dest="jimu_linear_joint_transport_check_start_state",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-jimu-linear-joint-transport-check-start-state",
        dest="jimu_linear_joint_transport_check_start_state",
        action="store_false",
    )
    parser.add_argument(
        "--jimu-pair-first-pregrasp-motiongen",
        dest="jimu_pair_first_pregrasp_motiongen",
        action="store_true",
        default=False,
        help=(
            "In Jimu pair-first mode, use cuRobo MotionGen from the current/start joint state "
            "to the IK-preselected pregrasp pose instead of accepting a raw joint interpolation."
        ),
    )
    parser.add_argument(
        "--no-jimu-pair-first-pregrasp-motiongen",
        dest="jimu_pair_first_pregrasp_motiongen",
        action="store_false",
        help="Restore the old fast-chain behavior that turns the IK-preselected pregrasp into a joint-linear path.",
    )
    parser.add_argument(
        "--jimu-pair-first-lazy-pregrasp-motiongen",
        dest="jimu_pair_first_lazy_pregrasp_motiongen",
        action="store_true",
        default=True,
        help=(
            "Try pair-first pregrasp MotionGen one ranked candidate at a time, and run the downstream "
            "lift/transport/place chain before planning the next pregrasp candidate."
        ),
    )
    parser.add_argument(
        "--no-jimu-pair-first-lazy-pregrasp-motiongen",
        dest="jimu_pair_first_lazy_pregrasp_motiongen",
        action="store_false",
    )
    parser.add_argument(
        "--jimu-pair-first-pregrasp-motiongen-initial-candidates",
        type=int,
        default=1,
        help=(
            "Number of ranked pair-first grasp candidates to try with normal MotionGen before stopping. "
            "This keeps the fast path from serially planning all top pairs."
        ),
    )
    parser.add_argument(
        "--jimu-pair-first-pregrasp-motiongen-max-candidates",
        type=int,
        default=8,
        help="Maximum ranked pair-first grasp candidates to consider when the initial pregrasp MotionGen pass fails.",
    )
    parser.add_argument(
        "--jimu-pair-first-pregrasp-motiongen-target-winners",
        type=int,
        default=1,
        help="Stop pregrasp MotionGen once this many ranked pair-first candidates have a valid current->pregrasp path.",
    )
    parser.add_argument(
        "--jimu-pair-first-pregrasp-motiongen-retry-max-attempts",
        type=int,
        default=4,
        help="MotionGen max_attempts for the failure-only retry pass from current/start q to pregrasp q.",
    )
    parser.add_argument(
        "--jimu-pair-first-pregrasp-motiongen-retry-trajopt-seeds",
        type=int,
        default=4,
        help="MotionGen trajopt seeds for the failure-only retry pass from current/start q to pregrasp q.",
    )
    parser.add_argument(
        "--jimu-pair-first-pregrasp-motiongen-retry-timeout",
        type=float,
        default=8.0,
        help="MotionGen timeout for the failure-only retry pass from current/start q to pregrasp q.",
    )
    parser.add_argument(
        "--jimu-retry-next-tray-source-on-grasp-failure",
        dest="jimu_retry_next_tray_source_on_grasp_failure",
        action="store_true",
        default=True,
        help="When a Jimu tray plate has no pregrasp/grasp IK before motion starts, remap the same logical target to the next tray slot and retry.",
    )
    parser.add_argument(
        "--no-jimu-retry-next-tray-source-on-grasp-failure",
        dest="jimu_retry_next_tray_source_on_grasp_failure",
        action="store_false",
    )
    parser.add_argument(
        "--jimu-planning-failure-source-retry-max",
        type=int,
        default=-1,
        help=(
            "Maximum same-family tray source swaps after a Jimu planning-only failure. "
            "Set negative to keep retrying unused same-family tray sources until none remain."
        ),
    )
    parser.add_argument(
        "--jimu-skip-physics-gripper-sync",
        dest="jimu_skip_physics_gripper_sync",
        action="store_true",
        default=True,
        help="Use a short lightweight visual sync instead of a long physics settle solely to open/close the gripper.",
    )
    parser.add_argument("--no-jimu-skip-physics-gripper-sync", dest="jimu_skip_physics_gripper_sync", action="store_false")
    parser.add_argument(
        "--jimu-gripper-visual-sync-steps",
        type=int,
        default=2,
        help="Deprecated compatibility option; Jimu gripper dry-run visual sync now writes qpos directly without stepping physics.",
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
    if bool(getattr(args, "jimu_base_support_obstacles", True)):
        default_scene_roles = [*default_scene_roles, *JIMU_BASE_SUPPORT_ROLES]
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
    if bool(getattr(args, "jimu_base_support_obstacles", True)):
        for role in JIMU_BASE_SUPPORT_ROLES:
            if role not in args.tracked_scene_object_names:
                args.tracked_scene_object_names.append(role)
    if _argv_has_option("--repeat-count"):
        args.repeat_count = max(int(getattr(args, "repeat_count", len(default_pick_roles))), len(args.cycle_object_names))
    else:
        args.repeat_count = len(args.cycle_object_names)
    if (
        hasattr(args, "skip_return_to_cycle_start_after_final_place")
        and not _argv_has_option("--skip-return-to-cycle-start-after-final-place")
        and not _argv_has_option("--no-skip-return-to-cycle-start-after-final-place")
    ):
        args.skip_return_to_cycle_start_after_final_place = False
    args.sam3_full_scene_keep_multi_instances = True
    args.sam3_max_masks_per_item = max(int(getattr(args, "sam3_max_masks_per_item", 1) or 1), len(args.jimu_scene_roles))
    if (
        not _argv_has_option("--jimu-sim-start-joints-deg")
        and _argv_has_option("--fixed-goal-joints-deg")
        and getattr(args, "fixed_goal_joints_deg", None) is not None
    ):
        args.jimu_sim_start_joints_deg = list(args.fixed_goal_joints_deg)
        print("[jimu config] using --fixed-goal-joints-deg as Jimu sim start qpos for compatibility")
    if not _argv_has_option("--return-start-clearance-lift-min-m"):
        args.return_start_clearance_lift_min_m = max(
            float(getattr(args, "return_start_clearance_lift_min_m", 0.03) or 0.03),
            0.080,
        )
    if not _argv_has_option("--return-start-clearance-lift-max-m"):
        args.return_start_clearance_lift_max_m = max(
            float(getattr(args, "return_start_clearance_lift_max_m", 0.09) or 0.09),
            0.120,
        )
    if not _argv_has_option("--return-start-clearance-lift-extra-m"):
        args.return_start_clearance_lift_extra_m = max(
            float(getattr(args, "return_start_clearance_lift_extra_m", 0.015) or 0.015),
            0.025,
        )
    args.return_to_start_staged_joint_fallback_for_real = bool(
        getattr(args, "jimu_return_to_start_staged_fallback_for_real", True)
    )
    if (
        bool(getattr(args, "jimu_live_sam6d", False))
        or bool(getattr(args, "jimu_tabletop_anchor_localization", False))
        or bool(getattr(args, "jimu_apriltag_anchor_localization", False))
    ):
        args.sam6d_fixed_scene_result_file = ""
    elif not str(getattr(args, "sam6d_fixed_scene_result_file", "") or "").strip():
        if PORTABLE_DEFAULT_SCENE_JSON.exists():
            args.sam6d_fixed_scene_result_file = str(PORTABLE_DEFAULT_SCENE_JSON)
        elif PORTABLE_LEGACY_DEFAULT_SCENE_JSON.exists():
            args.sam6d_fixed_scene_result_file = str(PORTABLE_LEGACY_DEFAULT_SCENE_JSON)
    fixed_scene_uses_measured_anchor_orientation = _fixed_scene_uses_measured_anchor_orientation(
        getattr(args, "sam6d_fixed_scene_result_file", None)
    )
    if (
        (
            bool(getattr(args, "jimu_tabletop_anchor_localization", False))
            or bool(getattr(args, "jimu_apriltag_anchor_localization", False))
            or fixed_scene_uses_measured_anchor_orientation
        )
        and not _argv_has_option("--jimu-canonical-snap-cardinal")
        and not _argv_has_option("--no-jimu-canonical-snap-cardinal")
    ):
        args.jimu_canonical_snap_cardinal = False
    if bool(getattr(args, "skip_foundationpose", False)):
        print("[jimu-sam6d] ignoring --skip-foundationpose; this entrypoint uses SAM6D camera poses")
        args.skip_foundationpose = False
    if (
        bool(getattr(args, "jimu_portable_maniskill_env", True))
        and PORTABLE_MANISKILL_ROOT.exists()
        and not _argv_has_option("--extra-maniskill-package-root")
    ):
        args.extra_maniskill_package_root = str(PORTABLE_MANISKILL_ROOT)
    if PORTABLE_MANISKILL_RM75_URDF.exists() and not _argv_has_option("--urdf-path"):
        args.urdf_path = str(PORTABLE_MANISKILL_RM75_URDF)
    if PORTABLE_MANISKILL_RM75_SRDF.exists() and not _argv_has_option("--srdf-path"):
        args.srdf_path = str(PORTABLE_MANISKILL_RM75_SRDF)
    if PORTABLE_MANISKILL_RM75_PLANNING_URDF.exists() and not _argv_has_option("--curobo-rm75-urdf"):
        args.curobo_rm75_urdf = PORTABLE_MANISKILL_RM75_PLANNING_URDF
    if not _argv_has_option("--curobo-rm75-robot-cfg"):
        args.curobo_rm75_robot_cfg = PICK_JIAOBANG_DIR / "curobo_rm75_config" / "rm75.yml"
    if PORTABLE_CAMERA_EXTRINSIC.exists() and not _argv_has_option("--camera-extrinsic-opencv-path"):
        args.camera_extrinsic_opencv_path = str(PORTABLE_CAMERA_EXTRINSIC)
    portable_env_loaded = install_portable_maniskill_env(bool(getattr(args, "jimu_portable_maniskill_env", True)))
    install_jimu_runtime_config(args)
    mesh_file = _jimu_mesh_file_override(args)
    sim_asset_file = _jimu_sim_asset_file_override(args)
    tray_mesh_file = _jimu_tray_mesh_file_override(args)
    base_assembly_mesh_file = _jimu_base_assembly_mesh_file_override(args)
    jimu_extents = _load_scaled_jimu_extents(args)
    fixed_scene = str(getattr(args, "sam6d_fixed_scene_result_file", "") or "")
    print(f"[jimu config] mesh_file={mesh_file or 'object_specs default'}")
    print(f"[jimu config] sim_asset_file={sim_asset_file or 'object_specs default'}")
    print(
        "[jimu config] localization="
        f"{str(args.jimu_localization_mode)}, "
        f"base_anchor={str(args.jimu_base_assembly_object_name)}, "
        f"tray_anchor={str(args.jimu_tray_object_name)}"
    )
    print(
        "[jimu config] assembly_assets="
        f"base={base_assembly_mesh_file or 'missing'}, "
        f"tray={tray_mesh_file or 'missing'}, "
        f"sam6d_loaded_tray={str(PORTABLE_DEFAULT_LOADED_TRAY_MESH_FILE) if PORTABLE_DEFAULT_LOADED_TRAY_MESH_FILE.exists() else 'auto-generate-on-live'}, "
        f"tray_scale={_jimu_tray_mesh_scale(args):.5f}, "
        f"slot_layout={int(args.jimu_tray_slot_columns)}x{int(args.jimu_tray_slot_rows)}, "
        f"render_tray={bool(args.jimu_render_tray_visual)}, "
        f"render_slot_plates={bool(args.jimu_render_tray_slot_visuals)}, "
        f"save_preview={bool(args.jimu_save_scene_preview)}"
    )
    print(
        "[jimu config] logical_plate_extents_mm="
        f"{np.round(jimu_extents * 1000.0, 2).tolist()} "
        f"(use_mesh_extents={bool(getattr(args, 'jimu_use_mesh_extents', False))})"
    )
    tray_bounds = _jimu_tray_bounds_scaled(args)
    tray_slot_poses = _jimu_tray_slot_local_poses(args)
    slot_centers = [np.asarray(T[:3, 3], dtype=np.float32) for T in tray_slot_poses.values()]
    slot_x_mm = sorted({round(float(center[0]) * 1000.0, 2) for center in slot_centers})
    slot_y_mm = sorted({round(float(center[1]) * 1000.0, 2) for center in slot_centers})
    print(
        "[jimu config] tray_bounds_mm="
        f"{np.round(tray_bounds * 1000.0, 2).tolist()}, "
        f"slot_x_mm={slot_x_mm}, slot_y_mm={slot_y_mm}"
    )
    print(f"[jimu config] scene_source={'live_sam6d' if not fixed_scene else fixed_scene}")
    if fixed_scene and "jimu_assembly_anchors_default_sam6d.json" in fixed_scene:
        print(
            "[jimu config][warn] using bundled synthetic fixed tray/base scene; "
            "for the physical new tray use --jimu-apriltag-anchor-localization"
        )
    print(
        "[jimu config] maniskill_env="
        f"{'portable' if portable_env_loaded else 'external'}, "
        f"extra_root={str(getattr(args, 'extra_maniskill_package_root', '') or '')}"
    )
    print(
        "[jimu config] rm75_assets="
        f"urdf={str(getattr(args, 'urdf_path', '') or '')}, "
        f"srdf={str(getattr(args, 'srdf_path', '') or '')}, "
        f"curobo_urdf={str(getattr(args, 'curobo_rm75_urdf', '') or '')}"
    )
    print(f"[jimu config] cad_to_sim_local_rpy_deg={list(_jimu_cad_to_sim_rpy_deg(args))}")
    print(f"[jimu config] snap_low_profile_objects_flat_on_table={bool(args.snap_low_profile_objects_flat_on_table)}")
    print(
        "[jimu config] build_layers="
        f"{str(args.jimu_build_layers)}, scene_roles={list(args.jimu_scene_roles)}, "
        f"tracked_roles={list(args.tracked_scene_object_names)}, "
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
        f"legacy_fallback={bool(args.fast_chain_allow_legacy_fallback)}, "
        f"pregrasp_motiongen={bool(args.jimu_pair_first_pregrasp_motiongen)}, "
        f"pregrasp_lazy={bool(args.jimu_pair_first_lazy_pregrasp_motiongen)}, "
        f"pregrasp_initial={int(args.jimu_pair_first_pregrasp_motiongen_initial_candidates)}, "
        f"pregrasp_max={int(args.jimu_pair_first_pregrasp_motiongen_max_candidates)}, "
        f"pregrasp_target_winners={int(args.jimu_pair_first_pregrasp_motiongen_target_winners)}"
    )
    print(
        "[jimu config] transport_q_goal_prefilter="
        f"{bool(args.transport_use_prefilter_q_goal)}, "
        f"trials={int(args.transport_prefilter_q_goal_max_trials)}, "
        f"timeout={float(args.transport_prefilter_q_goal_timeout):.2f}, "
        f"trajopt_seeds={int(args.transport_prefilter_q_goal_num_trajopt_seeds)}"
    )
    print(f"[jimu config] joint_search_validate_final_contact={bool(args.joint_search_validate_final_contact)}")
    print(f"[jimu config] empty_grasp_check_after_lift={bool(args.empty_grasp_check_after_lift)}")
    print(
        "[jimu config] constrained_short_segments="
        f"pose_metric_first={not bool(getattr(args, 'short_linear_endpoint_ik_first', True))}, "
        f"short_tol={float(getattr(args, 'strict_short_linear_waypoint_pos_tol_m', 0.0)):.4f}m, "
        f"final_contact_tol={float(getattr(args, 'strict_final_contact_waypoint_pos_tol_m', 0.0)):.4f}m, "
        f"locked_axis_tol={float(getattr(args, 'curobo_approach_metric_locked_axis_tol_m', 0.0)):.4f}m, "
        f"force_post_place_replan={bool(getattr(args, 'force_replan_post_place_clearance', False))}"
    )
    print(
        "[jimu config] jimu place rules use vertical_place auto-classification, "
        f"vertical_place_hover_height_m={float(args.vertical_place_hover_height_m):.3f}, "
        f"final_contact_clearance_m={float(args.final_contact_clearance_m):.3f}, "
        f"first_layer_bottom_clearance_m={float(args.jimu_first_layer_bottom_clearance_m):.4f}, "
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
        f"parallel_snap_yaw_90={bool(args.jimu_parallel_grasp_place_snap_yaw_90)}, "
        f"tilt_sym_bias={float(args.jimu_tilted_symmetry_score_bias):.2f}, "
        f"return_pair_weight={float(args.jimu_return_to_start_pair_score_weight):.2f}"
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
    print(
        "[jimu config] pregrasp_gripper="
        f"partial_enabled={bool(args.jimu_partial_open_before_grasp)}, "
        f"fraction={float(args.jimu_pregrasp_open_fraction):.2f}, "
        f"real_value={_jimu_pregrasp_partial_open_value(args):.3f}, "
        f"sim_value={_jimu_pregrasp_partial_open_sim_value(args):.3f}, "
        f"skip_physics_sync={bool(args.jimu_skip_physics_gripper_sync)}"
    )
    print(
        "[jimu config] dry_run_motion_render="
        f"{bool(args.jimu_render_dry_run_motion)}, "
        f"scale={float(args.jimu_render_motion_scale):.2f}, "
        f"min_s={float(args.jimu_render_motion_min_s):.2f}, "
        f"max_s={float(args.jimu_render_motion_max_s):.2f}, "
        f"fps={float(args.jimu_render_motion_fps):.1f}, "
        f"roof_return_linear_fallback={bool(args.jimu_dry_run_return_linear_fallback)}"
    )
    print(
        "[jimu config] near_ik_fallback="
        f"{bool(args.jimu_near_ik_fallback)}, "
        f"pos_threshold={float(args.jimu_near_ik_position_threshold):.6g}m, "
        f"rot_threshold={float(args.jimu_near_ik_rotation_threshold):.6g}rad"
    )
    print(
        "[jimu config] linear_joint_transport_fallback="
        f"{bool(args.jimu_linear_joint_transport_fallback)}, "
        f"step={float(args.jimu_linear_joint_transport_step_rad):.3f}rad, "
        f"check_start_state={bool(args.jimu_linear_joint_transport_check_start_state)}"
    )
    print(
        "[jimu config] return_to_start="
        f"prelift_min={float(args.return_start_clearance_lift_min_m):.3f}m, "
        f"prelift_extra={float(args.return_start_clearance_lift_extra_m):.3f}m, "
        f"prelift_max={float(args.return_start_clearance_lift_max_m):.3f}m, "
        f"retry={bool(args.jimu_return_to_start_curobo_retry)}, "
        f"retry_attempts={int(args.jimu_return_to_start_curobo_retry_max_attempts)}, "
        f"retry_trajopt={int(args.jimu_return_to_start_curobo_retry_trajopt_seeds)}, "
        f"retry_graph={bool(args.jimu_return_to_start_curobo_retry_enable_graph)}, "
        f"staged_real={bool(args.return_to_start_staged_joint_fallback_for_real)}"
    )
    print(f"[jimu config] legacy_mplib_demo_planner_disabled={bool(args.jimu_disable_mplib_demo_planner)}")
    direct._configure_curobo_torch_extensions(args)
    _install_jimu_near_ik_fallback(args)
    return args


def _jimu_q7_or_none(q) -> np.ndarray | None:
    helper = getattr(direct, "_q7_or_none", None)
    if callable(helper):
        return helper(q)
    if q is None:
        return None
    try:
        arr = np.asarray(q, dtype=np.float32).reshape(-1)[:7]
    except Exception:
        return None
    if arr.shape[0] < 7 or not np.all(np.isfinite(arr)):
        return None
    return arr.astype(np.float32, copy=True)


def _jimu_pair_first_pregrasp_cache_key(args, start_q) -> tuple:
    q = _jimu_q7_or_none(start_q)
    rounded = tuple(np.round(q if q is not None else np.zeros(7, dtype=np.float32), 5).tolist())
    return (id(args), str(getattr(args, "object_name", "") or ""), rounded)


def _jimu_preselected_grasp_label(item: dict) -> str:
    label = str(item.get("label", "") or "")
    if label:
        return label
    return f"candidate_{id(item)}"


def _jimu_force_pose_metric_final_approach(item: dict | None) -> dict | None:
    if item is None:
        return None
    out = dict(item)
    # Pair-first IK is still useful for ranking and q_pregrasp, but the real
    # contact descent must be planned with cuRobo PoseCostMetric instead of a
    # cached q_release/q_grasp joint interpolation.
    out.pop("deferred_final_approach_q_path", None)
    out["pair_first_ik_only"] = False
    out["jimu_pose_metric_final_approach"] = True
    return out


def make_ik_preselected_grasp_success_jimu(args, start_q, grasp_candidate: dict) -> dict | None:
    original = _ORIGINAL_MAKE_IK_PRESELECTED_GRASP_SUCCESS
    if not bool(getattr(args, "jimu_pair_first_pregrasp_motiongen", True)):
        if callable(original):
            return _jimu_force_pose_metric_final_approach(original(args, start_q, grasp_candidate))
        return None

    demo = getattr(_JIMU_RUNTIME_CONTEXT, "episode_demo", None)
    if demo is None:
        print("[jimu grasp] warning: no active demo context; falling back to pair-first joint interpolation")
        if callable(original):
            return _jimu_force_pose_metric_final_approach(original(args, start_q, grasp_candidate))
        return None

    cache_key = _jimu_pair_first_pregrasp_cache_key(args, start_q)
    cache = getattr(_JIMU_RUNTIME_CONTEXT, "pair_first_pregrasp_motiongen_cache", None)
    if cache is None:
        cache = {}
        _JIMU_RUNTIME_CONTEXT.pair_first_pregrasp_motiongen_cache = cache

    label = _jimu_preselected_grasp_label(grasp_candidate)
    lazy_pregrasp = bool(getattr(args, "jimu_pair_first_lazy_pregrasp_motiongen", False))
    cached_by_label = cache.get(cache_key)
    if cached_by_label is not None:
        cached = cached_by_label.get(label)
        if cached is not None:
            return dict(cached)
        if not lazy_pregrasp:
            return None

    if lazy_pregrasp:
        top_items = [grasp_candidate]
    else:
        top_items = list(grasp_candidate.get("_winner_chain_top_pair_grasps") or [grasp_candidate])
    unique: list[dict] = []
    seen: set[str] = set()
    for item in top_items:
        if not isinstance(item, dict):
            continue
        key = _jimu_preselected_grasp_label(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(dict(item))
    if label not in seen:
        unique.insert(0, dict(grasp_candidate))
    max_candidates = int(getattr(args, "jimu_pair_first_pregrasp_motiongen_max_candidates", len(unique)) or 0)
    if max_candidates > 0 and len(unique) > max_candidates:
        unique = unique[:max_candidates]
    initial_limit = int(getattr(args, "jimu_pair_first_pregrasp_motiongen_initial_candidates", 2) or 0)
    if initial_limit <= 0:
        initial_limit = len(unique)
    initial_limit = min(len(unique), max(1, initial_limit))
    target_winners = int(getattr(args, "jimu_pair_first_pregrasp_motiongen_target_winners", 2) or 0)
    target_winners = min(len(unique), max(1, target_winners))

    planner = direct._get_or_create_curobo_planner_serialized(args)
    print(
        "[jimu grasp] pair-first top grasps: planning current->IK-pregrasp-q with MotionGen "
        f"initial={initial_limit}/{len(unique)} candidate(s), target_winners={target_winners}; "
        "old joint interpolation disabled"
    )
    planned_by_label: dict[str, dict] = {}
    start_q_arr = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    with direct._profile_stage(
        args,
        "jimu_pair_first_pregrasp_motiongen",
        candidate_count=len(unique),
        max_attempts=int(getattr(args, "curobo_max_attempts", 2)),
        num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
        num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
        enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
    ) as prof:
        direct._refresh_curobo_world(
            planner,
            demo,
            args,
            label="jimu_pair_first_pregrasp_motiongen",
            include_active_object=True,
            include_table=False,
        )
        status_counts: dict[str, int] = {}
        attempt_records: list[dict] = []
        attempt_count = 0

        def plan_source(
            source: dict,
            *,
            pass_name: str,
            max_attempts: int,
            timeout: float,
            num_trajopt_seeds: int,
        ) -> bool:
            nonlocal attempt_count
            source_label = _jimu_preselected_grasp_label(source)
            if not callable(original):
                return False
            base_item = original(args, start_q_arr, source)
            if base_item is None:
                status_counts["MISSING_IK_PREGRASP"] = int(status_counts.get("MISSING_IK_PREGRASP", 0)) + 1
                return False
            q_pregrasp = _jimu_q7_or_none(base_item.get("q_pregrasp", base_item.get("deferred_pregrasp_q")))
            if q_pregrasp is None:
                status_counts["MISSING_Q_PREGRASP"] = int(status_counts.get("MISSING_Q_PREGRASP", 0)) + 1
                return False
            q_delta = np.abs(q_pregrasp - start_q_arr)
            attempt_record = {
                "source_label": source_label,
                "pass": str(pass_name),
                "q_delta_norm": float(np.linalg.norm(q_delta)),
                "q_delta_max": float(np.max(q_delta)),
            }
            for state_label, state_q in (("start", start_q_arr), ("goal", q_pregrasp)):
                try:
                    ok, state_status = planner.check_start_state(np.asarray(state_q, dtype=np.float32).reshape(-1)[:7])
                    attempt_record[f"{state_label}_state_valid"] = bool(ok)
                    attempt_record[f"{state_label}_state_status"] = str(state_status)
                except Exception as exc:
                    attempt_record[f"{state_label}_state_valid"] = False
                    attempt_record[f"{state_label}_state_status"] = f"{type(exc).__name__}: {exc}"
            attempt_count += 1
            try:
                result = direct._profile_plan_to_joint_state(
                    planner,
                    start_q_arr,
                    q_pregrasp,
                    enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
                    max_attempts=int(max_attempts),
                    timeout=float(timeout),
                    num_trajopt_seeds=int(num_trajopt_seeds),
                    num_graph_seeds=int(getattr(args, "curobo_num_graph_seeds", 1)),
                )
            except Exception as exc:
                status = type(exc).__name__
                status_counts[status] = int(status_counts.get(status, 0)) + 1
                attempt_record["status"] = status
                attempt_record["exception"] = str(exc)
                attempt_records.append(attempt_record)
                print(f"[jimu grasp] {source_label}: pregrasp MotionGen exception: {exc}")
                return False
            status = "Success" if bool(getattr(result, "success", False)) else str(getattr(result, "status", "FAIL"))
            status_counts[status] = int(status_counts.get(status, 0)) + 1
            attempt_record["status"] = status
            attempt_record["solve_time"] = float(getattr(result, "solve_time", 0.0) or 0.0)
            attempt_record["trajopt_time"] = float(getattr(result, "trajopt_time", 0.0) or 0.0)
            attempt_records.append(attempt_record)
            if not bool(getattr(result, "success", False)) or getattr(result, "joint_path", None) is None:
                print(
                    f"[jimu grasp] {source_label}: pregrasp MotionGen failed pass={pass_name} "
                    f"status={status}, q_delta_norm={attempt_record['q_delta_norm']:.3f}, "
                    f"q_delta_max={attempt_record['q_delta_max']:.3f}, "
                    f"start_state={attempt_record.get('start_state_status')}, "
                    f"goal_state={attempt_record.get('goal_state_status')}"
                )
                if status == "MotionGenStatus.INVALID_START_STATE_SELF_COLLISION":
                    try:
                        direct._print_start_state_self_collision_diagnostics(
                            planner,
                            start_q_arr,
                            label=f"{source_label}_jimu_pair_first_pregrasp_start",
                        )
                    except Exception as exc:
                        print(f"[jimu grasp] {source_label}: self-collision diagnosis failed: {exc}")
                return False
            q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in list(result.joint_path)]
            if not q_path:
                return False
            if not direct._validate_candidate_joint_path_with_demo_planner(
                demo,
                start_q_arr,
                q_path,
                use_attach=False,
                label=f"{source_label}_jimu_pair_first_pregrasp_motiongen",
            ):
                status_counts["DEMO_PATH_VALIDATION_FAIL"] = int(status_counts.get("DEMO_PATH_VALIDATION_FAIL", 0)) + 1
                return False
            metrics, path_score = direct._path_metrics_and_score(start_q_arr, q_path)
            planned = dict(base_item)
            planned["result"] = result
            planned["q_path"] = q_path
            planned["metrics"] = metrics
            planned["path_score"] = float(path_score)
            planned["q_pregrasp"] = np.asarray(q_path[-1], dtype=np.float32).reshape(-1)[:7]
            planned["deferred_pregrasp_q"] = planned["q_pregrasp"]
            planned["jimu_pair_first_pregrasp_motiongen"] = True
            planned = _jimu_force_pose_metric_final_approach(planned)
            planned["jimu_pair_first_pregrasp_motiongen_pass"] = str(pass_name)
            planned_by_label[source_label] = planned
            return True

        base_max_attempts = int(getattr(args, "curobo_max_attempts", 2))
        base_timeout = float(getattr(args, "curobo_timeout", 5.0))
        base_trajopt_seeds = int(getattr(args, "curobo_num_trajopt_seeds", 1))
        for source in unique[:initial_limit]:
            plan_source(
                source,
                pass_name="primary",
                max_attempts=base_max_attempts,
                timeout=base_timeout,
                num_trajopt_seeds=base_trajopt_seeds,
            )
            if len(planned_by_label) >= target_winners:
                break
        if not planned_by_label and unique:
            retry_max_attempts = max(base_max_attempts, int(getattr(args, "jimu_pair_first_pregrasp_motiongen_retry_max_attempts", 4) or 0))
            retry_timeout = max(base_timeout, float(getattr(args, "jimu_pair_first_pregrasp_motiongen_retry_timeout", 8.0) or 0.0))
            retry_trajopt_seeds = max(base_trajopt_seeds, int(getattr(args, "jimu_pair_first_pregrasp_motiongen_retry_trajopt_seeds", 4) or 0))
            prof["retry_pass_enabled"] = True
            prof["retry_max_attempts"] = int(retry_max_attempts)
            prof["retry_num_trajopt_seeds"] = int(retry_trajopt_seeds)
            prof["retry_timeout"] = float(retry_timeout)
            print(
                "[jimu grasp] primary pregrasp MotionGen found no winner; retrying ranked pair-first candidates "
                f"with max_attempts={retry_max_attempts}, trajopt_seeds={retry_trajopt_seeds}, timeout={retry_timeout:.1f}s"
            )
            for source in unique:
                plan_source(
                    source,
                    pass_name="retry",
                    max_attempts=retry_max_attempts,
                    timeout=retry_timeout,
                    num_trajopt_seeds=retry_trajopt_seeds,
                )
                if len(planned_by_label) >= target_winners:
                    break
        prof["winner_count"] = len(planned_by_label)
        prof["success"] = bool(planned_by_label)
        prof["status"] = "Success" if planned_by_label else "NO_WINNERS"
        prof["world_changed"] = bool(getattr(planner, "_last_world_changed", False))
        prof["cache_hit"] = bool(getattr(planner, "_last_world_cache_hit", False))
        if planned_by_label:
            first = next(iter(planned_by_label.values()))
            prof["path_waypoints"] = int((first.get("metrics") or {}).get("waypoint_count", 0) or 0)
            prof["path_score"] = float(first.get("path_score", first.get("score", 0.0)) or 0.0)
        prof["status_counts"] = dict(status_counts)
        prof["motiongen_attempt_count"] = int(attempt_count)
        prof["attempt_records"] = attempt_records[:16]
        prof["initial_candidate_limit"] = int(initial_limit)
        prof["target_winners"] = int(target_winners)

    source_by_label = {_jimu_preselected_grasp_label(item): item for item in unique}
    missing = [key for key in source_by_label if key not in planned_by_label]
    if missing:
        print(f"[jimu grasp] pregrasp MotionGen rejected {len(missing)}/{len(source_by_label)} pair-first candidate(s)")
    merged_by_label = dict(cached_by_label or {})
    merged_by_label.update(planned_by_label)
    cache[cache_key] = merged_by_label
    cached = merged_by_label.get(label)
    return dict(cached) if cached is not None else None


def main():
    global _ORIGINAL_TARGETED_BUILD_PLACE_PLAN_VARIANTS
    global _ORIGINAL_FAST_CHAIN_RANK_PAIRED_RELATION_CANDIDATES
    global _ORIGINAL_FAST_CHAIN_RELATION_MATCH_KEY
    global _ORIGINAL_EVALUATE_CUROBO_POSE_CANDIDATES_MULTI_START
    global _ORIGINAL_COPY_LAST_CANDIDATE_COUNTS_TO_PROFILE
    global _ORIGINAL_PROFILE_PLAN_TO_JOINT_STATE
    global _ORIGINAL_PLAN_RETURN_TO_START_JOINT_CUROBO
    global _ORIGINAL_PROFILE_PLAN_GOALSET_TO_POSES
    global _ORIGINAL_PROFILE_PLAN_BATCH_START_GOAL_PAIRS
    global _ORIGINAL_ATTACH_TRANSPORT_PAYLOAD_TO_CUROBO
    global _ORIGINAL_PLAN_SHORT_TCP_UP_AXIS_LIFT_IK
    global _ORIGINAL_BUILD_HOVER_POSE
    global _ORIGINAL_PROFILE_STAGE
    global _ORIGINAL_EXECUTE_POSE_PATH_STAGE
    global _ORIGINAL_EXECUTE_JOINT_PATH_STAGE
    global _ORIGINAL_PLAN_JOINT_PATH
    global _ORIGINAL_REALMAN_SET_GRIPPER
    global _ORIGINAL_SYNC_DEMO_GRIPPER_STATE
    global _ORIGINAL_SETTLE_RELEASED_ACTIVE_OBJECT
    global _ORIGINAL_SELECT_RANDOM_CYCLE_TARGET
    global _ORIGINAL_MAKE_IK_PRESELECTED_GRASP_SUCCESS
    global _ORIGINAL_BUILD_DIRECT_GRASP_CANDIDATES
    global _ORIGINAL_CREATE_DEMO
    global _ORIGINAL_CUROBO_SOLVE_IK
    global _ORIGINAL_CUROBO_SOLVE_BATCH_START_GOAL_IK
    global _ORIGINAL_CUROBO_SOLVE_BATCH_START_GOAL_IK_CUDA_GRAPH
    install_jimu_runtime_config()
    original_capture = direct.targeted.base.capture_or_reuse_foundationpose_scene
    original_create_demo = direct.targeted.base.create_demo
    original_relocalize = direct._relocalize_active_target_after_empty_grasp
    original_run_episode = direct.run_targeted_place_episode_curobo_direct
    original_single_scene_activate_object = getattr(direct, "_single_scene_activate_object", None)
    original_single_scene_restore_after_failed_attempt = getattr(direct, "_single_scene_restore_after_failed_attempt", None)
    original_provider_command = direct_sam6d._provider_command
    original_provider_needs_live_stdio = direct_sam6d._provider_needs_live_stdio
    original_parse_args = direct.parse_args
    original_build_place_variants = direct.targeted.build_targeted_place_plan_variants
    original_rank_paired = direct._fast_chain_rank_paired_relation_candidates
    original_relation_match_key = direct._fast_chain_relation_match_key
    original_evaluate_multi_start = direct._evaluate_curobo_pose_candidates_multi_start
    original_copy_candidate_counts = direct._copy_last_candidate_counts_to_profile
    original_profile_plan_joint = direct._profile_plan_to_joint_state
    original_plan_return_to_start_joint = direct._plan_return_to_start_joint_curobo
    original_profile_plan_goalset = direct._profile_plan_goalset_to_poses
    original_profile_plan_batch_pairs = direct._profile_plan_batch_start_goal_pairs
    original_attach_transport_payload = direct._attach_transport_payload_to_curobo
    original_plan_short_tcp_up_axis_lift_ik = direct._plan_short_tcp_up_axis_lift_ik
    original_build_hover_pose = direct.build_hover_pose
    original_profile_stage = direct._profile_stage
    original_execute_pose_path_stage = direct.targeted.base.execute_pose_path_stage
    original_execute_joint_path_stage = direct.targeted.base.execute_joint_path_stage
    original_plan_joint_path = direct.targeted.base.plan_joint_path
    original_realman_set_gripper = direct.targeted.base.RealmanJointExecutor.set_gripper
    original_sync_demo_gripper_state = direct.targeted.base.sync_demo_gripper_state
    original_settle_released_active_object = direct.targeted.base.settle_released_active_object_for_scene_cache
    original_tilt_pose_toward_robot = direct.targeted.base.tilt_pose_toward_robot
    original_select_random_cycle_target = direct.targeted._select_random_cycle_target
    original_make_ik_preselected_grasp_success = direct._make_ik_preselected_grasp_success
    original_build_direct_grasp_candidates = direct._build_direct_grasp_candidates
    original_install_dry_run_motion_window_wrappers = getattr(direct, "_install_dry_run_motion_window_wrappers", None)
    _ORIGINAL_TARGETED_BUILD_PLACE_PLAN_VARIANTS = original_build_place_variants
    _ORIGINAL_FAST_CHAIN_RANK_PAIRED_RELATION_CANDIDATES = original_rank_paired
    _ORIGINAL_FAST_CHAIN_RELATION_MATCH_KEY = original_relation_match_key
    _ORIGINAL_EVALUATE_CUROBO_POSE_CANDIDATES_MULTI_START = original_evaluate_multi_start
    _ORIGINAL_COPY_LAST_CANDIDATE_COUNTS_TO_PROFILE = original_copy_candidate_counts
    _ORIGINAL_PROFILE_PLAN_TO_JOINT_STATE = original_profile_plan_joint
    _ORIGINAL_PLAN_RETURN_TO_START_JOINT_CUROBO = original_plan_return_to_start_joint
    _ORIGINAL_PROFILE_PLAN_GOALSET_TO_POSES = original_profile_plan_goalset
    _ORIGINAL_PROFILE_PLAN_BATCH_START_GOAL_PAIRS = original_profile_plan_batch_pairs
    _ORIGINAL_ATTACH_TRANSPORT_PAYLOAD_TO_CUROBO = original_attach_transport_payload
    _ORIGINAL_PLAN_SHORT_TCP_UP_AXIS_LIFT_IK = original_plan_short_tcp_up_axis_lift_ik
    _ORIGINAL_BUILD_HOVER_POSE = original_build_hover_pose
    _ORIGINAL_PROFILE_STAGE = original_profile_stage
    _ORIGINAL_EXECUTE_POSE_PATH_STAGE = original_execute_pose_path_stage
    _ORIGINAL_EXECUTE_JOINT_PATH_STAGE = original_execute_joint_path_stage
    _ORIGINAL_PLAN_JOINT_PATH = original_plan_joint_path
    _ORIGINAL_REALMAN_SET_GRIPPER = original_realman_set_gripper
    _ORIGINAL_SYNC_DEMO_GRIPPER_STATE = original_sync_demo_gripper_state
    _ORIGINAL_SETTLE_RELEASED_ACTIVE_OBJECT = original_settle_released_active_object
    _ORIGINAL_SELECT_RANDOM_CYCLE_TARGET = original_select_random_cycle_target
    _ORIGINAL_MAKE_IK_PRESELECTED_GRASP_SUCCESS = original_make_ik_preselected_grasp_success
    _ORIGINAL_BUILD_DIRECT_GRASP_CANDIDATES = original_build_direct_grasp_candidates
    _ORIGINAL_CREATE_DEMO = original_create_demo

    def tilt_pose_toward_robot_jimu(demo, pose, angle_deg: float, *, direction: str = "toward_robot"):
        angle = float(angle_deg)
        direction_text = str(direction or "toward_robot").strip().lower()
        if angle < 0.0 and direction_text in {"toward", "toward_robot", "towards_robot"}:
            return original_tilt_pose_toward_robot(demo, pose, abs(angle), direction="away_robot")
        return original_tilt_pose_toward_robot(demo, pose, abs(angle), direction=direction)

    def _single_scene_activate_object_jimu(
        demo,
        bridge_mod,
        base_args,
        cycle_args,
        selected_name: str,
        obstacle_names,
        scene_capture_cache,
    ) -> bool:
        if not callable(original_single_scene_activate_object):
            return False
        ok = bool(
            original_single_scene_activate_object(
                demo,
                bridge_mod,
                base_args,
                cycle_args,
                selected_name,
                obstacle_names,
                scene_capture_cache,
            )
        )
        if ok:
            try:
                _sync_jimu_tray_slot_visuals_for_cycle(
                    demo.env,
                    demo,
                    cycle_args,
                    scene_capture_cache,
                    active_name=selected_name,
                )
            except Exception as exc:
                print(f"[jimu visual] warning: failed to sync tray slot visuals for {selected_name}: {exc}")
        return ok

    def _single_scene_restore_after_failed_attempt_jimu(
        demo,
        args,
        scene_capture_cache,
        selected_name: str,
        cycle_start_q,
        *restore_args,
        **restore_kwargs,
    ) -> None:
        prev_restore = getattr(_JIMU_RUNTIME_CONTEXT, "failed_attempt_restore", False)
        _JIMU_RUNTIME_CONTEXT.failed_attempt_restore = True
        if callable(original_single_scene_restore_after_failed_attempt):
            try:
                original_single_scene_restore_after_failed_attempt(
                    demo,
                    args,
                    scene_capture_cache,
                    selected_name,
                    cycle_start_q,
                    *restore_args,
                    **restore_kwargs,
                )
            finally:
                _JIMU_RUNTIME_CONTEXT.failed_attempt_restore = prev_restore
        else:
            _JIMU_RUNTIME_CONTEXT.failed_attempt_restore = prev_restore
        if _jimu_should_retry_next_source_after_failure(args):
            _jimu_prepare_next_tray_source_retry(
                demo,
                args,
                scene_capture_cache,
                selected_name,
            )

    def _install_dry_run_motion_window_wrappers_jimu() -> None:
        install_cuda_sync = getattr(direct, "_install_cuda_render_sync_wrappers", None)
        if callable(install_cuda_sync):
            install_cuda_sync()
        direct.targeted.base._direct_pre_place_dry_run_motion_window_wrapped = True
        print(
            "[jimu gripper] using Jimu dry-run motion executor; "
            "skipped direct dry-run wrapper so grasp partial-open state is preserved"
        )

    def _provider_command_jimu(args, object_names: list[str]) -> list[str]:
        cmd = list(original_provider_command(args, object_names))

        def append_option(flag: str, value) -> None:
            if flag in cmd:
                return
            cmd.extend([flag, str(value)])

        def set_option(flag: str, value) -> None:
            value_text = str(value)
            if flag not in cmd:
                cmd.extend([flag, value_text])
                return
            idx = cmd.index(flag)
            if idx + 1 < len(cmd):
                cmd[idx + 1] = value_text
            else:
                cmd.append(value_text)

        append_option(
            "--grounding-dino-model-id",
            str(getattr(args, "grounding_dino_model_id", "IDEA-Research/grounding-dino-base")),
        )
        append_option(
            "--grounding-dino-box-threshold",
            float(getattr(args, "grounding_dino_box_threshold", 0.25)),
        )
        append_option(
            "--grounding-dino-text-threshold",
            float(getattr(args, "grounding_dino_text_threshold", 0.20)),
        )
        if bool(getattr(args, "grounding_dino_local_files_only", True)):
            if "--grounding-dino-local-files-only" not in cmd and "--no-grounding-dino-local-files-only" not in cmd:
                cmd.append("--grounding-dino-local-files-only")
        elif "--no-grounding-dino-local-files-only" not in cmd:
            cmd.append("--no-grounding-dino-local-files-only")
        if bool(getattr(args, "jimu_apriltag_anchor_localization", False)):
            if "--jimu-apriltag-anchors" not in cmd:
                cmd.append("--jimu-apriltag-anchors")
            set_option("--jimu-apriltag-base-id", int(getattr(args, "jimu_apriltag_base_id", 1)))
            set_option("--jimu-apriltag-tray-id", int(getattr(args, "jimu_apriltag_tray_id", 0)))
            set_option("--jimu-apriltag-base-size-m", float(getattr(args, "jimu_apriltag_base_size_m", 0.052)))
            set_option("--jimu-apriltag-tray-size-m", float(getattr(args, "jimu_apriltag_tray_size_m", 0.06)))
            set_option("--jimu-apriltag-base-yaw-deg", float(getattr(args, "jimu_apriltag_base_yaw_deg", 0.0)))
            set_option("--jimu-apriltag-tray-yaw-deg", float(getattr(args, "jimu_apriltag_tray_yaw_deg", 90.0)))
            set_option(
                "--jimu-apriltag-tray-center-offset-x-m",
                float(getattr(args, "jimu_apriltag_tray_center_offset_x_m", PORTABLE_DEFAULT_TRAY_APRILTAG_CENTER_OFFSET_X_M)),
            )
            set_option(
                "--jimu-apriltag-tray-center-offset-y-m",
                float(getattr(args, "jimu_apriltag_tray_center_offset_y_m", PORTABLE_DEFAULT_TRAY_APRILTAG_CENTER_OFFSET_Y_M)),
            )
        if bool(getattr(args, "jimu_tabletop_anchor_localization", False)) and "--jimu-tabletop-anchors" not in cmd:
            cmd.append("--jimu-tabletop-anchors")
        if bool(getattr(args, "jimu_manual_sam6d_bboxes", False)) and "--jimu-manual-bboxes" not in cmd:
            cmd.append("--jimu-manual-bboxes")
        return cmd

    def _provider_needs_live_stdio_jimu(args, object_names: list[str]) -> bool:
        if bool(getattr(args, "jimu_apriltag_anchor_localization", False)):
            return True
        if bool(getattr(args, "jimu_tabletop_anchor_localization", False)):
            return True
        if bool(getattr(args, "jimu_manual_sam6d_bboxes", False)):
            return True
        return bool(original_provider_needs_live_stdio(args, object_names))

    wrapped_run_episode = direct_sam6d._wrap_run_targeted_place_episode_for_sam6d_prefetch_fail_fast(
        original_run_episode
    )

    def run_targeted_place_episode_curobo_direct_jimu(*call_args, **call_kwargs):
        demo = call_kwargs.get("demo")
        episode_args = call_kwargs.get("args")
        if demo is None and len(call_args) >= 1:
            demo = call_args[0]
        if episode_args is None and len(call_args) >= 4:
            episode_args = call_args[3]

        prev_demo = getattr(_JIMU_RUNTIME_CONTEXT, "episode_demo", None)
        prev_args = getattr(_JIMU_RUNTIME_CONTEXT, "episode_args", None)
        prev_cache = getattr(_JIMU_RUNTIME_CONTEXT, "pair_first_pregrasp_motiongen_cache", None)
        _JIMU_RUNTIME_CONTEXT.episode_demo = demo
        _JIMU_RUNTIME_CONTEXT.episode_args = episode_args
        _JIMU_RUNTIME_CONTEXT.pair_first_pregrasp_motiongen_cache = {}
        try:
            return wrapped_run_episode(*call_args, **call_kwargs)
        finally:
            if prev_demo is None:
                try:
                    delattr(_JIMU_RUNTIME_CONTEXT, "episode_demo")
                except AttributeError:
                    pass
            else:
                _JIMU_RUNTIME_CONTEXT.episode_demo = prev_demo
            if prev_args is None:
                try:
                    delattr(_JIMU_RUNTIME_CONTEXT, "episode_args")
                except AttributeError:
                    pass
            else:
                _JIMU_RUNTIME_CONTEXT.episode_args = prev_args
            if prev_cache is None:
                try:
                    delattr(_JIMU_RUNTIME_CONTEXT, "pair_first_pregrasp_motiongen_cache")
                except AttributeError:
                    pass
            else:
                _JIMU_RUNTIME_CONTEXT.pair_first_pregrasp_motiongen_cache = prev_cache

    try:
        direct.targeted.base.capture_or_reuse_foundationpose_scene = capture_or_reuse_jimu_sam6d_scene
        direct.targeted.base.create_demo = create_demo_jimu_no_mplib
        direct._relocalize_active_target_after_empty_grasp = relocalize_active_target_after_empty_grasp_jimu
        direct.run_targeted_place_episode_curobo_direct = run_targeted_place_episode_curobo_direct_jimu
        direct.parse_args = parse_args
        direct.targeted.build_targeted_place_plan_variants = build_targeted_place_plan_variants_jimu
        direct._fast_chain_rank_paired_relation_candidates = fast_chain_rank_paired_relation_candidates_jimu
        direct._fast_chain_relation_match_key = fast_chain_relation_match_key_jimu
        direct._make_ik_preselected_grasp_success = make_ik_preselected_grasp_success_jimu
        direct._build_direct_grasp_candidates = build_direct_grasp_candidates_jimu
        direct._evaluate_curobo_pose_candidates_multi_start = evaluate_curobo_pose_candidates_multi_start_jimu
        direct._copy_last_candidate_counts_to_profile = copy_last_candidate_counts_to_profile_jimu
        direct._profile_plan_to_joint_state = profile_plan_to_joint_state_jimu
        direct._plan_return_to_start_joint_curobo = plan_return_to_start_joint_curobo_jimu
        direct._profile_plan_goalset_to_poses = profile_plan_goalset_to_poses_jimu
        direct._profile_plan_batch_start_goal_pairs = profile_plan_batch_start_goal_pairs_jimu
        direct._attach_transport_payload_to_curobo = attach_transport_payload_to_curobo_jimu
        direct._plan_short_tcp_up_axis_lift_ik = plan_short_tcp_up_axis_lift_ik_jimu
        direct.build_hover_pose = build_hover_pose_jimu
        direct._profile_stage = profile_stage_jimu
        if callable(original_install_dry_run_motion_window_wrappers):
            direct._install_dry_run_motion_window_wrappers = _install_dry_run_motion_window_wrappers_jimu
        direct.targeted.base.execute_pose_path_stage = execute_pose_path_stage_jimu
        direct.targeted.base.execute_joint_path_stage = execute_joint_path_stage_jimu
        direct.targeted.base.plan_joint_path = plan_joint_path_jimu_no_mplib
        direct.targeted.base.RealmanJointExecutor.set_gripper = realman_set_gripper_jimu
        direct.targeted.base.sync_demo_gripper_state = sync_demo_gripper_state_jimu
        direct.targeted.base.settle_released_active_object_for_scene_cache = (
            settle_released_active_object_for_scene_cache_jimu
        )
        direct.targeted.base.tilt_pose_toward_robot = tilt_pose_toward_robot_jimu
        direct.targeted._select_random_cycle_target = select_random_cycle_target_jimu_layered
        direct_sam6d._provider_command = _provider_command_jimu
        direct_sam6d._provider_needs_live_stdio = _provider_needs_live_stdio_jimu
        if callable(original_single_scene_activate_object):
            direct._single_scene_activate_object = _single_scene_activate_object_jimu
        if callable(original_single_scene_restore_after_failed_attempt):
            direct._single_scene_restore_after_failed_attempt = _single_scene_restore_after_failed_attempt_jimu
        if _argv_has_option("--jimu-export-scene-only"):
            args = parse_args()
            _run_jimu_scene_export_only(args)
        else:
            direct.main()
    finally:
        direct.targeted.base.capture_or_reuse_foundationpose_scene = original_capture
        direct.targeted.base.create_demo = original_create_demo
        direct._relocalize_active_target_after_empty_grasp = original_relocalize
        direct.run_targeted_place_episode_curobo_direct = original_run_episode
        direct.parse_args = original_parse_args
        direct.targeted.build_targeted_place_plan_variants = original_build_place_variants
        direct._fast_chain_rank_paired_relation_candidates = original_rank_paired
        direct._fast_chain_relation_match_key = original_relation_match_key
        direct._make_ik_preselected_grasp_success = original_make_ik_preselected_grasp_success
        direct._build_direct_grasp_candidates = original_build_direct_grasp_candidates
        direct._evaluate_curobo_pose_candidates_multi_start = original_evaluate_multi_start
        direct._copy_last_candidate_counts_to_profile = original_copy_candidate_counts
        direct._profile_plan_to_joint_state = original_profile_plan_joint
        direct._plan_return_to_start_joint_curobo = original_plan_return_to_start_joint
        direct._profile_plan_goalset_to_poses = original_profile_plan_goalset
        direct._profile_plan_batch_start_goal_pairs = original_profile_plan_batch_pairs
        direct._attach_transport_payload_to_curobo = original_attach_transport_payload
        direct._plan_short_tcp_up_axis_lift_ik = original_plan_short_tcp_up_axis_lift_ik
        direct.build_hover_pose = original_build_hover_pose
        direct._profile_stage = original_profile_stage
        if callable(original_install_dry_run_motion_window_wrappers):
            direct._install_dry_run_motion_window_wrappers = original_install_dry_run_motion_window_wrappers
        direct.targeted.base.execute_pose_path_stage = original_execute_pose_path_stage
        direct.targeted.base.execute_joint_path_stage = original_execute_joint_path_stage
        direct.targeted.base.plan_joint_path = original_plan_joint_path
        direct.targeted.base.RealmanJointExecutor.set_gripper = original_realman_set_gripper
        direct.targeted.base.sync_demo_gripper_state = original_sync_demo_gripper_state
        direct.targeted.base.settle_released_active_object_for_scene_cache = original_settle_released_active_object
        direct.targeted.base.tilt_pose_toward_robot = original_tilt_pose_toward_robot
        direct.targeted._select_random_cycle_target = original_select_random_cycle_target
        direct_sam6d._provider_command = original_provider_command
        direct_sam6d._provider_needs_live_stdio = original_provider_needs_live_stdio
        if callable(original_single_scene_activate_object):
            direct._single_scene_activate_object = original_single_scene_activate_object
        if callable(original_single_scene_restore_after_failed_attempt):
            direct._single_scene_restore_after_failed_attempt = original_single_scene_restore_after_failed_attempt
        _restore_jimu_near_ik_fallback()
        _restore_jimu_no_mplib_collision_detection()
        _ORIGINAL_TARGETED_BUILD_PLACE_PLAN_VARIANTS = None
        _ORIGINAL_FAST_CHAIN_RANK_PAIRED_RELATION_CANDIDATES = None
        _ORIGINAL_FAST_CHAIN_RELATION_MATCH_KEY = None
        _ORIGINAL_EVALUATE_CUROBO_POSE_CANDIDATES_MULTI_START = None
        _ORIGINAL_COPY_LAST_CANDIDATE_COUNTS_TO_PROFILE = None
        _ORIGINAL_PROFILE_PLAN_TO_JOINT_STATE = None
        _ORIGINAL_PLAN_RETURN_TO_START_JOINT_CUROBO = None
        _ORIGINAL_PROFILE_PLAN_GOALSET_TO_POSES = None
        _ORIGINAL_PROFILE_PLAN_BATCH_START_GOAL_PAIRS = None
        _ORIGINAL_ATTACH_TRANSPORT_PAYLOAD_TO_CUROBO = None
        _ORIGINAL_PROFILE_STAGE = None
        _ORIGINAL_EXECUTE_POSE_PATH_STAGE = None
        _ORIGINAL_EXECUTE_JOINT_PATH_STAGE = None
        _ORIGINAL_PLAN_JOINT_PATH = None
        _ORIGINAL_REALMAN_SET_GRIPPER = None
        _ORIGINAL_SYNC_DEMO_GRIPPER_STATE = None
        _ORIGINAL_SETTLE_RELEASED_ACTIVE_OBJECT = None
        _ORIGINAL_SELECT_RANDOM_CYCLE_TARGET = None
        _ORIGINAL_MAKE_IK_PRESELECTED_GRASP_SUCCESS = None
        _ORIGINAL_BUILD_DIRECT_GRASP_CANDIDATES = None
        _ORIGINAL_CREATE_DEMO = None
        _ORIGINAL_CUROBO_SOLVE_IK = None
        _ORIGINAL_CUROBO_SOLVE_BATCH_START_GOAL_IK = None
        _ORIGINAL_CUROBO_SOLVE_BATCH_START_GOAL_IK_CUDA_GRAPH = None


if __name__ == "__main__":
    main()
