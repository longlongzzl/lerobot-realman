from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import sapien
import torch
from transforms3d.quaternions import mat2quat, quat2mat

import plan_single_wall_path_20s as single_wall_module
import record_realman_edge_grasp_open_cube as realman_module
from curobo_rm75_planner import DEFAULT_CUROBO_ROOT, RM75CuRoboPlanner, RM75CuRoboPlannerConfig
from magnetic_snap import LockedPanelPose, MagneticConnection
from plan_single_wall_path_20s import (
    GraspCandidate,
    _add_hold_segment,
    _add_joint_segment,
    _add_ramp_segment,
    _add_settle_segment,
    _candidate_tcp_for_edge_grasp,
    _current_q,
    _json_ready,
    _joint_distance,
    _make_grasp_candidates,
    _plan_motion_to_pose,
    _pregrasp_pose_for_candidate,
    _pose_to_pose_error,
    _release_collision_exclude_roles,
    _solve_ik,
    _tcp_axis_report,
    _world_obstacles_for_stage,
)
from record_realman_edge_grasp_open_cube import (
    CLOSED_GRIPPER,
    OPEN_GRIPPER,
    PLATE_SIZE,
    RM75_HOME,
    _active_connection_count_for_role,
    _actor_pose,
    _attach_payload_for_planning,
    _grasp_report,
    _initialize_staged_open_cube,
    _make_env,
    _pose_error,
    _offset_world,
    _set_robot_qpos,
    _pose_to_report,
    _tcp_pose,
    _step_action,
    _tilt_actor_pose_about_local_axis,
    _wall_release_actor_candidates,
    _world_to_robot_base,
)
from record_stepwise_house_assembly_sim import _append_frame


FIRST_LAYER_WALL_ROLES = {"right_wall", "back_wall", "left_wall", "front_wall"}
DEFAULT_LEROBOT_ROOT = "/home/zhangzhao/Desktop/lerobot"
DEFAULT_LEROBOT_SIM2REAL_ROOT = "/home/zhangzhao/Desktop/lerobot/lerobot-sim2real"


def _make_logger(out_dir: Path):
    started = time.perf_counter()
    log_path = out_dir / "phase_log.txt"

    def log(message: str) -> None:
        line = f"[multi_wall +{time.perf_counter() - started:7.3f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return log


def _profile_increment(profile: dict[str, Any] | None, key: str, elapsed: float) -> None:
    if profile is None:
        return
    profile[key] = float(profile.get(key, 0.0)) + float(elapsed)
    calls_key = f"{key}_calls"
    profile[calls_key] = int(profile.get(calls_key, 0)) + 1


def _profile_call(profile: dict[str, Any] | None, key: str, func: Any, *args: Any, **kwargs: Any) -> Any:
    started = time.perf_counter()
    try:
        return func(*args, **kwargs)
    finally:
        _profile_increment(profile, key, time.perf_counter() - started)


def _load_realman_base_module() -> Any:
    pick_dir = Path(__file__).resolve().parents[1] / "pick_jiaobang"
    if str(pick_dir) not in sys.path:
        sys.path.insert(0, str(pick_dir))
    import rm75_jiaobang_pick_real_with_foundationpose as real_base

    return real_base


def _sim_gripper_to_real(gripper: float, args: argparse.Namespace) -> float:
    alpha = (float(gripper) - float(OPEN_GRIPPER)) / max(float(CLOSED_GRIPPER - OPEN_GRIPPER), 1e-6)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    open_value = float(getattr(args, "real_gripper_open", 0.0))
    close_value = float(getattr(args, "real_gripper_close", 0.91))
    return float((1.0 - alpha) * open_value + alpha * close_value)


def _confirm_real_execution_start(args: argparse.Namespace, log: Any) -> None:
    if bool(getattr(args, "auto_execute", False)):
        return
    try:
        print(
            "[real] --execute-real will move the RM75 following this beta trajectory. "
            "Press Enter to connect/execute, or type q then Enter to abort: "
            "",
            flush=True,
        )
        answer = input().strip().lower()
    except EOFError:
        answer = "q"
    if answer in {"q", "quit", "n", "no"}:
        raise RuntimeError("user aborted before real execution")
    log("real execution confirmed by user")


def _start_real_executor(args: argparse.Namespace, log: Any) -> Any | None:
    if not bool(getattr(args, "execute_real", False)):
        return None
    _confirm_real_execution_start(args, log)
    real_base = _load_realman_base_module()
    real_exec = real_base.RealmanJointExecutor(args)
    real_exec.set_gripper(float(getattr(args, "real_gripper_open", 0.0)))
    if bool(getattr(args, "reset_real_before_start", True)):
        log("resetting real RM75 to hardware home before beta trajectory")
        real_exec.reset_robot(gripper_pos=float(getattr(args, "real_gripper_open", 0.0)))
    q_real = np.asarray(real_exec.get_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    q_delta = (RM75_HOME.astype(np.float32) - q_real + np.pi) % (2.0 * np.pi) - np.pi
    max_delta = float(np.max(np.abs(q_delta)))
    max_allowed = float(getattr(args, "real_start_max_delta", 0.12))
    log(f"real RM75 start delta to RM75_HOME: {max_delta:.4f} rad")
    if max_allowed > 0.0 and max_delta > max_allowed:
        real_exec.close()
        raise RuntimeError(
            f"real robot start is too far from RM75_HOME: max_delta={max_delta:.4f} rad > {max_allowed:.4f}. "
            "Use --reset-real-before-start or move the robot to home first."
        )
    return real_exec


def _runtime_collision_contact_force(base_env: Any, a: Any, b: Any) -> float:
    try:
        force = base_env.scene.get_pairwise_contact_forces(a, b)
        return float(torch.linalg.norm(force, dim=1).detach().cpu().numpy().reshape(-1)[0])
    except Exception:
        return 0.0


def _runtime_collision_actor_name(actor: Any) -> str:
    for attr in ("name", "_name"):
        try:
            value = getattr(actor, attr, None)
            if value:
                return str(value)
        except Exception:
            pass
    return type(actor).__name__


def _runtime_collision_begin_role(
    base_env: Any,
    *,
    args: argparse.Namespace,
    locked: dict[str, Any],
    role: str,
    completed_roles: list[str],
) -> None:
    enabled = bool(getattr(args, "runtime_collision_monitor", False))
    if not enabled:
        base_env._runtime_collision_monitor = {"enabled": False}
        return
    include_floor = bool(getattr(args, "runtime_collision_monitor_include_floor", False))
    target_roles = [
        item
        for item in list(dict.fromkeys(completed_roles))
        if item in locked and item != role and (include_floor or item != "floor")
    ]
    baseline_poses = {
        target_role: _actor_pose(locked[target_role].actor)
        for target_role in target_roles
        if target_role in locked
    }
    base_env._runtime_collision_monitor = {
        "enabled": True,
        "role": str(role),
        "stage": str(role),
        "locked": locked,
        "target_roles": target_roles,
        "baseline_poses": baseline_poses,
        "sample_period": max(int(getattr(args, "runtime_collision_monitor_period", 5)), 1),
        "force_threshold": float(getattr(args, "runtime_collision_monitor_force_threshold", 0.05)),
        "max_events": max(int(getattr(args, "runtime_collision_monitor_max_events", 24)), 0),
        "step": 0,
        "sampled_steps": 0,
        "contact_steps": 0,
        "max_force": 0.0,
        "events": [],
        "sources": {},
    }


def _runtime_collision_set_stage(base_env: Any, stage: str) -> str | None:
    monitor = getattr(base_env, "_runtime_collision_monitor", None)
    if not isinstance(monitor, dict) or not monitor.get("enabled"):
        return None
    previous = monitor.get("stage")
    monitor["stage"] = str(stage)
    return str(previous) if previous is not None else None


def _runtime_collision_sample(base_env: Any, profile: dict[str, Any] | None = None) -> None:
    monitor = getattr(base_env, "_runtime_collision_monitor", None)
    if not isinstance(monitor, dict) or not monitor.get("enabled"):
        return
    monitor["step"] = int(monitor.get("step", 0)) + 1
    period = max(int(monitor.get("sample_period", 5)), 1)
    if int(monitor["step"]) % period != 0:
        return
    started = time.perf_counter()
    try:
        locked = monitor.get("locked", {})
        if not isinstance(locked, dict):
            return
        current_role = str(monitor.get("role", ""))
        current_actor = locked.get(current_role).actor if current_role in locked else None
        target_roles = [role for role in list(monitor.get("target_roles", [])) if role in locked]
        if not target_roles:
            return
        agent = base_env.agent
        sources: list[tuple[str, Any]] = []
        if current_actor is not None:
            sources.append(("held_actor", current_actor))
        for source_name, link_attr in (("left_finger", "finger1_link"), ("right_finger", "finger2_link")):
            link = getattr(agent, link_attr, None)
            if link is not None:
                sources.append((source_name, link))
        threshold = float(monitor.get("force_threshold", 0.05))
        step_hits: list[dict[str, Any]] = []
        for target_role in target_roles:
            target_actor = locked[target_role].actor
            for source_name, source_actor in sources:
                if source_actor is target_actor:
                    continue
                force = _runtime_collision_contact_force(base_env, source_actor, target_actor)
                if force <= threshold:
                    continue
                hit = {
                    "step": int(monitor.get("step", 0)),
                    "stage": str(monitor.get("stage", "")),
                    "current_role": current_role,
                    "hit_role": str(target_role),
                    "source": source_name,
                    "source_actor": _runtime_collision_actor_name(source_actor),
                    "target_actor": _runtime_collision_actor_name(target_actor),
                    "force": float(force),
                }
                step_hits.append(hit)
                monitor["max_force"] = max(float(monitor.get("max_force", 0.0)), float(force))
                source_summary = monitor.setdefault("sources", {}).setdefault(source_name, {"contact_steps": 0, "max_force": 0.0})
                source_summary["max_force"] = max(float(source_summary.get("max_force", 0.0)), float(force))
        monitor["sampled_steps"] = int(monitor.get("sampled_steps", 0)) + 1
        if step_hits:
            monitor["contact_steps"] = int(monitor.get("contact_steps", 0)) + 1
            for source_name in {str(item["source"]) for item in step_hits}:
                source_summary = monitor.setdefault("sources", {}).setdefault(source_name, {"contact_steps": 0, "max_force": 0.0})
                source_summary["contact_steps"] = int(source_summary.get("contact_steps", 0)) + 1
            events = monitor.setdefault("events", [])
            max_events = int(monitor.get("max_events", 24))
            for hit in step_hits:
                if len(events) < max_events:
                    events.append(hit)
    finally:
        _profile_increment(profile, "runtime_collision_monitor_sec", time.perf_counter() - started)


def _runtime_collision_finish_role(base_env: Any) -> dict[str, Any]:
    monitor = getattr(base_env, "_runtime_collision_monitor", None)
    if not isinstance(monitor, dict) or not monitor.get("enabled"):
        return {"enabled": False}
    locked = monitor.get("locked", {})
    baseline_poses = monitor.get("baseline_poses", {})
    drift_reports: dict[str, Any] = {}
    max_drift_position = 0.0
    max_drift_orientation = 0.0
    if isinstance(locked, dict) and isinstance(baseline_poses, dict):
        for target_role, baseline_pose in baseline_poses.items():
            if target_role not in locked:
                continue
            try:
                drift = _pose_to_pose_error(_actor_pose(locked[target_role].actor), baseline_pose)
            except Exception:
                continue
            drift_reports[str(target_role)] = drift
            max_drift_position = max(max_drift_position, float(drift.get("position_error_m", 0.0)))
            max_drift_orientation = max(max_drift_orientation, float(drift.get("orientation_error_deg", 0.0)))
    summary = {
        "enabled": True,
        "role": str(monitor.get("role", "")),
        "target_roles": list(monitor.get("target_roles", [])),
        "sample_period": int(monitor.get("sample_period", 5)),
        "force_threshold": float(monitor.get("force_threshold", 0.05)),
        "total_steps": int(monitor.get("step", 0)),
        "sampled_steps": int(monitor.get("sampled_steps", 0)),
        "contact_steps": int(monitor.get("contact_steps", 0)),
        "detected": int(monitor.get("contact_steps", 0)) > 0,
        "max_force": float(monitor.get("max_force", 0.0)),
        "sources": monitor.get("sources", {}),
        "events": list(monitor.get("events", [])),
        "completed_role_drift": drift_reports,
        "max_completed_role_drift_position_m": float(max_drift_position),
        "max_completed_role_drift_orientation_deg": float(max_drift_orientation),
    }
    base_env._runtime_collision_monitor = {"enabled": False}
    return summary


def _accumulate_profile(dest: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if key.endswith("_calls"):
            dest[key] = int(dest.get(key, 0)) + int(value)
        else:
            dest[key] = float(dest.get(key, 0.0)) + float(value)


def _stop_actor(actor: Any) -> None:
    try:
        actor.set_linear_velocity(np.zeros(3, dtype=np.float32))
    except Exception:
        pass
    try:
        actor.set_angular_velocity(np.zeros(3, dtype=np.float32))
    except Exception:
        pass


def _copy_pose(pose: Any) -> sapien.Pose | None:
    try:
        position = np.asarray(pose.p, dtype=np.float32).reshape(-1)[:3]
        quaternion = np.asarray(pose.q, dtype=np.float32).reshape(-1)[:4]
    except Exception:
        return None
    if position.size != 3 or quaternion.size != 4:
        return None
    return sapien.Pose(p=position.tolist(), q=quaternion.tolist())


def _set_actor_pose_and_stop(actor: Any, pose: sapien.Pose) -> None:
    actor.set_pose(pose)
    _stop_actor(actor)


def _sync_locked_panel_pose(base_env: Any, role: str, pose: sapien.Pose) -> bool:
    snap = getattr(base_env, "magnetic_snap", None)
    if snap is None or not hasattr(snap, "locked_panel_poses"):
        return False
    position = np.asarray(pose.p, dtype=np.float32).reshape(-1)[:3]
    quaternion = np.asarray(pose.q, dtype=np.float32).reshape(-1)[:4]
    updated: list[Any] = []
    changed = False
    for locked_pose in list(snap.locked_panel_poses):
        if getattr(locked_pose, "role", None) == role:
            updated.append(
                LockedPanelPose(
                    role=role,
                    actor=locked_pose.actor,
                    position=position.copy(),
                    quaternion=quaternion.copy(),
                )
            )
            changed = True
        else:
            updated.append(locked_pose)
    if not changed:
        return False
    snap.locked_panel_poses = updated
    try:
        snap._refresh_locked_panel_map()
    except Exception:
        pass
    return True


def _set_active_pregrasp_pose_freeze(
    *,
    base_env: Any,
    actor: Any | None,
    enabled: bool,
) -> dict[str, Any]:
    if actor is None or not enabled:
        setattr(base_env, "_multi_wall_active_pregrasp_freeze", None)
        return {"enabled": False}
    pose = _copy_pose(_actor_pose(actor))
    if pose is None:
        setattr(base_env, "_multi_wall_active_pregrasp_freeze", None)
        return {"enabled": False, "reason": "pose_unavailable"}
    _set_actor_pose_and_stop(actor, pose)
    freeze = {
        "actor": actor,
        "pose": pose,
        "refresh_count": 0,
    }
    setattr(base_env, "_multi_wall_active_pregrasp_freeze", freeze)
    return {
        "enabled": True,
        "mode": "pose_refresh",
        "release_phase": "after_gripper_close",
        "pose": _pose_to_report(pose),
    }


def _enforce_active_pregrasp_pose_freeze(env: Any) -> None:
    try:
        base_env = env.unwrapped
    except Exception:
        return
    freeze = getattr(base_env, "_multi_wall_active_pregrasp_freeze", None)
    if not isinstance(freeze, dict):
        return
    actor = freeze.get("actor")
    pose = freeze.get("pose")
    if actor is None or pose is None:
        return
    try:
        _set_actor_pose_and_stop(actor, pose)
        freeze["refresh_count"] = int(freeze.get("refresh_count", 0)) + 1
    except Exception:
        return


def _clear_active_pregrasp_pose_freeze(base_env: Any, actor: Any | None = None) -> dict[str, Any]:
    freeze = getattr(base_env, "_multi_wall_active_pregrasp_freeze", None)
    if not isinstance(freeze, dict):
        return {"enabled": False}
    setattr(base_env, "_multi_wall_active_pregrasp_freeze", None)
    freeze_actor = freeze.get("actor")
    if actor is None:
        actor = freeze_actor
    if actor is not None:
        _stop_actor(actor)
    return {
        "enabled": True,
        "cleared": True,
        "mode": "pose_refresh",
        "refresh_count": int(freeze.get("refresh_count", 0)),
    }


def _enforce_non_current_role_freeze(env: Any) -> None:
    try:
        base_env = env.unwrapped
    except Exception:
        return
    frozen = getattr(base_env, "_multi_wall_frozen_role_poses", None)
    if not frozen:
        return
    every_n = int(getattr(base_env, "_multi_wall_freeze_enforce_every_n_steps", 0) or 0)
    if every_n <= 0:
        return
    count = int(getattr(base_env, "_multi_wall_freeze_enforce_step_count", 0)) + 1
    setattr(base_env, "_multi_wall_freeze_enforce_step_count", count)
    if count % every_n != 0:
        return
    for item in list(frozen.values()):
        try:
            _set_actor_pose_and_stop(item["actor"], item["pose"])
        except Exception:
            continue


def _set_non_current_role_freeze(
    *,
    base_env: Any,
    locked: dict[str, Any],
    active_role: str,
    args: argparse.Namespace,
    log: Any,
) -> dict[str, Any]:
    if not bool(getattr(args, "freeze_non_current_roles", True)):
        setattr(base_env, "_multi_wall_frozen_role_poses", {})
        setattr(base_env, "_multi_wall_freeze_enforce_every_n_steps", 0)
        return {"enabled": False}
    include_floor = bool(getattr(args, "freeze_non_current_include_floor", True))
    enforce_every_n = max(int(getattr(args, "freeze_non_current_enforce_every_n_steps", 0) or 0), 0)
    frozen: dict[str, dict[str, Any]] = {}
    for frozen_role, locked_item in locked.items():
        if frozen_role == active_role:
            continue
        if frozen_role == "floor" and not include_floor:
            continue
        actor = getattr(locked_item, "actor", None)
        if actor is None:
            continue
        pose = _copy_pose(_actor_pose(actor))
        if pose is None:
            continue
        try:
            previous_axes = actor.get_locked_motion_axes()
        except Exception:
            previous_axes = None
        frozen[frozen_role] = {"actor": actor, "pose": pose, "previous_locked_motion_axes": previous_axes}
        _set_actor_pose_and_stop(actor, pose)
        try:
            actor.set_locked_motion_axes([True, True, True, True, True, True])
        except Exception:
            pass
    setattr(base_env, "_multi_wall_frozen_role_poses", frozen)
    setattr(base_env, "_multi_wall_freeze_enforce_every_n_steps", enforce_every_n)
    setattr(base_env, "_multi_wall_freeze_enforce_step_count", 0)
    report = {
        "enabled": True,
        "active_role": active_role,
        "roles": sorted(frozen),
        "mode": "locked_motion_axes",
        "enforce_every_n_steps": enforce_every_n,
    }
    if frozen:
        log(f"{active_role}: frozen non-current roles: {report['roles']}")
    return report


def _clear_non_current_role_freeze(base_env: Any) -> None:
    frozen = getattr(base_env, "_multi_wall_frozen_role_poses", None) or {}
    for item in list(frozen.values()):
        actor = item.get("actor")
        previous_axes = item.get("previous_locked_motion_axes")
        if actor is None or previous_axes is None:
            continue
        try:
            actor.set_locked_motion_axes(previous_axes)
        except Exception:
            pass
        _stop_actor(actor)
    setattr(base_env, "_multi_wall_frozen_role_poses", {})
    setattr(base_env, "_multi_wall_freeze_enforce_every_n_steps", 0)
    setattr(base_env, "_multi_wall_freeze_enforce_step_count", 0)


def _set_loaded_role_pose_freeze(
    *,
    base_env: Any,
    locked: dict[str, Any],
    roles: list[str],
) -> dict[str, Any]:
    frozen = getattr(base_env, "_multi_wall_loaded_role_freeze", None)
    if not isinstance(frozen, dict):
        frozen = {}
    for role in roles:
        locked_item = locked.get(role)
        actor = getattr(locked_item, "actor", None)
        if actor is None:
            continue
        pose = _copy_pose(_actor_pose(actor))
        if pose is None:
            continue
        try:
            previous_axes = actor.get_locked_motion_axes()
        except Exception:
            previous_axes = None
        _set_actor_pose_and_stop(actor, pose)
        try:
            actor.set_locked_motion_axes([True, True, True, True, True, True])
        except Exception:
            pass
        frozen[role] = {
            "actor": actor,
            "pose": pose,
            "previous_locked_motion_axes": previous_axes,
            "refresh_count": 0,
        }
    setattr(base_env, "_multi_wall_loaded_role_freeze", frozen)
    return {
        "enabled": bool(frozen),
        "roles": sorted(frozen),
        "mode": "locked_motion_axes_plus_pose_refresh",
    }


def _clear_loaded_role_pose_freeze(base_env: Any, role: str) -> dict[str, Any]:
    frozen = getattr(base_env, "_multi_wall_loaded_role_freeze", None)
    if not isinstance(frozen, dict) or role not in frozen:
        return {"enabled": False}
    item = frozen.pop(role)
    actor = item.get("actor")
    previous_axes = item.get("previous_locked_motion_axes")
    if actor is not None and previous_axes is not None:
        try:
            actor.set_locked_motion_axes(previous_axes)
        except Exception:
            pass
        _stop_actor(actor)
    setattr(base_env, "_multi_wall_loaded_role_freeze", frozen)
    return {
        "enabled": True,
        "cleared": True,
        "role": role,
        "refresh_count": int(item.get("refresh_count", 0)),
    }


def _enforce_loaded_role_pose_freeze(env: Any) -> None:
    try:
        base_env = env.unwrapped
    except Exception:
        return
    frozen = getattr(base_env, "_multi_wall_loaded_role_freeze", None)
    if not isinstance(frozen, dict) or not frozen:
        return
    for item in list(frozen.values()):
        actor = item.get("actor")
        pose = item.get("pose")
        if actor is None or pose is None:
            continue
        try:
            _set_actor_pose_and_stop(actor, pose)
            item["refresh_count"] = int(item.get("refresh_count", 0)) + 1
        except Exception:
            continue


class _NextRolePrefetchManager:
    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bbs-next-role-prefetch")
        self._lock = threading.Lock()
        self._future: Any = None
        self._future_role: str | None = None

    def submit(
        self,
        *,
        role: str,
        planner: RM75CuRoboPlanner,
        base_env: Any,
        locked: dict[str, Any],
        fixtures: list[dict[str, Any]],
        actor: Any,
        target_actor_pose: sapien.Pose | None,
        start_q: np.ndarray,
        args: argparse.Namespace,
    ) -> None:
        if not self.enabled or target_actor_pose is None:
            return
        role_name = str(role)
        start_q_np = np.asarray(start_q, dtype=np.float32).reshape(7)
        candidate_pool = list(_grasp_candidates_for_role(args, role_name))

        def worker() -> dict[str, Any]:
            primary, fallback, report = _fast_chain_preselect_grasp_candidates(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                actor=actor,
                role=role_name,
                candidate_pool=candidate_pool,
                start_q=start_q_np,
                target_actor_pose=target_actor_pose,
                args=args,
            )
            return {
                "role": role_name,
                "start_q": start_q_np.tolist(),
                "candidate_count": len(candidate_pool),
                "primary": primary,
                "fallback": fallback,
                "report": report,
            }

        with self._lock:
            if self._future is not None and not self._future.done():
                return
            self._future_role = role_name
            self._future = self._executor.submit(worker)

    def consume(self, role: str) -> dict[str, Any] | None:
        with self._lock:
            if self._future is None or self._future_role != str(role):
                return None
            future = self._future
            self._future = None
            self._future_role = None
        try:
            started = time.perf_counter()
            result = future.result()
            result["consume_wait_sec"] = time.perf_counter() - started
            return result
        except Exception as exc:
            print(f"[prefetch] next-role preselect for {role} failed: {exc}")
            return None

    def close(self) -> None:
        with self._lock:
            future = self._future
            self._future = None
            self._future_role = None
        if future is not None:
            try:
                future.cancel()
            except Exception:
                pass
        self._executor.shutdown(wait=True, cancel_futures=True)


def _install_live_recorder(
    env: Any,
    writer: Any | None,
    record_every: int,
    profile: dict[str, Any] | None,
    *,
    args: argparse.Namespace | None = None,
    real_exec: Any | None = None,
    log: Any | None = None,
    human_render: bool = False,
    human_render_every: int = 1,
) -> Any:
    original_local = globals()["_step_action"]
    original_single = single_wall_module._step_action
    original_realman = realman_module._step_action
    snap = getattr(env.unwrapped, "magnetic_snap", None)
    original_snap_apply = getattr(snap, "apply", None) if snap is not None else None
    counter = {"step": 0}

    if original_snap_apply is not None:

        def profiled_snap_apply() -> None:
            started = time.perf_counter()
            original_snap_apply()
            _profile_increment(profile, "magnetic_apply_sec", time.perf_counter() - started)
            _profile_increment(profile, "magnetic_apply_calls", 1.0)

        snap.apply = profiled_snap_apply

    def wrapped_step_action(step_env: Any, target_q: np.ndarray, gripper: float, _writer: Any | None, _record_every: int, index: int) -> None:
        started = time.perf_counter()
        base_env = step_env.unwrapped
        action = np.zeros(step_env.action_space.shape, dtype=step_env.action_space.dtype)
        phase_started = time.perf_counter()
        action[:7] = realman_module._format_arm_action(base_env, target_q)
        _profile_increment(profile, "step_format_arm_sec", time.perf_counter() - phase_started)
        phase_started = time.perf_counter()
        action[-1] = realman_module._format_gripper_action(base_env, gripper)
        _profile_increment(profile, "step_format_gripper_sec", time.perf_counter() - phase_started)
        phase_started = time.perf_counter()
        step_env.step(action)
        _enforce_loaded_role_pose_freeze(step_env)
        _enforce_non_current_role_freeze(step_env)
        _enforce_active_pregrasp_pose_freeze(step_env)
        realman_module._apply_held_actor_pose_lock(base_env)
        _enforce_active_pregrasp_pose_freeze(step_env)
        _enforce_non_current_role_freeze(step_env)
        _enforce_loaded_role_pose_freeze(step_env)
        _runtime_collision_sample(base_env, profile)
        _profile_increment(profile, "env_step_sec", time.perf_counter() - phase_started)
        if real_exec is not None:
            real_started = time.perf_counter()
            try:
                real_gripper = None if args is None else _sim_gripper_to_real(float(gripper), args)
                real_exec.send_action(
                    arm_q=np.asarray(target_q, dtype=np.float32).reshape(-1)[:7],
                    gripper_pos=real_gripper,
                )
                hz = float(max(getattr(args, "real_control_hz", 30.0) if args is not None else 30.0, 1e-3))
                time.sleep(1.0 / hz)
            except Exception as exc:
                if callable(log):
                    log(f"real execution failed at step={counter['step']}: {type(exc).__name__}: {exc}")
                raise RuntimeError(f"real execution failed at step={counter['step']}: {exc}") from exc
            finally:
                _profile_increment(profile, "real_step_action_sec", time.perf_counter() - real_started)
        _profile_increment(profile, "step_action_sec", time.perf_counter() - started)
        if writer is not None and counter["step"] % max(int(record_every), 1) == 0:
            _profile_call(profile, "append_frame_sec", _append_frame, writer, step_env)
        if human_render and counter["step"] % max(int(human_render_every), 1) == 0:
            _profile_call(profile, "human_render_sec", step_env.render)
        counter["step"] += 1

    globals()["_step_action"] = wrapped_step_action
    single_wall_module._step_action = wrapped_step_action
    realman_module._step_action = wrapped_step_action

    def restore() -> int:
        globals()["_step_action"] = original_local
        single_wall_module._step_action = original_single
        realman_module._step_action = original_realman
        if snap is not None and original_snap_apply is not None:
            snap.apply = original_snap_apply
        return int(counter["step"])

    return restore


def _densify_joint_path(path: np.ndarray, max_joint_step: float) -> np.ndarray:
    path = np.asarray(path, dtype=np.float32).reshape(-1, 7)
    if path.shape[0] <= 1 or float(max_joint_step) <= 0.0:
        return path
    dense: list[np.ndarray] = [path[0, :7].astype(np.float32)]
    for target in path[1:]:
        start = dense[-1]
        delta = float(np.linalg.norm(np.asarray(target[:7], dtype=np.float32) - start))
        count = max(int(np.ceil(delta / float(max_joint_step))), 1)
        for index in range(1, count + 1):
            alpha = index / float(count)
            dense.append((start * (1.0 - alpha) + target[:7] * alpha).astype(np.float32))
    return np.asarray(dense, dtype=np.float32).reshape(-1, 7)


def _adaptive_step_count(
    *,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    base_steps: int,
    max_joint_step: float,
    max_steps: int,
) -> int:
    steps = max(int(base_steps), 1)
    if float(max_joint_step) > 0.0:
        distance = _joint_distance(np.asarray(goal_q, dtype=np.float32), np.asarray(start_q, dtype=np.float32))
        steps = max(steps, int(np.ceil(distance / float(max_joint_step))))
    if int(max_steps) > 0:
        steps = min(steps, int(max_steps))
    return max(steps, 1)


def _add_adaptive_joint_segment(
    *,
    env: Any,
    arrays: dict[str, np.ndarray],
    segments: list[dict[str, Any]],
    name: str,
    goal_q: np.ndarray,
    gripper: float,
    base_steps: int,
    action_repeat: int,
    final_hold: int,
    args: argparse.Namespace,
    max_joint_step_override: float | None = None,
    max_steps_override: int | None = None,
) -> dict[str, Any]:
    base_env = env.unwrapped
    start_q = _current_q(base_env)
    goal_q = _nearest_equivalent_joint_configuration(np.asarray(goal_q, dtype=np.float32).reshape(7), start_q)
    max_joint_step = float(max_joint_step_override) if max_joint_step_override is not None and float(max_joint_step_override) > 0.0 else float(getattr(args, "max_joint_step", 0.06))
    max_steps = int(max_steps_override) if max_steps_override is not None and int(max_steps_override) > 0 else int(getattr(args, "max_segment_steps", 420))
    steps = _adaptive_step_count(
        start_q=start_q,
        goal_q=np.asarray(goal_q, dtype=np.float32).reshape(7),
        base_steps=int(base_steps),
        max_joint_step=max_joint_step,
        max_steps=max_steps,
    )
    previous_stage = _runtime_collision_set_stage(base_env, name)
    try:
        segment = _add_joint_segment(
            env=env,
            arrays=arrays,
            segments=segments,
            name=name,
            goal_q=goal_q,
            gripper=gripper,
            steps=steps,
            action_repeat=action_repeat,
            final_hold=final_hold,
        )
    finally:
        if previous_stage is not None:
            _runtime_collision_set_stage(base_env, previous_stage)
    segment["base_steps"] = int(base_steps)
    segment["adaptive_steps"] = int(steps)
    segment["joint_distance"] = _joint_distance(np.asarray(goal_q, dtype=np.float32), start_q)
    segment["adaptive_max_joint_step"] = float(max_joint_step)
    segment["adaptive_max_steps"] = int(max_steps)
    return segment


def _wrap_to_pi(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _wrapped_joint_delta(goal_q: np.ndarray, start_q: np.ndarray) -> np.ndarray:
    goal = np.asarray(goal_q, dtype=np.float32).reshape(-1)
    start = np.asarray(start_q, dtype=np.float32).reshape(-1)
    delta = goal - start
    return np.asarray([_wrap_to_pi(value) for value in delta], dtype=np.float32)


def _nearest_equivalent_joint_configuration(goal_q: np.ndarray, start_q: np.ndarray) -> np.ndarray:
    goal = np.asarray(goal_q, dtype=np.float32).reshape(-1).copy()
    start = np.asarray(start_q, dtype=np.float32).reshape(-1)
    delta = _wrapped_joint_delta(goal, start)
    return (start + delta).astype(np.float32)


def _max_abs_joint_delta_deg(goal_q: np.ndarray, start_q: np.ndarray, joint_indices: list[int] | tuple[int, ...] | None = None) -> float:
    delta = _wrapped_joint_delta(goal_q, start_q)
    if joint_indices:
        delta = delta[list(joint_indices)]
    return float(np.rad2deg(np.max(np.abs(delta)))) if delta.size else 0.0


def _record_existing_joint_path(
    *,
    env: Any,
    arrays: dict[str, np.ndarray],
    segments: list[dict[str, Any]],
    name: str,
    path: np.ndarray,
    gripper: float,
    final_hold: int,
    max_joint_step: float = 0.06,
    max_waypoints: int = 0,
) -> np.ndarray:
    base_env = env.unwrapped
    previous_stage = _runtime_collision_set_stage(base_env, name)
    original_waypoints = int(np.asarray(path, dtype=np.float32).reshape(-1, 7).shape[0])
    path = _densify_joint_path(path, float(max_joint_step))
    densified_waypoints = int(path.shape[0])
    if int(max_waypoints) > 1 and path.shape[0] > int(max_waypoints):
        indices = np.linspace(0, path.shape[0] - 1, int(max_waypoints)).round().astype(int)
        path = path[indices]
    key = f"q_{len([item for item in segments if item.get('type') == 'joint_path']):03d}_{name}"
    arrays[key] = path.astype(np.float32)
    try:
        for target in path:
            _step_action(env, target[:7], float(gripper), None, 1, 0)
        final = path[-1, :7]
        for index in range(max(int(final_hold), 0)):
            _step_action(env, final, float(gripper), None, 1, index)
    finally:
        if previous_stage is not None:
            _runtime_collision_set_stage(base_env, previous_stage)
    segments.append(
        {
            "type": "joint_path",
            "name": name,
            "array_key": key,
            "gripper": float(gripper),
            "action_repeat": 1,
            "final_hold": int(final_hold),
            "waypoints": int(path.shape[0]),
            "original_waypoints": int(original_waypoints),
            "densified_waypoints": int(densified_waypoints),
            "max_waypoints": int(max_waypoints),
            "max_joint_step": float(max_joint_step),
            "no_collapse": True,
        }
    )
    return path[-1, :7].astype(np.float32)


def _plan_joint_motion_to_goal(
    *,
    planner: RM75CuRoboPlanner,
    base_env: Any,
    locked: dict[str, Any],
    fixtures: list[dict[str, Any]],
    start_q: np.ndarray,
    goal_q: np.ndarray,
    timeout: float,
    max_attempts: int,
    enable_graph: bool,
    num_graph_seeds: int,
) -> tuple[bool, np.ndarray | None, dict[str, Any]]:
    obstacles = _world_obstacles_for_stage(
        base_env,
        locked,
        fixtures,
        exclude_role=None,
        exclude_roles=None,
    )
    planner.set_world_from_obstacles(cuboids=obstacles)
    result = planner.plan_to_joint(
        np.asarray(start_q, dtype=np.float32).reshape(7),
        np.asarray(goal_q, dtype=np.float32).reshape(7),
        enable_graph=bool(enable_graph),
        max_attempts=int(max_attempts),
        timeout=float(timeout),
        num_trajopt_seeds=1,
        num_graph_seeds=int(num_graph_seeds),
    )
    report = {
        "success": bool(result.success),
        "status": result.status,
        "solve_time": float(result.solve_time),
        "ik_time": float(result.ik_time),
        "trajopt_time": float(result.trajopt_time),
        "obstacle_count": len(obstacles),
        "debug": result.debug,
        "planner_mode": "motion_gen_joint_collision_checked",
        "enable_graph": bool(enable_graph),
        "max_attempts": int(max_attempts),
        "num_graph_seeds": int(num_graph_seeds),
        "goal_joint": np.asarray(goal_q, dtype=np.float32).reshape(7).tolist(),
    }
    if not result.success or result.joint_path is None:
        return False, None, report
    path = np.asarray(result.joint_path, dtype=np.float32).reshape(-1, 7)
    report["waypoints"] = int(path.shape[0])
    report["final_goal_joint_distance"] = _joint_distance(path[-1, :7], np.asarray(goal_q, dtype=np.float32).reshape(7))
    return True, path, report


def _args_for_role(args: argparse.Namespace, role: str) -> argparse.Namespace:
    role_args = argparse.Namespace(**vars(args))
    mapping_text = str(getattr(args, "release_candidate_indices", "") or "")
    for item in mapping_text.split(","):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        if key.strip() == role:
            values = _int_csv(value)
            if len(values) == 1:
                role_args.release_candidate_index = int(values[0])
            elif len(values) > 1:
                role_args.release_candidate_index = -1
            break
    ignore_mapping_text = str(getattr(args, "release_ignore_roles_by_role", "") or "")
    for item in ignore_mapping_text.split(";"):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        if key.strip() == role:
            role_args.release_ignore_roles = value.strip()
            break
    return role_args


def _role_supports(role: str, completed: list[str]) -> list[str]:
    neighbors = {
        "right_wall": {"back_wall", "front_wall"},
        "back_wall": {"right_wall", "left_wall"},
        "left_wall": {"back_wall", "front_wall"},
        "front_wall": {"left_wall", "right_wall"},
    }
    supports = ["floor"]
    for item in completed:
        if item in neighbors.get(role, set()) and item not in supports:
            supports.append(item)
    return supports


def _release_ignore_mapping_for_roles(roles: list[str], completed: list[str]) -> str:
    mapping: list[str] = []
    completed_so_far = list(completed)
    for role in roles:
        supports = _role_supports(role, completed_so_far)
        mapping.append(f"{role}:{','.join(supports)}")
        if role not in completed_so_far:
            completed_so_far.append(role)
    return ";".join(mapping)


def _active_structure_roles(base_env: Any) -> set[str]:
    roles = {"floor"}
    for active_connection in base_env.magnetic_snap.active_connections:
        if not active_connection.active:
            continue
        connection = active_connection.connection
        roles.add(connection.parent)
        roles.add(connection.child)
    return roles


def _wall_connection_potential(base_env: Any, role: str, built_roles: set[str] | None = None) -> int:
    if role not in FIRST_LAYER_WALL_ROLES:
        return 0
    built = set(built_roles) if built_roles is not None else _active_structure_roles(base_env)
    built.add("floor")
    snap = base_env.magnetic_snap
    seen: set[tuple[tuple[str, str, str], tuple[str, str, str]]] = set()
    count = 0
    for connection in snap.connections:
        if connection.parent != role and connection.child != role:
            continue
        other = connection.child if connection.parent == role else connection.parent
        if other not in built:
            continue
        key = snap._connection_key(connection)
        if key in seen:
            continue
        seen.add(key)
        count += 1
    return count


def _required_active_connections_for_role(base_env: Any, role: str, args: argparse.Namespace) -> int:
    if role == "floor":
        return 0
    min_connections = max(int(getattr(args, "min_active_connections", 1)), 0)
    if role in FIRST_LAYER_WALL_ROLES:
        potential = _wall_connection_potential(base_env, role)
        return max(min_connections, 2 if potential >= 2 else 1)
    return max(min_connections, 1)


def _release_connection_geometry_report(
    base_env: Any,
    role: str,
    release_actor_pose: sapien.Pose,
    args: argparse.Namespace,
) -> dict[str, Any]:
    snap = getattr(base_env, "magnetic_snap", None)
    if snap is None:
        return {"enabled": False}
    locked_by_role = snap._locked_by_role()
    release_position = np.asarray(release_actor_pose.p, dtype=np.float32).reshape(3)
    release_quaternion = np.asarray(release_actor_pose.q, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(release_quaternion))
    if norm > 1e-8:
        release_quaternion = release_quaternion / norm
    connection_reports: list[dict[str, Any]] = []
    for connection in snap.connections:
        if connection.parent != role and connection.child != role:
            continue
        other_role = connection.child if connection.parent == role else connection.parent
        if other_role in snap.disabled_roles:
            continue
        if not snap._connection_allowed(connection):
            continue
        if not snap._edge_allowed(connection.parent, connection.parent_edge):
            continue
        if not snap._edge_allowed(connection.child, connection.child_edge):
            continue
        parent = locked_by_role.get(connection.parent)
        child = locked_by_role.get(connection.child)
        if parent is None or child is None:
            continue
        parent_points, child_points = snap._matched_connection_points(
            connection,
            parent,
            child,
            points_per_edge=1 if snap.magnet_mode == "edge_drive" else 2,
            use_locked_pose=False,
        )
        if connection.parent == role:
            parent_world = snap._transform_local_points(release_position, release_quaternion, parent_points)
            child_world = snap._current_world_points_array(child.actor, child_points)
        else:
            parent_world = snap._current_world_points_array(parent.actor, parent_points)
            child_world = snap._transform_local_points(release_position, release_quaternion, child_points)
        direct_error = sum(float(np.linalg.norm(a - b)) for a, b in zip(parent_world, child_world))
        flipped_error = sum(float(np.linalg.norm(a - b)) for a, b in zip(parent_world, reversed(child_world)))
        if flipped_error < direct_error:
            child_world = list(reversed(child_world))
        distances = snap._matched_point_distances(list(parent_world), list(child_world))
        if distances.size == 0:
            continue
        max_error = float(np.max(distances))
        mean_error = float(np.mean(distances))
        min_pair_distance = float(snap._min_pairwise_point_distance(list(parent_world), list(child_world)))
        connection_reports.append(
            {
                "parent": connection.parent,
                "parent_edge": connection.parent_edge,
                "child": connection.child,
                "child_edge": connection.child_edge,
                "other_role": other_role,
                "max_point_error": max_error,
                "mean_point_error": mean_error,
                "min_pair_distance": min_pair_distance,
                "within_attach": bool(max_error <= float(snap.attach_distance)),
                "within_attract": bool(min_pair_distance <= float(snap.attach_distance) or max_error <= float(snap.attract_distance)),
            }
        )
    connection_reports.sort(
        key=lambda item: (
            0 if bool(item["within_attach"]) else 1,
            0 if bool(item["within_attract"]) else 1,
            float(item["max_point_error"]),
            float(item["min_pair_distance"]),
        )
    )
    required = min(_required_active_connections_for_role(base_env, role, args), len(connection_reports))
    attach_count = sum(1 for item in connection_reports if bool(item["within_attach"]))
    attract_count = sum(1 for item in connection_reports if bool(item["within_attract"]))
    sorted_max_errors = sorted(float(item["max_point_error"]) for item in connection_reports)
    required_max_error = (
        sorted_max_errors[max(required - 1, 0)]
        if required > 0 and sorted_max_errors
        else float("inf")
    )
    best_max_error = sorted_max_errors[0] if sorted_max_errors else float("inf")
    best_min_pair_distance = min((float(item["min_pair_distance"]) for item in connection_reports), default=float("inf"))
    missing_attach = max(required - attach_count, 0)
    missing_attract = max(required - attract_count, 0)
    return {
        "enabled": True,
        "required_connections": int(required),
        "attach_distance": float(snap.attach_distance),
        "attract_distance": float(snap.attract_distance),
        "attachable_count": int(attach_count),
        "attractable_count": int(attract_count),
        "missing_attach_count": int(missing_attach),
        "missing_attract_count": int(missing_attract),
        "best_max_point_error": float(best_max_error),
        "required_max_point_error": float(required_max_error),
        "best_min_pair_distance": float(best_min_pair_distance),
        "connections": connection_reports[:4],
        "rank": (
            int(missing_attach),
            int(missing_attract),
            float(required_max_error),
            float(best_max_error),
            float(best_min_pair_distance),
        ),
    }


def _order_roles_by_connection_potential(
    base_env: Any,
    roles: list[str],
    *,
    built_roles: list[str],
) -> list[str]:
    remaining = list(roles)
    ordered: list[str] = []
    active_roles = set(built_roles)
    active_roles.add("floor")
    original_index = {role: index for index, role in enumerate(roles)}
    while remaining:
        best_index = 0
        best_key: tuple[int, int] | None = None
        for index, role in enumerate(remaining):
            if role in FIRST_LAYER_WALL_ROLES:
                key = (_wall_connection_potential(base_env, role, active_roles), -original_index.get(role, index))
            else:
                key = (-1, -original_index.get(role, index))
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        role = remaining.pop(best_index)
        ordered.append(role)
        if role in FIRST_LAYER_WALL_ROLES:
            active_roles.add(role)
    return ordered


def _apply_float_override(target: Any, attr: str, value: float | None, overrides: dict[str, float]) -> None:
    if value is None:
        return
    setattr(target, attr, float(value))
    overrides[attr] = float(getattr(target, attr))


def _apply_int_override(target: Any, attr: str, value: int | None, overrides: dict[str, float]) -> None:
    if value is None:
        return
    setattr(target, attr, int(value))
    overrides[attr] = int(getattr(target, attr))


def _write_run_outputs(
    *,
    summary_path: Path,
    manifest_path: Path,
    arrays_path: Path,
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    reports: list[dict[str, Any]],
    final: dict[str, Any],
) -> dict[str, Any]:
    np.savez_compressed(arrays_path, **arrays)
    manifest_path.write_text(json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    result = {
        "summary": str(summary_path),
        "manifest": str(manifest_path),
        "arrays": str(arrays_path),
        "reports": reports,
        "final": final,
    }
    summary_path.write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _actor_velocity(actor: Any, name: str) -> list[float]:
    getter = getattr(actor, name, None)
    if getter is None:
        return [0.0, 0.0, 0.0]
    try:
        return np.asarray(getter(), dtype=np.float32).reshape(-1)[:3].tolist()
    except Exception:
        return [0.0, 0.0, 0.0]


def _pose_delta_error(current: sapien.Pose, reference: sapien.Pose) -> dict[str, float]:
    cp, cq = _pose_arrays(current)
    rp, rq = _pose_arrays(reference)
    dot = min(max(abs(float(np.dot(cq, rq))), -1.0), 1.0)
    return {
        "position_error_m": float(np.linalg.norm(cp - rp)),
        "orientation_error_deg": float(np.rad2deg(2.0 * np.arccos(dot))),
    }


def _predicted_actor_pose_error_from_tcp_q(
    planner: RM75CuRoboPlanner,
    base_env: Any,
    q: np.ndarray,
    actor_to_tcp: sapien.Pose,
    target_actor_pose: sapien.Pose,
) -> dict[str, Any]:
    fk = planner.fk(np.asarray(q, dtype=np.float32).reshape(-1)[:7])
    tcp_pose_base = sapien.Pose(p=fk["position"], q=fk["quaternion"])
    tcp_pose = base_env.agent.robot.pose.sp * tcp_pose_base
    predicted_actor_pose = tcp_pose * actor_to_tcp.inv()
    return {
        "pose_error": _pose_to_pose_error(predicted_actor_pose, target_actor_pose),
        "predicted_actor_pose": _pose_to_report(predicted_actor_pose),
        "predicted_tcp_pose": _pose_to_report(tcp_pose),
    }


def _release_prediction_gate_failed(report: dict[str, Any], args: argparse.Namespace) -> bool:
    max_position_error = float(getattr(args, "release_screen_max_predicted_actor_position_error", 0.0) or 0.0)
    max_orientation_error = float(getattr(args, "release_screen_max_predicted_actor_orientation_error_deg", 0.0) or 0.0)
    pose_error = dict((report or {}).get("pose_error") or {})
    position_error = float(pose_error.get("position_error_m", 0.0) or 0.0)
    orientation_error = float(pose_error.get("orientation_error_deg", 0.0) or 0.0)
    return bool(
        (max_position_error > 0.0 and position_error > max_position_error)
        or (max_orientation_error > 0.0 and orientation_error > max_orientation_error)
    )


def _grasp_quality_report(
    *,
    base_env: Any,
    actor: Any,
    nominal_actor_to_tcp: sapien.Pose,
    grasp_report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    live_actor_to_tcp = _actor_pose(actor).inv() * _tcp_pose(base_env)
    live_position, live_quaternion = _pose_arrays(live_actor_to_tcp)
    nominal_position, nominal_quaternion = _pose_arrays(nominal_actor_to_tcp)
    dot = min(max(abs(float(np.dot(live_quaternion, nominal_quaternion))), -1.0), 1.0)
    position_delta = float(np.linalg.norm(live_position - nominal_position))
    orientation_delta = float(np.rad2deg(2.0 * np.arccos(dot)))
    left_force = float(grasp_report.get("left_force", 0.0) or 0.0)
    right_force = float(grasp_report.get("right_force", 0.0) or 0.0)
    force_sum = left_force + right_force
    force_balance_error = abs(left_force - right_force) / max(force_sum, 1e-6)
    max_position_delta = float(getattr(args, "grasp_quality_max_tcp_position_delta", 0.0) or 0.0)
    max_orientation_delta = float(getattr(args, "grasp_quality_max_tcp_orientation_delta_deg", 0.0) or 0.0)
    min_force = float(getattr(args, "grasp_quality_min_finger_force", 0.0) or 0.0)
    max_force_balance_error = float(getattr(args, "grasp_quality_max_force_balance_error", 0.0) or 0.0)
    is_grasped = bool(grasp_report.get("is_grasped"))
    position_ok = not (max_position_delta > 0.0 and position_delta > max_position_delta)
    orientation_ok = not (max_orientation_delta > 0.0 and orientation_delta > max_orientation_delta)
    force_ok = not (min_force > 0.0 and min(left_force, right_force) < min_force)
    balance_ok = not (max_force_balance_error > 0.0 and force_balance_error > max_force_balance_error)
    success = bool(is_grasped and position_ok and orientation_ok and force_ok and balance_ok)
    live_lock_recovered = False
    live_lock_max_delta = float(getattr(args, "grasp_quality_live_lock_max_tcp_position_delta", 0.0) or 0.0)
    if (
        not success
        and bool(getattr(args, "grasp_quality_allow_live_lock_recovery", False))
        and bool(getattr(args, "lock_held_actor_after_grasp", False))
        and is_grasped
        and not position_ok
        and orientation_ok
        and force_ok
        and balance_ok
        and live_lock_max_delta > 0.0
        and position_delta <= live_lock_max_delta
    ):
        success = True
        live_lock_recovered = True
    return {
        "success": success,
        "live_lock_recovered": bool(live_lock_recovered),
        "live_actor_to_tcp": _pose_to_report(live_actor_to_tcp),
        "nominal_actor_to_tcp": _pose_to_report(nominal_actor_to_tcp),
        "tcp_position_delta_m": position_delta,
        "tcp_orientation_delta_deg": orientation_delta,
        "force_balance_error": float(force_balance_error),
        "thresholds": {
            "max_tcp_position_delta_m": max_position_delta,
            "max_tcp_orientation_delta_deg": max_orientation_delta,
            "min_finger_force": min_force,
            "max_force_balance_error": max_force_balance_error,
            "live_lock_max_tcp_position_delta_m": live_lock_max_delta,
        },
    }


def _velocity_norm(actor: Any, name: str) -> float:
    return float(np.linalg.norm(np.asarray(_actor_velocity(actor, name), dtype=np.float32)))


def _validate_roles_after_settle(
    *,
    env: Any,
    base_env: Any,
    locked: dict[str, Any],
    targets: dict[str, Any],
    roles: list[str],
    segments: list[dict[str, Any]],
    args: argparse.Namespace,
    name: str,
    gripper: float,
    log: Any,
) -> dict[str, Any]:
    role_list = [role for role in roles if role in locked]
    baseline = {role: _actor_pose(locked[role].actor) for role in role_list}
    steps = int(getattr(args, "final_all_roles_stability_steps", 0))
    if steps > 0:
        _add_hold_segment(env, segments, name, gripper, steps)
    max_target_position = float(getattr(args, "all_roles_max_position_error", getattr(args, "max_position_error", 0.035)))
    max_target_orientation = float(
        getattr(args, "all_roles_max_orientation_error_deg", getattr(args, "max_orientation_error_deg", 35.0))
    )
    max_drift_position = float(getattr(args, "all_roles_max_drift_position", 0.008))
    max_drift_orientation = float(getattr(args, "all_roles_max_drift_orientation_deg", 5.0))
    max_linear_speed = float(getattr(args, "all_roles_max_linear_speed", 0.08))
    max_angular_speed = float(getattr(args, "all_roles_max_angular_speed", 1.0))
    min_connections = int(getattr(args, "min_active_connections", 1))
    role_reports: dict[str, Any] = {}
    success = True
    for role in role_list:
        actor = locked[role].actor
        current_pose = _actor_pose(actor)
        target_error = _pose_error(actor, targets[role]) if role in targets else None
        drift_error = _pose_delta_error(current_pose, baseline[role])
        linear_speed = _velocity_norm(actor, "get_linear_velocity")
        angular_speed = _velocity_norm(actor, "get_angular_velocity")
        active_connections = _active_connection_count_for_role(base_env, role)
        needs_connection = role != "floor"
        required_connections = max(min_connections, int(base_env.magnetic_snap.desired_active_connections_by_role.get(role, min_connections)))
        role_success = bool(
            (target_error is None or (
                target_error["position_error_m"] <= max_target_position
                and target_error["orientation_error_deg"] <= max_target_orientation
            ))
            and drift_error["position_error_m"] <= max_drift_position
            and drift_error["orientation_error_deg"] <= max_drift_orientation
            and linear_speed <= max_linear_speed
            and angular_speed <= max_angular_speed
            and (not needs_connection or active_connections >= required_connections)
            and role not in base_env.magnetic_snap.suspended_roles
        )
        role_reports[role] = {
            "success": role_success,
            "target_pose_error": target_error,
            "drift_after_settle": drift_error,
            "linear_speed": linear_speed,
            "angular_speed": angular_speed,
            "active_connection_count": active_connections,
            "suspended": role in base_env.magnetic_snap.suspended_roles,
            "thresholds": {
                "max_target_position": max_target_position,
                "max_target_orientation": max_target_orientation,
                "max_drift_position": max_drift_position,
                "max_drift_orientation": max_drift_orientation,
                "max_linear_speed": max_linear_speed,
                "max_angular_speed": max_angular_speed,
                "min_connections": required_connections if needs_connection else 0,
            },
        }
        success = success and role_success
    magnetic_snap_report = base_env.get_magnetic_snap_report()
    connection_error = dict(magnetic_snap_report.get("connection_error", {}))
    max_connection_point_error = float(connection_error.get("max_point_error", 0.0) or 0.0)
    max_connection_normal_error = float(connection_error.get("max_normal_angle_error_deg", 0.0) or 0.0)
    max_connection_edge_error = float(connection_error.get("max_edge_angle_error_deg", 0.0) or 0.0)
    max_allowed_connection_point_error = float(getattr(args, "final_max_connection_point_error", 0.0) or 0.0)
    max_allowed_connection_normal_error = float(getattr(args, "final_max_connection_normal_angle_deg", 0.0) or 0.0)
    max_allowed_connection_edge_error = float(getattr(args, "final_max_connection_edge_angle_deg", 0.0) or 0.0)
    connection_quality_success = True
    if max_allowed_connection_point_error > 0.0 and max_connection_point_error > max_allowed_connection_point_error:
        connection_quality_success = False
    if max_allowed_connection_normal_error > 0.0 and max_connection_normal_error > max_allowed_connection_normal_error:
        connection_quality_success = False
    if max_allowed_connection_edge_error > 0.0 and max_connection_edge_error > max_allowed_connection_edge_error:
        connection_quality_success = False
    success = success and connection_quality_success
    report = {
        "success": bool(success),
        "name": name,
        "steps": steps,
        "roles": role_list,
        "role_reports": role_reports,
        "connection_quality": {
            "success": bool(connection_quality_success),
            "max_point_error": max_connection_point_error,
            "max_normal_angle_error_deg": max_connection_normal_error,
            "max_edge_angle_error_deg": max_connection_edge_error,
            "thresholds": {
                "max_point_error": max_allowed_connection_point_error,
                "max_normal_angle_error_deg": max_allowed_connection_normal_error,
                "max_edge_angle_error_deg": max_allowed_connection_edge_error,
            },
        },
        "magnetic_snap_report": magnetic_snap_report,
    }
    log(f"{name}: all-role stability success={report['success']} roles={role_list}")
    return report


def _save_assembly_state(
    *,
    path: str,
    base_env: Any,
    locked: dict[str, Any],
    targets: dict[str, Any],
    reports: list[dict[str, Any]],
    roles: list[str],
    log: Any,
    loaded_completed_roles: list[str] | None = None,
    save_roles: list[str] | None = None,
    completed_roles_override: list[str] | None = None,
) -> None:
    if not path:
        return
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    save_role_set = set(save_roles or [])
    snap_report = base_env.get_magnetic_snap_report()
    if save_role_set:
        snap_report = dict(snap_report)
        snap_report["connections"] = [
            item
            for item in snap_report.get("connections", [])
            if item.get("parent") in save_role_set and item.get("child") in save_role_set
        ]
        connection_error = dict(snap_report.get("connection_error", {}))
        connection_error["connections"] = [
            item
            for item in connection_error.get("connections", [])
            if item.get("parent") in save_role_set and item.get("child") in save_role_set
        ]
        point_errors = [float(item.get("max_point_error", 0.0)) for item in connection_error["connections"]]
        mean_errors = [float(item.get("mean_point_error", 0.0)) for item in connection_error["connections"]]
        connection_error["max_point_error"] = max(point_errors) if point_errors else 0.0
        connection_error["mean_point_error"] = float(np.mean(mean_errors)) if mean_errors else 0.0
        snap_report["connection_error"] = connection_error
        keys = {
            tuple(
                sorted(
                    [
                        (item.get("parent"), item.get("parent_edge"), item.get("parent_lane", "rim")),
                        (item.get("child"), item.get("child_edge"), item.get("child_lane", "rim")),
                    ]
                )
            )
            for item in snap_report["connections"]
        }
        snap_report["active_connection_count"] = len(keys)
    role_states = {}
    for role, locked_item in locked.items():
        if save_role_set and role not in save_role_set:
            continue
        actor = locked_item.actor
        role_states[role] = {
            "pose": _pose_to_report(_actor_pose(actor)),
            "linear_velocity": _actor_velocity(actor, "get_linear_velocity"),
            "angular_velocity": _actor_velocity(actor, "get_angular_velocity"),
            "target_pose_error": _pose_error(actor, targets[role]) if role in targets else None,
        }
    completed_roles = list(
        dict.fromkeys(
            completed_roles_override
            if completed_roles_override is not None
            else [
                *(loaded_completed_roles or []),
                *[item["role"] for item in reports if item.get("success")],
            ]
        )
    )
    payload = {
        "schema": "jimu_assembly_state_v1",
        "roles": list(save_roles or roles),
        "completed_roles": completed_roles,
        "robot_qpos": base_env.agent.robot.get_qpos().detach().cpu().numpy()[0].astype(np.float32).tolist(),
        "role_states": role_states,
        "magnetic_snap_report": snap_report,
    }
    if save_role_set:
        payload["completed_roles"] = [
            role
            for role in payload["completed_roles"]
            if role in save_role_set and role != "floor"
        ]
    state_path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"saved assembly state: {state_path}")


def _load_assembly_state(
    *,
    path: str,
    base_env: Any,
    locked: dict[str, Any],
    targets: dict[str, Any] | None = None,
    args: argparse.Namespace | None = None,
    log: Any,
    restore_robot_qpos: bool = False,
) -> list[str]:
    if not path:
        return []
    state_path = Path(path)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    role_states = payload.get("role_states", {})
    restored_roles: list[str] = []
    for role, role_state in role_states.items():
        if role not in locked:
            continue
        pose_data = role_state.get("pose", {})
        position = pose_data.get("position")
        quaternion = pose_data.get("quaternion")
        if position is None or quaternion is None:
            continue
        actor = locked[role].actor
        pose = sapien.Pose(p=position, q=quaternion)
        actor.set_pose(pose)
        actor.set_linear_velocity(np.asarray(role_state.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float32))
        actor.set_angular_velocity(np.asarray(role_state.get("angular_velocity", [0.0, 0.0, 0.0]), dtype=np.float32))
        _sync_locked_panel_pose(base_env, role, pose)
        restored_roles.append(role)
    setattr(base_env, "_loaded_assembly_restored_roles", list(restored_roles))
    loaded_freeze_report = _set_loaded_role_pose_freeze(base_env=base_env, locked=locked, roles=restored_roles)
    target_alignment_report = _align_targets_to_loaded_floor(
        locked=locked,
        targets=targets,
        restored_roles=restored_roles,
        args=args,
        log=log,
    )
    base_env.magnetic_snap.suspended_roles.clear()
    if restore_robot_qpos and payload.get("robot_qpos") is not None:
        try:
            qpos = np.asarray(payload["robot_qpos"], dtype=np.float32).reshape(1, -1)
            base_env.agent.robot.set_qpos(torch.as_tensor(qpos, dtype=torch.float32, device=base_env.device))
            if hasattr(base_env.agent.robot, "set_qvel") and hasattr(base_env.agent.robot, "get_qvel"):
                base_env.agent.robot.set_qvel(torch.zeros_like(base_env.agent.robot.get_qvel()))
            log(f"restored robot qpos from assembly state: dof={qpos.shape[1]}")
        except Exception as exc:
            log(f"failed to restore robot qpos from assembly state: {type(exc).__name__}: {exc}")
    completed_roles = [
        role
        for role in payload.get("completed_roles", [])
        if role in locked
    ]
    log(
        f"loaded assembly state: {state_path} restored_roles={restored_roles} "
        f"completed_roles={completed_roles} loaded_freeze={loaded_freeze_report} "
        f"target_alignment={target_alignment_report}"
    )
    return completed_roles


def _align_targets_to_loaded_floor(
    *,
    locked: dict[str, Any],
    targets: dict[str, Any] | None,
    restored_roles: list[str],
    args: argparse.Namespace | None,
    log: Any,
) -> dict[str, Any]:
    if targets is None or "floor" not in targets or "floor" not in locked or "floor" not in restored_roles:
        return {"enabled": False, "reason": "missing_loaded_floor_or_targets"}
    if not bool(getattr(args, "align_targets_to_loaded_floor", True)):
        return {"enabled": False, "reason": "disabled"}
    mode = str(getattr(args, "align_targets_to_loaded_floor_mode", "xy") or "xy").strip().lower()
    target_role_text = str(
        getattr(args, "align_targets_to_loaded_floor_roles", "floor,right_wall,back_wall,left_wall,front_wall,top_lid")
        or ""
    )
    target_roles = _role_list(target_role_text)
    loaded_floor_position, _ = _pose_arrays(_actor_pose(locked["floor"].actor))
    target_floor_position, _ = _pose_arrays(targets["floor"])
    offset = loaded_floor_position - target_floor_position
    if mode == "xy":
        offset[2] = 0.0
    elif mode == "xyz":
        pass
    else:
        mode = "xy"
        offset[2] = 0.0
    shifted_roles: list[str] = []
    for role in target_roles:
        if role not in targets:
            continue
        position, quaternion = _pose_arrays(targets[role])
        targets[role] = sapien.Pose(p=(position + offset).tolist(), q=quaternion.tolist())
        shifted_roles.append(role)
    report = {
        "enabled": True,
        "mode": mode,
        "offset": offset.astype(float).tolist(),
        "roles": shifted_roles,
        "loaded_floor_position": loaded_floor_position.astype(float).tolist(),
        "previous_target_floor_position": target_floor_position.astype(float).tolist(),
    }
    log(f"aligned assembly targets to loaded floor: {report}")
    return report


def _restore_loaded_magnetic_connections(
    *,
    path: str,
    base_env: Any,
    locked: dict[str, Any],
    log: Any,
) -> list[dict[str, Any]]:
    if not path:
        return []
    state_path = Path(path)
    if not state_path.is_file():
        return []
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"failed to read magnetic connections from assembly state: {type(exc).__name__}: {exc}")
        return []
    snap_report = payload.get("magnetic_snap_report") if isinstance(payload, dict) else None
    connection_items = snap_report.get("connections", []) if isinstance(snap_report, dict) else []
    if not isinstance(connection_items, list):
        return []
    snap = base_env.magnetic_snap
    scene = getattr(snap, "scene", None)
    if scene is None:
        return []
    locked_by_role = {role: item for role, item in locked.items()}
    restored: list[dict[str, Any]] = []
    for item in connection_items:
        if not isinstance(item, dict):
            continue
        parent_role = str(item.get("parent", ""))
        child_role = str(item.get("child", ""))
        parent = locked_by_role.get(parent_role)
        child = locked_by_role.get(child_role)
        if parent is None or child is None:
            continue
        connection = MagneticConnection(
            parent=parent_role,
            parent_edge=str(item.get("parent_edge", "")),
            child=child_role,
            child_edge=str(item.get("child_edge", "")),
            mode=str(item.get("mode", "")),
            parent_lane=str(item.get("parent_lane", "rim")),
            child_lane=str(item.get("child_lane", "rim")),
        )
        if snap._find_active_connection(connection) is not None:
            continue
        try:
            snap._create_runtime_edge_connection(scene, parent, child, connection)
            restored.append(
                {
                    "parent": connection.parent,
                    "parent_edge": connection.parent_edge,
                    "child": connection.child,
                    "child_edge": connection.child_edge,
                    "mode": connection.mode,
                    "parent_lane": connection.parent_lane,
                    "child_lane": connection.child_lane,
                    "point_error_m": float(snap._connection_point_error(connection)),
                }
            )
        except Exception as exc:
            restored.append(
                {
                    "parent": connection.parent,
                    "parent_edge": connection.parent_edge,
                    "child": connection.child,
                    "child_edge": connection.child_edge,
                    "mode": connection.mode,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if restored:
        log(f"restored loaded magnetic connections: {restored}")
    return restored


def _configure_full_base_connection_targets(base_env: Any, roles: list[str], log: Any) -> dict[str, int]:
    base_roles = {"floor", "right_wall", "back_wall", "left_wall", "front_wall"}
    active_roles = [role for role in roles if role in base_roles]
    if not active_roles:
        return {}
    desired = {
        "floor": 4,
        "right_wall": 3,
        "back_wall": 3,
        "left_wall": 3,
        "front_wall": 3,
    }
    configured = {
        role: desired[role]
        for role in active_roles
        if role in desired
    }
    base_env.magnetic_snap.desired_active_connections_by_role.update(configured)
    log(f"base connection targets: {configured}")
    return configured


def _role_list(text: str) -> list[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def _role_in_csv(role: str, text: str) -> bool:
    roles = set(_role_list(text))
    return "all" in roles or role in roles


def _apply_loaded_state_perturbation(
    *,
    base_env: Any,
    locked: dict[str, Any],
    targets: dict[str, Any],
    args: argparse.Namespace,
    log: Any,
) -> dict[str, Any]:
    dx = float(getattr(args, "loaded_state_perturb_dx", 0.0))
    dy = float(getattr(args, "loaded_state_perturb_dy", 0.0))
    dz = float(getattr(args, "loaded_state_perturb_dz", 0.0))
    yaw_deg = float(getattr(args, "loaded_state_perturb_yaw_deg", 0.0))
    if abs(dx) <= 1e-9 and abs(dy) <= 1e-9 and abs(dz) <= 1e-9 and abs(yaw_deg) <= 1e-9:
        return {}

    perturb_roles = _role_list(getattr(args, "loaded_state_perturb_roles", "floor,right_wall,back_wall,left_wall"))
    target_roles = _role_list(
        getattr(args, "loaded_state_perturb_target_roles", "floor,right_wall,back_wall,left_wall,front_wall")
    )
    origin_role = str(getattr(args, "loaded_state_perturb_origin_role", "floor") or "floor")
    if origin_role in locked:
        origin = np.asarray(_actor_pose(locked[origin_role].actor).p, dtype=np.float32).reshape(3)
    else:
        origin = np.zeros(3, dtype=np.float32)
    yaw = np.deg2rad(yaw_deg)
    rot = np.asarray(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    offset = np.asarray([dx, dy, dz], dtype=np.float32)

    def transform_pose(pose: sapien.Pose) -> sapien.Pose:
        position, quaternion = _pose_arrays(pose)
        new_position = origin + rot @ (position - origin) + offset
        new_quaternion = mat2quat(rot @ quat2mat(quaternion)).astype(np.float32)
        return sapien.Pose(p=new_position.tolist(), q=new_quaternion.tolist())

    actor_errors: dict[str, Any] = {}
    for role in perturb_roles:
        if role not in locked:
            continue
        actor = locked[role].actor
        actor.set_pose(transform_pose(_actor_pose(actor)))
        actor.set_linear_velocity(np.zeros(3, dtype=np.float32))
        actor.set_angular_velocity(np.zeros(3, dtype=np.float32))
    for role in target_roles:
        if role in targets:
            targets[role] = transform_pose(targets[role])
    base_env.magnetic_snap.suspended_roles.clear()
    for role in perturb_roles:
        if role in locked and role in targets:
            actor_errors[role] = _pose_error(locked[role].actor, targets[role])
    report = {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "yaw_deg": yaw_deg,
        "origin_role": origin_role,
        "origin": origin.tolist(),
        "perturb_roles": perturb_roles,
        "target_roles": target_roles,
        "actor_target_errors_after_perturb": actor_errors,
    }
    log(f"applied loaded-state perturbation: {report}")
    return report


def _apply_initial_assembly_offset(
    *,
    locked: dict[str, Any],
    targets: dict[str, Any],
    args: argparse.Namespace,
    log: Any,
) -> dict[str, Any]:
    dx = float(getattr(args, "initial_assembly_offset_x", 0.0))
    dy = float(getattr(args, "initial_assembly_offset_y", 0.0))
    dz = float(getattr(args, "initial_assembly_offset_z", 0.0))
    if abs(dx) <= 1e-9 and abs(dy) <= 1e-9 and abs(dz) <= 1e-9:
        return {}
    actor_roles = _role_list(getattr(args, "initial_assembly_offset_actor_roles", "floor"))
    target_roles = _role_list(
        getattr(args, "initial_assembly_offset_target_roles", "floor,right_wall,back_wall,left_wall,front_wall")
    )
    offset = np.asarray([dx, dy, dz], dtype=np.float32)
    actor_errors: dict[str, Any] = {}
    for role in target_roles:
        if role in targets:
            position, quaternion = _pose_arrays(targets[role])
            targets[role] = sapien.Pose(p=(position + offset).tolist(), q=quaternion.tolist())
    for role in actor_roles:
        if role in locked:
            actor = locked[role].actor
            position, quaternion = _pose_arrays(_actor_pose(actor))
            actor.set_pose(sapien.Pose(p=(position + offset).tolist(), q=quaternion.tolist()))
            actor.set_linear_velocity(np.zeros(3, dtype=np.float32))
            actor.set_angular_velocity(np.zeros(3, dtype=np.float32))
    for role in sorted(set(actor_roles) | set(target_roles)):
        if role in locked and role in targets:
            actor_errors[role] = _pose_error(locked[role].actor, targets[role])
    report = {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "actor_roles": actor_roles,
        "target_roles": target_roles,
        "actor_target_errors_after_offset": actor_errors,
    }
    log(f"applied initial assembly offset: {report}")
    return report


def _apply_initial_actor_jitter(
    *,
    locked: dict[str, Any],
    targets: dict[str, Any],
    fixtures: list[dict[str, Any]],
    args: argparse.Namespace,
    log: Any,
) -> dict[str, Any]:
    radius = float(getattr(args, "initial_actor_jitter_xy", 0.0) or 0.0)
    if radius <= 0.0:
        return {}
    roles = _role_list(getattr(args, "initial_actor_jitter_roles", "right_wall,back_wall,left_wall,front_wall"))
    if not roles:
        return {}
    seed = int(getattr(args, "initial_actor_jitter_seed", 0) or 0)
    max_attempts = max(int(getattr(args, "initial_actor_jitter_max_sample_attempts", 100) or 100), 1)
    min_start_distance = max(float(getattr(args, "initial_actor_jitter_min_start_distance", 0.0) or 0.0), 0.0)
    min_target_distance = max(float(getattr(args, "initial_actor_jitter_min_target_distance", 0.0) or 0.0), 0.0)
    rng = np.random.default_rng(seed)
    baseline_positions = {
        role: np.asarray(_actor_pose(locked[role].actor).p, dtype=np.float32).reshape(3)
        for role in roles
        if role in locked
    }
    sampled_offsets: dict[str, np.ndarray] = {}

    def sample_offset() -> np.ndarray:
        for _ in range(max_attempts):
            offset_xy = rng.uniform(-radius, radius, size=2).astype(np.float32)
            if float(np.linalg.norm(offset_xy)) <= radius:
                return np.asarray([offset_xy[0], offset_xy[1], 0.0], dtype=np.float32)
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        distance = float(rng.uniform(0.0, radius))
        return np.asarray([np.cos(angle) * distance, np.sin(angle) * distance, 0.0], dtype=np.float32)

    def is_far_enough(role: str, candidate_position: np.ndarray) -> bool:
        if min_start_distance > 0.0:
            for other_role, other_offset in sampled_offsets.items():
                other_position = baseline_positions[other_role] + other_offset
                if float(np.linalg.norm(candidate_position[:2] - other_position[:2])) < min_start_distance:
                    return False
        if min_target_distance > 0.0:
            for target_role, target_pose in targets.items():
                if target_role == role:
                    continue
                target_position = np.asarray(target_pose.p, dtype=np.float32).reshape(3)
                if float(np.linalg.norm(candidate_position[:2] - target_position[:2])) < min_target_distance:
                    return False
        return True

    for role in roles:
        if role not in baseline_positions:
            continue
        accepted_offset: np.ndarray | None = None
        for _ in range(max_attempts):
            offset = sample_offset()
            if is_far_enough(role, baseline_positions[role] + offset):
                accepted_offset = offset
                break
        if accepted_offset is None:
            accepted_offset = np.zeros(3, dtype=np.float32)
        sampled_offsets[role] = accepted_offset

    actor_errors: dict[str, Any] = {}
    report_roles: dict[str, Any] = {}
    moved_fixtures: dict[str, list[str]] = {}
    for role, offset in sampled_offsets.items():
        actor = locked[role].actor
        position, quaternion = _pose_arrays(_actor_pose(actor))
        new_position = position + offset
        actor.set_pose(sapien.Pose(p=new_position.tolist(), q=quaternion.tolist()))
        actor.set_linear_velocity(np.zeros(3, dtype=np.float32))
        actor.set_angular_velocity(np.zeros(3, dtype=np.float32))
        fixture_prefix = f"fixture_{role}_"
        for fixture in fixtures:
            name = str(fixture.get("name", ""))
            if not name.startswith(fixture_prefix):
                continue
            fixture_actor = fixture.get("actor")
            if fixture_actor is None:
                continue
            fixture_position, fixture_quaternion = _pose_arrays(_actor_pose(fixture_actor))
            fixture_new_position = fixture_position + offset
            fixture_actor.set_pose(sapien.Pose(p=fixture_new_position.tolist(), q=fixture_quaternion.tolist()))
            pose = list(fixture.get("pose", []))
            if len(pose) >= 7:
                pose[0] = float(fixture_new_position[0])
                pose[1] = float(fixture_new_position[1])
                pose[2] = float(fixture_new_position[2])
                fixture["pose"] = pose
            moved_fixtures.setdefault(role, []).append(name)
        if role in targets:
            actor_errors[role] = _pose_error(actor, targets[role])
        report_roles[role] = {
            "dx": float(offset[0]),
            "dy": float(offset[1]),
            "dz": float(offset[2]),
            "start_position": new_position.astype(np.float32).tolist(),
        }
    report = {
        "seed": seed,
        "radius_xy": radius,
        "roles": roles,
        "min_start_distance": min_start_distance,
        "min_target_distance": min_target_distance,
        "offsets": report_roles,
        "moved_fixtures": moved_fixtures,
        "actor_target_errors_after_jitter": actor_errors,
    }
    log(f"applied initial actor jitter: {report}")
    return report


def _tcp_retreat_pose(base_env: Any, distance: float, direction_sign: float) -> sapien.Pose:
    tcp_pose = _tcp_pose(base_env)
    position, quaternion = _pose_arrays(tcp_pose)
    rotation = quat2mat(quaternion).astype(np.float32)
    retreat = rotation[:, 2].astype(np.float32) * float(direction_sign) * float(distance)
    return sapien.Pose(p=(position + retreat).tolist(), q=quaternion.tolist())


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        raise ValueError("cannot normalize zero vector")
    return (vec / norm).astype(np.float32)


def _pose_from_axes(position: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> sapien.Pose:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.stack(
        [_normalize_vector(x_axis), _normalize_vector(y_axis), _normalize_vector(z_axis)],
        axis=1,
    )
    matrix[:3, 3] = np.asarray(position, dtype=np.float32).reshape(3)
    return sapien.Pose(matrix)


def _robot_pose_arrays(base_env: Any) -> tuple[np.ndarray, np.ndarray]:
    pose = base_env.agent.robot.pose.sp
    position = np.asarray(pose.p, dtype=np.float32).reshape(3)
    quaternion = np.asarray(pose.q, dtype=np.float32).reshape(4)
    quaternion = quaternion / max(float(np.linalg.norm(quaternion)), 1e-8)
    return position, quaternion


def _add_rear_collision_wall_fixture(
    *,
    base_env: Any,
    targets: dict[str, sapien.Pose],
    fixtures: list[dict[str, Any]],
    args: argparse.Namespace,
    log: Any,
) -> dict[str, Any] | None:
    if not bool(getattr(args, "add_rear_collision_wall", False)):
        return None
    width = float(getattr(args, "rear_collision_wall_width", 1.80))
    thickness = float(getattr(args, "rear_collision_wall_thickness", 0.06))
    height = float(getattr(args, "rear_collision_wall_height", 1.20))
    distance = float(getattr(args, "rear_collision_wall_distance", 0.35))
    placement_frame = str(getattr(args, "rear_collision_wall_frame", "world_back_x") or "world_back_x").strip()
    x_offset = float(getattr(args, "rear_collision_wall_x_offset", 0.0))
    y_offset = float(getattr(args, "rear_collision_wall_y_offset", 0.0))
    z_bottom = float(getattr(args, "rear_collision_wall_z_bottom", 0.0))
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    if placement_frame == "world_back_x":
        robot_position, _ = _robot_pose_arrays(base_env)
        x_sign = float(getattr(args, "rear_collision_wall_robot_back_sign", -1.0))
        center = robot_position + np.asarray([x_sign * distance + x_offset, y_offset, 0.0], dtype=np.float32)
        center[2] = z_bottom + height * 0.5
        dims = np.asarray([thickness, width, height], dtype=np.float32)
        wall_pose = sapien.Pose(p=center.tolist(), q=[1.0, 0.0, 0.0, 0.0])
    elif placement_frame == "floor_back_y":
        floor_pose = targets.get("floor")
        if floor_pose is None:
            return None
        floor_position = np.asarray(floor_pose.p, dtype=np.float32).reshape(3)
        center = np.asarray(
            [
                floor_position[0] + x_offset,
                floor_position[1] + float(getattr(args, "rear_collision_wall_floor_y_offset", distance)),
                z_bottom + height * 0.5,
            ],
            dtype=np.float32,
        )
        dims = np.asarray([width, thickness, height], dtype=np.float32)
        wall_pose = sapien.Pose(p=center.tolist(), q=[1.0, 0.0, 0.0, 0.0])
    else:
        robot_position, robot_quaternion = _robot_pose_arrays(base_env)
        robot_rotation = quat2mat(robot_quaternion).astype(np.float32)
        back_axis = robot_rotation @ np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        back_axis[2] = 0.0
        if float(np.linalg.norm(back_axis)) <= 1e-6:
            back_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        back_axis = _normalize_vector(back_axis) * float(getattr(args, "rear_collision_wall_robot_back_sign", -1.0))
        lateral_axis = np.cross(up, back_axis)
        if float(np.linalg.norm(lateral_axis)) <= 1e-6:
            lateral_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        lateral_axis = _normalize_vector(lateral_axis)
        center = robot_position + back_axis * distance + np.asarray([x_offset, y_offset, 0.0], dtype=np.float32)
        center[2] = z_bottom + height * 0.5
        dims = np.asarray([thickness, width, height], dtype=np.float32)
        wall_pose = _pose_from_axes(center, back_axis, lateral_axis, up)
    _, quaternion = _pose_arrays(wall_pose)
    material = sapien.pysapien.physx.PhysxMaterial(static_friction=1.4, dynamic_friction=1.0, restitution=0.0)
    visual = sapien.render.RenderMaterial(base_color=[0.18, 0.22, 0.28, 0.72])
    builder = base_env.scene.create_actor_builder()
    builder.set_scene_idxs([0])
    builder.initial_pose = wall_pose
    builder.add_box_collision(half_size=(dims * 0.5).tolist(), material=material)
    builder.add_box_visual(half_size=(dims * 0.5).tolist(), material=visual)
    actor = builder.build_kinematic(name="rear_collision_wall_robot_back")
    base_env.remove_from_state_dict_registry(actor)
    fixture = {
        "name": "rear_collision_wall_robot_back",
        "actor": actor,
        "dims": dims.tolist(),
        "pose": [
            float(center[0]),
            float(center[1]),
            float(center[2]),
            float(quaternion[0]),
            float(quaternion[1]),
            float(quaternion[2]),
            float(quaternion[3]),
        ],
        "placement_frame": placement_frame,
        "distance": distance,
    }
    fixtures.append(fixture)
    report = {k: v for k, v in fixture.items() if k != "actor"}
    log(f"added rear collision wall fixture: {report}")
    return report


def _add_overhead_collision_wall_fixture(
    *,
    base_env: Any,
    fixtures: list[dict[str, Any]],
    args: argparse.Namespace,
    log: Any,
) -> dict[str, Any] | None:
    if not bool(getattr(args, "add_overhead_collision_wall", False)):
        return None
    size_x = float(getattr(args, "overhead_collision_wall_size_x", 1.20))
    size_y = float(getattr(args, "overhead_collision_wall_size_y", 1.20))
    thickness = float(getattr(args, "overhead_collision_wall_thickness", 0.045))
    z = float(getattr(args, "overhead_collision_wall_z", 0.82))
    x_offset = float(getattr(args, "overhead_collision_wall_x_offset", 0.0))
    y_offset = float(getattr(args, "overhead_collision_wall_y_offset", 0.0))
    frame = str(getattr(args, "overhead_collision_wall_frame", "robot_base") or "robot_base").strip()
    robot_position, robot_quaternion = _robot_pose_arrays(base_env)
    robot_rotation = quat2mat(robot_quaternion).astype(np.float32)
    if frame == "world":
        center = np.asarray([x_offset, y_offset, z], dtype=np.float32)
        wall_pose = sapien.Pose(p=center.tolist(), q=[1.0, 0.0, 0.0, 0.0])
    else:
        local = np.asarray([x_offset, y_offset, 0.0], dtype=np.float32)
        center = robot_position + robot_rotation @ local
        center[2] = z
        wall_pose = sapien.Pose(p=center.tolist(), q=robot_quaternion.tolist())
    dims = np.asarray([size_x, size_y, thickness], dtype=np.float32)
    _, quaternion = _pose_arrays(wall_pose)
    material = sapien.pysapien.physx.PhysxMaterial(static_friction=1.4, dynamic_friction=1.0, restitution=0.0)
    visual = sapien.render.RenderMaterial(base_color=[0.16, 0.18, 0.22, 0.62])
    builder = base_env.scene.create_actor_builder()
    builder.set_scene_idxs([0])
    builder.initial_pose = wall_pose
    builder.add_box_collision(half_size=(dims * 0.5).tolist(), material=material)
    builder.add_box_visual(half_size=(dims * 0.5).tolist(), material=visual)
    actor = builder.build_kinematic(name="overhead_collision_wall")
    base_env.remove_from_state_dict_registry(actor)
    fixture = {
        "name": "overhead_collision_wall",
        "actor": actor,
        "dims": dims.tolist(),
        "pose": [
            float(center[0]),
            float(center[1]),
            float(center[2]),
            float(quaternion[0]),
            float(quaternion[1]),
            float(quaternion[2]),
            float(quaternion[3]),
        ],
        "placement_frame": frame,
    }
    fixtures.append(fixture)
    report = {k: v for k, v in fixture.items() if k != "actor"}
    log(f"added overhead collision wall fixture: {report}")
    return report


def _centered_grasp_candidates(max_candidates: int, args: argparse.Namespace) -> list[GraspCandidate]:
    if not bool(getattr(args, "prefer_center_wall_grasp", True)):
        return _make_grasp_candidates(max_candidates)
    requested = max(int(max_candidates), 1)
    center_limit = min(requested, max(int(getattr(args, "max_center_wall_grasp_candidates", 8)), 1))
    center_offset = float(getattr(args, "wall_grasp_center_offset", 0.006))
    center_offsets = [
        (0.0, 0.0, 0.0),
        (-center_offset, 0.0, 0.0),
        (center_offset, 0.0, 0.0),
        (0.0, -center_offset, 0.0),
        (0.0, center_offset, 0.0),
        (-center_offset, -center_offset, 0.0),
        (-center_offset, center_offset, 0.0),
        (center_offset, -center_offset, 0.0),
        (center_offset, center_offset, 0.0),
    ]
    if bool(getattr(args, "enable_center_grasp_yaw_candidates", False)):
        center_offsets = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 4.0),
            (0.0, 0.0, -4.0),
            *center_offsets[1:],
        ]
    center_candidates: list[GraspCandidate] = []
    for template in _make_grasp_candidates(center_limit):
        for local_x, local_y, yaw_deg in center_offsets:
            label = (
                f"center_{template.label}_"
                f"cx_{local_x:+.3f}_cy_{local_y:+.3f}_cyaw_{yaw_deg:+.1f}"
            )
            center_candidates.append(
                GraspCandidate(
                    label=label,
                    local_x=float(local_x),
                    local_y=float(local_y),
                    thin_bias=float(template.thin_bias),
                    approach_bias=float(template.approach_bias),
                    yaw_deg=float(yaw_deg),
                    pregrasp_distance=float(template.pregrasp_distance),
                    approach=str(template.approach),
                    approach_tilt_deg=float(template.approach_tilt_deg),
                )
            )
            if len(center_candidates) >= center_limit:
                break
        if len(center_candidates) >= center_limit:
            break
    edge_candidates = _make_grasp_candidates(max(requested - len(center_candidates), 0))
    return [*center_candidates, *edge_candidates][:requested]


def _center_distance(candidate: Any) -> float:
    return float(np.hypot(float(candidate.local_x), float(candidate.local_y)))


def _int_csv(text: str) -> list[int]:
    values: list[int] = []
    for item in str(text or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError:
            continue
    return values


def _release_candidate_index_filter(args: argparse.Namespace, role: str) -> list[int]:
    mapping_text = str(getattr(args, "release_candidate_indices", "") or "")
    if not mapping_text.strip():
        return []
    role_values: list[int] = []
    global_values: list[int] = []
    for item in mapping_text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            global_values.extend(_int_csv(item))
            continue
        key, value = item.split(":", 1)
        if key.strip() == role:
            role_values.extend(_int_csv(value))
    values = role_values or global_values
    unique: list[int] = []
    for value in values:
        if value >= 0 and value not in unique:
            unique.append(value)
    return unique


def _rank_near(value: float, preferred: list[float]) -> tuple[float, int]:
    if not preferred:
        return (abs(float(value)), 0)
    distances = [abs(float(value) - float(item)) for item in preferred]
    index = int(np.argmin(np.asarray(distances, dtype=np.float32)))
    return (float(distances[index]), index)


def _wall_grasp_prior_key(candidate: Any, role: str = "") -> tuple[Any, ...]:
    label = str(getattr(candidate, "label", ""))
    local_x = float(getattr(candidate, "local_x", 0.0))
    local_y = float(getattr(candidate, "local_y", 0.0))
    pregrasp = float(getattr(candidate, "pregrasp_distance", 0.0))
    tilt = float(getattr(candidate, "approach_tilt_deg", 0.0))
    thin = float(getattr(candidate, "thin_bias", 0.0))
    yaw = float(getattr(candidate, "yaw_deg", 0.0))
    approach_bias = float(getattr(candidate, "approach_bias", 0.0))
    is_center = 0 if label.startswith("center_") else 1
    center_targets = [
        (0.004, -0.004),
        (-0.004, -0.004),
        (0.0, 0.0),
        (0.0, 0.004),
        (0.0, -0.004),
    ]
    plain_x_targets = [0.025, -0.025, 0.019, -0.019]
    center_distance_rank = min(
        abs(local_x - target_x) + abs(local_y - target_y)
        for target_x, target_y in center_targets
    )
    plain_x_rank = min(abs(local_x - target_x) + 0.25 * abs(local_y) for target_x in plain_x_targets)
    geometry_rank = center_distance_rank if is_center == 0 else plain_x_rank
    approach_rank = _rank_near(approach_bias, [-0.024, -0.018, -0.012])
    preferred_sign = 1.0 if role == "front_wall" else -1.0
    sign_rank = 0 if local_x * preferred_sign > 0.0 else 1
    return (
        abs(pregrasp - 0.100),
        abs(tilt),
        abs(thin),
        abs(yaw),
        sign_rank,
        min(center_distance_rank, plain_x_rank),
        is_center,
        geometry_rank,
        approach_rank[0],
        approach_rank[1],
        _center_distance(candidate),
        label,
    )


def _apply_wall_grasp_prior(args: argparse.Namespace, role: str, candidates: list[Any]) -> list[Any]:
    if role not in FIRST_LAYER_WALL_ROLES:
        return candidates
    mode = str(getattr(args, "wall_grasp_prior_mode", "none") or "none").strip().lower()
    if mode in {"", "none", "off"}:
        return candidates
    if mode != "mined_success_v1":
        return candidates
    ranked = sorted(candidates, key=lambda candidate: _wall_grasp_prior_key(candidate, role))
    limit = int(getattr(args, "wall_grasp_prior_max_candidates", 0) or 0)
    if limit > 0:
        return ranked[:limit]
    return ranked


def _extra_tilt_values_for_wall_grasp(args: argparse.Namespace) -> list[float]:
    text = str(getattr(args, "wall_grasp_extra_tilt_degs", "") or "").strip()
    if not text:
        return []
    max_abs = abs(float(getattr(args, "wall_grasp_extra_tilt_max_abs_deg", 30.0) or 30.0))
    values: list[float] = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError:
            continue
        if abs(value) <= 1e-6:
            continue
        value = max(-max_abs, min(max_abs, value))
        if value not in values:
            values.append(value)
    return values


def _append_extra_tilt_wall_grasp_candidates(args: argparse.Namespace, candidates: list[Any]) -> list[Any]:
    tilt_values = _extra_tilt_values_for_wall_grasp(args)
    if not tilt_values:
        return candidates
    expanded: list[Any] = []
    seen = {
        (
            round(float(getattr(item, "local_x", 0.0)), 6),
            round(float(getattr(item, "local_y", 0.0)), 6),
            round(float(getattr(item, "thin_bias", 0.0)), 6),
            round(float(getattr(item, "approach_bias", 0.0)), 6),
            round(float(getattr(item, "yaw_deg", 0.0)), 6),
            round(float(getattr(item, "pregrasp_distance", 0.0)), 6),
            str(getattr(item, "approach", "")),
            round(float(getattr(item, "approach_tilt_deg", 0.0)), 6),
        )
        for item in candidates
    }
    for template in candidates:
        expanded.append(template)
        for tilt_deg in tilt_values:
            key = (
                round(float(template.local_x), 6),
                round(float(template.local_y), 6),
                round(float(template.thin_bias), 6),
                round(float(template.approach_bias), 6),
                round(float(template.yaw_deg), 6),
                round(float(template.pregrasp_distance), 6),
                str(template.approach),
                round(float(tilt_deg), 6),
            )
            if key in seen:
                continue
            seen.add(key)
            expanded.append(
                GraspCandidate(
                    label=f"{template.label}_extra_tilt_{tilt_deg:+.0f}",
                    local_x=float(template.local_x),
                    local_y=float(template.local_y),
                    thin_bias=float(template.thin_bias),
                    approach_bias=float(template.approach_bias),
                    yaw_deg=float(template.yaw_deg),
                    pregrasp_distance=float(template.pregrasp_distance),
                    approach=str(template.approach),
                    approach_tilt_deg=float(tilt_deg),
                )
            )
    return expanded


def _grasp_candidates_for_role(args: argparse.Namespace, role: str) -> list[Any]:
    start_index = max(int(getattr(args, "grasp_candidate_start_index", 0)), 0)
    mapping_text = str(getattr(args, "grasp_candidate_start_indices", "") or "")
    for item in mapping_text.split(","):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        if key.strip() == role:
            start_index = max(int(value.strip()), 0)
            break
    def axis_filtered(candidates: list[Any]) -> list[Any]:
        axis = str(getattr(args, "square_wall_grasp_edge_axis", "any") or "any").strip().lower()
        if role == "top_lid" or axis in {"", "any"}:
            return candidates
        center_window = float(getattr(args, "wall_grasp_center_axis_keep_radius", 0.010))
        centered = [item for item in candidates if _center_distance(item) <= center_window]
        if axis == "x":
            filtered = [item for item in candidates if abs(float(item.local_x)) > abs(float(item.local_y))]
        elif axis == "y":
            filtered = [item for item in candidates if abs(float(item.local_y)) > abs(float(item.local_x))]
        else:
            filtered = candidates
        if centered:
            filtered = [*centered, *[item for item in filtered if item not in centered]]
        return filtered or candidates
    if role != "top_lid":
        if role != "front_wall" or not bool(getattr(args, "diversify_wall_grasp_candidates", True)):
            candidates = _centered_grasp_candidates(args.max_grasp_candidates + start_index, args)
            filtered = _apply_wall_grasp_prior(args, role, axis_filtered(candidates))
            filtered = _append_extra_tilt_wall_grasp_candidates(args, filtered)
            return filtered[start_index : start_index + max(int(args.max_grasp_candidates), 1)]
        pool_size = min(
            max(int(getattr(args, "wall_grasp_diversify_pool_size", 512)), int(args.max_grasp_candidates)),
            18000,
        )
        pool = _centered_grasp_candidates(pool_size, args)
        pregrasp_rank = {0.080: 0, 0.060: 1, 0.100: 2, 0.120: 3}
        thin_rank = {0.0: 0, 0.0035: 1, -0.0035: 2, 0.007: 3, -0.007: 4}
        approach_rank = {-0.018: 0, -0.012: 1, -0.024: 2, -0.006: 3, 0.0: 4}
        tilt_rank = {0.0: 0, -8.0: 1, 8.0: 2, -14.0: 3, 14.0: 4}
        yaw_rank = {0.0: 0, -4.0: 1, 4.0: 2}

        def edge_rank(candidate: Any) -> int:
            distance_to_edge = PLATE_SIZE / 2.0 - max(abs(float(candidate.local_x)), abs(float(candidate.local_y)))
            preferred = [0.012, 0.018, 0.006]
            return min(range(len(preferred)), key=lambda index: abs(distance_to_edge - preferred[index]))

        def rank_value(mapping: dict[float, int], value: float) -> int:
            rounded = round(float(value), 4)
            return mapping.get(rounded, len(mapping) + 1)

        def candidate_key(candidate: Any) -> tuple[Any, ...]:
            core_ranks = [
                rank_value(pregrasp_rank, candidate.pregrasp_distance),
                rank_value(thin_rank, candidate.thin_bias),
                rank_value(approach_rank, candidate.approach_bias),
                rank_value(tilt_rank, candidate.approach_tilt_deg),
                rank_value(yaw_rank, candidate.yaw_deg),
            ]
            return (
                max(core_ranks),
                sum(core_ranks),
                _center_distance(candidate),
                rank_value(tilt_rank, candidate.approach_tilt_deg),
                rank_value(yaw_rank, candidate.yaw_deg),
                rank_value(approach_rank, candidate.approach_bias),
                rank_value(pregrasp_rank, candidate.pregrasp_distance),
                rank_value(thin_rank, candidate.thin_bias),
                edge_rank(candidate),
            )

        def front_wall_candidate_key(candidate: Any) -> tuple[Any, ...]:
            core_ranks = [
                rank_value(pregrasp_rank, candidate.pregrasp_distance),
                rank_value(thin_rank, candidate.thin_bias),
                rank_value(approach_rank, candidate.approach_bias),
                rank_value(tilt_rank, candidate.approach_tilt_deg),
                rank_value(yaw_rank, candidate.yaw_deg),
            ]
            return (
                max(core_ranks),
                sum(core_ranks),
                _center_distance(candidate),
                rank_value(pregrasp_rank, candidate.pregrasp_distance),
                rank_value(approach_rank, candidate.approach_bias),
                rank_value(tilt_rank, candidate.approach_tilt_deg),
                rank_value(yaw_rank, candidate.yaw_deg),
                rank_value(thin_rank, candidate.thin_bias),
                edge_rank(candidate),
            )

        key_fn = front_wall_candidate_key if role == "front_wall" else candidate_key
        filtered = _apply_wall_grasp_prior(args, role, axis_filtered(sorted(pool, key=key_fn)))
        filtered = _append_extra_tilt_wall_grasp_candidates(args, filtered)
        return filtered[start_index : start_index + max(int(args.max_grasp_candidates), 1)]
    if bool(getattr(args, "defer_top_lid_release_screen", False)) and not bool(
        getattr(args, "top_lid_prefer_tilted_grasp", False)
    ):
        return _make_grasp_candidates(args.max_grasp_candidates + start_index)[start_index:]
    pool = _make_grasp_candidates(max(int(args.max_grasp_candidates), 144))
    tilt_rank = {8.0: 0, -8.0: 1, 14.0: 2, -14.0: 3, 0.0: 4}
    yaw_rank = {0.0: 0, 4.0: 1, -4.0: 2}
    return sorted(
        pool,
        key=lambda candidate: (
            tilt_rank.get(float(candidate.approach_tilt_deg), 9),
            yaw_rank.get(float(candidate.yaw_deg), 9),
            abs(float(candidate.thin_bias)),
            abs(float(candidate.approach_bias) + 0.018),
            abs(float(candidate.local_x)) + abs(float(candidate.local_y)),
        ),
    )[start_index : start_index + max(int(args.max_grasp_candidates), 1)]


def _ik_error_score(result: Any) -> float:
    debug = getattr(result, "debug", {}) if result is not None else {}
    if not isinstance(debug, dict):
        debug = {}
    pos = debug.get("position_error", np.nan)
    rot = debug.get("rotation_error", np.nan)
    pos_score = 0.0 if pos is None or not np.isfinite(float(pos)) else float(pos)
    rot_score = 0.0 if rot is None or not np.isfinite(float(rot)) else float(rot)
    return float(pos_score + 0.02 * rot_score)


def _batch_ik_to_poses(
    *,
    planner: RM75CuRoboPlanner,
    base_env: Any,
    locked: dict[str, Any],
    fixtures: list[dict[str, Any]],
    start_qs: list[np.ndarray],
    poses: list[sapien.Pose],
    num_seeds: int,
    exclude_role: str | None = None,
    exclude_roles: set[str] | None = None,
    use_cuda_graph_batch: bool = False,
    cuda_graph_batch_size: int | None = None,
    cuda_graph_fixed_batch_size: int | None = None,
) -> list[Any]:
    if not start_qs or not poses:
        return []
    obstacles = _world_obstacles_for_stage(
        base_env,
        locked,
        fixtures,
        exclude_role=exclude_role,
        exclude_roles=exclude_roles,
    )
    planner.set_world_from_obstacles(cuboids=obstacles)
    robot_base_poses = [_world_to_robot_base(base_env, pose) for pose in poses]
    return planner.solve_batch_start_goal_ik(
        start_qs,
        robot_base_poses,
        num_seeds=int(num_seeds),
        use_cuda_graph_batch=bool(use_cuda_graph_batch),
        cuda_graph_batch_size=None if cuda_graph_batch_size is None else int(cuda_graph_batch_size),
        cuda_graph_fixed_batch_size=None if cuda_graph_fixed_batch_size is None else int(cuda_graph_fixed_batch_size),
    )


def _fast_chain_preselect_grasp_candidates(
    *,
    planner: RM75CuRoboPlanner,
    base_env: Any,
    locked: dict[str, Any],
    fixtures: list[dict[str, Any]],
    actor: Any,
    role: str,
    candidate_pool: list[Any],
    start_q: np.ndarray,
    target_actor_pose: sapien.Pose | None,
    args: argparse.Namespace,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    report: dict[str, Any] = {
        "enabled": bool(getattr(args, "fast_chain_screening", True)),
        "candidate_count_in": len(candidate_pool),
        "top_k": int(getattr(args, "fast_chain_top_grasp_candidates", 0)),
        "pair_probe_candidates": int(getattr(args, "fast_chain_pair_probe_candidates", 0)),
    }
    if (
        not bool(getattr(args, "fast_chain_screening", True))
        or int(getattr(args, "fast_chain_top_grasp_candidates", 0)) <= 0
        or len(candidate_pool) <= int(getattr(args, "fast_chain_top_grasp_candidates", 0))
    ):
        report["status"] = "BYPASS"
        return candidate_pool, [], report
    started = time.perf_counter()
    top_k = max(int(getattr(args, "fast_chain_top_grasp_candidates", 4)), 1)
    ik_seeds = max(int(getattr(args, "fast_chain_ik_seeds", min(int(args.ik_seeds), 16))), 1)
    connection_potential = _wall_connection_potential(base_env, role)
    supported_wall_max_actor_to_tcp_y = float(
        getattr(args, "fast_chain_supported_wall_max_actor_to_tcp_y", 0.0) or 0.0
    )
    report["connection_potential"] = int(connection_potential)
    report["supported_wall_max_actor_to_tcp_y"] = float(supported_wall_max_actor_to_tcp_y)
    start_q = np.asarray(start_q, dtype=np.float32).reshape(7)
    try:
        staged: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidate_pool):
            grasp_tcp = _candidate_tcp_for_edge_grasp(actor, candidate)
            staged.append(
                {
                    "index": index,
                    "candidate": candidate,
                    "grasp_tcp": grasp_tcp,
                    "pregrasp": _pregrasp_pose_for_candidate(grasp_tcp, candidate),
                }
            )
        pre_results = _batch_ik_to_poses(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            start_qs=[start_q for _ in staged],
            poses=[item["pregrasp"] for item in staged],
            num_seeds=ik_seeds,
            exclude_role=None,
            use_cuda_graph_batch=bool(getattr(args, "use_cuda_graph_batch_ik", False)),
            cuda_graph_batch_size=int(getattr(args, "cuda_graph_batch_ik_max_batch", 0) or 0) or None,
            cuda_graph_fixed_batch_size=int(getattr(args, "cuda_graph_batch_ik_fixed_batch_size", 0) or 0) or None,
        )
        pre_ok: list[dict[str, Any]] = []
        for item, result in zip(staged, pre_results):
            if bool(getattr(result, "success", False)) and getattr(result, "goal_joint", None) is not None:
                copied = dict(item)
                copied["q_pregrasp"] = np.asarray(result.goal_joint, dtype=np.float32).reshape(-1)[:7]
                copied["pregrasp_ik_score"] = _ik_error_score(result)
                pre_ok.append(copied)
        report["pregrasp_ik_ok_count"] = len(pre_ok)
        if pre_ok:
            grasp_results = _batch_ik_to_poses(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                start_qs=[item["q_pregrasp"] for item in pre_ok],
                poses=[item["grasp_tcp"] for item in pre_ok],
                num_seeds=ik_seeds,
                exclude_role=role,
                use_cuda_graph_batch=bool(getattr(args, "use_cuda_graph_batch_ik", False)),
                cuda_graph_batch_size=int(getattr(args, "cuda_graph_batch_ik_max_batch", 0) or 0) or None,
                cuda_graph_fixed_batch_size=int(getattr(args, "cuda_graph_batch_ik_fixed_batch_size", 0) or 0) or None,
            )
        else:
            grasp_results = []
        grasp_ok: list[dict[str, Any]] = []
        for item, result in zip(pre_ok, grasp_results):
            if bool(getattr(result, "success", False)) and getattr(result, "goal_joint", None) is not None:
                copied = dict(item)
                copied["q_grasp"] = np.asarray(result.goal_joint, dtype=np.float32).reshape(-1)[:7]
                copied["grasp_ik_score"] = _ik_error_score(result)
                grasp_ok.append(copied)
        report["grasp_ik_ok_count"] = len(grasp_ok)
        if grasp_ok:
            lift_poses = [
                type(item["grasp_tcp"])(
                    p=[
                        item["grasp_tcp"].p[0],
                        item["grasp_tcp"].p[1],
                        item["grasp_tcp"].p[2] + args.lift_height,
                    ],
                    q=item["grasp_tcp"].q,
                )
                for item in grasp_ok
            ]
            lift_results = _batch_ik_to_poses(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                start_qs=[item["q_grasp"] for item in grasp_ok],
                poses=lift_poses,
                num_seeds=ik_seeds,
                exclude_role=role,
                use_cuda_graph_batch=bool(getattr(args, "use_cuda_graph_batch_ik", False)),
                cuda_graph_batch_size=int(getattr(args, "cuda_graph_batch_ik_max_batch", 0) or 0) or None,
                cuda_graph_fixed_batch_size=int(getattr(args, "cuda_graph_batch_ik_fixed_batch_size", 0) or 0) or None,
            )
        else:
            lift_results = []
        ranked: list[dict[str, Any]] = []
        for item, result in zip(grasp_ok, lift_results):
            if not bool(getattr(result, "success", False)) or getattr(result, "goal_joint", None) is None:
                continue
            q_lift = np.asarray(result.goal_joint, dtype=np.float32).reshape(-1)[:7]
            candidate = item["candidate"]
            score = (
                0.001 * float(item["index"])
                + _joint_distance(item["q_pregrasp"], start_q)
                + 0.35 * _joint_distance(item["q_grasp"], item["q_pregrasp"])
                + 0.20 * _joint_distance(q_lift, item["q_grasp"])
                + float(item.get("pregrasp_ik_score", 0.0))
                + float(item.get("grasp_ik_score", 0.0))
                + _ik_error_score(result)
                + float(getattr(args, "fast_chain_center_distance_weight", 0.28)) * _center_distance(candidate)
                + 0.06 * abs(float(candidate.approach_bias))
                + 0.02 * abs(float(candidate.thin_bias))
            )
            ranked.append(
                {
                    "index": int(item["index"]),
                    "label": str(candidate.label),
                    "score": float(score),
                    "candidate": candidate,
                    "grasp_tcp": item["grasp_tcp"],
                    "q_lift": q_lift,
                    "nominal_actor_to_tcp": _actor_pose(actor).inv() * item["grasp_tcp"],
                }
            )
            actor_to_tcp_y = abs(float(ranked[-1]["nominal_actor_to_tcp"].p[1]))
            ranked[-1]["actor_to_tcp_y"] = float(actor_to_tcp_y)
            ranked[-1]["supported_wall_grasp_depth_risk"] = bool(
                role in FIRST_LAYER_WALL_ROLES
                and int(connection_potential) >= 2
                and supported_wall_max_actor_to_tcp_y > 0.0
                and actor_to_tcp_y > supported_wall_max_actor_to_tcp_y
            )
        ranked.sort(key=lambda item: (float(item["score"]), int(item["index"])))
        report["lift_ik_ok_count"] = len(ranked)
        if not ranked:
            report["status"] = "NO_BATCH_IK_WINNER_FALLBACK_FULL"
            report["elapsed_sec"] = time.perf_counter() - started
            return candidate_pool, [], report
        probe_count = max(int(getattr(args, "fast_chain_pair_probe_candidates", 0)), 0)
        if target_actor_pose is not None and probe_count > 0:
            probe_reports: list[dict[str, Any]] = []
            release_screen_by_label: dict[str, dict[str, Any]] = {}
            stop_after_success = bool(getattr(args, "fast_chain_stop_pair_probe_after_success", False))
            min_successful_probes = max(int(getattr(args, "fast_chain_min_successful_pair_probes", 0) or 0), 0)
            for item in ranked[:probe_count]:
                release_screen = _screen_release_for_grasp(
                    planner=planner,
                    base_env=base_env,
                    locked=locked,
                    fixtures=fixtures,
                    role=role,
                    actor_to_tcp=item["nominal_actor_to_tcp"],
                    target_actor_pose=target_actor_pose,
                    start_q=item["q_lift"],
                    args=args,
                )
                item["release_screen"] = release_screen
                release_screen_by_label[str(item["label"])] = release_screen
                probe_reports.append(
                    {
                        "index": int(item["index"]),
                        "label": str(item["label"]),
                        "success": bool(release_screen.get("success")),
                        "selected_score": release_screen.get("selected_score"),
                    }
                )
                success_count_so_far = sum(1 for probe in probe_reports if bool(probe.get("success")))
                if (
                    stop_after_success
                    and bool(release_screen.get("success"))
                    and success_count_so_far >= max(min_successful_probes, 1)
                ):
                    report["pair_release_probe_stopped_early"] = True
                    break
            for item in ranked:
                release_screen = item.get("release_screen", {})
                if not bool(release_screen.get("success")):
                    continue
                selected_score = release_screen.get("selected_score")
                try:
                    release_probe_score = float(selected_score)
                except (TypeError, ValueError):
                    release_probe_score = 0.0
                item["release_probe_score"] = float(release_probe_score)
                item["score"] = float(item["score"]) + float(getattr(args, "fast_chain_release_probe_weight", 0.6)) * float(
                    release_probe_score
                )
            if any(bool(item.get("release_screen", {}).get("success")) for item in ranked):
                ranked.sort(
                    key=lambda item: (
                        1 if bool(item.get("supported_wall_grasp_depth_risk", False)) else 0,
                        0 if bool(item.get("release_screen", {}).get("success")) else 1,
                        float(item.get("release_probe_score", float("inf"))),
                        float(item["score"]),
                        int(item["index"]),
                    )
                )
            elif supported_wall_max_actor_to_tcp_y > 0.0 and int(connection_potential) >= 2:
                ranked.sort(
                    key=lambda item: (
                        1 if bool(item.get("supported_wall_grasp_depth_risk", False)) else 0,
                        float(item["score"]),
                        int(item["index"]),
                    )
                )
            report["pair_release_probe_count"] = len(probe_reports)
            report["pair_release_success_count"] = sum(
                1 for item in ranked if bool(item.get("release_screen", {}).get("success"))
            )
            report["pair_release_probe_reports"] = probe_reports
            report["release_screen_by_label"] = release_screen_by_label
        selected_ranked = ranked[:top_k]
        selected_indices = {int(item["index"]) for item in selected_ranked}
        primary = [candidate_pool[int(item["index"])] for item in selected_ranked]
        fallback = [candidate for index, candidate in enumerate(candidate_pool) if index not in selected_indices]
        report["status"] = "Success"
        report["selected_count"] = len(primary)
        report["fallback_count"] = len(fallback)
        report["selected"] = [
            {
                "index": int(item["index"]),
                "label": str(item["label"]),
                "score": float(item["score"]),
                "actor_to_tcp_y": float(item.get("actor_to_tcp_y", 0.0)),
                "supported_wall_grasp_depth_risk": bool(item.get("supported_wall_grasp_depth_risk", False)),
                "pair_release_success": bool(item.get("release_screen", {}).get("success")),
                "release_probe_score": item.get("release_probe_score"),
            }
            for item in selected_ranked
        ]
        report["elapsed_sec"] = time.perf_counter() - started
        return primary, fallback if bool(getattr(args, "fast_chain_allow_fallback", True)) else [], report
    except Exception as exc:
        report["status"] = type(exc).__name__
        report["error"] = str(exc)
        report["elapsed_sec"] = time.perf_counter() - started
        return candidate_pool, [], report


def _release_candidate_entries(
    args: argparse.Namespace,
    role: str,
    target_actor_pose: sapien.Pose,
) -> list[tuple[int, str, sapien.Pose, dict[str, Any]]]:
    entries: list[tuple[int, str, sapien.Pose, dict[str, Any]]] = []
    index_filter = _release_candidate_index_filter(args, role)
    index_set = set(index_filter)
    for index, (label, release_actor_pose) in enumerate(_release_candidates_for_role(args, role, target_actor_pose)):
        if index_filter:
            if index not in index_set:
                if index > max(index_filter):
                    break
                continue
        elif index >= int(args.max_release_candidates):
            break
        entries.append((index, label, release_actor_pose, {}))
        if index_filter and len(entries) >= len(index_filter):
            break
    return entries


def _release_candidate_order_rank(args: argparse.Namespace, role: str, index: int) -> int:
    mapping_text = str(getattr(args, "release_candidate_indices", "") or "")
    if not mapping_text.strip():
        return 100000 + int(index)
    role_values: list[int] = []
    global_values: list[int] = []
    for item in mapping_text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            global_values.extend(_int_csv(item))
            continue
        key, value = item.split(":", 1)
        if key.strip() == role:
            role_values.extend(_int_csv(value))
    values = role_values or global_values
    for rank, value in enumerate(values):
        if int(value) == int(index):
            return rank
    return 100000 + int(index)


def _fixed_release_preplace_params(label: str) -> dict[str, float] | None:
    text = str(label or "")
    if text.startswith("fixed_top_h"):
        try:
            height_mm = float(text.split("fixed_top_h", 1)[1].split("mm", 1)[0])
        except Exception:
            return None
        return {"kind": "top", "height": height_mm / 1000.0, "retreat": 0.0}
    if text.startswith("fixed_out_r"):
        try:
            retreat_mm = float(text.split("fixed_out_r", 1)[1].split("mm", 1)[0])
            height_mm = float(text.split("_h", 1)[1].split("mm", 1)[0])
        except Exception:
            return None
        return {"kind": "outside", "height": height_mm / 1000.0, "retreat": retreat_mm / 1000.0}
    return None


def _fast_chain_preselect_release_candidates(
    *,
    planner: RM75CuRoboPlanner,
    base_env: Any,
    locked: dict[str, Any],
    fixtures: list[dict[str, Any]],
    role: str,
    actor_to_tcp: Any,
    target_actor_pose: sapien.Pose,
    start_q: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[tuple[int, str, sapien.Pose, dict[str, Any]]], list[tuple[int, str, sapien.Pose, dict[str, Any]]], dict[str, Any]]:
    entries = _release_candidate_entries(args, role, target_actor_pose)
    release_args = _args_for_role(args, role)
    report: dict[str, Any] = {
        "enabled": bool(getattr(args, "fast_chain_screening", True)),
        "candidate_count_in": len(entries),
        "top_k": int(getattr(args, "fast_chain_top_release_candidates", 0)),
    }
    if (
        not bool(getattr(args, "fast_chain_screening", True))
        or int(getattr(args, "fast_chain_top_release_candidates", 0)) <= 0
        or len(entries) <= int(getattr(args, "fast_chain_top_release_candidates", 0))
    ):
        report["status"] = "BYPASS"
        return entries, [], report
    started = time.perf_counter()
    top_k = max(int(getattr(args, "fast_chain_top_release_candidates", 4)), 1)
    ik_seeds = max(int(getattr(args, "fast_chain_ik_seeds", min(int(args.ik_seeds), 16))), 1)
    release_exclude_roles = _release_collision_exclude_roles(release_args, role)
    release_preplace_max_joint_delta = float(getattr(release_args, "release_preplace_max_joint_delta", 0.0))
    release_max_joint_delta = float(getattr(release_args, "release_max_joint_delta", 0.0))
    start_q = np.asarray(start_q, dtype=np.float32).reshape(7)
    try:
        staged: list[dict[str, Any]] = []
        for index, label, release_actor_pose, _cache in entries:
            target_tcp = release_actor_pose * actor_to_tcp
            preplace, approach_report = _release_preplace_pose_for_candidate(
                base_env=base_env,
                role=role,
                release_actor_pose=release_actor_pose,
                target_tcp=target_tcp,
                args=args,
                candidate_label=label,
            )
            staged.append(
                {
                    "index": index,
                    "label": label,
                    "release_actor_pose": release_actor_pose,
                    "target_tcp": target_tcp,
                    "preplace": preplace,
                    "release_approach": approach_report,
                    "pose_error": _pose_to_pose_error(release_actor_pose, target_actor_pose),
                }
            )
        pre_results = _batch_ik_to_poses(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            start_qs=[start_q for _ in staged],
            poses=[item["preplace"] for item in staged],
            num_seeds=ik_seeds,
            exclude_role=role,
            use_cuda_graph_batch=bool(getattr(args, "use_cuda_graph_batch_ik", False)),
            cuda_graph_batch_size=int(getattr(args, "cuda_graph_batch_ik_max_batch", 0) or 0) or None,
            cuda_graph_fixed_batch_size=int(getattr(args, "cuda_graph_batch_ik_fixed_batch_size", 0) or 0) or None,
        )
        pre_ok: list[dict[str, Any]] = []
        for item, result in zip(staged, pre_results):
            if not bool(getattr(result, "success", False)) or getattr(result, "goal_joint", None) is None:
                continue
            q_preplace = np.asarray(result.goal_joint, dtype=np.float32).reshape(-1)[:7]
            preplace_joint_delta = _joint_distance(q_preplace, start_q)
            if release_preplace_max_joint_delta > 0.0 and preplace_joint_delta > release_preplace_max_joint_delta:
                continue
            copied = dict(item)
            copied["q_preplace"] = q_preplace
            copied["pre_report"] = {
                "success": True,
                "status": str(getattr(result, "status", "Success")),
                "solve_time": float(getattr(result, "solve_time", 0.0) or 0.0),
                "ik_time": float(getattr(result, "ik_time", 0.0) or 0.0),
                "debug": getattr(result, "debug", {}),
                "fast_chain_cached": True,
                "joint_distance_to_start": float(preplace_joint_delta),
            }
            copied["preplace_joint_delta"] = float(preplace_joint_delta)
            pre_ok.append(copied)
        report["preplace_ik_ok_count"] = len(pre_ok)
        if pre_ok:
            place_results = _batch_ik_to_poses(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                start_qs=[item["q_preplace"] for item in pre_ok],
                poses=[item["target_tcp"] for item in pre_ok],
                num_seeds=ik_seeds,
                exclude_role=None,
                exclude_roles=release_exclude_roles,
                use_cuda_graph_batch=bool(getattr(args, "use_cuda_graph_batch_ik", False)),
                cuda_graph_batch_size=int(getattr(args, "cuda_graph_batch_ik_max_batch", 0) or 0) or None,
                cuda_graph_fixed_batch_size=int(getattr(args, "cuda_graph_batch_ik_fixed_batch_size", 0) or 0) or None,
            )
        else:
            place_results = []
        ranked: list[dict[str, Any]] = []
        cache_by_index: dict[int, dict[str, Any]] = {}
        for item, result in zip(pre_ok, place_results):
            if not bool(getattr(result, "success", False)) or getattr(result, "goal_joint", None) is None:
                continue
            q_place = np.asarray(result.goal_joint, dtype=np.float32).reshape(-1)[:7]
            place_joint_delta = _joint_distance(q_place, item["q_preplace"])
            score = _release_candidate_score(
                label=item["label"],
                pose_error=item["pose_error"],
                preplace_joint_delta=float(item["preplace_joint_delta"]),
                place_joint_delta=float(place_joint_delta),
                preplace_joint_weight=float(getattr(args, "release_score_preplace_joint_weight", 0.005)),
                place_joint_weight=float(getattr(args, "release_score_place_joint_weight", 0.0015)),
            )
            score += _release_ik_error_penalty(
                preplace_report=item.get("pre_report"),
                place_report={"debug": getattr(result, "debug", {})},
                args=args,
            )
            connection_geometry = _release_connection_geometry_report(
                base_env=base_env,
                role=role,
                release_actor_pose=item["release_actor_pose"],
                args=args,
            )
            if release_max_joint_delta > 0.0 and place_joint_delta > release_max_joint_delta:
                continue
            cache = {
                "q_preplace": item["q_preplace"],
                "q_place": q_place,
                "pre_report": item["pre_report"],
                "place_report": {
                    "success": True,
                    "status": str(getattr(result, "status", "Success")),
                    "solve_time": float(getattr(result, "solve_time", 0.0) or 0.0),
                    "ik_time": float(getattr(result, "ik_time", 0.0) or 0.0),
                    "debug": getattr(result, "debug", {}),
                    "fast_chain_cached": True,
                    "joint_distance_to_start": float(place_joint_delta),
                },
                "preplace_joint_delta": float(item["preplace_joint_delta"]),
                "place_joint_delta": float(place_joint_delta),
                "score": float(score),
                "connection_geometry": connection_geometry,
            }
            cache_by_index[int(item["index"])] = cache
            ranked.append(
                {
                    "index": int(item["index"]),
                    "label": str(item["label"]),
                    "score": float(score),
                    "connection_rank": tuple(connection_geometry.get("rank", (99, 99, float("inf"), float("inf"), float("inf")))),
                    "connection_geometry": connection_geometry,
                }
            )
        connection_potential = _wall_connection_potential(base_env, role)
        prefer_index_order = bool(
            role in FIRST_LAYER_WALL_ROLES
            and int(connection_potential) >= 2
            and bool(getattr(args, "release_prefer_candidate_index_order_for_multi_connection", False))
        )
        ranked.sort(
            key=lambda item: (
                tuple(item.get("connection_rank", (99, 99, float("inf"), float("inf"), float("inf")))),
                _release_candidate_order_rank(args, role, int(item["index"])) if prefer_index_order else 0,
                float(item["score"]),
                int(item["index"]),
            )
        )
        report["place_ik_ok_count"] = len(ranked)
        report["elapsed_sec"] = time.perf_counter() - started
        if not ranked:
            report["status"] = "NO_BATCH_IK_WINNER_FALLBACK_FULL"
            return entries, [], report
        forced_index = int(getattr(release_args, "release_candidate_index", -1))
        selected_ranked = ranked[:top_k]
        if forced_index >= 0 and forced_index in cache_by_index and all(int(item["index"]) != forced_index for item in selected_ranked):
            forced_item = next(item for item in ranked if int(item["index"]) == forced_index)
            selected_ranked = [forced_item, *selected_ranked[: max(top_k - 1, 0)]]
        selected_indices = {int(item["index"]) for item in selected_ranked}

        def with_cache(entry: tuple[int, str, sapien.Pose, dict[str, Any]]) -> tuple[int, str, sapien.Pose, dict[str, Any]]:
            index, label, release_actor_pose, cache = entry
            merged = dict(cache)
            if index in cache_by_index:
                merged.update(cache_by_index[index])
            return index, label, release_actor_pose, merged

        entry_by_index = {int(entry[0]): entry for entry in entries}
        primary = [with_cache(entry_by_index[int(item["index"])]) for item in selected_ranked if int(item["index"]) in entry_by_index]
        fallback = [with_cache(entry) for entry in entries if entry[0] not in selected_indices]
        report["status"] = "Success"
        report["selected_count"] = len(primary)
        report["fallback_count"] = len(fallback)
        report["selected"] = selected_ranked
        return primary, fallback if bool(getattr(args, "fast_chain_allow_fallback", True)) else [], report
    except Exception as exc:
        report["status"] = type(exc).__name__
        report["error"] = str(exc)
        report["elapsed_sec"] = time.perf_counter() - started
        return entries, [], report


def _actor_to_tcp_with_local_offset(
    actor_to_tcp: sapien.Pose,
    *,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    yaw_deg: float = 0.0,
) -> sapien.Pose:
    position, quaternion = _pose_arrays(actor_to_tcp)
    shifted = position.astype(np.float32).copy()
    shifted[0] += float(offset_x)
    shifted[1] += float(offset_y)
    shifted[2] += float(offset_z)
    if abs(yaw_deg) > 1e-9:
        yaw = np.deg2rad(float(yaw_deg))
        yaw_rotation = np.asarray(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        quaternion = mat2quat(yaw_rotation @ quat2mat(quaternion)).astype(np.float32)
    return sapien.Pose(p=shifted.tolist(), q=quaternion.tolist())


def _release_robust_actor_to_tcp_offset_grid(args: argparse.Namespace) -> list[dict[str, float]]:
    x_offsets = _float_csv(str(getattr(args, "release_robust_actor_to_tcp_x_offsets", "") or ""), [0.0])
    y_offsets = _float_csv(str(getattr(args, "release_robust_actor_to_tcp_y_offsets", "") or ""), [0.0])
    z_offsets = _float_csv(str(getattr(args, "release_robust_actor_to_tcp_z_offsets", "") or ""), [0.0])
    yaw_offsets = _float_csv(str(getattr(args, "release_robust_actor_to_tcp_yaw_deg_offsets", "") or ""), [0.0])
    normalized_x: list[float] = []
    normalized_y: list[float] = []
    normalized_z: list[float] = []
    normalized_yaw: list[float] = []
    for value in x_offsets:
        if all(abs(float(value) - existing) > 1e-9 for existing in normalized_x):
            normalized_x.append(float(value))
    for value in y_offsets:
        if all(abs(float(value) - existing) > 1e-9 for existing in normalized_y):
            normalized_y.append(float(value))
    for value in z_offsets:
        if all(abs(float(value) - existing) > 1e-9 for existing in normalized_z):
            normalized_z.append(float(value))
    for value in yaw_offsets:
        if all(abs(float(value) - existing) > 1e-9 for existing in normalized_yaw):
            normalized_yaw.append(float(value))
    if all(abs(item) > 1e-9 for item in normalized_x):
        normalized_x.insert(0, 0.0)
    if all(abs(item) > 1e-9 for item in normalized_y):
        normalized_y.insert(0, 0.0)
    if all(abs(item) > 1e-9 for item in normalized_z):
        normalized_z.insert(0, 0.0)
    if all(abs(item) > 1e-9 for item in normalized_yaw):
        normalized_yaw.insert(0, 0.0)
    grid: list[dict[str, float]] = []
    for offset_x in normalized_x or [0.0]:
        for offset_y in normalized_y or [0.0]:
            for offset_z in normalized_z or [0.0]:
                for yaw_deg in normalized_yaw or [0.0]:
                    grid.append(
                        {
                            "offset_x": float(offset_x),
                            "offset_y": float(offset_y),
                            "offset_z": float(offset_z),
                            "yaw_deg": float(yaw_deg),
                        }
                    )
    return grid or [{"offset_x": 0.0, "offset_y": 0.0, "offset_z": 0.0, "yaw_deg": 0.0}]


def _release_robust_actor_to_tcp_x_offset_grid(args: argparse.Namespace) -> list[dict[str, float]]:
    x_offsets = _float_csv(str(getattr(args, "release_robust_actor_to_tcp_x_offsets", "") or ""), [0.0])
    normalized_x: list[float] = []
    for value in x_offsets:
        if all(abs(float(value) - existing) > 1e-9 for existing in normalized_x):
            normalized_x.append(float(value))
    if all(abs(item) > 1e-9 for item in normalized_x):
        normalized_x.insert(0, 0.0)
    return [{"offset_x": float(offset_x), "offset_y": 0.0, "offset_z": 0.0, "yaw_deg": 0.0} for offset_x in (normalized_x or [0.0])]


def _screen_release_for_grasp(
    *,
    planner: RM75CuRoboPlanner,
    base_env: Any,
    locked: dict[str, Any],
    fixtures: list[dict[str, Any]],
    role: str,
    actor_to_tcp: Any,
    target_actor_pose: Any,
    start_q: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    offsets = _release_robust_actor_to_tcp_offset_grid(args)
    x_fallback_offsets = _release_robust_actor_to_tcp_x_offset_grid(args)
    min_successes = max(int(getattr(args, "release_robust_min_variant_successes", 1)), 1)
    robust_weight = float(getattr(args, "release_robust_score_weight", 0.0))
    preferred_y = float(getattr(args, "release_robust_preferred_actor_to_tcp_y_offset", 0.0) or 0.0)
    require_preferred = bool(getattr(args, "release_robust_require_preferred_actor_to_tcp_y_offset", False))
    require_preferred_max_connection_potential = int(
        getattr(args, "release_robust_require_preferred_max_connection_potential", 0) or 0
    )
    connection_potential = _wall_connection_potential(base_env, role)
    if (
        require_preferred
        and require_preferred_max_connection_potential > 0
        and int(connection_potential) > require_preferred_max_connection_potential
    ):
        require_preferred = False
    has_preferred_y = abs(preferred_y) > 1e-9
    early_stop_on_first_success = bool(getattr(args, "release_robust_early_stop_on_first_success", False))
    if len(offsets) <= 1 and min_successes <= 1:
        result = _screen_release_for_actor_to_tcp(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            role=role,
            actor_to_tcp=actor_to_tcp,
            target_actor_pose=target_actor_pose,
            start_q=start_q,
            args=args,
        )
        result["robust_actor_to_tcp"] = {
            "enabled": False,
            "offsets": offsets,
            "success_count": 1 if bool(result.get("success")) else 0,
            "min_successes": min_successes,
        }
        return result

    variant_results: list[dict[str, Any]] = []
    successful: list[dict[str, Any]] = []
    def append_variant(offset: dict[str, float], result: dict[str, Any], *, stage: str) -> None:
        compact = {
            "offset_x": float(offset.get("offset_x", 0.0)),
            "offset_y": float(offset.get("offset_y", 0.0)),
            "offset_z": float(offset.get("offset_z", 0.0)),
            "yaw_deg": float(offset.get("yaw_deg", 0.0)),
            "offset_stage": stage,
            "success": bool(result.get("success")),
            "selected": result.get("selected"),
            "selected_index": result.get("selected_index"),
            "selected_score": result.get("selected_score"),
            "selected_release_mode": result.get("selected_release_mode"),
            "release_approach": result.get("release_approach", {}),
            "fast_chain_release_preselect": result.get("fast_chain_release_preselect", {}),
            "reports": result.get("reports", []),
        }
        variant_results.append(compact)
        if bool(result.get("success")):
            successful.append(
                {
                    "offset_x": float(offset.get("offset_x", 0.0)),
                    "offset_y": float(offset.get("offset_y", 0.0)),
                    "offset_z": float(offset.get("offset_z", 0.0)),
                    "yaw_deg": float(offset.get("yaw_deg", 0.0)),
                    "offset_stage": stage,
                    "result": result,
                }
            )

    for offset in offsets:
        variant_actor_to_tcp = _actor_to_tcp_with_local_offset(
            actor_to_tcp,
            offset_x=float(offset.get("offset_x", 0.0)),
            offset_y=float(offset.get("offset_y", 0.0)),
            offset_z=float(offset.get("offset_z", 0.0)),
            yaw_deg=float(offset.get("yaw_deg", 0.0)),
        )
        result = _screen_release_for_actor_to_tcp(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            role=role,
            actor_to_tcp=variant_actor_to_tcp,
            target_actor_pose=target_actor_pose,
            start_q=start_q,
            args=args,
        )
        append_variant(offset, result, stage="primary")
        if (
            early_stop_on_first_success
            and len(successful) >= min_successes
            and not bool(getattr(args, "release_robust_require_same_release_index", False))
            and (
                not require_preferred
                or not has_preferred_y
                or abs(float(offset.get("offset_y", 0.0)) - preferred_y) <= 1e-6
            )
        ):
            break

    if (
        role in FIRST_LAYER_WALL_ROLES
        and len(successful) < min_successes
        and any(abs(float(item.get("offset_x", 0.0))) > 1e-9 for item in x_fallback_offsets)
    ):
        for offset in x_fallback_offsets:
            variant_actor_to_tcp = _actor_to_tcp_with_local_offset(
                actor_to_tcp,
                offset_x=float(offset.get("offset_x", 0.0)),
                offset_y=0.0,
                offset_z=0.0,
                yaw_deg=0.0,
            )
            result = _screen_release_for_actor_to_tcp(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                role=role,
                actor_to_tcp=variant_actor_to_tcp,
                target_actor_pose=target_actor_pose,
                start_q=start_q,
                args=args,
            )
            append_variant(offset, result, stage="x_fallback")

    success_count = len(successful)
    robust_report = {
        "enabled": True,
        "offsets": offsets,
        "x_fallback_offsets": x_fallback_offsets,
        "success_count": int(success_count),
        "min_successes": int(min_successes),
        "score_weight": float(robust_weight),
        "variants": variant_results,
        "early_stop_on_first_success": bool(early_stop_on_first_success),
    }
    if has_preferred_y:
        robust_report["preferred_actor_to_tcp_y_offset"] = float(preferred_y)
        robust_report["require_preferred_max_connection_potential"] = int(require_preferred_max_connection_potential)
        robust_report["connection_potential"] = int(connection_potential)
    if require_preferred and has_preferred_y:
        preferred_successful = [
            item
            for item in successful
            if abs(float(item.get("offset_y", 0.0)) - preferred_y) <= 1e-6
        ]
        robust_report["require_preferred_actor_to_tcp_y_offset"] = True
        robust_report["preferred_success_count"] = len(preferred_successful)
        if not preferred_successful:
            nominal = next(
                (item for item in variant_results if abs(float(item["offset_y"])) <= 1e-9),
                variant_results[0] if variant_results else {},
            )
            return {
                "success": False,
                "failed_at": "no_robust_grasp_release_pair",
                "selected": nominal.get("selected"),
                "selected_index": nominal.get("selected_index"),
                "selected_score": nominal.get("selected_score"),
                "reports": nominal.get("reports", []),
                "variant_reports": variant_results,
                "robust_actor_to_tcp": robust_report,
            }
        successful = preferred_successful
        success_count = len(successful)
        robust_report["success_count"] = int(success_count)
    if bool(getattr(args, "release_robust_require_same_release_index", False)) and successful:
        by_index: dict[int, list[dict[str, Any]]] = {}
        for item in successful:
            selected_index = item["result"].get("selected_index")
            if selected_index is None:
                continue
            by_index.setdefault(int(selected_index), []).append(item)
        branch_summaries = [
            {
                "selected_index": int(index),
                "success_count": len(items),
                "offsets": [float(entry["offset_y"]) for entry in items],
                "best_score": min(float(entry["result"].get("selected_score", float("inf"))) for entry in items),
            }
            for index, items in by_index.items()
        ]
        branch_summaries.sort(key=lambda item: (-int(item["success_count"]), float(item["best_score"]), int(item["selected_index"])))
        robust_report["require_same_release_index"] = True
        robust_report["branch_summaries"] = branch_summaries
        if not branch_summaries or int(branch_summaries[0]["success_count"]) < min_successes:
            nominal = next(
                (item for item in variant_results if abs(float(item["offset_y"])) <= 1e-9),
                variant_results[0] if variant_results else {},
            )
            return {
                "success": False,
                "failed_at": "no_robust_grasp_release_pair",
                "selected": nominal.get("selected"),
                "selected_index": nominal.get("selected_index"),
                "selected_score": nominal.get("selected_score"),
                "reports": nominal.get("reports", []),
                "variant_reports": variant_results,
                "robust_actor_to_tcp": robust_report,
            }
        best_index = int(branch_summaries[0]["selected_index"])
        successful = by_index[best_index]
        success_count = len(successful)
        robust_report["success_count"] = int(success_count)
    if success_count < min_successes:
        nominal = next(
            (item for item in variant_results if abs(float(item["offset_y"])) <= 1e-9 and abs(float(item.get("offset_z", 0.0))) <= 1e-9 and abs(float(item.get("yaw_deg", 0.0))) <= 1e-9),
            variant_results[0] if variant_results else {},
        )
        return {
            "success": False,
            "failed_at": "no_robust_grasp_release_pair",
            "selected": nominal.get("selected"),
            "selected_index": nominal.get("selected_index"),
            "selected_score": nominal.get("selected_score"),
            "reports": nominal.get("reports", []),
            "variant_reports": variant_results,
            "robust_actor_to_tcp": robust_report,
        }

    nominal_success = next(
        (
            item
            for item in successful
            if abs(float(item["offset_y"])) <= 1e-9
            and abs(float(item["offset_z"])) <= 1e-9
            and abs(float(item["yaw_deg"])) <= 1e-9
        ),
        None,
    )
    prefer_best_variant = bool(getattr(args, "release_robust_prefer_best_variant", False))
    chosen = min(
        successful,
        key=lambda item: (
            abs(float(item["offset_y"]) - preferred_y) if has_preferred_y else 0.0,
            float(item["result"].get("selected_score", float("inf"))),
            0 if abs(float(item["offset_y"])) <= 1e-9 and abs(float(item["offset_z"])) <= 1e-9 and abs(float(item["yaw_deg"])) <= 1e-9 else 1,
            abs(float(item["offset_y"])),
        ),
    ) if prefer_best_variant else (nominal_success or min(
        successful,
        key=lambda item: (
            float(item["result"].get("selected_score", float("inf"))),
            abs(float(item["offset_y"])),
        ),
    ))
    chosen_result = dict(chosen["result"])
    base_score = float(chosen_result.get("selected_score", 0.0) or 0.0)
    robust_score = base_score + robust_weight * float(max(len(variant_results) - success_count, 0))
    chosen_result["selected_score_nominal"] = base_score
    chosen_result["selected_score"] = float(robust_score)
    chosen_result["robust_actor_to_tcp"] = robust_report
    chosen_result["variant_reports"] = variant_results
    chosen_result["selected_actor_to_tcp_offset_x"] = float(chosen["offset_x"])
    chosen_result["selected_actor_to_tcp_offset_y"] = float(chosen["offset_y"])
    chosen_result["selected_actor_to_tcp_offset_z"] = float(chosen["offset_z"])
    chosen_result["selected_actor_to_tcp_yaw_deg"] = float(chosen["yaw_deg"])
    return chosen_result


def _screen_release_for_actor_to_tcp(
    *,
    planner: RM75CuRoboPlanner,
    base_env: Any,
    locked: dict[str, Any],
    fixtures: list[dict[str, Any]],
    role: str,
    actor_to_tcp: Any,
    target_actor_pose: Any,
    start_q: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    predicted_gate_feasible: list[dict[str, Any]] = []
    base_connection_potential = _wall_connection_potential(base_env, role)
    approach_modes = _wall_release_approach_modes(args, role, base_connection_potential)
    evaluate_all_modes = str(getattr(args, "wall_release_approach_mode", "auto") or "auto").strip().lower() == "both"
    for release_mode in approach_modes:
        if feasible and not evaluate_all_modes and not _needs_fixed_outside_release_search(
            feasible=feasible,
            args=args,
            role=role,
            connection_potential=base_connection_potential,
        ):
            break
        release_args = _args_for_release_mode(args, role, release_mode)
        release_exclude_roles = _release_collision_exclude_roles(release_args, role)
        release_preplace_max_joint_delta = float(getattr(release_args, "release_preplace_max_joint_delta", 0.0))
        release_max_joint_delta = float(getattr(release_args, "release_max_joint_delta", 0.0))
        primary_entries, fallback_entries, fast_report = _fast_chain_preselect_release_candidates(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            role=role,
            actor_to_tcp=actor_to_tcp,
            target_actor_pose=target_actor_pose,
            start_q=start_q,
            args=release_args,
        )
        for pass_name, entries in [("fast_chain", primary_entries), ("fallback", fallback_entries)]:
            if feasible and not evaluate_all_modes and not _needs_fixed_outside_release_search(
                feasible=feasible,
                args=args,
                role=role,
                connection_potential=base_connection_potential,
            ):
                break
            if not entries:
                continue
            for index, label, release_actor_pose, cache in entries:
                target_tcp = release_actor_pose * actor_to_tcp
                preplace, approach_report = _release_preplace_pose_for_candidate(
                    base_env=base_env,
                    role=role,
                    release_actor_pose=release_actor_pose,
                    target_tcp=target_tcp,
                    args=release_args,
                    candidate_label=label,
                )
                pose_error = _pose_to_pose_error(release_actor_pose, target_actor_pose)
                candidate_report: dict[str, Any] = {
                    "index": index,
                    "label": label,
                    "target_pose_error": pose_error,
                    "fast_chain_pass": pass_name,
                    "release_mode": release_mode,
                    "release_approach": approach_report,
                }
                if _fixed_release_too_close_for_single_connection(release_args, approach_report, base_connection_potential):
                    candidate_report["success"] = False
                    candidate_report["failed_at"] = "fixed_single_connection_release_too_close"
                    reports.append(candidate_report)
                    continue
                if cache.get("q_preplace") is not None and cache.get("pre_report") is not None:
                    q_preplace = np.asarray(cache["q_preplace"], dtype=np.float32).reshape(-1)[:7]
                    pre_report = dict(cache["pre_report"])
                    ok = True
                else:
                    ok, q_preplace, pre_report = _solve_ik(
                        planner=planner,
                        base_env=base_env,
                        locked=locked,
                        fixtures=fixtures,
                        exclude_role=role,
                        target_pose=preplace,
                        num_seeds=args.ik_seeds,
                        start_q=start_q,
                    )
                candidate_report["preplace_ik"] = pre_report
                if not ok or q_preplace is None:
                    reports.append(candidate_report)
                    continue
                preplace_joint_delta = float(cache.get("preplace_joint_delta", _joint_distance(q_preplace, start_q)))
                candidate_report["preplace_joint_delta_from_lift"] = float(preplace_joint_delta)
                if release_preplace_max_joint_delta > 0.0 and preplace_joint_delta > release_preplace_max_joint_delta:
                    candidate_report["success"] = False
                    candidate_report["failed_at"] = "preplace_joint_branch_jump"
                    candidate_report["release_preplace_max_joint_delta"] = release_preplace_max_joint_delta
                    reports.append(candidate_report)
                    continue
                used_cached_place_ik = bool(cache.get("q_place") is not None and cache.get("place_report") is not None)
                if used_cached_place_ik:
                    q_place = np.asarray(cache["q_place"], dtype=np.float32).reshape(-1)[:7]
                    place_report = dict(cache["place_report"])
                    ok = True
                else:
                    ok, q_place, place_report = _solve_ik(
                        planner=planner,
                        base_env=base_env,
                        locked=locked,
                        fixtures=fixtures,
                        exclude_role=None,
                        exclude_roles=release_exclude_roles,
                        target_pose=target_tcp,
                        num_seeds=args.ik_seeds,
                        start_q=q_preplace,
                    )
                candidate_report["place_ik"] = place_report
                if not ok or q_place is None:
                    reports.append(candidate_report)
                    continue
                place_joint_delta = float(cache.get("place_joint_delta", _joint_distance(q_place, q_preplace)))
                release_max_joint_delta = float(getattr(release_args, "release_max_joint_delta", 0.0))
                if used_cached_place_ik and release_max_joint_delta > 0.0 and place_joint_delta > release_max_joint_delta:
                    repair_ok, repair_q_place, repair_report = _solve_ik(
                        planner=planner,
                        base_env=base_env,
                        locked=locked,
                        fixtures=fixtures,
                        exclude_role=None,
                        exclude_roles=release_exclude_roles,
                        target_pose=target_tcp,
                        num_seeds=args.ik_seeds,
                        start_q=q_preplace,
                    )
                    repair_delta = (
                        float(_joint_distance(repair_q_place, q_preplace))
                        if repair_ok and repair_q_place is not None
                        else float("inf")
                    )
                    candidate_report["cached_place_ik_branch_repair"] = {
                        "attempted": True,
                        "success": bool(repair_ok and repair_q_place is not None),
                        "cached_joint_delta": float(place_joint_delta),
                        "repaired_joint_delta": float(repair_delta),
                        "ik": repair_report,
                    }
                    if repair_ok and repair_q_place is not None and repair_delta < place_joint_delta:
                        q_place = repair_q_place
                        place_report = repair_report
                        candidate_report["place_ik"] = place_report
                        place_joint_delta = repair_delta
                candidate_report["place_joint_delta_from_preplace"] = float(place_joint_delta)
                branch_motion_min_connection_potential = int(
                    getattr(args, "release_motion_plan_on_branch_jump_min_connection_potential", 0) or 0
                )
                branch_motion_allowed_for_role = (
                    branch_motion_min_connection_potential <= 0
                    or int(base_connection_potential) >= branch_motion_min_connection_potential
                )
                allow_branch_motion_plan = bool(getattr(args, "release_motion_plan_preplace", False)) or (
                    bool(getattr(args, "release_motion_plan_on_branch_jump", False))
                    and branch_motion_allowed_for_role
                )
                if (
                    release_max_joint_delta > 0.0
                    and place_joint_delta > release_max_joint_delta
                    and not allow_branch_motion_plan
                ):
                    candidate_report["success"] = False
                    candidate_report["failed_at"] = "place_joint_branch_jump"
                    candidate_report["release_max_joint_delta"] = release_max_joint_delta
                    reports.append(candidate_report)
                    continue
                requires_release_motion_plan = bool(release_max_joint_delta > 0.0 and place_joint_delta > release_max_joint_delta)
                if requires_release_motion_plan:
                    candidate_report["place_joint_branch_jump_warning"] = True
                    candidate_report["release_max_joint_delta"] = release_max_joint_delta
                single_connection_max_place_joint_delta = float(
                    getattr(args, "release_screen_single_connection_max_place_joint_delta", 0.0) or 0.0
                )
                if (
                    int(base_connection_potential) <= 1
                    and single_connection_max_place_joint_delta > 0.0
                    and place_joint_delta > single_connection_max_place_joint_delta
                ):
                    candidate_report["success"] = False
                    candidate_report["failed_at"] = "single_connection_place_joint_delta"
                    candidate_report["release_screen_single_connection_max_place_joint_delta"] = (
                        single_connection_max_place_joint_delta
                    )
                    reports.append(candidate_report)
                    continue
                predicted_report = _predicted_actor_pose_error_from_tcp_q(
                    planner=planner,
                    base_env=base_env,
                    q=q_place,
                    actor_to_tcp=actor_to_tcp,
                    target_actor_pose=target_actor_pose,
                )
                candidate_report["predicted_actor_pose_after_place"] = predicted_report
                connection_geometry = dict(cache.get("connection_geometry") or {})
                if not connection_geometry:
                    connection_geometry = _release_connection_geometry_report(
                        base_env=base_env,
                        role=role,
                        release_actor_pose=release_actor_pose,
                        args=release_args,
                    )
                candidate_report["connection_geometry"] = connection_geometry
                score = float(
                    cache.get(
                        "score",
                        _release_candidate_score(
                            label=label,
                            pose_error=pose_error,
                            preplace_joint_delta=float(preplace_joint_delta),
                            place_joint_delta=float(place_joint_delta),
                            preplace_joint_weight=float(getattr(release_args, "release_score_preplace_joint_weight", 0.005)),
                            place_joint_weight=float(getattr(release_args, "release_score_place_joint_weight", 0.0015)),
                        ),
                    )
                )
                if "score" not in cache:
                    score += _release_ik_error_penalty(
                        preplace_report=pre_report,
                        place_report=place_report,
                        args=release_args,
                    )
                candidate_report["release_score"] = float(score)
                if _release_prediction_gate_failed(predicted_report, release_args):
                    connection_geometry = dict(cache.get("connection_geometry") or {})
                    if not connection_geometry:
                        connection_geometry = _release_connection_geometry_report(
                            base_env=base_env,
                            role=role,
                            release_actor_pose=release_actor_pose,
                            args=release_args,
                        )
                    candidate_report["connection_geometry"] = connection_geometry
                    score = float(
                        cache.get(
                            "score",
                            _release_candidate_score(
                                label=label,
                                pose_error=pose_error,
                                preplace_joint_delta=float(preplace_joint_delta),
                                place_joint_delta=float(place_joint_delta),
                                preplace_joint_weight=float(getattr(release_args, "release_score_preplace_joint_weight", 0.005)),
                                place_joint_weight=float(getattr(release_args, "release_score_place_joint_weight", 0.0015)),
                            ),
                        )
                    )
                    if "score" not in cache:
                        score += _release_ik_error_penalty(
                            preplace_report=pre_report,
                            place_report=place_report,
                            args=release_args,
                        )
                    candidate_report["release_score"] = float(score)
                    candidate_report["success"] = False
                    candidate_report["failed_at"] = "predicted_actor_pose_gate"
                    candidate_report["release_screen_max_predicted_actor_position_error"] = float(
                        getattr(release_args, "release_screen_max_predicted_actor_position_error", 0.0) or 0.0
                    )
                    candidate_report["release_screen_max_predicted_actor_orientation_error_deg"] = float(
                        getattr(release_args, "release_screen_max_predicted_actor_orientation_error_deg", 0.0) or 0.0
                    )
                    predicted_gate_feasible.append(
                        {
                            "score": float(score),
                            "index": index,
                            "label": label,
                            "preplace": preplace,
                            "release_actor_pose": release_actor_pose,
                            "target_tcp": target_tcp,
                            "q_preplace": q_preplace,
                            "q_place": q_place,
                            "predicted_actor_pose_after_place": predicted_report,
                            "release_mode": release_mode,
                            "release_approach": approach_report,
                            "release_exclude_roles": sorted(release_exclude_roles),
                            "requires_release_motion_plan": requires_release_motion_plan,
                            "connection_geometry": connection_geometry,
                            "connection_rank": tuple(
                                connection_geometry.get("rank", (99, 99, float("inf"), float("inf"), float("inf")))
                            ),
                            "prediction_gate_fallback": True,
                        }
                    )
                    reports.append(candidate_report)
                    continue
                candidate_report["success"] = True
                reports.append(candidate_report)
                feasible.append({"index": index, "label": label, "score": float(score), "release_mode": release_mode, "release_approach": approach_report})
                feasible[-1]["release_actor_pose"] = release_actor_pose
                feasible[-1]["target_tcp"] = target_tcp
                feasible[-1]["q_preplace"] = q_preplace
                feasible[-1]["q_place"] = q_place
                feasible[-1]["predicted_actor_pose_after_place"] = predicted_report
                feasible[-1]["requires_release_motion_plan"] = requires_release_motion_plan
                feasible[-1]["connection_geometry"] = connection_geometry
                feasible[-1]["connection_rank"] = tuple(
                    connection_geometry.get("rank", (99, 99, float("inf"), float("inf"), float("inf")))
                )
    prediction_gate_fallback_min_connection_potential = int(
        getattr(args, "release_prediction_gate_fallback_min_connection_potential", 0) or 0
    )
    prediction_gate_fallback_allowed = (
        prediction_gate_fallback_min_connection_potential <= 0
        or int(base_connection_potential) >= prediction_gate_fallback_min_connection_potential
    )
    if not feasible and predicted_gate_feasible and prediction_gate_fallback_allowed:
        feasible = predicted_gate_feasible
        fast_report["prediction_gate_fallback_used"] = True
        fast_report["prediction_gate_fallback_count"] = len(predicted_gate_feasible)
    if not feasible:
        return {"success": False, "reports": reports, "fast_chain_release_preselect": fast_report}
    forced_index = int(getattr(release_args, "release_candidate_index", -1))
    forced = [item for item in feasible if int(item["index"]) == forced_index]
    selected = forced[0] if forced else min(
        feasible,
        key=lambda item: (
            _fixed_release_selection_rank(
                item=item,
                args=args,
                role=role,
                connection_potential=base_connection_potential,
            ),
            tuple(item.get("connection_rank", (99, 99, float("inf"), float("inf"), float("inf")))),
            _release_candidate_order_rank(args, role, int(item["index"]))
            if bool(getattr(args, "release_prefer_candidate_index_order_for_multi_connection", False))
            and int(base_connection_potential) >= 2
            else 0,
            float(item["score"]),
            int(item["index"]),
        ),
    )
    return {
        "success": True,
        "selected_index": int(selected["index"]),
        "selected": selected["label"],
        "selected_score": float(selected["score"]),
        "forced_release_candidate_index": forced_index if forced else None,
        "selected_release_mode": selected.get("release_mode"),
        "release_approach": selected.get("release_approach", {}),
        "selected_connection_geometry": selected.get("connection_geometry", {}),
        "selected_predicted_actor_pose_after_place": selected.get("predicted_actor_pose_after_place", {}),
        "release_actor_pose": selected.get("release_actor_pose"),
        "target_tcp_pose": selected.get("target_tcp"),
        "q_preplace": selected.get("q_preplace"),
        "q_place": selected.get("q_place"),
        "reports": reports,
        "fast_chain_release_preselect": fast_report,
    }


def _select_release_with_live_rollout_for_role(
    *,
    env: Any,
    planner: RM75CuRoboPlanner,
    locked: dict[str, Any],
    fixtures: list[dict[str, Any]],
    role: str,
    actor_to_tcp: sapien.Pose,
    target_actor_pose: sapien.Pose,
    arrays: dict[str, np.ndarray],
    segments: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    base_env = env.unwrapped
    reports: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    predicted_gate_feasible: list[dict[str, Any]] = []
    base_connection_potential = _wall_connection_potential(base_env, role)
    approach_modes = _wall_release_approach_modes(args, role, base_connection_potential)
    evaluate_all_modes = str(getattr(args, "wall_release_approach_mode", "auto") or "auto").strip().lower() == "both"
    for release_mode in approach_modes:
        if feasible and not evaluate_all_modes and not _needs_fixed_outside_release_search(
            feasible=feasible,
            args=args,
            role=role,
            connection_potential=base_connection_potential,
        ):
            break
        release_args = _args_for_release_mode(args, role, release_mode)
        release_exclude_roles = _release_collision_exclude_roles(release_args, role)
        release_preplace_max_joint_delta = float(getattr(release_args, "release_preplace_max_joint_delta", 0.0))
        release_max_joint_delta = float(getattr(release_args, "release_max_joint_delta", 0.0))
        primary_entries, fallback_entries, fast_report = _fast_chain_preselect_release_candidates(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            role=role,
            actor_to_tcp=actor_to_tcp,
            target_actor_pose=target_actor_pose,
            start_q=_current_q(base_env),
            args=release_args,
        )
        for pass_name, entries in [("fast_chain", primary_entries), ("fallback", fallback_entries)]:
            if feasible and not evaluate_all_modes and not _needs_fixed_outside_release_search(
                feasible=feasible,
                args=args,
                role=role,
                connection_potential=base_connection_potential,
            ):
                break
            if not entries:
                continue
            for index, label, release_actor_pose, cache in entries:
                target_tcp = release_actor_pose * actor_to_tcp
                preplace, approach_report = _release_preplace_pose_for_candidate(
                    base_env=base_env,
                    role=role,
                    release_actor_pose=release_actor_pose,
                    target_tcp=target_tcp,
                    args=release_args,
                    candidate_label=label,
                )
                pose_error = _pose_to_pose_error(release_actor_pose, target_actor_pose)
                candidate_report: dict[str, Any] = {
                    "index": index,
                    "label": label,
                    "release_actor_pose": _pose_to_report(release_actor_pose),
                    "target_pose_error": pose_error,
                    "fast_chain_pass": pass_name,
                    "release_approach": approach_report,
                }
                if _fixed_release_too_close_for_single_connection(release_args, approach_report, base_connection_potential):
                    candidate_report["success"] = False
                    candidate_report["failed_at"] = "fixed_single_connection_release_too_close"
                    reports.append(candidate_report)
                    continue
                if cache.get("q_preplace") is not None and cache.get("pre_report") is not None:
                    q_preplace = np.asarray(cache["q_preplace"], dtype=np.float32).reshape(-1)[:7]
                    pre_report = dict(cache["pre_report"])
                    ok = True
                else:
                    ok, q_preplace, pre_report = _solve_ik(
                        planner=planner,
                        base_env=base_env,
                        locked=locked,
                        fixtures=fixtures,
                        exclude_role=role,
                        target_pose=preplace,
                        num_seeds=args.ik_seeds,
                        start_q=_current_q(base_env),
                    )
                candidate_report["preplace_ik"] = pre_report
                if not ok or q_preplace is None:
                    reports.append(candidate_report)
                    continue
                preplace_joint_delta = float(cache.get("preplace_joint_delta", _joint_distance(q_preplace, _current_q(base_env))))
                candidate_report["preplace_joint_delta_from_current"] = float(preplace_joint_delta)
                if release_preplace_max_joint_delta > 0.0 and preplace_joint_delta > release_preplace_max_joint_delta:
                    candidate_report["success"] = False
                    candidate_report["failed_at"] = "preplace_joint_branch_jump"
                    candidate_report["release_preplace_max_joint_delta"] = release_preplace_max_joint_delta
                    reports.append(candidate_report)
                    continue
                used_cached_place_ik = bool(cache.get("q_place") is not None and cache.get("place_report") is not None)
                if used_cached_place_ik:
                    q_place = np.asarray(cache["q_place"], dtype=np.float32).reshape(-1)[:7]
                    place_report = dict(cache["place_report"])
                    ok = True
                else:
                    ok, q_place, place_report = _solve_ik(
                        planner=planner,
                        base_env=base_env,
                        locked=locked,
                        fixtures=fixtures,
                        exclude_role=None,
                        exclude_roles=release_exclude_roles,
                        target_pose=target_tcp,
                        num_seeds=args.ik_seeds,
                        start_q=q_preplace,
                    )
                candidate_report["release_collision_excluded_roles"] = sorted(release_exclude_roles)
                candidate_report["place_ik"] = place_report
                if not ok or q_place is None:
                    if not bool(getattr(args, "release_motion_plan_preplace", False)):
                        reports.append(candidate_report)
                        continue
                    candidate_report["place_ik_fallback_to_motion_plan"] = True
                    q_place = q_preplace
                place_joint_delta = float(cache.get("place_joint_delta", _joint_distance(q_place, q_preplace)))
                if used_cached_place_ik and release_max_joint_delta > 0.0 and place_joint_delta > release_max_joint_delta:
                    repair_ok, repair_q_place, repair_report = _solve_ik(
                        planner=planner,
                        base_env=base_env,
                        locked=locked,
                        fixtures=fixtures,
                        exclude_role=None,
                        exclude_roles=release_exclude_roles,
                        target_pose=target_tcp,
                        num_seeds=args.ik_seeds,
                        start_q=q_preplace,
                    )
                    repair_delta = (
                        float(_joint_distance(repair_q_place, q_preplace))
                        if repair_ok and repair_q_place is not None
                        else float("inf")
                    )
                    candidate_report["cached_place_ik_branch_repair"] = {
                        "attempted": True,
                        "success": bool(repair_ok and repair_q_place is not None),
                        "cached_joint_delta": float(place_joint_delta),
                        "repaired_joint_delta": float(repair_delta),
                        "ik": repair_report,
                    }
                    if repair_ok and repair_q_place is not None and repair_delta < place_joint_delta:
                        q_place = repair_q_place
                        place_report = repair_report
                        candidate_report["place_ik"] = place_report
                        place_joint_delta = repair_delta
                candidate_report["place_joint_delta_from_preplace"] = float(place_joint_delta)
                branch_motion_min_connection_potential = int(
                    getattr(args, "release_motion_plan_on_branch_jump_min_connection_potential", 0) or 0
                )
                branch_motion_allowed_for_role = (
                    branch_motion_min_connection_potential <= 0
                    or int(base_connection_potential) >= branch_motion_min_connection_potential
                )
                allow_branch_motion_plan = bool(getattr(args, "release_motion_plan_preplace", False)) or (
                    bool(getattr(args, "release_motion_plan_on_branch_jump", False))
                    and branch_motion_allowed_for_role
                )
                if release_max_joint_delta > 0.0 and place_joint_delta > release_max_joint_delta and not allow_branch_motion_plan:
                    candidate_report["success"] = False
                    candidate_report["failed_at"] = "place_joint_branch_jump"
                    candidate_report["release_max_joint_delta"] = release_max_joint_delta
                    reports.append(candidate_report)
                    continue
                requires_release_motion_plan = bool(release_max_joint_delta > 0.0 and place_joint_delta > release_max_joint_delta)
                if requires_release_motion_plan:
                    candidate_report["place_joint_branch_jump_warning"] = True
                    candidate_report["release_max_joint_delta"] = release_max_joint_delta
                single_connection_max_place_joint_delta = float(
                    getattr(args, "release_screen_single_connection_max_place_joint_delta", 0.0) or 0.0
                )
                if (
                    int(base_connection_potential) <= 1
                    and single_connection_max_place_joint_delta > 0.0
                    and place_joint_delta > single_connection_max_place_joint_delta
                ):
                    candidate_report["success"] = False
                    candidate_report["failed_at"] = "single_connection_place_joint_delta"
                    candidate_report["release_screen_single_connection_max_place_joint_delta"] = (
                        single_connection_max_place_joint_delta
                    )
                    reports.append(candidate_report)
                    continue
                predicted_report = _predicted_actor_pose_error_from_tcp_q(
                    planner=planner,
                    base_env=base_env,
                    q=q_place,
                    actor_to_tcp=actor_to_tcp,
                    target_actor_pose=target_actor_pose,
                )
                candidate_report["predicted_actor_pose_after_place"] = predicted_report
                connection_geometry = dict(cache.get("connection_geometry") or {})
                if not connection_geometry:
                    connection_geometry = _release_connection_geometry_report(
                        base_env=base_env,
                        role=role,
                        release_actor_pose=release_actor_pose,
                        args=release_args,
                    )
                candidate_report["connection_geometry"] = connection_geometry
                score = float(
                    cache.get(
                        "score",
                        _release_candidate_score(
                            label=label,
                            pose_error=pose_error,
                            preplace_joint_delta=float(preplace_joint_delta),
                            place_joint_delta=float(place_joint_delta),
                            preplace_joint_weight=float(getattr(release_args, "release_score_preplace_joint_weight", 0.005)),
                            place_joint_weight=float(getattr(release_args, "release_score_place_joint_weight", 0.0015)),
                        ),
                    )
                )
                if "score" not in cache:
                    score += _release_ik_error_penalty(
                        preplace_report=pre_report,
                        place_report=place_report,
                        args=release_args,
                    )
                candidate_report["release_score"] = float(score)
                if _release_prediction_gate_failed(predicted_report, release_args):
                    candidate_report["success"] = False
                    candidate_report["failed_at"] = "predicted_actor_pose_gate"
                    candidate_report["release_screen_max_predicted_actor_position_error"] = float(
                        getattr(release_args, "release_screen_max_predicted_actor_position_error", 0.0) or 0.0
                    )
                    candidate_report["release_screen_max_predicted_actor_orientation_error_deg"] = float(
                        getattr(release_args, "release_screen_max_predicted_actor_orientation_error_deg", 0.0) or 0.0
                    )
                    predicted_gate_feasible.append(
                        {
                            "score": float(score),
                            "index": index,
                            "label": label,
                            "preplace": preplace,
                            "release_actor_pose": release_actor_pose,
                            "target_tcp": target_tcp,
                            "q_preplace": q_preplace,
                            "q_place": q_place,
                            "predicted_actor_pose_after_place": predicted_report,
                            "release_mode": release_mode,
                            "release_approach": approach_report,
                            "release_exclude_roles": sorted(release_exclude_roles),
                            "requires_release_motion_plan": requires_release_motion_plan,
                            "connection_geometry": connection_geometry,
                            "connection_rank": tuple(
                                connection_geometry.get("rank", (99, 99, float("inf"), float("inf"), float("inf")))
                            ),
                            "prediction_gate_fallback": True,
                        }
                    )
                    reports.append(candidate_report)
                    continue
                connection_geometry = dict(cache.get("connection_geometry") or {})
                if not connection_geometry:
                    connection_geometry = _release_connection_geometry_report(
                        base_env=base_env,
                        role=role,
                        release_actor_pose=release_actor_pose,
                        args=release_args,
                    )
                candidate_report["connection_geometry"] = connection_geometry
                score = float(
                    cache.get(
                        "score",
                        _release_candidate_score(
                            label=label,
                            pose_error=pose_error,
                            preplace_joint_delta=float(preplace_joint_delta),
                            place_joint_delta=float(place_joint_delta),
                            preplace_joint_weight=float(getattr(release_args, "release_score_preplace_joint_weight", 0.005)),
                            place_joint_weight=float(getattr(release_args, "release_score_place_joint_weight", 0.0015)),
                        ),
                    )
                )
                if "score" not in cache:
                    score += _release_ik_error_penalty(
                        preplace_report=pre_report,
                        place_report=place_report,
                        args=release_args,
                    )
                candidate_report["release_score"] = float(score)
                candidate_report["success"] = True
                feasible.append(
                    {
                        "score": float(score),
                        "index": index,
                        "label": label,
                        "preplace": preplace,
                        "release_actor_pose": release_actor_pose,
                        "target_tcp": target_tcp,
                        "q_preplace": q_preplace,
                        "q_place": q_place,
                        "predicted_actor_pose_after_place": predicted_report,
                        "release_mode": release_mode,
                        "release_approach": approach_report,
                        "release_exclude_roles": sorted(release_exclude_roles),
                        "requires_release_motion_plan": requires_release_motion_plan,
                        "connection_geometry": connection_geometry,
                        "connection_rank": tuple(
                            connection_geometry.get("rank", (99, 99, float("inf"), float("inf"), float("inf")))
                        ),
                    }
                )
                reports.append(candidate_report)
    prediction_gate_fallback_min_connection_potential = int(
        getattr(args, "release_prediction_gate_fallback_min_connection_potential", 0) or 0
    )
    prediction_gate_fallback_allowed = (
        prediction_gate_fallback_min_connection_potential <= 0
        or int(base_connection_potential) >= prediction_gate_fallback_min_connection_potential
    )
    if not feasible and predicted_gate_feasible and prediction_gate_fallback_allowed:
        feasible = predicted_gate_feasible
        fast_report["prediction_gate_fallback_used"] = True
        fast_report["prediction_gate_fallback_count"] = len(predicted_gate_feasible)
    if not feasible:
        return {"success": False, "reports": reports, "fast_chain_release_preselect": fast_report}
    forced_index = int(getattr(args, "release_candidate_index", -1))
    forced = [item for item in feasible if int(item["index"]) == forced_index]
    ranked_feasible = forced if forced else sorted(
        feasible,
        key=lambda item: (
            _fixed_release_selection_rank(
                item=item,
                args=args,
                role=role,
                connection_potential=base_connection_potential,
            ),
            tuple(item.get("connection_rank", (99, 99, float("inf"), float("inf"), float("inf")))),
            _release_candidate_order_rank(args, role, int(item["index"]))
            if bool(getattr(args, "release_prefer_candidate_index_order_for_multi_connection", False))
            and int(base_connection_potential) >= 2
            else 0,
            float(item["score"]),
            int(item["index"]),
        ),
    )
    selected = ranked_feasible[0]
    preplace_motion_plan_report: dict[str, Any] | None = None
    place_motion_planned = False
    should_motion_plan_release = bool(getattr(args, "release_motion_plan_preplace", False)) or bool(
        selected.get("requires_release_motion_plan")
    )
    if should_motion_plan_release:
        motion_attempts: list[dict[str, Any]] = []
        selected = None
        for item in ranked_feasible:
            motion_started = time.perf_counter()
            ok, q_preplace_path, candidate_motion_report = _plan_motion_to_pose(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                exclude_role=role,
                target_pose=item["preplace"],
                start_q=_current_q(base_env),
                timeout=float(getattr(args, "release_motion_plan_timeout", 4.0)),
                num_seeds=args.ik_seeds,
                enable_graph=False,
                max_attempts=1,
                num_graph_seeds=1,
            )
            candidate_motion_report["elapsed_sec"] = float(time.perf_counter() - motion_started)
            attempt_report = {
                "index": int(item["index"]),
                "label": item["label"],
                "score": float(item["score"]),
                "requires_release_motion_plan": bool(item.get("requires_release_motion_plan")),
                "preplace_motion_plan": candidate_motion_report,
            }
            if not ok or q_preplace_path is None:
                motion_attempts.append(attempt_report)
                continue
            place_motion_started = time.perf_counter()
            ok, q_place_path, place_motion_plan_report = _plan_motion_to_pose(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                exclude_role=None,
                exclude_roles=set(item.get("release_exclude_roles", [])),
                target_pose=item["target_tcp"],
                start_q=q_preplace_path[-1, :7],
                timeout=float(getattr(args, "release_motion_plan_timeout", 4.0)),
                num_seeds=args.ik_seeds,
                enable_graph=False,
                max_attempts=1,
                num_graph_seeds=1,
            )
            place_motion_plan_report["elapsed_sec"] = float(time.perf_counter() - place_motion_started)
            candidate_motion_report["place_motion_plan"] = place_motion_plan_report
            attempt_report["place_motion_plan"] = place_motion_plan_report
            if not ok or q_place_path is None:
                motion_attempts.append(attempt_report)
                continue
            item["q_preplace_path"] = q_preplace_path
            item["q_place_path"] = q_place_path
            selected = item
            preplace_motion_plan_report = candidate_motion_report
            motion_attempts.append(attempt_report)
            break
        if selected is None:
            return {
                "success": False,
                "failed_at": "release_motion_plan",
                "reports": reports,
                "motion_plan_attempts": motion_attempts,
                "fast_chain_release_preselect": fast_report,
            }
        q_preplace_path = selected["q_preplace_path"]
        q_place_path = selected["q_place_path"]
        _record_existing_joint_path(
            env=env,
            arrays=arrays,
            segments=segments,
            name=f"preplace_plan_{int(selected['index']):02d}",
            path=q_preplace_path,
            gripper=CLOSED_GRIPPER,
            final_hold=args.final_hold_steps,
            max_joint_step=float(getattr(args, "max_joint_step", 0.06)),
            max_waypoints=int(getattr(args, "release_motion_plan_max_waypoints", 0)),
        )
        _record_existing_joint_path(
            env=env,
            arrays=arrays,
            segments=segments,
            name=f"place_plan_{int(selected['index']):02d}",
            path=q_place_path,
            gripper=CLOSED_GRIPPER,
            final_hold=args.final_hold_steps,
            max_joint_step=float(getattr(args, "max_joint_step", 0.06)),
            max_waypoints=int(getattr(args, "release_motion_plan_max_waypoints", 0)),
        )
        selected["q_place"] = q_place_path[-1, :7]
        place_motion_planned = True
    else:
        _add_adaptive_joint_segment(
            env=env,
            arrays=arrays,
            segments=segments,
            name=f"preplace_{int(selected['index']):02d}",
            goal_q=np.asarray(selected["q_preplace"], dtype=np.float32),
            gripper=CLOSED_GRIPPER,
            base_steps=args.move_steps,
            action_repeat=args.action_repeat,
            final_hold=args.final_hold_steps,
            args=args,
        )
    if not place_motion_planned:
        _add_adaptive_joint_segment(
            env=env,
            arrays=arrays,
            segments=segments,
            name=f"place_{int(selected['index']):02d}",
            goal_q=np.asarray(selected["q_place"], dtype=np.float32),
            gripper=CLOSED_GRIPPER,
            base_steps=args.release_steps,
            action_repeat=args.action_repeat,
            final_hold=args.final_hold_steps,
            args=args,
        )
    return {
        "success": True,
        "selected": selected["label"],
        "selected_index": selected["index"],
        "selected_score": selected["score"],
        "forced_release_candidate_index": forced_index if forced else None,
        "selected_release_mode": selected.get("release_mode"),
        "release_approach": selected.get("release_approach", {}),
        "selected_connection_geometry": selected.get("connection_geometry", {}),
        "selected_predicted_actor_pose_after_place": selected.get("predicted_actor_pose_after_place", {}),
        "prediction_gate_fallback": bool(selected.get("prediction_gate_fallback", False)),
        "reports": reports,
        "release_actor_pose": _pose_to_report(selected["release_actor_pose"]),
        "target_tcp_pose": selected["target_tcp"],
        "q_preplace": np.asarray(selected["q_preplace"], dtype=np.float32),
        "q_place": np.asarray(selected["q_place"], dtype=np.float32),
        "preplace_motion_plan": preplace_motion_plan_report,
        "motion_plan_attempts": motion_attempts if should_motion_plan_release else [],
        "fast_chain_release_preselect": fast_report,
    }


def _release_candidate_score(
    *,
    label: str,
    pose_error: dict[str, float],
    preplace_joint_delta: float,
    place_joint_delta: float,
    preplace_joint_weight: float = 0.0050,
    place_joint_weight: float = 0.0015,
) -> float:
    """Prefer near-target edge-first releases that stay inside magnetic capture."""
    text = str(label or "")
    lift_mm = 0.0
    normal_mm = 0.0
    tilt_deg = 0.0
    if "generic_edge_lift_" in text:
        try:
            lift_mm = float(text.split("generic_edge_lift_", 1)[1].split("mm", 1)[0])
        except Exception:
            lift_mm = 0.0
        try:
            normal_mm = float(text.split("normal_", 1)[1].split("mm", 1)[0])
        except Exception:
            normal_mm = 0.0
        try:
            tilt_deg = float(text.split("tilt_", 1)[1].split("deg", 1)[0])
        except Exception:
            tilt_deg = 0.0
    exact_penalty = 0.008 if text == "exact_target" else 0.0
    lift_penalty = abs(lift_mm - 0.5) * 0.0008
    normal_penalty = 0.0020 * abs(normal_mm) + 0.0040 * max(abs(normal_mm) - 1.5, 0.0)
    tilt_penalty = abs(abs(tilt_deg) - 6.0) * 0.00020
    if abs(tilt_deg) > 10.0:
        tilt_penalty += (abs(tilt_deg) - 10.0) * 0.001
    openout_mm = 0.0
    if "openout_" in text:
        try:
            openout_mm = float(text.split("openout_", 1)[1].split("mm", 1)[0])
        except Exception:
            openout_mm = 0.0
    openout_penalty = 0.0020 * abs(openout_mm)
    tangent_mm = 0.0
    if "_edge_" in text:
        try:
            tangent_mm = float(text.split("_edge_", 1)[1].split("mm", 1)[0])
        except Exception:
            tangent_mm = 0.0
    tangent_penalty = 0.0015 * abs(tangent_mm)
    yaw_deg = 0.0
    if "_yaw_" in text:
        try:
            yaw_deg = float(text.split("_yaw_", 1)[1].split("deg", 1)[0])
        except Exception:
            yaw_deg = 0.0
    release_yaw_penalty = 0.0008 * abs(yaw_deg)
    joint_penalty = float(preplace_joint_weight) * float(preplace_joint_delta) + float(place_joint_weight) * float(place_joint_delta)
    pose_penalty = float(pose_error["position_error_m"]) + 0.00045 * float(pose_error["orientation_error_deg"])
    return float(
        exact_penalty
        + pose_penalty
        + lift_penalty
        + normal_penalty
        + tilt_penalty
        + openout_penalty
        + tangent_penalty
        + release_yaw_penalty
        + joint_penalty
    )


def _release_ik_error_penalty(
    *,
    preplace_report: dict[str, Any] | None,
    place_report: dict[str, Any] | None,
    args: argparse.Namespace,
) -> float:
    position_weight = float(getattr(args, "release_score_ik_position_weight", 0.25))
    rotation_weight = float(getattr(args, "release_score_ik_rotation_weight", 0.010))

    def one(report: dict[str, Any] | None) -> float:
        if not isinstance(report, dict):
            return 0.0
        debug = report.get("debug", {})
        if not isinstance(debug, dict):
            return 0.0
        score = 0.0
        if debug.get("position_error") is not None:
            score += position_weight * float(debug.get("position_error"))
        if debug.get("rotation_error") is not None:
            score += rotation_weight * float(debug.get("rotation_error"))
        return float(score)

    return one(preplace_report) + one(place_report)


def _fixed_release_kind(item: dict[str, Any]) -> str:
    approach = item.get("release_approach", {})
    if isinstance(approach, dict):
        kind = str(approach.get("kind", "") or "").strip().lower()
        if kind:
            return kind
    label = str(item.get("label", "") or "")
    if label.startswith("fixed_out_"):
        return "outside"
    if label.startswith("fixed_top_"):
        return "top"
    return ""


def _fixed_release_label_openout_mm(item: dict[str, Any]) -> float:
    label = str(item.get("label", "") or "")
    if "openout_" not in label:
        return 0.0
    try:
        return float(label.split("openout_", 1)[1].split("mm", 1)[0])
    except Exception:
        return 0.0


def _fixed_release_label_tangent_mm(item: dict[str, Any]) -> float:
    label = str(item.get("label", "") or "")
    if "_edge_" not in label:
        return 0.0
    try:
        return float(label.split("_edge_", 1)[1].split("mm", 1)[0])
    except Exception:
        return 0.0


def _fixed_release_label_yaw_deg(item: dict[str, Any]) -> float:
    label = str(item.get("label", "") or "")
    if "_yaw_" not in label:
        return 0.0
    try:
        return float(label.split("_yaw_", 1)[1].split("deg", 1)[0])
    except Exception:
        return 0.0


def _fixed_release_too_close_for_single_connection(args: argparse.Namespace, approach_report: dict[str, Any], connection_potential: int) -> bool:
    profile = str(getattr(args, "wall_release_profile", "legacy") or "legacy").strip().lower()
    if profile != "fixed_top_down" or int(connection_potential) > 1:
        return False
    min_clearance = float(getattr(args, "fixed_single_connection_release_min_clearance", 0.0) or 0.0)
    if min_clearance <= 0.0:
        return False
    if str(approach_report.get("kind", "") or "").strip().lower() != "outside":
        return False
    height = float(approach_report.get("height", 0.0) or 0.0)
    retreat = float(approach_report.get("retreat", 0.0) or 0.0)
    return height < min_clearance or retreat < min_clearance


def _prefer_fixed_outside_release(args: argparse.Namespace, role: str, connection_potential: int) -> bool:
    profile = str(getattr(args, "wall_release_profile", "legacy") or "legacy").strip().lower()
    return bool(role in FIRST_LAYER_WALL_ROLES and profile == "fixed_top_down")


def _needs_fixed_outside_release_search(
    *,
    feasible: list[dict[str, Any]],
    args: argparse.Namespace,
    role: str,
    connection_potential: int,
) -> bool:
    if not feasible or not _prefer_fixed_outside_release(args, role, connection_potential):
        return False
    return not any(_fixed_release_kind(item) == "outside" for item in feasible)


def _fixed_release_selection_rank(
    *,
    item: dict[str, Any],
    args: argparse.Namespace,
    role: str,
    connection_potential: int,
) -> int:
    if not _prefer_fixed_outside_release(args, role, connection_potential):
        return 0
    if _fixed_release_kind(item) != "outside":
        return 4
    openout_mm = _fixed_release_label_openout_mm(item)
    tangent_mm = _fixed_release_label_tangent_mm(item)
    yaw_deg = _fixed_release_label_yaw_deg(item)
    if int(connection_potential) <= 1:
        params = _fixed_release_preplace_params(str(item.get("label", "") or ""))
        if not params:
            return 4
        height_mm = 1000.0 * float(params.get("height", 0.0))
        retreat_mm = 1000.0 * float(params.get("retreat", 0.0))
        too_close = height_mm < 30.0 or retreat_mm < 30.0
        if abs(openout_mm) <= 1e-6 and abs(tangent_mm) <= 1e-6 and abs(yaw_deg) <= 1e-6:
            if too_close:
                return 1
            return 0
        if abs(yaw_deg) > 1e-6 and abs(openout_mm) <= 1e-6 and abs(tangent_mm) <= 1e-6:
            return 2
        if not too_close:
            return 3
        return 4
    if abs(openout_mm) <= 1e-6 and abs(tangent_mm) <= 1e-6:
        return 0
    if openout_mm > 1e-6 and abs(tangent_mm) <= 1e-6:
        return 1
    if abs(tangent_mm) > 1e-6:
        return 2
    return 3


def _wall_release_approach_mode(args: argparse.Namespace, role: str, connection_potential: int) -> str:
    mode = str(getattr(args, "wall_release_approach_mode", "auto") or "auto").strip().lower()
    if mode in {"top_down", "side_push", "both"}:
        return mode
    profile = str(getattr(args, "wall_release_profile", "legacy") or "legacy").strip().lower()
    if role in FIRST_LAYER_WALL_ROLES and profile == "fixed_top_down":
        return "top_down"
    if (
        role in FIRST_LAYER_WALL_ROLES
        and profile == "generic"
    ):
        if int(connection_potential) <= 1:
            return "side_push"
        if int(connection_potential) >= 3:
            return "side_push"
        return "top_down"
    return "top_down"


def _wall_release_approach_modes(args: argparse.Namespace, role: str, connection_potential: int) -> list[str]:
    mode = str(getattr(args, "wall_release_approach_mode", "auto") or "auto").strip().lower()
    if mode == "both":
        return ["top_down", "side_push"]
    if mode in {"top_down", "side_push"}:
        return [mode]
    profile = str(getattr(args, "wall_release_profile", "legacy") or "legacy").strip().lower()
    if role in FIRST_LAYER_WALL_ROLES and profile == "fixed_top_down":
        return ["top_down"]
    if (
        role in FIRST_LAYER_WALL_ROLES
        and profile == "generic"
    ):
        if int(connection_potential) <= 1:
            return ["side_push"]
        if int(connection_potential) >= 3:
            return ["side_push", "top_down"]
        return ["top_down", "side_push"]
    return ["top_down"]


def _args_for_release_mode(args: argparse.Namespace, role: str, mode: str) -> argparse.Namespace:
    release_args = _args_for_role(args, role)
    release_args.wall_release_approach_mode = mode
    return release_args


def _release_preplace_pose_for_candidate(
    *,
    base_env: Any,
    role: str,
    release_actor_pose: sapien.Pose,
    target_tcp: sapien.Pose,
    args: argparse.Namespace,
    candidate_label: str = "",
) -> tuple[sapien.Pose, dict[str, Any]]:
    connection_potential = _wall_connection_potential(base_env, role)
    required_active_connections = _required_active_connections_for_role(base_env, role, args)
    mode = _wall_release_approach_mode(args, role, connection_potential)
    target_position, target_quaternion = _pose_arrays(target_tcp)
    fixed_preplace = _fixed_release_preplace_params(candidate_label)
    if fixed_preplace is not None:
        _, release_quaternion = _pose_arrays(release_actor_pose)
        rotation = quat2mat(release_quaternion).astype(np.float32)
        face_normal = rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        face_normal[2] = 0.0
        if float(np.linalg.norm(face_normal)) <= 1e-6:
            face_normal = rotation @ np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            face_normal[2] = 0.0
        face_normal = _normalize_vector(face_normal)
        release_position, _ = _pose_arrays(release_actor_pose)
        outward = np.asarray([release_position[0], release_position[1], 0.0], dtype=np.float32)
        if float(np.linalg.norm(outward)) <= 1e-6:
            outward = face_normal
        outward = _normalize_vector(outward)
        offset = np.asarray([0.0, 0.0, float(fixed_preplace["height"])], dtype=np.float32)
        if str(fixed_preplace["kind"]) == "outside":
            offset = offset + outward * float(fixed_preplace["retreat"])
        return sapien.Pose(p=(target_position + offset).tolist(), q=target_quaternion.tolist()), {
            "mode": "top_down",
            "profile": "fixed_top_down",
            "kind": str(fixed_preplace["kind"]),
            "connection_potential": int(connection_potential),
            "required_active_connections": int(required_active_connections),
            "height": float(fixed_preplace["height"]),
            "retreat": float(fixed_preplace["retreat"]),
            "outward_axis": outward.tolist(),
        }
    if mode == "side_push":
        robot_position, _ = _robot_pose_arrays(base_env)
        approach = np.asarray(robot_position[:2] - target_position[:2], dtype=np.float32)
        approach = np.asarray([approach[0], approach[1], 0.0], dtype=np.float32)
        if float(np.linalg.norm(approach)) <= 1e-6:
            _, release_quaternion = _pose_arrays(release_actor_pose)
            rotation = quat2mat(release_quaternion).astype(np.float32)
            approach = rotation @ np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
            approach[2] = 0.0
        if float(np.linalg.norm(approach)) <= 1e-6:
            approach = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
        approach = _normalize_vector(approach)
        distance = float(getattr(args, "wall_release_side_approach_distance", max(float(getattr(args, "preplace_height", 0.055)), 0.035)))
        lift = float(getattr(args, "wall_release_side_approach_lift", 0.010))
        offset = approach * distance + np.asarray([0.0, 0.0, lift], dtype=np.float32)
        return sapien.Pose(p=(target_position + offset).tolist(), q=target_quaternion.tolist()), {
            "mode": mode,
            "connection_potential": int(connection_potential),
            "required_active_connections": int(required_active_connections),
            "side_distance": float(distance),
            "side_lift": float(lift),
        }
    height = float(getattr(args, "preplace_height", 0.055))
    return _offset_world(target_tcp, np.asarray([0.0, 0.0, height], dtype=np.float32)), {
        "mode": mode,
        "connection_potential": int(connection_potential),
        "required_active_connections": int(required_active_connections),
        "height": float(height),
    }


def _pose_arrays(pose: sapien.Pose) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(pose.p, dtype=np.float32).reshape(-1, 3)[0],
        np.asarray(pose.q, dtype=np.float32).reshape(-1, 4)[0],
    )


def _float_csv(text: str, default: list[float]) -> list[float]:
    values: list[float] = []
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values if values else list(default)


def _release_open_gripper_value(args: argparse.Namespace, role: str) -> float:
    if role == "top_lid" and getattr(args, "top_lid_release_open_gripper_value", None) is not None:
        return float(args.top_lid_release_open_gripper_value)
    return float(getattr(args, "release_open_gripper_value", OPEN_GRIPPER))


def _post_open_pullback_pose(
    *,
    base_env: Any,
    target_tcp_pose: sapien.Pose,
    release_approach: dict[str, Any] | None,
    pullback_distance: float,
    lift_distance: float,
) -> tuple[sapien.Pose, dict[str, Any]]:
    target_position, target_quaternion = _pose_arrays(target_tcp_pose)
    approach = dict(release_approach or {})
    pull_axis = None
    outward = approach.get("outward_axis")
    if isinstance(outward, (list, tuple)) and len(outward) == 3:
        outward_vec = np.asarray(outward, dtype=np.float32)
        if float(np.linalg.norm(outward_vec)) > 1e-6:
            pull_axis = _normalize_vector(outward_vec)
    if pull_axis is None:
        robot_position, _ = _robot_pose_arrays(base_env)
        horizontal = np.asarray(robot_position[:2] - target_position[:2], dtype=np.float32)
        pull_axis = np.asarray([horizontal[0], horizontal[1], 0.0], dtype=np.float32)
        if float(np.linalg.norm(pull_axis)) <= 1e-6:
            pull_axis = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
        pull_axis = _normalize_vector(pull_axis)
    offset = pull_axis * float(pullback_distance) + np.asarray([0.0, 0.0, float(lift_distance)], dtype=np.float32)
    pose = sapien.Pose(p=(target_position + offset).tolist(), q=target_quaternion.tolist())
    return pose, {
        "mode": "release_pullback_lift",
        "pull_axis": pull_axis.tolist(),
        "pullback_distance_m": float(pullback_distance),
        "lift_distance_m": float(lift_distance),
        "target_pose": _pose_to_report(pose),
    }


def _release_candidates_for_role(args: argparse.Namespace, role: str, target_pose: sapien.Pose) -> list[tuple[str, sapien.Pose]]:
    if role != "top_lid" or not str(getattr(args, "top_lid_hinge_tilt_degs", "") or "").strip():
        profile = str(getattr(args, "wall_release_profile", "legacy") or "legacy").strip().lower()
        if (
            role in {"right_wall", "left_wall", "back_wall", "front_wall"}
            and profile == "fixed_top_down"
        ):
            return _fixed_topdown_wall_release_actor_candidates(args, target_pose)
        if (
            role in {"right_wall", "left_wall", "back_wall", "front_wall"}
            and profile == "generic"
        ):
            return _generic_wall_release_actor_candidates(target_pose)
        return _wall_release_actor_candidates(role, target_pose)
    candidates: list[tuple[str, sapien.Pose]] = []
    front_edge_center = np.asarray([0.0, -PLATE_SIZE / 2.0, 0.0], dtype=np.float32)
    front_edge_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    tilts = _float_csv(str(args.top_lid_hinge_tilt_degs), [45.0])
    lifts = _float_csv(str(getattr(args, "top_lid_hinge_lift_mms", "0.0")), [0.0])
    inward_offsets = _float_csv(str(getattr(args, "top_lid_hinge_inward_mms", "0.0")), [0.0])
    x_offsets = _float_csv(str(getattr(args, "top_lid_hinge_x_mms", "0.0")), [0.0])
    for tilt_deg in tilts:
        for lift_mm in lifts:
            for inward_mm in inward_offsets:
                for x_mm in x_offsets:
                    offset = np.asarray([x_mm / 1000.0, inward_mm / 1000.0, lift_mm / 1000.0], dtype=np.float32)
                    candidates.append(
                        (
                            f"top_lid_front_hinge_tilt_{tilt_deg:+.0f}deg_lift_{lift_mm:.1f}mm_y_{inward_mm:+.1f}mm_x_{x_mm:+.1f}mm",
                            _tilt_actor_pose_about_local_axis(
                                target_pose,
                                local_pivot=front_edge_center,
                                local_axis=front_edge_axis,
                                angle_deg=tilt_deg,
                                world_offset=offset,
                            ),
                        )
                    )
    return candidates


def _fixed_topdown_wall_release_actor_candidates(args: argparse.Namespace, target_pose: sapien.Pose) -> list[tuple[str, sapien.Pose]]:
    p, q = _pose_arrays(target_pose)
    open_gap = float(getattr(args, "fixed_top_down_open_gap", 0.0))
    outward = np.asarray([p[0], p[1], 0.0], dtype=np.float32)
    if float(np.linalg.norm(outward)) <= 1e-6:
        outward = np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)
    outward = _normalize_vector(outward)
    samples: list[tuple[str, float, float, float, float, float]] = [
        ("top", 0.0, 25.0, 0.0, 0.0, 0.0),
        ("outside", 35.0, 35.0, 0.0, 0.0, 0.0),
        ("outside", 45.0, 35.0, 0.0, 0.0, 0.0),
        ("outside", 35.0, 25.0, 0.0, 0.0, 0.0),
        ("outside", 35.0, 35.0, 0.0, 0.0, 2.0),
        ("outside", 35.0, 35.0, 0.0, 0.0, -2.0),
        ("top", 0.0, 35.0, 0.0, 0.0, 0.0),
        ("outside", 25.0, 35.0, 0.0, 0.0, 0.0),
        ("outside", 35.0, 45.0, 0.0, 0.0, 0.0),
        ("outside", 45.0, 35.0, 0.0, 0.0, 2.0),
        ("outside", 45.0, 35.0, 0.0, 0.0, -2.0),
        ("outside", 35.0, 35.0, 3.0, 0.0, 0.0),
        ("outside", 35.0, 35.0, -3.0, 0.0, 0.0),
        ("top", 0.0, 45.0, 0.0, 0.0, 0.0),
        ("outside", 45.0, 35.0, 3.0, 0.0, 0.0),
        ("outside", 45.0, 35.0, -3.0, 0.0, 0.0),
        ("outside", 25.0, 25.0, 0.0, 0.0, 0.0),
        ("outside", 35.0, 35.0, 6.0, 0.0, 0.0),
        ("top", 0.0, 55.0, 0.0, 0.0, 0.0),
        ("outside", 45.0, 45.0, 0.0, 0.0, 0.0),
        ("outside", 55.0, 45.0, 0.0, 0.0, 0.0),
        ("outside", 55.0, 35.0, 0.0, 0.0, 0.0),
        ("outside", 35.0, 35.0, 0.0, 0.0, 4.0),
        ("outside", 35.0, 35.0, 0.0, 0.0, -4.0),
        ("outside", 45.0, 45.0, 3.0, 0.0, 0.0),
        ("outside", 45.0, 45.0, -3.0, 0.0, 0.0),
        ("outside", 35.0, 45.0, 3.0, 0.0, 0.0),
        ("outside", 35.0, 45.0, -3.0, 0.0, 0.0),
        ("outside", 55.0, 45.0, 3.0, 0.0, 0.0),
        ("outside", 55.0, 45.0, -3.0, 0.0, 0.0),
        ("outside", 45.0, 35.0, 0.0, 3.0, 0.0),
        ("outside", 45.0, 35.0, 0.0, -3.0, 0.0),
        ("outside", 55.0, 35.0, 0.0, 3.0, 0.0),
        ("outside", 55.0, 35.0, 0.0, -3.0, 0.0),
        ("outside", 45.0, 35.0, 0.0, 6.0, 0.0),
        ("outside", 45.0, 35.0, 0.0, -6.0, 0.0),
    ]
    if not bool(getattr(args, "enable_release_yaw_candidates", False)):
        samples = [item for item in samples if abs(float(item[5])) <= 1e-6]
    candidates: list[tuple[str, sapien.Pose]] = []
    tangent = np.asarray([-outward[1], outward[0], 0.0], dtype=np.float32)
    tangent = _normalize_vector(tangent)
    for index, (kind, retreat_mm, height_mm, open_out_mm, tangent_mm, yaw_deg) in enumerate(samples):
        open_position = (
            p
            + np.asarray([0.0, 0.0, open_gap], dtype=np.float32)
            + outward * (open_out_mm / 1000.0)
            + tangent * (tangent_mm / 1000.0)
        )
        release_q = q
        if abs(yaw_deg) > 1e-6:
            yaw = np.deg2rad(yaw_deg)
            yaw_rotation = np.asarray(
                [
                    [np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw), np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            release_q = mat2quat(yaw_rotation @ quat2mat(q)).astype(np.float32)
        fixed_pose = sapien.Pose(p=open_position.tolist(), q=release_q.tolist())
        tangent_suffix = "" if abs(tangent_mm) <= 1e-6 else f"_edge_{tangent_mm:+.1f}mm"
        yaw_suffix = "" if abs(yaw_deg) <= 1e-6 else f"_yaw_{yaw_deg:+.1f}deg"
        if kind == "top":
            label = f"fixed_top_h{height_mm:.0f}mm_gap_{open_gap * 1000.0:.1f}mm_openout_{open_out_mm:+.1f}mm{tangent_suffix}{yaw_suffix}_{index:02d}"
        else:
            label = (
                f"fixed_out_r{retreat_mm:.0f}mm_h{height_mm:.0f}mm_"
                f"gap_{open_gap * 1000.0:.1f}mm_openout_{open_out_mm:+.1f}mm{tangent_suffix}{yaw_suffix}_{index:02d}"
            )
        candidates.append((label, fixed_pose))
    tilt_degs = [
        value
        for value in _float_csv(str(getattr(args, "fixed_top_down_extra_tilt_degs", "") or ""), [])
        if abs(float(value)) <= abs(float(getattr(args, "fixed_top_down_extra_tilt_max_abs_deg", 30.0) or 30.0))
    ]
    if tilt_degs:
        local_bottom_center = np.asarray([0.0, -PLATE_SIZE / 2.0, 0.0], dtype=np.float32)
        local_bottom_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        lift_mms = _float_csv(str(getattr(args, "fixed_top_down_extra_tilt_lift_mms", "1.0,2.0") or ""), [1.0, 2.0])
        normal_bias_mms = _float_csv(
            str(getattr(args, "fixed_top_down_extra_tilt_normal_bias_mms", "0.0,1.5,-1.5") or ""),
            [0.0, 1.5, -1.5],
        )
        target_rotation = quat2mat(q).astype(np.float32)
        face_normal = _normalize_vector(target_rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
        for lift_mm in lift_mms:
            for normal_bias_mm in normal_bias_mms:
                for tilt_deg in tilt_degs:
                    world_offset = (
                        np.asarray([0.0, 0.0, open_gap + float(lift_mm) / 1000.0], dtype=np.float32)
                        + face_normal * (float(normal_bias_mm) / 1000.0)
                    )
                    label = (
                        f"fixed_top_tilt_lift_{float(lift_mm):.1f}mm_"
                        f"normal_{float(normal_bias_mm):+.1f}mm_tilt_{float(tilt_deg):+.0f}deg"
                    )
                    candidates.append(
                        (
                            label,
                            _tilt_actor_pose_about_local_axis(
                                target_pose,
                                local_pivot=local_bottom_center,
                                local_axis=local_bottom_axis,
                                angle_deg=float(tilt_deg),
                                world_offset=world_offset,
                            ),
                        )
                    )
    return candidates


def _generic_wall_release_actor_candidates(target_pose: sapien.Pose) -> list[tuple[str, sapien.Pose]]:
    candidates: list[tuple[str, sapien.Pose]] = []
    local_bottom_center = np.asarray([0.0, -PLATE_SIZE / 2.0, 0.0], dtype=np.float32)
    local_bottom_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    p, q = _pose_arrays(target_pose)
    rotation = quat2mat(q).astype(np.float32)
    face_normal = _normalize_vector(rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    horizontal_normal = face_normal.copy()
    horizontal_normal[2] = 0.0
    if float(np.linalg.norm(horizontal_normal)) <= 1e-6:
        horizontal_normal = rotation @ np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        horizontal_normal[2] = 0.0
    horizontal_normal = _normalize_vector(horizontal_normal)
    candidates.append(("exact_target", sapien.Pose(p=p.tolist(), q=q.tolist())))
    for bottom_lift_mm in [0.5, 1.0, 2.0, 3.0]:
        for tilt_deg in [0.0, 6.0, -6.0, 10.0, -10.0, 14.0, -14.0]:
            for normal_bias_mm in [0.0, 1.5, -1.5, 3.0, -3.0]:
                world_offset = np.asarray([0.0, 0.0, bottom_lift_mm / 1000.0], dtype=np.float32)
                world_offset = world_offset + horizontal_normal * (normal_bias_mm / 1000.0)
                label = (
                    f"generic_edge_lift_{bottom_lift_mm:.1f}mm_"
                    f"normal_{normal_bias_mm:+.1f}mm_tilt_{tilt_deg:+.0f}deg"
                )
                candidates.append(
                    (
                        label,
                        _tilt_actor_pose_about_local_axis(
                            target_pose,
                            local_pivot=local_bottom_center,
                            local_axis=local_bottom_axis,
                            angle_deg=tilt_deg,
                            world_offset=world_offset,
                        ),
                    )
                )
    return candidates


def _configure_anchor_drive(drive: Any, stiffness: float, damping: float, force_limit: float) -> None:
    drive.set_drive_property_x(float(stiffness), float(damping), float(force_limit))
    drive.set_drive_property_y(float(stiffness), float(damping), float(force_limit))
    drive.set_drive_property_z(float(stiffness), float(damping), float(force_limit))
    for item in getattr(drive, "_objs", []):
        item.set_drive_property_slerp(float(stiffness) * 0.08, float(damping) * 0.08, float(force_limit) * 0.08)


def _set_role_magnets_enabled(base_env: Any, role: str, enabled: bool) -> None:
    snap = base_env.magnetic_snap
    if enabled:
        snap.disabled_roles.discard(role)
        snap.suspended_roles.discard(role)
        return
    snap.disabled_roles.add(role)
    snap.suspended_roles.discard(role)
    for active_connection in snap.active_connections:
        connection = active_connection.connection
        if connection.parent != role and connection.child != role:
            continue
        active_connection.active = False
        for drive in active_connection.drives:
            snap._disable_drive(drive)


def _anchor_floor_for_build(
    *,
    base_env: Any,
    locked: dict[str, Any],
    target_pose: sapien.Pose,
    args: argparse.Namespace,
    log: Any,
) -> dict[str, Any] | None:
    if not bool(getattr(args, "anchor_floor_during_build", False)):
        return None
    floor = locked.get("floor")
    if floor is None:
        return None
    actor = floor.actor
    position, quaternion = _pose_arrays(target_pose)
    builder = base_env.scene.create_actor_builder()
    builder.set_scene_idxs([0])
    builder.initial_pose = sapien.Pose(p=position.tolist(), q=quaternion.tolist())
    anchor = builder.build_kinematic(name="jimu_floor_build_anchor")
    base_env.remove_from_state_dict_registry(anchor)
    drive = base_env.scene.create_drive(anchor, sapien.Pose(), actor, sapien.Pose())
    _configure_anchor_drive(
        drive,
        stiffness=float(getattr(args, "floor_anchor_stiffness", 1200.0)),
        damping=float(getattr(args, "floor_anchor_damping", 90.0)),
        force_limit=float(getattr(args, "floor_anchor_force_limit", 35.0)),
    )
    anchor.set_pose(sapien.Pose(p=position.tolist(), q=quaternion.tolist()))
    log(
        "floor: enabled build anchor "
        f"stiffness={float(getattr(args, 'floor_anchor_stiffness', 1200.0)):.1f} "
        f"damping={float(getattr(args, 'floor_anchor_damping', 90.0)):.1f} "
        f"force_limit={float(getattr(args, 'floor_anchor_force_limit', 35.0)):.1f}"
    )
    return {
        "anchor": anchor,
        "drive": drive,
        "target_pose": {"p": position.tolist(), "q": quaternion.tolist()},
    }


def _build_one_role(
    *,
    env: Any,
    planner: RM75CuRoboPlanner,
    targets: dict[str, Any],
    locked: dict[str, Any],
    stage_targets: dict[str, Any],
    fixtures: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    segments: list[dict[str, Any]],
    role: str,
    has_next_role: bool,
    next_role: str | None,
    completed_roles_before: list[str],
    args: argparse.Namespace,
    log: Any,
    prefetch_manager: _NextRolePrefetchManager | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_env = env.unwrapped
    actor = locked[role].actor
    report: dict[str, Any] = {"role": role, "segments_start": len(segments), "candidate_reports": []}
    loaded_role_freeze_release = _clear_loaded_role_pose_freeze(base_env, role)
    if loaded_role_freeze_release.get("cleared"):
        report["loaded_role_freeze_release_for_active"] = loaded_role_freeze_release
        log(f"{role}: released loaded-state pose freeze for active pickup")
    role_timing: dict[str, Any] = {}
    role_started = time.perf_counter()
    _runtime_collision_begin_role(base_env, args=args, locked=locked, role=role, completed_roles=completed_roles_before)

    def finalize_report() -> dict[str, Any]:
        freeze_release = _clear_active_pregrasp_pose_freeze(base_env, actor)
        if freeze_release.get("enabled"):
            report["active_stage_lock_release_on_finalize"] = freeze_release
        report["runtime_collision"] = _runtime_collision_finish_role(base_env)
        return report

    prefetched_fast_chain: dict[str, Any] | None = prefetch_manager.consume(role) if prefetch_manager is not None else None

    if role in stage_targets:
        report["stage_pose_error_before_pick"] = _pose_error(actor, stage_targets[role])
    if prefetched_fast_chain is not None:
        role_timing["next_cycle_prefetch_wait_sec"] = float(prefetched_fast_chain.get("consume_wait_sec", 0.0) or 0.0)
        report["next_cycle_prefetch_consumed"] = True
        report["next_cycle_prefetch_wait_sec"] = float(role_timing["next_cycle_prefetch_wait_sec"])
    if "floor" in locked and "floor" in targets:
        report["floor_pose_error_before_pick"] = _pose_error(locked["floor"].actor, targets["floor"])
    held_role_magnets_suspended = bool(getattr(args, "disable_held_role_magnets_until_release", False))
    if held_role_magnets_suspended:
        _set_role_magnets_enabled(base_env, role, False)
        log(f"{role}: held-role magnets disabled until release pose")
    active_stage_lock = _set_active_pregrasp_pose_freeze(
        base_env=base_env,
        actor=actor,
        enabled=bool(getattr(args, "lock_active_staged_object_until_grasp", True)) and role in stage_targets,
    )
    report["active_stage_lock_until_grasp"] = active_stage_lock
    if active_stage_lock.get("enabled"):
        log(f"{role}: pose-froze active staged object until gripper close completes")

    selected: dict[str, Any] | None = None
    start_q = _current_q(base_env)
    graph_fallback_only = bool(getattr(args, "pregrasp_graph_fallback_only", False))
    pass_modes = [False]
    if bool(getattr(args, "pregrasp_enable_graph", False)):
        pass_modes = [False, True] if graph_fallback_only else [True]
    candidate_pool = list(_grasp_candidates_for_role(args, role))
    fast_chain_candidate_passes: list[tuple[str, list[Any]]] = [("direct", candidate_pool)]
    fast_chain_grasp_preselect: dict[str, Any] | None = None
    fast_chain_release_screen_by_label: dict[str, dict[str, Any]] = {}
    if prefetched_fast_chain is not None:
        fast_chain_grasp_preselect = dict(prefetched_fast_chain.get("report", {}))
        fast_chain_primary = list(prefetched_fast_chain.get("primary") or [])
        fast_chain_fallback = list(prefetched_fast_chain.get("fallback") or [])
        fast_chain_release_screen_by_label = dict(fast_chain_grasp_preselect.get("release_screen_by_label", {}))
        if fast_chain_grasp_preselect:
            fast_chain_grasp_preselect["prefetched"] = True
            fast_chain_grasp_preselect["prefetched_role"] = role
            report["fast_chain_grasp_preselect"] = fast_chain_grasp_preselect
        if fast_chain_primary:
            fast_chain_candidate_passes = [("fast_chain", list(fast_chain_primary))]
            if fast_chain_fallback:
                fast_chain_candidate_passes.append(("fallback", list(fast_chain_fallback)))
        elif fast_chain_fallback:
            fast_chain_candidate_passes = [("fallback", list(fast_chain_fallback))]
    elif bool(getattr(args, "fast_chain_screening", True)):
        fast_chain_primary, fast_chain_fallback, fast_chain_grasp_preselect = _profile_call(
            role_timing,
            "fast_chain_grasp_preselect_sec",
            _fast_chain_preselect_grasp_candidates,
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            actor=actor,
            role=role,
            candidate_pool=candidate_pool,
            start_q=start_q,
            target_actor_pose=targets.get(role),
            args=args,
        )
        if fast_chain_grasp_preselect is not None:
            report["fast_chain_grasp_preselect"] = fast_chain_grasp_preselect
        fast_chain_release_screen_by_label = (
            dict(fast_chain_grasp_preselect.get("release_screen_by_label", {}))
            if fast_chain_grasp_preselect is not None
            else {}
        )
        if fast_chain_primary:
            fast_chain_candidate_passes = [("fast_chain", list(fast_chain_primary))]
            if fast_chain_fallback:
                fast_chain_candidate_passes.append(("fallback", list(fast_chain_fallback)))
    candidate_screen_started = time.perf_counter()
    candidate_screen_stop_reason = ""
    candidate_screen_release_failures = 0
    candidate_screen_no_robust_failures = 0
    max_release_failures = max(int(getattr(args, "candidate_screen_max_release_failures", 0) or 0), 0)
    max_no_robust_failures = max(int(getattr(args, "candidate_screen_max_no_robust_failures", 0) or 0), 0)
    for pass_label, candidate_subset in fast_chain_candidate_passes:
        if selected is not None or candidate_screen_stop_reason:
            break
        for use_graph in pass_modes:
            if selected is not None or candidate_screen_stop_reason:
                break
            for candidate_index, candidate in enumerate(candidate_subset):
                if candidate_screen_stop_reason:
                    break
                grasp_tcp = _candidate_tcp_for_edge_grasp(actor, candidate)
                nominal_actor_to_tcp = _actor_pose(actor).inv() * grasp_tcp
                pregrasp = _pregrasp_pose_for_candidate(grasp_tcp, candidate)
                candidate_report: dict[str, Any] = {
                    "candidate_index": candidate_index,
                    "candidate": candidate.__dict__,
                    "grasp_tcp_axes": _tcp_axis_report(grasp_tcp),
                    "pregrasp_pass": "graph" if use_graph else "direct",
                    "fast_chain_pass": pass_label,
                }

                log(f"{role}: screening candidate {candidate_index} pass={candidate_report['pregrasp_pass']} {candidate.label}")
                ok, q_path, pre_report = _profile_call(
                    role_timing,
                    "pregrasp_motion_plan_sec",
                    _plan_motion_to_pose,
                    planner=planner,
                    base_env=base_env,
                    locked=locked,
                    fixtures=fixtures,
                    exclude_role=None,
                    target_pose=pregrasp,
                    start_q=start_q,
                    timeout=args.pregrasp_timeout,
                    num_seeds=args.ik_seeds,
                    enable_graph=bool(use_graph),
                    max_attempts=int(getattr(args, "pregrasp_max_attempts", 1)),
                    num_graph_seeds=int(getattr(args, "pregrasp_graph_seeds", 1)),
                )
                candidate_report["pregrasp_plan"] = pre_report
                if not ok or q_path is None:
                    candidate_report["failed_at"] = "pregrasp_motion_plan"
                    report["candidate_reports"].append(candidate_report)
                    continue
                ok, q_grasp, grasp_report = _profile_call(
                    role_timing,
                    "edge_grasp_ik_sec",
                    _solve_ik,
                    planner=planner,
                    base_env=base_env,
                    locked=locked,
                    fixtures=fixtures,
                    exclude_role=role,
                    target_pose=grasp_tcp,
                    num_seeds=args.ik_seeds,
                    start_q=q_path[-1, :7],
                )
                candidate_report["edge_grasp_ik"] = grasp_report
                if not ok or q_grasp is None:
                    candidate_report["failed_at"] = "edge_grasp_ik"
                    report["candidate_reports"].append(candidate_report)
                    continue
                grasp_joint_delta = _joint_distance(q_grasp, q_path[-1, :7])
                candidate_report["grasp_joint_delta_from_pregrasp"] = float(grasp_joint_delta)
                max_grasp_joint_delta = float(getattr(args, "grasp_max_joint_delta", 0.0))
                if max_grasp_joint_delta > 0.0 and grasp_joint_delta > max_grasp_joint_delta:
                    candidate_report["failed_at"] = "edge_grasp_joint_branch_jump"
                    candidate_report["grasp_max_joint_delta"] = max_grasp_joint_delta
                    report["candidate_reports"].append(candidate_report)
                    continue
                lift = type(grasp_tcp)(p=[grasp_tcp.p[0], grasp_tcp.p[1], grasp_tcp.p[2] + args.lift_height], q=grasp_tcp.q)
                ok, q_lift, lift_report = _profile_call(
                    role_timing,
                    "lift_ik_sec",
                    _solve_ik,
                    planner=planner,
                    base_env=base_env,
                    locked=locked,
                    fixtures=fixtures,
                    exclude_role=role,
                    target_pose=lift,
                    num_seeds=args.ik_seeds,
                    start_q=q_grasp,
                )
                candidate_report["lift_ik"] = lift_report
                if not ok or q_lift is None:
                    candidate_report["failed_at"] = "lift_ik"
                    report["candidate_reports"].append(candidate_report)
                    continue
                lift_joint_delta = _joint_distance(q_lift, q_grasp)
                candidate_report["lift_joint_delta_from_grasp"] = float(lift_joint_delta)
                max_lift_joint_delta = float(getattr(args, "lift_max_joint_delta", 0.0))
                if max_lift_joint_delta > 0.0 and lift_joint_delta > max_lift_joint_delta:
                    candidate_report["failed_at"] = "lift_joint_branch_jump"
                    candidate_report["lift_max_joint_delta"] = max_lift_joint_delta
                    report["candidate_reports"].append(candidate_report)
                    continue
                preselected_release_screen = fast_chain_release_screen_by_label.get(str(candidate.label))
                if preselected_release_screen is not None:
                    release_screen = dict(preselected_release_screen)
                elif role == "top_lid" and bool(getattr(args, "defer_top_lid_release_screen", False)):
                    release_screen = {"success": False, "deferred_until_after_lift": True}
                else:
                    release_screen = _profile_call(
                        role_timing,
                        "release_screen_sec",
                        _screen_release_for_grasp,
                        planner=planner,
                        base_env=base_env,
                        locked=locked,
                        fixtures=fixtures,
                        role=role,
                        actor_to_tcp=nominal_actor_to_tcp,
                        target_actor_pose=targets[role],
                        start_q=q_lift,
                        args=args,
                    )
                candidate_report["release_screen"] = release_screen
                if not release_screen.get("success") and not (
                    role == "top_lid" and bool(getattr(args, "defer_top_lid_release_screen", False))
                ):
                    candidate_report["failed_at"] = "release_screen"
                    report["candidate_reports"].append(candidate_report)
                    candidate_screen_release_failures += 1
                    if str((release_screen or {}).get("failed_at", "") or "") == "no_robust_grasp_release_pair":
                        candidate_screen_no_robust_failures += 1
                    if max_no_robust_failures > 0 and candidate_screen_no_robust_failures >= max_no_robust_failures:
                        candidate_screen_stop_reason = "max_no_robust_release_screen_failures"
                    elif max_release_failures > 0 and candidate_screen_release_failures >= max_release_failures:
                        candidate_screen_stop_reason = "max_release_screen_failures"
                    if candidate_screen_stop_reason:
                        report["candidate_screen_stopped_early"] = True
                        report["candidate_screen_stop_reason"] = candidate_screen_stop_reason
                        report["candidate_screen_release_failures"] = int(candidate_screen_release_failures)
                        report["candidate_screen_no_robust_failures"] = int(candidate_screen_no_robust_failures)
                        report["candidate_screen_max_release_failures"] = int(max_release_failures)
                        report["candidate_screen_max_no_robust_failures"] = int(max_no_robust_failures)
                    continue
                if not release_screen.get("success"):
                    candidate_report["release_screen_deferred_until_after_lift"] = True
                else:
                    selected_release_index = release_screen.get("selected_index")
                    selected_release_report = None
                    for item in release_screen.get("reports", []):
                        if int(item.get("index", -999)) == int(selected_release_index):
                            selected_release_report = item
                            break
                    max_screen_preplace_delta = float(getattr(args, "release_screen_max_preplace_joint_delta", 0.0))
                    screen_preplace_delta = (
                        selected_release_report.get("preplace_joint_delta_from_lift")
                        if isinstance(selected_release_report, dict)
                        else None
                    )
                    if (
                        max_screen_preplace_delta > 0.0
                        and screen_preplace_delta is not None
                        and float(screen_preplace_delta) > max_screen_preplace_delta
                    ):
                        candidate_report["failed_at"] = "release_screen_preplace_branch_jump"
                        candidate_report["release_screen_max_preplace_joint_delta"] = max_screen_preplace_delta
                        candidate_report["release_screen_preplace_joint_delta"] = float(screen_preplace_delta)
                        report["candidate_reports"].append(candidate_report)
                        continue
                candidate_report["success"] = True
                report["candidate_reports"].append(candidate_report)
                selected = {
                    "candidate_index": candidate_index,
                    "candidate": candidate,
                    "grasp_tcp": grasp_tcp,
                    "nominal_actor_to_tcp": nominal_actor_to_tcp,
                    "q_path": q_path,
                    "q_grasp": q_grasp,
                    "q_lift": q_lift,
                    "pregrasp_plan": pre_report,
                    "edge_grasp_ik": grasp_report,
                    "lift_ik": lift_report,
                    "pregrasp_pass": candidate_report["pregrasp_pass"],
                    "release_screen": release_screen,
                }
                break

    role_timing["candidate_screen_sec"] = time.perf_counter() - candidate_screen_started
    role_timing["candidate_reports_count"] = float(len(report["candidate_reports"]))
    role_timing["candidate_screen_release_failures"] = float(candidate_screen_release_failures)
    role_timing["candidate_screen_no_robust_failures"] = float(candidate_screen_no_robust_failures)
    if selected is None:
        report["success"] = False
        report["failed_at"] = "candidate_screen"
        robust_failures = [
            item
            for item in report.get("candidate_reports", [])
            if str(item.get("failed_at", "") or "") == "release_screen"
            and str((item.get("release_screen") or {}).get("failed_at", "") or "") == "no_robust_grasp_release_pair"
        ]
        report["failure_classification"] = (
            "no_robust_grasp_release_pair" if robust_failures else "live_release_place_ik_fail"
        )
        role_timing["role_total_sec"] = time.perf_counter() - role_started
        report["timing"] = role_timing
        if profile is not None:
            _accumulate_profile(profile, role_timing)
        return finalize_report()
    candidate = selected["candidate"]
    grasp_tcp = selected["grasp_tcp"]
    nominal_actor_to_tcp = selected["nominal_actor_to_tcp"]
    held_actor_pose_lock_enabled = False
    q_path = selected["q_path"]
    q_grasp = selected["q_grasp"]
    q_lift = selected["q_lift"]
    report["candidate_index"] = int(selected["candidate_index"])
    report["candidate"] = candidate.__dict__
    report["pregrasp_plan"] = selected["pregrasp_plan"]
    report["pregrasp_pass"] = selected.get("pregrasp_pass", "direct")
    report["edge_grasp_ik"] = selected["edge_grasp_ik"]
    report["lift_ik"] = selected["lift_ik"]
    log(f"{role}: selected candidate {report['candidate_index']} {candidate.label}")
    pregrasp_max_waypoints = int(getattr(args, "max_existing_path_waypoints", 0))
    if bool(getattr(args, "free_space_motion_window", False)):
        light_waypoints = int(getattr(args, "free_space_pregrasp_max_waypoints", 0))
        if light_waypoints > 1:
            pregrasp_max_waypoints = light_waypoints if pregrasp_max_waypoints <= 1 else min(pregrasp_max_waypoints, light_waypoints)
        report["free_space_motion_window"] = True
        report["free_space_pregrasp_max_waypoints"] = int(pregrasp_max_waypoints)
    _profile_call(
        role_timing,
        "pregrasp_path_exec_sec",
        _record_existing_joint_path,
        env=env,
        arrays=arrays,
        segments=segments,
        name=f"{role}_pregrasp",
        path=q_path,
        gripper=OPEN_GRIPPER,
        final_hold=args.final_hold_steps,
        max_joint_step=float(getattr(args, "max_joint_step", 0.06)),
        max_waypoints=pregrasp_max_waypoints,
    )

    _profile_call(
        role_timing,
        "edge_grasp_exec_sec",
        _add_adaptive_joint_segment,
        env=env,
        arrays=arrays,
        segments=segments,
        name=f"{role}_edge_grasp",
        goal_q=q_grasp,
        gripper=OPEN_GRIPPER,
        base_steps=args.short_steps,
        action_repeat=args.action_repeat,
        final_hold=args.final_hold_steps,
        args=args,
    )

    _profile_call(role_timing, "close_gripper_sec", _add_hold_segment, env, segments, f"{role}_close_gripper", CLOSED_GRIPPER, args.close_steps)
    report["active_stage_lock_release_after_close"] = _clear_active_pregrasp_pose_freeze(base_env, actor)
    if report["active_stage_lock_release_after_close"].get("cleared"):
        log(f"{role}: released active staged object pose-freeze after gripper close")
    report["grasp_after_close"] = _grasp_report(base_env, actor)
    report["grasp_quality_after_close"] = _grasp_quality_report(
        base_env=base_env,
        actor=actor,
        nominal_actor_to_tcp=nominal_actor_to_tcp,
        grasp_report=report["grasp_after_close"],
        args=args,
    )
    log(f"{role}: grasp after close {report['grasp_after_close']}")
    if not report["grasp_quality_after_close"]["success"]:
        report["success"] = False
        report["failed_at"] = "physical_grasp_check"
        role_timing["role_total_sec"] = time.perf_counter() - role_started
        report["timing"] = role_timing
        if profile is not None:
            _accumulate_profile(profile, role_timing)
        return finalize_report()
    if bool(getattr(args, "lock_held_actor_after_grasp", False)):
        lock_reference = str(getattr(args, "held_actor_lock_reference", "live") or "live").strip().lower()
        locked_actor_to_tcp = nominal_actor_to_tcp if lock_reference == "nominal" else (_actor_pose(actor).inv() * _tcp_pose(base_env))
        realman_module._enable_held_actor_pose_lock(base_env, actor, locked_actor_to_tcp)
        realman_module._apply_held_actor_pose_lock(base_env)
        held_actor_pose_lock_enabled = True
        report["held_actor_pose_lock"] = {
            "enabled": True,
            "stage": "after_close",
            "reference": lock_reference,
            "actor_to_tcp": _pose_to_report(locked_actor_to_tcp),
        }

    _profile_call(
        role_timing,
        "lift_exec_sec",
        _add_adaptive_joint_segment,
        env=env,
        arrays=arrays,
        segments=segments,
        name=f"{role}_lift",
        goal_q=q_lift,
        gripper=CLOSED_GRIPPER,
        base_steps=args.move_steps,
        action_repeat=args.action_repeat,
        final_hold=args.final_hold_steps,
        args=args,
    )
    report["grasp_after_lift"] = _grasp_report(base_env, actor)
    report["grasp_quality_after_lift"] = _grasp_quality_report(
        base_env=base_env,
        actor=actor,
        nominal_actor_to_tcp=nominal_actor_to_tcp,
        grasp_report=report["grasp_after_lift"],
        args=args,
    )
    log(f"{role}: grasp after lift {report['grasp_after_lift']}")
    if not report["grasp_quality_after_lift"]["success"]:
        if held_actor_pose_lock_enabled:
            lock_report = realman_module._disable_held_actor_pose_lock(base_env)
            report["held_actor_pose_lock_released"] = {
                "stage": "lift_quality_failure",
                "step_count": int(lock_report.get("step_count", 0)) if isinstance(lock_report, dict) else 0,
            }
        report["success"] = False
        report["failed_at"] = "physical_grasp_lost_during_lift"
        role_timing["role_total_sec"] = time.perf_counter() - role_started
        report["timing"] = role_timing
        if profile is not None:
            _accumulate_profile(profile, role_timing)
        return finalize_report()

    attached_payload_for_release = False
    if bool(getattr(args, "attach_held_payload_for_release_planning", False)):
        payload_report = _attach_payload_for_planning(planner, base_env, actor)
        attached_payload_for_release = bool(payload_report.get("enabled"))
        report["release_payload_collision"] = payload_report

    release_connection_potential = _wall_connection_potential(base_env, role)
    live_actor_to_tcp_min_connection_potential = int(
        getattr(args, "live_actor_to_tcp_after_lift_min_connection_potential", 0) or 0
    )
    use_live_actor_to_tcp = bool(getattr(args, "use_live_actor_to_tcp_after_lift", True)) and (
        live_actor_to_tcp_min_connection_potential <= 0
        or int(release_connection_potential) >= live_actor_to_tcp_min_connection_potential
    )
    if use_live_actor_to_tcp:
        actor_to_tcp = _actor_pose(actor).inv() * _tcp_pose(base_env)
        report["live_actor_to_tcp_after_lift"] = True
    else:
        actor_to_tcp = nominal_actor_to_tcp
        report["live_actor_to_tcp_after_lift"] = False
    report["live_actor_to_tcp_after_lift_min_connection_potential"] = int(live_actor_to_tcp_min_connection_potential)
    required_active_connections = _required_active_connections_for_role(base_env, role, args)
    base_env.magnetic_snap.desired_active_connections_by_role[role] = required_active_connections
    report["release_connection_potential"] = int(release_connection_potential)
    report["required_active_connections_for_role"] = int(required_active_connections)
    log(f"{role}: planning and executing release")
    release_args = _args_for_role(args, role)
    cached_screen = selected.get("release_screen") if isinstance(selected.get("release_screen"), dict) else None
    cached_q_preplace = cached_screen.get("q_preplace") if cached_screen is not None else None
    cached_q_place = cached_screen.get("q_place") if cached_screen is not None else None
    if (
        bool(getattr(args, "use_cached_pair_release", False))
        and cached_screen is not None
        and cached_screen.get("success")
        and cached_q_preplace is not None
        and cached_q_place is not None
    ):
        _profile_call(
            role_timing,
            "release_exec_sec",
            _add_adaptive_joint_segment,
            env=env,
            arrays=arrays,
            segments=segments,
            name=f"preplace_pair_{int(cached_screen.get('selected_index', -1)):02d}",
            goal_q=np.asarray(cached_q_preplace, dtype=np.float32),
            gripper=CLOSED_GRIPPER,
            base_steps=args.move_steps,
            action_repeat=args.action_repeat,
            final_hold=args.final_hold_steps,
            args=args,
        )
        _profile_call(
            role_timing,
            "release_exec_sec",
            _add_adaptive_joint_segment,
            env=env,
            arrays=arrays,
            segments=segments,
            name=f"place_pair_{int(cached_screen.get('selected_index', -1)):02d}",
            goal_q=np.asarray(cached_q_place, dtype=np.float32),
            gripper=CLOSED_GRIPPER,
            base_steps=args.release_steps,
            action_repeat=args.action_repeat,
            final_hold=args.final_hold_steps,
            args=args,
        )
        release_report = {
            "success": True,
            "selected": cached_screen.get("selected"),
            "selected_index": cached_screen.get("selected_index"),
            "selected_score": cached_screen.get("selected_score"),
            "selected_release_mode": cached_screen.get("selected_release_mode"),
            "release_approach": cached_screen.get("release_approach", {}),
            "cached_pair_primary": True,
        }
    else:
        release_report = _profile_call(
            role_timing,
            "release_exec_sec",
            _select_release_with_live_rollout_for_role,
            env=env,
            planner=planner,
            locked=locked,
            fixtures=fixtures,
            role=role,
            actor_to_tcp=actor_to_tcp,
            target_actor_pose=targets[role],
            arrays=arrays,
            segments=segments,
            args=release_args,
        )
    report["release"] = release_report
    if not release_report.get("success"):
        nominal_fallback_max_connection_potential = int(
            getattr(args, "allow_nominal_release_fallback_max_connection_potential", 0) or 0
        )
        nominal_fallback_allowed = bool(getattr(args, "allow_nominal_release_fallback", False)) and (
            nominal_fallback_max_connection_potential <= 0
            or int(release_connection_potential) <= nominal_fallback_max_connection_potential
        )
        if bool(getattr(args, "use_live_actor_to_tcp_after_lift", True)) and nominal_fallback_allowed:
            nominal_release_report = _profile_call(
                role_timing,
                "release_exec_sec",
                _select_release_with_live_rollout_for_role,
                env=env,
                planner=planner,
                locked=locked,
                fixtures=fixtures,
                role=role,
                actor_to_tcp=nominal_actor_to_tcp,
                target_actor_pose=targets[role],
                arrays=arrays,
                segments=segments,
                args=release_args,
            )
            report["release_live_actor_to_tcp_failed"] = release_report
            report["release_nominal_actor_to_tcp_fallback"] = nominal_release_report
            report["allow_nominal_release_fallback_max_connection_potential"] = int(
                nominal_fallback_max_connection_potential
            )
            release_report = nominal_release_report
            report["release"] = release_report
        if release_report.get("success"):
            report["release_used_nominal_actor_to_tcp_fallback"] = True
        elif bool(getattr(args, "use_cached_pair_release", False)) and isinstance(selected.get("release_screen"), dict):
            cached_screen = selected["release_screen"]
            cached_q_preplace = cached_screen.get("q_preplace")
            cached_q_place = cached_screen.get("q_place")
            if cached_screen.get("success") and cached_q_preplace is not None and cached_q_place is not None:
                _profile_call(
                    role_timing,
                    "release_exec_sec",
                    _add_adaptive_joint_segment,
                    env=env,
                    arrays=arrays,
                    segments=segments,
                    name=f"preplace_cached_{int(cached_screen.get('selected_index', -1)):02d}",
                    goal_q=np.asarray(cached_q_preplace, dtype=np.float32),
                    gripper=CLOSED_GRIPPER,
                    base_steps=args.move_steps,
                    action_repeat=args.action_repeat,
                    final_hold=args.final_hold_steps,
                    args=args,
                )
                _profile_call(
                    role_timing,
                    "release_exec_sec",
                    _add_adaptive_joint_segment,
                    env=env,
                    arrays=arrays,
                    segments=segments,
                    name=f"place_cached_{int(cached_screen.get('selected_index', -1)):02d}",
                    goal_q=np.asarray(cached_q_place, dtype=np.float32),
                    gripper=CLOSED_GRIPPER,
                    base_steps=args.release_steps,
                    action_repeat=args.action_repeat,
                    final_hold=args.final_hold_steps,
                    args=args,
                )
                release_report = {
                    "success": True,
                    "selected": cached_screen.get("selected"),
                    "selected_index": cached_screen.get("selected_index"),
                    "selected_score": cached_screen.get("selected_score"),
                    "selected_release_mode": cached_screen.get("selected_release_mode"),
                    "release_approach": cached_screen.get("release_approach", {}),
                    "cached_pair_replay": True,
                    "live_release_failed": report.get("release_live_actor_to_tcp_failed"),
                    "nominal_release_failed": report.get("release_nominal_actor_to_tcp_fallback"),
                }
                report["release_cached_pair_fallback"] = release_report
                report["release"] = release_report
    if not release_report.get("success"):
        if held_actor_pose_lock_enabled:
            report["held_actor_pose_lock_released"] = {
                "stage": "kept_locked_on_release_failure",
                "step_count": int(
                    getattr(base_env, "_held_actor_pose_lock", {}).get("step_count", 0)
                    if isinstance(getattr(base_env, "_held_actor_pose_lock", None), dict)
                    else 0
                ),
            }
            report["held_actor_pose_lock_still_active"] = True
        if attached_payload_for_release:
            planner.detach_object_from_robot()
        report["success"] = False
        report["failed_at"] = "release"
        release_failed_at = str(release_report.get("failed_at", "") or "")
        if release_failed_at == "no_robust_grasp_release_pair":
            report["failure_classification"] = "no_robust_grasp_release_pair"
        else:
            report["failure_classification"] = "live_release_place_ik_fail"
        role_timing["role_total_sec"] = time.perf_counter() - role_started
        report["timing"] = role_timing
        if profile is not None:
            _accumulate_profile(profile, role_timing)
        return finalize_report()
    report["pose_error_after_initial_release_move"] = _pose_error(actor, targets[role])
    max_recoverable_position_error = float(getattr(args, "max_recoverable_release_position_error", 0.18))
    max_recoverable_orientation_error = float(getattr(args, "max_recoverable_release_orientation_error_deg", 90.0))
    if (
        report["pose_error_after_initial_release_move"]["position_error_m"] > max_recoverable_position_error
        or report["pose_error_after_initial_release_move"]["orientation_error_deg"] > max_recoverable_orientation_error
    ):
        if held_actor_pose_lock_enabled:
            lock_report = realman_module._disable_held_actor_pose_lock(base_env)
            held_actor_pose_lock_enabled = False
            report["held_actor_pose_lock_released"] = {
                "stage": "actor_state_invalid_after_initial_release",
                "step_count": int(lock_report.get("step_count", 0)) if isinstance(lock_report, dict) else 0,
            }
        report["success"] = False
        report["failed_at"] = "actor_state_invalid_after_initial_release"
        report["failure_classification"] = "actor_state_invalid_after_release"
        role_timing["role_total_sec"] = time.perf_counter() - role_started
        report["timing"] = role_timing
        if profile is not None:
            _accumulate_profile(profile, role_timing)
        return finalize_report()
    release_corrections: list[dict[str, Any]] = []
    for correction_index in range(max(int(getattr(args, "release_correction_attempts", 0)), 0)):
        current_error = _pose_error(actor, targets[role])
        if (
            current_error["position_error_m"] <= float(getattr(args, "release_correction_position_threshold", 0.008))
            and current_error["orientation_error_deg"] <= float(getattr(args, "release_correction_orientation_threshold_deg", 8.0))
        ):
            break
        live_actor_to_tcp = _actor_pose(actor).inv() * _tcp_pose(base_env)
        desired_tcp = targets[role] * live_actor_to_tcp
        correction_args = _args_for_role(args, role)
        ok, q_correction, correction_ik = _profile_call(
            role_timing,
            "release_correction_ik_sec",
            _solve_ik,
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            exclude_role=None,
            exclude_roles=_release_collision_exclude_roles(correction_args, role),
            target_pose=desired_tcp,
            num_seeds=args.ik_seeds,
            start_q=_current_q(base_env),
        )
        correction_report: dict[str, Any] = {
            "attempt": correction_index + 1,
            "type": "closed_gripper_relative_servo",
            "pose_error_before_correction": current_error,
            "ik": correction_ik,
            "success": bool(ok and q_correction is not None),
        }
        if ok and q_correction is not None:
            _profile_call(
                role_timing,
                "release_correction_exec_sec",
                _add_adaptive_joint_segment,
                env=env,
                arrays=arrays,
                segments=segments,
                name=f"{role}_release_servo_correction_{correction_index + 1}",
                goal_q=q_correction,
                gripper=CLOSED_GRIPPER,
                base_steps=int(getattr(args, "release_correction_steps", 8)),
                action_repeat=args.action_repeat,
                final_hold=args.final_hold_steps,
                args=args,
            )
        correction_report["pose_error_after_correction"] = _pose_error(actor, targets[role])
        release_corrections.append(correction_report)
        log(
            f"{role}: release correction {correction_index + 1} "
            f"before={current_error} after={correction_report['pose_error_after_correction']}"
        )
        if (
            correction_report["pose_error_after_correction"]["position_error_m"] > max_recoverable_position_error
            or correction_report["pose_error_after_correction"]["orientation_error_deg"] > max_recoverable_orientation_error
            or correction_report["pose_error_after_correction"]["position_error_m"]
            > current_error["position_error_m"] + float(getattr(args, "release_correction_revert_tolerance", 0.01))
        ):
            correction_report["success"] = False
            correction_report["stopped"] = "correction_worsened_or_left_recoverable_range"
            break
        if not correction_report["success"]:
            break
    report["release_corrections"] = release_corrections

    def enable_role_capture(phase: str) -> None:
        if role == "top_lid":
            snap = base_env.magnetic_snap
            if float(getattr(args, "top_lid_attach_distance", 0.0)) > 0.0:
                snap.attach_distance = float(args.top_lid_attach_distance)
            if float(getattr(args, "top_lid_attract_distance", 0.0)) > 0.0:
                snap.attract_distance = float(args.top_lid_attract_distance)
            if float(getattr(args, "top_lid_detach_distance", 0.0)) > 0.0:
                snap.detach_distance = float(args.top_lid_detach_distance)
            if float(getattr(args, "top_lid_normal_torque_stiffness", 0.0)) > 0.0:
                snap.attract_normal_torque_stiffness = float(args.top_lid_normal_torque_stiffness)
            if float(getattr(args, "top_lid_normal_torque_limit", 0.0)) > 0.0:
                snap.attract_normal_torque_limit = float(args.top_lid_normal_torque_limit)
            snap.drive_angular_stiffness = float(getattr(args, "top_lid_drive_angular_stiffness", 0.0))
            snap.drive_angular_damping = float(getattr(args, "top_lid_drive_angular_damping", 0.0))
            snap.drive_angular_force_limit = float(getattr(args, "top_lid_drive_angular_force_limit", 0.0))
            desired_connections = int(getattr(args, "top_lid_desired_active_connections", 0))
            if desired_connections > 0:
                snap.desired_active_connections_by_role["top_lid"] = desired_connections
            log(
                f"top_lid: capture distances phase={phase} "
                f"attach={snap.attach_distance:.3f} attract={snap.attract_distance:.3f} detach={snap.detach_distance:.3f}"
            )
            edge_filter = str(getattr(args, "top_lid_release_edge", "") or "").strip()
            target_role_filter = str(getattr(args, "top_lid_release_target_role", "") or "").strip()
            target_edge_filter = str(getattr(args, "top_lid_release_target_edge", "") or "").strip()
            if phase == "after_open" and edge_filter:
                snap.allowed_edges_by_role["top_lid"] = {edge_filter}
                allowed_keys = set()
                for connection in snap.connections:
                    if connection.parent != "top_lid" and connection.child != "top_lid":
                        continue
                    top_edge = connection.parent_edge if connection.parent == "top_lid" else connection.child_edge
                    other_role = connection.child if connection.parent == "top_lid" else connection.parent
                    other_edge = connection.child_edge if connection.parent == "top_lid" else connection.parent_edge
                    if top_edge != edge_filter:
                        continue
                    if target_role_filter and other_role != target_role_filter:
                        continue
                    if target_edge_filter and other_edge != target_edge_filter:
                        continue
                    allowed_keys.add(snap._connection_key(connection))
                snap.allowed_connection_keys = allowed_keys or None
                if not hasattr(snap, "single_edge_supported_roles"):
                    snap.single_edge_supported_roles = set()
                snap.single_edge_supported_roles.add("top_lid")
                log(
                    "top_lid: first-stage connection gate "
                    f"edge={edge_filter} target_role={target_role_filter or '*'} "
                    f"target_edge={target_edge_filter or '*'} allowed_pairs={len(allowed_keys)}"
                )
            elif phase == "after_hinge":
                snap.allowed_edges_by_role.pop("top_lid", None)
                snap.allowed_connection_keys = None
            force_edge_now = (
                bool(getattr(args, "force_top_lid_connections_at_release", False))
                and phase == "after_open"
            )
            if force_edge_now:
                locked_by_role = {item.role: item for item in snap.locked_panel_poses}
                forced_count = 0
                skipped_far_count = 0
                force_max_error = float(getattr(args, "force_top_lid_max_point_error", 0.0))
                for connection in list(snap.connections):
                    if connection.parent != "top_lid" and connection.child != "top_lid":
                        continue
                    if edge_filter:
                        top_edge = connection.parent_edge if connection.parent == "top_lid" else connection.child_edge
                        if top_edge != edge_filter:
                            continue
                    if target_role_filter:
                        other_role = connection.child if connection.parent == "top_lid" else connection.parent
                        if other_role != target_role_filter:
                            continue
                    if target_edge_filter:
                        other_edge = connection.child_edge if connection.parent == "top_lid" else connection.parent_edge
                        if other_edge != target_edge_filter:
                            continue
                    parent = locked_by_role.get(connection.parent)
                    child = locked_by_role.get(connection.child)
                    if parent is None or child is None:
                        continue
                    if snap._find_active_connection(connection) is not None:
                        continue
                    point_error = float(snap._connection_point_error(connection))
                    if force_max_error > 0.0 and point_error > force_max_error:
                        skipped_far_count += 1
                        continue
                    snap._create_runtime_edge_connection(base_env.scene, parent, child, connection)
                    forced_count += 1
                snap.suspended_roles.discard("top_lid")
                log(
                    f"top_lid: forced magnetic edge connections phase={phase} "
                    f"count={forced_count} skipped_far={skipped_far_count} max_error={force_max_error:.3f}"
                )
        _set_role_magnets_enabled(base_env, role, True)
        log(f"{role}: held-role magnets re-enabled phase={phase}")

    capture_after_open_roles = {
        item.strip()
        for item in str(getattr(args, "enable_capture_after_open_roles", "") or "").split(",")
        if item.strip()
    }
    enable_after_open = (
        (role == "top_lid" and bool(getattr(args, "enable_top_lid_capture_after_open", False)))
        or "all" in capture_after_open_roles
        or role in capture_after_open_roles
        or (
            int(getattr(args, "capture_after_open_min_connection_potential", 0)) > 0
            and int(release_connection_potential) >= int(getattr(args, "capture_after_open_min_connection_potential", 0))
        )
    )
    if held_role_magnets_suspended and not enable_after_open:
        _profile_call(role_timing, "capture_before_open_sec", enable_role_capture, "before_open")
    _profile_call(
        role_timing,
        "pre_open_hold_sec",
        _add_hold_segment,
        env,
        segments,
        f"{role}_pre_open_magnetic_hold",
        CLOSED_GRIPPER,
        0 if enable_after_open else getattr(args, "pre_open_hold_steps", 0),
    )
    report["snap_report_before_open"] = base_env.get_magnetic_snap_report()
    report["pose_error_before_open"] = _pose_error(actor, targets[role])
    pre_open_connection_corrections: list[dict[str, Any]] = []
    if not enable_after_open:
        min_pre_open_connections = int(getattr(args, "pre_open_min_active_connections", 1))
        max_correction_pose_error = float(getattr(args, "pre_open_connection_correction_max_pose_error", 0.08))
        max_correction_orientation_error = float(getattr(args, "pre_open_connection_correction_max_orientation_error_deg", 70.0))
        trigger_pose_error = float(getattr(args, "pre_open_connection_correction_trigger_pose_error", 0.012))
        trigger_orientation_error = float(
            getattr(args, "pre_open_connection_correction_trigger_orientation_error_deg", 6.0)
        )
        pre_open_connection_corrections_started = time.perf_counter()
        for correction_index in range(max(int(getattr(args, "pre_open_connection_correction_attempts", 0)), 0)):
            active_count = int(_active_connection_count_for_role(base_env, role))
            current_error = _pose_error(actor, targets[role])
            if active_count >= min_pre_open_connections:
                break
            correction_report: dict[str, Any] = {
                "attempt": correction_index + 1,
                "active_connections_before": active_count,
                "pose_error_before": current_error,
            }
            if (
                active_count >= min_pre_open_connections
                and current_error["position_error_m"] <= trigger_pose_error
                and current_error["orientation_error_deg"] <= trigger_orientation_error
            ):
                correction_report["success"] = False
                correction_report["skipped"] = "pose_error_already_settled_for_open"
                pre_open_connection_corrections.append(correction_report)
                break
            if (
                current_error["position_error_m"] > max_correction_pose_error
                or current_error["orientation_error_deg"] > max_correction_orientation_error
            ):
                correction_report["success"] = False
                correction_report["skipped"] = "pose_error_too_large_for_closed_loop_correction"
                pre_open_connection_corrections.append(correction_report)
                break
            live_actor_to_tcp = _actor_pose(actor).inv() * _tcp_pose(base_env)
            desired_tcp = targets[role] * live_actor_to_tcp
            correction_args = _args_for_role(args, role)
            ok, q_correction, correction_ik = _profile_call(
                role_timing,
                "pre_open_connection_correction_ik_sec",
                _solve_ik,
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                exclude_role=None,
                exclude_roles=_release_collision_exclude_roles(correction_args, role),
                target_pose=desired_tcp,
                num_seeds=args.ik_seeds,
                start_q=_current_q(base_env),
            )
            correction_report["ik"] = correction_ik
            correction_report["success"] = bool(ok and q_correction is not None)
            if ok and q_correction is not None:
                correction_joint_delta = float(_joint_distance(q_correction, _current_q(base_env)))
                correction_report["joint_delta_from_current"] = correction_joint_delta
                max_correction_joint_delta = float(
                    getattr(args, "pre_open_connection_correction_max_joint_delta", 1.4)
                )
                if max_correction_joint_delta > 0.0 and correction_joint_delta > max_correction_joint_delta:
                    correction_report["success"] = False
                    correction_report["skipped"] = "joint_delta_too_large_for_closed_loop_correction"
                    correction_report["max_joint_delta"] = max_correction_joint_delta
                    pre_open_connection_corrections.append(correction_report)
                    break
                _profile_call(
                    role_timing,
                    "pre_open_connection_correction_exec_sec",
                    _add_adaptive_joint_segment,
                    env=env,
                    arrays=arrays,
                    segments=segments,
                    name=f"{role}_pre_open_connection_correction_{correction_index + 1}",
                    goal_q=q_correction,
                    gripper=CLOSED_GRIPPER,
                    base_steps=int(getattr(args, "pre_open_connection_correction_steps", 12)),
                    action_repeat=args.action_repeat,
                    final_hold=args.final_hold_steps,
                    args=args,
                )
                _profile_call(
                    role_timing,
                    "pre_open_connection_correction_hold_sec",
                    _add_hold_segment,
                    env,
                    segments,
                    f"{role}_pre_open_connection_correction_hold_{correction_index + 1}",
                    CLOSED_GRIPPER,
                    int(getattr(args, "pre_open_connection_correction_hold_steps", 20)),
                )
            correction_report["active_connections_after"] = int(_active_connection_count_for_role(base_env, role))
            correction_report["pose_error_after"] = _pose_error(actor, targets[role])
            pre_open_connection_corrections.append(correction_report)
            log(
                f"{role}: pre-open connection correction {correction_index + 1} "
                f"active_before={active_count} active_after={correction_report['active_connections_after']} "
                f"pose_before={current_error} pose_after={correction_report['pose_error_after']}"
            )
            if not correction_report["success"]:
                break
        role_timing["pre_open_connection_corrections_sec"] = time.perf_counter() - pre_open_connection_corrections_started
    report["pre_open_connection_corrections"] = pre_open_connection_corrections
    report["active_connection_count_before_open"] = int(_active_connection_count_for_role(base_env, role))
    if bool(getattr(args, "require_active_connection_before_open", False)):
        required_before_open = int(report.get("required_active_connections_for_role", args.min_active_connections))
        if report["active_connection_count_before_open"] < required_before_open:
            if held_actor_pose_lock_enabled:
                lock_report = realman_module._disable_held_actor_pose_lock(base_env)
                held_actor_pose_lock_enabled = False
                report["held_actor_pose_lock_released"] = {
                    "stage": "pre_open_active_connection_gate",
                    "step_count": int(lock_report.get("step_count", 0)) if isinstance(lock_report, dict) else 0,
                }
            report["success"] = False
            report["failed_at"] = "pre_open_active_connection_gate"
            report["failure_classification"] = "missing_connection_after_open"
            report["pre_open_active_connection_gate"] = {
                "active_connection_count_before_open": int(report["active_connection_count_before_open"]),
                "required_active_connections_for_role": required_before_open,
            }
            role_timing["role_total_sec"] = time.perf_counter() - role_started
            report["timing"] = role_timing
            if profile is not None:
                _accumulate_profile(profile, role_timing)
            return finalize_report()
    release_open_gripper = _release_open_gripper_value(args, role)
    report["release_open_gripper_value"] = release_open_gripper
    unlock_stage = str(getattr(args, "held_actor_unlock_stage", "before_open_gripper") or "before_open_gripper").strip().lower()
    unlock_after_open = unlock_stage == "after_open_gripper"
    if held_actor_pose_lock_enabled and not unlock_after_open:
        lock_report = realman_module._disable_held_actor_pose_lock(base_env)
        held_actor_pose_lock_enabled = False
        report["held_actor_pose_lock_released"] = {
            "stage": "before_open_gripper",
            "step_count": int(lock_report.get("step_count", 0)) if isinstance(lock_report, dict) else 0,
        }
    _profile_call(
        role_timing,
        "open_gripper_sec",
        _add_ramp_segment,
        env,
        segments,
        f"{role}_open_gripper",
        CLOSED_GRIPPER,
        release_open_gripper,
        args.open_steps,
    )
    if held_actor_pose_lock_enabled and unlock_after_open:
        lock_report = realman_module._disable_held_actor_pose_lock(base_env)
        held_actor_pose_lock_enabled = False
        report["held_actor_pose_lock_released"] = {
            "stage": "after_open_gripper",
            "step_count": int(lock_report.get("step_count", 0)) if isinstance(lock_report, dict) else 0,
        }
    if attached_payload_for_release:
        planner.detach_object_from_robot()
    if held_role_magnets_suspended and enable_after_open:
        _profile_call(role_timing, "capture_after_open_sec", enable_role_capture, "after_open")
        _profile_call(
            role_timing,
            "post_open_hold_sec",
            _add_hold_segment,
            env,
            segments,
            f"{role}_post_open_magnetic_hold",
            release_open_gripper,
            getattr(args, "pre_open_hold_steps", 0),
        )
        report["snap_report_after_open_capture"] = base_env.get_magnetic_snap_report()
    full_open_before_retreat_steps = int(getattr(args, "full_open_after_retreat_steps", 0))
    if (
        role in FIRST_LAYER_WALL_ROLES
        and release_open_gripper > float(OPEN_GRIPPER)
        and full_open_before_retreat_steps > 0
    ):
        _profile_call(
            role_timing,
            "full_open_before_retreat_sec",
            _add_ramp_segment,
            env,
            segments,
            f"{role}_full_open_before_retreat",
            release_open_gripper,
            OPEN_GRIPPER,
            full_open_before_retreat_steps,
        )
        report["full_open_before_retreat"] = {
            "enabled": True,
            "from": float(release_open_gripper),
            "to": float(OPEN_GRIPPER),
            "steps": int(full_open_before_retreat_steps),
        }
        retreat_gripper = OPEN_GRIPPER
    else:
        report["full_open_before_retreat"] = {
            "enabled": False,
            "reason": "not_first_layer_wall_or_steps_zero_or_already_open",
        }
        retreat_gripper = release_open_gripper
    retreat_report: dict[str, Any] = {"mode": "world_z_preplace"}
    q_retreat = release_report.get("q_preplace")
    current_q_after_open = _current_q(base_env)
    if q_retreat is not None:
        q_retreat = _nearest_equivalent_joint_configuration(
            np.asarray(q_retreat, dtype=np.float32).reshape(7),
            current_q_after_open,
        )
        retreat_report["wrapped_to_current_branch"] = True
        retreat_report["wrapped_max_joint_delta_deg"] = _max_abs_joint_delta_deg(q_retreat, current_q_after_open)
    if role in FIRST_LAYER_WALL_ROLES and bool(getattr(args, "wall_post_open_pullback_retreat", False)):
        target_tcp_pose = release_report.get("target_tcp_pose")
        release_approach = release_report.get("release_approach", {})
        if target_tcp_pose is not None:
            pullback_pose, pullback_meta = _post_open_pullback_pose(
                base_env=base_env,
                target_tcp_pose=target_tcp_pose,
                release_approach=release_approach,
                pullback_distance=float(getattr(args, "wall_post_open_pullback_distance", 0.035)),
                lift_distance=float(getattr(args, "wall_post_open_pullback_lift", 0.025)),
            )
            retreat_args = _args_for_role(args, role)
            ok_pullback, tcp_q_pullback, tcp_pullback_ik = _solve_ik(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                exclude_role=None,
                exclude_roles=_release_collision_exclude_roles(retreat_args, role),
                target_pose=pullback_pose,
                num_seeds=args.ik_seeds,
                start_q=current_q_after_open,
            )
            if tcp_q_pullback is not None:
                tcp_q_pullback = _nearest_equivalent_joint_configuration(
                    np.asarray(tcp_q_pullback, dtype=np.float32).reshape(7),
                    current_q_after_open,
                )
            pullback_joint_delta = (
                _joint_distance(tcp_q_pullback, current_q_after_open)
                if tcp_q_pullback is not None
                else None
            )
            pullback_wrist_delta_deg = (
                _max_abs_joint_delta_deg(tcp_q_pullback, current_q_after_open, [4, 5, 6])
                if tcp_q_pullback is not None
                else None
            )
            retreat_max_wrist_delta_deg = float(getattr(args, "retreat_max_wrist_joint_delta_deg", 0.0) or 0.0)
            retreat_max_joint_delta = float(getattr(args, "tcp_retreat_max_joint_delta", 1.2))
            pullback_joint_ok = (
                pullback_joint_delta is None
                or retreat_max_joint_delta <= 0.0
                or pullback_joint_delta <= retreat_max_joint_delta
            )
            pullback_wrist_ok = (
                pullback_wrist_delta_deg is None
                or retreat_max_wrist_delta_deg <= 0.0
                or pullback_wrist_delta_deg <= retreat_max_wrist_delta_deg
            )
            retreat_report = {
                **pullback_meta,
                "ik": tcp_pullback_ik,
                "joint_delta_from_current": pullback_joint_delta,
                "max_joint_delta": retreat_max_joint_delta,
                "wrist_joint_delta_deg": pullback_wrist_delta_deg,
                "max_wrist_joint_delta_deg": retreat_max_wrist_delta_deg,
                "success": bool(ok_pullback and tcp_q_pullback is not None and pullback_joint_ok and pullback_wrist_ok),
                "fallback_mode": "world_z_preplace",
            }
            if ok_pullback and tcp_q_pullback is not None and pullback_joint_ok and pullback_wrist_ok:
                q_retreat = tcp_q_pullback
    tcp_retreat_roles = str(getattr(args, "tcp_retreat_after_open_roles", "top_lid") or "")
    if _role_in_csv(role, tcp_retreat_roles):
        retreat_distance = float(getattr(args, "tcp_retreat_distance", 0.08))
        retreat_direction_sign = float(getattr(args, "tcp_retreat_direction_sign", -1.0))
        retreat_pose = _tcp_retreat_pose(base_env, retreat_distance, retreat_direction_sign)
        retreat_args = _args_for_role(args, role)
        ok_retreat, tcp_q_retreat, tcp_retreat_ik = _solve_ik(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            exclude_role=None,
            exclude_roles=_release_collision_exclude_roles(retreat_args, role),
            target_pose=retreat_pose,
            num_seeds=args.ik_seeds,
            start_q=_current_q(base_env),
        )
        retreat_joint_delta = None
        if tcp_q_retreat is not None:
            tcp_q_retreat = _nearest_equivalent_joint_configuration(
                np.asarray(tcp_q_retreat, dtype=np.float32).reshape(7),
                current_q_after_open,
            )
            retreat_joint_delta = _joint_distance(tcp_q_retreat, current_q_after_open)
        retreat_max_wrist_delta_deg = float(getattr(args, "retreat_max_wrist_joint_delta_deg", 0.0) or 0.0)
        retreat_wrist_delta_deg = (
            _max_abs_joint_delta_deg(tcp_q_retreat, current_q_after_open, [4, 5, 6])
            if tcp_q_retreat is not None
            else None
        )
        retreat_max_joint_delta = float(getattr(args, "tcp_retreat_max_joint_delta", 1.2))
        retreat_joint_ok = retreat_joint_delta is None or retreat_max_joint_delta <= 0.0 or retreat_joint_delta <= retreat_max_joint_delta
        retreat_wrist_ok = (
            retreat_wrist_delta_deg is None
            or retreat_max_wrist_delta_deg <= 0.0
            or retreat_wrist_delta_deg <= retreat_max_wrist_delta_deg
        )
        retreat_report = {
            "mode": "tcp_axis",
            "distance_m": retreat_distance,
            "direction_sign": retreat_direction_sign,
            "max_joint_delta": retreat_max_joint_delta,
            "joint_delta_from_current": retreat_joint_delta,
            "max_wrist_joint_delta_deg": retreat_max_wrist_delta_deg,
            "wrist_joint_delta_deg": retreat_wrist_delta_deg,
            "target_pose": _pose_to_report(retreat_pose),
            "ik": tcp_retreat_ik,
            "success": bool(ok_retreat and tcp_q_retreat is not None and retreat_joint_ok and retreat_wrist_ok),
        }
        if ok_retreat and tcp_q_retreat is not None and retreat_joint_ok and retreat_wrist_ok:
            q_retreat = tcp_q_retreat
        elif role == "top_lid":
            q_retreat = None
    report["post_open_retreat"] = retreat_report
    post_open_retreat_executed = False
    if q_retreat is not None and int(getattr(args, "return_home_steps", 0)) > 0:
        _profile_call(
            role_timing,
            "retreat_exec_sec",
            _add_adaptive_joint_segment,
            env=env,
            arrays=arrays,
            segments=segments,
            name=f"{role}_post_open_retreat",
            goal_q=np.asarray(q_retreat, dtype=np.float32),
            gripper=retreat_gripper,
            base_steps=int(args.return_home_steps),
            action_repeat=args.action_repeat,
            final_hold=args.final_hold_steps,
            args=args,
        )
        post_open_retreat_executed = True
    full_open_after_retreat_steps = int(getattr(args, "full_open_after_retreat_steps", 0))
    if (
        post_open_retreat_executed
        and full_open_after_retreat_steps > 0
        and abs(float(retreat_gripper) - float(OPEN_GRIPPER)) > 1e-6
    ):
        _profile_call(
            role_timing,
            "full_open_after_retreat_sec",
            _add_ramp_segment,
            env,
            segments,
            f"{role}_full_open_after_retreat",
            release_open_gripper,
            OPEN_GRIPPER,
            full_open_after_retreat_steps,
        )
        report["full_open_after_retreat"] = {
            "enabled": True,
            "from": float(release_open_gripper),
            "to": float(OPEN_GRIPPER),
            "steps": int(full_open_after_retreat_steps),
        }
    else:
        report["full_open_after_retreat"] = {
            "enabled": False,
            "reason": (
                "retreat_not_executed"
                if not post_open_retreat_executed
                else "already_open_before_retreat"
                if abs(float(retreat_gripper) - float(OPEN_GRIPPER)) <= 1e-6
                else "steps_zero"
            ),
        }
    if role == "top_lid" and bool(getattr(args, "top_lid_enable_all_connections_after_hinge", False)):
        _profile_call(role_timing, "capture_after_hinge_sec", enable_role_capture, "after_hinge")
        _profile_call(
            role_timing,
            "all_connections_hold_sec",
            _add_hold_segment,
            env,
            segments,
            f"{role}_all_connections_hold",
            release_open_gripper,
            getattr(args, "top_lid_all_connections_hold_steps", 80),
        )
        report["snap_report_after_all_top_lid_connections"] = base_env.get_magnetic_snap_report()
    _profile_call(role_timing, "stability_check_sec", _add_settle_segment, env, segments, f"{role}_stability_check", args.stability_steps)

    report["pose_error_after_release"] = _pose_error(actor, targets[role])
    if "floor" in locked and "floor" in targets:
        report["floor_pose_error_after_release"] = _pose_error(locked["floor"].actor, targets["floor"])
    report["active_connection_count_for_role"] = _active_connection_count_for_role(base_env, role)
    report["suspended_roles"] = sorted(base_env.magnetic_snap.suspended_roles)
    report["success"] = bool(
        report["pose_error_after_release"]["position_error_m"] <= args.max_position_error
        and report["pose_error_after_release"]["orientation_error_deg"] <= args.max_orientation_error_deg
        and report["active_connection_count_for_role"] >= int(report.get("required_active_connections_for_role", args.min_active_connections))
        and role not in base_env.magnetic_snap.suspended_roles
    )
    if not report["success"]:
        report["failed_at"] = "final_stability_check"
        required_connections = int(report.get("required_active_connections_for_role", args.min_active_connections))
        if int(report["active_connection_count_for_role"]) < required_connections:
            report["failure_classification"] = "missing_connection_after_open"
        else:
            report["failure_classification"] = "final_connection_quality_fail"
    elif bool(getattr(args, "return_neutral_after_role", False)):
        skip_final_return = bool(getattr(args, "return_neutral_skip_final_role", False)) and not bool(has_next_role)
        if skip_final_return:
            report["returned_neutral_after_success"] = False
            report["return_neutral_skipped_final_role"] = True
        else:
            if prefetch_manager is not None and next_role and next_role in locked:
                prefetch_manager.submit(
                    role=next_role,
                    planner=planner,
                    base_env=base_env,
                    locked=locked,
                    fixtures=fixtures,
                    actor=locked[next_role].actor,
                    target_actor_pose=targets.get(next_role),
                    start_q=RM75_HOME,
                    args=args,
                )
                report["next_cycle_prefetch_submitted"] = next_role
            return_neutral_max_joint_step = float(getattr(args, "return_neutral_max_joint_step", 0.0) or 0.0)
            if return_neutral_max_joint_step <= 0.0:
                return_neutral_max_joint_step = float(getattr(args, "max_joint_step", 0.06))
            if bool(getattr(args, "free_space_motion_window", False)):
                return_neutral_max_joint_step = float(
                    getattr(args, "free_space_return_neutral_max_joint_step", return_neutral_max_joint_step)
                )
            return_neutral_steps = int(getattr(args, "return_neutral_steps", 24))
            if bool(getattr(args, "free_space_motion_window", False)):
                return_neutral_steps = min(
                    return_neutral_steps,
                    max(int(getattr(args, "free_space_return_neutral_steps", return_neutral_steps)), 1),
                )
            planned_return_executed = False
            if bool(getattr(args, "return_neutral_motion_plan", True)):
                ok_return, q_return_path, return_plan_report = _profile_call(
                    role_timing,
                    "return_neutral_motion_plan_sec",
                    _plan_joint_motion_to_goal,
                    planner=planner,
                    base_env=base_env,
                    locked=locked,
                    fixtures=fixtures,
                    start_q=_current_q(base_env),
                    goal_q=RM75_HOME,
                    timeout=float(getattr(args, "return_neutral_motion_plan_timeout", 2.5)),
                    max_attempts=int(getattr(args, "return_neutral_motion_plan_max_attempts", 1)),
                    enable_graph=bool(getattr(args, "return_neutral_motion_plan_enable_graph", True)),
                    num_graph_seeds=int(getattr(args, "return_neutral_motion_plan_graph_seeds", 1)),
                )
                report["return_neutral_motion_plan"] = return_plan_report
                if ok_return and q_return_path is not None:
                    _profile_call(
                        role_timing,
                        "return_neutral_sec",
                        _record_existing_joint_path,
                        env=env,
                        arrays=arrays,
                        segments=segments,
                        name=f"{role}_return_neutral_plan_after_success",
                        path=q_return_path,
                        gripper=OPEN_GRIPPER,
                        final_hold=args.final_hold_steps,
                        max_joint_step=return_neutral_max_joint_step,
                        max_waypoints=int(getattr(args, "return_neutral_motion_plan_max_waypoints", 0)),
                    )
                    planned_return_executed = True
                    report["return_neutral_mode"] = "motion_plan_collision_checked"
            if not planned_return_executed:
                report["return_neutral_mode"] = "joint_interpolation_fallback"
                _profile_call(
                    role_timing,
                    "return_neutral_sec",
                    _add_adaptive_joint_segment,
                    env=env,
                    arrays=arrays,
                    segments=segments,
                    name=f"{role}_return_neutral_after_success",
                    goal_q=RM75_HOME,
                    gripper=OPEN_GRIPPER,
                    base_steps=return_neutral_steps,
                    action_repeat=args.action_repeat,
                    final_hold=args.final_hold_steps,
                    args=args,
                    max_joint_step_override=return_neutral_max_joint_step,
                )
            report["returned_neutral_after_success"] = True
            report["return_neutral_max_joint_step"] = float(return_neutral_max_joint_step)
    report["segments_end"] = len(segments)
    role_timing["role_total_sec"] = time.perf_counter() - role_started
    report["timing"] = role_timing
    if profile is not None:
        _accumulate_profile(profile, role_timing)
    return finalize_report()


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _make_logger(out_dir)
    run_started = time.perf_counter()
    run_profile: dict[str, Any] = {}
    summary_path = out_dir / "multi_wall_summary.json"
    manifest_path = out_dir / "multi_wall_manifest.json"
    arrays_path = out_dir / "multi_wall_arrays.npz"
    video_path = out_dir / "multi_wall_live.mp4"
    roles = [item.strip() for item in args.roles.split(",") if item.strip()]
    save_roles = _role_list(str(getattr(args, "save_assembly_roles", "") or ""))
    save_completed_roles = _role_list(str(getattr(args, "save_assembly_completed_roles", "") or ""))
    arrays: dict[str, np.ndarray] = {}
    segments: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    requested_render_mode = str(getattr(args, "render_mode", "none") or "none").strip().lower()
    if requested_render_mode in {"", "off", "null"}:
        requested_render_mode = "none"
    env_render_mode = None if requested_render_mode == "none" else requested_render_mode
    if env_render_mode is None and bool(getattr(args, "record_live", False)):
        env_render_mode = "rgb_array"
    env = _make_env(bool(getattr(args, "record_live", False)), render_mode=env_render_mode)
    prefetch_manager = _NextRolePrefetchManager(bool(getattr(args, "next_cycle_plan_prefetch", False)))
    writer = None
    restore_recorder = None
    real_exec = None
    executed_live_steps = 0
    try:
        _profile_call(run_profile, "env_reset_sec", env.reset)
        if env_render_mode == "human":
            _profile_call(run_profile, "human_render_sec", env.render)
        base_env = env.unwrapped
        snap = base_env.magnetic_snap
        snap_parameter_overrides = {}
        if float(getattr(args, "magnet_attach_distance", 0.0)) > 0.0:
            snap.attach_distance = float(args.magnet_attach_distance)
            snap_parameter_overrides["attach_distance"] = snap.attach_distance
        if float(getattr(args, "magnet_attract_distance", 0.0)) > 0.0:
            snap.attract_distance = float(args.magnet_attract_distance)
            snap_parameter_overrides["attract_distance"] = snap.attract_distance
        if float(getattr(args, "magnet_detach_distance", 0.0)) > 0.0:
            snap.detach_distance = float(args.magnet_detach_distance)
            snap_parameter_overrides["detach_distance"] = snap.detach_distance
        if float(getattr(args, "magnet_edge_sample_half_span", 0.0)) > 0.0:
            snap.edge_sample_half_span = float(args.magnet_edge_sample_half_span)
            snap_parameter_overrides["edge_sample_half_span"] = snap.edge_sample_half_span
        edge_sample_offsets = _float_csv(str(getattr(args, "magnet_edge_sample_offsets", "") or ""), [])
        if edge_sample_offsets:
            snap.edge_sample_offsets = edge_sample_offsets
            snap_parameter_overrides["edge_sample_offsets"] = list(
                snap._expanded_edge_sample_offsets(snap.edge_sample_offsets, snap.plate_size)
            )
        if float(getattr(args, "magnet_connect_edge_sample_half_span", 0.0)) > 0.0:
            snap.connection_edge_sample_half_span = float(args.magnet_connect_edge_sample_half_span)
            snap_parameter_overrides["connection_edge_sample_half_span"] = snap.connection_edge_sample_half_span
        connection_edge_sample_offsets = _float_csv(
            str(getattr(args, "magnet_connect_edge_sample_offsets", "") or ""),
            [],
        )
        if connection_edge_sample_offsets:
            snap.connection_edge_sample_offsets = connection_edge_sample_offsets
            snap_parameter_overrides["connection_edge_sample_offsets"] = list(
                snap._expanded_edge_sample_offsets(snap.connection_edge_sample_offsets, snap.plate_size)
            )
        if (
            "edge_sample_half_span" in snap_parameter_overrides
            or "connection_edge_sample_half_span" in snap_parameter_overrides
            or "edge_sample_offsets" in snap_parameter_overrides
            or "connection_edge_sample_offsets" in snap_parameter_overrides
        ):
            snap._disable_all_drives()
            snap.active_connections.clear()
            snap.drives.clear()
        _apply_float_override(
            snap,
            "attract_stiffness",
            getattr(args, "magnet_attract_stiffness", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "attract_force_limit",
            getattr(args, "magnet_attract_force_limit", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "attract_torque_stiffness",
            getattr(args, "magnet_attract_torque_stiffness", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "attract_torque_limit",
            getattr(args, "magnet_attract_torque_limit", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "attract_normal_torque_stiffness",
            getattr(args, "magnet_attract_normal_torque_stiffness", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "attract_normal_torque_limit",
            getattr(args, "magnet_attract_normal_torque_limit", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "active_magnet_stiffness",
            getattr(args, "magnet_active_stiffness", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "active_magnet_damping",
            getattr(args, "magnet_active_damping", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "active_magnet_force_limit",
            getattr(args, "magnet_active_force_limit", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "active_edge_torque_scale",
            getattr(args, "magnet_active_edge_torque_scale", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "active_edge_torque_min_scale",
            getattr(args, "magnet_active_edge_torque_min_scale", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "active_normal_torque_scale",
            getattr(args, "magnet_active_normal_torque_scale", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "active_normal_torque_min_scale",
            getattr(args, "magnet_active_normal_torque_min_scale", None),
            snap_parameter_overrides,
        )
        _apply_int_override(
            snap,
            "active_torque_delay_steps",
            getattr(args, "magnet_active_torque_delay_steps", None),
            snap_parameter_overrides,
        )
        _apply_int_override(
            snap,
            "active_torque_ramp_steps",
            getattr(args, "magnet_active_torque_ramp_steps", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "active_torque_max_point_error",
            getattr(args, "magnet_active_torque_max_point_error", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "floor_support_score",
            getattr(args, "magnet_floor_support_score", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "support_connection_score_scale",
            getattr(args, "magnet_support_connection_score_scale", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "multi_connection_support_bonus",
            getattr(args, "magnet_multi_connection_support_bonus", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "drive_stiffness",
            getattr(args, "magnet_drive_stiffness", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "drive_damping",
            getattr(args, "magnet_drive_damping", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "drive_force_limit",
            getattr(args, "magnet_drive_force_limit", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "drive_angular_stiffness",
            getattr(args, "magnet_drive_angular_stiffness", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "drive_angular_damping",
            getattr(args, "magnet_drive_angular_damping", None),
            snap_parameter_overrides,
        )
        _apply_float_override(
            snap,
            "drive_angular_force_limit",
            getattr(args, "magnet_drive_angular_force_limit", None),
            snap_parameter_overrides,
        )
        if snap_parameter_overrides:
            log(f"magnet parameter overrides: {snap_parameter_overrides}")
        _set_robot_qpos(base_env, RM75_HOME, gripper_open=True)
        real_exec = _start_real_executor(args, log)
        targets, locked, fixtures = _initialize_staged_open_cube(base_env)
        initial_assembly_offset = _apply_initial_assembly_offset(
            locked=locked,
            targets=targets,
            args=args,
            log=log,
        )
        initial_actor_jitter = _apply_initial_actor_jitter(
            locked=locked,
            targets=targets,
            fixtures=fixtures,
            args=args,
            log=log,
        )
        rear_collision_wall = _add_rear_collision_wall_fixture(
            base_env=base_env,
            targets=targets,
            fixtures=fixtures,
            args=args,
            log=log,
        )
        overhead_collision_wall = _add_overhead_collision_wall_fixture(
            base_env=base_env,
            fixtures=fixtures,
            args=args,
            log=log,
        )
        loaded_state_roles = _load_assembly_state(
            path=str(getattr(args, "load_assembly_state", "") or ""),
            base_env=base_env,
            locked=locked,
            targets=targets,
            args=args,
            log=log,
            restore_robot_qpos=bool(getattr(args, "restore_loaded_robot_qpos", False)),
        )
        if env_render_mode == "human":
            _profile_call(run_profile, "human_render_sec", env.render)
        loaded_state_perturbation = _apply_loaded_state_perturbation(
            base_env=base_env,
            locked=locked,
            targets=targets,
            args=args,
            log=log,
        ) if loaded_state_roles else {}
        restored_loaded_magnetic_connections = (
            _restore_loaded_magnetic_connections(
                path=str(getattr(args, "load_assembly_state", "") or ""),
                base_env=base_env,
                locked=locked,
                log=log,
            )
            if loaded_state_roles and bool(getattr(args, "restore_loaded_magnetic_connections", True))
            else []
        )
        base_connection_targets = (
            _configure_full_base_connection_targets(
                base_env,
                ["floor", *loaded_state_roles],
                log,
            )
            if bool(getattr(args, "require_full_base_connections", False)) and loaded_state_roles
            else {}
        )
        if loaded_state_roles:
            _profile_call(
                run_profile,
                "loaded_state_settle_sec",
                _add_hold_segment,
                env,
                segments,
                "loaded_assembly_state_settle",
                OPEN_GRIPPER,
                args.loaded_state_settle_steps,
            )
            log(f"loaded state snap report: {base_env.get_magnetic_snap_report()}")
        if bool(getattr(args, "record_live", False)):
            writer = _profile_call(
                run_profile,
                "video_writer_init_sec",
                imageio.get_writer,
                video_path,
                fps=args.fps,
                codec="libx264",
                quality=8,
                macro_block_size=8,
            )
            for _ in range(max(int(args.fps), 1)):
                _profile_call(run_profile, "video_initial_frame_sec", _append_frame, writer, env)
        restore_recorder = _install_live_recorder(
            env,
            writer,
            args.record_every,
            run_profile,
            args=args,
            real_exec=real_exec,
            log=log,
            human_render=env_render_mode == "human",
            human_render_every=max(int(getattr(args, "human_render_every", 1)), 1),
        )
        if bool(getattr(args, "heal_loaded_state_only", False)):
            heal_role_text = str(getattr(args, "loaded_state_heal_roles", "") or "").strip()
            heal_roles = _role_list(heal_role_text) if heal_role_text else list(loaded_state_roles)
            if "floor" in locked:
                heal_roles = list(dict.fromkeys(["floor", *heal_roles]))
            heal_steps = max(int(getattr(args, "loaded_state_heal_steps", 0)), 0)
            if heal_steps > 0:
                _add_hold_segment(env, segments, "loaded_assembly_state_heal", OPEN_GRIPPER, heal_steps)
            heal_validation_steps = max(int(getattr(args, "loaded_state_heal_validate_steps", 0)), 0)
            original_final_steps = int(getattr(args, "final_all_roles_stability_steps", 0))
            if heal_validation_steps > 0:
                args.final_all_roles_stability_steps = heal_validation_steps
                all_role_stability = _profile_call(
                    run_profile,
                    "loaded_state_heal_validation_sec",
                    _validate_roles_after_settle,
                    env=env,
                    base_env=base_env,
                    locked=locked,
                    targets=targets,
                    roles=heal_roles,
                    segments=segments,
                    args=args,
                    name="loaded_state_heal_stability_check",
                    gripper=OPEN_GRIPPER,
                    log=log,
                )
                args.final_all_roles_stability_steps = original_final_steps
            else:
                all_role_stability = {
                    "success": bool(heal_roles),
                    "name": "loaded_state_heal_snapshot",
                    "steps": 0,
                    "roles": heal_roles,
                    "role_reports": {},
                    "magnetic_snap_report": base_env.get_magnetic_snap_report(),
                }
            if str(getattr(args, "save_assembly_state", "") or ""):
                _save_assembly_state(
                    path=str(getattr(args, "save_assembly_state", "") or ""),
                    base_env=base_env,
                    locked=locked,
                    targets=targets,
                    reports=[],
                    roles=heal_roles,
                    log=log,
                    loaded_completed_roles=loaded_state_roles,
                    save_roles=save_roles,
                    completed_roles_override=save_completed_roles or None,
                )
            final = {
                "success": bool(all_role_stability.get("success")),
                "roles": heal_roles,
                "completed_roles": [role for role in heal_roles if role != "floor"],
                "executed_steps_estimate": _estimate_steps(segments),
                "executed_live_steps": _estimate_steps(segments),
                "video": str(video_path) if bool(getattr(args, "record_live", False)) else None,
                "all_role_stability": all_role_stability,
                "floor_pose_error_final": _pose_error(locked["floor"].actor, targets["floor"]) if "floor" in locked else None,
                "healed_pose_errors_final": {
                    role: _pose_error(locked[role].actor, targets[role])
                    for role in heal_roles
                    if role in locked and role in targets
                },
            }
            timing_summary = dict(run_profile)
            timing_summary["wall_clock_sec"] = time.perf_counter() - run_started
            final["timing"] = timing_summary
            manifest = {
                "name": "multi_wall_loaded_state_heal",
                "roles": heal_roles,
                "control_mode": "pd_joint_pos_abs",
                "arrays": str(arrays_path),
                "segments": segments,
                "reports": reports,
                "floor_anchor": False,
                "magnet_gating": {
                    "parameter_overrides": snap_parameter_overrides,
                },
                "assembly_state": {
                    "loaded_roles": loaded_state_roles,
                    "load_path": str(getattr(args, "load_assembly_state", "") or ""),
                    "save_path": str(getattr(args, "save_assembly_state", "") or ""),
                    "save_roles": save_roles,
                    "save_completed_roles": save_completed_roles,
                    "initial_assembly_offset": initial_assembly_offset,
                    "initial_actor_jitter": initial_actor_jitter,
                    "perturbation": loaded_state_perturbation,
                    "heal_only": True,
                    "heal_steps": heal_steps,
                    "heal_validate_steps": heal_validation_steps,
                    "restored_loaded_magnetic_connections": restored_loaded_magnetic_connections,
                    "base_connection_targets": base_connection_targets,
                },
                "rear_collision_wall": rear_collision_wall,
                "overhead_collision_wall": overhead_collision_wall,
            }
            return _write_run_outputs(
                summary_path=summary_path,
                manifest_path=manifest_path,
                arrays_path=arrays_path,
                arrays=arrays,
                manifest=manifest,
                reports=reports,
                final=final,
            )
        stage_targets = {role: _actor_pose(locked[role].actor) for role in roles if role in locked}
        if bool(getattr(args, "disable_unused_roles", False)) or bool(getattr(args, "hide_unused_roles", False)):
            active_role_set = set(roles)
            for unused_role in sorted(set(locked) - active_role_set - {"floor"}):
                base_env.deactivate_magnetic_piece(unused_role, hide=bool(getattr(args, "hide_unused_roles", False)))
                action = "hidden/disabled" if bool(getattr(args, "hide_unused_roles", False)) else "disabled"
                log(f"{unused_role}: {action} because it is not part of this run")
        floor_anchor = _anchor_floor_for_build(
            base_env=base_env,
            locked=locked,
            target_pose=targets["floor"],
            args=args,
            log=log,
        )
        preplaced_roles = [item.strip() for item in str(getattr(args, "preplaced_roles", "")).split(",") if item.strip()]
        for role in preplaced_roles:
            if role not in locked or role not in targets:
                continue
            actor = locked[role].actor
            actor.set_pose(targets[role])
            actor.set_linear_velocity(np.zeros(3, dtype=np.float32))
            actor.set_angular_velocity(np.zeros(3, dtype=np.float32))
            log(f"{role}: preplaced at target for isolated downstream placement")
        if preplaced_roles:
            _add_hold_segment(env, segments, "preplaced_roles_snap_settle", OPEN_GRIPPER, args.preplaced_settle_steps)
        role_order_policy = str(getattr(args, "role_order_policy", "adaptive") or "adaptive").strip().lower()
        if role_order_policy == "given":
            log(f"role order policy keeps provided order: {roles}")
        elif role_order_policy == "far_from_robot":
            log(f"role order policy far_from_robot keeps wrapper-provided order: {roles}")
        elif bool(getattr(args, "adaptive_role_order", True)):
            ordered_roles = _order_roles_by_connection_potential(
                base_env,
                roles,
                built_roles=[*loaded_state_roles, *preplaced_roles],
            )
            if ordered_roles != roles:
                log(f"role order adjusted for connection potential: {roles} -> {ordered_roles}")
            roles = ordered_roles
        args.release_ignore_roles_by_role = _release_ignore_mapping_for_roles(roles, loaded_state_roles)
        if bool(getattr(args, "disable_unbuilt_role_magnets", False)):
            for role in roles:
                if role in preplaced_roles:
                    continue
                _set_role_magnets_enabled(base_env, role, False)
            extra_pending_roles = [
                item.strip()
                for item in str(getattr(args, "extra_pending_roles", "") or "").split(",")
                if item.strip()
            ]
            for role in extra_pending_roles:
                if role in locked and role not in roles and role not in preplaced_roles:
                    _set_role_magnets_enabled(base_env, role, False)
            log(
                "pending roles: magnets disabled until each role reaches its release pose: "
                f"{roles + [role for role in extra_pending_roles if role in locked and role not in roles]}"
            )
        planner = _profile_call(
            run_profile,
            "planner_init_sec",
            RM75CuRoboPlanner,
            RM75CuRoboPlannerConfig(
                curobo_root=Path(args.curobo_root),
                robot_cfg_path=Path(args.robot_cfg),
                position_threshold=args.ik_position_threshold,
                rotation_threshold=args.ik_rotation_threshold,
                num_ik_seeds=args.ik_seeds,
                use_cuda_graph_batch_ik=bool(getattr(args, "use_cuda_graph_batch_ik", False)),
                cuda_graph_batch_ik_max_batch=int(getattr(args, "cuda_graph_batch_ik_max_batch", 128)),
                cuda_graph_batch_ik_fixed_batch_size=(
                    int(getattr(args, "cuda_graph_batch_ik_fixed_batch_size", 0))
                    if int(getattr(args, "cuda_graph_batch_ik_fixed_batch_size", 0)) > 0
                    else None
                ),
                collision_activation_distance=0.018,
                build_motion_gen=True,
            ),
        )
        _add_hold_segment(env, segments, "initial_hold", OPEN_GRIPPER, args.initial_steps)
        for role_index, role in enumerate(roles):
            stop_before_role = str(getattr(args, "stop_before_role", "") or "").strip()
            if stop_before_role and role == stop_before_role:
                log(f"stopping before role={role} as requested")
                _save_assembly_state(
                    path=str(getattr(args, "save_assembly_state", "") or ""),
                    base_env=base_env,
                    locked=locked,
                    targets=targets,
                    reports=reports,
                    roles=roles,
                    log=log,
                    loaded_completed_roles=loaded_state_roles,
                    save_roles=save_roles,
                    completed_roles_override=save_completed_roles or None,
                )
                break
            completed_before_role = list(dict.fromkeys([*loaded_state_roles, *[item["role"] for item in reports if item.get("success")]]))
            freeze_report = _set_non_current_role_freeze(
                base_env=base_env,
                locked=locked,
                active_role=role,
                args=args,
                log=log,
            )
            try:
                role_report = _build_one_role(
                    env=env,
                    planner=planner,
                    targets=targets,
                    locked=locked,
                    stage_targets=stage_targets,
                    fixtures=fixtures,
                    arrays=arrays,
                    segments=segments,
                    role=role,
                    has_next_role=role_index < len(roles) - 1,
                    next_role=roles[role_index + 1] if role_index < len(roles) - 1 else None,
                    completed_roles_before=completed_before_role,
                    args=args,
                    log=log,
                    prefetch_manager=prefetch_manager,
                    profile=run_profile,
                )
            finally:
                _clear_non_current_role_freeze(base_env)
            role_report["non_current_role_freeze"] = freeze_report
            reports.append(role_report)
            if not role_report.get("success"):
                break
            if bool(getattr(args, "validate_all_completed_after_each_role", False)):
                completed_so_far = [item["role"] for item in reports if item.get("success")]
                validation_roles = list(dict.fromkeys(["floor", *loaded_state_roles, *completed_so_far])) if "floor" in locked else list(dict.fromkeys([*loaded_state_roles, *completed_so_far]))
                role_report["all_completed_stability_after_role"] = _profile_call(
                    run_profile,
                    "all_completed_validation_sec",
                    _validate_roles_after_settle,
                    env=env,
                    base_env=base_env,
                    locked=locked,
                    targets=targets,
                    roles=validation_roles,
                    segments=segments,
                    args=args,
                    name=f"{role}_all_completed_stability_check",
                    gripper=OPEN_GRIPPER,
                    log=log,
                )
                if not role_report["all_completed_stability_after_role"].get("success"):
                    role_report["success"] = False
                    role_report["failed_at"] = "all_completed_stability_after_role"
                    break
        completed_roles = [item["role"] for item in reports if item.get("success")]
        final_stability_roles = list(dict.fromkeys(["floor", *loaded_state_roles, *completed_roles])) if "floor" in locked else list(dict.fromkeys([*loaded_state_roles, *completed_roles]))
        failed_reports = [item for item in reports if not item.get("success")]
        keep_gripper_closed_after_failure = bool(
            failed_reports
            and failed_reports[-1].get("held_actor_pose_lock_still_active")
        )
        final_stability_gripper = CLOSED_GRIPPER if keep_gripper_closed_after_failure else OPEN_GRIPPER
        all_role_stability = _profile_call(
            run_profile,
            "final_all_roles_stability_sec",
            _validate_roles_after_settle,
            env=env,
            base_env=base_env,
            locked=locked,
            targets=targets,
            roles=final_stability_roles,
            segments=segments,
            args=args,
            name="final_all_roles_stability_check",
            gripper=final_stability_gripper,
            log=log,
        ) if completed_roles else {"success": False, "roles": [], "role_reports": {}, "steps": 0}
        if str(getattr(args, "save_assembly_state", "") or "") and not (
            str(getattr(args, "stop_before_role", "") or "").strip()
            and any(role == str(getattr(args, "stop_before_role", "") or "").strip() for role in roles)
        ):
            all_requested_roles_succeeded = bool(reports and all(item.get("success") for item in reports))
            if all_requested_roles_succeeded:
                state_save_roles = save_roles
                state_completed_roles = save_completed_roles or None
            else:
                partial_roles = list(dict.fromkeys(["floor", *loaded_state_roles, *completed_roles]))
                state_save_roles = partial_roles if save_roles else []
                state_completed_roles = [role for role in partial_roles if role != "floor"]
            _save_assembly_state(
                path=str(getattr(args, "save_assembly_state", "") or ""),
                base_env=base_env,
                locked=locked,
                targets=targets,
                reports=reports,
                roles=roles,
                log=log,
                loaded_completed_roles=loaded_state_roles,
                save_roles=state_save_roles,
                completed_roles_override=state_completed_roles,
            )
        final = {
            "success": bool(reports and all(item.get("success") for item in reports) and all_role_stability.get("success")),
            "roles": roles,
            "completed_roles": completed_roles,
            "executed_steps_estimate": _estimate_steps(segments),
            "executed_live_steps": _estimate_steps(segments),
            "video": str(video_path) if bool(getattr(args, "record_live", False)) else None,
            "all_role_stability": all_role_stability,
            "floor_pose_error_final": _pose_error(locked["floor"].actor, targets["floor"]) if "floor" in locked else None,
            "stage_pose_errors_final": {
                role: _pose_error(locked[role].actor, stage_targets[role])
                for role in roles
                if role in locked and role in stage_targets and role not in {item["role"] for item in reports if item.get("success")}
            },
            "initial_assembly_offset": initial_assembly_offset,
            "initial_actor_jitter": initial_actor_jitter,
        }
        timing_summary = dict(run_profile)
        timing_summary["wall_clock_sec"] = time.perf_counter() - run_started
        final["timing"] = timing_summary
        timing_items = [
            (key, value)
            for key, value in timing_summary.items()
            if not key.endswith("_calls") and isinstance(value, (int, float))
        ]
        timing_items.sort(key=lambda item: float(item[1]), reverse=True)
        if timing_items:
            log(
                "timing breakdown: "
                + ", ".join(f"{key}={float(value):.3f}s" for key, value in timing_items[:8])
            )
        manifest = {
            "name": "multi_wall_physical_path",
            "roles": roles,
            "control_mode": "pd_joint_pos_abs",
            "arrays": str(arrays_path),
            "segments": segments,
            "reports": reports,
            "floor_anchor": floor_anchor is not None,
            "real_execution": {
                "execute_real": bool(getattr(args, "execute_real", False)),
                "real_control_hz": float(getattr(args, "real_control_hz", 30.0)),
                "real_max_delta_per_step": float(getattr(args, "real_max_delta_per_step", 0.1)),
                "real_start_max_delta": float(getattr(args, "real_start_max_delta", 0.12)),
                "reset_real_before_start": bool(getattr(args, "reset_real_before_start", True)),
            },
            "magnet_gating": {
                "disable_held_role_magnets_until_release": bool(getattr(args, "disable_held_role_magnets_until_release", False)),
                "disable_unbuilt_role_magnets": bool(getattr(args, "disable_unbuilt_role_magnets", False)),
                "use_live_actor_to_tcp_after_lift": bool(getattr(args, "use_live_actor_to_tcp_after_lift", True)),
                "wall_release_profile": str(getattr(args, "wall_release_profile", "legacy") or "legacy"),
                "parameter_overrides": snap_parameter_overrides,
            },
            "assembly_state": {
                "loaded_roles": loaded_state_roles,
                "load_path": str(getattr(args, "load_assembly_state", "") or ""),
                "save_path": str(getattr(args, "save_assembly_state", "") or ""),
                "save_roles": save_roles,
                "save_completed_roles": save_completed_roles,
                "initial_assembly_offset": initial_assembly_offset,
                "initial_actor_jitter": initial_actor_jitter,
                "perturbation": loaded_state_perturbation,
                "restored_loaded_magnetic_connections": restored_loaded_magnetic_connections,
                "base_connection_targets": base_connection_targets,
            },
            "rear_collision_wall": rear_collision_wall,
            "overhead_collision_wall": overhead_collision_wall,
        }
        return _write_run_outputs(
            summary_path=summary_path,
            manifest_path=manifest_path,
            arrays_path=arrays_path,
            arrays=arrays,
            manifest=manifest,
            reports=reports,
            final=final,
        )
    finally:
        if restore_recorder is not None:
            executed_live_steps = restore_recorder()
        if real_exec is not None:
            try:
                real_exec.set_gripper(float(getattr(args, "real_gripper_open", 0.0)))
            except Exception:
                pass
            try:
                real_exec.close()
            except Exception:
                pass
        if writer is not None:
            for _ in range(max(int(getattr(args, "fps", 12)), 1)):
                _profile_call(run_profile, "video_finalize_frame_sec", _append_frame, writer, env)
            writer.close()
        prefetch_manager.close()
        wait_before_close = getattr(args, "wait_before_close", None)
        if wait_before_close is None:
            wait_before_close = env_render_mode == "human"
        if bool(wait_before_close):
            try:
                print("[multi_wall] run finished. Press Enter to close simulation...", flush=True)
                input()
            except EOFError:
                pass
        env.close()


def _estimate_steps(segments: list[dict[str, Any]]) -> int:
    total = 0
    for segment in segments:
        if segment.get("type") == "joint_path":
            total += int(segment.get("waypoints", 0)) * int(segment.get("action_repeat", 1)) + int(segment.get("final_hold", 0))
        elif segment.get("type") in {"hold", "gripper_ramp", "settle_zero_action"}:
            total += int(segment.get("steps", 0))
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "multi_wall_path_v1"))
    parser.add_argument("--robot-cfg", default=str(Path(__file__).resolve().parent / "curobo_rm75_config" / "rm75.yml"))
    parser.add_argument("--curobo-root", default=str(DEFAULT_CUROBO_ROOT))
    parser.add_argument("--roles", default="right_wall,left_wall")
    parser.add_argument("--stop-before-role", default="")
    parser.add_argument("--save-assembly-state", default="")
    parser.add_argument("--save-assembly-roles", default="")
    parser.add_argument("--save-assembly-completed-roles", default="")
    parser.add_argument("--load-assembly-state", default="")
    parser.add_argument("--restore-loaded-robot-qpos", action="store_true")
    parser.add_argument("--restore-loaded-magnetic-connections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--align-targets-to-loaded-floor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--align-targets-to-loaded-floor-mode", choices=["xy", "xyz"], default="xy")
    parser.add_argument("--align-targets-to-loaded-floor-roles", default="floor,right_wall,back_wall,left_wall,front_wall,top_lid")
    parser.add_argument("--require-full-base-connections", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loaded-state-settle-steps", type=int, default=40)
    parser.add_argument("--adaptive-role-order", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--role-order-policy", choices=["adaptive", "given", "far_from_robot"], default="adaptive")
    parser.add_argument("--heal-loaded-state-only", action="store_true")
    parser.add_argument("--loaded-state-heal-steps", type=int, default=20)
    parser.add_argument("--loaded-state-heal-validate-steps", type=int, default=0)
    parser.add_argument("--loaded-state-heal-roles", default="")
    parser.add_argument("--loaded-state-perturb-dx", type=float, default=0.0)
    parser.add_argument("--loaded-state-perturb-dy", type=float, default=0.0)
    parser.add_argument("--loaded-state-perturb-dz", type=float, default=0.0)
    parser.add_argument("--loaded-state-perturb-yaw-deg", type=float, default=0.0)
    parser.add_argument("--loaded-state-perturb-origin-role", default="floor")
    parser.add_argument("--loaded-state-perturb-roles", default="floor,right_wall,back_wall,left_wall")
    parser.add_argument("--loaded-state-perturb-target-roles", default="floor,right_wall,back_wall,left_wall,front_wall")
    parser.add_argument("--initial-assembly-offset-x", type=float, default=0.0)
    parser.add_argument("--initial-assembly-offset-y", type=float, default=0.0)
    parser.add_argument("--initial-assembly-offset-z", type=float, default=0.0)
    parser.add_argument("--initial-assembly-offset-actor-roles", default="floor")
    parser.add_argument("--initial-assembly-offset-target-roles", default="floor,right_wall,back_wall,left_wall,front_wall")
    parser.add_argument("--initial-actor-jitter-xy", type=float, default=0.0)
    parser.add_argument("--initial-actor-jitter-seed", type=int, default=0)
    parser.add_argument("--initial-actor-jitter-roles", default="right_wall,back_wall,left_wall,front_wall")
    parser.add_argument("--initial-actor-jitter-min-start-distance", type=float, default=0.055)
    parser.add_argument("--initial-actor-jitter-min-target-distance", type=float, default=0.035)
    parser.add_argument("--initial-actor-jitter-max-sample-attempts", type=int, default=100)
    parser.add_argument("--add-rear-collision-wall", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rear-collision-wall-frame", choices=["world_back_x", "robot_back", "floor_back_y"], default="world_back_x")
    parser.add_argument("--rear-collision-wall-distance", type=float, default=0.35)
    parser.add_argument("--rear-collision-wall-robot-back-sign", type=float, default=-1.0)
    parser.add_argument("--rear-collision-wall-y-offset", type=float, default=0.0)
    parser.add_argument("--rear-collision-wall-x-offset", type=float, default=0.0)
    parser.add_argument("--rear-collision-wall-floor-y-offset", type=float, default=0.35)
    parser.add_argument("--rear-collision-wall-width", type=float, default=1.80)
    parser.add_argument("--rear-collision-wall-thickness", type=float, default=0.06)
    parser.add_argument("--rear-collision-wall-height", type=float, default=1.20)
    parser.add_argument("--rear-collision-wall-z-bottom", type=float, default=0.0)
    parser.add_argument("--add-overhead-collision-wall", action="store_true")
    parser.add_argument("--overhead-collision-wall-frame", choices=["robot_base", "world"], default="robot_base")
    parser.add_argument("--overhead-collision-wall-size-x", type=float, default=1.20)
    parser.add_argument("--overhead-collision-wall-size-y", type=float, default=1.20)
    parser.add_argument("--overhead-collision-wall-thickness", type=float, default=0.045)
    parser.add_argument("--overhead-collision-wall-z", type=float, default=0.82)
    parser.add_argument("--overhead-collision-wall-x-offset", type=float, default=0.0)
    parser.add_argument("--overhead-collision-wall-y-offset", type=float, default=0.0)
    parser.add_argument("--preplaced-roles", default="")
    parser.add_argument("--disable-unused-roles", action="store_true")
    parser.add_argument("--hide-unused-roles", action="store_true")
    parser.add_argument("--extra-pending-roles", default="")
    parser.add_argument("--preplaced-settle-steps", type=int, default=20)
    parser.add_argument("--record-live", action="store_true")
    parser.add_argument("--execute-real", action="store_true", help="Connect to RM75 and stream this beta trajectory to the real robot.")
    parser.add_argument("--auto-execute", action="store_true", help="Skip the startup confirmation prompt for --execute-real.")
    parser.add_argument("--robot-ip", type=str, default=None)
    parser.add_argument("--lerobot-root", type=str, default=DEFAULT_LEROBOT_ROOT)
    parser.add_argument("--lerobot-sim2real-root", type=str, default=DEFAULT_LEROBOT_SIM2REAL_ROOT)
    parser.add_argument("--real-gripper-open", type=float, default=0.0)
    parser.add_argument("--real-gripper-close", type=float, default=0.91)
    parser.add_argument("--reset-real-before-start", dest="reset_real_before_start", action="store_true", default=True)
    parser.add_argument("--no-reset-real-before-start", dest="reset_real_before_start", action="store_false")
    parser.add_argument("--real-start-max-delta", type=float, default=0.12)
    parser.add_argument("--real-control-hz", type=float, default=30.0)
    parser.add_argument("--real-max-delta-per-step", type=float, default=0.1)
    parser.add_argument("--render-mode", choices=["none", "rgb_array", "human"], default="none")
    parser.add_argument("--human-render-every", type=int, default=1)
    parser.add_argument("--wait-before-close", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--record-every", type=int, default=3)
    parser.add_argument("--max-grasp-candidates", type=int, default=72)
    parser.add_argument("--square-wall-grasp-edge-axis", choices=["any", "x", "y"], default="any")
    parser.add_argument("--diversify-wall-grasp-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-chain-screening", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-chain-top-grasp-candidates", type=int, default=6)
    parser.add_argument("--fast-chain-top-release-candidates", type=int, default=6)
    parser.add_argument("--fast-chain-ik-seeds", type=int, default=16)
    parser.add_argument("--fast-chain-allow-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-chain-pair-probe-candidates", type=int, default=0)
    parser.add_argument("--fast-chain-stop-pair-probe-after-success", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fast-chain-min-successful-pair-probes", type=int, default=0)
    parser.add_argument("--fast-chain-center-distance-weight", type=float, default=0.28)
    parser.add_argument("--fast-chain-release-probe-weight", type=float, default=0.6)
    parser.add_argument("--fast-chain-supported-wall-max-actor-to-tcp-y", type=float, default=0.0)
    parser.add_argument("--use-cuda-graph-batch-ik", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cuda-graph-batch-ik-max-batch", type=int, default=128)
    parser.add_argument("--cuda-graph-batch-ik-fixed-batch-size", type=int, default=0)
    parser.add_argument("--prefer-center-wall-grasp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-center-grasp-yaw-candidates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wall-grasp-center-offset", type=float, default=0.004)
    parser.add_argument("--wall-grasp-center-axis-keep-radius", type=float, default=0.012)
    parser.add_argument("--max-center-wall-grasp-candidates", type=int, default=12)
    parser.add_argument("--wall-grasp-diversify-pool-size", type=int, default=512)
    parser.add_argument("--wall-grasp-prior-mode", choices=["none", "mined_success_v1"], default="none")
    parser.add_argument("--wall-grasp-prior-max-candidates", type=int, default=0)
    parser.add_argument("--grasp-quality-max-tcp-position-delta", type=float, default=0.0)
    parser.add_argument("--grasp-quality-max-tcp-orientation-delta-deg", type=float, default=0.0)
    parser.add_argument("--grasp-quality-min-finger-force", type=float, default=0.0)
    parser.add_argument("--grasp-quality-max-force-balance-error", type=float, default=0.0)
    parser.add_argument("--grasp-quality-allow-live-lock-recovery", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grasp-quality-live-lock-max-tcp-position-delta", type=float, default=0.0)
    parser.add_argument("--grasp-candidate-start-index", type=int, default=0)
    parser.add_argument("--grasp-candidate-start-indices", default="")
    parser.add_argument("--ik-seeds", type=int, default=64)
    parser.add_argument("--ik-position-threshold", type=float, default=0.008)
    parser.add_argument("--ik-rotation-threshold", type=float, default=0.12)
    parser.add_argument("--initial-steps", type=int, default=2)
    parser.add_argument("--close-steps", type=int, default=18)
    parser.add_argument("--open-steps", type=int, default=12)
    parser.add_argument("--pre-open-hold-steps", type=int, default=0)
    parser.add_argument("--disable-held-role-magnets-until-release", action="store_true")
    parser.add_argument("--disable-unbuilt-role-magnets", action="store_true")
    parser.add_argument("--defer-top-lid-release-screen", action="store_true")
    parser.add_argument("--top-lid-prefer-tilted-grasp", action="store_true")
    parser.add_argument("--allow-top-lid-preplace-drop", action="store_true")
    parser.add_argument("--force-top-lid-preplace-drop", action="store_true")
    parser.add_argument("--top-lid-drop-retreat-height", type=float, default=0.06)
    parser.add_argument("--top-lid-attach-distance", type=float, default=0.0)
    parser.add_argument("--top-lid-attract-distance", type=float, default=0.0)
    parser.add_argument("--top-lid-detach-distance", type=float, default=0.0)
    parser.add_argument("--top-lid-normal-torque-stiffness", type=float, default=0.0)
    parser.add_argument("--top-lid-normal-torque-limit", type=float, default=0.0)
    parser.add_argument("--top-lid-drive-angular-stiffness", type=float, default=0.0)
    parser.add_argument("--top-lid-drive-angular-damping", type=float, default=0.0)
    parser.add_argument("--top-lid-drive-angular-force-limit", type=float, default=0.0)
    parser.add_argument("--force-top-lid-connections-at-release", action="store_true")
    parser.add_argument("--force-top-lid-max-point-error", type=float, default=0.0)
    parser.add_argument("--enable-top-lid-capture-after-open", action="store_true")
    parser.add_argument("--enable-capture-after-open-roles", default="")
    parser.add_argument("--capture-after-open-min-connection-potential", type=int, default=0)
    parser.add_argument("--top-lid-release-edge", default="")
    parser.add_argument("--top-lid-release-target-role", default="front_wall")
    parser.add_argument("--top-lid-release-target-edge", default="top_edge")
    parser.add_argument("--top-lid-enable-all-connections-after-hinge", action="store_true")
    parser.add_argument("--top-lid-all-connections-hold-steps", type=int, default=80)
    parser.add_argument("--top-lid-desired-active-connections", type=int, default=0)
    parser.add_argument("--live-actor-to-tcp-after-lift", dest="use_live_actor_to_tcp_after_lift", action="store_true")
    parser.add_argument("--no-live-actor-to-tcp-after-lift", dest="use_live_actor_to_tcp_after_lift", action="store_false")
    parser.set_defaults(use_live_actor_to_tcp_after_lift=True)
    parser.add_argument("--live-actor-to-tcp-after-lift-min-connection-potential", type=int, default=0)
    parser.add_argument("--allow-nominal-release-fallback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-nominal-release-fallback-max-connection-potential", type=int, default=0)
    parser.add_argument("--move-steps", type=int, default=22)
    parser.add_argument("--short-steps", type=int, default=18)
    parser.add_argument("--release-steps", type=int, default=24)
    parser.add_argument("--max-joint-step", type=float, default=0.06)
    parser.add_argument("--max-segment-steps", type=int, default=420)
    parser.add_argument("--max-existing-path-waypoints", type=int, default=0)
    parser.add_argument("--free-space-motion-window", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--free-space-pregrasp-max-waypoints", type=int, default=12)
    parser.add_argument("--free-space-return-neutral-steps", type=int, default=4)
    parser.add_argument("--free-space-return-neutral-max-joint-step", type=float, default=0.12)
    parser.add_argument("--grasp-max-joint-delta", type=float, default=3.0)
    parser.add_argument("--lift-max-joint-delta", type=float, default=0.0)
    parser.add_argument("--release-correction-attempts", type=int, default=0)
    parser.add_argument("--release-correction-steps", type=int, default=8)
    parser.add_argument("--release-correction-position-threshold", type=float, default=0.02)
    parser.add_argument("--release-correction-orientation-threshold-deg", type=float, default=15.0)
    parser.add_argument("--release-correction-revert-tolerance", type=float, default=0.01)
    parser.add_argument("--max-recoverable-release-position-error", type=float, default=0.18)
    parser.add_argument("--max-recoverable-release-orientation-error-deg", type=float, default=90.0)
    parser.add_argument("--pre-open-connection-correction-attempts", type=int, default=0)
    parser.add_argument("--pre-open-connection-correction-steps", type=int, default=12)
    parser.add_argument("--pre-open-connection-correction-hold-steps", type=int, default=20)
    parser.add_argument("--pre-open-connection-correction-max-joint-delta", type=float, default=1.4)
    parser.add_argument("--pre-open-min-active-connections", type=int, default=1)
    parser.add_argument("--pre-open-connection-correction-max-pose-error", type=float, default=0.08)
    parser.add_argument("--pre-open-connection-correction-max-orientation-error-deg", type=float, default=70.0)
    parser.add_argument("--pre-open-connection-correction-trigger-pose-error", type=float, default=0.012)
    parser.add_argument("--pre-open-connection-correction-trigger-orientation-error-deg", type=float, default=5.0)
    parser.add_argument("--require-active-connection-before-open", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-open-gripper-value", type=float, default=OPEN_GRIPPER)
    parser.add_argument("--attach-held-payload-for-release-planning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--top-lid-release-open-gripper-value", type=float, default=0.0)
    parser.add_argument("--top-lid-hinge-tilt-degs", default="")
    parser.add_argument("--top-lid-hinge-lift-mms", default="0.0")
    parser.add_argument("--top-lid-hinge-inward-mms", default="0.0")
    parser.add_argument("--top-lid-hinge-x-mms", default="0.0")
    parser.add_argument("--tcp-retreat-after-open-roles", default="top_lid")
    parser.add_argument("--tcp-retreat-distance", type=float, default=0.02)
    parser.add_argument("--tcp-retreat-direction-sign", type=float, default=-1.0)
    parser.add_argument("--tcp-retreat-max-joint-delta", type=float, default=1.2)
    parser.add_argument("--full-open-after-retreat-steps", type=int, default=0)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--final-hold-steps", type=int, default=2)
    parser.add_argument("--stability-steps", type=int, default=18)
    parser.add_argument("--final-all-roles-stability-steps", type=int, default=50)
    parser.add_argument("--validate-all-completed-after-each-role", action="store_true")
    parser.add_argument("--all-roles-max-position-error", type=float, default=0.035)
    parser.add_argument("--all-roles-max-orientation-error-deg", type=float, default=35.0)
    parser.add_argument("--all-roles-max-drift-position", type=float, default=0.008)
    parser.add_argument("--all-roles-max-drift-orientation-deg", type=float, default=5.0)
    parser.add_argument("--all-roles-max-linear-speed", type=float, default=0.08)
    parser.add_argument("--all-roles-max-angular-speed", type=float, default=1.0)
    parser.add_argument("--final-max-connection-point-error", type=float, default=0.0)
    parser.add_argument("--final-max-connection-normal-angle-deg", type=float, default=0.0)
    parser.add_argument("--final-max-connection-edge-angle-deg", type=float, default=0.0)
    parser.add_argument("--return-home-steps", type=int, default=0)
    parser.add_argument("--return-neutral-after-role", action="store_true")
    parser.add_argument("--return-neutral-steps", type=int, default=24)
    parser.add_argument("--return-neutral-max-joint-step", type=float, default=0.0)
    parser.add_argument("--return-neutral-skip-final-role", action="store_true")
    parser.add_argument("--return-neutral-motion-plan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return-neutral-motion-plan-timeout", type=float, default=2.5)
    parser.add_argument("--return-neutral-motion-plan-max-attempts", type=int, default=1)
    parser.add_argument("--return-neutral-motion-plan-enable-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return-neutral-motion-plan-graph-seeds", type=int, default=1)
    parser.add_argument("--return-neutral-motion-plan-max-waypoints", type=int, default=0)
    parser.add_argument("--next-cycle-plan-prefetch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--anchor-floor-during-build", action="store_true")
    parser.add_argument("--floor-anchor-stiffness", type=float, default=1200.0)
    parser.add_argument("--floor-anchor-damping", type=float, default=90.0)
    parser.add_argument("--floor-anchor-force-limit", type=float, default=35.0)
    parser.add_argument("--lift-height", type=float, default=0.12)
    parser.add_argument("--preplace-height", type=float, default=0.055)
    parser.add_argument(
        "--wall-release-approach-mode",
        choices=["auto", "top_down", "side_push", "both"],
        default="auto",
    )
    parser.add_argument("--wall-release-side-approach-distance", type=float, default=0.035)
    parser.add_argument("--wall-release-side-approach-lift", type=float, default=0.010)
    parser.add_argument("--pregrasp-timeout", type=float, default=2.5)
    parser.add_argument("--pregrasp-enable-graph", action="store_true")
    parser.add_argument("--pregrasp-graph-fallback-only", action="store_true")
    parser.add_argument("--pregrasp-max-attempts", type=int, default=1)
    parser.add_argument("--pregrasp-graph-seeds", type=int, default=1)
    parser.add_argument("--max-release-candidates", type=int, default=24)
    parser.add_argument("--release-candidate-index", type=int, default=6)
    parser.add_argument("--release-candidate-indices", default="")
    parser.add_argument("--release-prefer-candidate-index-order-for-multi-connection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wall-release-profile", choices=["legacy", "generic", "fixed_top_down"], default="legacy")
    parser.add_argument("--fixed-top-down-open-gap", type=float, default=0.008)
    parser.add_argument("--wall-grasp-extra-tilt-degs", default="")
    parser.add_argument("--wall-grasp-extra-tilt-max-abs-deg", type=float, default=30.0)
    parser.add_argument("--fixed-top-down-extra-tilt-degs", default="")
    parser.add_argument("--fixed-top-down-extra-tilt-max-abs-deg", type=float, default=30.0)
    parser.add_argument("--fixed-top-down-extra-tilt-lift-mms", default="1.0,2.0")
    parser.add_argument("--fixed-top-down-extra-tilt-normal-bias-mms", default="0.0,1.5,-1.5")
    parser.add_argument("--fixed-single-connection-release-min-clearance", type=float, default=0.0)
    parser.add_argument("--release-preplace-max-joint-delta", type=float, default=0.0)
    parser.add_argument("--release-max-joint-delta", type=float, default=2.2)
    parser.add_argument("--release-score-preplace-joint-weight", type=float, default=0.0050)
    parser.add_argument("--release-score-place-joint-weight", type=float, default=0.0015)
    parser.add_argument("--release-motion-plan-preplace", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-motion-plan-on-branch-jump", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-motion-plan-on-branch-jump-min-connection-potential", type=int, default=0)
    parser.add_argument("--release-motion-plan-timeout", type=float, default=4.0)
    parser.add_argument("--release-motion-plan-max-waypoints", type=int, default=24)
    parser.add_argument("--enable-release-yaw-candidates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-robust-actor-to-tcp-y-offsets", default="0.0")
    parser.add_argument("--release-robust-actor-to-tcp-x-offsets", default="0.0")
    parser.add_argument("--release-robust-actor-to-tcp-z-offsets", default="0.0")
    parser.add_argument("--release-robust-actor-to-tcp-yaw-deg-offsets", default="0.0")
    parser.add_argument("--release-robust-min-variant-successes", type=int, default=1)
    parser.add_argument("--release-robust-score-weight", type=float, default=0.0)
    parser.add_argument("--release-robust-require-same-release-index", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-robust-prefer-best-variant", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-robust-preferred-actor-to-tcp-y-offset", type=float, default=0.0)
    parser.add_argument("--release-robust-require-preferred-actor-to-tcp-y-offset", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-robust-require-preferred-max-connection-potential", type=int, default=0)
    parser.add_argument("--release-robust-early-stop-on-first-success", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-screen-max-preplace-joint-delta", type=float, default=0.0)
    parser.add_argument("--release-screen-single-connection-max-place-joint-delta", type=float, default=0.0)
    parser.add_argument("--release-screen-max-predicted-actor-position-error", type=float, default=0.0)
    parser.add_argument("--release-screen-max-predicted-actor-orientation-error-deg", type=float, default=0.0)
    parser.add_argument("--candidate-screen-max-release-failures", type=int, default=0)
    parser.add_argument("--candidate-screen-max-no-robust-failures", type=int, default=0)
    parser.add_argument("--release-prediction-gate-fallback-min-connection-potential", type=int, default=0)
    parser.add_argument("--release-ignore-roles", default="floor")
    parser.add_argument("--release-ignore-roles-by-role", default="")
    parser.add_argument("--release-ignore-roles-as-collision-exclusions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-cached-pair-release", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--runtime-collision-monitor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--runtime-collision-monitor-period", type=int, default=5)
    parser.add_argument("--runtime-collision-monitor-force-threshold", type=float, default=0.05)
    parser.add_argument("--runtime-collision-monitor-max-events", type=int, default=24)
    parser.add_argument("--runtime-collision-monitor-include-floor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze-non-current-roles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-non-current-include-floor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-non-current-enforce-every-n-steps", type=int, default=0)
    parser.add_argument("--lock-active-staged-object-until-grasp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lock-held-actor-after-grasp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--held-actor-lock-reference", choices=["live", "nominal"], default="live")
    parser.add_argument("--held-actor-unlock-stage", choices=["before_open_gripper", "after_open_gripper"], default="before_open_gripper")
    parser.add_argument("--retreat-max-wrist-joint-delta-deg", type=float, default=120.0)
    parser.add_argument("--wall-post-open-pullback-retreat", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wall-post-open-pullback-distance", type=float, default=0.035)
    parser.add_argument("--wall-post-open-pullback-lift", type=float, default=0.025)
    parser.add_argument("--max-position-error", type=float, default=0.035)
    parser.add_argument("--max-orientation-error-deg", type=float, default=35.0)
    parser.add_argument("--min-active-connections", type=int, default=1)
    parser.add_argument("--magnet-attach-distance", type=float, default=0.0)
    parser.add_argument("--magnet-attract-distance", type=float, default=0.0)
    parser.add_argument("--magnet-detach-distance", type=float, default=0.0)
    parser.add_argument("--magnet-edge-sample-half-span", type=float, default=0.0)
    parser.add_argument("--magnet-connect-edge-sample-half-span", type=float, default=0.0)
    parser.add_argument("--magnet-edge-sample-offsets", default="")
    parser.add_argument("--magnet-connect-edge-sample-offsets", default="")
    parser.add_argument("--magnet-attract-stiffness", type=float, default=None)
    parser.add_argument("--magnet-attract-force-limit", type=float, default=None)
    parser.add_argument("--magnet-attract-torque-stiffness", type=float, default=None)
    parser.add_argument("--magnet-attract-torque-limit", type=float, default=None)
    parser.add_argument("--magnet-attract-normal-torque-stiffness", type=float, default=None)
    parser.add_argument("--magnet-attract-normal-torque-limit", type=float, default=None)
    parser.add_argument("--magnet-active-stiffness", type=float, default=None)
    parser.add_argument("--magnet-active-damping", type=float, default=None)
    parser.add_argument("--magnet-active-force-limit", type=float, default=None)
    parser.add_argument("--magnet-active-edge-torque-scale", type=float, default=None)
    parser.add_argument("--magnet-active-edge-torque-min-scale", type=float, default=None)
    parser.add_argument("--magnet-active-normal-torque-scale", type=float, default=None)
    parser.add_argument("--magnet-active-normal-torque-min-scale", type=float, default=None)
    parser.add_argument("--magnet-active-torque-delay-steps", type=int, default=None)
    parser.add_argument("--magnet-active-torque-ramp-steps", type=int, default=None)
    parser.add_argument("--magnet-active-torque-max-point-error", type=float, default=None)
    parser.add_argument("--magnet-floor-support-score", type=float, default=None)
    parser.add_argument("--magnet-support-connection-score-scale", type=float, default=None)
    parser.add_argument("--magnet-multi-connection-support-bonus", type=float, default=None)
    parser.add_argument("--magnet-drive-stiffness", type=float, default=None)
    parser.add_argument("--magnet-drive-damping", type=float, default=None)
    parser.add_argument("--magnet-drive-force-limit", type=float, default=None)
    parser.add_argument("--magnet-drive-angular-stiffness", type=float, default=None)
    parser.add_argument("--magnet-drive-angular-damping", type=float, default=None)
    parser.add_argument("--magnet-drive-angular-force-limit", type=float, default=None)
    args = parser.parse_args()
    if bool(getattr(args, "execute_real", False)):
        real_max_delta = float(getattr(args, "real_max_delta_per_step", 0.0) or 0.0)
        if real_max_delta > 0.0:
            args.max_joint_step = min(float(getattr(args, "max_joint_step", 0.06) or 0.06), real_max_delta)
    result = run(args)
    print(json.dumps({"summary": result["summary"], "final": result["final"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
