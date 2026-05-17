from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sapien
from transforms3d.quaternions import mat2quat, quat2mat

from curobo_rm75_planner import DEFAULT_CUROBO_ROOT, RM75CuRoboPlanner, RM75CuRoboPlannerConfig
from jimu_pick_cube_env import PLATE_SIZE
from record_realman_edge_grasp_open_cube import (
    BUILD_ROLES,
    CLOSED_GRIPPER,
    OPEN_GRIPPER,
    RM75_HOME,
    _active_connection_count_for_role,
    _actor_pose,
    _axis_angle_matrix,
    _grasp_report,
    _initialize_staged_open_cube,
    _make_env,
    _normalize,
    _offset_along_tcp_z,
    _offset_world,
    _pose_arrays,
    _pose_error,
    _pose_to_report,
    _set_robot_qpos,
    _stage_pose,
    _step_action,
    _tcp_pose,
    _wall_release_actor_candidates,
    _world_obstacles,
    _world_to_robot_base,
)


@dataclass(frozen=True)
class GraspCandidate:
    label: str
    local_x: float
    local_y: float
    thin_bias: float
    approach_bias: float
    yaw_deg: float
    pregrasp_distance: float
    approach: str = "top_down"
    approach_tilt_deg: float = 0.0


def _make_logger(out_dir: Path):
    started = time.perf_counter()
    log_path = out_dir / "phase_log.txt"

    def log(message: str) -> None:
        line = f"[plan_single_wall +{time.perf_counter() - started:7.3f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return log


def _deadline(started: float, budget_sec: float) -> float:
    return float(started) + max(float(budget_sec), 0.1)


def _time_left(deadline: float) -> float:
    return float(deadline) - time.perf_counter()


def _candidate_tcp_for_edge_grasp(actor: Any, candidate: GraspCandidate) -> sapien.Pose:
    pose = _actor_pose(actor)
    position, quaternion = _pose_arrays(pose)
    rotation = quat2mat(quaternion).astype(np.float32)
    thin_normal = _normalize(rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    local_point = np.asarray([candidate.local_x, candidate.local_y, 0.0], dtype=np.float32)
    grasp_center = position + rotation @ local_point
    tilted_top_down = candidate.approach == "top_down" and abs(float(candidate.approach_tilt_deg)) > 1e-6
    if candidate.approach == "top_down":
        approaching = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
        if tilted_top_down:
            approaching = approaching - thin_normal * float(np.dot(approaching, thin_normal))
            if float(np.linalg.norm(approaching)) <= 1e-5:
                approaching = rotation @ np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
                approaching = approaching - thin_normal * float(np.dot(approaching, thin_normal))
            if float(np.linalg.norm(approaching)) <= 1e-5:
                approaching = rotation @ np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
                approaching = approaching - thin_normal * float(np.dot(approaching, thin_normal))
            approaching = _normalize(approaching)
            robot_direction = np.asarray([-grasp_center[0], -grasp_center[1], 0.0], dtype=np.float32)
            if float(np.linalg.norm(robot_direction)) <= 1e-5:
                robot_direction = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
            robot_direction = _normalize(robot_direction)
            robot_direction = robot_direction - thin_normal * float(np.dot(robot_direction, thin_normal))
            robot_direction = robot_direction - approaching * float(np.dot(robot_direction, approaching))
            if float(np.linalg.norm(robot_direction)) <= 1e-5:
                robot_direction = rotation @ np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
                robot_direction = robot_direction - thin_normal * float(np.dot(robot_direction, thin_normal))
                robot_direction = robot_direction - approaching * float(np.dot(robot_direction, approaching))
            if float(np.linalg.norm(robot_direction)) <= 1e-5:
                robot_direction = rotation @ np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
                robot_direction = robot_direction - thin_normal * float(np.dot(robot_direction, thin_normal))
                robot_direction = robot_direction - approaching * float(np.dot(robot_direction, approaching))
            if float(np.linalg.norm(robot_direction)) <= 1e-5:
                robot_direction = np.cross(thin_normal, approaching)
            robot_direction = _normalize(robot_direction)
            tilt = float(np.deg2rad(candidate.approach_tilt_deg))
            approaching = _normalize(np.cos(tilt) * approaching + np.sin(tilt) * robot_direction)
            closing_axis = thin_normal
            ortho_axis = _normalize(np.cross(closing_axis, approaching))
        else:
            closing_axis = thin_normal - approaching * float(np.dot(thin_normal, approaching))
            if float(np.linalg.norm(closing_axis)) <= 1e-5:
                closing_axis = rotation @ np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
                closing_axis = closing_axis - approaching * float(np.dot(closing_axis, approaching))
            closing_axis = _normalize(closing_axis)
            ortho_axis = _normalize(np.cross(closing_axis, approaching))
            closing_axis = _normalize(np.cross(approaching, ortho_axis))
    else:
        approach = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(thin_normal, approach))) > 0.96:
            approach = _normalize(rotation @ np.asarray([0.0, -1.0, 0.0], dtype=np.float32))
        ortho_axis = _normalize(np.cross(thin_normal, approach))
        closing_axis = thin_normal
        approaching = _normalize(np.cross(ortho_axis, closing_axis))
    if abs(float(candidate.yaw_deg)) > 1e-6:
        if tilted_top_down:
            delta = _axis_angle_matrix(closing_axis, np.deg2rad(float(candidate.yaw_deg)))
        else:
            delta = _axis_angle_matrix(approaching, np.deg2rad(float(candidate.yaw_deg)))
        ortho_axis = _normalize(delta @ ortho_axis)
        if tilted_top_down:
            approaching = _normalize(delta @ approaching)
        else:
            closing_axis = _normalize(delta @ closing_axis)
    if tilted_top_down:
        closing_axis = _normalize(thin_normal)
        ortho_axis = _normalize(np.cross(closing_axis, approaching))
        approaching = _normalize(np.cross(ortho_axis, closing_axis))
    grasp_center = grasp_center + thin_normal * float(candidate.thin_bias) + approaching * float(candidate.approach_bias)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.stack([ortho_axis, closing_axis, approaching], axis=1)
    matrix[:3, 3] = grasp_center.astype(np.float32)
    return sapien.Pose(p=matrix[:3, 3].tolist(), q=mat2quat(matrix[:3, :3]).astype(np.float32).tolist())


def _pregrasp_pose_for_candidate(grasp_tcp: sapien.Pose, candidate: GraspCandidate) -> sapien.Pose:
    if candidate.approach != "top_down":
        return _offset_along_tcp_z(grasp_tcp, candidate.pregrasp_distance)
    p, q = _pose_arrays(grasp_tcp)
    rotation = quat2mat(q)
    retreat = -rotation[:, 2].astype(np.float32) * float(candidate.pregrasp_distance)
    return sapien.Pose(p=(p + retreat).tolist(), q=q.tolist())


def _tcp_axis_report(pose: sapien.Pose) -> dict[str, Any]:
    _, q = _pose_arrays(pose)
    rotation = quat2mat(q).astype(np.float32)
    approaching = rotation[:, 2].astype(np.float32)
    down = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    return {
        "tcp_x_axis_world": rotation[:, 0].astype(float).tolist(),
        "tcp_y_closing_axis_world": rotation[:, 1].astype(float).tolist(),
        "tcp_z_approach_axis_world": approaching.astype(float).tolist(),
        "tcp_z_down_dot": float(np.dot(approaching, down)),
        "tcp_z_down_angle_deg": float(np.rad2deg(np.arccos(np.clip(np.dot(approaching, down), -1.0, 1.0)))),
    }


def _make_grasp_candidates(max_candidates: int) -> list[GraspCandidate]:
    candidates: list[GraspCandidate] = []
    edge_points = [
        (-PLATE_SIZE / 2.0 + 0.012, 0.0),
        (PLATE_SIZE / 2.0 - 0.012, 0.0),
        (0.0, -PLATE_SIZE / 2.0 + 0.012),
        (0.0, PLATE_SIZE / 2.0 - 0.012),
        (-PLATE_SIZE / 2.0 + 0.018, 0.0),
        (PLATE_SIZE / 2.0 - 0.018, 0.0),
        (0.0, -PLATE_SIZE / 2.0 + 0.018),
        (0.0, PLATE_SIZE / 2.0 - 0.018),
        (-PLATE_SIZE / 2.0 + 0.006, 0.0),
        (PLATE_SIZE / 2.0 - 0.006, 0.0),
        (0.0, -PLATE_SIZE / 2.0 + 0.006),
        (0.0, PLATE_SIZE / 2.0 - 0.006),
    ]
    thin_bias_values = [0.0, 0.0035, -0.0035, 0.007, -0.007]
    approach_bias_values = [-0.024, -0.018, -0.012, -0.006, 0.0]
    wrist_roll_values = [0.0, 4.0, -4.0]
    top_down_tilts = [0.0, 8.0, -8.0, 14.0, -14.0]
    for pregrasp_distance in [0.100, 0.080, 0.120, 0.060]:
        for thin_bias in thin_bias_values:
            for approach_bias in approach_bias_values:
                for tilt_deg in top_down_tilts:
                    for yaw_deg in wrist_roll_values:
                        for local_x, local_y in edge_points:
                            label = (
                                f"top_pre_{pregrasp_distance:.3f}_tilt_{tilt_deg:+.0f}_"
                                f"x_{local_x:+.3f}_y_{local_y:+.3f}_"
                                f"thin_{thin_bias:+.4f}_approach_{approach_bias:+.3f}_yaw_{yaw_deg:+.1f}"
                            )
                            candidates.append(
                                GraspCandidate(
                                    label=label,
                                    local_x=float(local_x),
                                    local_y=float(local_y),
                                    thin_bias=float(thin_bias),
                                    approach_bias=float(approach_bias),
                                    yaw_deg=float(yaw_deg),
                                    pregrasp_distance=float(pregrasp_distance),
                                    approach="top_down",
                                    approach_tilt_deg=float(tilt_deg),
                                )
                            )
    return candidates[: max(int(max_candidates), 1)]


def _current_q(base_env: Any) -> np.ndarray:
    return base_env.agent.robot.get_qpos().detach().cpu().numpy()[0, :7].astype(np.float32)


def _excluded_role_set(exclude_role: str | None = None, exclude_roles: set[str] | None = None) -> set[str]:
    roles = {str(role) for role in (exclude_roles or set()) if str(role)}
    if exclude_role:
        roles.add(str(exclude_role))
    return roles


def _world_obstacles_for_stage(
    base_env: Any,
    locked: dict[str, Any],
    fixtures: list[dict[str, Any]],
    *,
    exclude_role: str | None = None,
    exclude_roles: set[str] | None = None,
) -> list[dict[str, Any]]:
    roles = _excluded_role_set(exclude_role, exclude_roles)
    if not roles:
        return _world_obstacles(base_env, locked, fixtures, exclude_role=None)
    obstacles = _world_obstacles(base_env, locked, fixtures, exclude_role=None)
    excluded_names = {f"plate_{role}" for role in roles}
    return [obstacle for obstacle in obstacles if str(obstacle.get("name", "")) not in excluded_names]


def _release_collision_exclude_roles(args: argparse.Namespace, role: str) -> set[str]:
    configured: set[str] = set()
    if bool(getattr(args, "release_ignore_roles_as_collision_exclusions", True)):
        configured = {
            item.strip()
            for item in str(getattr(args, "release_ignore_roles", "")).split(",")
            if item.strip()
        }
    configured.add(str(role))
    return configured


def _interpolate(start: np.ndarray, goal: np.ndarray, steps: int) -> np.ndarray:
    count = max(int(steps), 2)
    return np.stack(
        [
            (1.0 - float(alpha)) * start.astype(np.float32) + float(alpha) * goal.astype(np.float32)
            for alpha in np.linspace(0.0, 1.0, count, dtype=np.float32)[1:]
        ],
        axis=0,
    ).astype(np.float32)


def _wrapped_joint_delta(q: np.ndarray, reference_q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(-1)[:7]
    reference_q = np.asarray(reference_q, dtype=np.float32).reshape(-1)[:7]
    return ((q - reference_q + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _joint_distance(q: np.ndarray, reference_q: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float32).reshape(-1)[:7]
    reference_q = np.asarray(reference_q, dtype=np.float32).reshape(-1)[:7]
    return float(np.linalg.norm(q - reference_q))


def _solve_ik(
    *,
    planner: RM75CuRoboPlanner,
    base_env: Any,
    locked: dict[str, Any],
    fixtures: list[dict[str, Any]],
    exclude_role: str | None,
    target_pose: sapien.Pose,
    num_seeds: int,
    start_q: np.ndarray | None = None,
    exclude_roles: set[str] | None = None,
) -> tuple[bool, np.ndarray | None, dict[str, Any]]:
    obstacles = _world_obstacles_for_stage(
        base_env,
        locked,
        fixtures,
        exclude_role=exclude_role,
        exclude_roles=exclude_roles,
    )
    planner.set_world_from_obstacles(cuboids=obstacles)
    use_start_q = _current_q(base_env) if start_q is None else np.asarray(start_q, dtype=np.float32).reshape(7)
    result = planner.solve_ik(use_start_q, _world_to_robot_base(base_env, target_pose), num_seeds=int(num_seeds))
    report = {
        "success": bool(result.success),
        "status": result.status,
        "solve_time": float(result.solve_time),
        "ik_time": float(result.ik_time),
        "obstacle_count": len(obstacles),
        "debug": result.debug,
    }
    if not result.success or result.goal_joint is None:
        return False, None, report
    goal_q = np.asarray(result.goal_joint, dtype=np.float32).reshape(-1)[:7]
    report["joint_distance_to_start"] = _joint_distance(goal_q, use_start_q)
    return True, goal_q, report


def _plan_motion_to_pose(
    *,
    planner: RM75CuRoboPlanner,
    base_env: Any,
    locked: dict[str, Any],
    fixtures: list[dict[str, Any]],
    exclude_role: str | None,
    target_pose: sapien.Pose,
    start_q: np.ndarray,
    timeout: float,
    num_seeds: int,
    exclude_roles: set[str] | None = None,
    enable_graph: bool = False,
    max_attempts: int = 1,
    num_graph_seeds: int = 1,
) -> tuple[bool, np.ndarray | None, dict[str, Any]]:
    excluded_roles = _excluded_role_set(exclude_role, exclude_roles)
    obstacles = _world_obstacles_for_stage(
        base_env,
        locked,
        fixtures,
        exclude_role=exclude_role,
        exclude_roles=exclude_roles,
    )
    planner.set_world_from_obstacles(cuboids=obstacles)
    result = planner.plan_to_pose(
        np.asarray(start_q, dtype=np.float32).reshape(7),
        _world_to_robot_base(base_env, target_pose),
        enable_graph=bool(enable_graph),
        max_attempts=int(max_attempts),
        timeout=float(timeout),
        num_ik_seeds=int(num_seeds),
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
        "planner_mode": "motion_gen_collision_checked",
        "excluded_roles": sorted(excluded_roles),
        "target_object_in_world": not excluded_roles,
        "enable_graph": bool(enable_graph),
        "max_attempts": int(max_attempts),
        "num_graph_seeds": int(num_graph_seeds),
    }
    if not result.success or result.joint_path is None:
        return False, None, report
    return True, np.asarray(result.joint_path, dtype=np.float32).reshape(-1, 7), report


def _add_joint_segment(
    *,
    env: Any,
    arrays: dict[str, np.ndarray],
    segments: list[dict[str, Any]],
    name: str,
    goal_q: np.ndarray,
    gripper: float,
    steps: int,
    action_repeat: int,
    final_hold: int,
) -> dict[str, Any]:
    start_q = _current_q(env.unwrapped)
    path = _interpolate(start_q, np.asarray(goal_q, dtype=np.float32).reshape(7), steps)
    key = f"q_{len([item for item in segments if item.get('type') == 'joint_path']):03d}_{name}"
    arrays[key] = path.astype(np.float32)
    for target in path:
        for index in range(max(int(action_repeat), 1)):
            _step_action(env, target[:7], float(gripper), None, 1, index)
    final = path[-1, :7]
    for index in range(max(int(final_hold), 0)):
        _step_action(env, final, float(gripper), None, 1, index)
    segment = {
        "type": "joint_path",
        "name": name,
        "array_key": key,
        "gripper": float(gripper),
        "action_repeat": int(action_repeat),
        "final_hold": int(final_hold),
        "waypoints": int(path.shape[0]),
    }
    segments.append(segment)
    return segment


def _record_joint_segment(
    *,
    arrays: dict[str, np.ndarray],
    segments: list[dict[str, Any]],
    name: str,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    gripper: float,
    steps: int,
    action_repeat: int,
    final_hold: int,
) -> np.ndarray:
    path = _interpolate(np.asarray(start_q, dtype=np.float32).reshape(7), np.asarray(goal_q, dtype=np.float32).reshape(7), steps)
    key = f"q_{len([item for item in segments if item.get('type') == 'joint_path']):03d}_{name}"
    arrays[key] = path.astype(np.float32)
    segments.append(
        {
            "type": "joint_path",
            "name": name,
            "array_key": key,
            "gripper": float(gripper),
            "action_repeat": int(action_repeat),
            "final_hold": int(final_hold),
            "waypoints": int(path.shape[0]),
        }
    )
    return path[-1, :7].astype(np.float32)


def _record_existing_joint_path(
    *,
    arrays: dict[str, np.ndarray],
    segments: list[dict[str, Any]],
    name: str,
    path: np.ndarray,
    gripper: float,
    action_repeat: int,
    final_hold: int,
    no_collapse: bool = True,
) -> np.ndarray:
    path = np.asarray(path, dtype=np.float32).reshape(-1, 7)
    key = f"q_{len([item for item in segments if item.get('type') == 'joint_path']):03d}_{name}"
    arrays[key] = path.astype(np.float32)
    segment: dict[str, Any] = {
        "type": "joint_path",
        "name": name,
        "array_key": key,
        "gripper": float(gripper),
        "action_repeat": int(action_repeat),
        "final_hold": int(final_hold),
        "waypoints": int(path.shape[0]),
    }
    if no_collapse:
        segment["no_collapse"] = True
    segments.append(segment)
    return path[-1, :7].astype(np.float32)


def _record_hold_segment(segments: list[dict[str, Any]], name: str, gripper: float, steps: int) -> None:
    if int(steps) <= 0:
        return
    segments.append({"type": "hold", "name": name, "gripper": float(gripper), "steps": int(steps)})


def _record_ramp_segment(
    segments: list[dict[str, Any]],
    name: str,
    start_gripper: float,
    end_gripper: float,
    steps: int,
) -> None:
    segments.append(
        {
            "type": "gripper_ramp",
            "name": name,
            "start_gripper": float(start_gripper),
            "end_gripper": float(end_gripper),
            "steps": int(steps),
        }
    )


def _record_settle_segment(segments: list[dict[str, Any]], name: str, steps: int) -> None:
    segments.append({"type": "settle_zero_action", "name": name, "steps": int(steps)})


def _add_hold_segment(env: Any, segments: list[dict[str, Any]], name: str, gripper: float, steps: int) -> None:
    if int(steps) <= 0:
        return
    q = _current_q(env.unwrapped)
    for index in range(max(int(steps), 0)):
        _step_action(env, q, float(gripper), None, 1, index)
    segments.append({"type": "hold", "name": name, "gripper": float(gripper), "steps": int(steps)})


def _add_ramp_segment(
    env: Any,
    segments: list[dict[str, Any]],
    name: str,
    start_gripper: float,
    end_gripper: float,
    steps: int,
) -> None:
    q = _current_q(env.unwrapped)
    for index, gripper in enumerate(np.linspace(float(start_gripper), float(end_gripper), max(int(steps), 1), dtype=np.float32)):
        _step_action(env, q, float(gripper), None, 1, index)
    segments.append(
        {
            "type": "gripper_ramp",
            "name": name,
            "start_gripper": float(start_gripper),
            "end_gripper": float(end_gripper),
            "steps": int(steps),
        }
    )


def _add_settle_segment(env: Any, segments: list[dict[str, Any]], name: str, steps: int) -> None:
    q = _current_q(env.unwrapped)
    for index in range(max(int(steps), 0)):
        _step_action(env, q, OPEN_GRIPPER, None, 1, index)
    segments.append({"type": "settle_zero_action", "name": name, "steps": int(steps)})


def _make_attempt_env() -> tuple[Any, dict[str, sapien.Pose], dict[str, Any], list[dict[str, Any]]]:
    env = _make_env(False)
    env.reset()
    base_env = env.unwrapped
    _set_robot_qpos(base_env, RM75_HOME, gripper_open=True)
    targets, locked, fixtures = _initialize_staged_open_cube(base_env)
    return env, targets, locked, fixtures


def _try_candidate(
    *,
    candidate: GraspCandidate,
    planner: RM75CuRoboPlanner,
    args: argparse.Namespace,
    deadline: float,
    log: Any,
) -> tuple[bool, dict[str, Any], dict[str, np.ndarray]]:
    env = None
    arrays: dict[str, np.ndarray] = {}
    segments: list[dict[str, Any]] = []
    report: dict[str, Any] = {"candidate": candidate.__dict__, "segments": segments}
    role = args.role
    try:
        if _time_left(deadline) <= 0.0:
            report["failed_at"] = "budget_expired_before_env"
            return False, report, arrays
        log(f"{candidate.label}: creating env")
        env, targets, locked, fixtures = _make_attempt_env()
        log(f"{candidate.label}: env ready")
        base_env = env.unwrapped
        actor = locked[role].actor
        log(f"{candidate.label}: initial hold")
        _add_hold_segment(env, segments, "initial_hold", OPEN_GRIPPER, args.initial_steps)
        grasp_tcp = _candidate_tcp_for_edge_grasp(actor, candidate)
        pregrasp = _pregrasp_pose_for_candidate(grasp_tcp, candidate)
        report["grasp_tcp_pose"] = _pose_to_report(grasp_tcp)
        report["grasp_tcp_axes"] = _tcp_axis_report(grasp_tcp)
        report["pregrasp_pose"] = _pose_to_report(pregrasp)
        for label, pose, gripper, steps in [
            ("pregrasp", pregrasp, OPEN_GRIPPER, args.move_steps),
            ("edge_grasp", grasp_tcp, OPEN_GRIPPER, args.short_steps),
        ]:
            if _time_left(deadline) <= 0.0:
                report["failed_at"] = f"budget_expired_before_{label}"
                return False, report, arrays
            log(f"{candidate.label}: solving {label}")
            ok, q_goal, ik_report = _solve_ik(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                exclude_role=role,
                target_pose=pose,
                num_seeds=args.ik_seeds,
            )
            report[f"{label}_ik"] = ik_report
            if not ok or q_goal is None:
                report["failed_at"] = f"{label}_ik"
                return False, report, arrays
            log(f"{candidate.label}: executing {label}")
            _add_joint_segment(
                env=env,
                arrays=arrays,
                segments=segments,
                name=label,
                goal_q=q_goal,
                gripper=gripper,
                steps=steps,
                action_repeat=args.action_repeat,
                final_hold=args.final_hold_steps,
            )
            log(f"{candidate.label}: executed {label}")
        if _time_left(deadline) <= 0.0:
            report["failed_at"] = "budget_expired_before_close"
            return False, report, arrays
        log(f"{candidate.label}: closing gripper")
        _add_hold_segment(env, segments, "close_gripper", CLOSED_GRIPPER, args.close_steps)
        report["grasp_after_close"] = _grasp_report(base_env, actor)
        log(f"{candidate.label}: grasp after close {report['grasp_after_close']}")
        if not report["grasp_after_close"]["is_grasped"]:
            report["failed_at"] = "physical_grasp_check"
            return False, report, arrays
        lift = _offset_world(grasp_tcp, np.asarray([0.0, 0.0, args.lift_height], dtype=np.float32))
        if _time_left(deadline) <= 0.0:
            report["failed_at"] = "budget_expired_before_lift"
            return False, report, arrays
        log(f"{candidate.label}: solving lift")
        ok, q_lift, ik_report = _solve_ik(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            exclude_role=role,
            target_pose=lift,
            num_seeds=args.ik_seeds,
        )
        report["lift_ik"] = ik_report
        if not ok or q_lift is None:
            report["failed_at"] = "lift_ik"
            return False, report, arrays
        before_lift_z = float(_pose_arrays(_actor_pose(actor))[0][2])
        log(f"{candidate.label}: executing lift")
        _add_joint_segment(
            env=env,
            arrays=arrays,
            segments=segments,
            name="lift",
            goal_q=q_lift,
            gripper=CLOSED_GRIPPER,
            steps=args.move_steps,
            action_repeat=args.action_repeat,
            final_hold=args.final_hold_steps,
        )
        log(f"{candidate.label}: executed lift")
        after_lift_z = float(_pose_arrays(_actor_pose(actor))[0][2])
        report["grasp_after_lift"] = _grasp_report(base_env, actor)
        report["lift_delta_m"] = after_lift_z - before_lift_z
        if not report["grasp_after_lift"]["is_grasped"] or report["lift_delta_m"] < args.min_lift_delta:
            report["failed_at"] = "physical_lift_check"
            return False, report, arrays
        actor_to_tcp = _actor_pose(actor).inv() * _tcp_pose(base_env)
        if _time_left(deadline) <= 0.0:
            report["failed_at"] = "budget_expired_before_release"
            return False, report, arrays
        log(f"{candidate.label}: selecting release")
        release_report = _select_release_with_live_rollout(
            env=env,
            planner=planner,
            locked=locked,
            fixtures=fixtures,
            actor=actor,
            role=role,
            actor_to_tcp=actor_to_tcp,
            target_actor_pose=targets[role],
            arrays=arrays,
            segments=segments,
            args=args,
        )
        report["release"] = release_report
        if not release_report.get("success"):
            report["failed_at"] = "release_rollout"
            return False, report, arrays
        _add_hold_segment(env, segments, "pre_open_magnetic_hold", CLOSED_GRIPPER, args.pre_open_hold_steps)
        report["snap_report_before_open"] = base_env.get_magnetic_snap_report()
        report["pose_error_before_open"] = _pose_error(actor, targets[role])
        log(f"{candidate.label}: opening gripper")
        _add_ramp_segment(env, segments, "open_gripper", CLOSED_GRIPPER, OPEN_GRIPPER, args.open_steps)
        if args.plan_retreat:
            retreat_pose = _offset_world(release_report["target_tcp_pose"], np.asarray([0.0, 0.0, args.retreat_height], dtype=np.float32))
            log(f"{candidate.label}: solving retreat")
            ok, q_retreat, ik_report = _solve_ik(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                exclude_role=None,
                exclude_roles=_release_collision_exclude_roles(args, role),
                target_pose=retreat_pose,
                num_seeds=args.ik_seeds,
            )
            report["retreat_ik"] = ik_report
            if ok and q_retreat is not None:
                _add_joint_segment(
                    env=env,
                    arrays=arrays,
                    segments=segments,
                    name="retreat",
                    goal_q=q_retreat,
                    gripper=OPEN_GRIPPER,
                    steps=args.short_steps,
                    action_repeat=args.action_repeat,
                    final_hold=args.final_hold_steps,
                )
        log(f"{candidate.label}: stability check steps={args.stability_steps}")
        _add_settle_segment(env, segments, "stability_check", args.stability_steps)
        report["pose_error_after_release"] = _pose_error(actor, targets[role])
        report["active_connection_count_for_role"] = _active_connection_count_for_role(base_env, role)
        report["snap_report"] = base_env.get_magnetic_snap_report()
        report["suspended_roles"] = sorted(base_env.magnetic_snap.suspended_roles)
        success = (
            report["pose_error_after_release"]["position_error_m"] <= args.max_position_error
            and report["pose_error_after_release"]["orientation_error_deg"] <= args.max_orientation_error_deg
            and role not in base_env.magnetic_snap.suspended_roles
            and report["active_connection_count_for_role"] >= args.min_active_connections
        )
        report["success"] = bool(success)
        if not success:
            report["failed_at"] = "final_stability_check"
        return bool(success), report, arrays
    except Exception as exc:
        report.update({"success": False, "error_type": type(exc).__name__, "error": str(exc)})
        return False, report, arrays
    finally:
        if env is not None:
            try:
                log(f"{candidate.label}: closing env")
                env.close()
                log(f"{candidate.label}: env closed")
            except Exception as exc:
                log(f"env close failed for {candidate.label}: {type(exc).__name__}: {exc}")


def _select_release_with_live_rollout(
    *,
    env: Any,
    planner: RM75CuRoboPlanner,
    locked: dict[str, Any],
    fixtures: list[dict[str, Any]],
    actor: Any,
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
    release_exclude_roles = _release_collision_exclude_roles(args, role)
    release_max_joint_delta = float(getattr(args, "release_max_joint_delta", 0.0))
    for index, (label, release_actor_pose) in enumerate(_wall_release_actor_candidates(role, target_actor_pose)):
        if index >= int(args.max_release_candidates):
            break
        target_tcp = release_actor_pose * actor_to_tcp
        preplace = _offset_world(target_tcp, np.asarray([0.0, 0.0, args.preplace_height], dtype=np.float32))
        pose_error = _pose_to_pose_error(release_actor_pose, target_actor_pose)
        candidate_report: dict[str, Any] = {
            "index": index,
            "label": label,
            "release_actor_pose": _pose_to_report(release_actor_pose),
            "target_pose_error": pose_error,
        }
        ok, q_preplace, pre_report = _solve_ik(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            exclude_role=role,
            target_pose=preplace,
            num_seeds=args.ik_seeds,
        )
        candidate_report["preplace_ik"] = pre_report
        if not ok or q_preplace is None:
            reports.append(candidate_report)
            continue
        preplace_joint_delta = _joint_distance(q_preplace, _current_q(base_env))
        candidate_report["preplace_joint_delta_from_current"] = float(preplace_joint_delta)
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
        if role == "top_lid" and bool(getattr(args, "force_top_lid_preplace_drop", False)):
            ok = False
            q_place = None
        if not ok or q_place is None:
            if role == "top_lid" and bool(getattr(args, "allow_top_lid_preplace_drop", False)):
                retreat_pose = _offset_world(preplace, np.asarray([0.0, 0.0, float(getattr(args, "top_lid_drop_retreat_height", 0.06))], dtype=np.float32))
                ok_retreat, q_retreat, retreat_report = _solve_ik(
                    planner=planner,
                    base_env=base_env,
                    locked=locked,
                    fixtures=fixtures,
                    exclude_role=role,
                    target_pose=retreat_pose,
                    num_seeds=args.ik_seeds,
                    start_q=q_preplace,
                )
                candidate_report["drop_retreat_ik"] = retreat_report
                candidate_report["preplace_drop_fallback"] = True
                score = pose_error["position_error_m"] + 0.002 * pose_error["orientation_error_deg"] + float(args.preplace_height)
                candidate_report["release_score"] = float(score)
                candidate_report["success"] = True
                feasible.append(
                    {
                        "score": float(score),
                        "index": index,
                        "label": f"{label}_preplace_drop",
                        "release_actor_pose": release_actor_pose,
                        "target_tcp": preplace,
                        "q_preplace": q_retreat if ok_retreat and q_retreat is not None else q_preplace,
                        "q_place": q_preplace,
                    }
                )
            reports.append(candidate_report)
            continue
        place_joint_delta = _joint_distance(q_place, q_preplace)
        candidate_report["place_joint_delta_from_preplace"] = float(place_joint_delta)
        if release_max_joint_delta > 0.0 and place_joint_delta > release_max_joint_delta:
            candidate_report["success"] = False
            candidate_report["failed_at"] = "place_joint_branch_jump"
            candidate_report["release_max_joint_delta"] = release_max_joint_delta
            reports.append(candidate_report)
            continue
        score = pose_error["position_error_m"] + 0.002 * pose_error["orientation_error_deg"]
        candidate_report["release_score"] = float(score)
        candidate_report["success"] = True
        feasible.append(
            {
                "score": float(score),
                "index": index,
                "label": label,
                "release_actor_pose": release_actor_pose,
                "target_tcp": target_tcp,
                "q_preplace": q_preplace,
                "q_place": q_place,
            }
        )
        reports.append(candidate_report)
    if not feasible:
        return {"success": False, "reports": reports}
    forced_index = int(getattr(args, "release_candidate_index", -1))
    forced = [item for item in feasible if int(item["index"]) == forced_index]
    selected = forced[0] if forced else min(feasible, key=lambda item: (float(item["score"]), int(item["index"])))
    _add_joint_segment(
        env=env,
        arrays=arrays,
        segments=segments,
        name=f"preplace_{int(selected['index']):02d}",
        goal_q=np.asarray(selected["q_preplace"], dtype=np.float32),
        gripper=CLOSED_GRIPPER,
        steps=args.move_steps,
        action_repeat=args.action_repeat,
        final_hold=args.final_hold_steps,
    )
    _add_joint_segment(
        env=env,
        arrays=arrays,
        segments=segments,
        name=f"place_{int(selected['index']):02d}",
        goal_q=np.asarray(selected["q_place"], dtype=np.float32),
        gripper=CLOSED_GRIPPER,
        steps=args.release_steps,
        action_repeat=args.action_repeat,
        final_hold=args.final_hold_steps,
    )
    return {
        "success": True,
        "selected": selected["label"],
        "selected_index": selected["index"],
        "selected_score": selected["score"],
        "forced_release_candidate_index": forced_index if forced else None,
        "reports": reports,
        "release_actor_pose": _pose_to_report(selected["release_actor_pose"]),
        "target_tcp_pose": selected["target_tcp"],
        "q_preplace": np.asarray(selected["q_preplace"], dtype=np.float32),
        "q_place": np.asarray(selected["q_place"], dtype=np.float32),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, sapien.Pose):
        return _pose_to_report(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    return value


def _pose_to_pose_error(current: sapien.Pose, target: sapien.Pose) -> dict[str, float]:
    cp, cq = _pose_arrays(current)
    tp, tq = _pose_arrays(target)
    dot = min(max(abs(float(np.dot(cq, tq))), -1.0), 1.0)
    return {
        "position_error_m": float(np.linalg.norm(cp - tp)),
        "orientation_error_deg": float(np.rad2deg(2.0 * np.arccos(dot))),
    }


def _plan_release_without_rollout(
    *,
    base_env: Any,
    planner: RM75CuRoboPlanner,
    locked: dict[str, Any],
    fixtures: list[dict[str, Any]],
    role: str,
    actor_to_tcp: sapien.Pose,
    target_actor_pose: sapien.Pose,
    start_q: np.ndarray,
    arrays: dict[str, np.ndarray],
    segments: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[bool, np.ndarray, dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    release_exclude_roles = _release_collision_exclude_roles(args, role)
    for index, (label, release_actor_pose) in enumerate(_wall_release_actor_candidates(role, target_actor_pose)):
        if index >= int(args.max_release_candidates):
            break
        target_tcp = release_actor_pose * actor_to_tcp
        preplace = _offset_world(target_tcp, np.asarray([0.0, 0.0, args.preplace_height], dtype=np.float32))
        pose_error = _pose_to_pose_error(release_actor_pose, target_actor_pose)
        candidate_report: dict[str, Any] = {
            "index": index,
            "label": label,
            "release_actor_pose": _pose_to_report(release_actor_pose),
            "target_tcp_pose": _pose_to_report(target_tcp),
            "target_pose_error": pose_error,
        }
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
            reports.append(candidate_report)
            continue
        score = pose_error["position_error_m"] + 0.002 * pose_error["orientation_error_deg"]
        candidate_report["release_score"] = float(score)
        candidate_report["success"] = True
        reports.append(candidate_report)
        feasible.append(
            {
                "score": float(score),
                "index": index,
                "label": label,
                "target_tcp": target_tcp,
                "q_preplace": q_preplace,
                "q_place": q_place,
            }
        )
    if feasible:
        forced_index = int(getattr(args, "release_candidate_index", -1))
        forced = [item for item in feasible if int(item["index"]) == forced_index]
        selected = forced[0] if forced else min(feasible, key=lambda item: (float(item["score"]), int(item["index"])))
        current_q = _record_joint_segment(
            arrays=arrays,
            segments=segments,
            name=f"preplace_{int(selected['index']):02d}",
            start_q=start_q,
            goal_q=np.asarray(selected["q_preplace"], dtype=np.float32),
            gripper=CLOSED_GRIPPER,
            steps=args.move_steps,
            action_repeat=args.action_repeat,
            final_hold=args.final_hold_steps,
        )
        current_q = _record_joint_segment(
            arrays=arrays,
            segments=segments,
            name=f"place_{int(selected['index']):02d}",
            start_q=current_q,
            goal_q=np.asarray(selected["q_place"], dtype=np.float32),
            gripper=CLOSED_GRIPPER,
            steps=args.release_steps,
            action_repeat=args.action_repeat,
            final_hold=args.final_hold_steps,
        )
        return (
            True,
            current_q,
            {
                "selected": selected["label"],
                "selected_index": selected["index"],
                "selected_score": selected["score"],
                "forced_release_candidate_index": forced_index if forced else None,
                "reports": reports,
                "target_tcp_pose": selected["target_tcp"],
            },
        )
    return False, start_q.astype(np.float32), {"selected": None, "reports": reports}


def _try_candidate_planning_only(
    *,
    candidate: GraspCandidate,
    planner: RM75CuRoboPlanner,
    args: argparse.Namespace,
    deadline: float,
    log: Any,
) -> tuple[bool, dict[str, Any], dict[str, np.ndarray]]:
    env = None
    arrays: dict[str, np.ndarray] = {}
    segments: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "candidate": candidate.__dict__,
        "segments": segments,
        "mode": "planning_only_no_step_validation",
    }
    role = args.role
    try:
        if _time_left(deadline) <= 0.0:
            report["failed_at"] = "budget_expired_before_env"
            return False, report, arrays
        log(f"{candidate.label}: creating planning env")
        env, targets, locked, fixtures = _make_attempt_env()
        base_env = env.unwrapped
        for placed_role in [item.strip() for item in str(getattr(args, "preplaced_roles", "")).split(",") if item.strip()]:
            if placed_role in locked and placed_role in targets:
                placed_actor = locked[placed_role].actor
                placed_actor.set_pose(targets[placed_role])
                placed_actor.set_linear_velocity(np.zeros(3, dtype=np.float32))
                placed_actor.set_angular_velocity(np.zeros(3, dtype=np.float32))
        actor = locked[role].actor
        if int(getattr(args, "stage_index", -1)) >= 0:
            target_stage = int(args.stage_index)
            source_stage = BUILD_ROLES.index(role) if role in BUILD_ROLES else target_stage
            swap_role = BUILD_ROLES[target_stage] if 0 <= target_stage < len(BUILD_ROLES) else None
            if swap_role and swap_role in locked and swap_role != role:
                swap_actor = locked[swap_role].actor
                swap_actor.set_pose(_stage_pose(swap_role, source_stage, targets))
                swap_actor.set_linear_velocity(np.zeros(3, dtype=np.float32))
                swap_actor.set_angular_velocity(np.zeros(3, dtype=np.float32))
            actor.set_pose(_stage_pose(role, target_stage, targets))
            actor.set_linear_velocity(np.zeros(3, dtype=np.float32))
            actor.set_angular_velocity(np.zeros(3, dtype=np.float32))
        current_q = RM75_HOME.astype(np.float32).copy()
        _record_hold_segment(segments, "initial_hold", OPEN_GRIPPER, args.initial_steps)
        grasp_tcp = _candidate_tcp_for_edge_grasp(actor, candidate)
        pregrasp = _pregrasp_pose_for_candidate(grasp_tcp, candidate)
        report["grasp_tcp_pose"] = _pose_to_report(grasp_tcp)
        report["grasp_tcp_axes"] = _tcp_axis_report(grasp_tcp)
        report["pregrasp_pose"] = _pose_to_report(pregrasp)
        if _time_left(deadline) <= 0.0:
            report["failed_at"] = "budget_expired_before_pregrasp"
            return False, report, arrays
        log(f"{candidate.label}: planning pregrasp with target object as obstacle")
        if args.motion_plan_pregrasp:
            ok, q_path, pregrasp_report = _plan_motion_to_pose(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                exclude_role=None,
                target_pose=pregrasp,
                start_q=current_q,
                timeout=args.pregrasp_timeout,
                num_seeds=args.ik_seeds,
            )
            report["pregrasp_plan"] = pregrasp_report
            if not ok or q_path is None:
                report["failed_at"] = "pregrasp_motion_plan"
                return False, report, arrays
            current_q = _record_existing_joint_path(
                arrays=arrays,
                segments=segments,
                name="pregrasp",
                path=q_path,
                gripper=OPEN_GRIPPER,
                action_repeat=args.action_repeat,
                final_hold=args.final_hold_steps,
                no_collapse=True,
            )
        else:
            ok, q_goal, pregrasp_report = _solve_ik(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                exclude_role=None,
                target_pose=pregrasp,
                num_seeds=args.ik_seeds,
                start_q=current_q,
            )
            report["pregrasp_ik"] = pregrasp_report
            if not ok or q_goal is None:
                report["failed_at"] = "pregrasp_ik"
                return False, report, arrays
            current_q = _record_joint_segment(
                arrays=arrays,
                segments=segments,
                name="pregrasp",
                start_q=current_q,
                goal_q=q_goal,
                gripper=OPEN_GRIPPER,
                steps=args.move_steps,
                action_repeat=args.action_repeat,
                final_hold=args.final_hold_steps,
            )
            segments[-1]["no_collapse"] = True
        if _time_left(deadline) <= 0.0:
            report["failed_at"] = "budget_expired_before_edge_grasp"
            return False, report, arrays
        log(f"{candidate.label}: planning short edge_grasp final approach")
        ok, q_goal, ik_report = _solve_ik(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            exclude_role=role,
            target_pose=grasp_tcp,
            num_seeds=args.ik_seeds,
            start_q=current_q,
        )
        report["edge_grasp_ik"] = ik_report
        if not ok or q_goal is None:
            report["failed_at"] = "edge_grasp_ik"
            return False, report, arrays
        current_q = _record_joint_segment(
            arrays=arrays,
            segments=segments,
            name="edge_grasp",
            start_q=current_q,
            goal_q=q_goal,
            gripper=OPEN_GRIPPER,
            steps=args.short_steps,
            action_repeat=args.action_repeat,
            final_hold=args.final_hold_steps,
        )
        segments[-1]["no_collapse"] = True
        _record_hold_segment(segments, "close_gripper", CLOSED_GRIPPER, args.close_steps)
        if _time_left(deadline) <= 0.0:
            report["failed_at"] = "budget_expired_before_lift"
            return False, report, arrays
        lift = _offset_world(grasp_tcp, np.asarray([0.0, 0.0, args.lift_height], dtype=np.float32))
        log(f"{candidate.label}: planning lift")
        ok, q_lift, ik_report = _solve_ik(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            exclude_role=role,
            target_pose=lift,
            num_seeds=args.ik_seeds,
            start_q=current_q,
        )
        report["lift_ik"] = ik_report
        if not ok or q_lift is None:
            report["failed_at"] = "lift_ik"
            return False, report, arrays
        current_q = _record_joint_segment(
            arrays=arrays,
            segments=segments,
            name="lift",
            start_q=current_q,
            goal_q=q_lift,
            gripper=CLOSED_GRIPPER,
            steps=args.move_steps,
            action_repeat=args.action_repeat,
            final_hold=args.final_hold_steps,
        )
        actor_to_tcp = _actor_pose(actor).inv() * grasp_tcp
        log(f"{candidate.label}: planning release")
        ok, current_q, release_report = _plan_release_without_rollout(
            base_env=base_env,
            planner=planner,
            locked=locked,
            fixtures=fixtures,
            role=role,
            actor_to_tcp=actor_to_tcp,
            target_actor_pose=targets[role],
            start_q=current_q,
            arrays=arrays,
            segments=segments,
            args=args,
        )
        report["release"] = release_report
        if not ok:
            report["failed_at"] = "release_ik"
            return False, report, arrays
        _record_hold_segment(segments, "pre_open_magnetic_hold", CLOSED_GRIPPER, args.pre_open_hold_steps)
        _record_ramp_segment(segments, "open_gripper", CLOSED_GRIPPER, OPEN_GRIPPER, args.open_steps)
        target_tcp_pose = release_report["target_tcp_pose"]
        if args.plan_retreat:
            retreat_pose = _offset_world(target_tcp_pose, np.asarray([0.0, 0.0, args.retreat_height], dtype=np.float32))
            log(f"{candidate.label}: planning retreat")
            ok, q_retreat, ik_report = _solve_ik(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                exclude_role=None,
                exclude_roles=_release_collision_exclude_roles(args, role),
                target_pose=retreat_pose,
                num_seeds=args.ik_seeds,
                start_q=current_q,
            )
            report["retreat_ik"] = ik_report
            if ok and q_retreat is not None:
                current_q = _record_joint_segment(
                    arrays=arrays,
                    segments=segments,
                    name="retreat",
                    start_q=current_q,
                    goal_q=q_retreat,
                    gripper=OPEN_GRIPPER,
                    steps=args.short_steps,
                    action_repeat=args.action_repeat,
                    final_hold=args.final_hold_steps,
                )
        _record_settle_segment(segments, "stability_check", args.stability_steps)
        report["success"] = True
        report["success_definition"] = "ik_feasible_path_saved; run replay_single_wall_path.py for physical step validation"
        return True, report, arrays
    except Exception as exc:
        report.update({"success": False, "error_type": type(exc).__name__, "error": str(exc)})
        return False, report, arrays
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                log(f"planning env close failed for {candidate.label}: {type(exc).__name__}: {exc}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _make_logger(out_dir)
    manifest_path = out_dir / "single_wall_path_manifest.json"
    arrays_path = out_dir / "single_wall_path_arrays.npz"
    summary_path = out_dir / "single_wall_path_search_summary.json"
    search_deadline = _deadline(started, args.planning_budget_sec)
    attempts: list[dict[str, Any]] = []
    best_arrays: dict[str, np.ndarray] = {}
    final: dict[str, Any] = {
        "success": False,
        "path_manifest": str(manifest_path),
        "path_arrays": str(arrays_path),
    }
    planner = None
    try:
        log("creating CuRobo IK planner")
        planner = RM75CuRoboPlanner(
            RM75CuRoboPlannerConfig(
                curobo_root=Path(args.curobo_root),
                robot_cfg_path=Path(args.robot_cfg),
                position_threshold=args.ik_position_threshold,
                rotation_threshold=args.ik_rotation_threshold,
                num_ik_seeds=args.ik_seeds,
                num_trajopt_seeds=1,
                num_graph_seeds=1,
                collision_activation_distance=0.018,
                build_motion_gen=bool(args.motion_plan_pregrasp or args.validate_during_search),
            )
        )
        final["curobo_collision_enabled"] = bool(planner.collision_enabled)
        final["configured_collision_links"] = planner.configured_collision_links
        log("planner ready")
        search_started = time.perf_counter()
        search_deadline = _deadline(search_started, args.planning_budget_sec)
        final["budget_start"] = "after_curobo_planner_ready"
        for index, candidate in enumerate(_make_grasp_candidates(args.max_candidates)):
            if _time_left(search_deadline) <= 0.0:
                log("planning budget exhausted")
                break
            log(f"trying candidate {index}: {candidate.label}, time_left={_time_left(search_deadline):.2f}s")
            if args.validate_during_search:
                success, report, arrays = _try_candidate(
                    candidate=candidate,
                    planner=planner,
                    args=args,
                    deadline=search_deadline,
                    log=log,
                )
            else:
                success, report, arrays = _try_candidate_planning_only(
                    candidate=candidate,
                    planner=planner,
                    args=args,
                    deadline=search_deadline,
                    log=log,
                )
            report["candidate_index"] = index
            attempts.append(_json_ready(report))
            log(f"candidate {index} success={success} failed_at={report.get('failed_at')}")
            if success:
                best_arrays = arrays
                manifest = {
                    "name": "single_wall_physical_path",
                    "created_by": Path(__file__).name,
                    "role": args.role,
                    "robot": "RM75",
                    "control_mode": "pd_joint_pos_abs",
                    "no_kinematic_attach": True,
                    "no_placement_correction": True,
                    "validated_during_search": bool(args.validate_during_search),
                    "physical_replay_required": not bool(args.validate_during_search),
                    "arrays": str(arrays_path),
                    "stage_index": int(args.stage_index),
                    "preplaced_roles": [item.strip() for item in str(args.preplaced_roles).split(",") if item.strip()],
                    "candidate_index": index,
                    "candidate": report.get("candidate"),
                    "segments": report.get("segments", []),
                    "validation": {
                        "physically_grasped": (
                            bool(report.get("grasp_after_close", {}).get("is_grasped", False))
                            if args.validate_during_search
                            else None
                        ),
                        "lift_delta_m": float(report.get("lift_delta_m", 0.0)) if args.validate_during_search else None,
                        "pose_error_after_release": report.get("pose_error_after_release"),
                        "active_connection_count_for_role": report.get("active_connection_count_for_role"),
                        "suspended_roles": report.get("suspended_roles", []),
                    },
                }
                manifest_path.write_text(json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
                np.savez_compressed(arrays_path, **best_arrays)
                final.update(
                    {
                        "success": True,
                        "success_definition": (
                            "physical_rollout_validated"
                            if args.validate_during_search
                            else "ik_feasible_path_saved; physical replay still required"
                        ),
                        "validated_during_search": bool(args.validate_during_search),
                        "accepted_candidate_index": index,
                        "accepted_candidate": candidate.label,
                        "elapsed_sec": round(time.perf_counter() - started, 3),
                    }
                )
                break
    except Exception as exc:
        final.update({"success": False, "error_type": type(exc).__name__, "error": str(exc)})
        log(f"exception: {final}")
    result = {
        "name": "plan_single_wall_path_20s",
        "summary": str(summary_path),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "planning_budget_sec": float(args.planning_budget_sec),
        "attempts": attempts,
        "final": final,
    }
    summary_path.write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "single_wall_path_20s_v1"))
    parser.add_argument("--robot-cfg", default=str(Path(__file__).resolve().parent / "curobo_rm75_config" / "rm75.yml"))
    parser.add_argument("--curobo-root", default=str(DEFAULT_CUROBO_ROOT))
    parser.add_argument("--role", default="right_wall")
    parser.add_argument("--stage-index", type=int, default=-1)
    parser.add_argument("--preplaced-roles", default="")
    parser.add_argument("--planning-budget-sec", type=float, default=20.0)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--max-release-candidates", type=int, default=24)
    parser.add_argument("--ik-seeds", type=int, default=64)
    parser.add_argument("--ik-position-threshold", type=float, default=0.008)
    parser.add_argument("--ik-rotation-threshold", type=float, default=0.12)
    parser.add_argument("--initial-steps", type=int, default=2)
    parser.add_argument("--close-steps", type=int, default=18)
    parser.add_argument("--open-steps", type=int, default=12)
    parser.add_argument("--move-steps", type=int, default=22)
    parser.add_argument("--short-steps", type=int, default=18)
    parser.add_argument("--release-steps", type=int, default=24)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--final-hold-steps", type=int, default=2)
    parser.add_argument("--pre-open-hold-steps", type=int, default=0)
    parser.add_argument("--stability-steps", type=int, default=18)
    parser.add_argument("--lift-height", type=float, default=0.12)
    parser.add_argument("--min-lift-delta", type=float, default=0.03)
    parser.add_argument("--preplace-height", type=float, default=0.055)
    parser.add_argument("--retreat-height", type=float, default=0.10)
    parser.add_argument("--plan-retreat", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-ignore-roles", default="floor")
    parser.add_argument("--release-ignore-roles-as-collision-exclusions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--release-candidate-index", type=int, default=6)
    parser.add_argument("--release-max-joint-delta", type=float, default=2.2)
    parser.add_argument("--max-position-error", type=float, default=0.035)
    parser.add_argument("--max-orientation-error-deg", type=float, default=35.0)
    parser.add_argument("--min-active-connections", type=int, default=1)
    parser.add_argument("--validate-during-search", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--motion-plan-pregrasp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pregrasp-timeout", type=float, default=2.5)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"summary": result["summary"], "elapsed_sec": result["elapsed_sec"], "final": result["final"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
