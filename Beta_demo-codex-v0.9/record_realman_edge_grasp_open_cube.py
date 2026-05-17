from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import sapien
import torch
from transforms3d.quaternions import axangle2quat, mat2quat, quat2mat

import jimu_pick_cube_env  # noqa: F401
from curobo_rm75_planner import DEFAULT_CUROBO_ROOT, RM75CuRoboPlanner, RM75CuRoboPlannerConfig
from jimu_pick_cube_env import PLATE_SIZE, PLATE_THICKNESS
from magnetic_snap import LockedPanelPose, MagneticConnection
from record_stepwise_house_assembly_sim import _append_frame


ROLES = ["floor", "right_wall", "left_wall", "back_wall", "front_wall", "top_lid"]
BUILD_ROLES = ["right_wall", "left_wall", "back_wall", "front_wall", "top_lid"]
RM75_HOME_BASE_YAW_DEG = float(os.environ.get("JIMU_RM75_HOME_BASE_YAW_DEG", "90.0"))
RM75_HOME = np.asarray(
    [np.deg2rad(RM75_HOME_BASE_YAW_DEG), 0.0, 0.0, -np.pi / 2.0, 0.0, -np.pi / 2.0, np.pi / 3.0],
    dtype=np.float32,
)
OPEN_GRIPPER = -1.0
CLOSED_GRIPPER = 1.0
ACTION_DELTA_RAD = 0.05


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        raise ValueError("cannot normalize zero vector")
    return (vec / norm).astype(np.float32)


def _pose_arrays(pose: sapien.Pose) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pose.p, dtype=np.float32).reshape(3)
    q = np.asarray(pose.q, dtype=np.float32).reshape(4)
    q = q / max(float(np.linalg.norm(q)), 1e-8)
    return p, q


def _actor_pose(actor: Any) -> sapien.Pose:
    p = actor.pose.p.detach().cpu().numpy().reshape(-1, 3)[0].astype(np.float32)
    q = actor.pose.q.detach().cpu().numpy().reshape(-1, 4)[0].astype(np.float32)
    return sapien.Pose(p=p.tolist(), q=(q / max(float(np.linalg.norm(q)), 1e-8)).tolist())


def _tcp_pose(base_env: Any) -> sapien.Pose:
    pose = base_env.agent.tcp.pose
    p = pose.p.detach().cpu().numpy().reshape(-1, 3)[0].astype(np.float32)
    q = pose.q.detach().cpu().numpy().reshape(-1, 4)[0].astype(np.float32)
    return sapien.Pose(p=p.tolist(), q=(q / max(float(np.linalg.norm(q)), 1e-8)).tolist())


def _enable_held_actor_pose_lock(base_env: Any, actor: Any, actor_to_tcp: sapien.Pose) -> None:
    base_env._held_actor_pose_lock = {
        "actor": actor,
        "actor_to_tcp": actor_to_tcp,
        "step_count": 0,
    }


def _disable_held_actor_pose_lock(base_env: Any) -> dict[str, Any] | None:
    lock = getattr(base_env, "_held_actor_pose_lock", None)
    if lock is not None:
        try:
            delattr(base_env, "_held_actor_pose_lock")
        except Exception:
            base_env._held_actor_pose_lock = None
    return lock if isinstance(lock, dict) else None


def _apply_held_actor_pose_lock(base_env: Any) -> None:
    lock = getattr(base_env, "_held_actor_pose_lock", None)
    if not isinstance(lock, dict):
        return
    actor = lock.get("actor")
    actor_to_tcp = lock.get("actor_to_tcp")
    if actor is None or actor_to_tcp is None:
        return
    target_pose = _tcp_pose(base_env) * actor_to_tcp.inv()
    actor.set_pose(target_pose)
    try:
        actor.set_linear_velocity(np.zeros(3, dtype=np.float32))
        actor.set_angular_velocity(np.zeros(3, dtype=np.float32))
    except Exception:
        pass
    lock["step_count"] = int(lock.get("step_count", 0)) + 1


def _make_pose(position: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> sapien.Pose:
    x_axis = _normalize(x_axis)
    y_axis = _normalize(y_axis)
    z_axis = _normalize(z_axis)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    matrix[:3, 3] = position.astype(np.float32)
    return sapien.Pose(matrix)


def _axis_angle_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize(axis)
    x, y, z = axis.tolist()
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )


def _tilt_actor_pose_about_local_axis(
    pose: sapien.Pose,
    *,
    local_pivot: np.ndarray,
    local_axis: np.ndarray,
    angle_deg: float,
    world_offset: np.ndarray | None = None,
) -> sapien.Pose:
    position, quaternion = _pose_arrays(pose)
    rotation = quat2mat(quaternion).astype(np.float32)
    pivot_world = position + rotation @ local_pivot.astype(np.float32)
    axis_world = _normalize(rotation @ local_axis.astype(np.float32))
    delta = _axis_angle_matrix(axis_world, np.deg2rad(float(angle_deg)))
    new_position = pivot_world + delta @ (position - pivot_world)
    if world_offset is not None:
        new_position = new_position + world_offset.astype(np.float32)
    new_rotation = delta @ rotation
    return sapien.Pose(p=new_position.astype(np.float32).tolist(), q=mat2quat(new_rotation).astype(np.float32).tolist())


def _tcp_for_edge_grasp(actor: Any, *, target_pose: sapien.Pose | None = None) -> sapien.Pose:
    pose = target_pose if target_pose is not None else _actor_pose(actor)
    position, quaternion = _pose_arrays(pose)
    rotation = quat2mat(quaternion)
    thin_normal = _normalize(rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    local_top_edge = np.asarray([-PLATE_SIZE / 2.0 + 0.012, 0.0, 0.0], dtype=np.float32)
    grasp_center = position + rotation @ local_top_edge
    approach = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(thin_normal, approach))) > 0.96:
        approach = _normalize(rotation @ np.asarray([0.0, -1.0, 0.0], dtype=np.float32))
    x_axis = _normalize(np.cross(thin_normal, approach))
    z_axis = _normalize(np.cross(x_axis, thin_normal))
    return _make_pose(grasp_center, x_axis, thin_normal, z_axis)


def _offset_world(pose: sapien.Pose, offset: np.ndarray) -> sapien.Pose:
    p, q = _pose_arrays(pose)
    return sapien.Pose(p=(p + offset.astype(np.float32)).tolist(), q=q.tolist())


def _offset_along_tcp_z(pose: sapien.Pose, distance: float) -> sapien.Pose:
    p, q = _pose_arrays(pose)
    rotation = quat2mat(q)
    return sapien.Pose(p=(p + rotation[:, 2].astype(np.float32) * float(distance)).tolist(), q=q.tolist())


def _current_arm_qpos(base_env: Any) -> np.ndarray:
    return base_env.agent.robot.get_qpos().detach().cpu().numpy()[0, :7].astype(np.float32)


def _controller_config(base_env: Any, name: str) -> Any:
    return base_env.agent.controller.configs[name]


def _scale_to_normalized(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return (2.0 * (values - low) / np.maximum(high - low, 1e-8) - 1.0).astype(np.float32)


def _format_arm_action(base_env: Any, target_q: np.ndarray) -> np.ndarray:
    config = _controller_config(base_env, "arm")
    target_q = np.asarray(target_q, dtype=np.float32).reshape(7)
    current = _current_arm_qpos(base_env)
    if bool(getattr(config, "use_delta", False)):
        lower = np.broadcast_to(np.asarray(config.lower, dtype=np.float32), target_q.shape)
        upper = np.broadcast_to(np.asarray(config.upper, dtype=np.float32), target_q.shape)
        delta = np.clip(target_q - current, lower, upper)
        if bool(getattr(config, "normalize_action", False)):
            return _scale_to_normalized(delta, lower, upper)
        return delta.astype(np.float32)
    if bool(getattr(config, "normalize_action", False)):
        limits = base_env.agent.robot.get_qlimits().detach().cpu().numpy()[0, :7].astype(np.float32)
        lower = limits[:, 0]
        upper = limits[:, 1]
        if getattr(config, "lower", None) is not None:
            lower[:] = np.asarray(config.lower, dtype=np.float32)
        if getattr(config, "upper", None) is not None:
            upper[:] = np.asarray(config.upper, dtype=np.float32)
        return _scale_to_normalized(target_q, lower, upper)
    return target_q.astype(np.float32)


def _format_gripper_action(base_env: Any, gripper: float) -> float:
    config = _controller_config(base_env, "gripper")
    value = float(gripper)
    if bool(getattr(config, "normalize_action", False)):
        return float(np.clip(value, -1.0, 1.0))
    lower = float(np.asarray(config.lower, dtype=np.float32).reshape(-1)[0])
    upper = float(np.asarray(config.upper, dtype=np.float32).reshape(-1)[0])
    normalized = float(np.clip(value, -1.0, 1.0))
    return lower + (normalized + 1.0) * 0.5 * (upper - lower)


def _set_robot_qpos(base_env: Any, arm_qpos: np.ndarray, gripper_open: bool) -> None:
    full = np.zeros((1, 13), dtype=np.float32)
    full[0, :7] = arm_qpos.astype(np.float32)
    full[0, 7:] = 0.0 if gripper_open else 0.6
    base_env.agent.robot.set_qpos(torch.as_tensor(full, dtype=torch.float32, device=base_env.device))
    if hasattr(base_env.agent.robot, "set_qvel"):
        base_env.agent.robot.set_qvel(torch.zeros_like(base_env.agent.robot.get_qvel()))


def _step_action(env: Any, target_q: np.ndarray, gripper: float, writer: Any | None, record_every: int, index: int) -> None:
    base_env = env.unwrapped
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    action[:7] = _format_arm_action(base_env, target_q)
    action[-1] = _format_gripper_action(base_env, gripper)
    env.step(action)
    _apply_held_actor_pose_lock(base_env)
    if writer is not None and index % max(record_every, 1) == 0:
        _append_frame(writer, env)


def _follow_joint_path(env: Any, path: np.ndarray, gripper: float, writer: Any | None, record_every: int) -> None:
    step_index = 0
    for target in path:
        for _ in range(4):
            _step_action(env, target[:7], gripper, writer, record_every, step_index)
            step_index += 1
    final = path[-1, :7]
    for _ in range(12):
        _step_action(env, final, gripper, writer, record_every, step_index)
        step_index += 1


def _hold(env: Any, gripper: float, steps: int, writer: Any | None, record_every: int) -> None:
    q = _current_arm_qpos(env.unwrapped)
    for index in range(steps):
        _step_action(env, q, gripper, writer, record_every, index)


def _ramp_gripper(
    env: Any,
    start_gripper: float,
    end_gripper: float,
    steps: int,
    writer: Any | None,
    record_every: int,
) -> None:
    q = _current_arm_qpos(env.unwrapped)
    count = max(int(steps), 1)
    for index, gripper in enumerate(np.linspace(float(start_gripper), float(end_gripper), count, dtype=np.float32)):
        _step_action(env, q, float(gripper), writer, record_every, index)


def _interpolate_joint_path(start: np.ndarray, goal: np.ndarray, *, steps: int) -> np.ndarray:
    count = max(int(steps), 2)
    return np.stack(
        [
            (1.0 - alpha) * start.astype(np.float32) + alpha * goal.astype(np.float32)
            for alpha in np.linspace(0.0, 1.0, count, dtype=np.float32)[1:]
        ],
        axis=0,
    ).astype(np.float32)


def _stage_xy(role: str, index: int) -> tuple[float, float]:
    role_key = role.upper()
    layout = os.environ.get("JIMU_STAGE_LAYOUT", "right_side_row").strip().lower()
    if layout in {"side_row", "right_side_row", "row_x"}:
        x0 = float(os.environ.get("JIMU_STAGE_ROW_X_START", "-0.52"))
        dx = float(os.environ.get("JIMU_STAGE_ROW_X_STEP", "0.12"))
        default_y = float(os.environ.get("JIMU_STAGE_ROW_Y", "-0.29"))
        x = x0 + index * dx
        y = default_y
    elif layout in {"left_side_row"}:
        x0 = float(os.environ.get("JIMU_STAGE_ROW_X_START", "-0.52"))
        dx = float(os.environ.get("JIMU_STAGE_ROW_X_STEP", "0.12"))
        default_y = float(os.environ.get("JIMU_STAGE_ROW_Y", "0.34"))
        x = x0 + index * dx
        y = default_y
    elif layout in {"front_row", "row_y"}:
        default_x = float(os.environ.get("JIMU_STAGE_ROW_X", "-0.44"))
        y0 = float(os.environ.get("JIMU_STAGE_ROW_Y_START", "-0.30"))
        dy = float(os.environ.get("JIMU_STAGE_ROW_Y_STEP", "0.12"))
        x = default_x
        y = y0 + index * dy
    else:
        x0 = float(os.environ.get("JIMU_STAGE_START_X", "-0.35"))
        dx = float(os.environ.get("JIMU_STAGE_STEP_X", "0.09"))
        default_y = float(os.environ.get("JIMU_STAGE_Y", "-0.20"))
        x = x0 + index * dx
        y = default_y
    x = float(os.environ.get(f"JIMU_STAGE_X_{role_key}", x))
    y = float(os.environ.get(f"JIMU_STAGE_Y_{role_key}", y))
    return x, y


def _stage_pose(role: str, index: int, targets: dict[str, sapien.Pose] | None = None) -> sapien.Pose:
    role_key = role.upper()
    x, y = _stage_xy(role, index)
    yaw_deg = float(os.environ.get(f"JIMU_STAGE_YAW_DEG_{role_key}", os.environ.get("JIMU_STAGE_YAW_DEG", "90.0")))
    base = quat2mat(axangle2quat([0.0, 1.0, 0.0], np.deg2rad(90.0))).astype(np.float32)
    yaw = _axis_angle_matrix(np.asarray([0.0, 0.0, 1.0], dtype=np.float32), np.deg2rad(yaw_deg))
    q = mat2quat(base @ yaw).astype(np.float32)
    return sapien.Pose(p=[x, y, PLATE_SIZE / 2.0], q=q.tolist())


def _locked_by_role(base_env: Any) -> dict[str, Any]:
    return {locked.role: locked for locked in base_env.magnetic_snap.locked_panel_poses if locked.role in ROLES}


def _add_top_lid_role(base_env: Any) -> None:
    snap = base_env.magnetic_snap
    if any(locked.role == "top_lid" for locked in snap.locked_panel_poses):
        return
    if len(base_env.plates) < 6:
        raise ValueError("top lid requires num_plates >= 6")
    wall_top_z = PLATE_THICKNESS + PLATE_SIZE
    top_position = np.asarray([0.0, 0.0, wall_top_z + PLATE_THICKNESS / 2.0], dtype=np.float32)
    top_quaternion = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    snap.locked_panel_poses.append(
        LockedPanelPose(
            role="top_lid",
            actor=base_env.plates[5],
            position=top_position,
            quaternion=top_quaternion,
        )
    )
    snap.connections.extend(
        [
            MagneticConnection("top_lid", "right_edge", "right_wall", "top_edge", "right_angle", "face_b", "face_a"),
            MagneticConnection("top_lid", "left_edge", "left_wall", "top_edge", "right_angle", "face_b", "face_a"),
            MagneticConnection("top_lid", "top_edge", "back_wall", "top_edge", "right_angle", "face_b", "face_a"),
            MagneticConnection("top_lid", "bottom_edge", "front_wall", "top_edge", "right_angle", "face_b", "face_a"),
        ]
    )


def _create_staging_fixtures(base_env: Any, targets: dict[str, sapien.Pose]) -> list[dict[str, Any]]:
    fixture_specs: list[dict[str, Any]] = []
    material = sapien.pysapien.physx.PhysxMaterial(static_friction=1.6, dynamic_friction=1.2, restitution=0.0)
    visual = sapien.render.RenderMaterial(base_color=[0.12, 0.12, 0.12, 1.0])
    for index, role in enumerate(BUILD_ROLES):
        stage = _stage_pose(role, index, targets)
        center = np.asarray(stage.p, dtype=np.float32)
        _, stage_q = _pose_arrays(stage)
        stage_rot = quat2mat(stage_q)
        thin_axis = _normalize(stage_rot @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
        thin_axis[2] = 0.0
        thin_axis = _normalize(thin_axis)
        long_axis = stage_rot @ np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        long_axis[2] = 0.0
        if float(np.linalg.norm(long_axis)) <= 1e-6:
            long_axis = stage_rot @ np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            long_axis[2] = 0.0
        long_axis = _normalize(long_axis)
        rail_q = np.asarray(_make_pose(np.zeros(3, dtype=np.float32), thin_axis, long_axis, np.asarray([0.0, 0.0, 1.0], dtype=np.float32)).q, dtype=np.float32)
        for side, sign in [("left", -1.0), ("right", 1.0)]:
            rail_center = np.asarray(
                [
                    center[0] + sign * (PLATE_THICKNESS * 0.5 + 0.0045) * thin_axis[0],
                    center[1] + sign * (PLATE_THICKNESS * 0.5 + 0.0045) * thin_axis[1],
                    0.012,
                ],
                dtype=np.float32,
            )
            dims = np.asarray([0.005, PLATE_SIZE + 0.028, 0.024], dtype=np.float32)
            builder = base_env.scene.create_actor_builder()
            builder.set_scene_idxs([0])
            builder.initial_pose = sapien.Pose(p=rail_center.tolist(), q=rail_q.tolist())
            builder.add_box_collision(half_size=(dims * 0.5).tolist(), material=material)
            builder.add_box_visual(half_size=(dims * 0.5).tolist(), material=visual)
            actor = builder.build_kinematic(name=f"staging_slot_{role}_{side}")
            base_env.remove_from_state_dict_registry(actor)
            fixture_specs.append(
                {
                    "name": f"fixture_{role}_{side}",
                    "actor": actor,
                    "dims": dims.tolist(),
                    "pose": [
                        float(rail_center[0]),
                        float(rail_center[1]),
                        float(rail_center[2]),
                        float(rail_q[0]),
                        float(rail_q[1]),
                        float(rail_q[2]),
                        float(rail_q[3]),
                    ],
                }
            )
    return fixture_specs


def _initialize_staged_open_cube(base_env: Any) -> tuple[dict[str, sapien.Pose], dict[str, Any], list[dict[str, Any]]]:
    _add_top_lid_role(base_env)
    locked = _locked_by_role(base_env)
    targets = {
        role: sapien.Pose(p=locked[role].position.tolist(), q=locked[role].quaternion.tolist())
        for role in ROLES
    }
    assembly_offset = np.asarray([-0.20, 0.0, 0.0], dtype=np.float32)
    targets = {
        role: sapien.Pose(
            p=(np.asarray(pose.p, dtype=np.float32) + assembly_offset).tolist(),
            q=np.asarray(pose.q, dtype=np.float32).tolist(),
        )
        for role, pose in targets.items()
    }
    snap = base_env.magnetic_snap
    snap._disable_all_drives()
    for active in snap.active_connections:
        active.active = False
    snap.suspended_roles.clear()
    floor = locked["floor"].actor
    floor.set_pose(targets["floor"])
    floor.set_linear_velocity(np.zeros(3, dtype=np.float32))
    floor.set_angular_velocity(np.zeros(3, dtype=np.float32))
    for index, role in enumerate(BUILD_ROLES):
        actor = locked[role].actor
        actor.set_pose(_stage_pose(role, index, targets))
        actor.set_linear_velocity(np.zeros(3, dtype=np.float32))
        actor.set_angular_velocity(np.zeros(3, dtype=np.float32))
    fixtures = _create_staging_fixtures(base_env, targets)
    return targets, locked, fixtures


def _cuboid_for_plate(role: str, actor: Any, *, inflate: float = 0.006) -> dict[str, Any]:
    pose = _actor_pose(actor)
    p, q = _pose_arrays(pose)
    return {
        "name": f"plate_{role}",
        "dims": [PLATE_SIZE + inflate, PLATE_SIZE + inflate, PLATE_THICKNESS + inflate],
        "pose": [float(p[0]), float(p[1]), float(p[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])],
    }


def _world_to_robot_base(base_env: Any, pose: sapien.Pose) -> sapien.Pose:
    return base_env.agent.robot.pose.sp.inv() * pose


def _cuboid_in_robot_base(base_env: Any, cuboid: dict[str, Any]) -> dict[str, Any]:
    pose = sapien.Pose(p=cuboid["pose"][:3], q=cuboid["pose"][3:7])
    base_pose = _world_to_robot_base(base_env, pose)
    p, q = _pose_arrays(base_pose)
    converted = dict(cuboid)
    converted["pose"] = [float(p[0]), float(p[1]), float(p[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])]
    return converted


def _world_obstacles(
    base_env: Any,
    locked: dict[str, Any],
    fixtures: list[dict[str, Any]],
    *,
    exclude_role: str | None,
) -> list[dict[str, Any]]:
    obstacles = [
        {
            "name": "table_guard",
            "dims": [1.2, 1.2, 0.035],
            "pose": [0.0, 0.0, -0.02, 1.0, 0.0, 0.0, 0.0],
        }
    ]
    for role, locked_pose in locked.items():
        if role == exclude_role:
            continue
        obstacles.append(_cuboid_for_plate(role, locked_pose.actor))
    obstacles.extend({k: v for k, v in fixture.items() if k != "actor"} for fixture in fixtures)
    return [_cuboid_in_robot_base(base_env, obstacle) for obstacle in obstacles]


def _plan_and_execute(
    *,
    env: Any,
    planner: RM75CuRoboPlanner,
    locked: dict[str, Any],
    fixtures: list[dict[str, Any]],
    exclude_role: str | None,
    target_pose: sapien.Pose,
    gripper: float,
    writer: Any | None,
    record_every: int,
    label: str,
    planner_mode: str,
) -> dict[str, Any]:
    base_env = env.unwrapped
    start_q = _current_arm_qpos(base_env)
    obstacles = _world_obstacles(base_env, locked, fixtures, exclude_role=exclude_role)
    planner.set_world_from_obstacles(cuboids=obstacles)
    target_pose_base = _world_to_robot_base(base_env, target_pose)
    segment_name = label.split(":")[-1]
    use_ik = planner_mode == "ik" or (
        planner_mode == "hybrid" and segment_name in {"preplace", "place", "release_retreat", "retreat"}
    )
    if use_ik:
        result = planner.solve_ik(start_q, target_pose_base, num_seeds=64)
        report = {
            "label": label,
            "success": bool(result.success),
            "status": result.status,
            "solve_time": float(result.solve_time),
            "ik_time": float(result.ik_time),
            "trajopt_time": 0.0,
            "obstacle_count": len(obstacles),
            "debug": result.debug,
            "planner_mode": "ik_interpolate_checked" if planner_mode != "hybrid" else "hybrid_ik_release",
        }
        if not result.success or result.goal_joint is None:
            return report
        path = _interpolate_joint_path(start_q, result.goal_joint, steps=24)
        _follow_joint_path(env, path, gripper, writer, record_every)
        report["executed"] = True
        return report
    result = planner.plan_to_pose(start_q, target_pose_base, enable_graph=False, max_attempts=1, timeout=2.5)
    report = {
        "label": label,
        "success": bool(result.success),
        "status": result.status,
        "solve_time": float(result.solve_time),
        "ik_time": float(result.ik_time),
        "trajopt_time": float(result.trajopt_time),
        "obstacle_count": len(obstacles),
        "debug": result.debug,
        "planner_mode": "motion_gen",
    }
    if not result.success or result.joint_path is None:
        return report
    _follow_joint_path(env, result.joint_path, gripper, writer, record_every)
    report["executed"] = True
    return report


def _grasp_report(base_env: Any, actor: Any) -> dict[str, Any]:
    agent = base_env.agent
    l_force = base_env.scene.get_pairwise_contact_forces(agent.finger1_link, actor)
    r_force = base_env.scene.get_pairwise_contact_forces(agent.finger2_link, actor)
    is_grasped = agent.is_grasping(actor)
    return {
        "is_grasped": bool(is_grasped.detach().cpu().numpy().reshape(-1)[0]),
        "left_force": float(torch.linalg.norm(l_force, dim=1).detach().cpu().numpy().reshape(-1)[0]),
        "right_force": float(torch.linalg.norm(r_force, dim=1).detach().cpu().numpy().reshape(-1)[0]),
    }


def _pose_error(actor: Any, target: sapien.Pose) -> dict[str, float]:
    current = _actor_pose(actor)
    cp, cq = _pose_arrays(current)
    tp, tq = _pose_arrays(target)
    dot = min(max(abs(float(np.dot(cq, tq))), -1.0), 1.0)
    return {
        "position_error_m": float(np.linalg.norm(cp - tp)),
        "orientation_error_deg": float(np.rad2deg(2.0 * np.arccos(dot))),
    }


def _active_connection_count_for_role(base_env: Any, role: str) -> int:
    keys = set()
    for active_connection in base_env.magnetic_snap.active_connections:
        if not active_connection.active:
            continue
        connection = active_connection.connection
        if connection.parent == role or connection.child == role:
            keys.add(base_env.magnetic_snap._connection_key(connection))
    return len(keys)


def _wall_release_actor_candidates(role: str, target_pose: sapien.Pose) -> list[tuple[str, sapien.Pose]]:
    if role == "top_lid":
        candidates: list[tuple[str, sapien.Pose]] = []
        front_edge_center = np.asarray([0.0, -PLATE_SIZE / 2.0, 0.0], dtype=np.float32)
        front_edge_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        for tilt_deg in [35.0, 45.0, 55.0, 65.0, 75.0]:
            for lift_mm in [0.0, 2.0, 4.0, 8.0]:
                for inward_mm in [0.0, 1.5, -1.5, 3.0, -3.0]:
                    offset = np.asarray([0.0, inward_mm / 1000.0, lift_mm / 1000.0], dtype=np.float32)
                    candidates.append(
                        (
                            f"top_lid_front_hinge_tilt_{tilt_deg:+.0f}deg_lift_{lift_mm:.1f}mm_y_{inward_mm:+.1f}mm",
                            _tilt_actor_pose_about_local_axis(
                                target_pose,
                                local_pivot=front_edge_center,
                                local_axis=front_edge_axis,
                                angle_deg=tilt_deg,
                                world_offset=offset,
                            ),
                        )
                    )
        candidates.append(("exact_target", target_pose))
        for lift_mm in [14.0, 20.0, 28.0, 40.0, 55.0]:
            for tilt_deg in [8.0, -8.0, 0.0, 12.0, -12.0, 20.0, -20.0, 32.0, -32.0, 45.0, -45.0]:
                for dx_mm, dy_mm in [
                    (0.0, 0.0),
                    (1.5, 0.0),
                    (-1.5, 0.0),
                    (0.0, 1.5),
                    (0.0, -1.5),
                    (4.0, 0.0),
                    (-4.0, 0.0),
                    (0.0, 4.0),
                    (0.0, -4.0),
                ]:
                    pose = target_pose * sapien.Pose(
                        p=[dx_mm / 1000.0, dy_mm / 1000.0, lift_mm / 1000.0],
                        q=axangle2quat([0.0, 1.0, 0.0], np.deg2rad(tilt_deg)).tolist(),
                    )
                    candidates.append(
                        (
                            f"top_lid_lift_{lift_mm:.1f}mm_tilt_{tilt_deg:+.0f}deg_dx_{dx_mm:+.1f}mm_dy_{dy_mm:+.1f}mm",
                            pose,
                        )
                    )
        return candidates
    if role not in {"right_wall", "left_wall", "back_wall", "front_wall"}:
        return [("target", target_pose)]
    candidates: list[tuple[str, sapien.Pose]] = []
    local_bottom_center = np.asarray([0.0, -PLATE_SIZE / 2.0, 0.0], dtype=np.float32)
    local_bottom_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    p, q = _pose_arrays(target_pose)
    rotation = quat2mat(q).astype(np.float32)
    local_face_normal = _normalize(rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    inward_by_role = {
        "right_wall": np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
        "left_wall": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "back_wall": np.asarray([0.0, -1.0, 0.0], dtype=np.float32),
        "front_wall": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    }
    inward = inward_by_role[role]
    inward_sign = 1.0 if float(np.dot(local_face_normal, inward)) >= 0.0 else -1.0
    candidates.append(("exact_target", sapien.Pose(p=p.tolist(), q=q.tolist())))
    tilt_order = [0.0, 6.0 * inward_sign, -6.0 * inward_sign, 10.0 * inward_sign, -10.0 * inward_sign, 14.0 * inward_sign, -14.0 * inward_sign]
    for bottom_lift_mm in [0.5, 1.0, 2.0, 3.0]:
        for tilt_deg in tilt_order:
            for inward_bias_mm in [0.0, 1.5, -1.5]:
                world_offset = np.asarray([0.0, 0.0, bottom_lift_mm / 1000.0], dtype=np.float32)
                world_offset = world_offset + inward * (inward_bias_mm / 1000.0)
                label = (
                    f"edge_first_lift_{bottom_lift_mm:.1f}mm_"
                    f"bias_{inward_bias_mm:+.1f}mm_tilt_{tilt_deg:+.0f}deg"
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


def _screen_release_candidate(
    *,
    base_env: Any,
    planner: RM75CuRoboPlanner,
    start_q: np.ndarray,
    actor_to_tcp: sapien.Pose,
    actor_pose: sapien.Pose,
) -> tuple[bool, dict[str, Any], sapien.Pose, sapien.Pose, np.ndarray | None, np.ndarray | None]:
    target_tcp = actor_pose * actor_to_tcp
    preplace = _offset_world(target_tcp, np.asarray([0.0, 0.0, 0.055], dtype=np.float32))
    pre_result = planner.solve_ik(start_q, _world_to_robot_base(base_env, preplace), num_seeds=48)
    report: dict[str, Any] = {
        "preplace_ik": bool(pre_result.success),
        "preplace_status": pre_result.status,
        "preplace_debug": pre_result.debug,
    }
    if not pre_result.success or pre_result.goal_joint is None:
        return False, report, preplace, target_tcp, None, None
    place_result = planner.solve_ik(pre_result.goal_joint, _world_to_robot_base(base_env, target_tcp), num_seeds=48)
    report.update(
        {
            "place_ik": bool(place_result.success),
            "place_status": place_result.status,
            "place_debug": place_result.debug,
        }
    )
    if not place_result.success or place_result.goal_joint is None:
        return False, report, preplace, target_tcp, pre_result.goal_joint, None
    return True, report, preplace, target_tcp, pre_result.goal_joint, place_result.goal_joint


def _select_release_candidate_fast_batch(
    *,
    env: Any,
    planner: RM75CuRoboPlanner,
    locked: dict[str, Any],
    fixtures: list[dict[str, Any]],
    role: str,
    actor_to_tcp: sapien.Pose,
    target_pose: sapien.Pose,
    num_seeds: int,
) -> tuple[sapien.Pose | None, sapien.Pose | None, dict[str, np.ndarray] | None, dict[str, Any]]:
    base_env = env.unwrapped
    obstacles = _world_obstacles(base_env, locked, fixtures, exclude_role=role)
    planner.set_world_from_obstacles(cuboids=obstacles)
    start_q = _current_arm_qpos(base_env)
    candidates = [
        {
            "label": label,
            "actor_pose": actor_pose,
            "target_tcp": actor_pose * actor_to_tcp,
        }
        for label, actor_pose in _wall_release_actor_candidates(role, target_pose)
    ]
    if not candidates:
        return None, None, None, {"selected": None, "strategy": "fast_batch", "candidates": []}
    for item in candidates:
        item["preplace"] = _offset_world(item["target_tcp"], np.asarray([0.0, 0.0, 0.055], dtype=np.float32))
    pre_results = planner.solve_batch_start_goal_ik(
        [start_q for _ in candidates],
        [_world_to_robot_base(base_env, item["preplace"]) for item in candidates],
        num_seeds=int(num_seeds),
    )
    pre_ok: list[tuple[dict[str, Any], Any]] = []
    reports: list[dict[str, Any]] = []
    for item, result in zip(candidates, pre_results):
        report = {
            "candidate": item["label"],
            "actor_position": _pose_arrays(item["actor_pose"])[0].tolist(),
            "preplace_ik": bool(result.success),
            "preplace_status": result.status,
            "preplace_debug": result.debug,
        }
        reports.append(report)
        if result.success and result.goal_joint is not None:
            item["preplace_q"] = np.asarray(result.goal_joint, dtype=np.float32).reshape(-1)[:7]
            pre_ok.append((item, result))
    if not pre_ok:
        return None, None, None, {"selected": None, "strategy": "fast_batch", "candidates": reports}
    place_results = planner.solve_batch_start_goal_ik(
        [item["preplace_q"] for item, _ in pre_ok],
        [_world_to_robot_base(base_env, item["target_tcp"]) for item, _ in pre_ok],
        num_seeds=int(num_seeds),
    )
    report_by_label = {item["candidate"]: item for item in reports}
    ranked: list[dict[str, Any]] = []
    for (item, _), place_result in zip(pre_ok, place_results):
        report = report_by_label[item["label"]]
        report.update(
            {
                "place_ik": bool(place_result.success),
                "place_status": place_result.status,
                "place_debug": place_result.debug,
            }
        )
        if not place_result.success or place_result.goal_joint is None:
            continue
        place_q = np.asarray(place_result.goal_joint, dtype=np.float32).reshape(-1)[:7]
        score = float(np.linalg.norm(item["preplace_q"] - start_q)) + 0.35 * float(np.linalg.norm(place_q - item["preplace_q"]))
        report["fast_batch_score"] = score
        ranked.append(
            {
                "score": score,
                "item": item,
                "place_q": place_q,
            }
        )
    if not ranked:
        return None, None, None, {"selected": None, "strategy": "fast_batch", "candidates": reports}
    ranked.sort(key=lambda entry: (float(entry["score"]), str(entry["item"]["label"])))
    selected = ranked[0]
    item = selected["item"]
    return (
        item["preplace"],
        item["target_tcp"],
        {
            "preplace": np.asarray(item["preplace_q"], dtype=np.float32),
            "place": np.asarray(selected["place_q"], dtype=np.float32),
        },
        {
            "selected": item["label"],
            "selected_actor_pose": _pose_to_report(item["actor_pose"]),
            "strategy": "fast_batch",
            "candidate_count": len(candidates),
            "preplace_ik_count": len(pre_ok),
            "place_ik_count": len(ranked),
            "selected_score": float(selected["score"]),
            "candidates": reports,
        },
    )


def _pose_to_report(pose: sapien.Pose) -> dict[str, list[float]]:
    p, q = _pose_arrays(pose)
    return {"position": p.tolist(), "quaternion": q.tolist()}


def _pose_from_report(data: dict[str, Any]) -> sapien.Pose:
    return sapien.Pose(p=data["position"], q=data["quaternion"])


def _select_release_candidate(
    *,
    env: Any,
    planner: RM75CuRoboPlanner,
    locked: dict[str, Any],
    fixtures: list[dict[str, Any]],
    role: str,
    actor_to_tcp: sapien.Pose,
    target_pose: sapien.Pose,
) -> tuple[sapien.Pose | None, sapien.Pose | None, dict[str, np.ndarray] | None, dict[str, Any]]:
    base_env = env.unwrapped
    obstacles = _world_obstacles(base_env, locked, fixtures, exclude_role=role)
    planner.set_world_from_obstacles(cuboids=obstacles)
    start_q = _current_arm_qpos(base_env)
    reports: list[dict[str, Any]] = []
    for label, actor_pose in _wall_release_actor_candidates(role, target_pose):
        ok, report, preplace, target_tcp, preplace_q, place_q = _screen_release_candidate(
            base_env=base_env,
            planner=planner,
            start_q=start_q,
            actor_to_tcp=actor_to_tcp,
            actor_pose=actor_pose,
        )
        report["candidate"] = label
        report["actor_position"] = _pose_arrays(actor_pose)[0].tolist()
        reports.append(report)
        if ok:
            return (
                preplace,
                target_tcp,
                {"preplace": preplace_q.astype(np.float32), "place": place_q.astype(np.float32)},
                {"selected": label, "selected_actor_pose": _pose_to_report(actor_pose), "candidates": reports},
            )
    return None, None, None, {"selected": None, "candidates": reports}


def _execute_joint_goal(
    *,
    env: Any,
    goal_q: np.ndarray,
    gripper: float,
    writer: Any | None,
    record_every: int,
    label: str,
    steps: int,
) -> dict[str, Any]:
    start_q = _current_arm_qpos(env.unwrapped)
    path = _interpolate_joint_path(start_q, goal_q, steps=steps)
    _follow_joint_path(env, path, gripper, writer, record_every)
    return {
        "label": label,
        "success": True,
        "status": "screened_ik_goal_executed",
        "planner_mode": "screened_ik_interpolation",
        "executed": True,
    }


def _plan_short_ik_and_execute(
    *,
    env: Any,
    planner: RM75CuRoboPlanner,
    locked: dict[str, Any],
    fixtures: list[dict[str, Any]],
    exclude_role: str | None,
    target_pose: sapien.Pose,
    gripper: float,
    writer: Any | None,
    record_every: int,
    label: str,
    steps: int,
    num_seeds: int,
) -> dict[str, Any]:
    base_env = env.unwrapped
    start_q = _current_arm_qpos(base_env)
    obstacles = _world_obstacles(base_env, locked, fixtures, exclude_role=exclude_role)
    planner.set_world_from_obstacles(cuboids=obstacles)
    result = planner.solve_ik(start_q, _world_to_robot_base(base_env, target_pose), num_seeds=int(num_seeds))
    report = {
        "label": label,
        "success": bool(result.success),
        "status": result.status,
        "solve_time": float(result.solve_time),
        "ik_time": float(result.ik_time),
        "trajopt_time": 0.0,
        "obstacle_count": len(obstacles),
        "debug": result.debug,
        "planner_mode": "fast_short_endpoint_ik",
    }
    if not result.success or result.goal_joint is None:
        return report
    seg = _execute_joint_goal(
        env=env,
        goal_q=np.asarray(result.goal_joint, dtype=np.float32).reshape(-1)[:7],
        gripper=gripper,
        writer=writer,
        record_every=record_every,
        label=label,
        steps=steps,
    )
    report["executed"] = True
    report["execution"] = seg
    return report


def _servo_held_actor_to_pose(
    *,
    env: Any,
    planner: RM75CuRoboPlanner,
    locked: dict[str, Any],
    fixtures: list[dict[str, Any]],
    role: str,
    actor: Any,
    target_actor_pose: sapien.Pose,
    writer: Any | None,
    record_every: int,
    iterations: int,
    steps: int,
) -> dict[str, Any]:
    base_env = env.unwrapped
    attempts: list[dict[str, Any]] = []
    obstacles = _world_obstacles(base_env, locked, fixtures, exclude_role=role)
    planner.set_world_from_obstacles(cuboids=obstacles)
    for index in range(max(int(iterations), 0)):
        current_error = _pose_error(actor, target_actor_pose)
        grasp = _grasp_report(base_env, actor)
        attempt: dict[str, Any] = {
            "iteration": index,
            "pose_error_before": current_error,
            "grasp": grasp,
        }
        attempts.append(attempt)
        if current_error["position_error_m"] <= 0.006 and current_error["orientation_error_deg"] <= 10.0:
            attempt["skipped"] = "already_within_servo_tolerance"
            break
        if not grasp["is_grasped"]:
            attempt["failed_at"] = "grasp_lost"
            break
        live_actor_to_tcp = _actor_pose(actor).inv() * _tcp_pose(base_env)
        desired_tcp = target_actor_pose * live_actor_to_tcp
        start_q = _current_arm_qpos(base_env)
        result = planner.solve_ik(start_q, _world_to_robot_base(base_env, desired_tcp), num_seeds=64)
        attempt.update(
            {
                "ik_success": bool(result.success),
                "ik_status": result.status,
                "ik_debug": result.debug,
            }
        )
        if not result.success or result.goal_joint is None:
            attempt["failed_at"] = "ik"
            break
        seg = _execute_joint_goal(
            env=env,
            goal_q=result.goal_joint.astype(np.float32),
            gripper=CLOSED_GRIPPER,
            writer=writer,
            record_every=record_every,
            label=f"{role}:closed_loop_servo_{index}",
            steps=steps,
        )
        attempt["segment"] = seg
        _hold(env, CLOSED_GRIPPER, 8, writer, record_every)
        attempt["pose_error_after"] = _pose_error(actor, target_actor_pose)
    return {
        "attempts": attempts,
        "final_error_to_release_pose": _pose_error(actor, target_actor_pose),
    }


def _attach_payload_for_planning(planner: RM75CuRoboPlanner, base_env: Any, actor: Any) -> dict[str, Any]:
    if getattr(planner, "motion_gen", None) is None:
        return {
            "enabled": False,
            "attached_sphere_count": 0,
            "note": "skipped: ik_only_direct_control",
        }
    q = _current_arm_qpos(base_env)
    ok = planner.attach_object_box_to_robot(
        q,
        [PLATE_SIZE, PLATE_SIZE, PLATE_THICKNESS],
        object_pose_world=_actor_pose(actor),
        linear_sphere_count=6,
        linear_sphere_radius_scale=0.42,
        linear_length_scale=0.92,
    )
    return {
        "enabled": bool(ok),
        "attached_sphere_count": int(planner.get_attached_sphere_count()),
        "note": "planner_collision_payload_only; simulation object remains fully dynamic",
    }


def _make_env(record: bool, render_mode: str | None = None) -> Any:
    drive_stiffness = float(os.environ.get("JIMU_DRIVE_STIFFNESS", "60.0"))
    drive_damping = float(os.environ.get("JIMU_DRIVE_DAMPING", "10.0"))
    drive_force_limit = float(os.environ.get("JIMU_DRIVE_FORCE_LIMIT", "1.0"))
    drive_angular_stiffness = float(os.environ.get("JIMU_DRIVE_ANGULAR_STIFFNESS", "0.22"))
    drive_angular_damping = float(os.environ.get("JIMU_DRIVE_ANGULAR_DAMPING", "0.025"))
    drive_angular_force_limit = float(os.environ.get("JIMU_DRIVE_ANGULAR_FORCE_LIMIT", "0.16"))
    magnet_dynamic_update_period = int(os.environ.get("JIMU_MAGNET_DYNAMIC_UPDATE_PERIOD", "5"))
    selected_render_mode = render_mode
    if selected_render_mode in {"", "none", "off", "null"}:
        selected_render_mode = None
    if selected_render_mode is None and record:
        selected_render_mode = "rgb_array"
    kwargs = {
        "obs_mode": "state",
        "render_mode": selected_render_mode,
        "control_mode": "pd_joint_pos_abs",
        "robot_uids": "RM75",
        "assembly_mode": "open_cube",
        "magnet_mode": "edge_pair_drive",
        "drive_stiffness": drive_stiffness,
        "drive_damping": drive_damping,
        "drive_force_limit": drive_force_limit,
        "drive_angular_stiffness": drive_angular_stiffness,
        "drive_angular_damping": drive_angular_damping,
        "drive_angular_force_limit": drive_angular_force_limit,
        "magnet_dynamic_update_period": magnet_dynamic_update_period,
        "num_plates": 6,
        "num_triangles": 0,
        "num_envs": 1,
        "max_episode_steps": 100000,
    }
    if selected_render_mode == "rgb_array":
        kwargs["render_backend"] = "cpu"
    return gym.make("JimuPickCube-v1", **kwargs)


def run_realman_open_cube(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "realman_edge_grasp_open_cube.mp4"
    summary_path = out_dir / "realman_edge_grasp_open_cube_summary.json"
    env = _make_env(args.record)
    writer = None
    stage_reports: list[dict[str, Any]] = []
    final: dict[str, Any] = {}
    try:
        env.reset()
        base_env = env.unwrapped
        _set_robot_qpos(base_env, RM75_HOME, gripper_open=True)
        targets, locked, fixtures = _initialize_staged_open_cube(base_env)
        planner = RM75CuRoboPlanner(
            RM75CuRoboPlannerConfig(
                curobo_root=Path(args.curobo_root),
                robot_cfg_path=Path(args.robot_cfg),
                position_threshold=0.008,
                rotation_threshold=0.12,
                num_ik_seeds=64,
                num_trajopt_seeds=2,
                num_graph_seeds=1,
                collision_activation_distance=0.018,
                build_motion_gen=not bool(args.fast_single_wall),
            )
        )
        final["curobo_collision_enabled"] = bool(planner.collision_enabled)
        final["configured_collision_links"] = planner.configured_collision_links
        if args.record:
            writer = imageio.get_writer(video_path, fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
            for _ in range(args.fps):
                _append_frame(writer, env)
        _hold(env, OPEN_GRIPPER, args.initial_steps, writer, args.record_every)
        placed_roles: list[str] = ["floor"]
        for role in BUILD_ROLES[: max(0, int(args.max_build_roles))]:
            use_fast_single_wall = bool(getattr(args, "fast_single_wall", False)) and role == "right_wall"
            actor = locked[role].actor
            grasp_tcp = _tcp_for_edge_grasp(actor)
            pregrasp = _offset_along_tcp_z(grasp_tcp, 0.035)
            actor_pose_at_grasp = _actor_pose(actor)
            actor_to_tcp = actor_pose_at_grasp.inv() * grasp_tcp
            target_actor_pose = targets[role]
            if role == "top_lid":
                target_actor_pose = sapien.Pose(
                    p=[targets[role].p[0], targets[role].p[1], targets[role].p[2] + 0.014],
                    q=(targets[role] * sapien.Pose(q=axangle2quat([0.0, 1.0, 0.0], np.deg2rad(8.0)).tolist())).q,
                )
            lift = _offset_world(grasp_tcp, np.asarray([0.0, 0.0, 0.16], dtype=np.float32))
            role_report: dict[str, Any] = {"role": role, "segments": [], "fast_single_wall": use_fast_single_wall}
            for label, pose, gripper in [
                ("pregrasp", pregrasp, OPEN_GRIPPER),
                ("edge_grasp", grasp_tcp, OPEN_GRIPPER),
            ]:
                if use_fast_single_wall:
                    seg = _plan_short_ik_and_execute(
                        env=env,
                        planner=planner,
                        locked=locked,
                        fixtures=fixtures,
                        exclude_role=role,
                        target_pose=pose,
                        gripper=gripper,
                        writer=writer,
                        record_every=args.record_every,
                        label=f"{role}:{label}",
                        steps=args.fast_short_ik_steps,
                        num_seeds=args.fast_batch_ik_seeds,
                    )
                else:
                    seg = _plan_and_execute(
                        env=env,
                        planner=planner,
                        locked=locked,
                        fixtures=fixtures,
                        exclude_role=role,
                        target_pose=pose,
                        gripper=gripper,
                        writer=writer,
                        record_every=args.record_every,
                        label=f"{role}:{label}",
                        planner_mode=args.planner_mode,
                    )
                role_report["segments"].append(seg)
                if not seg.get("success"):
                    role_report["failed_at"] = label
                    stage_reports.append(role_report)
                    raise RuntimeError(f"{role} failed at {label}: {seg.get('status')}")
            _hold(env, CLOSED_GRIPPER, args.close_steps, writer, args.record_every)
            role_report["grasp_after_close"] = _grasp_report(base_env, actor)
            if not role_report["grasp_after_close"]["is_grasped"]:
                role_report["failed_at"] = "physical_grasp_check"
                stage_reports.append(role_report)
                raise RuntimeError(f"{role} was not physically grasped at the thin edge")
            if use_fast_single_wall:
                lift_segment = _plan_short_ik_and_execute(
                    env=env,
                    planner=planner,
                    locked=locked,
                    fixtures=fixtures,
                    exclude_role=role,
                    target_pose=lift,
                    gripper=CLOSED_GRIPPER,
                    writer=writer,
                    record_every=args.record_every,
                    label=f"{role}:lift",
                    steps=args.fast_short_ik_steps,
                    num_seeds=args.fast_batch_ik_seeds,
                )
            else:
                lift_segment = _plan_and_execute(
                    env=env,
                    planner=planner,
                    locked=locked,
                    fixtures=fixtures,
                    exclude_role=role,
                    target_pose=lift,
                    gripper=CLOSED_GRIPPER,
                    writer=writer,
                    record_every=args.record_every,
                    label=f"{role}:lift",
                    planner_mode=args.planner_mode,
                )
            role_report["segments"].append(lift_segment)
            if not lift_segment.get("success"):
                role_report["failed_at"] = "lift"
                stage_reports.append(role_report)
                raise RuntimeError(f"{role} failed at lift: {lift_segment.get('status')}")
            role_report["grasp_after_lift"] = _grasp_report(base_env, actor)
            if not role_report["grasp_after_lift"]["is_grasped"]:
                role_report["failed_at"] = "physical_grasp_lost_during_lift"
                stage_reports.append(role_report)
                raise RuntimeError(f"{role} lost the physical edge grasp during lift")
            actor_to_tcp = _actor_pose(actor).inv() * _tcp_pose(base_env)
            actor_to_tcp_p, actor_to_tcp_q = _pose_arrays(actor_to_tcp)
            role_report["actor_to_tcp_after_lift"] = {
                "position": actor_to_tcp_p.tolist(),
                "quaternion": actor_to_tcp_q.tolist(),
                "note": "recalibrated from live actor pose after physical lift",
            }
            role_report["payload_collision_model"] = _attach_payload_for_planning(planner, base_env, actor)
            release_actor_pose = target_actor_pose
            if role == "top_lid":
                target_tcp = target_actor_pose * actor_to_tcp
                preplace = _offset_world(target_tcp, np.asarray([0.0, 0.0, 0.11], dtype=np.float32))
                release_q_goals = None
            else:
                if use_fast_single_wall:
                    preplace, target_tcp, release_q_goals, release_selection = _select_release_candidate_fast_batch(
                        env=env,
                        planner=planner,
                        locked=locked,
                        fixtures=fixtures,
                        role=role,
                        actor_to_tcp=actor_to_tcp,
                        target_pose=target_actor_pose,
                        num_seeds=args.fast_batch_ik_seeds,
                    )
                else:
                    preplace, target_tcp, release_q_goals, release_selection = _select_release_candidate(
                        env=env,
                        planner=planner,
                        locked=locked,
                        fixtures=fixtures,
                        role=role,
                        actor_to_tcp=actor_to_tcp,
                        target_pose=target_actor_pose,
                    )
                role_report["release_candidate_selection"] = release_selection
                if preplace is None or target_tcp is None:
                    role_report["failed_at"] = "release_candidate_ik_screen"
                    stage_reports.append(role_report)
                    raise RuntimeError(f"{role} has no IK-feasible bottom-contact release candidate")
                if "selected_actor_pose" in release_selection:
                    release_actor_pose = _pose_from_report(release_selection["selected_actor_pose"])
            if release_q_goals is not None:
                for label in ["preplace", "place"]:
                    seg = _execute_joint_goal(
                        env=env,
                        goal_q=release_q_goals[label],
                        gripper=CLOSED_GRIPPER,
                        writer=writer,
                        record_every=args.record_every,
                        label=f"{role}:{label}",
                        steps=args.ik_exec_steps,
                    )
                    role_report["segments"].append(seg)
            if release_q_goals is not None:
                role_report["closed_loop_release_servo"] = _servo_held_actor_to_pose(
                    env=env,
                    planner=planner,
                    locked=locked,
                    fixtures=fixtures,
                    role=role,
                    actor=actor,
                    target_actor_pose=release_actor_pose,
                    writer=writer,
                    record_every=args.record_every,
                    iterations=args.release_servo_iters,
                    steps=args.release_servo_steps,
                )
                _hold(env, CLOSED_GRIPPER, args.pre_release_settle_steps, writer, args.record_every)
                role_report["pose_error_before_open_to_release_pose"] = _pose_error(actor, release_actor_pose)
                role_report["pose_error_before_open"] = _pose_error(actor, targets[role])
                role_report["snap_report_before_open"] = base_env.get_magnetic_snap_report()
            else:
                for label, pose in [
                    ("preplace", preplace),
                    ("place", target_tcp),
                ]:
                    seg = _plan_and_execute(
                        env=env,
                        planner=planner,
                        locked=locked,
                        fixtures=fixtures,
                        exclude_role=role,
                        target_pose=pose,
                        gripper=CLOSED_GRIPPER,
                        writer=writer,
                        record_every=args.record_every,
                        label=f"{role}:{label}",
                        planner_mode=args.planner_mode,
                    )
                    role_report["segments"].append(seg)
                    if not seg.get("success"):
                        role_report["failed_at"] = label
                        stage_reports.append(role_report)
                        raise RuntimeError(f"{role} failed at {label}: {seg.get('status')}")
                role_report["closed_loop_release_servo"] = _servo_held_actor_to_pose(
                    env=env,
                    planner=planner,
                    locked=locked,
                    fixtures=fixtures,
                    role=role,
                    actor=actor,
                    target_actor_pose=release_actor_pose,
                    writer=writer,
                    record_every=args.record_every,
                    iterations=args.release_servo_iters,
                    steps=args.release_servo_steps,
                )
                _hold(env, CLOSED_GRIPPER, args.pre_release_settle_steps, writer, args.record_every)
                role_report["pose_error_before_open_to_release_pose"] = _pose_error(actor, release_actor_pose)
                role_report["pose_error_before_open"] = _pose_error(actor, targets[role])
                role_report["snap_report_before_open"] = base_env.get_magnetic_snap_report()
            _ramp_gripper(env, CLOSED_GRIPPER, OPEN_GRIPPER, args.open_steps, writer, args.record_every)
            if getattr(planner, "motion_gen", None) is not None:
                planner.detach_object_from_robot()
            _plan_and_execute(
                env=env,
                planner=planner,
                locked=locked,
                fixtures=fixtures,
                exclude_role=None,
                target_pose=_offset_world(target_tcp, np.asarray([0.0, 0.0, 0.12], dtype=np.float32)),
                gripper=OPEN_GRIPPER,
                writer=writer,
                record_every=args.record_every,
                label=f"{role}:retreat",
                planner_mode=args.planner_mode,
            )
            for _ in range(args.settle_steps):
                env.step(np.zeros(env.action_space.shape, dtype=env.action_space.dtype))
                if writer is not None:
                    _append_frame(writer, env)
            placed_roles.append(role)
            role_report["placed_roles"] = placed_roles.copy()
            role_report["pose_error_after_release"] = _pose_error(actor, targets[role])
            role_report["snap_report"] = base_env.get_magnetic_snap_report()
            role_report["active_connection_count_for_role"] = _active_connection_count_for_role(base_env, role)
            role_report["accepted_physical_place"] = (
                role_report["pose_error_after_release"]["position_error_m"] <= 0.035
                and role_report["pose_error_after_release"]["orientation_error_deg"] <= 35.0
                and role not in base_env.magnetic_snap.suspended_roles
                and role_report["active_connection_count_for_role"] >= 1
            )
            stage_reports.append(role_report)
        snap = base_env.magnetic_snap
        placed_without_floor = [role for role in placed_roles if role != "floor"]
        placed_reports = [item for item in stage_reports if item.get("role") in placed_without_floor]
        all_physical_places_ok = all(bool(item.get("accepted_physical_place", False)) for item in placed_reports)
        final.update(
            {
                "success": bool(all_physical_places_ok),
                "placed_roles": placed_roles,
                "active_connection_count": sum(1 for item in snap.active_connections if item.active),
                "suspended_roles": sorted(snap.suspended_roles),
            }
        )
    except Exception as exc:
        final.update({"success": False, "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        if writer is not None:
            for _ in range(args.fps):
                _append_frame(writer, env)
            writer.close()
        env.close()
    report = {
        "name": "realman_edge_grasp_open_cube",
        "video": str(video_path) if args.record else None,
        "summary": str(summary_path),
        "robot": "RM75",
        "grasp_constraint": "edge_only: tcp y axis is aligned with plate thin normal and grasp point is offset near a 74mm plate edge",
        "no_kinematic_attach": True,
        "no_placement_correction": True,
        "stage_reports": stage_reports,
        "final": final,
    }
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "videos_realman_edge_open_cube_v1"))
    parser.add_argument("--robot-cfg", default=str(Path(__file__).resolve().parent / "curobo_rm75_config" / "rm75.yml"))
    parser.add_argument("--curobo-root", default=str(DEFAULT_CUROBO_ROOT))
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--record-every", type=int, default=3)
    parser.add_argument("--initial-steps", type=int, default=20)
    parser.add_argument("--close-steps", type=int, default=24)
    parser.add_argument("--open-steps", type=int, default=24)
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--max-build-roles", type=int, default=len(BUILD_ROLES))
    parser.add_argument("--planner-mode", choices=["hybrid", "ik", "motion"], default="hybrid")
    parser.add_argument("--ik-exec-steps", type=int, default=96)
    parser.add_argument("--pre-release-settle-steps", type=int, default=40)
    parser.add_argument("--release-servo-iters", type=int, default=3)
    parser.add_argument("--release-servo-steps", type=int, default=48)
    parser.add_argument("--fast-single-wall", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fast-batch-ik-seeds", type=int, default=32)
    parser.add_argument("--fast-short-ik-steps", type=int, default=48)
    args = parser.parse_args()
    report = run_realman_open_cube(args)
    print(json.dumps({"summary": report["summary"], "video": report["video"], "final": report["final"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
