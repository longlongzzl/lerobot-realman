from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import sapien
from transforms3d.quaternions import mat2quat, quat2mat

import jimu_pick_cube_env  # noqa: F401
import plan_multi_wall_path_200step as multi_wall_module
import plan_single_wall_path_20s as single_wall_module
from curobo_rm75_planner import DEFAULT_CUROBO_ROOT, DEFAULT_RETRACT_CONFIG, RM75CuRoboPlanner, RM75CuRoboPlannerConfig
from jimu_pick_cube_env import PLATE_SIZE, PLATE_THICKNESS, TRIANGLE_HEIGHT, TRIANGLE_THICKNESS, TRIANGLE_WIDTH
from magnetic_snap import LockedPanelPose, MagneticConnection, _triangle_roof_quat
from plan_multi_wall_path_200step import (
    _anchor_floor_for_build,
    _add_rear_collision_wall_fixture,
    _load_assembly_state,
    _make_logger,
    _record_existing_joint_path,
    _save_assembly_state,
    _validate_roles_after_settle,
)
from plan_single_wall_path_20s import (
    _add_hold_segment,
    _add_joint_segment,
    _add_ramp_segment,
    _current_q,
    _joint_distance,
    _plan_motion_to_pose,
    _solve_ik,
)
from record_realman_edge_grasp_open_cube import (
    CLOSED_GRIPPER,
    OPEN_GRIPPER,
    RM75_HOME,
    _active_connection_count_for_role,
    _actor_pose,
    _grasp_report,
    _initialize_staged_open_cube,
    _normalize,
    _offset_world,
    _pose_arrays,
    _pose_error,
    _pose_to_report,
    _set_robot_qpos,
    _step_action,
    _tcp_pose,
    _world_to_robot_base,
)
from record_stepwise_house_assembly_sim import _append_frame


TRIANGLE_BUILD_SPECS = [
    ("right_second_triangle", "right_wall", "top_edge", [0.0, 1.0, 0.0]),
    ("back_second_triangle", "back_wall", "top_edge", [-1.0, 0.0, 0.0]),
    ("left_second_triangle", "left_wall", "top_edge", [0.0, -1.0, 0.0]),
    ("front_second_triangle", "front_wall", "top_edge", [1.0, 0.0, 0.0]),
]


@dataclass(frozen=True)
class TriangleGraspCandidate:
    label: str
    local_x: float
    local_z: float
    thin_bias: float
    approach_bias: float
    yaw_deg: float
    pregrasp_distance: float
    approach_tilt_deg: float = 0.0


def _make_env(record: bool) -> Any:
    kwargs = {
        "obs_mode": "state",
        "render_mode": "rgb_array" if record else None,
        "control_mode": "pd_joint_pos_abs",
        "robot_uids": "RM75",
        "assembly_mode": "open_cube",
        "magnet_mode": "edge_pair_drive",
        "drive_stiffness": float(os.environ.get("JIMU_DRIVE_STIFFNESS", "60.0")),
        "drive_damping": float(os.environ.get("JIMU_DRIVE_DAMPING", "10.0")),
        "drive_force_limit": float(os.environ.get("JIMU_DRIVE_FORCE_LIMIT", "1.0")),
        "drive_angular_stiffness": float(os.environ.get("JIMU_DRIVE_ANGULAR_STIFFNESS", "0.22")),
        "drive_angular_damping": float(os.environ.get("JIMU_DRIVE_ANGULAR_DAMPING", "0.025")),
        "drive_angular_force_limit": float(os.environ.get("JIMU_DRIVE_ANGULAR_FORCE_LIMIT", "0.16")),
        "attract_stiffness": float(os.environ.get("JIMU_ATTRACT_STIFFNESS", "1.6")),
        "attract_force_limit": float(os.environ.get("JIMU_ATTRACT_FORCE_LIMIT", "0.65")),
        "attract_torque_stiffness": float(os.environ.get("JIMU_ATTRACT_TORQUE_STIFFNESS", "0.055")),
        "attract_torque_limit": float(os.environ.get("JIMU_ATTRACT_TORQUE_LIMIT", "0.035")),
        "attract_normal_torque_stiffness": float(os.environ.get("JIMU_ATTRACT_NORMAL_TORQUE_STIFFNESS", "0.16")),
        "attract_normal_torque_limit": float(os.environ.get("JIMU_ATTRACT_NORMAL_TORQUE_LIMIT", "0.12")),
        "panel_linear_damping": float(os.environ.get("JIMU_PANEL_LINEAR_DAMPING", "0.15")),
        "panel_angular_damping": float(os.environ.get("JIMU_PANEL_ANGULAR_DAMPING", "0.45")),
        "panel_density": float(os.environ.get("JIMU_PANEL_DENSITY", "1000.0")),
        "num_plates": 6,
        "num_triangles": 4,
        "num_envs": 1,
        "max_episode_steps": 100000,
    }
    if record:
        kwargs["render_backend"] = "cpu"
    return gym.make("JimuPickCube-v1", **kwargs)


def _remove_free_triangle_roles(base_env: Any) -> None:
    snap = base_env.magnetic_snap
    snap.locked_panel_poses = [
        locked for locked in snap.locked_panel_poses if not locked.role.startswith("free_triangle_")
    ]
    snap.connections = [
        connection
        for connection in snap.connections
        if not connection.parent.startswith("free_triangle_") and not connection.child.startswith("free_triangle_")
    ]
    snap.active_connections = [
        active
        for active in snap.active_connections
        if not active.connection.parent.startswith("free_triangle_")
        and not active.connection.child.startswith("free_triangle_")
    ]


def _edge_center_from_actor(base_env: Any, role: str, edge_name: str) -> np.ndarray:
    start, end = _edge_points_from_actor(base_env, role, edge_name)
    return ((start + end) * 0.5).astype(np.float32)


def _edge_points_from_actor(base_env: Any, role: str, edge_name: str) -> tuple[np.ndarray, np.ndarray]:
    snap = base_env.magnetic_snap
    locked_by_role = {locked.role: locked for locked in snap.locked_panel_poses}
    locked = locked_by_role[role]
    edge = snap._edge_spec(role, edge_name)
    actor_pose = _actor_pose(locked.actor)
    start = np.asarray((actor_pose * sapien.Pose(p=edge.start.tolist())).p, dtype=np.float32)
    end = np.asarray((actor_pose * sapien.Pose(p=edge.end.tolist())).p, dtype=np.float32)
    return start, end


def _triangle_target_pose(base_center: np.ndarray, base_dir: list[float], lift: float = 0.0) -> sapien.Pose:
    position = np.asarray(base_center, dtype=np.float32).copy()
    position[2] += float(lift)
    quaternion = _triangle_roof_quat(base_dir, [0.0, 0.0, TRIANGLE_HEIGHT])
    return sapien.Pose(p=position.tolist(), q=quaternion.tolist())


def _triangle_target_pose_from_parent_edge(
    base_env: Any,
    *,
    parent_role: str,
    parent_edge: str,
    base_dir_hint: list[float],
    lift: float = 0.0,
) -> sapien.Pose:
    start, end = _edge_points_from_actor(base_env, parent_role, parent_edge)
    center = ((start + end) * 0.5).astype(np.float32)
    x_axis = _normalize(end - start)
    hint = _normalize(np.asarray(base_dir_hint, dtype=np.float32))
    if float(np.dot(x_axis, hint)) < 0.0:
        x_axis = -x_axis
    snap = base_env.magnetic_snap
    locked_by_role = {locked.role: locked for locked in snap.locked_panel_poses}
    parent = locked_by_role[parent_role]
    parent_center_z = float(_actor_pose(parent.actor).p[2])
    parent_edge_length = float(np.linalg.norm(end - start))
    center[2] = parent_center_z + parent_edge_length * 0.5
    parent_normal = _normalize(snap._world_face_normal(parent.actor, parent_role))
    y_axis = -parent_normal
    y_axis = y_axis - x_axis * float(np.dot(y_axis, x_axis))
    if float(np.linalg.norm(y_axis)) <= 1e-6:
        y_axis = np.cross(np.asarray([0.0, 0.0, 1.0], dtype=np.float32), x_axis)
    y_axis = _normalize(y_axis)
    z_axis = _normalize(np.cross(x_axis, y_axis))
    if float(np.dot(z_axis, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))) < 0.0:
        x_axis = -x_axis
        z_axis = _normalize(np.cross(x_axis, y_axis))
    position = center + z_axis * float(lift)
    rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
    return sapien.Pose(p=position.astype(np.float32).tolist(), q=mat2quat(rotation).astype(np.float32).tolist())


def _triangle_parent_spec(role: str) -> tuple[str, str, list[float]] | None:
    for spec_role, parent_role, parent_edge, base_dir in TRIANGLE_BUILD_SPECS:
        if spec_role == role:
            return parent_role, parent_edge, base_dir
    return None


def _triangle_parent_edge_report(base_env: Any, role: str) -> dict[str, Any]:
    spec = _triangle_parent_spec(role)
    if spec is None:
        return {"role": role, "success": False}
    parent_role, parent_edge, _ = spec
    start, end = _edge_points_from_actor(base_env, parent_role, parent_edge)
    center = ((start + end) * 0.5).astype(np.float32)
    return {
        "role": role,
        "parent_role": parent_role,
        "parent_edge": parent_edge,
        "start": start.tolist(),
        "end": end.tolist(),
        "center": center.tolist(),
        "center_z_m": float(center[2]),
        "min_z_m": float(min(start[2], end[2])),
        "max_z_m": float(max(start[2], end[2])),
        "height_span_m": float(abs(start[2] - end[2])),
        "edge_length_m": float(np.linalg.norm(end - start)),
    }


def _triangle_stage_pose(index: int, *, x_start: float, x_step: float, y: float) -> sapien.Pose:
    position = np.asarray([float(x_start) + int(index) * float(x_step), float(y), 0.0], dtype=np.float32)
    rotation = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return sapien.Pose(p=position.tolist(), q=mat2quat(rotation).astype(np.float32).tolist())


def _create_triangle_staging_fixtures(
    base_env: Any,
    stage_targets: dict[str, sapien.Pose],
    *,
    enabled: bool,
    log: Any,
) -> list[dict[str, Any]]:
    if not enabled:
        return []
    material = sapien.pysapien.physx.PhysxMaterial(static_friction=1.6, dynamic_friction=1.2, restitution=0.0)
    visual = sapien.render.RenderMaterial(base_color=[0.10, 0.13, 0.16, 1.0])
    fixtures: list[dict[str, Any]] = []
    dims = np.asarray([0.005, TRIANGLE_WIDTH + 0.028, 0.026], dtype=np.float32)
    for role, stage in stage_targets.items():
        center = np.asarray(stage.p, dtype=np.float32).reshape(3)
        for side, sign in [("left", -1.0), ("right", 1.0)]:
            rail_center = np.asarray(
                [
                    center[0] + sign * (TRIANGLE_THICKNESS * 0.5 + 0.0045),
                    center[1],
                    dims[2] * 0.5,
                ],
                dtype=np.float32,
            )
            builder = base_env.scene.create_actor_builder()
            builder.set_scene_idxs([0])
            builder.initial_pose = sapien.Pose(p=rail_center.tolist(), q=[1.0, 0.0, 0.0, 0.0])
            builder.add_box_collision(half_size=(dims * 0.5).tolist(), material=material)
            builder.add_box_visual(half_size=(dims * 0.5).tolist(), material=visual)
            actor = builder.build_kinematic(name=f"triangle_staging_slot_{role}_{side}")
            base_env.remove_from_state_dict_registry(actor)
            fixture = {
                "name": f"triangle_fixture_{role}_{side}",
                "actor": actor,
                "dims": dims.tolist(),
                "pose": [
                    float(rail_center[0]),
                    float(rail_center[1]),
                    float(rail_center[2]),
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            }
            fixtures.append(fixture)
    log(f"added triangle staging fixtures: count={len(fixtures)}")
    return fixtures


def _remove_second_layer_unused_square_artifacts(
    base_env: Any,
    locked: dict[str, Any],
    targets: dict[str, sapien.Pose],
    fixtures: list[dict[str, Any]],
    log: Any,
) -> list[dict[str, Any]]:
    snap = base_env.magnetic_snap
    removed_roles: list[str] = []
    top_lid = locked.pop("top_lid", None)
    targets.pop("top_lid", None)
    if top_lid is not None:
        top_lid.actor.set_pose(sapien.Pose(p=[1.8, 1.8, -2.0], q=[1.0, 0.0, 0.0, 0.0]))
        top_lid.actor.set_linear_velocity(np.zeros(3, dtype=np.float32))
        top_lid.actor.set_angular_velocity(np.zeros(3, dtype=np.float32))
        removed_roles.append("top_lid")
    snap.locked_panel_poses = [item for item in snap.locked_panel_poses if item.role != "top_lid"]
    snap.connections = [
        item
        for item in snap.connections
        if item.parent != "top_lid" and item.child != "top_lid"
    ]
    retained_active_connections = []
    for active_connection in snap.active_connections:
        connection = active_connection.connection
        if connection.parent == "top_lid" or connection.child == "top_lid":
            active_connection.active = False
            for drive in active_connection.drives:
                snap._disable_drive(drive)
            continue
        retained_active_connections.append(active_connection)
    snap.active_connections = retained_active_connections
    moved_fixtures = 0
    for index, fixture in enumerate(fixtures):
        actor = fixture.get("actor")
        if actor is None:
            continue
        actor.set_pose(sapien.Pose(p=[1.8 + 0.03 * index, 1.9, -2.0], q=[1.0, 0.0, 0.0, 0.0]))
        moved_fixtures += 1
    if removed_roles or moved_fixtures:
        log(f"removed second-layer unused square artifacts: roles={removed_roles} moved_fixtures={moved_fixtures}")
    return []


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
        steps = max(steps, int(np.ceil(_joint_distance(goal_q, start_q) / float(max_joint_step))))
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
) -> dict[str, Any]:
    start_q = _current_q(env.unwrapped)
    goal = np.asarray(goal_q, dtype=np.float32).reshape(7)
    steps = _adaptive_step_count(
        start_q=start_q,
        goal_q=goal,
        base_steps=int(base_steps),
        max_joint_step=float(getattr(args, "max_joint_step", 0.06)),
        max_steps=int(getattr(args, "max_segment_steps", 420)),
    )
    segment = _add_joint_segment(
        env=env,
        arrays=arrays,
        segments=segments,
        name=name,
        goal_q=goal,
        gripper=gripper,
        steps=steps,
        action_repeat=action_repeat,
        final_hold=final_hold,
    )
    segment["base_steps"] = int(base_steps)
    segment["adaptive_steps"] = int(steps)
    segment["joint_distance"] = _joint_distance(goal, start_q)
    return segment


def _apply_float_override(target: Any, attr: str, value: float | None, overrides: dict[str, float]) -> None:
    if value is None:
        return
    setattr(target, attr, float(value))
    overrides[attr] = float(getattr(target, attr))


def _disable_overstretched_active_connections(base_env: Any, *, threshold: float, log: Any) -> list[dict[str, Any]]:
    snap = base_env.magnetic_snap
    disabled: list[dict[str, Any]] = []
    for active_connection in snap.active_connections:
        if not active_connection.active:
            continue
        connection = active_connection.connection
        try:
            point_error = float(snap._connection_point_error(connection))
        except Exception as exc:
            disabled.append(
                {
                    "parent": connection.parent,
                    "parent_edge": connection.parent_edge,
                    "child": connection.child,
                    "child_edge": connection.child_edge,
                    "mode": connection.mode,
                    "error": f"{type(exc).__name__}: {exc}",
                    "disabled": False,
                }
            )
            continue
        if point_error <= float(threshold):
            continue
        active_connection.active = False
        for drive in active_connection.drives:
            snap._disable_drive(drive)
        item = {
            "parent": connection.parent,
            "parent_edge": connection.parent_edge,
            "child": connection.child,
            "child_edge": connection.child_edge,
            "mode": connection.mode,
            "point_error_m": point_error,
            "threshold_m": float(threshold),
            "disabled": True,
        }
        disabled.append(item)
        log(
            "disabled overstretched active connection "
            f"{connection.parent}:{connection.parent_edge} -> {connection.child}:{connection.child_edge} "
            f"error={point_error:.4f} threshold={float(threshold):.4f}"
        )
    return disabled


def _restore_loaded_magnetic_connections(
    *,
    state_path: str,
    base_env: Any,
    locked: dict[str, LockedPanelPose],
    log: Any,
) -> list[dict[str, Any]]:
    if not state_path:
        return []
    path = Path(state_path)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
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


def _configure_loaded_base_connection_targets(base_env: Any, loaded_roles: list[str], locked: dict[str, LockedPanelPose], log: Any) -> dict[str, int]:
    base_roles = {"floor", "right_wall", "back_wall", "left_wall", "front_wall"}
    desired = {
        "floor": 4,
        "right_wall": 3,
        "back_wall": 3,
        "left_wall": 3,
        "front_wall": 3,
    }
    active_roles = [role for role in ["floor", *loaded_roles] if role in base_roles and role in locked]
    configured = {role: desired[role] for role in active_roles if role in desired}
    if configured:
        base_env.magnetic_snap.desired_active_connections_by_role.update(configured)
        log(f"loaded base full-connection targets: {configured}")
    return configured


def _loaded_base_connection_report(base_env: Any, loaded_roles: list[str]) -> dict[str, Any]:
    base_roles = {"floor", "right_wall", "back_wall", "left_wall", "front_wall"}
    active_roles = set(["floor", *loaded_roles]) & base_roles
    if not active_roles:
        return {
            "roles": [],
            "expected_connection_count": 0,
            "active_connection_count": 0,
            "max_point_error_m": 0.0,
            "mean_point_error_m": 0.0,
            "connections": [],
        }
    snap = base_env.magnetic_snap
    active_keys = {
        snap._connection_key(active_connection.connection)
        for active_connection in snap.active_connections
        if active_connection.active
    }
    seen_keys: set[Any] = set()
    connections: list[dict[str, Any]] = []
    point_errors: list[float] = []
    for connection in snap.connections:
        if connection.parent not in active_roles or connection.child not in active_roles:
            continue
        key = snap._connection_key(connection)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        try:
            point_error = float(snap._connection_point_error(connection))
            point_errors.append(point_error)
            item: dict[str, Any] = {
                "parent": connection.parent,
                "parent_edge": connection.parent_edge,
                "child": connection.child,
                "child_edge": connection.child_edge,
                "mode": connection.mode,
                "parent_lane": connection.parent_lane,
                "child_lane": connection.child_lane,
                "point_error_m": point_error,
                "active": bool(key in active_keys),
            }
        except Exception as exc:
            item = {
                "parent": connection.parent,
                "parent_edge": connection.parent_edge,
                "child": connection.child,
                "child_edge": connection.child_edge,
                "mode": connection.mode,
                "parent_lane": connection.parent_lane,
                "child_lane": connection.child_lane,
                "error": f"{type(exc).__name__}: {exc}",
                "active": bool(key in active_keys),
            }
        connections.append(item)
    active_connection_count = sum(1 for item in connections if item.get("active"))
    return {
        "roles": sorted(active_roles),
        "expected_connection_count": len(connections),
        "active_connection_count": int(active_connection_count),
        "max_point_error_m": float(max(point_errors)) if point_errors else 0.0,
        "mean_point_error_m": float(np.mean(point_errors)) if point_errors else 0.0,
        "connections": connections,
    }


def _loaded_base_validation_report(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    expected = int(report.get("expected_connection_count", 0))
    configured_min = int(getattr(args, "loaded_base_min_active_connections", 0) or 0)
    required_active = configured_min if configured_min > 0 else expected
    max_allowed = float(getattr(args, "loaded_base_max_point_error", 0.0) or 0.0)
    active_count = int(report.get("active_connection_count", 0) or 0)
    max_error = float(report.get("max_point_error_m", 0.0) or 0.0)
    failures: list[str] = []
    if required_active > 0 and active_count < required_active:
        failures.append("active_connection_count")
    if max_allowed > 0.0 and max_error > max_allowed:
        failures.append("max_point_error")
    return {
        "success": not failures,
        "failures": failures,
        "required_active_connections": required_active,
        "active_connection_count": active_count,
        "max_allowed_point_error_m": max_allowed,
        "max_point_error_m": max_error,
    }


def _argv_option_explicit(argv: list[str], flag: str) -> bool:
    prefix = f"{flag}="
    negated = f"--no-{flag[2:]}" if flag.startswith("--") else None
    for item in argv:
        if item == flag or item.startswith(prefix) or (negated is not None and item == negated):
            return True
    return False


def _apply_strategy_preset(args: argparse.Namespace, argv: list[str]) -> None:
    preset = str(getattr(args, "strategy_preset", "none") or "none")
    if preset == "none":
        return
    presets: dict[str, dict[str, Any]] = {
        "stable_fast_v1": {
            "max_grasp_candidates": 48,
            "fast_screen_max_grasp_candidates": 8,
            "ik_seeds": 64,
            "fast_screen_ik_seeds": 24,
            "triangle_candidate_pool_size": 3,
            "use_screened_release_after_lift": True,
            "replan_release_after_preplace": True,
            "allow_screened_release_fallback": False,
            "require_active_connection_before_open": True,
            "enable_magnets_after_initial_open": True,
            "open_before_magnet_steps": 52,
            "open_before_magnet_fraction": 1.0,
            "post_magnet_capture_hold_steps": 96,
            "pre_open_hold_steps": 0,
            "preplace_staging_mode": "fallback",
            "magnetic_capture_nudge_attempts": 0,
            "magnetic_capture_nudge_step": 0.0022,
            "magnetic_capture_nudge_steps": 14,
            "magnetic_capture_hold_steps": 48,
            "magnetic_capture_max_angle_step_deg": 2.8,
            "magnetic_capture_max_edge_error": 0.0100,
            "magnetic_capture_max_center_error": 0.0090,
            "magnetic_capture_max_edge_angle_deg": 8.0,
            "magnetic_capture_max_joint_delta": 1.4,
            "magnetic_capture_revert_if_worse": True,
            "allow_partial_open_settle_after_active_capture": False,
            "partial_open_settle_fraction": 0.15,
            "partial_open_settle_open_steps": 36,
            "partial_open_settle_hold_steps": 48,
            "partial_open_settle_nudge_attempts": 7,
            "partial_open_settle_nudge_step": 0.0012,
            "partial_open_settle_after_success_max_edge_error": 0.0055,
            "partial_open_settle_after_success_max_center_error": 0.0050,
            "partial_open_settle_after_success_max_edge_angle_deg": 5.0,
            "release_correction_attempts": 0,
            "edge_seating_attempts": 0,
            "edge_seating_steps": 14,
            "edge_seating_max_step": 0.002,
            "edge_seating_max_error": 0.006,
            "edge_seating_center_error": 0.006,
            "edge_seating_angle_error_deg": 4.0,
            "edge_seating_max_angle_step_deg": 3.0,
            "edge_seating_rotation_mode": "world_z",
            "release_gap_mms": "8",
            "release_open_gripper_value": OPEN_GRIPPER,
            "open_steps": 52,
            "post_open_hold_steps": 64,
            "post_success_safe_lift_steps": 48,
            "post_success_safe_lift_height": 0.06,
            "post_success_safe_lift_max_joint_delta": 1.6,
            "post_success_safe_lift_settle_steps": 30,
            "release_yaw_degs": "0,-1,1",
            "max_release_candidates": 72,
            "release_ik_max_position_error": 0.009,
            "release_ik_max_rotation_error": 0.22,
            "preplace_ik_max_position_error": 0.012,
            "preplace_ik_max_rotation_error": 0.22,
            "min_parent_top_edge_z": 0.045,
            "release_preplace_max_joint_delta": 5.3,
            "release_max_joint_delta": 2.8,
            "final_max_base_edge_error": 0.010,
            "final_max_base_center_error": 0.0095,
            "final_max_base_edge_angle_deg": 8.0,
            "pre_open_max_base_edge_error": 0.028,
            "pre_open_max_base_center_error": 0.022,
            "pre_open_max_base_edge_angle_deg": 22.0,
            "require_loaded_base_full_connections": True,
            "loaded_state_settle_steps": 0,
            "loaded_state_post_prune_settle_steps": 8,
            "loaded_base_min_active_connections": 8,
            "loaded_base_max_point_error": 0.0105,
            "magnet_attach_distance": 0.010,
            "magnet_attract_distance": 0.024,
            "magnet_detach_distance": 0.015,
            "magnet_loaded_base_detach_distance": 0.020,
            "magnet_attract_stiffness": 7.5,
            "magnet_attract_force_limit": 3.2,
            "magnet_attract_torque_stiffness": 0.30,
            "magnet_attract_torque_limit": 0.22,
            "magnet_attract_normal_torque_stiffness": 0.72,
            "magnet_attract_normal_torque_limit": 0.45,
            "magnet_active_stiffness": 0.95,
            "magnet_active_damping": 0.18,
            "magnet_active_force_limit": 0.65,
            "magnet_drive_stiffness": 220.0,
            "magnet_drive_damping": 36.0,
            "magnet_drive_force_limit": 22.0,
            "magnet_drive_angular_stiffness": 1.35,
            "magnet_drive_angular_damping": 0.18,
            "magnet_drive_angular_force_limit": 1.8,
            "magnetic_capture_max_downward_step": 0.0,
        },
    }
    preset_values = presets.get(preset)
    if preset_values is None:
        raise ValueError(f"Unknown strategy preset: {preset}")
    for key, value in preset_values.items():
        flag = f"--{key.replace('_', '-')}"
        if _argv_option_explicit(argv, flag):
            continue
        setattr(args, key, value)


def _prepare_triangle_roles(
    base_env: Any,
    *,
    target_lift: float,
    stage_x_start: float,
    stage_x_step: float,
    stage_y: float,
    log: Any,
) -> tuple[dict[str, LockedPanelPose], dict[str, sapien.Pose], dict[str, sapien.Pose]]:
    _remove_free_triangle_roles(base_env)
    snap = base_env.magnetic_snap
    locked_by_role = {locked.role: locked for locked in snap.locked_panel_poses}
    targets: dict[str, sapien.Pose] = {}
    stages: dict[str, sapien.Pose] = {}
    for index, (role, parent_role, parent_edge, base_dir) in enumerate(TRIANGLE_BUILD_SPECS):
        actor = base_env.triangles[index]
        target = _triangle_target_pose_from_parent_edge(
            base_env,
            parent_role=parent_role,
            parent_edge=parent_edge,
            base_dir_hint=base_dir,
            lift=target_lift,
        )
        stage = _triangle_stage_pose(index, x_start=stage_x_start, x_step=stage_x_step, y=stage_y)
        actor.set_pose(stage)
        actor.set_linear_velocity(np.zeros(3, dtype=np.float32))
        actor.set_angular_velocity(np.zeros(3, dtype=np.float32))
        p, q = _pose_arrays(target)
        locked = LockedPanelPose(role=role, actor=actor, position=p, quaternion=q)
        snap.locked_panel_poses.append(locked)
        snap.connections.append(MagneticConnection(parent_role, parent_edge, role, "base_edge", "single_support_base"))
        locked_by_role[role] = locked
        targets[role] = target
        stages[role] = stage
        log(f"{role}: staged at {_pose_to_report(stage)} target={_pose_to_report(target)}")
    return locked_by_role, targets, stages


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


def _current_pose_targets_for_roles(locked: dict[str, Any], roles: list[str]) -> dict[str, sapien.Pose]:
    targets: dict[str, sapien.Pose] = {}
    for role in roles:
        if role not in locked:
            continue
        targets[role] = _actor_pose(locked[role].actor)
    return targets


def _refresh_unloaded_triangle_targets(
    base_env: Any,
    locked: dict[str, Any],
    targets: dict[str, sapien.Pose],
    loaded_roles: list[str],
    *,
    target_lift: float,
    log: Any,
) -> None:
    loaded = set(loaded_roles)
    snap = base_env.magnetic_snap
    for role, parent_role, parent_edge, base_dir in TRIANGLE_BUILD_SPECS:
        if role in loaded or role not in locked:
            continue
        target = _triangle_target_pose_from_parent_edge(
            base_env,
            parent_role=parent_role,
            parent_edge=parent_edge,
            base_dir_hint=base_dir,
            lift=target_lift,
        )
        position, quaternion = _pose_arrays(target)
        refreshed = LockedPanelPose(role=role, actor=locked[role].actor, position=position, quaternion=quaternion)
        snap.locked_panel_poses = [refreshed if item.role == role else item for item in snap.locked_panel_poses]
        locked[role] = refreshed
        targets[role] = target
        log(f"{role}: refreshed target from restored parent edge {_pose_to_report(target)}")


def _refresh_triangle_target_for_role(
    base_env: Any,
    locked: dict[str, Any],
    targets: dict[str, sapien.Pose],
    role: str,
    *,
    target_lift: float,
    log: Any,
) -> None:
    for spec_role, parent_role, parent_edge, base_dir in TRIANGLE_BUILD_SPECS:
        if spec_role != role or role not in locked:
            continue
        target = _triangle_target_pose_from_parent_edge(
            base_env,
            parent_role=parent_role,
            parent_edge=parent_edge,
            base_dir_hint=base_dir,
            lift=target_lift,
        )
        position, quaternion = _pose_arrays(target)
        refreshed = LockedPanelPose(role=role, actor=locked[role].actor, position=position, quaternion=quaternion)
        snap = base_env.magnetic_snap
        snap.locked_panel_poses = [refreshed if item.role == role else item for item in snap.locked_panel_poses]
        locked[role] = refreshed
        targets[role] = target
        log(f"{role}: refreshed target before build from live parent edge {_pose_to_report(target)}")
        return


def _cuboid_for_actor(role: str, actor: Any) -> dict[str, Any]:
    pose = _actor_pose(actor)
    p, q = _pose_arrays(pose)
    if "triangle" in role:
        dims = [TRIANGLE_WIDTH + 0.008, TRIANGLE_THICKNESS + 0.006, TRIANGLE_HEIGHT + 0.008]
        name = f"triangle_{role}"
    else:
        dims = [PLATE_SIZE + 0.006, PLATE_SIZE + 0.006, PLATE_THICKNESS + 0.006]
        name = f"plate_{role}"
    return {
        "name": name,
        "dims": dims,
        "pose": [float(p[0]), float(p[1]), float(p[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])],
    }


def _mesh_for_actor(
    base_env: Any,
    role: str,
    actor: Any,
    *,
    scale: float | list[float] | tuple[float, ...] = 1.0,
) -> dict[str, Any] | None:
    mesh_path = getattr(base_env, "triangle_mesh_path", None)
    if "triangle" not in str(role) or mesh_path is None:
        return None
    mesh_file = Path(mesh_path).expanduser().resolve()
    if not mesh_file.is_file():
        return None
    pose = _actor_pose(actor)
    p, q = _pose_arrays(pose)
    scale_arr = np.asarray(scale, dtype=np.float32).reshape(-1)
    if scale_arr.size == 1:
        scale_arr = np.repeat(scale_arr, 3)
    if scale_arr.size != 3:
        raise ValueError(f"triangle mesh obstacle scale must have 1 or 3 values, got {scale_arr.shape!r}")
    return {
        "name": f"triangle_mesh_{role}",
        "file_path": str(mesh_file),
        "scale": scale_arr.astype(np.float32).tolist(),
        "pose": [float(p[0]), float(p[1]), float(p[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])],
    }


def _robot_base_pose(base_env: Any) -> sapien.Pose:
    pose = base_env.agent.robot.pose.sp
    position = np.asarray(pose.p, dtype=np.float32).reshape(3)
    quaternion = np.asarray(pose.q, dtype=np.float32).reshape(4)
    quaternion = quaternion / max(float(np.linalg.norm(quaternion)), 1e-8)
    return sapien.Pose(p=position.tolist(), q=quaternion.tolist())


def _robot_base_pose_to_world(base_env: Any, pose_in_robot_base: sapien.Pose) -> sapien.Pose:
    return _robot_base_pose(base_env) * pose_in_robot_base


def _planner_fk_world_pose(
    planner: RM75CuRoboPlanner,
    base_env: Any,
    q: np.ndarray,
) -> sapien.Pose:
    pose_dict = planner.fk(np.asarray(q, dtype=np.float32).reshape(7))
    tcp_in_robot_base = sapien.Pose(
        p=np.asarray(pose_dict["position"], dtype=np.float32).reshape(3).tolist(),
        q=np.asarray(pose_dict["quaternion"], dtype=np.float32).reshape(4).tolist(),
    )
    return _robot_base_pose_to_world(base_env, tcp_in_robot_base)


def _world_obstacles(
    base_env: Any,
    locked: dict[str, LockedPanelPose],
    fixtures: list[dict[str, Any]],
    *,
    exclude_roles: set[str] | None = None,
    use_triangle_meshes: bool = False,
    triangle_mesh_scale: float = 1.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    excluded = set(exclude_roles or set())
    cuboids: list[dict[str, Any]] = [
        {
            "name": "table_guard",
            "dims": [1.2, 1.2, 0.035],
            "pose": [0.0, 0.0, -0.02, 1.0, 0.0, 0.0, 0.0],
        }
    ]
    meshes: list[dict[str, Any]] = []
    for role, locked_pose in locked.items():
        if role in excluded:
            continue
        if use_triangle_meshes and "triangle" in str(role):
            mesh = _mesh_for_actor(base_env, role, locked_pose.actor, scale=float(triangle_mesh_scale))
            if mesh is not None:
                meshes.append(mesh)
                continue
        cuboids.append(_cuboid_for_actor(role, locked_pose.actor))
    cuboids.extend({k: v for k, v in fixture.items() if k != "actor"} for fixture in fixtures)
    converted_cuboids = []
    for obstacle in cuboids:
        pose = sapien.Pose(p=obstacle["pose"][:3], q=obstacle["pose"][3:7])
        base_pose = _world_to_robot_base(base_env, pose)
        p, q = _pose_arrays(base_pose)
        item = dict(obstacle)
        item["pose"] = [float(p[0]), float(p[1]), float(p[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        converted_cuboids.append(item)
    converted_meshes = []
    for mesh in meshes:
        pose = sapien.Pose(p=mesh["pose"][:3], q=mesh["pose"][3:7])
        base_pose = _world_to_robot_base(base_env, pose)
        p, q = _pose_arrays(base_pose)
        item = dict(mesh)
        item["pose"] = [float(p[0]), float(p[1]), float(p[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        converted_meshes.append(item)
    return converted_cuboids, converted_meshes


def _set_planner_world(
    planner: RM75CuRoboPlanner,
    base_env: Any,
    locked: dict[str, LockedPanelPose],
    fixtures: list[dict[str, Any]],
    exclude_roles: set[str] | None,
    *,
    args: argparse.Namespace | None = None,
) -> int:
    use_triangle_meshes = bool(getattr(args, "use_triangle_mesh_obstacles", False))
    triangle_mesh_scale = float(getattr(args, "triangle_mesh_obstacle_scale", 1.0))
    cuboids, meshes = _world_obstacles(
        base_env,
        locked,
        fixtures,
        exclude_roles=exclude_roles,
        use_triangle_meshes=use_triangle_meshes,
        triangle_mesh_scale=triangle_mesh_scale,
    )
    planner.set_world_from_obstacles(cuboids=cuboids, meshes=meshes)
    return len(cuboids) + len(meshes)


def _solve_triangle_ik(
    *,
    planner: RM75CuRoboPlanner,
    base_env: Any,
    locked: dict[str, LockedPanelPose],
    fixtures: list[dict[str, Any]],
    target_pose: sapien.Pose,
    start_q: np.ndarray,
    ik_seeds: int,
    exclude_roles: set[str] | None,
    args: argparse.Namespace | None = None,
) -> tuple[bool, np.ndarray | None, dict[str, Any]]:
    obstacle_count = _set_planner_world(planner, base_env, locked, fixtures, exclude_roles, args=args)
    result = planner.solve_ik(start_q, _world_to_robot_base(base_env, target_pose), num_seeds=int(ik_seeds))
    report = {
        "success": bool(result.success),
        "status": result.status,
        "solve_time": float(result.solve_time),
        "ik_time": float(result.ik_time),
        "obstacle_count": obstacle_count,
        "debug": result.debug,
    }
    if not result.success or result.goal_joint is None:
        return False, None, report
    q = np.asarray(result.goal_joint, dtype=np.float32).reshape(-1)[:7]
    report["joint_distance_to_start"] = _joint_distance(q, start_q)
    return True, q, report


def _ik_debug_error_status(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    debug = report.get("debug") if isinstance(report, dict) else {}
    if not isinstance(debug, dict):
        debug = {}
    max_position = float(getattr(args, "release_ik_max_position_error", 0.0))
    max_rotation = float(getattr(args, "release_ik_max_rotation_error", 0.0))
    position_error = float(debug.get("position_error", float("inf")))
    rotation_error = float(debug.get("rotation_error", float("inf")))
    position_ok = bool(max_position <= 0.0 or position_error <= max_position)
    rotation_ok = bool(max_rotation <= 0.0 or rotation_error <= max_rotation)
    return {
        "position_error": position_error,
        "rotation_error": rotation_error,
        "max_position_error": max_position,
        "max_rotation_error": max_rotation,
        "position_ok": position_ok,
        "rotation_ok": rotation_ok,
        "ok": bool(position_ok and rotation_ok),
    }


def _preplace_ik_debug_error_status(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    debug = report.get("debug") if isinstance(report, dict) else {}
    if not isinstance(debug, dict):
        debug = {}
    max_position = float(
        getattr(
            args,
            "preplace_ik_max_position_error",
            getattr(args, "release_ik_max_position_error", 0.0),
        )
    )
    max_rotation = float(
        getattr(
            args,
            "preplace_ik_max_rotation_error",
            getattr(args, "release_ik_max_rotation_error", 0.0),
        )
    )
    position_error = float(debug.get("position_error", float("inf")))
    rotation_error = float(debug.get("rotation_error", float("inf")))
    position_ok = bool(max_position <= 0.0 or position_error <= max_position)
    rotation_ok = bool(max_rotation <= 0.0 or rotation_error <= max_rotation)
    return {
        "position_error": position_error,
        "rotation_error": rotation_error,
        "max_position_error": max_position,
        "max_rotation_error": max_rotation,
        "position_ok": position_ok,
        "rotation_ok": rotation_ok,
        "ok": bool(position_ok and rotation_ok),
    }


def _plan_triangle_motion(
    *,
    planner: RM75CuRoboPlanner,
    base_env: Any,
    locked: dict[str, LockedPanelPose],
    fixtures: list[dict[str, Any]],
    target_pose: sapien.Pose,
    start_q: np.ndarray,
    timeout: float,
    ik_seeds: int,
    exclude_roles: set[str] | None,
    args: argparse.Namespace | None = None,
) -> tuple[bool, np.ndarray | None, dict[str, Any]]:
    obstacle_count = _set_planner_world(planner, base_env, locked, fixtures, exclude_roles, args=args)
    result = planner.plan_to_pose(
        np.asarray(start_q, dtype=np.float32).reshape(7),
        _world_to_robot_base(base_env, target_pose),
        enable_graph=False,
        max_attempts=1,
        timeout=float(timeout),
        num_ik_seeds=int(ik_seeds),
        num_trajopt_seeds=1,
        num_graph_seeds=1,
    )
    report = {
        "success": bool(result.success),
        "status": result.status,
        "solve_time": float(result.solve_time),
        "ik_time": float(result.ik_time),
        "trajopt_time": float(result.trajopt_time),
        "obstacle_count": obstacle_count,
        "debug": result.debug,
    }
    if not result.success or result.joint_path is None:
        return False, None, report
    return True, np.asarray(result.joint_path, dtype=np.float32).reshape(-1, 7), report


def _grasp_screen_passes(args: argparse.Namespace) -> list[dict[str, Any]]:
    full_candidates = max(int(getattr(args, "max_grasp_candidates", 1)), 1)
    full_ik_seeds = max(int(getattr(args, "ik_seeds", 1)), 1)
    fast_candidates = int(getattr(args, "fast_screen_max_grasp_candidates", 0))
    fast_ik_seeds = int(getattr(args, "fast_screen_ik_seeds", 0))
    passes: list[dict[str, Any]] = []
    if 0 < fast_candidates < full_candidates:
        passes.append(
            {
                "label": "fast",
                "max_grasp_candidates": int(fast_candidates),
                "ik_seeds": max(1, min(int(fast_ik_seeds) if fast_ik_seeds > 0 else full_ik_seeds, full_ik_seeds)),
            }
        )
    passes.append(
        {
            "label": "full",
            "max_grasp_candidates": int(full_candidates),
            "ik_seeds": int(full_ik_seeds),
        }
    )
    return passes


def _build_preplace_staging_pose(
    current_tcp_pose: sapien.Pose,
    preplace_pose: sapien.Pose,
    args: argparse.Namespace,
) -> sapien.Pose:
    current_p = np.asarray(current_tcp_pose.p, dtype=np.float32).reshape(3).copy()
    preplace_p = np.asarray(preplace_pose.p, dtype=np.float32).reshape(3)
    z_margin = float(max(getattr(args, "preplace_staging_z_margin", 0.04), 0.0))
    current_p[2] = max(float(current_p[2]), float(preplace_p[2])) + z_margin
    return sapien.Pose(p=current_p.tolist(), q=np.asarray(preplace_pose.q, dtype=np.float32).reshape(4).tolist())


def _solve_triangle_release_option(
    *,
    planner: RM75CuRoboPlanner,
    base_env: Any,
    locked: dict[str, LockedPanelPose],
    fixtures: list[dict[str, Any]],
    role: str,
    release_index: int,
    label: str,
    release_actor_pose: sapien.Pose,
    actor_to_tcp: sapien.Pose,
    preplace_index: int,
    preplace_height: float,
    start_q: np.ndarray,
    ik_seeds: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    _ = role
    target_tcp = release_actor_pose * actor_to_tcp
    preplace_pose = _offset_world(target_tcp, np.asarray([0.0, 0.0, preplace_height], dtype=np.float32))
    staging_mode = str(getattr(args, "preplace_staging_mode", "fallback") or "fallback")
    use_staging = bool(getattr(args, "preplace_staging", True))
    allow_direct_fallback = bool(getattr(args, "allow_direct_preplace_fallback", True))
    search_staging_enabled = bool(use_staging and staging_mode != "off")

    def solve_direct(start_q_value: np.ndarray) -> dict[str, Any]:
        ok_pre, q_pre, pre_ik = _solve_triangle_ik(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            target_pose=preplace_pose,
            start_q=np.asarray(start_q_value, dtype=np.float32).reshape(7),
            ik_seeds=ik_seeds,
            exclude_roles={role},
            args=args,
        )
        ok_rel, q_rel, rel_ik = (False, None, {"skipped": True})
        if ok_pre and q_pre is not None:
            ok_rel, q_rel, rel_ik = _solve_triangle_ik(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                target_pose=target_tcp,
                start_q=q_pre,
                ik_seeds=ik_seeds,
                exclude_roles={role},
                args=args,
            )
        return {
            "path_mode": "direct",
            "stage_pose": None,
            "stage_ik": {"skipped": True, "reason": "direct_first_release_search"},
            "q_stage": None,
            "q_preplace": q_pre,
            "q_release": q_rel,
            "preplace_ik": pre_ik,
            "release_ik": rel_ik,
            "ok_pre": bool(ok_pre),
            "ok_rel": bool(ok_rel),
            "stage_joint_delta_from_start": None,
            "staging_enabled": bool(search_staging_enabled),
            "staging_used": False,
            "staging_direct_fallback_allowed": bool(allow_direct_fallback),
            "release_actor_pose": release_actor_pose,
        }

    def solve_staged(start_q_value: np.ndarray, *, direct_fallback_allowed: bool) -> dict[str, Any]:
        start_q_value = np.asarray(start_q_value, dtype=np.float32).reshape(7)
        current_tcp_pose = _planner_fk_world_pose(planner, base_env, start_q_value)
        stage_pose = _build_preplace_staging_pose(current_tcp_pose, preplace_pose, args)
        stage_ok, q_stage, stage_ik = _solve_triangle_ik(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            target_pose=stage_pose,
            start_q=start_q_value,
            ik_seeds=ik_seeds,
            exclude_roles={role},
            args=args,
        )
        stage_joint_delta = _joint_distance(q_stage, start_q_value) if q_stage is not None else None
        used_staging = bool(stage_ok and q_stage is not None)
        if used_staging:
            preplace_start_q = q_stage
            ok_pre, q_pre, pre_ik = _solve_triangle_ik(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                target_pose=preplace_pose,
                start_q=preplace_start_q,
                ik_seeds=ik_seeds,
                exclude_roles={role},
                args=args,
            )
        elif direct_fallback_allowed:
            ok_pre, q_pre, pre_ik = _solve_triangle_ik(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                target_pose=preplace_pose,
                start_q=start_q_value,
                ik_seeds=ik_seeds,
                exclude_roles={role},
                args=args,
            )
        else:
            ok_pre, q_pre, pre_ik = (
                False,
                None,
                {"skipped": True, "reason": "preplace_staging_failed", "stage_ik": stage_ik},
            )
        ok_rel, q_rel, rel_ik = (False, None, {"skipped": True})
        if ok_pre and q_pre is not None:
            ok_rel, q_rel, rel_ik = _solve_triangle_ik(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                target_pose=target_tcp,
                start_q=q_pre,
                ik_seeds=ik_seeds,
                exclude_roles={role},
                args=args,
            )
        return {
            "path_mode": "staged",
            "stage_pose": stage_pose,
            "stage_ik": stage_ik,
            "q_stage": q_stage,
            "q_preplace": q_pre,
            "q_release": q_rel,
            "preplace_ik": pre_ik,
            "release_ik": rel_ik,
            "ok_pre": bool(ok_pre),
            "ok_rel": bool(ok_rel),
            "stage_joint_delta_from_start": None if stage_joint_delta is None else float(stage_joint_delta),
            "staging_enabled": bool(search_staging_enabled),
            "staging_used": bool(used_staging),
            "staging_direct_fallback_allowed": bool(direct_fallback_allowed),
            "release_actor_pose": release_actor_pose,
        }

    direct_attempt = solve_direct(np.asarray(start_q, dtype=np.float32).reshape(7))
    staged_attempt: dict[str, Any] | None = None
    chosen = direct_attempt
    if staging_mode == "fallback":
        if not direct_attempt["ok_pre"] or direct_attempt["q_preplace"] is None or not direct_attempt["ok_rel"] or direct_attempt["q_release"] is None:
            if search_staging_enabled:
                staged_attempt = solve_staged(start_q, direct_fallback_allowed=False)
                chosen = staged_attempt
    elif staging_mode == "always":
        staged_attempt = solve_staged(start_q, direct_fallback_allowed=allow_direct_fallback)
        chosen = staged_attempt
        if not staged_attempt["ok_pre"] or staged_attempt["q_preplace"] is None or not staged_attempt["ok_rel"] or staged_attempt["q_release"] is None:
            if allow_direct_fallback:
                direct_attempt = solve_direct(start_q)
                if direct_attempt["ok_pre"] and direct_attempt["q_preplace"] is not None and direct_attempt["ok_rel"] and direct_attempt["q_release"] is not None:
                    chosen = direct_attempt
    else:
        chosen = direct_attempt

    q_stage = chosen.get("q_stage")
    q_pre = chosen.get("q_preplace")
    q_rel = chosen.get("q_release")
    stage_ik = chosen.get("stage_ik", {"skipped": True})
    pre_ik = chosen.get("preplace_ik", {"skipped": True})
    rel_ik = chosen.get("release_ik", {"skipped": True})
    stage_pose = chosen.get("stage_pose")
    stage_joint_delta = chosen.get("stage_joint_delta_from_start")
    staging_enabled = bool(chosen.get("staging_enabled", search_staging_enabled))
    staging_used = bool(chosen.get("staging_used", False))
    staging_direct_fallback_allowed = bool(chosen.get("staging_direct_fallback_allowed", allow_direct_fallback))
    preplace_joint_delta = _joint_distance(q_pre, start_q) if q_pre is not None else None
    release_joint_delta = _joint_distance(q_rel, q_pre) if q_pre is not None and q_rel is not None else None
    max_stage_joint_delta = float(getattr(args, "preplace_staging_max_joint_delta", 0.0))
    max_release_preplace_delta = float(getattr(args, "release_preplace_max_joint_delta", 0.0))
    max_release_delta = float(getattr(args, "release_max_joint_delta", 0.0))
    joint_delta_ok = True
    failed_at = None
    if stage_joint_delta is not None and max_stage_joint_delta > 0.0 and stage_joint_delta > max_stage_joint_delta:
        joint_delta_ok = False
        failed_at = "preplace_staging_joint_branch_jump"
    if preplace_joint_delta is not None and max_release_preplace_delta > 0.0 and preplace_joint_delta > max_release_preplace_delta:
        joint_delta_ok = False
        if failed_at is None:
            failed_at = "preplace_joint_branch_jump"
    if release_joint_delta is not None and max_release_delta > 0.0 and release_joint_delta > max_release_delta:
        joint_delta_ok = False
        if failed_at is None:
            failed_at = "release_joint_branch_jump"
    pre_ik_error = _preplace_ik_debug_error_status(pre_ik, args)
    rel_ik_error = _ik_debug_error_status(rel_ik, args)
    ik_error_ok = bool(pre_ik_error["ok"] and rel_ik_error["ok"])
    if not ik_error_ok and failed_at is None:
        failed_at = "release_ik_residual"
    success = bool(chosen.get("ok_pre") and q_pre is not None and chosen.get("ok_rel") and q_rel is not None and joint_delta_ok and ik_error_ok)
    report = {
        "index": release_index * 100 + preplace_index,
        "release_index": release_index,
        "preplace_index": preplace_index,
        "label": f"{label}_preplace_{preplace_height:.3f}m",
        "release_label": label,
        "preplace_height_m": float(preplace_height),
        "staging_enabled": bool(staging_enabled),
        "staging_used": bool(staging_used),
        "staging_direct_fallback_allowed": bool(staging_direct_fallback_allowed),
        "stage_pose": None if stage_pose is None else _pose_to_report(stage_pose),
        "stage_ik": stage_ik,
        "preplace_ik": pre_ik,
        "release_ik": rel_ik,
        "preplace_ik_error": pre_ik_error,
        "release_ik_error": rel_ik_error,
        "ik_error_ok": ik_error_ok,
        "stage_joint_delta_from_start": None if stage_joint_delta is None else float(stage_joint_delta),
        "preplace_joint_delta_from_lift": None if preplace_joint_delta is None else float(preplace_joint_delta),
        "release_joint_delta_from_preplace": None if release_joint_delta is None else float(release_joint_delta),
        "success": success,
        "failed_at": failed_at,
        "q_stage": q_stage,
        "q_preplace": q_pre,
        "q_release": q_rel,
        "release_actor_pose": release_actor_pose,
        "path_mode": chosen.get("path_mode", "direct"),
    }
    if staged_attempt is not None:
        report["staged_attempt"] = {k: v for k, v in staged_attempt.items() if k not in {"q_stage", "q_preplace", "q_release", "release_actor_pose"}}
    report["direct_attempt"] = {k: v for k, v in direct_attempt.items() if k not in {"q_stage", "q_preplace", "q_release", "release_actor_pose"}}
    return report


def _triangle_grasp_candidates(max_candidates: int) -> list[TriangleGraspCandidate]:
    candidates: list[TriangleGraspCandidate] = []
    seen_labels: set[str] = set()
    z_values = [0.062, 0.074, 0.052, 0.088]
    pregrasp_distances = [0.095, 0.075, 0.115]
    x_biases = [0.0, -0.010, 0.010, -0.018, 0.018]
    thin_biases = [0.0, 0.0025, -0.0025]
    approach_biases = [-0.008, -0.014, -0.020, 0.0]
    tilts = [0.0, -6.0, 6.0, -10.0, 10.0]
    yaws = [0.0, 4.0, -4.0]
    variant_order = sorted(
        [
            (thin_bias, approach_bias, tilt, yaw)
            for thin_bias in thin_biases
            for approach_bias in approach_biases
            for tilt in tilts
            for yaw in yaws
        ],
        key=lambda item: (
            abs(float(item[1]) + 0.008),
            0 if abs(float(item[3])) <= 1e-6 else 1,
            abs(float(item[2])),
            0 if abs(float(item[0])) <= 1e-6 else 1,
            abs(float(item[3])),
            abs(float(item[0])),
            abs(float(item[1])),
        ),
    )
    base_specs: list[tuple[float, float, float]] = []
    for local_z in z_values:
        max_x = max(TRIANGLE_WIDTH * 0.5 * (1.0 - local_z / TRIANGLE_HEIGHT) - 0.006, 0.0)
        for x_bias in x_biases:
            local_x = float(np.clip(x_bias, -max_x, max_x))
            for pregrasp_distance in pregrasp_distances:
                base_specs.append((pregrasp_distance, local_z, local_x))
    limit = max(int(max_candidates), 1)
    max_band = len(base_specs) + len(variant_order) - 1
    for band in range(max_band):
        start_base_index = min(band, len(base_specs) - 1)
        min_base_index = max(0, band - (len(variant_order) - 1))
        for base_index in range(start_base_index, min_base_index - 1, -1):
            variant_index = band - base_index
            if variant_index < 0 or variant_index >= len(variant_order):
                continue
            pregrasp_distance, local_z, local_x = base_specs[base_index]
            thin_bias, approach_bias, tilt, yaw = variant_order[variant_index]
            label = (
                f"tri_top_pre_{pregrasp_distance:.3f}_z_{local_z:.3f}_"
                f"x_{local_x:+.3f}_thin_{thin_bias:+.4f}_"
                f"approach_{approach_bias:+.3f}_tilt_{tilt:+.0f}_yaw_{yaw:+.0f}"
            )
            if label in seen_labels:
                continue
            seen_labels.add(label)
            candidates.append(
                TriangleGraspCandidate(
                    label=label,
                    local_x=local_x,
                    local_z=local_z,
                    thin_bias=thin_bias,
                    approach_bias=approach_bias,
                    yaw_deg=yaw,
                    pregrasp_distance=pregrasp_distance,
                    approach_tilt_deg=tilt,
                )
            )
            if len(candidates) >= limit:
                return candidates
    return candidates


def _triangle_grasp_tcp(actor: Any, candidate: TriangleGraspCandidate) -> sapien.Pose:
    actor_pose = _actor_pose(actor)
    position, quaternion = _pose_arrays(actor_pose)
    rotation = quat2mat(quaternion).astype(np.float32)
    local_point = np.asarray([candidate.local_x, 0.0, candidate.local_z], dtype=np.float32)
    grasp_center = position + rotation @ local_point
    thin_normal = _normalize(rotation @ np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    approaching = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    if abs(float(candidate.approach_tilt_deg)) > 1e-6:
        robot_direction = np.asarray([-grasp_center[0], -grasp_center[1], 0.0], dtype=np.float32)
        if float(np.linalg.norm(robot_direction)) <= 1e-5:
            robot_direction = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
        robot_direction = _normalize(robot_direction)
        tilt = float(np.deg2rad(candidate.approach_tilt_deg))
        approaching = _normalize(np.cos(tilt) * approaching + np.sin(tilt) * robot_direction)
    closing_axis = thin_normal - approaching * float(np.dot(thin_normal, approaching))
    if float(np.linalg.norm(closing_axis)) <= 1e-5:
        closing_axis = rotation @ np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        closing_axis = closing_axis - approaching * float(np.dot(closing_axis, approaching))
    closing_axis = _normalize(closing_axis)
    ortho_axis = _normalize(np.cross(closing_axis, approaching))
    closing_axis = _normalize(np.cross(approaching, ortho_axis))
    if abs(float(candidate.yaw_deg)) > 1e-6:
        delta = _axis_angle_matrix(approaching, np.deg2rad(float(candidate.yaw_deg)))
        ortho_axis = _normalize(delta @ ortho_axis)
        closing_axis = _normalize(delta @ closing_axis)
    grasp_center = grasp_center + thin_normal * float(candidate.thin_bias) + approaching * float(candidate.approach_bias)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.stack([ortho_axis, closing_axis, approaching], axis=1)
    matrix[:3, 3] = grasp_center.astype(np.float32)
    return sapien.Pose(p=matrix[:3, 3].tolist(), q=mat2quat(matrix[:3, :3]).astype(np.float32).tolist())


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize(np.asarray(axis, dtype=np.float32))
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    C = 1.0 - c
    return np.asarray(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float32,
    )


def _signed_angle_around_axis(source: np.ndarray, target: np.ndarray, axis: np.ndarray) -> float:
    axis = _normalize(np.asarray(axis, dtype=np.float32))
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    source = source - axis * float(np.dot(source, axis))
    target = target - axis * float(np.dot(target, axis))
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm <= 1e-8 or target_norm <= 1e-8:
        return 0.0
    source = source / source_norm
    target = target / target_norm
    sin_term = float(np.dot(axis, np.cross(source, target)))
    cos_term = float(np.clip(np.dot(source, target), -1.0, 1.0))
    return float(np.arctan2(sin_term, cos_term))


def _pregrasp_pose(grasp_tcp: sapien.Pose, candidate: TriangleGraspCandidate) -> sapien.Pose:
    p, q = _pose_arrays(grasp_tcp)
    rotation = quat2mat(q).astype(np.float32)
    retreat = -rotation[:, 2].astype(np.float32) * float(candidate.pregrasp_distance)
    return sapien.Pose(p=(p + retreat).tolist(), q=q.tolist())


def _release_gap_values(args: argparse.Namespace | None) -> list[float]:
    if args is None:
        return []
    raw = str(getattr(args, "release_gap_mms", "") or "").strip()
    values: list[float] = []
    if raw:
        for item in raw.split(","):
            text = item.strip()
            if text:
                values.append(float(text) / 1000.0)
    elif float(getattr(args, "release_gap_m", 0.0)) > 0.0:
        values.append(float(getattr(args, "release_gap_m", 0.0)))
    deduped: list[float] = []
    for value in values:
        if value <= 0.0:
            continue
        if not any(abs(value - existing) < 1e-6 for existing in deduped):
            deduped.append(value)
    return deduped


def _release_edge_offset_values_mm() -> list[float]:
    plate_thickness_mm = float(PLATE_THICKNESS) * 1000.0
    triangle_thickness_mm = float(TRIANGLE_THICKNESS) * 1000.0
    values = [
        0.0,
        -1.0,
        1.0,
        -2.0,
        2.0,
        -3.0,
        3.0,
        -4.0,
        4.0,
        -plate_thickness_mm,
        plate_thickness_mm,
        -2.0 * plate_thickness_mm,
        2.0 * plate_thickness_mm,
        -triangle_thickness_mm,
        triangle_thickness_mm,
        -2.0 * triangle_thickness_mm,
        2.0 * triangle_thickness_mm,
    ]
    deduped: list[float] = []
    for value in values:
        if not any(abs(value - existing) < 1e-6 for existing in deduped):
            deduped.append(value)
    return deduped


def _release_yaw_values(args: argparse.Namespace | None) -> list[float]:
    if args is None:
        return [0.0]
    raw = str(getattr(args, "release_yaw_degs", "") or "").strip()
    values: list[float] = []
    if raw:
        for item in raw.split(","):
            text = item.strip()
            if text:
                values.append(float(text))
    else:
        values.append(0.0)
    deduped: list[float] = []
    for value in values:
        if not any(abs(value - existing) < 1e-6 for existing in deduped):
            deduped.append(value)
    return deduped or [0.0]


def _release_candidates(
    role: str,
    target_pose: sapien.Pose,
    args: argparse.Namespace | None = None,
) -> list[tuple[str, sapien.Pose]]:
    _ = role
    target_p, target_q = _pose_arrays(target_pose)
    target_rot = quat2mat(target_q).astype(np.float32)
    triangle_normal = _normalize(target_rot @ np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    world_z = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    gap_values = _release_gap_values(args)
    if gap_values:
        edge_axis = _normalize(target_rot @ np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
        yaw_values = _release_yaw_values(args)
        candidates: list[tuple[str, sapien.Pose]] = []
        edge_offsets_mm = _release_edge_offset_values_mm()
        for gap in gap_values:
            for yaw_deg in yaw_values:
                for normal_mm in [0.0, 1.5, -1.5, 3.0]:
                    for edge_mm in edge_offsets_mm:
                        yaw_rot = _axis_angle_matrix(world_z, float(np.deg2rad(yaw_deg))) @ target_rot
                        offset = (
                            world_z * float(gap)
                            + triangle_normal * float(normal_mm / 1000.0)
                            + edge_axis * float(edge_mm / 1000.0)
                        )
                        label = (
                            f"gap_{gap * 1000.0:.1f}mm_edge_{edge_mm:+.1f}mm_"
                            f"normal_{normal_mm:+.1f}mm_yaw_{yaw_deg:+.1f}deg"
                        )
                        candidates.append(
                            (
                                label,
                                sapien.Pose(
                                    p=(target_p + offset).astype(np.float32).tolist(),
                                    q=mat2quat(yaw_rot).astype(np.float32).tolist(),
                                ),
                            ),
                        )
        max_candidates = int(getattr(args, "max_release_candidates", 0))
        if max_candidates > 0:
            candidates = sorted(
                candidates,
                key=lambda item: (
                    _predicted_triangle_base_edge_alignment(item[1], target_pose)["max_point_error_m"],
                    _predicted_triangle_base_edge_alignment(item[1], target_pose)["center_error_m"],
                    _predicted_triangle_base_edge_alignment(item[1], target_pose)["edge_parallel_error_deg"],
                ),
            )
        if max_candidates > 0:
            candidates = candidates[:max_candidates]
        return candidates
    offsets: list[tuple[str, np.ndarray]] = [("exact_target", np.zeros(3, dtype=np.float32))]
    for lift_mm in [-1.5, 1.0, 2.0, 4.0]:
        offsets.append((f"z_{lift_mm:+.1f}mm", np.asarray([0.0, 0.0, lift_mm / 1000.0], dtype=np.float32)))
    for normal_mm in [-5.0, -3.0, 3.0, 5.0]:
        offsets.append((f"normal_{normal_mm:+.1f}mm", triangle_normal * float(normal_mm / 1000.0)))
    for normal_mm in [-3.0, 3.0]:
        for lift_mm in [-1.0, 1.5]:
            offset = triangle_normal * float(normal_mm / 1000.0) + np.asarray([0.0, 0.0, lift_mm / 1000.0], dtype=np.float32)
            offsets.append((f"normal_{normal_mm:+.1f}mm_z_{lift_mm:+.1f}mm", offset))
    candidates: list[tuple[str, sapien.Pose]] = []
    for label, offset in offsets:
        candidates.append((label, sapien.Pose(p=(target_p + offset).astype(np.float32).tolist(), q=target_q.tolist())))
    world_z = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    for yaw_deg in [-4.0, -3.0, -2.0, 2.0, 3.0, 4.0]:
        yaw_rot = _axis_angle_matrix(world_z, float(np.deg2rad(yaw_deg))) @ target_rot
        candidates.append(
            (
                f"yaw_{yaw_deg:+.1f}deg",
                sapien.Pose(p=target_p.astype(np.float32).tolist(), q=mat2quat(yaw_rot).astype(np.float32).tolist()),
            )
        )
    edge_axis = _normalize(target_rot @ np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    edge_offsets_mm = _release_edge_offset_values_mm()
    for normal_mm in [1.0, 2.0, 3.0, 5.0]:
        for edge_mm in edge_offsets_mm:
            offset = edge_axis * float(edge_mm / 1000.0) + triangle_normal * float(normal_mm / 1000.0)
            candidates.append(
                (
                    f"edge_{edge_mm:+.1f}mm_normal_{normal_mm:+.1f}mm",
                    sapien.Pose(p=(target_p + offset).astype(np.float32).tolist(), q=target_q.tolist()),
                )
            )
    for normal_mm in [-1.0, -2.0, -3.0]:
        for edge_mm in edge_offsets_mm:
            offset = edge_axis * float(edge_mm / 1000.0) + triangle_normal * float(normal_mm / 1000.0)
            candidates.append(
                (
                    f"edge_{edge_mm:+.1f}mm_normal_{normal_mm:+.1f}mm",
                    sapien.Pose(p=(target_p + offset).astype(np.float32).tolist(), q=target_q.tolist()),
                )
            )
    for lift_mm in [3.0]:
        offset = np.asarray([0.0, 0.0, lift_mm / 1000.0], dtype=np.float32)
        candidates.append((f"z_{lift_mm:+.1f}mm_late", sapien.Pose(p=(target_p + offset).astype(np.float32).tolist(), q=target_q.tolist())))
    for normal_mm in [1.0, 2.0, 3.0]:
        for edge_mm in edge_offsets_mm:
            for lift_mm in [2.0, 3.0]:
                offset = (
                    edge_axis * float(edge_mm / 1000.0)
                    + triangle_normal * float(normal_mm / 1000.0)
                    + np.asarray([0.0, 0.0, lift_mm / 1000.0], dtype=np.float32)
                )
                candidates.append(
                    (
                        f"edge_{edge_mm:+.1f}mm_normal_{normal_mm:+.1f}mm_z_{lift_mm:+.1f}mm",
                        sapien.Pose(p=(target_p + offset).astype(np.float32).tolist(), q=target_q.tolist()),
                    )
                )
    for lift_mm in [3.0]:
        for yaw_deg in [-2.0, -1.0, 1.0, 2.0]:
            yaw_rot = _axis_angle_matrix(world_z, float(np.deg2rad(yaw_deg))) @ target_rot
            offset = np.asarray([0.0, 0.0, lift_mm / 1000.0], dtype=np.float32)
            candidates.append(
                (
                    f"z_{lift_mm:+.1f}mm_yaw_{yaw_deg:+.1f}deg",
                    sapien.Pose(p=(target_p + offset).astype(np.float32).tolist(), q=mat2quat(yaw_rot).astype(np.float32).tolist()),
                )
            )
    return candidates


def _preplace_heights(args: argparse.Namespace) -> list[float]:
    raw = str(getattr(args, "preplace_heights", "") or "").strip()
    values: list[float] = []
    if raw:
        for item in raw.replace("|", ",").replace(";", ",").split(","):
            text = item.strip()
            if not text:
                continue
            values.append(float(text))
    values.append(float(getattr(args, "preplace_height", 0.055)))
    deduped: list[float] = []
    for value in values:
        if value < 0.0:
            continue
        if not any(abs(value - existing) < 1e-6 for existing in deduped):
            deduped.append(value)
    return deduped or [float(getattr(args, "preplace_height", 0.055))]


def _predicted_triangle_base_edge_alignment(release_pose: sapien.Pose, target_pose: sapien.Pose) -> dict[str, float]:
    half_width = float(TRIANGLE_WIDTH) / 2.0
    local_points = [
        np.asarray([-half_width, 0.0, 0.0], dtype=np.float32),
        np.asarray([half_width, 0.0, 0.0], dtype=np.float32),
    ]
    release_p, release_q = _pose_arrays(release_pose)
    target_p, target_q = _pose_arrays(target_pose)
    release_rot = quat2mat(release_q).astype(np.float32)
    target_rot = quat2mat(target_q).astype(np.float32)
    release_world = [release_p + release_rot @ point for point in local_points]
    target_world = [target_p + target_rot @ point for point in local_points]
    deltas = [
        np.asarray(release_point, dtype=np.float32) - np.asarray(target_point, dtype=np.float32)
        for release_point, target_point in zip(release_world, target_world)
    ]
    z_clearances = [float(delta[2]) for delta in deltas]
    distances = [float(np.linalg.norm(delta)) for delta in deltas]
    center_delta = np.mean(np.asarray(deltas, dtype=np.float32), axis=0)
    release_dir = _normalize(release_world[1] - release_world[0])
    target_dir = _normalize(target_world[1] - target_world[0])
    if float(np.dot(release_dir, target_dir)) < 0.0:
        release_dir = -release_dir
    edge_dot = float(np.clip(np.dot(release_dir, target_dir), -1.0, 1.0))
    edge_angle = float(np.arccos(edge_dot))
    return {
        "max_point_error_m": float(max(distances)) if distances else 0.0,
        "center_error_m": float(np.linalg.norm(center_delta)),
        "edge_parallel_error_deg": float(np.rad2deg(edge_angle)),
        "min_world_z_clearance_m": float(min(z_clearances)) if z_clearances else 0.0,
        "mean_world_z_clearance_m": float(np.mean(z_clearances)) if z_clearances else 0.0,
    }


def _triangle_release_option_score(option: dict[str, Any], target_pose: sapien.Pose) -> tuple[float, float, float, int]:
    release_pose = option["release_actor_pose"]
    release_p, _ = _pose_arrays(release_pose)
    target_p, target_q = _pose_arrays(target_pose)
    target_rot = quat2mat(target_q).astype(np.float32)
    normal_axis = _normalize(target_rot @ np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    edge_axis = _normalize(target_rot @ np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    delta = release_p - target_p
    normal_offset = float(np.dot(delta, normal_axis))
    edge_offset = float(np.dot(delta, edge_axis))
    z_offset = float(delta[2])
    pose_error = option.get("pose_error") or _pose_error_proxy(release_pose, target_pose)
    pre_delta = float(option.get("preplace_joint_delta_from_lift") or option.get("preplace_joint_delta_from_current") or 0.0)
    release_delta = float(option.get("release_joint_delta_from_preplace") or 0.0)
    preplace_height = float(option.get("preplace_height_m", 0.0))
    predicted_alignment = _predicted_triangle_base_edge_alignment(release_pose, target_pose)
    min_world_z_clearance = float(predicted_alignment.get("min_world_z_clearance_m", 0.0))
    release_ik_error = option.get("release_ik_error") if isinstance(option.get("release_ik_error"), dict) else {}
    release_ik_position_error = float(release_ik_error.get("position_error", 0.0))
    release_ik_rotation_error = float(release_ik_error.get("rotation_error", 0.0))
    score = 0.0
    score += float(predicted_alignment["max_point_error_m"]) * 1.2
    score += float(predicted_alignment["center_error_m"]) * 0.8
    score += float(predicted_alignment["edge_parallel_error_deg"]) * 0.0009
    score += max(0.008 - min_world_z_clearance, 0.0) * 6.0
    score += abs(edge_offset) * 1.8
    score += abs(normal_offset) * 0.75
    score += abs(z_offset - 0.008) * 0.12
    score += max(-normal_offset, 0.0) * 1.5
    score += max(-z_offset, 0.0) * 1.0
    score += float(pose_error["position_error_m"]) * 0.55
    score += float(pose_error["orientation_error_deg"]) * 0.0008
    score += release_ik_position_error * 2.0
    score += release_ik_rotation_error * 0.018
    score += pre_delta * 0.0015
    score += release_delta * 0.0010
    if bool(option.get("staging_used", False)):
        score += 0.0040
    score += max(pre_delta - 4.0, 0.0) * 0.0400
    score += max(release_delta - 2.0, 0.0) * 0.0250
    score += abs(preplace_height - 0.040) * 0.0030
    return (float(score), pre_delta, release_delta, int(option["index"]))


def _triangle_bundle_score(bundle: dict[str, Any], target_pose: sapien.Pose) -> tuple[float, float, float, float, int]:
    release_score = _triangle_release_option_score(bundle["release"], target_pose)
    pregrasp_distance = float(bundle.get("pregrasp_joint_distance", 0.0))
    grasp_delta = float(bundle.get("grasp_joint_delta_from_pregrasp", 0.0))
    lift_delta = float(bundle.get("lift_joint_delta_from_grasp", 0.0))
    release = bundle["release"]
    preplace_delta = float(release.get("preplace_joint_delta_from_lift") or 0.0)
    release_delta = float(release.get("release_joint_delta_from_preplace") or 0.0)
    candidate: TriangleGraspCandidate = bundle["candidate"]
    clearance_preference = abs(float(candidate.pregrasp_distance) - 0.095) * 0.0100
    short_clearance_penalty = max(0.090 - float(candidate.pregrasp_distance), 0.0) * 0.0600
    tilt_penalty = abs(float(candidate.approach_tilt_deg)) * 0.0010
    yaw_penalty = abs(float(candidate.yaw_deg)) * 0.0008
    smoothness = (
        pregrasp_distance * 0.0030
        + grasp_delta * 0.0060
        + lift_delta * 0.0040
        + preplace_delta * 0.0100
        + release_delta * 0.0080
        + clearance_preference
        + short_clearance_penalty
        + tilt_penalty
        + yaw_penalty
    )
    return (
        float(release_score[0] + smoothness),
        preplace_delta,
        release_delta,
        pregrasp_distance + grasp_delta + lift_delta,
        int(bundle["candidate_index"]),
    )


def _select_release_after_preplace(
    *,
    planner: RM75CuRoboPlanner,
    base_env: Any,
    locked: dict[str, LockedPanelPose],
    fixtures: list[dict[str, Any]],
    role: str,
    actor: Any,
    target_pose: sapien.Pose,
    preplace_height: float,
    start_q: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    actor_to_tcp = _actor_pose(actor).inv() * _tcp_pose(base_env)
    release_options: list[dict[str, Any]] = []
    for release_index, (label, release_actor_pose) in enumerate(_release_candidates(role, target_pose, args)):
        if not _release_index_allowed(args, release_index):
            continue
        desired_tcp = release_actor_pose * actor_to_tcp
        ok_rel, q_rel, rel_ik = _solve_triangle_ik(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            target_pose=desired_tcp,
            start_q=start_q,
            ik_seeds=args.ik_seeds,
            exclude_roles={role},
            args=args,
        )
        release_delta = None if q_rel is None else float(_joint_distance(q_rel, start_q))
        max_release_delta = float(getattr(args, "release_max_joint_delta", 0.0))
        rel_ik_error = _ik_debug_error_status(rel_ik, args)
        success = bool(
            ok_rel
            and q_rel is not None
            and (max_release_delta <= 0.0 or (release_delta is not None and release_delta <= max_release_delta))
            and rel_ik_error["ok"]
        )
        option = {
            "index": release_index * 1000,
            "release_index": release_index,
            "preplace_index": -1,
            "label": f"{label}_replan_from_preplace",
            "release_label": label,
            "preplace_height_m": float(preplace_height),
            "staging_enabled": False,
            "staging_used": False,
            "staging_direct_fallback_allowed": False,
            "stage_pose": None,
            "stage_ik": {"skipped": True, "reason": "replan after preplace"},
            "preplace_ik": {"skipped": True, "reason": "already at preplace"},
            "release_ik": rel_ik,
            "release_ik_error": rel_ik_error,
            "ik_error_ok": bool(rel_ik_error["ok"]),
            "stage_joint_delta_from_start": None,
            "preplace_joint_delta_from_current": 0.0,
            "release_joint_delta_from_preplace": None if release_delta is None else float(release_delta),
            "success": bool(success),
            "failed_at": None if success else ("release_joint_branch_jump" if release_delta is not None and max_release_delta > 0.0 and release_delta > max_release_delta else "release_ik"),
            "q_stage": None,
            "q_preplace": np.asarray(start_q, dtype=np.float32).reshape(7),
            "q_release": q_rel,
            "release_actor_pose": release_actor_pose,
            "path_mode": "replanned_release_only",
            "pose_error": _pose_error_proxy(release_actor_pose, target_pose),
        }
        release_options.append(option)
    selected = min(
        [option for option in release_options if option["success"]],
        key=lambda option: _triangle_release_option_score(option, target_pose),
        default=None,
    )
    return selected, release_options


def _current_start_state_report(
    *,
    planner: RM75CuRoboPlanner,
    base_env: Any,
    locked: dict[str, LockedPanelPose],
    fixtures: list[dict[str, Any]],
    exclude_roles: set[str] | None = None,
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    current_q = _current_q(base_env)
    obstacle_count = _set_planner_world(planner, base_env, locked, fixtures, exclude_roles=exclude_roles or set(), args=args)
    valid, status = planner.check_start_state(current_q)
    return {
        "valid": bool(valid),
        "status": status,
        "obstacle_count": int(obstacle_count),
        "exclude_roles": sorted(exclude_roles or set()),
        "q": current_q.astype(np.float32).tolist(),
    }


def _recover_to_valid_start_state(
    *,
    env: Any,
    planner: RM75CuRoboPlanner,
    base_env: Any,
    locked: dict[str, LockedPanelPose],
    fixtures: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    segments: list[dict[str, Any]],
    role: str,
    args: argparse.Namespace,
    segment_prefix: str | None = None,
) -> dict[str, Any]:
    current_q = _current_q(base_env)
    candidate_qs = [
        ("rm75_home", np.asarray(RM75_HOME, dtype=np.float32).reshape(7)),
        ("default_retract", np.asarray(DEFAULT_RETRACT_CONFIG, dtype=np.float32).reshape(7)),
    ]
    options: list[dict[str, Any]] = []
    _set_planner_world(planner, base_env, locked, fixtures, exclude_roles=set(), args=args)
    for label, q in candidate_qs:
        valid, status = planner.check_start_state(q)
        joint_delta = float(_joint_distance(q, current_q))
        options.append(
            {
                "label": label,
                "valid": bool(valid),
                "status": status,
                "joint_delta": joint_delta,
                "q": q,
            }
        )
    selected = min(
        [option for option in options if option["valid"]],
        key=lambda option: (float(option["joint_delta"]), str(option["label"])),
        default=None,
    )
    report: dict[str, Any] = {
        "enabled": True,
        "options": [{key: value for key, value in option.items() if key != "q"} for option in options],
        "selected": None if selected is None else {key: value for key, value in selected.items() if key != "q"},
    }
    if selected is None:
        report["success"] = False
        report["after"] = _current_start_state_report(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            args=args,
        )
        return report
    recovery_args = argparse.Namespace(**vars(args))
    recovery_args.max_joint_step = min(float(getattr(args, "max_joint_step", 0.032)), 0.024)
    recovery_args.max_segment_steps = max(int(getattr(args, "max_segment_steps", 900)), 1200)
    _add_adaptive_joint_segment(
        env=env,
        arrays=arrays,
        segments=segments,
        name=f"{role}_{segment_prefix or 'post_execution_valid_start_recovery'}",
        goal_q=selected["q"],
        gripper=float(getattr(args, "release_open_gripper_value", 0.0)),
        base_steps=max(int(getattr(args, "post_success_safe_lift_steps", 36)), 56),
        action_repeat=int(getattr(args, "action_repeat", 1)),
        final_hold=int(getattr(args, "final_hold_steps", 0)),
        args=recovery_args,
    )
    _add_hold_segment(
        env,
        segments,
        f"{role}_{segment_prefix or 'post_execution_valid_start_recovery'}_settle",
        float(getattr(args, "release_open_gripper_value", 0.0)),
        max(int(getattr(args, "post_success_safe_lift_settle_steps", 0)), 20),
    )
    report["after"] = _current_start_state_report(
        planner=planner,
        base_env=base_env,
        locked=locked,
        fixtures=fixtures,
        args=args,
    )
    report["success"] = bool(report["after"]["valid"])
    return report


def _role_mapping(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in str(text or "").split(","):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key:
            mapping[key] = value
    return mapping


def _role_int_list_mapping(text: str) -> dict[str, set[int]]:
    mapping: dict[str, set[int]] = {}
    for item in str(text or "").split(","):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        role = key.strip()
        if not role:
            continue
        values: set[int] = set()
        for raw in value.replace("|", ";").split(";"):
            raw = raw.strip()
            if not raw:
                continue
            values.add(int(raw))
        if values:
            mapping[role] = values
    return mapping


def _allowed_release_candidate_indices(args: argparse.Namespace) -> set[int] | None:
    values = getattr(args, "allowed_release_candidate_indices", None)
    if values is None:
        return None
    if isinstance(values, set):
        return set(int(item) for item in values)
    if isinstance(values, (list, tuple)):
        return set(int(item) for item in values)
    text = str(values).strip()
    if not text:
        return None
    return {int(item.strip()) for item in text.replace("|", ";").split(";") if item.strip()}


def _release_index_allowed(args: argparse.Namespace, release_index: int) -> bool:
    allowed = _allowed_release_candidate_indices(args)
    if allowed is not None:
        return int(release_index) in allowed
    return int(getattr(args, "release_candidate_index", -1)) < 0 or int(release_index) == int(args.release_candidate_index)


def _apply_role_mapping(role_args: argparse.Namespace, role: str, attr: str, mapping_text: str, cast: Any) -> None:
    mapping = _role_mapping(mapping_text)
    if role not in mapping:
        return
    setattr(role_args, attr, cast(mapping[role]))


def _args_for_role(args: argparse.Namespace, role: str) -> argparse.Namespace:
    role_args = argparse.Namespace(**vars(args))
    if str(getattr(args, "auto_release_profile", "") or "").strip() == "v6_physical":
        role_args.release_candidate_index = -1
        role_args.open_steps = max(int(role_args.open_steps), 36)
        role_args.pre_open_hold_steps = max(int(role_args.pre_open_hold_steps), 50)
        role_args.post_open_hold_steps = max(int(role_args.post_open_hold_steps), 40)
        role_args.retreat_mode = "none"
        role_args.retreat_distance = 0.0
    _apply_role_mapping(role_args, role, "release_candidate_index", getattr(args, "release_candidate_indices", ""), int)
    _apply_role_mapping(role_args, role, "release_open_gripper_value", getattr(args, "release_open_gripper_values", ""), float)
    _apply_role_mapping(role_args, role, "retreat_mode", getattr(args, "retreat_modes", ""), str)
    _apply_role_mapping(role_args, role, "retreat_distance", getattr(args, "retreat_distances", ""), float)
    _apply_role_mapping(role_args, role, "open_steps", getattr(args, "open_steps_by_role", ""), int)
    _apply_role_mapping(role_args, role, "pre_open_hold_steps", getattr(args, "pre_open_hold_steps_by_role", ""), int)
    _apply_role_mapping(role_args, role, "post_open_hold_steps", getattr(args, "post_open_hold_steps_by_role", ""), int)
    _apply_role_mapping(role_args, role, "edge_seating_attempts", getattr(args, "edge_seating_attempts_by_role", ""), int)
    _apply_role_mapping(role_args, role, "release_correction_attempts", getattr(args, "release_correction_attempts_by_role", ""), int)
    _apply_role_mapping(role_args, role, "preplace_heights", getattr(args, "preplace_heights_by_role", ""), str)
    _apply_role_mapping(role_args, role, "preplace_height", getattr(args, "preplace_height_by_role", ""), float)
    allowed_mapping = _role_int_list_mapping(getattr(args, "release_candidate_index_groups", ""))
    if role in allowed_mapping:
        role_args.allowed_release_candidate_indices = allowed_mapping[role]
    return role_args


def _retreat_pose_for_release(
    *,
    tcp_pose: sapien.Pose,
    actor_pose: sapien.Pose,
    mode: str,
    distance: float,
) -> sapien.Pose:
    tcp_p, tcp_q = _pose_arrays(tcp_pose)
    actor_p, actor_q = _pose_arrays(actor_pose)
    tcp_rot = quat2mat(tcp_q).astype(np.float32)
    actor_rot = quat2mat(actor_q).astype(np.float32)
    mode = str(mode or "world_z")
    if mode == "none" or float(distance) <= 0.0:
        direction = np.zeros(3, dtype=np.float32)
    elif mode == "tcp_back":
        direction = _normalize(-tcp_rot[:, 2].astype(np.float32))
    elif mode == "tcp_forward":
        direction = _normalize(tcp_rot[:, 2].astype(np.float32))
    elif mode == "tcp_open_axis_pos":
        direction = _normalize(tcp_rot[:, 1].astype(np.float32))
    elif mode == "tcp_open_axis_neg":
        direction = _normalize(-tcp_rot[:, 1].astype(np.float32))
    elif mode == "triangle_normal_pos":
        direction = _normalize(actor_rot @ np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    elif mode == "triangle_normal_neg":
        direction = _normalize(-(actor_rot @ np.asarray([0.0, 1.0, 0.0], dtype=np.float32)))
    else:
        direction = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return sapien.Pose(p=(tcp_p + direction * float(distance)).astype(np.float32).tolist(), q=tcp_q.tolist())


def _release_retreat_options(
    *,
    tcp_pose: sapien.Pose,
    actor_pose: sapien.Pose,
    primary_mode: str,
    primary_distance: float,
) -> list[tuple[str, sapien.Pose]]:
    modes = [
        str(primary_mode or "tcp_back"),
        "tcp_back",
        "world_z",
        "triangle_normal_pos",
        "triangle_normal_neg",
        "tcp_open_axis_pos",
        "tcp_open_axis_neg",
    ]
    distances = [
        float(primary_distance),
        max(float(primary_distance) * 0.5, 0.0),
        0.025,
        0.035,
        0.055,
        0.070,
    ]
    options: list[tuple[str, sapien.Pose]] = []
    seen: set[tuple[str, float]] = set()
    for mode in modes:
        for distance in distances:
            if mode == "none" or distance <= 1e-6:
                continue
            key = (mode, round(float(distance), 4))
            if key in seen:
                continue
            seen.add(key)
            options.append(
                (
                    f"{mode}_{float(distance):.3f}m",
                    _retreat_pose_for_release(
                        tcp_pose=tcp_pose,
                        actor_pose=actor_pose,
                        mode=mode,
                        distance=float(distance),
                    ),
                )
            )
    options.append(
        (
            "none",
            _retreat_pose_for_release(
                tcp_pose=tcp_pose,
                actor_pose=actor_pose,
                mode="none",
                distance=0.0,
            ),
        )
    )
    return options


def _role_connection_point_error(base_env: Any, role: str) -> dict[str, Any]:
    snap = base_env.magnetic_snap
    items: list[dict[str, Any]] = []
    for connection in snap.connections:
        if connection.parent != role and connection.child != role:
            continue
        try:
            point_error = float(snap._connection_point_error(connection))
        except Exception as exc:
            items.append(
                {
                    "parent": connection.parent,
                    "parent_edge": connection.parent_edge,
                    "child": connection.child,
                    "child_edge": connection.child_edge,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        active = any(
            active_connection.active
            and active_connection.connection.parent == connection.parent
            and active_connection.connection.parent_edge == connection.parent_edge
            and active_connection.connection.child == connection.child
            and active_connection.connection.child_edge == connection.child_edge
            for active_connection in snap.active_connections
        )
        items.append(
            {
                "parent": connection.parent,
                "parent_edge": connection.parent_edge,
                "child": connection.child,
                "child_edge": connection.child_edge,
                "mode": connection.mode,
                "point_error_m": point_error,
                "active": bool(active),
            }
        )
    return {
        "role": role,
        "attach_distance_m": float(snap.attach_distance),
        "connections": items,
    }


def _triangle_base_edge_alignment(base_env: Any, locked: dict[str, LockedPanelPose], role: str) -> dict[str, Any]:
    snap = base_env.magnetic_snap
    locked_by_role = {item.role: item for item in snap.locked_panel_poses}
    parent = None
    child = locked_by_role.get(role) or locked.get(role)
    connection = None
    for item in snap.connections:
        if item.child == role and item.child_edge == "base_edge":
            connection = item
            parent = locked_by_role.get(item.parent) or locked.get(item.parent)
            break
        if item.parent == role and item.parent_edge == "base_edge":
            connection = item
            parent = locked_by_role.get(item.child) or locked.get(item.child)
            break
    if parent is None or child is None or connection is None:
        return {"success": False, "role": role, "error": "missing base-edge connection"}
    parent_points, child_points = snap._matched_connection_points(
        connection,
        parent if connection.parent != role else child,
        child if connection.child == role else parent,
        points_per_edge=2,
        use_locked_pose=False,
    )
    if connection.parent == role:
        parent_points, child_points = child_points, parent_points
    parent_world = snap._current_world_points(parent.actor, parent_points)
    child_world = snap._current_world_points(child.actor, child_points)
    deltas = [np.asarray(child_point, dtype=np.float32) - np.asarray(parent_point, dtype=np.float32) for parent_point, child_point in zip(parent_world, child_world)]
    distances = [float(np.linalg.norm(delta)) for delta in deltas]
    center_delta = np.mean(np.asarray(deltas, dtype=np.float32), axis=0).astype(np.float32)
    parent_edge = snap._edge_spec(parent.role, connection.parent_edge if connection.parent != role else connection.child_edge, "rim")
    child_edge = snap._edge_spec(child.role, "base_edge", "rim")
    parent_endpoints = snap._current_world_points(parent.actor, [parent_edge.start, parent_edge.end])
    child_endpoints = snap._current_world_points(child.actor, [child_edge.start, child_edge.end])
    parent_dir = _normalize(parent_endpoints[1] - parent_endpoints[0])
    child_dir = _normalize(child_endpoints[1] - child_endpoints[0])
    if float(np.dot(parent_dir, child_dir)) < 0.0:
        child_dir = -child_dir
    edge_dot = float(np.clip(np.dot(child_dir, parent_dir), -1.0, 1.0))
    edge_axis = np.cross(child_dir, parent_dir).astype(np.float32)
    edge_axis_norm = float(np.linalg.norm(edge_axis))
    edge_angle = float(np.arctan2(edge_axis_norm, edge_dot))
    if edge_axis_norm > 1e-8:
        edge_axis = (edge_axis / edge_axis_norm).astype(np.float32)
    else:
        edge_axis = np.zeros(3, dtype=np.float32)
    return {
        "success": True,
        "role": role,
        "parent": parent.role,
        "parent_edge": connection.parent_edge if connection.parent != role else connection.child_edge,
        "child": child.role,
        "child_edge": "base_edge",
        "point_errors_m": distances,
        "max_point_error_m": float(max(distances)) if distances else 0.0,
        "mean_point_error_m": float(np.mean(distances)) if distances else 0.0,
        "center_delta_child_minus_parent_m": center_delta.tolist(),
        "center_error_m": float(np.linalg.norm(center_delta)),
        "edge_parallel_error_deg": float(np.rad2deg(abs(edge_angle))),
        "edge_rotation_axis_world": edge_axis.tolist(),
        "edge_rotation_angle_rad": edge_angle,
        "parent_edge_dir_world": parent_dir.tolist(),
        "child_edge_dir_world": child_dir.tolist(),
    }


def _clamp_vector(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if max_norm <= 0.0:
        return np.zeros_like(vec, dtype=np.float32)
    if norm <= 1e-8 or norm <= max_norm:
        return vec.astype(np.float32)
    return (vec / norm * float(max_norm)).astype(np.float32)


def _edge_alignment_score(alignment: dict[str, Any]) -> float:
    if not alignment.get("success"):
        return float("inf")
    max_error = float(alignment.get("max_point_error_m", float("inf")))
    center_error = float(alignment.get("center_error_m", float("inf")))
    angle_error = float(alignment.get("edge_parallel_error_deg", float("inf")))
    return max_error + center_error + 0.0012 * angle_error


def _role_connection_score(connection_error: dict[str, Any]) -> float:
    point_errors = [
        float(item["point_error_m"])
        for item in connection_error.get("connections", [])
        if isinstance(item.get("point_error_m"), (int, float)) and np.isfinite(float(item["point_error_m"]))
    ]
    return float(min(point_errors)) if point_errors else float("inf")


def _role_active_connection_report(base_env: Any, role: str) -> dict[str, Any]:
    connection_error = _role_connection_point_error(base_env, role)
    attach_distance = float(connection_error.get("attach_distance_m", float("inf")))
    active_errors = [
        float(item["point_error_m"])
        for item in connection_error.get("connections", [])
        if item.get("active")
        and isinstance(item.get("point_error_m"), (int, float))
        and np.isfinite(float(item["point_error_m"]))
    ]
    return {
        "role": role,
        "attach_distance_m": attach_distance,
        "active_count": int(len(active_errors)),
        "min_active_point_error_m": float(min(active_errors)) if active_errors else float("inf"),
        "connection_point_error": connection_error,
        "satisfied": bool(active_errors and min(active_errors) <= attach_distance + 1e-6),
    }


def _release_correction_score(pose_error: dict[str, Any], connection_error: dict[str, Any]) -> float:
    return (
        float(pose_error.get("position_error_m", float("inf")))
        + _role_connection_score(connection_error)
        + 0.0008 * float(pose_error.get("orientation_error_deg", float("inf")))
    )


def _pose_error_proxy(current: sapien.Pose, target: sapien.Pose) -> dict[str, float]:
    current_p, current_q = _pose_arrays(current)
    target_p, target_q = _pose_arrays(target)
    current_rot = quat2mat(current_q).astype(np.float32)
    target_rot = quat2mat(target_q).astype(np.float32)
    delta = current_rot.T @ target_rot
    trace = float(np.trace(delta))
    angle = float(np.rad2deg(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))))
    return {
        "position_error_m": float(np.linalg.norm(current_p - target_p)),
        "orientation_error_deg": angle,
    }


def _actor_state_report(actor: Any, target: sapien.Pose, base_env: Any, role: str) -> dict[str, Any]:
    pose_error = _pose_error(actor, target)
    actor_p, actor_q = _pose_arrays(_actor_pose(actor))
    finite = bool(np.all(np.isfinite(actor_p)) and np.all(np.isfinite(actor_q)))
    workspace_radius = float(np.linalg.norm(actor_p))
    connection_error = _role_connection_point_error(base_env, role)
    point_errors = [
        float(item["point_error_m"])
        for item in connection_error.get("connections", [])
        if isinstance(item.get("point_error_m"), (int, float)) and np.isfinite(float(item["point_error_m"]))
    ]
    return {
        "pose_error": pose_error,
        "position": actor_p.tolist(),
        "quaternion": actor_q.tolist(),
        "finite": finite,
        "workspace_radius_m": workspace_radius,
        "connection_point_error": connection_error,
        "min_connection_point_error_m": float(min(point_errors)) if point_errors else float("inf"),
    }


def _actor_state_is_plausible(report: dict[str, Any], args: argparse.Namespace) -> bool:
    if not bool(report.get("finite")):
        return False
    if float(report.get("workspace_radius_m", float("inf"))) > float(getattr(args, "max_actor_workspace_radius", 1.5)):
        return False
    pose_error = report.get("pose_error", {})
    if float(pose_error.get("position_error_m", float("inf"))) > float(getattr(args, "max_recoverable_release_position_error", 0.18)):
        return False
    if float(pose_error.get("orientation_error_deg", float("inf"))) > float(getattr(args, "max_recoverable_release_orientation_error_deg", 90.0)):
        return False
    min_point_error = float(report.get("min_connection_point_error_m", float("inf")))
    if np.isfinite(min_point_error) and min_point_error > float(getattr(args, "max_recoverable_connection_point_error", 0.18)):
        return False
    return True


def _post_success_lift_score(option: dict[str, Any]) -> tuple[int, float, int]:
    label = str(option.get("label", ""))
    joint_delta = float(option.get("joint_delta") or 0.0)
    if "tcp_back" in label:
        priority = 0
    elif label.startswith("world_z"):
        priority = 1
    elif label.startswith("world_"):
        priority = 2
    elif "tcp_open_axis" in label:
        priority = 3
    elif "triangle_normal" in label:
        priority = 4
    else:
        priority = 5
    return (priority, -joint_delta, int(option.get("index", 0)))


def _add_post_success_safe_lift(
    *,
    env: Any,
    planner: RM75CuRoboPlanner,
    base_env: Any,
    locked: dict[str, LockedPanelPose],
    fixtures: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    segments: list[dict[str, Any]],
    role: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    steps = int(getattr(args, "post_success_safe_lift_steps", 0))
    if steps <= 0:
        return {"enabled": False}
    current_q = _current_q(base_env)
    height = float(getattr(args, "post_success_safe_lift_height", 0.06))
    lateral = min(max(height * 0.7, 0.03), 0.06)
    tcp_pose = _tcp_pose(base_env)
    actor = locked[role].actor if role in locked else None
    actor_pose = _actor_pose(actor) if actor is not None else sapien.Pose()
    target_poses: list[tuple[str, sapien.Pose]] = [
        ("world_z", _offset_world(tcp_pose, np.asarray([0.0, 0.0, height], dtype=np.float32))),
        ("world_y_pos", _offset_world(tcp_pose, np.asarray([0.0, lateral, height * 0.5], dtype=np.float32))),
        ("world_y_neg", _offset_world(tcp_pose, np.asarray([0.0, -lateral, height * 0.5], dtype=np.float32))),
        ("world_x_pos", _offset_world(tcp_pose, np.asarray([lateral, 0.0, height * 0.5], dtype=np.float32))),
        ("world_x_neg", _offset_world(tcp_pose, np.asarray([-lateral, 0.0, height * 0.5], dtype=np.float32))),
        ("world_z_high", _offset_world(tcp_pose, np.asarray([0.0, 0.0, height * 1.5], dtype=np.float32))),
    ]
    for label, pose in _release_retreat_options(
        tcp_pose=tcp_pose,
        actor_pose=actor_pose,
        primary_mode=str(getattr(args, "retreat_mode", "tcp_back")),
        primary_distance=max(float(getattr(args, "retreat_distance", 0.045)), 0.025),
    ):
        if label != "none":
            target_poses.append((f"local_{label}", pose))
    max_joint_delta = float(getattr(args, "post_success_safe_lift_max_joint_delta", 1.6))
    options: list[dict[str, Any]] = []
    seen_targets: set[tuple[float, float, float, float, float, float, float]] = set()
    for index, (label, target_pose) in enumerate(target_poses):
        target_p, target_q = _pose_arrays(target_pose)
        target_key = tuple(np.round(np.concatenate([target_p, target_q]), 4).astype(float).tolist())
        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)
        ok, q_lift, ik_report = _solve_triangle_ik(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            target_pose=target_pose,
            start_q=current_q,
            ik_seeds=int(getattr(args, "ik_seeds", 64)),
            exclude_roles=set(),
        )
        joint_delta = None if q_lift is None else float(_joint_distance(q_lift, current_q))
        full_world_valid = False
        full_world_status = None
        if q_lift is not None:
            _set_planner_world(planner, base_env, locked, fixtures, exclude_roles=set())
            full_world_valid, full_world_status = planner.check_start_state(q_lift)
        success = bool(
            ok
            and q_lift is not None
            and full_world_valid
            and (max_joint_delta <= 0.0 or float(joint_delta or 0.0) <= max_joint_delta)
        )
        options.append(
            {
                "index": index,
                "label": label,
                "target_pose": _pose_to_report(target_pose),
                "success": success,
                "joint_delta": joint_delta,
                "max_joint_delta": max_joint_delta,
                "full_world_start_valid": bool(full_world_valid),
                "full_world_start_status": full_world_status,
                "ik": ik_report,
                "q": q_lift,
            }
        )
    _set_planner_world(planner, base_env, locked, fixtures, exclude_roles=set())
    current_full_world_valid, current_full_world_status = planner.check_start_state(current_q)
    current_clearance = {
        "index": len(options),
        "label": "current_clearance",
        "target_pose": _pose_to_report(tcp_pose),
        "success": bool(current_full_world_valid),
        "joint_delta": 0.0,
        "max_joint_delta": max_joint_delta,
        "full_world_start_valid": bool(current_full_world_valid),
        "full_world_start_status": current_full_world_status,
        "ik": {"skipped": True, "reason": "current collision-free retreat pose"},
        "q": current_q,
    }
    options.append(current_clearance)
    selected = min(
        [option for option in options if option["success"] and option["label"] != "current_clearance"],
        key=_post_success_lift_score,
        default=None,
    )
    if selected is None and current_clearance["success"]:
        selected = current_clearance
    report = {
        "enabled": True,
        "options": [{key: value for key, value in option.items() if key != "q"} for option in options],
        "selected": None if selected is None else {key: value for key, value in selected.items() if key != "q"},
    }
    if (
        selected is not None
        and selected.get("q") is not None
        and selected.get("label") != "current_clearance"
    ):
        _add_adaptive_joint_segment(
            env=env,
            arrays=arrays,
            segments=segments,
            name=f"{role}_post_success_safe_lift",
            goal_q=selected["q"],
            gripper=float(getattr(args, "release_open_gripper_value", 0.0)),
            base_steps=steps,
            action_repeat=int(getattr(args, "action_repeat", 1)),
            final_hold=int(getattr(args, "final_hold_steps", 0)),
            args=args,
        )
        settle_steps = int(getattr(args, "post_success_safe_lift_settle_steps", 0))
        if settle_steps > 0:
            _add_hold_segment(
                env,
                segments,
                f"{role}_post_success_safe_lift_settle",
                float(getattr(args, "release_open_gripper_value", 0.0)),
                settle_steps,
            )
    return report


def _magnetic_capture_status(
    base_env: Any,
    locked: dict[str, LockedPanelPose],
    role: str,
    args: argparse.Namespace,
    *,
    require_active: bool = True,
) -> dict[str, Any]:
    alignment = _triangle_base_edge_alignment(base_env, locked, role)
    active = _role_active_connection_report(base_env, role)
    max_edge_error = float(alignment.get("max_point_error_m", float("inf")))
    center_error = float(alignment.get("center_error_m", float("inf")))
    edge_angle_deg = float(alignment.get("edge_parallel_error_deg", float("inf")))
    success = bool(
        alignment.get("success")
        and (active.get("satisfied") or not require_active)
        and max_edge_error <= float(getattr(args, "magnetic_capture_max_edge_error", 0.010))
        and center_error <= float(getattr(args, "magnetic_capture_max_center_error", 0.010))
        and edge_angle_deg <= float(getattr(args, "magnetic_capture_max_edge_angle_deg", 10.0))
    )
    return {
        "success": success,
        "alignment": alignment,
        "active_connection": active,
    }


def _activate_close_role_connection(
    base_env: Any,
    locked: dict[str, LockedPanelPose],
    role: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    snap = getattr(base_env, "magnetic_snap", None)
    scene = getattr(snap, "scene", None) if snap is not None else None
    if snap is None or scene is None:
        return {"attempted": False, "reason": "magnetic_snap_unavailable"}
    alignment = _triangle_base_edge_alignment(base_env, locked, role)
    if not alignment.get("success"):
        return {"attempted": False, "reason": "alignment_unavailable", "alignment": alignment}
    max_edge_error = float(alignment.get("max_point_error_m", float("inf")))
    center_error = float(alignment.get("center_error_m", float("inf")))
    edge_angle_deg = float(alignment.get("edge_parallel_error_deg", float("inf")))
    attach_distance = float(getattr(snap, "attach_distance", 0.010))
    if (
        max_edge_error > attach_distance
        or center_error > float(getattr(args, "magnetic_capture_max_center_error", 0.010))
        or edge_angle_deg > float(getattr(args, "magnetic_capture_max_edge_angle_deg", 10.0))
    ):
        return {
            "attempted": False,
            "reason": "alignment_outside_attach_threshold",
            "attach_distance_m": attach_distance,
            "alignment": alignment,
        }
    candidates = []
    for connection in getattr(snap, "connections", []):
        if connection.parent == role or connection.child == role:
            if snap._find_active_connection(connection) is not None:
                return {
                    "attempted": False,
                    "reason": "connection_already_active",
                    "connection": connection.__dict__,
                    "alignment": alignment,
                }
            try:
                point_error = float(snap._connection_point_error(connection))
            except Exception:
                point_error = float("inf")
            if point_error <= attach_distance:
                candidates.append((point_error, connection))
    if not candidates:
        return {
            "attempted": False,
            "reason": "no_close_defined_connection",
            "attach_distance_m": attach_distance,
            "alignment": alignment,
        }
    candidates.sort(key=lambda item: item[0])
    point_error, connection = candidates[0]
    parent = locked.get(connection.parent)
    child = locked.get(connection.child)
    if parent is None or child is None:
        return {
            "attempted": False,
            "reason": "locked_pose_missing",
            "connection": connection.__dict__,
            "alignment": alignment,
        }
    snap._create_runtime_edge_connection(scene, parent, child, connection)
    return {
        "attempted": True,
        "activated": snap._find_active_connection(connection) is not None,
        "connection": connection.__dict__,
        "point_error_m": point_error,
        "alignment": alignment,
    }


def _run_pre_open_magnetic_capture(
    *,
    env: Any,
    base_env: Any,
    planner: RM75CuRoboPlanner,
    locked: dict[str, LockedPanelPose],
    fixtures: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    segments: list[dict[str, Any]],
    role: str,
    args: argparse.Namespace,
    gripper: float,
    require_active: bool,
    name_prefix: str,
    log: Any,
) -> dict[str, Any]:
    actor = locked[role].actor
    attempts = max(int(getattr(args, "magnetic_capture_nudge_attempts", 0)), 0)
    hold_steps = max(int(getattr(args, "magnetic_capture_hold_steps", 0)), 0)
    max_step = float(getattr(args, "magnetic_capture_nudge_step", 0.0))
    max_angle_step = float(np.deg2rad(float(getattr(args, "magnetic_capture_max_angle_step_deg", 0.0))))
    max_joint_delta = float(getattr(args, "magnetic_capture_max_joint_delta", 0.0))
    max_downward_step = float(getattr(args, "magnetic_capture_max_downward_step", float("inf")))
    revert_if_worse = bool(getattr(args, "magnetic_capture_revert_if_worse", True))
    revert_tolerance = float(getattr(args, "magnetic_capture_revert_tolerance", 0.0007))
    report: dict[str, Any] = {
        "attempts_requested": attempts,
        "hold_steps": hold_steps,
        "nudge_step_m": max_step,
        "max_angle_step_deg": float(getattr(args, "magnetic_capture_max_angle_step_deg", 0.0)),
        "require_active": bool(require_active),
        "revert_if_worse": bool(revert_if_worse),
        "revert_tolerance": float(revert_tolerance),
        "steps": [],
    }
    direct_activation = _activate_close_role_connection(base_env, locked, role, args)
    report["direct_activation"] = direct_activation
    direct_status = _magnetic_capture_status(base_env, locked, role, args, require_active=require_active)
    direct_status["attempt"] = 0
    if direct_status["success"]:
        report["steps"].append(direct_status)
        report["success"] = True
        report["final_status"] = direct_status
        return report
    if hold_steps > 0:
        _add_hold_segment(env, segments, f"{role}_{name_prefix}_initial_hold", gripper, hold_steps)
    for attempt_index in range(attempts + 1):
        status = _magnetic_capture_status(base_env, locked, role, args, require_active=require_active)
        status["attempt"] = attempt_index
        if status["success"] or attempt_index >= attempts:
            report["steps"].append(status)
            report["success"] = bool(status["success"])
            report["final_status"] = status
            break
        alignment = status["alignment"]
        before_score = _edge_alignment_score(alignment)
        center_delta = np.asarray(alignment.get("center_delta_child_minus_parent_m", [0.0, 0.0, 0.0]), dtype=np.float32)
        correction_delta = _clamp_vector(-center_delta, max_step)
        if np.isfinite(max_downward_step):
            correction_delta[2] = max(correction_delta[2], -max(max_downward_step, 0.0))
        if not require_active:
            parent_dir = np.asarray(alignment.get("parent_edge_dir_world", [1.0, 0.0, 0.0]), dtype=np.float32)
            parent_dir = _normalize(parent_dir)
            vertical_dir = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
            correction_delta = (
                parent_dir * float(np.dot(correction_delta, parent_dir))
                + vertical_dir * float(np.dot(correction_delta, vertical_dir))
            ).astype(np.float32)
            correction_delta = _clamp_vector(correction_delta, max_step)
        actor_pose = _actor_pose(actor)
        actor_p, actor_q = _pose_arrays(actor_pose)
        actor_rot = quat2mat(actor_q).astype(np.float32)
        parent_dir = np.asarray(alignment.get("parent_edge_dir_world", [1.0, 0.0, 0.0]), dtype=np.float32)
        child_dir = np.asarray(alignment.get("child_edge_dir_world", [1.0, 0.0, 0.0]), dtype=np.float32)
        rotation_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        rotation_angle = _signed_angle_around_axis(child_dir, parent_dir, rotation_axis)
        if max_angle_step <= 0.0:
            rotation_angle = 0.0
        else:
            rotation_angle = float(np.clip(rotation_angle, -max_angle_step, max_angle_step))
        if float(np.linalg.norm(rotation_axis)) > 1e-6 and abs(rotation_angle) > 1e-6:
            target_rot = _axis_angle_matrix(rotation_axis, rotation_angle) @ actor_rot
            target_q = mat2quat(target_rot).astype(np.float32)
        else:
            target_q = actor_q
        target_actor_pose = sapien.Pose(p=(actor_p + correction_delta).astype(np.float32).tolist(), q=target_q.tolist())
        live_actor_to_tcp = actor_pose.inv() * _tcp_pose(base_env)
        desired_tcp = target_actor_pose * live_actor_to_tcp
        ok, q_nudge, ik_report = _solve_triangle_ik(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            target_pose=desired_tcp,
            start_q=_current_q(base_env),
            ik_seeds=args.ik_seeds,
            exclude_roles={role},
        )
        q_before_nudge = _current_q(base_env)
        joint_delta = None if q_nudge is None else float(_joint_distance(q_nudge, _current_q(base_env)))
        joint_delta_ok = bool(joint_delta is not None and (max_joint_delta <= 0.0 or joint_delta <= max_joint_delta))
        step_report = {
            **status,
            "correction_delta_m": correction_delta.tolist(),
            "rotation_angle_deg": float(np.rad2deg(rotation_angle)),
            "ik": ik_report,
            "joint_delta_from_current": joint_delta,
            "max_joint_delta": max_joint_delta,
            "executed": bool(ok and q_nudge is not None and joint_delta_ok),
        }
        report["steps"].append(step_report)
        if not step_report["executed"]:
            report["success"] = False
            report["final_status"] = _magnetic_capture_status(base_env, locked, role, args, require_active=require_active)
            break
        _add_adaptive_joint_segment(
            env=env,
            arrays=arrays,
            segments=segments,
            name=f"{role}_{name_prefix}_nudge_{attempt_index + 1}",
            goal_q=q_nudge,
            gripper=gripper,
            base_steps=int(getattr(args, "magnetic_capture_nudge_steps", 10)),
            action_repeat=args.action_repeat,
            final_hold=args.final_hold_steps,
            args=args,
        )
        if hold_steps > 0:
            _add_hold_segment(env, segments, f"{role}_{name_prefix}_hold_{attempt_index + 1}", gripper, hold_steps)
        after_status = _magnetic_capture_status(base_env, locked, role, args, require_active=require_active)
        after_score = _edge_alignment_score(after_status["alignment"])
        step_report["after_nudge_status"] = after_status
        step_report["before_score"] = float(before_score)
        step_report["after_score"] = float(after_score)
        if revert_if_worse and after_score > before_score + revert_tolerance:
            step_report["reverted"] = True
            step_report["revert_reason"] = "edge_alignment_worsened"
            _add_adaptive_joint_segment(
                env=env,
                arrays=arrays,
                segments=segments,
                name=f"{role}_{name_prefix}_revert_{attempt_index + 1}",
                goal_q=q_before_nudge,
                gripper=gripper,
                base_steps=int(getattr(args, "magnetic_capture_nudge_steps", 10)),
                action_repeat=args.action_repeat,
                final_hold=args.final_hold_steps,
                args=args,
            )
            if hold_steps > 0:
                _add_hold_segment(env, segments, f"{role}_{name_prefix}_revert_hold_{attempt_index + 1}", gripper, hold_steps)
            report["success"] = False
            report["final_status"] = _magnetic_capture_status(base_env, locked, role, args, require_active=require_active)
            break
        log(
            f"{role}: {name_prefix} nudge {attempt_index + 1} "
            f"delta={correction_delta.tolist()} rot_deg={float(np.rad2deg(rotation_angle)):.2f}"
        )
    return report


def _build_triangle_role(
    *,
    env: Any,
    planner: RM75CuRoboPlanner,
    locked: dict[str, LockedPanelPose],
    fixtures: list[dict[str, Any]],
    targets: dict[str, sapien.Pose],
    arrays: dict[str, np.ndarray],
    segments: list[dict[str, Any]],
    role: str,
    args: argparse.Namespace,
    log: Any,
) -> dict[str, Any]:
    base_env = env.unwrapped
    _refresh_triangle_target_for_role(
        base_env,
        locked,
        targets,
        role,
        target_lift=float(getattr(args, "target_lift", 0.0)),
        log=log,
    )
    actor = locked[role].actor
    parent_edge_report = _triangle_parent_edge_report(base_env, role)
    report: dict[str, Any] = {
        "role": role,
        "candidate_reports": [],
        "target_pose": _pose_to_report(targets[role]),
        "parent_edge_measurement": parent_edge_report,
        "release_gap_mms": str(getattr(args, "release_gap_mms", "") or ""),
    }
    min_parent_edge_z = float(getattr(args, "min_parent_top_edge_z", 0.0))
    if min_parent_edge_z > 0.0 and float(parent_edge_report.get("center_z_m", 0.0)) < min_parent_edge_z:
        report["success"] = False
        report["failed_at"] = "parent_edge_height"
        report["failure_reasons"] = ["parent_top_edge_below_threshold"]
        return report
    _set_role_magnets_enabled(base_env, role, False)
    pre_start_report = _current_start_state_report(
        planner=planner,
        base_env=base_env,
        locked=locked,
        fixtures=fixtures,
        args=args,
    )
    report["pre_role_start_state"] = pre_start_report
    if not pre_start_report["valid"]:
        recovery_report = _recover_to_valid_start_state(
            env=env,
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            arrays=arrays,
            segments=segments,
            role=role,
            args=args,
            segment_prefix="pre_role_valid_start_recovery",
        )
        report["pre_role_start_state_recovery"] = recovery_report
        report["pre_role_start_state_after_recovery"] = recovery_report.get("after")
        if not recovery_report.get("success"):
            report["success"] = False
            report["failed_at"] = "pre_role_start_state_recovery"
            report["failure_reasons"] = ["invalid_start_state_before_candidate_screen"]
            return report
    start_q = _current_q(base_env)
    selected: dict[str, Any] | None = None
    feasible_bundles: list[dict[str, Any]] = []
    candidate_pool_size = max(int(getattr(args, "triangle_candidate_pool_size", 1)), 1)
    screen_passes = _grasp_screen_passes(args)
    report["screen_passes"] = [
        {
            "label": str(screen_pass["label"]),
            "max_grasp_candidates": int(screen_pass["max_grasp_candidates"]),
            "ik_seeds": int(screen_pass["ik_seeds"]),
        }
        for screen_pass in screen_passes
    ]
    for screen_pass in screen_passes:
        if feasible_bundles:
            break
        screen_label = str(screen_pass["label"])
        screen_ik_seeds = int(screen_pass["ik_seeds"])
        for index, candidate in enumerate(_triangle_grasp_candidates(screen_pass["max_grasp_candidates"])):
            grasp_tcp = _triangle_grasp_tcp(actor, candidate)
            actor_to_tcp = _actor_pose(actor).inv() * grasp_tcp
            pregrasp = _pregrasp_pose(grasp_tcp, candidate)
            log(f"{role}: [{screen_label}] screening triangle grasp {index} {candidate.label}")
            ok, q_path, pre_report = _plan_triangle_motion(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                target_pose=pregrasp,
                start_q=start_q,
                timeout=args.pregrasp_timeout,
                ik_seeds=screen_ik_seeds,
                exclude_roles=set(),
                args=args,
            )
            candidate_report: dict[str, Any] = {
                "index": index,
                "candidate": candidate.__dict__,
                "screen_pass": screen_label,
                "screen_ik_seeds": int(screen_ik_seeds),
                "pregrasp_plan": pre_report,
            }
            if not ok or q_path is None:
                candidate_report["failed_at"] = "pregrasp_plan"
                report["candidate_reports"].append(candidate_report)
                continue
            ok, q_grasp, grasp_report = _solve_triangle_ik(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                target_pose=grasp_tcp,
                start_q=q_path[-1, :7],
                ik_seeds=screen_ik_seeds,
                exclude_roles={role},
                args=args,
            )
            candidate_report["grasp_ik"] = grasp_report
            if not ok or q_grasp is None:
                candidate_report["failed_at"] = "grasp_ik"
                report["candidate_reports"].append(candidate_report)
                continue
            grasp_joint_delta = _joint_distance(q_grasp, q_path[-1, :7])
            candidate_report["grasp_joint_delta_from_pregrasp"] = float(grasp_joint_delta)
            max_grasp_joint_delta = float(getattr(args, "grasp_max_joint_delta", 0.0))
            if max_grasp_joint_delta > 0.0 and grasp_joint_delta > max_grasp_joint_delta:
                candidate_report["failed_at"] = "grasp_joint_branch_jump"
                candidate_report["grasp_max_joint_delta"] = max_grasp_joint_delta
                report["candidate_reports"].append(candidate_report)
                continue
            lift_pose = sapien.Pose(p=[grasp_tcp.p[0], grasp_tcp.p[1], grasp_tcp.p[2] + args.lift_height], q=grasp_tcp.q)
            ok, q_lift, lift_report = _solve_triangle_ik(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                target_pose=lift_pose,
                start_q=q_grasp,
                ik_seeds=screen_ik_seeds,
                exclude_roles={role},
                args=args,
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
            release_options = []
            for release_index, (label, release_actor_pose) in enumerate(_release_candidates(role, targets[role], args)):
                if not _release_index_allowed(args, release_index):
                    continue
                for preplace_index, preplace_height in enumerate(_preplace_heights(args)):
                    option = _solve_triangle_release_option(
                        planner=planner,
                        base_env=base_env,
                        locked=locked,
                        fixtures=fixtures,
                        role=role,
                        release_index=release_index,
                        label=label,
                        release_actor_pose=release_actor_pose,
                        actor_to_tcp=actor_to_tcp,
                        preplace_index=preplace_index,
                        preplace_height=float(preplace_height),
                        start_q=q_lift,
                        ik_seeds=screen_ik_seeds,
                        args=args,
                    )
                    option["pose_error"] = _pose_error_proxy(release_actor_pose, targets[role])
                    release_options.append(option)
            candidate_report["release_options"] = [
                {k: v for k, v in option.items() if k not in {"q_stage", "q_preplace", "q_release", "release_actor_pose"}}
                for option in release_options
            ]
            candidate_report["release_screening_stats"] = {
                "option_count": int(len(release_options)),
                "success_count": int(sum(1 for option in release_options if option.get("success"))),
                "stage_ik_time_sum": float(
                    sum(
                        float((option.get("stage_ik") or {}).get("ik_time", 0.0) or 0.0)
                        for option in release_options
                        if isinstance(option.get("stage_ik"), dict)
                    )
                ),
                "preplace_ik_time_sum": float(
                    sum(
                        float((option.get("preplace_ik") or {}).get("ik_time", 0.0) or 0.0)
                        for option in release_options
                        if isinstance(option.get("preplace_ik"), dict)
                    )
                ),
                "release_ik_time_sum": float(
                    sum(
                        float((option.get("release_ik") or {}).get("ik_time", 0.0) or 0.0)
                        for option in release_options
                        if isinstance(option.get("release_ik"), dict)
                    )
                ),
                "release_candidate_index_filter": int(getattr(args, "release_candidate_index", -1)),
                "allowed_release_candidate_indices": None
                if _allowed_release_candidate_indices(args) is None
                else sorted(int(item) for item in _allowed_release_candidate_indices(args)),
                "preplace_heights": _preplace_heights(args),
            }
            feasible_release_options = [option for option in release_options if option["success"]]
            release_selected = min(
                feasible_release_options,
                key=lambda option: _triangle_release_option_score(option, targets[role]),
                default=None,
            )
            if release_selected is None:
                candidate_report["failed_at"] = "release_ik"
                report["candidate_reports"].append(candidate_report)
                continue
            pregrasp_joint_distance = float(_joint_distance(q_path[-1, :7], start_q))
            bundle = {
                "candidate_index": index,
                "candidate": candidate,
                "screen_pass": screen_label,
                "screen_ik_seeds": int(screen_ik_seeds),
                "q_path": q_path,
                "q_grasp": q_grasp,
                "q_lift": q_lift,
                "actor_to_tcp": actor_to_tcp,
                "release": release_selected,
                "pregrasp_joint_distance": pregrasp_joint_distance,
                "grasp_joint_delta_from_pregrasp": float(grasp_joint_delta),
                "lift_joint_delta_from_grasp": float(lift_joint_delta),
            }
            bundle_score = _triangle_bundle_score(bundle, targets[role])
            candidate_report["success"] = True
            candidate_report["bundle_score"] = list(bundle_score)
            report["candidate_reports"].append(candidate_report)
            feasible_bundles.append(bundle)
            if len(feasible_bundles) >= candidate_pool_size:
                break
        if len(feasible_bundles) >= candidate_pool_size:
            break
    if feasible_bundles:
        selected = min(feasible_bundles, key=lambda bundle: _triangle_bundle_score(bundle, targets[role]))
        report["candidate_pool_size_requested"] = int(candidate_pool_size)
        report["candidate_pool_size_collected"] = int(len(feasible_bundles))
        report["selected_bundle_score"] = list(_triangle_bundle_score(selected, targets[role]))
    if selected is None:
        report["success"] = False
        report["failed_at"] = "candidate_screen"
        return report

    report["candidate_index"] = int(selected["candidate_index"])
    report["candidate"] = selected["candidate"].__dict__
    report["release_option"] = {
        k: v
        for k, v in selected["release"].items()
        if k not in {"q_stage", "q_preplace", "q_release", "release_actor_pose"}
    }
    report["release_strategy"] = {
        "release_candidate_index": int(getattr(args, "release_candidate_index", -1)),
        "release_open_gripper_value": float(getattr(args, "release_open_gripper_value", 0.0)),
        "open_steps": int(getattr(args, "open_steps", 0)),
        "pre_open_hold_steps": int(getattr(args, "pre_open_hold_steps", 0)),
        "post_open_hold_steps": int(getattr(args, "post_open_hold_steps", 0)),
        "retreat_mode": str(getattr(args, "retreat_mode", "")),
        "retreat_distance": float(getattr(args, "retreat_distance", 0.0)),
        "retreat_max_joint_delta": float(getattr(args, "retreat_max_joint_delta", 0.0)),
        "edge_seating_attempts": int(getattr(args, "edge_seating_attempts", 0)),
        "release_correction_attempts": int(getattr(args, "release_correction_attempts", 0)),
    }
    _record_existing_joint_path(
        env=env,
        arrays=arrays,
        segments=segments,
        name=f"{role}_pregrasp",
        path=selected["q_path"],
        gripper=OPEN_GRIPPER,
        final_hold=args.final_hold_steps,
        max_joint_step=float(getattr(args, "max_joint_step", 0.06)),
    )
    _add_adaptive_joint_segment(
        env=env,
        arrays=arrays,
        segments=segments,
        name=f"{role}_grasp",
        goal_q=selected["q_grasp"],
        gripper=OPEN_GRIPPER,
        base_steps=args.short_steps,
        action_repeat=args.action_repeat,
        final_hold=args.final_hold_steps,
        args=args,
    )
    _add_hold_segment(env, segments, f"{role}_close_gripper", CLOSED_GRIPPER, args.close_steps)
    report["grasp_after_close"] = _grasp_report(base_env, actor)
    log(f"{role}: grasp after close {report['grasp_after_close']}")
    if not report["grasp_after_close"]["is_grasped"]:
        report["success"] = False
        report["failed_at"] = "grasp_check"
        return report
    _add_adaptive_joint_segment(
        env=env,
        arrays=arrays,
        segments=segments,
        name=f"{role}_lift",
        goal_q=selected["q_lift"],
        gripper=CLOSED_GRIPPER,
        base_steps=args.move_steps,
        action_repeat=args.action_repeat,
        final_hold=args.final_hold_steps,
        args=args,
    )
    report["grasp_after_lift"] = _grasp_report(base_env, actor)
    log(f"{role}: grasp after lift {report['grasp_after_lift']}")
    if not report["grasp_after_lift"]["is_grasped"]:
        report["success"] = False
        report["failed_at"] = "grasp_lost_after_lift"
        return report
    _refresh_triangle_target_for_role(
        base_env,
        locked,
        targets,
        role,
        target_lift=float(getattr(args, "target_lift", 0.0)),
        log=log,
    )
    report["target_pose_after_lift_refresh"] = _pose_to_report(targets[role])
    report["parent_edge_measurement_after_lift_refresh"] = _triangle_parent_edge_report(base_env, role)
    live_release_options: list[dict[str, Any]] = []
    if bool(getattr(args, "use_screened_release_after_lift", True)):
        q_stage = selected["release"].get("q_stage")
        q_pre = selected["release"]["q_preplace"]
        q_rel = selected["release"]["q_release"]
        stage_ik = selected["release"].get("stage_ik", {"reused_from_candidate_screen": True})
        pre_ik = {"reused_from_candidate_screen": True}
        rel_ik = {"reused_from_candidate_screen": True}
        stage_used = bool(selected["release"].get("staging_used", False))
        ok_pre = q_pre is not None
        ok_rel = q_rel is not None
        if ok_pre and ok_rel:
            release_pose = selected["release"]["release_actor_pose"]
            live_release_options.append(
                {
                    "index": int(selected["release"]["index"]),
                    "label": str(selected["release"]["label"]),
                    "staging_enabled": bool(selected["release"].get("staging_enabled", False)),
                    "staging_used": stage_used,
                    "staging_direct_fallback_allowed": bool(selected["release"].get("staging_direct_fallback_allowed", True)),
                    "stage_pose": selected["release"].get("stage_pose"),
                    "stage_ik": stage_ik,
                    "preplace_ik": pre_ik,
                    "release_ik": rel_ik,
                    "stage_joint_delta_from_start": selected["release"].get("stage_joint_delta_from_start"),
                    "preplace_joint_delta_from_current": _joint_distance(q_pre, _current_q(base_env)),
                    "release_joint_delta_from_preplace": _joint_distance(q_rel, q_pre),
                    "pose_error": _pose_error_proxy(release_pose, targets[role]),
                    "success": True,
                    "failed_at": None,
                    "q_stage": q_stage,
                    "q_preplace": q_pre,
                    "q_release": q_rel,
                    "release_actor_pose": release_pose,
                }
            )
    else:
        q_stage = None
        ok_pre, q_pre, pre_ik = (False, None, {"skipped": True, "reason": "force live release search"})
        ok_rel, q_rel, rel_ik = (False, None, {"skipped": True, "reason": "force live release search"})
        stage_ik = {"skipped": True, "reason": "force live release search"}
        stage_used = False
    live_preplace_delta = _joint_distance(q_pre, _current_q(base_env)) if q_pre is not None else None
    live_release_delta = _joint_distance(q_rel, q_pre) if q_pre is not None and q_rel is not None else None
    max_release_preplace_delta = float(getattr(args, "release_preplace_max_joint_delta", 0.0))
    max_release_delta = float(getattr(args, "release_max_joint_delta", 0.0))
    live_joint_delta_ok = True
    if live_preplace_delta is not None and max_release_preplace_delta > 0.0 and live_preplace_delta > max_release_preplace_delta:
        live_joint_delta_ok = False
    if live_release_delta is not None and max_release_delta > 0.0 and live_release_delta > max_release_delta:
        live_joint_delta_ok = False
    if not (ok_pre and q_pre is not None and ok_rel and q_rel is not None and live_joint_delta_ok):
        live_actor_to_tcp = _actor_pose(actor).inv() * _tcp_pose(base_env)
        current_q = _current_q(base_env)
        live_release_options = []
        for release_index, (label, release_actor_pose) in enumerate(_release_candidates(role, targets[role], args)):
            if not _release_index_allowed(args, release_index):
                continue
            for preplace_index, preplace_height in enumerate(_preplace_heights(args)):
                option = _solve_triangle_release_option(
                    planner=planner,
                    base_env=base_env,
                    locked=locked,
                    fixtures=fixtures,
                    role=role,
                    release_index=release_index,
                    label=label,
                    release_actor_pose=release_actor_pose,
                    actor_to_tcp=live_actor_to_tcp,
                    preplace_index=preplace_index,
                    preplace_height=float(preplace_height),
                    start_q=current_q,
                    ik_seeds=args.ik_seeds,
                    args=args,
                )
                option["pose_error"] = _pose_error_proxy(release_actor_pose, targets[role])
                option["preplace_joint_delta_from_current"] = option.pop("preplace_joint_delta_from_lift", None)
                live_release_options.append(option)
        live_selected = min(
            [option for option in live_release_options if option["success"]],
            key=lambda option: _triangle_release_option_score(option, targets[role]),
            default=None,
        )
        if live_selected is not None:
            q_stage = live_selected.get("q_stage")
            q_pre = live_selected["q_preplace"]
            q_rel = live_selected["q_release"]
            stage_ik = live_selected.get("stage_ik", {"selected_from_live_release_search": True})
            pre_ik = live_selected["preplace_ik"]
            rel_ik = live_selected["release_ik"]
            stage_used = bool(live_selected.get("staging_used", False))
            ok_pre = True
            ok_rel = True
            live_preplace_delta = _joint_distance(q_pre, current_q) if q_pre is not None else None
            live_release_delta = _joint_distance(q_rel, q_pre) if q_pre is not None and q_rel is not None else None
            live_joint_delta_ok = True
        elif bool(getattr(args, "allow_screened_release_fallback", False)):
            q_stage = selected["release"].get("q_stage")
            screened_q_pre = selected["release"]["q_preplace"]
            screened_q_rel = selected["release"]["q_release"]
            stage_ik = selected["release"].get("stage_ik", {"fallback_reused_from_candidate_screen": True})
            stage_used = bool(selected["release"].get("staging_used", False))
            screened_pre_delta = _joint_distance(screened_q_pre, current_q) if screened_q_pre is not None else None
            screened_rel_delta = (
                _joint_distance(screened_q_rel, screened_q_pre)
                if screened_q_pre is not None and screened_q_rel is not None
                else None
            )
            screened_delta_ok = bool(screened_q_pre is not None and screened_q_rel is not None)
            if screened_pre_delta is not None and max_release_preplace_delta > 0.0 and screened_pre_delta > max_release_preplace_delta:
                screened_delta_ok = False
            if screened_rel_delta is not None and max_release_delta > 0.0 and screened_rel_delta > max_release_delta:
                screened_delta_ok = False
            if screened_delta_ok:
                q_pre = screened_q_pre
                q_rel = screened_q_rel
                pre_ik = {"fallback_reused_from_candidate_screen": True}
                rel_ik = {"fallback_reused_from_candidate_screen": True}
                ok_pre = True
                ok_rel = True
                live_preplace_delta = screened_pre_delta
                live_release_delta = screened_rel_delta
                live_joint_delta_ok = True
    report["live_release_ik"] = {
        "stage": stage_ik,
        "preplace": pre_ik,
        "release": rel_ik,
        "staging_used": bool(stage_used),
        "stage_joint_delta_from_start": None
        if not live_release_options
        else min(
            [option for option in live_release_options if option.get("success")],
            key=lambda option: _triangle_release_option_score(option, targets[role]),
            default=live_release_options[0],
        ).get("stage_joint_delta_from_start"),
        "preplace_joint_delta_from_current": None if live_preplace_delta is None else float(live_preplace_delta),
        "release_joint_delta_from_preplace": None if live_release_delta is None else float(live_release_delta),
        "joint_delta_ok": bool(live_joint_delta_ok),
        "use_screened_release_after_lift": bool(getattr(args, "use_screened_release_after_lift", True)),
        "selected": None
        if not live_release_options
        else {
            k: v
            for k, v in min(
                [option for option in live_release_options if option["success"]],
                key=lambda option: _triangle_release_option_score(option, targets[role]),
                default=live_release_options[0],
            ).items()
            if k not in {"q_stage", "q_preplace", "q_release", "release_actor_pose"}
        },
        "live_release_option_count": len(live_release_options),
        "options": [
            {k: v for k, v in option.items() if k not in {"q_stage", "q_preplace", "q_release", "release_actor_pose"}}
            for option in live_release_options
        ],
    }
    if not (ok_pre and q_pre is not None and ok_rel and q_rel is not None and live_joint_delta_ok):
        report["success"] = False
        report["failed_at"] = "live_release_ik"
        return report
    if stage_used and q_stage is not None:
        _add_adaptive_joint_segment(
            env=env,
            arrays=arrays,
            segments=segments,
            name=f"{role}_preplace_stage",
            goal_q=q_stage,
            gripper=CLOSED_GRIPPER,
            base_steps=args.move_steps,
            action_repeat=args.action_repeat,
            final_hold=args.final_hold_steps,
            args=args,
        )
        report["grasp_after_preplace_stage"] = _grasp_report(base_env, actor)
        report["actor_state_after_preplace_stage"] = _actor_state_report(actor, targets[role], base_env, role)
        if not report["grasp_after_preplace_stage"]["is_grasped"]:
            report["success"] = False
            report["failed_at"] = "grasp_lost_after_preplace_stage"
            return report
        if not _actor_state_is_plausible(report["actor_state_after_preplace_stage"], args):
            report["success"] = False
            report["failed_at"] = "actor_state_invalid_after_preplace_stage"
            return report
    _add_adaptive_joint_segment(
        env=env,
        arrays=arrays,
        segments=segments,
        name=f"{role}_preplace",
        goal_q=q_pre,
        gripper=CLOSED_GRIPPER,
        base_steps=args.move_steps,
        action_repeat=args.action_repeat,
        final_hold=args.final_hold_steps,
        args=args,
    )
    report["grasp_after_preplace"] = _grasp_report(base_env, actor)
    report["actor_state_after_preplace"] = _actor_state_report(actor, targets[role], base_env, role)
    if not report["grasp_after_preplace"]["is_grasped"]:
        report["success"] = False
        report["failed_at"] = "grasp_lost_after_preplace"
        return report
    if not _actor_state_is_plausible(report["actor_state_after_preplace"], args):
        report["success"] = False
        report["failed_at"] = "actor_state_invalid_after_preplace"
        return report
    release_replan_selected: dict[str, Any] | None = None
    release_replan_options: list[dict[str, Any]] = []
    if bool(getattr(args, "replan_release_after_preplace", False)):
        _refresh_triangle_target_for_role(
            base_env,
            locked,
            targets,
            role,
            target_lift=float(getattr(args, "target_lift", 0.0)),
            log=log,
        )
        report["target_pose_after_preplace_refresh"] = _pose_to_report(targets[role])
        release_replan_selected, release_replan_options = _select_release_after_preplace(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            role=role,
            actor=actor,
            target_pose=targets[role],
            preplace_height=float(report["release_option"].get("preplace_height_m", 0.0)),
            start_q=_current_q(base_env),
            args=args,
        )
        report["release_replan_after_preplace"] = {
            "selected": None
            if release_replan_selected is None
            else {
                key: value
                for key, value in release_replan_selected.items()
                if key not in {"q_stage", "q_preplace", "q_release", "release_actor_pose"}
            },
            "option_count": int(len(release_replan_options)),
            "options": [
                {key: value for key, value in option.items() if key not in {"q_stage", "q_preplace", "q_release", "release_actor_pose"}}
                for option in release_replan_options
            ],
        }
        if release_replan_selected is not None and release_replan_selected.get("q_release") is not None:
            q_rel = release_replan_selected["q_release"]
            rel_ik = release_replan_selected["release_ik"]
            live_release_delta = release_replan_selected.get("release_joint_delta_from_preplace")
            live_joint_delta_ok = True
            report["release_option"] = {
                key: value
                for key, value in release_replan_selected.items()
                if key not in {"q_stage", "q_preplace", "q_release", "release_actor_pose"}
            }
    _add_adaptive_joint_segment(
        env=env,
        arrays=arrays,
        segments=segments,
        name=f"{role}_release_pose",
        goal_q=q_rel,
        gripper=CLOSED_GRIPPER,
        base_steps=args.release_steps,
        action_repeat=args.action_repeat,
        final_hold=args.final_hold_steps,
        args=args,
    )
    report["pose_error_before_open"] = _pose_error(actor, targets[role])
    report["grasp_before_open"] = _grasp_report(base_env, actor)
    report["actor_state_before_open"] = _actor_state_report(actor, targets[role], base_env, role)
    if not report["grasp_before_open"]["is_grasped"]:
        report["success"] = False
        report["failed_at"] = "grasp_lost_before_open"
        return report
    if not _actor_state_is_plausible(report["actor_state_before_open"], args):
        report["success"] = False
        report["failed_at"] = "actor_state_invalid_before_open"
        return report
    correction_reports: list[dict[str, Any]] = []
    for correction_index in range(max(int(args.release_correction_attempts), 0)):
        current_error = _pose_error(actor, targets[role])
        current_connection_error = _role_connection_point_error(base_env, role)
        current_point_errors = [
            float(item["point_error_m"])
            for item in current_connection_error.get("connections", [])
            if isinstance(item.get("point_error_m"), (int, float))
        ]
        current_max_point_error = min(current_point_errors) if current_point_errors else float("inf")
        if (
            current_error["position_error_m"] <= float(args.release_correction_position_threshold)
            and current_error["orientation_error_deg"] <= float(args.release_correction_orientation_threshold_deg)
            and current_max_point_error <= float(args.release_correction_edge_threshold)
        ):
            break
        q_before_correction = _current_q(base_env)
        live_actor_to_tcp = _actor_pose(actor).inv() * _tcp_pose(base_env)
        desired_tcp = targets[role] * live_actor_to_tcp
        ok_corr, q_corr, corr_ik = _solve_triangle_ik(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            target_pose=desired_tcp,
            start_q=q_before_correction,
            ik_seeds=args.ik_seeds,
            exclude_roles={role},
        )
        correction_joint_delta = _joint_distance(q_corr, q_before_correction) if q_corr is not None else None
        max_correction_joint_delta = float(getattr(args, "release_correction_max_joint_delta", 0.0))
        correction_joint_delta_ok = (
            correction_joint_delta is None
            or max_correction_joint_delta <= 0.0
            or correction_joint_delta <= max_correction_joint_delta
        )
        correction_report: dict[str, Any] = {
            "attempt": correction_index + 1,
            "pose_error_before": current_error,
            "connection_point_error_before": current_connection_error,
            "ik": corr_ik,
            "joint_delta_from_current": None if correction_joint_delta is None else float(correction_joint_delta),
            "max_joint_delta": max_correction_joint_delta,
            "joint_delta_ok": bool(correction_joint_delta_ok),
            "success": bool(ok_corr and q_corr is not None and correction_joint_delta_ok),
        }
        if ok_corr and q_corr is not None and correction_joint_delta_ok:
            _add_adaptive_joint_segment(
                env=env,
                arrays=arrays,
                segments=segments,
                name=f"{role}_release_correction_{correction_index + 1}",
                goal_q=q_corr,
                gripper=CLOSED_GRIPPER,
                base_steps=args.release_correction_steps,
                action_repeat=args.action_repeat,
                final_hold=args.final_hold_steps,
                args=args,
            )
        correction_report["pose_error_after"] = _pose_error(actor, targets[role])
        correction_report["connection_point_error_after"] = _role_connection_point_error(base_env, role)
        score_before = _release_correction_score(current_error, current_connection_error)
        score_after = _release_correction_score(
            correction_report["pose_error_after"],
            correction_report["connection_point_error_after"],
        )
        correction_report["score_before"] = float(score_before)
        correction_report["score_after"] = float(score_after)
        if (
            bool(getattr(args, "release_correction_revert_if_worse", True))
            and correction_report["success"]
            and score_after > score_before + float(getattr(args, "release_correction_revert_tolerance", 0.0005))
        ):
            _add_adaptive_joint_segment(
                env=env,
                arrays=arrays,
                segments=segments,
                name=f"{role}_release_correction_{correction_index + 1}_revert",
                goal_q=q_before_correction,
                gripper=CLOSED_GRIPPER,
                base_steps=args.release_correction_steps,
                action_repeat=args.action_repeat,
                final_hold=args.final_hold_steps,
                args=args,
            )
            correction_report["revert_if_worse"] = {
                "triggered": True,
                "score_before": float(score_before),
                "score_after": float(score_after),
                "pose_error_after_revert": _pose_error(actor, targets[role]),
                "connection_point_error_after_revert": _role_connection_point_error(base_env, role),
            }
        correction_reports.append(correction_report)
        log(
            f"{role}: release correction {correction_index + 1} "
            f"before={current_error} edge_before={current_max_point_error:.4f} "
            f"after={correction_report['pose_error_after']} "
            f"score_before={score_before:.4f} score_after={score_after:.4f}"
        )
        if not correction_report["success"]:
            break
        if correction_report.get("revert_if_worse", {}).get("triggered"):
            break
    report["release_corrections"] = correction_reports
    report["pose_error_before_open_after_corrections"] = _pose_error(actor, targets[role])
    edge_seating_reports: list[dict[str, Any]] = []
    for seating_index in range(max(int(getattr(args, "edge_seating_attempts", 0)), 0)):
        alignment_before = _triangle_base_edge_alignment(base_env, locked, role)
        max_edge_error = float(alignment_before.get("max_point_error_m", float("inf")))
        center_error = float(alignment_before.get("center_error_m", float("inf")))
        edge_angle_deg = float(alignment_before.get("edge_parallel_error_deg", float("inf")))
        if (
            max_edge_error <= float(getattr(args, "edge_seating_max_error", 0.0))
            and center_error <= float(getattr(args, "edge_seating_center_error", 0.0))
            and edge_angle_deg <= float(getattr(args, "edge_seating_angle_error_deg", 180.0))
        ):
            break
        center_delta = np.asarray(alignment_before.get("center_delta_child_minus_parent_m", [0.0, 0.0, 0.0]), dtype=np.float32)
        correction_delta = _clamp_vector(-center_delta, float(getattr(args, "edge_seating_max_step", 0.0)))
        actor_pose = _actor_pose(actor)
        actor_p, actor_q = _pose_arrays(actor_pose)
        actor_rot = quat2mat(actor_q).astype(np.float32)
        rotation_axis = np.asarray(alignment_before.get("edge_rotation_axis_world", [0.0, 0.0, 0.0]), dtype=np.float32)
        rotation_angle = float(alignment_before.get("edge_rotation_angle_rad", 0.0))
        rotation_mode = str(getattr(args, "edge_seating_rotation_mode", "auto") or "auto")
        if rotation_mode == "auto":
            parent_dir = np.asarray(alignment_before.get("parent_edge_dir_world", [1.0, 0.0, 0.0]), dtype=np.float32)
            child_dir = np.asarray(alignment_before.get("child_edge_dir_world", [1.0, 0.0, 0.0]), dtype=np.float32)
            if abs(float(parent_dir[2])) < 0.35 and abs(float(child_dir[2])) < 0.35:
                rotation_mode = "world_z"
            else:
                rotation_mode = "edge_cross"
        if rotation_mode == "world_z":
            parent_dir = np.asarray(alignment_before.get("parent_edge_dir_world", [1.0, 0.0, 0.0]), dtype=np.float32)
            child_dir = np.asarray(alignment_before.get("child_edge_dir_world", [1.0, 0.0, 0.0]), dtype=np.float32)
            rotation_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
            rotation_angle = _signed_angle_around_axis(child_dir, parent_dir, rotation_axis)
        elif rotation_mode != "edge_cross":
            rotation_mode = "edge_cross"
        max_rotation = float(np.deg2rad(float(getattr(args, "edge_seating_max_angle_step_deg", 0.0))))
        if max_rotation <= 0.0 or edge_angle_deg <= float(getattr(args, "edge_seating_angle_error_deg", 180.0)):
            rotation_angle = 0.0
        else:
            rotation_angle = float(np.clip(rotation_angle, -max_rotation, max_rotation))
            rotation_angle *= float(getattr(args, "edge_seating_rotation_sign", 1.0))
        if float(np.linalg.norm(rotation_axis)) > 1e-6 and abs(rotation_angle) > 1e-6:
            target_rot = _axis_angle_matrix(rotation_axis, rotation_angle) @ actor_rot
            target_q = mat2quat(target_rot).astype(np.float32)
        else:
            target_q = actor_q
        target_actor_pose = sapien.Pose(p=(actor_p + correction_delta).astype(np.float32).tolist(), q=target_q.tolist())
        live_actor_to_tcp = actor_pose.inv() * _tcp_pose(base_env)
        desired_tcp = target_actor_pose * live_actor_to_tcp
        q_before_seating = _current_q(base_env)
        ok_seat, q_seat, seat_ik = _solve_triangle_ik(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            target_pose=desired_tcp,
            start_q=_current_q(base_env),
            ik_seeds=args.ik_seeds,
            exclude_roles={role},
        )
        seating_report: dict[str, Any] = {
            "attempt": seating_index + 1,
            "alignment_before": alignment_before,
            "correction_delta_m": correction_delta.tolist(),
            "rotation_mode": rotation_mode,
            "rotation_angle_deg": float(np.rad2deg(rotation_angle)),
            "ik": seat_ik,
            "success": bool(ok_seat and q_seat is not None),
        }
        if ok_seat and q_seat is not None:
            _add_adaptive_joint_segment(
                env=env,
                arrays=arrays,
                segments=segments,
                name=f"{role}_edge_seating_{seating_index + 1}",
                goal_q=q_seat,
                gripper=CLOSED_GRIPPER,
                base_steps=args.edge_seating_steps,
                action_repeat=args.action_repeat,
                final_hold=args.final_hold_steps,
                args=args,
            )
        seating_report["alignment_after"] = _triangle_base_edge_alignment(base_env, locked, role)
        if (
            bool(getattr(args, "edge_seating_revert_if_worse", True))
            and seating_report["success"]
            and _edge_alignment_score(seating_report["alignment_after"])
            > _edge_alignment_score(alignment_before) + float(getattr(args, "edge_seating_revert_tolerance", 0.0005))
        ):
            seating_report["revert_if_worse"] = {
                "triggered": True,
                "score_before": _edge_alignment_score(alignment_before),
                "score_after": _edge_alignment_score(seating_report["alignment_after"]),
                "success": True,
            }
            _add_adaptive_joint_segment(
                env=env,
                arrays=arrays,
                segments=segments,
                name=f"{role}_edge_seating_{seating_index + 1}_revert",
                goal_q=q_before_seating,
                gripper=CLOSED_GRIPPER,
                base_steps=args.edge_seating_steps,
                action_repeat=args.action_repeat,
                final_hold=args.final_hold_steps,
                args=args,
            )
            seating_report["alignment_after_revert"] = _triangle_base_edge_alignment(base_env, locked, role)
        edge_seating_reports.append(seating_report)
        after = seating_report["alignment_after"]
        log(
            f"{role}: edge seating {seating_index + 1} "
            f"edge_before={max_edge_error:.4f} center_before={center_error:.4f} angle_before={edge_angle_deg:.2f} "
            f"delta={correction_delta.tolist()} "
            f"rot_mode={rotation_mode} "
            f"rot_deg={float(np.rad2deg(rotation_angle)):.2f} "
            f"edge_after={float(after.get('max_point_error_m', float('inf'))):.4f} "
            f"center_after={float(after.get('center_error_m', float('inf'))):.4f} "
            f"angle_after={float(after.get('edge_parallel_error_deg', float('inf'))):.2f}"
        )
        if not seating_report["success"]:
            break
        if seating_report.get("revert_if_worse", {}).get("triggered"):
            break
    report["edge_seating_corrections"] = edge_seating_reports
    report["edge_alignment_before_open"] = _triangle_base_edge_alignment(base_env, locked, role)
    report["connection_point_error_before_open"] = _role_connection_point_error(base_env, role)
    pre_open_edge_error = float(report["edge_alignment_before_open"].get("max_point_error_m", float("inf")))
    pre_open_center_error = float(report["edge_alignment_before_open"].get("center_error_m", float("inf")))
    pre_open_edge_angle = float(report["edge_alignment_before_open"].get("edge_parallel_error_deg", float("inf")))
    if (
        pre_open_edge_error > float(getattr(args, "pre_open_max_base_edge_error", 0.035))
        or pre_open_center_error > float(getattr(args, "pre_open_max_base_center_error", 0.030))
        or pre_open_edge_angle > float(getattr(args, "pre_open_max_base_edge_angle_deg", 25.0))
    ):
        report["success"] = False
        report["failed_at"] = "pre_open_alignment"
        report["failure_reasons"] = ["pre_open_base_alignment"]
        return report
    enable_after_initial_open = bool(getattr(args, "enable_magnets_after_initial_open", False))
    open_start_gripper = CLOSED_GRIPPER
    partial_open_used = False
    if enable_after_initial_open:
        _set_role_magnets_enabled(base_env, role, False)
        report["edge_alignment_before_magnet_enable"] = _triangle_base_edge_alignment(base_env, locked, role)
        report["connection_point_error_before_magnet_enable"] = _role_connection_point_error(base_env, role)
        if int(args.pre_open_hold_steps) > 0:
            _add_hold_segment(env, segments, f"{role}_pre_open_gap_hold_no_magnet", CLOSED_GRIPPER, args.pre_open_hold_steps)
        first_open_steps = min(max(int(getattr(args, "open_before_magnet_steps", 0)), 0), int(args.open_steps))
        open_fraction = min(max(float(getattr(args, "open_before_magnet_fraction", 0.0)), 0.0), 1.0)
        capture_gripper = float(CLOSED_GRIPPER) + (float(args.release_open_gripper_value) - float(CLOSED_GRIPPER)) * open_fraction
        if first_open_steps > 0 and open_fraction > 0.0:
            _add_ramp_segment(
                env,
                segments,
                f"{role}_initial_open_before_magnet",
                CLOSED_GRIPPER,
                capture_gripper,
                first_open_steps,
            )
        else:
            capture_gripper = CLOSED_GRIPPER
        pre_magnet_attempts = int(getattr(args, "pre_magnet_geometric_capture_attempts", 0))
        if pre_magnet_attempts > 0:
            pre_magnet_alignment = _triangle_base_edge_alignment(base_env, locked, role)
            pre_magnet_angle = float(pre_magnet_alignment.get("edge_parallel_error_deg", float("inf")))
            if pre_magnet_angle <= float(getattr(args, "pre_magnet_geometric_capture_max_angle_deg", 10.0)):
                original_attempts = int(getattr(args, "magnetic_capture_nudge_attempts", 0))
                original_hold_steps = int(getattr(args, "magnetic_capture_hold_steps", 0))
                args.magnetic_capture_nudge_attempts = pre_magnet_attempts
                args.magnetic_capture_hold_steps = 0
                pre_magnet_report = _run_pre_open_magnetic_capture(
                    env=env,
                    base_env=base_env,
                    planner=planner,
                    locked=locked,
                    fixtures=fixtures,
                    arrays=arrays,
                    segments=segments,
                    role=role,
                    args=args,
                    gripper=capture_gripper,
                    require_active=False,
                    name_prefix="pre_magnet_geometric_capture",
                    log=log,
                )
                args.magnetic_capture_nudge_attempts = original_attempts
                args.magnetic_capture_hold_steps = original_hold_steps
                report["pre_magnet_geometric_capture"] = pre_magnet_report
            else:
                report["pre_magnet_geometric_capture"] = {
                    "skipped": True,
                    "reason": "edge_angle_too_large_for_gripper_closed_nudge",
                    "max_angle_deg": float(getattr(args, "pre_magnet_geometric_capture_max_angle_deg", 10.0)),
                    "alignment": pre_magnet_alignment,
                }
        _set_role_magnets_enabled(base_env, role, True)
        capture_hold_steps = max(int(getattr(args, "post_magnet_capture_hold_steps", 0)), 0)
        if capture_hold_steps > 0:
            _add_hold_segment(env, segments, f"{role}_magnetic_capture_after_initial_open", capture_gripper, capture_hold_steps)
        capture_report = _run_pre_open_magnetic_capture(
            env=env,
            base_env=base_env,
            planner=planner,
            locked=locked,
            fixtures=fixtures,
            arrays=arrays,
            segments=segments,
            role=role,
            args=args,
            gripper=capture_gripper,
            require_active=True,
            name_prefix="magnetic_capture",
            log=log,
        )
        report["magnetic_capture_before_open"] = capture_report
        final_status = capture_report.get("final_status") if isinstance(capture_report.get("final_status"), dict) else {}
        final_alignment = final_status.get("alignment") if isinstance(final_status.get("alignment"), dict) else {}
        final_active = final_status.get("active_connection") if isinstance(final_status.get("active_connection"), dict) else {}
        max_partial_edge_error = float(final_alignment.get("max_point_error_m", float("inf")))
        partial_center_error = float(final_alignment.get("center_error_m", float("inf")))
        partial_edge_angle = float(final_alignment.get("edge_parallel_error_deg", float("inf")))
        needs_post_success_settle = bool(
            capture_report.get("success")
            and (
                max_partial_edge_error
                > float(getattr(args, "partial_open_settle_after_success_max_edge_error", float("inf")))
                or partial_center_error
                > float(getattr(args, "partial_open_settle_after_success_max_center_error", float("inf")))
                or partial_edge_angle
                > float(getattr(args, "partial_open_settle_after_success_max_edge_angle_deg", 180.0))
            )
        )
        should_try_partial_open_settle = bool(
            (not capture_report.get("success"))
            or (
                needs_post_success_settle
                and bool(getattr(args, "allow_partial_open_settle_after_active_capture", True))
            )
        )
        initial_open_is_full = bool(
            first_open_steps > 0
            and open_fraction >= 0.999
            and abs(float(capture_gripper) - float(args.release_open_gripper_value)) <= 1e-6
        )
        if initial_open_is_full and should_try_partial_open_settle:
            report["partial_open_settle_after_active_capture"] = {
                "skipped": True,
                "reason": "disabled_after_full_initial_open",
                "max_edge_error_m": max_partial_edge_error,
                "center_error_m": partial_center_error,
                "edge_angle_deg": partial_edge_angle,
            }
            should_try_partial_open_settle = False
        if capture_report.get("success") and needs_post_success_settle:
            report["partial_open_settle_after_active_capture"] = {
                "skipped": not bool(getattr(args, "allow_partial_open_settle_after_active_capture", True)),
                "reason": "active_capture_already_succeeded",
                "max_edge_error_m": max_partial_edge_error,
                "center_error_m": partial_center_error,
                "edge_angle_deg": partial_edge_angle,
            }
        if should_try_partial_open_settle:
            can_settle = bool(
                (
                    int(final_active.get("active_count", 0)) > 0
                    or max_partial_edge_error <= float(getattr(args, "partial_open_settle_without_active_max_edge_error", 0.015))
                )
                and max_partial_edge_error
                <= float(getattr(args, "partial_open_settle_max_edge_error", 0.022))
                and partial_center_error
                <= float(getattr(args, "partial_open_settle_max_center_error", 0.018))
                and partial_edge_angle
                <= float(getattr(args, "partial_open_settle_max_edge_angle_deg", 18.0))
            )
            if can_settle:
                open_fraction = min(max(float(getattr(args, "partial_open_settle_fraction", 0.45)), 0.0), 1.0)
                open_start_gripper = float(CLOSED_GRIPPER) + (float(args.release_open_gripper_value) - float(CLOSED_GRIPPER)) * open_fraction
                _add_ramp_segment(
                    env,
                    segments,
                    f"{role}_partial_open_settle_after_active_capture",
                    CLOSED_GRIPPER,
                    open_start_gripper,
                    int(getattr(args, "partial_open_settle_open_steps", 12)),
                )
                partial_open_used = True
                settle_steps = int(getattr(args, "partial_open_settle_hold_steps", 30))
                if settle_steps > 0:
                    _add_hold_segment(env, segments, f"{role}_partial_open_settle_hold", open_start_gripper, settle_steps)
                original_attempts = int(getattr(args, "magnetic_capture_nudge_attempts", 0))
                original_nudge_step = float(getattr(args, "magnetic_capture_nudge_step", 0.002))
                args.magnetic_capture_nudge_attempts = max(original_attempts, int(getattr(args, "partial_open_settle_nudge_attempts", 2)))
                args.magnetic_capture_nudge_step = min(
                    max(original_nudge_step, float(getattr(args, "partial_open_settle_nudge_step", 0.0012))),
                    float(getattr(args, "partial_open_settle_nudge_step", 0.0012)),
                )
                settle_capture_report = _run_pre_open_magnetic_capture(
                    env=env,
                    base_env=base_env,
                    planner=planner,
                    locked=locked,
                    fixtures=fixtures,
                    arrays=arrays,
                    segments=segments,
                    role=role,
                    args=args,
                    gripper=open_start_gripper,
                    require_active=True,
                    name_prefix="magnetic_capture_after_partial_open_settle",
                    log=log,
                )
                args.magnetic_capture_nudge_attempts = original_attempts
                args.magnetic_capture_nudge_step = original_nudge_step
                report["magnetic_capture_after_partial_open_settle"] = settle_capture_report
                capture_report = settle_capture_report
        if bool(getattr(args, "require_active_connection_before_open", False)) and not capture_report.get("success"):
            latest_status = capture_report.get("final_status") if isinstance(capture_report.get("final_status"), dict) else {}
            latest_active = latest_status.get("active_connection") if isinstance(latest_status.get("active_connection"), dict) else {}
            latest_alignment = latest_status.get("alignment") if isinstance(latest_status.get("alignment"), dict) else {}
            active_count = int(latest_active.get("active_count", 0) or 0)
            max_edge_error = float(latest_alignment.get("max_point_error_m", float("inf")))
            can_continue_after_full_open = bool(
                initial_open_is_full
                and active_count > 0
                and max_edge_error <= float(getattr(args, "pre_open_max_base_edge_error", 0.028))
            )
            if not can_continue_after_full_open:
                report["success"] = False
                report["failed_at"] = "magnetic_capture_before_open"
                report["failure_reasons"] = ["active_connection_not_captured_before_open"]
                return report
            report["magnetic_capture_before_open_continue"] = {
                "reason": "active_connection_present_after_full_open",
                "active_count": active_count,
                "max_edge_error_m": max_edge_error,
                "final_validation_required": True,
            }
        remaining_open_steps = max(int(args.open_steps) - first_open_steps, 0)
        if remaining_open_steps > 0:
            _add_ramp_segment(
                env,
                segments,
                f"{role}_open_gripper_after_magnet_capture",
                capture_gripper,
                args.release_open_gripper_value,
                remaining_open_steps,
            )
        report["connection_point_error_after_magnetic_hold"] = _role_connection_point_error(base_env, role)
        report["snap_report_before_open"] = base_env.get_magnetic_snap_report()
        report["magnet_enable_timing"] = {
            "mode": "after_initial_open",
            "first_open_steps": int(first_open_steps),
            "open_fraction": float(open_fraction),
            "capture_gripper": float(capture_gripper),
            "capture_hold_steps": int(capture_hold_steps),
            "remaining_open_steps": int(remaining_open_steps),
        }
    else:
        pre_magnet_attempts = int(getattr(args, "pre_magnet_geometric_capture_attempts", 0))
        if pre_magnet_attempts > 0:
            pre_magnet_alignment = _triangle_base_edge_alignment(base_env, locked, role)
            pre_magnet_angle = float(pre_magnet_alignment.get("edge_parallel_error_deg", float("inf")))
            if pre_magnet_angle <= float(getattr(args, "pre_magnet_geometric_capture_max_angle_deg", 10.0)):
                original_attempts = int(getattr(args, "magnetic_capture_nudge_attempts", 0))
                original_hold_steps = int(getattr(args, "magnetic_capture_hold_steps", 0))
                args.magnetic_capture_nudge_attempts = pre_magnet_attempts
                args.magnetic_capture_hold_steps = 0
                pre_magnet_report = _run_pre_open_magnetic_capture(
                    env=env,
                    base_env=base_env,
                    planner=planner,
                    locked=locked,
                    fixtures=fixtures,
                    arrays=arrays,
                    segments=segments,
                    role=role,
                    args=args,
                    gripper=CLOSED_GRIPPER,
                    require_active=False,
                    name_prefix="pre_magnet_geometric_capture",
                    log=log,
                )
                args.magnetic_capture_nudge_attempts = original_attempts
                args.magnetic_capture_hold_steps = original_hold_steps
                report["pre_magnet_geometric_capture"] = pre_magnet_report
            else:
                report["pre_magnet_geometric_capture"] = {
                    "skipped": True,
                    "reason": "edge_angle_too_large_for_gripper_closed_nudge",
                    "max_angle_deg": float(getattr(args, "pre_magnet_geometric_capture_max_angle_deg", 10.0)),
                    "alignment": pre_magnet_alignment,
                }
        _set_role_magnets_enabled(base_env, role, True)
        _add_hold_segment(env, segments, f"{role}_pre_open_magnetic_hold", CLOSED_GRIPPER, args.pre_open_hold_steps)
        capture_report = _run_pre_open_magnetic_capture(
            env=env,
            base_env=base_env,
            planner=planner,
            locked=locked,
            fixtures=fixtures,
            arrays=arrays,
            segments=segments,
            role=role,
            args=args,
            gripper=CLOSED_GRIPPER,
            require_active=True,
            name_prefix="magnetic_capture",
            log=log,
        )
        report["magnetic_capture_before_open"] = capture_report
        final_status = capture_report.get("final_status") if isinstance(capture_report.get("final_status"), dict) else {}
        final_alignment = final_status.get("alignment") if isinstance(final_status.get("alignment"), dict) else {}
        final_active = final_status.get("active_connection") if isinstance(final_status.get("active_connection"), dict) else {}
        max_partial_edge_error = float(final_alignment.get("max_point_error_m", float("inf")))
        partial_center_error = float(final_alignment.get("center_error_m", float("inf")))
        partial_edge_angle = float(final_alignment.get("edge_parallel_error_deg", float("inf")))
        needs_post_success_settle = bool(
            capture_report.get("success")
            and (
                max_partial_edge_error
                > float(getattr(args, "partial_open_settle_after_success_max_edge_error", float("inf")))
                or partial_center_error
                > float(getattr(args, "partial_open_settle_after_success_max_center_error", float("inf")))
                or partial_edge_angle
                > float(getattr(args, "partial_open_settle_after_success_max_edge_angle_deg", 180.0))
            )
        )
        should_try_partial_open_settle = bool(
            (not capture_report.get("success"))
            or (
                needs_post_success_settle
                and bool(getattr(args, "allow_partial_open_settle_after_active_capture", True))
            )
        )
        if capture_report.get("success") and needs_post_success_settle:
            report["partial_open_settle_after_active_capture"] = {
                "skipped": not bool(getattr(args, "allow_partial_open_settle_after_active_capture", True)),
                "reason": "active_capture_already_succeeded",
                "max_edge_error_m": max_partial_edge_error,
                "center_error_m": partial_center_error,
                "edge_angle_deg": partial_edge_angle,
            }
        if should_try_partial_open_settle:
            can_settle = bool(
                (
                    int(final_active.get("active_count", 0)) > 0
                    or max_partial_edge_error <= float(getattr(args, "partial_open_settle_without_active_max_edge_error", 0.015))
                )
                and max_partial_edge_error
                <= float(getattr(args, "partial_open_settle_max_edge_error", 0.022))
                and partial_center_error
                <= float(getattr(args, "partial_open_settle_max_center_error", 0.018))
                and partial_edge_angle
                <= float(getattr(args, "partial_open_settle_max_edge_angle_deg", 18.0))
            )
            if can_settle:
                open_fraction = min(max(float(getattr(args, "partial_open_settle_fraction", 0.45)), 0.0), 1.0)
                open_start_gripper = float(CLOSED_GRIPPER) + (float(args.release_open_gripper_value) - float(CLOSED_GRIPPER)) * open_fraction
                _add_ramp_segment(
                    env,
                    segments,
                    f"{role}_partial_open_settle_after_active_capture",
                    CLOSED_GRIPPER,
                    open_start_gripper,
                    int(getattr(args, "partial_open_settle_open_steps", 12)),
                )
                partial_open_used = True
                settle_steps = int(getattr(args, "partial_open_settle_hold_steps", 30))
                if settle_steps > 0:
                    _add_hold_segment(env, segments, f"{role}_partial_open_settle_hold", open_start_gripper, settle_steps)
                original_attempts = int(getattr(args, "magnetic_capture_nudge_attempts", 0))
                original_nudge_step = float(getattr(args, "magnetic_capture_nudge_step", 0.002))
                args.magnetic_capture_nudge_attempts = max(original_attempts, int(getattr(args, "partial_open_settle_nudge_attempts", 2)))
                args.magnetic_capture_nudge_step = min(
                    max(original_nudge_step, float(getattr(args, "partial_open_settle_nudge_step", 0.0012))),
                    float(getattr(args, "partial_open_settle_nudge_step", 0.0012)),
                )
                settle_capture_report = _run_pre_open_magnetic_capture(
                    env=env,
                    base_env=base_env,
                    planner=planner,
                    locked=locked,
                    fixtures=fixtures,
                    arrays=arrays,
                    segments=segments,
                    role=role,
                    args=args,
                    gripper=open_start_gripper,
                    require_active=True,
                    name_prefix="magnetic_capture_after_partial_open_settle",
                    log=log,
                )
                args.magnetic_capture_nudge_attempts = original_attempts
                args.magnetic_capture_nudge_step = original_nudge_step
                report["magnetic_capture_after_partial_open_settle"] = settle_capture_report
                capture_report = settle_capture_report
        if bool(getattr(args, "require_active_connection_before_open", False)) and not capture_report.get("success"):
            report["success"] = False
            report["failed_at"] = "magnetic_capture_before_open"
            report["failure_reasons"] = ["active_connection_not_captured_before_open"]
            return report
        report["connection_point_error_after_magnetic_hold"] = _role_connection_point_error(base_env, role)
        report["snap_report_before_open"] = base_env.get_magnetic_snap_report()
        remaining_open_steps = int(args.open_steps)
        if partial_open_used:
            remaining_open_steps = max(remaining_open_steps - int(getattr(args, "partial_open_settle_open_steps", 12)), 1)
        _add_ramp_segment(env, segments, f"{role}_open_gripper", open_start_gripper, args.release_open_gripper_value, remaining_open_steps)
        report["magnet_enable_timing"] = {
            "mode": "before_open",
            "pre_open_hold_steps": int(args.pre_open_hold_steps),
        }
    _add_hold_segment(env, segments, f"{role}_post_open_hold", args.release_open_gripper_value, args.post_open_hold_steps)
    current_q = _current_q(base_env)
    retreat_options: list[dict[str, Any]] = []
    retreat_max_joint_delta = float(getattr(args, "retreat_max_joint_delta", 0.0))
    for retreat_label, retreat_pose in _release_retreat_options(
        tcp_pose=_tcp_pose(base_env),
        actor_pose=_actor_pose(actor),
        primary_mode=args.retreat_mode,
        primary_distance=args.retreat_distance,
    ):
        if retreat_label == "none":
            _set_planner_world(planner, base_env, locked, fixtures, exclude_roles=set())
            full_world_valid, full_world_status = planner.check_start_state(current_q)
            retreat_options.append(
                {
                    "label": retreat_label,
                    "ik": {"skipped": True, "reason": "stay at current joint state"},
                    "success": bool(full_world_valid),
                    "joint_delta_from_current": 0.0,
                    "max_joint_delta": retreat_max_joint_delta,
                    "full_world_start_valid": bool(full_world_valid),
                    "full_world_start_status": full_world_status,
                    "q": current_q,
                }
            )
            continue
        ok_retreat, q_retreat, retreat_ik = _solve_triangle_ik(
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            target_pose=retreat_pose,
            start_q=current_q,
            ik_seeds=args.ik_seeds,
            exclude_roles={role},
        )
        joint_delta = None if q_retreat is None else float(_joint_distance(q_retreat, current_q))
        full_world_valid = False
        full_world_status = None
        if q_retreat is not None:
            _set_planner_world(planner, base_env, locked, fixtures, exclude_roles=set())
            full_world_valid, full_world_status = planner.check_start_state(q_retreat)
        joint_delta_ok = bool(
            joint_delta is not None
            and (retreat_max_joint_delta <= 0.0 or joint_delta <= retreat_max_joint_delta)
        )
        retreat_options.append(
            {
                "label": retreat_label,
                "ik": retreat_ik,
                "success": bool(ok_retreat and q_retreat is not None and joint_delta_ok and full_world_valid),
                "joint_delta_from_current": joint_delta,
                "max_joint_delta": retreat_max_joint_delta,
                "full_world_start_valid": bool(full_world_valid),
                "full_world_start_status": full_world_status,
                "q": q_retreat,
            }
        )
    nonzero_options = [
        option
        for option in retreat_options
        if option["success"] and str(option["label"]) != "none"
    ]
    selected_retreat = min(
        nonzero_options,
        key=lambda option: (float(option["joint_delta_from_current"] or 0.0), str(option["label"])),
        default=None,
    )
    if selected_retreat is None:
        selected_retreat = next((option for option in retreat_options if option["success"]), None)
    report["retreat_options"] = [
        {key: value for key, value in option.items() if key != "q"}
        for option in retreat_options
    ]
    report["retreat_selected"] = None if selected_retreat is None else {
        key: value for key, value in selected_retreat.items() if key != "q"
    }
    if selected_retreat is None:
        report["success"] = False
        report["failed_at"] = "retreat_clearance"
        return report
    if selected_retreat is not None and selected_retreat.get("q") is not None and str(selected_retreat["label"]) != "none":
        _add_adaptive_joint_segment(
            env=env,
            arrays=arrays,
            segments=segments,
            name=f"{role}_retreat",
            goal_q=selected_retreat["q"],
            gripper=args.release_open_gripper_value,
            base_steps=args.return_home_steps,
            action_repeat=args.action_repeat,
            final_hold=args.final_hold_steps,
            args=args,
        )
    _add_hold_segment(env, segments, f"{role}_stability_hold", args.release_open_gripper_value, args.stability_steps)
    report["pose_error_after_release"] = _pose_error(actor, targets[role])
    report["active_connection_count_for_role"] = _active_connection_count_for_role(base_env, role)
    report["connection_point_error_after_release"] = _role_connection_point_error(base_env, role)
    report["edge_alignment_after_release"] = _triangle_base_edge_alignment(base_env, locked, role)
    report["snap_report_after_release"] = base_env.get_magnetic_snap_report()
    final_edge_error = float(report["edge_alignment_after_release"].get("max_point_error_m", float("inf")))
    final_center_error = float(report["edge_alignment_after_release"].get("center_error_m", float("inf")))
    final_edge_angle = float(report["edge_alignment_after_release"].get("edge_parallel_error_deg", float("inf")))
    report["success"] = (
        report["pose_error_after_release"]["position_error_m"] <= args.max_position_error
        and report["pose_error_after_release"]["orientation_error_deg"] <= args.max_orientation_error_deg
        and report["active_connection_count_for_role"] >= args.min_active_connections
        and (
            float(getattr(args, "final_max_base_edge_error", 0.0)) <= 0.0
            or final_edge_error <= float(getattr(args, "final_max_base_edge_error", 0.0))
        )
        and (
            float(getattr(args, "final_max_base_center_error", 0.0)) <= 0.0
            or final_center_error <= float(getattr(args, "final_max_base_center_error", 0.0))
        )
        and (
            float(getattr(args, "final_max_base_edge_angle_deg", 0.0)) <= 0.0
            or final_edge_angle <= float(getattr(args, "final_max_base_edge_angle_deg", 0.0))
        )
    )
    if not report["success"]:
        failures: list[str] = []
        if report["pose_error_after_release"]["position_error_m"] > args.max_position_error:
            failures.append("position_error")
        if report["pose_error_after_release"]["orientation_error_deg"] > args.max_orientation_error_deg:
            failures.append("orientation_error")
        if report["active_connection_count_for_role"] < args.min_active_connections:
            failures.append("active_connection_count")
        if float(getattr(args, "final_max_base_edge_error", 0.0)) > 0.0 and final_edge_error > float(getattr(args, "final_max_base_edge_error", 0.0)):
            failures.append("base_edge_error")
        if float(getattr(args, "final_max_base_center_error", 0.0)) > 0.0 and final_center_error > float(getattr(args, "final_max_base_center_error", 0.0)):
            failures.append("base_center_error")
        if float(getattr(args, "final_max_base_edge_angle_deg", 0.0)) > 0.0 and final_edge_angle > float(getattr(args, "final_max_base_edge_angle_deg", 0.0)):
            failures.append("base_edge_angle")
        report["failed_at"] = "final_validation"
        report["failure_reasons"] = failures
    if report["success"]:
        safe_lift_report = _add_post_success_safe_lift(
            env=env,
            planner=planner,
            base_env=base_env,
            locked=locked,
            fixtures=fixtures,
            arrays=arrays,
            segments=segments,
            role=role,
            args=args,
        )
        report["post_success_safe_lift"] = safe_lift_report
        if safe_lift_report.get("enabled") and safe_lift_report.get("selected") is None:
            report["success"] = False
            report["failed_at"] = "post_success_safe_lift"
        if report["success"]:
            post_start_report = _current_start_state_report(
                planner=planner,
                base_env=base_env,
                locked=locked,
                fixtures=fixtures,
                args=args,
            )
            report["post_execution_start_state"] = post_start_report
            if not post_start_report["valid"]:
                recovery_report = _recover_to_valid_start_state(
                    env=env,
                    planner=planner,
                    base_env=base_env,
                    locked=locked,
                    fixtures=fixtures,
                    arrays=arrays,
                    segments=segments,
                    role=role,
                    args=args,
                )
                report["post_execution_start_state_recovery"] = recovery_report
                report["post_execution_start_state_after_recovery"] = recovery_report.get("after")
                if not recovery_report.get("success"):
                    report["success"] = False
                    report["failed_at"] = "post_execution_start_state_recovery"
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _make_logger(out_dir)
    summary_path = out_dir / "triangle_wall_summary.json"
    manifest_path = out_dir / "triangle_wall_manifest.json"
    arrays_path = out_dir / "triangle_wall_arrays.npz"
    video_path = out_dir / "triangle_wall_live.mp4"
    arrays: dict[str, np.ndarray] = {}
    segments: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    roles = [item.strip() for item in args.roles.split(",") if item.strip()]
    env = _make_env(bool(args.record_live))
    writer = None
    restore_step = None
    loaded_state_perturbation: dict[str, Any] = {}
    try:
        env.reset()
        base_env = env.unwrapped
        _set_robot_qpos(base_env, RM75_HOME, gripper_open=True)
        square_targets, square_locked, fixtures = _initialize_staged_open_cube(base_env)
        fixtures = _remove_second_layer_unused_square_artifacts(
            base_env,
            square_locked,
            square_targets,
            fixtures,
            log,
        )
        rear_collision_wall = _add_rear_collision_wall_fixture(
            base_env=base_env,
            targets=square_targets,
            fixtures=fixtures,
            args=args,
            log=log,
        )
        snap_parameter_overrides = {}
        snap = base_env.magnetic_snap
        if float(getattr(args, "magnet_attach_distance", 0.0)) > 0.0:
            snap.attach_distance = float(args.magnet_attach_distance)
            snap_parameter_overrides["attach_distance"] = snap.attach_distance
        if float(getattr(args, "magnet_attract_distance", 0.0)) > 0.0:
            snap.attract_distance = float(args.magnet_attract_distance)
            snap_parameter_overrides["attract_distance"] = snap.attract_distance
        if float(getattr(args, "magnet_detach_distance", 0.0)) > 0.0:
            snap.detach_distance = float(args.magnet_detach_distance)
            snap_parameter_overrides["detach_distance"] = snap.detach_distance
        _apply_float_override(
            snap,
            "base_connection_detach_distance",
            getattr(args, "magnet_loaded_base_detach_distance", None),
            snap_parameter_overrides,
        )
        _apply_float_override(snap, "attract_stiffness", getattr(args, "magnet_attract_stiffness", None), snap_parameter_overrides)
        _apply_float_override(snap, "attract_force_limit", getattr(args, "magnet_attract_force_limit", None), snap_parameter_overrides)
        _apply_float_override(snap, "attract_torque_stiffness", getattr(args, "magnet_attract_torque_stiffness", None), snap_parameter_overrides)
        _apply_float_override(snap, "attract_torque_limit", getattr(args, "magnet_attract_torque_limit", None), snap_parameter_overrides)
        _apply_float_override(snap, "attract_normal_torque_stiffness", getattr(args, "magnet_attract_normal_torque_stiffness", None), snap_parameter_overrides)
        _apply_float_override(snap, "attract_normal_torque_limit", getattr(args, "magnet_attract_normal_torque_limit", None), snap_parameter_overrides)
        _apply_float_override(snap, "active_magnet_stiffness", getattr(args, "magnet_active_stiffness", None), snap_parameter_overrides)
        _apply_float_override(snap, "active_magnet_damping", getattr(args, "magnet_active_damping", None), snap_parameter_overrides)
        _apply_float_override(snap, "active_magnet_force_limit", getattr(args, "magnet_active_force_limit", None), snap_parameter_overrides)
        _apply_float_override(snap, "drive_stiffness", getattr(args, "magnet_drive_stiffness", None), snap_parameter_overrides)
        _apply_float_override(snap, "drive_damping", getattr(args, "magnet_drive_damping", None), snap_parameter_overrides)
        _apply_float_override(snap, "drive_force_limit", getattr(args, "magnet_drive_force_limit", None), snap_parameter_overrides)
        _apply_float_override(snap, "drive_angular_stiffness", getattr(args, "magnet_drive_angular_stiffness", None), snap_parameter_overrides)
        _apply_float_override(snap, "drive_angular_damping", getattr(args, "magnet_drive_angular_damping", None), snap_parameter_overrides)
        _apply_float_override(snap, "drive_angular_force_limit", getattr(args, "magnet_drive_angular_force_limit", None), snap_parameter_overrides)
        if snap_parameter_overrides:
            log(f"magnet parameter overrides: {snap_parameter_overrides}")
            snap.refresh_drive_properties()
            log("magnet drive properties refreshed after parameter overrides")
        loaded_roles = _load_assembly_state(
            path=args.load_assembly_state,
            base_env=base_env,
            locked=square_locked,
            log=log,
            restore_robot_qpos=bool(args.restore_loaded_robot_qpos),
        )
        locked, triangle_targets, stage_targets = _prepare_triangle_roles(
            base_env,
            target_lift=args.target_lift,
            stage_x_start=args.triangle_stage_x_start,
            stage_x_step=args.triangle_stage_x_step,
            stage_y=args.triangle_stage_y,
            log=log,
        )
        if args.load_assembly_state:
            loaded_roles = _load_assembly_state(
                path=args.load_assembly_state,
                base_env=base_env,
                locked=locked,
                log=log,
                restore_robot_qpos=bool(args.restore_loaded_robot_qpos),
            )
            loaded_state_perturbation = multi_wall_module._apply_loaded_state_perturbation(
                base_env=base_env,
                locked=locked,
                targets=square_targets,
                args=args,
                log=log,
            )
            _refresh_unloaded_triangle_targets(
                base_env,
                locked,
                triangle_targets,
                loaded_roles,
                target_lift=args.target_lift,
                log=log,
            )
            restored_loaded_magnetic_connections = (
                _restore_loaded_magnetic_connections(
                    state_path=args.load_assembly_state,
                    base_env=base_env,
                    locked=locked,
                    log=log,
                )
                if bool(getattr(args, "restore_loaded_magnetic_connections", True))
                else []
            )
        else:
            restored_loaded_magnetic_connections = []
        loaded_base_connection_targets = (
            _configure_loaded_base_connection_targets(base_env, loaded_roles, locked, log)
            if bool(getattr(args, "require_loaded_base_full_connections", False)) and loaded_roles
            else {}
        )
        loaded_pose_targets = _current_pose_targets_for_roles(locked, loaded_roles)
        floor_anchor = _anchor_floor_for_build(
            base_env=base_env,
            locked=locked,
            target_pose=loaded_pose_targets.get("floor", square_targets["floor"]),
            args=args,
            log=log,
        )
        triangle_stage_fixtures = _create_triangle_staging_fixtures(
            base_env,
            stage_targets,
            enabled=bool(args.add_triangle_staging_fixtures),
            log=log,
        )
        fixtures.extend(triangle_stage_fixtures)
        for role in roles:
            _set_role_magnets_enabled(base_env, role, False)
        _add_hold_segment(env, segments, "loaded_first_layer_settle", OPEN_GRIPPER, args.loaded_state_settle_steps)
        loaded_first_layer_disabled_connections: list[dict[str, Any]] = []
        if bool(getattr(args, "disable_overstretched_loaded_connections", True)):
            threshold = float(getattr(args, "loaded_connection_max_active_error", 0.0))
            if threshold <= 0.0:
                threshold = float(base_env.magnetic_snap.attach_distance)
            loaded_first_layer_disabled_connections = _disable_overstretched_active_connections(
                base_env,
                threshold=threshold,
                log=log,
            )
        loaded_base_connection_report_before_post_restore = _loaded_base_connection_report(base_env, loaded_roles)
        post_settle_restored_loaded_magnetic_connections: list[dict[str, Any]] = []
        if (
            bool(getattr(args, "require_loaded_base_full_connections", False))
            and bool(getattr(args, "restore_loaded_magnetic_connections", True))
            and loaded_roles
            and int(loaded_base_connection_report_before_post_restore.get("active_connection_count", 0))
            < int(loaded_base_connection_report_before_post_restore.get("expected_connection_count", 0))
        ):
            post_settle_restored_loaded_magnetic_connections = _restore_loaded_magnetic_connections(
                state_path=args.load_assembly_state,
                base_env=base_env,
                locked=locked,
                log=log,
            )
            post_restore_steps = max(int(getattr(args, "loaded_state_post_prune_settle_steps", 0)), 0)
            if post_settle_restored_loaded_magnetic_connections and post_restore_steps > 0:
                _add_hold_segment(
                    env,
                    segments,
                    "loaded_first_layer_post_restore_settle",
                    OPEN_GRIPPER,
                    post_restore_steps,
                )
        loaded_first_layer_snap_report = base_env.get_magnetic_snap_report()
        loaded_base_connection_report = _loaded_base_connection_report(base_env, loaded_roles)
        loaded_base_validation = _loaded_base_validation_report(loaded_base_connection_report, args)
        loaded_first_layer_active_connections = {
            role: _role_active_connection_report(base_env, role)
            for role in loaded_roles
            if role in locked
        }
        log(f"loaded first-layer snap report: {loaded_first_layer_snap_report}")
        log(f"loaded base connection report: {loaded_base_connection_report}")
        log(f"loaded base validation: {loaded_base_validation}")
        log(f"loaded first-layer active connections: {loaded_first_layer_active_connections}")
        if args.record_live:
            writer = imageio.get_writer(video_path, fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
            for _ in range(max(int(args.fps), 1)):
                _append_frame(writer, env)
            original_step = globals()["_step_action"]
            original_single_step = single_wall_module._step_action
            original_multi_step = multi_wall_module._step_action
            counter = {"step": 0}

            def wrapped_step(step_env: Any, target_q: np.ndarray, gripper: float, _writer: Any | None, _record_every: int, index: int) -> None:
                original_step(step_env, target_q, gripper, None, 1, index)
                if counter["step"] % max(int(args.record_every), 1) == 0:
                    _append_frame(writer, step_env)
                counter["step"] += 1

            globals()["_step_action"] = wrapped_step
            single_wall_module._step_action = wrapped_step
            multi_wall_module._step_action = wrapped_step

            def restore() -> int:
                globals()["_step_action"] = original_step
                single_wall_module._step_action = original_single_step
                multi_wall_module._step_action = original_multi_step
                return int(counter["step"])

            restore_step = restore
        if (
            bool(getattr(args, "require_loaded_base_full_connections", False))
            and loaded_roles
            and not bool(loaded_base_validation.get("success", True))
        ):
            final = {
                "success": False,
                "failed_at": "loaded_base_full_connection_gate",
                "failure_reasons": list(loaded_base_validation.get("failures", [])),
                "roles": roles,
                "completed_roles": [],
                "loaded_roles": loaded_roles,
                "video": str(video_path) if args.record_live else None,
                "stage_targets": {role: _pose_to_report(pose) for role, pose in stage_targets.items()},
                "triangle_stage_fixture_count": len(triangle_stage_fixtures),
                "floor_anchor": floor_anchor is not None,
                "loaded_state_perturbation": loaded_state_perturbation,
                "loaded_first_layer_snap_report": loaded_first_layer_snap_report,
                "loaded_base_connection_report_before_post_restore": loaded_base_connection_report_before_post_restore,
                "loaded_base_connection_report": loaded_base_connection_report,
                "loaded_base_validation": loaded_base_validation,
                "loaded_first_layer_active_connections": loaded_first_layer_active_connections,
                "loaded_first_layer_disabled_connections": loaded_first_layer_disabled_connections,
                "restored_loaded_magnetic_connections": restored_loaded_magnetic_connections,
                "post_settle_restored_loaded_magnetic_connections": post_settle_restored_loaded_magnetic_connections,
                "loaded_base_connection_targets": loaded_base_connection_targets,
            }
            np.savez_compressed(arrays_path, **arrays)
            manifest = {
                "name": "second_layer_triangle_wall_path",
                "control_mode": "pd_joint_pos_abs",
                "arrays": str(arrays_path),
                "segments": segments,
                "reports": reports,
                "final": final,
                "rear_collision_wall": rear_collision_wall,
                "loaded_state_perturbation": loaded_state_perturbation,
                "loaded_first_layer_snap_report": loaded_first_layer_snap_report,
                "loaded_base_connection_report_before_post_restore": loaded_base_connection_report_before_post_restore,
                "loaded_base_connection_report": loaded_base_connection_report,
                "loaded_base_validation": loaded_base_validation,
                "loaded_first_layer_active_connections": loaded_first_layer_active_connections,
                "loaded_first_layer_disabled_connections": loaded_first_layer_disabled_connections,
                "restored_loaded_magnetic_connections": restored_loaded_magnetic_connections,
                "post_settle_restored_loaded_magnetic_connections": post_settle_restored_loaded_magnetic_connections,
                "loaded_base_connection_targets": loaded_base_connection_targets,
            }
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            result = {"summary": str(summary_path), "manifest": str(manifest_path), "arrays": str(arrays_path), "reports": reports, "final": final}
            summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            return result
        if bool(args.preview_layout_only):
            _add_hold_segment(env, segments, "preview_triangle_staging_layout", OPEN_GRIPPER, args.preview_steps)
            final = {
                "success": True,
                "roles": roles,
                "completed_roles": [],
                "loaded_roles": loaded_roles,
                "video": str(video_path) if args.record_live else None,
                "stage_targets": {role: _pose_to_report(pose) for role, pose in stage_targets.items()},
                "triangle_stage_fixture_count": len(triangle_stage_fixtures),
                "preview_layout_only": True,
                "loaded_state_perturbation": loaded_state_perturbation,
            }
            manifest = {
                "name": "second_layer_triangle_wall_path_preview",
                "control_mode": "pd_joint_pos_abs",
                "segments": segments,
                "final": final,
            }
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            result = {"summary": str(summary_path), "manifest": str(manifest_path), "arrays": str(arrays_path), "reports": reports, "final": final}
            summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            return result
        planner = RM75CuRoboPlanner(
            RM75CuRoboPlannerConfig(
                curobo_root=Path(args.curobo_root),
                robot_cfg_path=Path(args.robot_cfg),
                position_threshold=args.ik_position_threshold,
                rotation_threshold=args.ik_rotation_threshold,
                num_ik_seeds=args.ik_seeds,
                collision_activation_distance=0.018,
                build_motion_gen=True,
            )
        )
        for role in roles:
            if role not in locked or role not in triangle_targets:
                log(f"{role}: skipped because role is not a triangle build target")
                continue
            role_args = _args_for_role(args, role)
            report = _build_triangle_role(
                env=env,
                planner=planner,
                locked=locked,
                fixtures=fixtures,
                targets=triangle_targets,
                arrays=arrays,
                segments=segments,
                role=role,
                args=role_args,
                log=log,
            )
            reports.append(report)
            if not report.get("success"):
                break
        completed_roles = [item["role"] for item in reports if item.get("success")]
        validation_targets = {**square_targets, **triangle_targets, **loaded_pose_targets}
        final_stability_roles = [role for role in ["floor", *loaded_roles, *completed_roles] if role in locked]
        if floor_anchor is not None and bool(args.release_floor_anchor_before_final):
            base_env.magnetic_snap._disable_drive(floor_anchor["drive"])
            _add_hold_segment(env, segments, "floor_anchor_release_settle", args.release_open_gripper_value, args.floor_anchor_release_settle_steps)
            log("floor: released build anchor before final stability check")
        all_role_stability = _validate_roles_after_settle(
            env=env,
            base_env=base_env,
            locked=locked,
            targets=validation_targets,
            roles=final_stability_roles,
            segments=segments,
            args=args,
            name="final_square_and_triangle_stability_check",
            gripper=args.release_open_gripper_value,
            log=log,
        ) if completed_roles else {"success": False, "roles": [], "role_reports": {}, "steps": 0}
        final_triangle_roles = list(
            dict.fromkeys(
                role
                for role in [*loaded_roles, *completed_roles]
                if role in locked and role in triangle_targets
            )
        )
        final_triangle_edge_alignment = {
            role: _triangle_base_edge_alignment(base_env, locked, role)
            for role in final_triangle_roles
        }
        final_triangle_connection_point_error = {
            role: _role_connection_point_error(base_env, role)
            for role in final_triangle_roles
        }
        final_triangle_edge_success = True
        for alignment in final_triangle_edge_alignment.values():
            if not alignment.get("success"):
                final_triangle_edge_success = False
                continue
            if (
                float(getattr(args, "final_max_base_edge_error", 0.0)) > 0.0
                and float(alignment.get("max_point_error_m", float("inf"))) > float(args.final_max_base_edge_error)
            ):
                final_triangle_edge_success = False
            if (
                float(getattr(args, "final_max_base_center_error", 0.0)) > 0.0
                and float(alignment.get("center_error_m", float("inf"))) > float(args.final_max_base_center_error)
            ):
                final_triangle_edge_success = False
            if (
                float(getattr(args, "final_max_base_edge_angle_deg", 0.0)) > 0.0
                and float(alignment.get("edge_parallel_error_deg", float("inf"))) > float(args.final_max_base_edge_angle_deg)
            ):
                final_triangle_edge_success = False
        final = {
            "success": bool(
                reports
                and all(item.get("success") for item in reports)
                and all_role_stability.get("success")
                and final_triangle_edge_success
            ),
            "roles": roles,
            "completed_roles": completed_roles,
            "loaded_roles": loaded_roles,
            "video": str(video_path) if args.record_live else None,
            "stage_targets": {role: _pose_to_report(pose) for role, pose in stage_targets.items()},
            "triangle_stage_fixture_count": len(triangle_stage_fixtures),
            "all_role_stability": all_role_stability,
            "final_triangle_edge_alignment": final_triangle_edge_alignment,
            "final_triangle_connection_point_error": final_triangle_connection_point_error,
            "final_triangle_edge_success": final_triangle_edge_success,
            "floor_anchor": floor_anchor is not None,
            "floor_anchor_released_before_final": bool(floor_anchor is not None and args.release_floor_anchor_before_final),
            "loaded_state_perturbation": loaded_state_perturbation,
            "loaded_first_layer_snap_report": loaded_first_layer_snap_report,
            "loaded_base_connection_report_before_post_restore": loaded_base_connection_report_before_post_restore,
            "loaded_base_connection_report": loaded_base_connection_report,
            "loaded_base_validation": loaded_base_validation,
            "loaded_first_layer_active_connections": loaded_first_layer_active_connections,
            "loaded_first_layer_disabled_connections": loaded_first_layer_disabled_connections,
            "restored_loaded_magnetic_connections": restored_loaded_magnetic_connections,
            "post_settle_restored_loaded_magnetic_connections": post_settle_restored_loaded_magnetic_connections,
            "loaded_base_connection_targets": loaded_base_connection_targets,
        }
        if str(getattr(args, "save_assembly_state", "") or ""):
            _save_assembly_state(
                path=args.save_assembly_state,
                base_env=base_env,
                locked=locked,
                targets=validation_targets,
                reports=reports,
                roles=roles,
                log=log,
                loaded_completed_roles=loaded_roles,
            )
        np.savez_compressed(arrays_path, **arrays)
        manifest = {
            "name": "second_layer_triangle_wall_path",
            "control_mode": "pd_joint_pos_abs",
            "arrays": str(arrays_path),
            "segments": segments,
            "reports": reports,
            "final": final,
            "rear_collision_wall": rear_collision_wall,
            "loaded_state_perturbation": loaded_state_perturbation,
            "loaded_first_layer_snap_report": loaded_first_layer_snap_report,
            "loaded_base_connection_report_before_post_restore": loaded_base_connection_report_before_post_restore,
            "loaded_base_connection_report": loaded_base_connection_report,
            "loaded_base_validation": loaded_base_validation,
            "loaded_first_layer_active_connections": loaded_first_layer_active_connections,
            "loaded_first_layer_disabled_connections": loaded_first_layer_disabled_connections,
            "restored_loaded_magnetic_connections": restored_loaded_magnetic_connections,
            "post_settle_restored_loaded_magnetic_connections": post_settle_restored_loaded_magnetic_connections,
            "loaded_base_connection_targets": loaded_base_connection_targets,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        result = {"summary": str(summary_path), "manifest": str(manifest_path), "arrays": str(arrays_path), "reports": reports, "final": final}
        summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return result
    finally:
        if restore_step is not None:
            restore_step()
        if writer is not None:
            for _ in range(max(int(getattr(args, "fps", 12)), 1)):
                _append_frame(writer, env)
            writer.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "triangle_wall_path_v1"))
    parser.add_argument("--robot-cfg", default=str(Path(__file__).resolve().parent / "curobo_rm75_config" / "rm75.yml"))
    parser.add_argument("--curobo-root", default=str(DEFAULT_CUROBO_ROOT))
    parser.add_argument("--load-assembly-state", default=str(Path(__file__).resolve().parent / "four_wall_checkpoint_build_v1" / "four_wall_state.json"))
    parser.add_argument("--save-assembly-state", default="")
    parser.add_argument("--restore-loaded-robot-qpos", action="store_true")
    parser.add_argument("--loaded-state-perturb-dx", type=float, default=0.0)
    parser.add_argument("--loaded-state-perturb-dy", type=float, default=0.0)
    parser.add_argument("--loaded-state-perturb-dz", type=float, default=0.0)
    parser.add_argument("--loaded-state-perturb-yaw-deg", type=float, default=0.0)
    parser.add_argument("--loaded-state-perturb-origin-role", default="floor")
    parser.add_argument("--loaded-state-perturb-roles", default="floor,right_wall,back_wall,left_wall,front_wall")
    parser.add_argument("--loaded-state-perturb-target-roles", default="floor,right_wall,back_wall,left_wall,front_wall")
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
    parser.add_argument("--roles", default="right_second_triangle")
    parser.add_argument("--triangle-stage-x-start", type=float, default=-0.70)
    parser.add_argument("--triangle-stage-x-step", type=float, default=0.10)
    parser.add_argument("--triangle-stage-y", type=float, default=0.19)
    parser.add_argument("--add-triangle-staging-fixtures", action="store_true")
    parser.add_argument("--preview-layout-only", action="store_true")
    parser.add_argument("--preview-steps", type=int, default=80)
    parser.add_argument("--target-lift", type=float, default=0.0)
    parser.add_argument("--record-live", action="store_true")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--record-every", type=int, default=3)
    parser.add_argument("--loaded-state-settle-steps", type=int, default=20)
    parser.add_argument("--loaded-state-post-prune-settle-steps", type=int, default=20)
    parser.add_argument("--disable-overstretched-loaded-connections", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loaded-connection-max-active-error", type=float, default=0.0)
    parser.add_argument("--restore-loaded-magnetic-connections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-loaded-base-full-connections", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loaded-base-min-active-connections", type=int, default=0)
    parser.add_argument("--loaded-base-max-point-error", type=float, default=0.0)
    parser.add_argument("--strategy-preset", choices=["none", "stable_fast_v1"], default="none")
    parser.add_argument("--max-grasp-candidates", type=int, default=24)
    parser.add_argument("--fast-screen-max-grasp-candidates", type=int, default=12)
    parser.add_argument("--ik-seeds", type=int, default=48)
    parser.add_argument("--fast-screen-ik-seeds", type=int, default=24)
    parser.add_argument("--ik-position-threshold", type=float, default=0.012)
    parser.add_argument("--ik-rotation-threshold", type=float, default=0.18)
    parser.add_argument("--release-ik-max-position-error", type=float, default=0.0)
    parser.add_argument("--release-ik-max-rotation-error", type=float, default=0.0)
    parser.add_argument("--preplace-ik-max-position-error", type=float, default=0.0)
    parser.add_argument("--preplace-ik-max-rotation-error", type=float, default=0.0)
    parser.add_argument("--pregrasp-timeout", type=float, default=2.5)
    parser.add_argument("--lift-height", type=float, default=0.10)
    parser.add_argument("--preplace-height", type=float, default=0.055)
    parser.add_argument("--preplace-staging-mode", choices=["fallback", "always", "off"], default="fallback")
    parser.add_argument("--preplace-staging", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preplace-staging-z-margin", type=float, default=0.04)
    parser.add_argument("--preplace-staging-max-joint-delta", type=float, default=0.0)
    parser.add_argument("--allow-direct-preplace-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preplace-heights", default="")
    parser.add_argument("--release-gap-mms", default="")
    parser.add_argument("--release-yaw-degs", default="")
    parser.add_argument("--max-release-candidates", type=int, default=0)
    parser.add_argument("--use-triangle-mesh-obstacles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triangle-mesh-obstacle-scale", type=float, default=1.0)
    parser.add_argument("--min-parent-top-edge-z", type=float, default=0.0)
    parser.add_argument("--move-steps", type=int, default=28)
    parser.add_argument("--short-steps", type=int, default=16)
    parser.add_argument("--release-steps", type=int, default=22)
    parser.add_argument("--max-joint-step", type=float, default=0.06)
    parser.add_argument("--max-segment-steps", type=int, default=420)
    parser.add_argument("--grasp-max-joint-delta", type=float, default=3.0)
    parser.add_argument("--lift-max-joint-delta", type=float, default=0.0)
    parser.add_argument("--release-preplace-max-joint-delta", type=float, default=2.8)
    parser.add_argument("--release-max-joint-delta", type=float, default=2.2)
    parser.add_argument("--triangle-candidate-pool-size", type=int, default=1)
    parser.add_argument("--use-screened-release-after-lift", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replan-release-after-preplace", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-screened-release-fallback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--close-steps", type=int, default=16)
    parser.add_argument("--open-steps", type=int, default=12)
    parser.add_argument("--enable-magnets-after-initial-open", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--open-before-magnet-steps", type=int, default=0)
    parser.add_argument("--open-before-magnet-fraction", type=float, default=0.0)
    parser.add_argument("--post-magnet-capture-hold-steps", type=int, default=0)
    parser.add_argument("--require-active-connection-before-open", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pre-magnet-geometric-capture-attempts", type=int, default=0)
    parser.add_argument("--pre-magnet-geometric-capture-max-angle-deg", type=float, default=10.0)
    parser.add_argument("--magnetic-capture-nudge-attempts", type=int, default=0)
    parser.add_argument("--magnetic-capture-nudge-step", type=float, default=0.002)
    parser.add_argument("--magnetic-capture-nudge-steps", type=int, default=10)
    parser.add_argument("--magnetic-capture-hold-steps", type=int, default=10)
    parser.add_argument("--magnetic-capture-max-angle-step-deg", type=float, default=2.0)
    parser.add_argument("--magnetic-capture-max-joint-delta", type=float, default=0.8)
    parser.add_argument("--magnetic-capture-max-downward-step", type=float, default=float("inf"))
    parser.add_argument("--magnetic-capture-revert-if-worse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--magnetic-capture-revert-tolerance", type=float, default=0.0007)
    parser.add_argument("--magnetic-capture-max-edge-error", type=float, default=0.010)
    parser.add_argument("--magnetic-capture-max-center-error", type=float, default=0.010)
    parser.add_argument("--magnetic-capture-max-edge-angle-deg", type=float, default=10.0)
    parser.add_argument("--allow-partial-open-settle-after-active-capture", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--partial-open-settle-fraction", type=float, default=0.75)
    parser.add_argument("--partial-open-settle-open-steps", type=int, default=12)
    parser.add_argument("--partial-open-settle-hold-steps", type=int, default=50)
    parser.add_argument("--partial-open-settle-nudge-attempts", type=int, default=5)
    parser.add_argument("--partial-open-settle-nudge-step", type=float, default=0.002)
    parser.add_argument("--partial-open-settle-max-edge-error", type=float, default=0.022)
    parser.add_argument("--partial-open-settle-without-active-max-edge-error", type=float, default=0.022)
    parser.add_argument("--partial-open-settle-max-center-error", type=float, default=0.018)
    parser.add_argument("--partial-open-settle-max-edge-angle-deg", type=float, default=18.0)
    parser.add_argument("--partial-open-settle-after-success-max-edge-error", type=float, default=float("inf"))
    parser.add_argument("--partial-open-settle-after-success-max-center-error", type=float, default=float("inf"))
    parser.add_argument("--partial-open-settle-after-success-max-edge-angle-deg", type=float, default=180.0)
    parser.add_argument("--pre-open-hold-steps", type=int, default=50)
    parser.add_argument("--post-open-hold-steps", type=int, default=20)
    parser.add_argument("--release-correction-attempts", type=int, default=2)
    parser.add_argument("--release-correction-steps", type=int, default=10)
    parser.add_argument("--release-correction-position-threshold", type=float, default=0.018)
    parser.add_argument("--release-correction-orientation-threshold-deg", type=float, default=15.0)
    parser.add_argument("--release-correction-edge-threshold", type=float, default=0.014)
    parser.add_argument("--release-correction-max-joint-delta", type=float, default=2.4)
    parser.add_argument("--release-correction-revert-if-worse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--release-correction-revert-tolerance", type=float, default=0.0005)
    parser.add_argument("--max-actor-workspace-radius", type=float, default=1.5)
    parser.add_argument("--max-recoverable-release-position-error", type=float, default=0.18)
    parser.add_argument("--max-recoverable-release-orientation-error-deg", type=float, default=90.0)
    parser.add_argument("--max-recoverable-connection-point-error", type=float, default=0.18)
    parser.add_argument("--edge-seating-attempts", type=int, default=0)
    parser.add_argument("--edge-seating-steps", type=int, default=14)
    parser.add_argument("--edge-seating-max-step", type=float, default=0.006)
    parser.add_argument("--edge-seating-max-error", type=float, default=0.013)
    parser.add_argument("--edge-seating-center-error", type=float, default=0.012)
    parser.add_argument("--edge-seating-angle-error-deg", type=float, default=10.0)
    parser.add_argument("--edge-seating-max-angle-step-deg", type=float, default=2.0)
    parser.add_argument("--edge-seating-rotation-mode", choices=["auto", "world_z", "edge_cross"], default="auto")
    parser.add_argument("--edge-seating-rotation-sign", type=float, default=1.0)
    parser.add_argument("--edge-seating-revert-if-worse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge-seating-revert-tolerance", type=float, default=0.0005)
    parser.add_argument("--stability-steps", type=int, default=120)
    parser.add_argument("--return-home-steps", type=int, default=18)
    parser.add_argument("--retreat-height", type=float, default=0.045)
    parser.add_argument("--retreat-distance", type=float, default=0.045)
    parser.add_argument("--retreat-max-joint-delta", type=float, default=1.2)
    parser.add_argument("--post-success-safe-lift-steps", type=int, default=0)
    parser.add_argument("--post-success-safe-lift-height", type=float, default=0.06)
    parser.add_argument("--post-success-safe-lift-max-joint-delta", type=float, default=1.6)
    parser.add_argument("--post-success-safe-lift-settle-steps", type=int, default=30)
    parser.add_argument(
        "--retreat-mode",
        choices=[
            "world_z",
            "tcp_back",
            "tcp_forward",
            "tcp_open_axis_pos",
            "tcp_open_axis_neg",
            "triangle_normal_pos",
            "triangle_normal_neg",
            "none",
        ],
        default="tcp_back",
    )
    parser.add_argument("--release-candidate-index", type=int, default=-1)
    parser.add_argument("--release-candidate-indices", default="")
    parser.add_argument("--release-candidate-index-groups", default="")
    parser.add_argument("--preplace-heights-by-role", default="")
    parser.add_argument("--preplace-height-by-role", default="")
    parser.add_argument("--release-open-gripper-values", default="")
    parser.add_argument("--retreat-modes", default="")
    parser.add_argument("--retreat-distances", default="")
    parser.add_argument("--open-steps-by-role", default="")
    parser.add_argument("--pre-open-hold-steps-by-role", default="")
    parser.add_argument("--post-open-hold-steps-by-role", default="")
    parser.add_argument("--edge-seating-attempts-by-role", default="")
    parser.add_argument("--release-correction-attempts-by-role", default="")
    parser.add_argument("--auto-release-profile", choices=["none", "v6_physical"], default="none")
    parser.add_argument("--release-open-gripper-value", type=float, default=0.0)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--final-hold-steps", type=int, default=2)
    parser.add_argument("--final-all-roles-stability-steps", type=int, default=50)
    parser.add_argument("--all-roles-max-position-error", type=float, default=0.035)
    parser.add_argument("--all-roles-max-orientation-error-deg", type=float, default=35.0)
    parser.add_argument("--all-roles-max-drift-position", type=float, default=0.008)
    parser.add_argument("--all-roles-max-drift-orientation-deg", type=float, default=5.0)
    parser.add_argument("--all-roles-max-linear-speed", type=float, default=0.08)
    parser.add_argument("--all-roles-max-angular-speed", type=float, default=1.0)
    parser.add_argument("--max-position-error", type=float, default=0.035)
    parser.add_argument("--max-orientation-error-deg", type=float, default=35.0)
    parser.add_argument("--min-active-connections", type=int, default=1)
    parser.add_argument("--final-max-base-edge-error", type=float, default=0.0)
    parser.add_argument("--final-max-base-center-error", type=float, default=0.0)
    parser.add_argument("--final-max-base-edge-angle-deg", type=float, default=0.0)
    parser.add_argument("--pre-open-max-base-edge-error", type=float, default=0.035)
    parser.add_argument("--pre-open-max-base-center-error", type=float, default=0.030)
    parser.add_argument("--pre-open-max-base-edge-angle-deg", type=float, default=25.0)
    parser.add_argument("--magnet-attach-distance", type=float, default=0.0)
    parser.add_argument("--magnet-attract-distance", type=float, default=0.0)
    parser.add_argument("--magnet-detach-distance", type=float, default=0.0)
    parser.add_argument("--magnet-loaded-base-detach-distance", type=float, default=0.0)
    parser.add_argument("--magnet-attract-stiffness", type=float, default=None)
    parser.add_argument("--magnet-attract-force-limit", type=float, default=None)
    parser.add_argument("--magnet-attract-torque-stiffness", type=float, default=None)
    parser.add_argument("--magnet-attract-torque-limit", type=float, default=None)
    parser.add_argument("--magnet-attract-normal-torque-stiffness", type=float, default=None)
    parser.add_argument("--magnet-attract-normal-torque-limit", type=float, default=None)
    parser.add_argument("--magnet-active-stiffness", type=float, default=None)
    parser.add_argument("--magnet-active-damping", type=float, default=None)
    parser.add_argument("--magnet-active-force-limit", type=float, default=None)
    parser.add_argument("--magnet-drive-stiffness", type=float, default=None)
    parser.add_argument("--magnet-drive-damping", type=float, default=None)
    parser.add_argument("--magnet-drive-force-limit", type=float, default=None)
    parser.add_argument("--magnet-drive-angular-stiffness", type=float, default=None)
    parser.add_argument("--magnet-drive-angular-damping", type=float, default=None)
    parser.add_argument("--magnet-drive-angular-force-limit", type=float, default=None)
    parser.add_argument("--anchor-floor-during-build", action="store_true")
    parser.add_argument("--floor-anchor-stiffness", type=float, default=1200.0)
    parser.add_argument("--floor-anchor-damping", type=float, default=90.0)
    parser.add_argument("--floor-anchor-force-limit", type=float, default=35.0)
    parser.add_argument("--release-floor-anchor-before-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--floor-anchor-release-settle-steps", type=int, default=20)
    args = parser.parse_args()
    _apply_strategy_preset(args, sys.argv[1:])
    result = run(args)
    print(json.dumps({"summary": result["summary"], "final": result["final"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
