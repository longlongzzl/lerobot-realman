from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import sapien
from transforms3d.quaternions import axangle2quat, qmult

import jimu_pick_cube_env  # noqa: F401
from jimu_pick_cube_env import PLATE_THICKNESS, TRIANGLE_THICKNESS
from record_house_restore_videos import _actor_pose
from record_house_restore_videos import _find_locked
from record_house_restore_videos import _set_role_pose


ASSEMBLY_ORDER = [
    "floor",
    "right_wall",
    "left_wall",
    "back_wall",
    "front_wall",
    "right_roof",
    "left_roof",
    "back_roof",
    "front_roof",
]

SQUARE_ROLES = {"floor", "right_wall", "left_wall", "back_wall", "front_wall"}


def _zero_action(env) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)


def _normalize_quat(quaternion: np.ndarray) -> np.ndarray:
    return quaternion / max(float(np.linalg.norm(quaternion)), 1e-8)


def _nlerp_quat(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    target = end
    if float(np.dot(start, end)) < 0.0:
        target = -end
    return _normalize_quat((1.0 - alpha) * start + alpha * target)


def _pose_to_arrays(pose: sapien.Pose) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(pose.p, dtype=np.float32), np.asarray(pose.q, dtype=np.float32)


def _pose_with_yaw(position: np.ndarray, quaternion: np.ndarray, yaw_deg: float) -> sapien.Pose:
    yaw = np.asarray(axangle2quat([0.0, 0.0, 1.0], np.deg2rad(yaw_deg)), dtype=np.float32)
    quat = _normalize_quat(np.asarray(qmult(yaw, quaternion), dtype=np.float32))
    return sapien.Pose(p=position.tolist(), q=quat.tolist())


def _zero_velocity(base_env, role: str) -> None:
    actor = _find_locked(base_env, role).actor
    actor.set_linear_velocity(np.zeros(3, dtype=np.float32))
    actor.set_angular_velocity(np.zeros(3, dtype=np.float32))


def _frame(env) -> np.ndarray:
    frame = env.render()
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    return frame.astype(np.uint8)


def _append_frame(writer, env, repeat: int = 1) -> None:
    frame = _frame(env)
    for _ in range(repeat):
        writer.append_data(frame)


def _step_record(env, writer, steps: int, sample_every: int) -> None:
    action = _zero_action(env)
    for step in range(steps):
        env.step(action)
        if step % sample_every == 0:
            _append_frame(writer, env)


def _move_role(
    *,
    env,
    writer,
    role: str,
    target: sapien.Pose,
    steps: int,
    sample_every: int,
) -> None:
    base_env = env.unwrapped
    actor = _find_locked(base_env, role).actor
    start_p, start_q = _actor_pose(actor)
    target_p, target_q = _pose_to_arrays(target)
    action = _zero_action(env)
    for step in range(steps):
        alpha = min(1.0, float(step + 1) / float(max(steps, 1)))
        position = (1.0 - alpha) * start_p + alpha * target_p
        quaternion = _nlerp_quat(start_q, target_q, alpha)
        _set_role_pose(base_env, role, sapien.Pose(p=position.tolist(), q=quaternion.tolist()))
        _zero_velocity(base_env, role)
        env.step(action)
        if step % sample_every == 0:
            _append_frame(writer, env)


def _lifted_pose_from_current(base_env, role: str, lift_height: float) -> sapien.Pose:
    position, quaternion = _actor_pose(_find_locked(base_env, role).actor)
    position[2] = max(float(position[2]), lift_height)
    return sapien.Pose(p=position.tolist(), q=quaternion.tolist())


def _lifted_target(target: sapien.Pose, lift_height: float) -> sapien.Pose:
    position, quaternion = _pose_to_arrays(target)
    position = position.copy()
    position[2] = max(float(position[2]), lift_height)
    return sapien.Pose(p=position.tolist(), q=quaternion.tolist())


def _place_role_via_clearance_path(
    *,
    env,
    writer,
    role: str,
    target: sapien.Pose,
    segment_steps: int,
    sample_every: int,
    lift_height: float,
) -> None:
    base_env = env.unwrapped
    _move_role(
        env=env,
        writer=writer,
        role=role,
        target=_lifted_pose_from_current(base_env, role, lift_height),
        steps=segment_steps,
        sample_every=sample_every,
    )
    _move_role(
        env=env,
        writer=writer,
        role=role,
        target=_lifted_target(target, lift_height),
        steps=segment_steps,
        sample_every=sample_every,
    )
    _move_role(
        env=env,
        writer=writer,
        role=role,
        target=target,
        steps=segment_steps,
        sample_every=sample_every,
    )


def _disable_all_connections(base_env) -> None:
    snap = base_env.magnetic_snap
    snap.disabled_roles.clear()
    snap.suspended_roles.clear()
    for active_connection in snap.active_connections:
        active_connection.active = False
        for drive in active_connection.drives:
            snap._disable_drive(drive)


def _scatter_pose(role: str, index: int) -> sapien.Pose:
    grid_positions = [
        (-0.26, -0.28),
        (-0.13, -0.28),
        (0.00, -0.28),
        (0.13, -0.28),
        (0.26, -0.28),
        (-0.20, 0.12),
        (-0.07, 0.12),
        (0.07, 0.12),
        (0.20, 0.12),
    ]
    x, y = grid_positions[index]
    yaw = np.asarray(axangle2quat([0.0, 0.0, 1.0], np.deg2rad((index % 5 - 2) * 13.0)), dtype=np.float32)
    if role in SQUARE_ROLES:
        z = PLATE_THICKNESS / 2.0
        quat = yaw
    else:
        lay_flat = np.asarray(axangle2quat([1.0, 0.0, 0.0], np.deg2rad(90.0)), dtype=np.float32)
        quat = _normalize_quat(np.asarray(qmult(yaw, lay_flat), dtype=np.float32))
        z = TRIANGLE_THICKNESS / 2.0
    return sapien.Pose(p=[x, y, z], q=quat.tolist())


def _scatter_all_pieces(base_env) -> None:
    _disable_all_connections(base_env)
    for index, role in enumerate(ASSEMBLY_ORDER):
        _set_role_pose(base_env, role, _scatter_pose(role, index))
        _zero_velocity(base_env, role)
    base_env.magnetic_snap.disabled_roles = set(ASSEMBLY_ORDER)
    base_env.magnetic_snap.suspended_roles.clear()


def _target_pose_from_house(base_env, role: str) -> sapien.Pose:
    locked = _find_locked(base_env, role)
    return sapien.Pose(p=locked.position.tolist(), q=locked.quaternion.tolist())


def _target_with_small_offset(target: sapien.Pose, role: str, index: int) -> sapien.Pose:
    return target


def _enable_role_for_magnetism(base_env, role: str) -> None:
    snap = base_env.magnetic_snap
    snap.disabled_roles.discard(role)
    snap.suspended_roles.discard(role)


def _activate_nearby_predefined_connections(base_env, placed_roles: set[str], threshold: float = 0.035) -> None:
    snap = base_env.magnetic_snap
    for active_connection in snap.active_connections:
        connection = active_connection.connection
        if connection.parent not in placed_roles or connection.child not in placed_roles:
            continue
        if active_connection.active:
            continue
        if snap._connection_point_error(connection) > threshold:
            continue
        active_connection.active = True
        for drive in active_connection.drives:
            snap._configure_drive(drive)


def _align_placed_roles_to_targets(base_env, targets: dict[str, sapien.Pose], placed_roles: list[str]) -> None:
    for role in placed_roles:
        _set_role_pose(base_env, role, targets[role])
        _zero_velocity(base_env, role)


def _collect_stage_report(base_env, placed_roles: list[str]) -> dict[str, Any]:
    snap = base_env.magnetic_snap
    errors = snap.connection_errors()
    return {
        "placed_roles": placed_roles,
        "active_connection_count": sum(1 for item in snap.active_connections if item.active),
        "max_connection_error": float(errors["max_point_error"]),
        "mean_connection_error": float(errors["mean_point_error"]),
        "suspended_roles": sorted(snap.suspended_roles),
        "active_counts": {role: snap._active_connection_count(role) for role in ASSEMBLY_ORDER},
    }


def record_stepwise_assembly(
    *,
    out_path: Path,
    summary_path: Path,
    fps: int,
    segment_steps: int,
    settle_steps: int,
    initial_steps: int,
    sample_every: int,
    lift_height: float,
) -> dict[str, Any]:
    env = gym.make(
        "JimuPickCube-v1",
        obs_mode="state",
        render_mode="rgb_array",
        render_backend="cpu",
        control_mode="pd_joint_pos",
        robot_uids="panda",
        assembly_mode="house",
        magnet_mode="edge_pair_drive",
        num_envs=1,
        max_episode_steps=100000,
    )
    reports = []
    try:
        env.reset()
        base_env = env.unwrapped
        targets = {role: _target_pose_from_house(base_env, role) for role in ASSEMBLY_ORDER}
        _scatter_all_pieces(base_env)
        action = _zero_action(env)
        with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, macro_block_size=8) as writer:
            for _ in range(fps):
                env.step(action)
                _append_frame(writer, env)
            _step_record(env, writer, initial_steps, sample_every)
            placed_roles = []
            for index, role in enumerate(ASSEMBLY_ORDER):
                print(f"[assemble] place {index + 1}/9 {role}")
                target = _target_with_small_offset(targets[role], role, index)
                _place_role_via_clearance_path(
                    env=env,
                    writer=writer,
                    role=role,
                    target=target,
                    segment_steps=segment_steps,
                    sample_every=sample_every,
                    lift_height=lift_height,
                )
                _set_role_pose(base_env, role, targets[role])
                _zero_velocity(base_env, role)
                _enable_role_for_magnetism(base_env, role)
                placed_roles.append(role)
                _align_placed_roles_to_targets(base_env, targets, placed_roles)
                _activate_nearby_predefined_connections(base_env, set(placed_roles))
                _step_record(env, writer, settle_steps, 1)
                reports.append(_collect_stage_report(base_env, placed_roles.copy()))
            for _ in range(fps):
                _append_frame(writer, env)
    finally:
        env.close()

    payload = {
        "video": str(out_path),
        "fps": fps,
        "assembly_order": ASSEMBLY_ORDER,
        "settle_steps_after_each_piece": settle_steps,
        "stage_reports": reports,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "videos_stepwise_sim"))
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--segment-steps", type=int, default=28)
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--initial-steps", type=int, default=40)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--lift-height", type=float, default=0.24)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stepwise_house_assembly_sim.mp4"
    summary_path = out_dir / "stepwise_house_assembly_summary.json"
    payload = record_stepwise_assembly(
        out_path=out_path,
        summary_path=summary_path,
        fps=args.fps,
        segment_steps=args.segment_steps,
        settle_steps=args.settle_steps,
        initial_steps=args.initial_steps,
        sample_every=args.sample_every,
        lift_height=args.lift_height,
    )
    print(json.dumps({"video": payload["video"], "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
