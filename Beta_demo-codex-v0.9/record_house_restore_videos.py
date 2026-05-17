from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib
import numpy as np
import sapien
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from transforms3d.quaternions import axangle2quat, qmult, quat2mat

import jimu_pick_cube_env  # noqa: F401
from jimu_pick_cube_env import PLATE_SIZE, PLATE_THICKNESS, TRIANGLE_HEIGHT, TRIANGLE_THICKNESS, TRIANGLE_WIDTH

matplotlib.use("Agg")


ROLE_COLORS = {
    "floor": "#8c0a0a",
    "right_wall": "#0d47a1",
    "left_wall": "#2e7d32",
    "back_wall": "#b26a00",
    "front_wall": "#6a1b9a",
    "right_roof": "#c62828",
    "left_roof": "#039be5",
    "back_roof": "#43a047",
    "front_roof": "#ef6c00",
}


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _actor_pose(actor) -> tuple[np.ndarray, np.ndarray]:
    return (
        _to_numpy(actor.pose.p).reshape(-1, 3)[0].astype(np.float32),
        _to_numpy(actor.pose.q).reshape(-1, 4)[0].astype(np.float32),
    )


def _zero_velocity(actor) -> None:
    if getattr(actor, "px_body_type", None) != "dynamic":
        return
    actor.set_linear_velocity(np.zeros(3, dtype=np.float32))
    actor.set_angular_velocity(np.zeros(3, dtype=np.float32))


def _pose_with_offset(position: np.ndarray, quaternion: np.ndarray, offset: np.ndarray, yaw_deg: float) -> sapien.Pose:
    yaw = np.asarray(axangle2quat([0.0, 0.0, 1.0], np.deg2rad(yaw_deg)), dtype=np.float32)
    quat = np.asarray(qmult(yaw, quaternion), dtype=np.float32)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-8)
    return sapien.Pose(p=(position + offset).tolist(), q=quat.tolist())


def _find_locked(base_env, role: str):
    for locked in base_env.magnetic_snap.locked_panel_poses:
        if locked.role == role:
            return locked
    raise KeyError(f"role not found: {role}")


def _set_role_pose(base_env, role: str, pose: sapien.Pose) -> None:
    actor = _find_locked(base_env, role).actor
    actor.set_pose(pose)
    _zero_velocity(actor)


def _active_connections_for(base_env, roles: set[str]) -> list[str]:
    items = []
    for active_connection in base_env.magnetic_snap.active_connections:
        if not active_connection.active:
            continue
        connection = active_connection.connection
        if connection.parent in roles or connection.child in roles:
            items.append(
                f"{connection.parent}.{connection.parent_edge}<->{connection.child}.{connection.child_edge}:{connection.mode}"
            )
    return sorted(items)


def _square_vertices() -> tuple[np.ndarray, list[list[int]]]:
    hx = PLATE_SIZE / 2.0
    hy = PLATE_SIZE / 2.0
    hz = PLATE_THICKNESS / 2.0
    vertices = np.asarray(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    faces = [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]
    return vertices, faces


def _triangle_vertices() -> tuple[np.ndarray, list[list[int]]]:
    hw = TRIANGLE_WIDTH / 2.0
    ht = TRIANGLE_THICKNESS / 2.0
    vertices = np.asarray(
        [
            [-hw, -ht, 0.0],
            [hw, -ht, 0.0],
            [0.0, -ht, TRIANGLE_HEIGHT],
            [-hw, ht, 0.0],
            [hw, ht, 0.0],
            [0.0, ht, TRIANGLE_HEIGHT],
        ],
        dtype=np.float32,
    )
    faces = [[0, 1, 2], [3, 5, 4], [0, 3, 4, 1], [1, 4, 5, 2], [2, 5, 3, 0]]
    return vertices, faces


def _transform(vertices: np.ndarray, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    rotation = quat2mat(quaternion)
    return (vertices @ rotation.T + position).astype(np.float32)


def _frame_from_env(base_env, title: str, roles: list[str]) -> np.ndarray:
    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=23, azim=-52)
    ax.set_xlim(-0.12, 0.52)
    ax.set_ylim(-0.38, 0.20)
    ax.set_zlim(0.0, 0.22)
    ax.set_box_aspect((0.64, 0.58, 0.22))
    ax.set_axis_off()
    ax.text2D(0.02, 0.96, title, transform=ax.transAxes, fontsize=12)
    ax.text2D(0.02, 0.91, "removed/restored: " + ", ".join(roles), transform=ax.transAxes, fontsize=9)
    table_x = [-0.15, 0.55, 0.55, -0.15]
    table_y = [-0.42, -0.42, 0.24, 0.24]
    table_z = [0.0, 0.0, 0.0, 0.0]
    table = Poly3DCollection([list(zip(table_x, table_y, table_z))], facecolor="#9c6b3f", alpha=0.22, edgecolor="none")
    ax.add_collection3d(table)

    square_vertices, square_faces = _square_vertices()
    triangle_vertices, triangle_faces = _triangle_vertices()
    for locked in base_env.magnetic_snap.locked_panel_poses:
        if locked.role.startswith("free_triangle"):
            continue
        position, quaternion = _actor_pose(locked.actor)
        if position[2] < -0.5:
            continue
        if "roof" in locked.role or "triangle" in locked.role:
            local_vertices, faces = triangle_vertices, triangle_faces
        else:
            local_vertices, faces = square_vertices, square_faces
        vertices = _transform(local_vertices, position, quaternion)
        polygons = [[vertices[index] for index in face] for face in faces]
        color = ROLE_COLORS.get(locked.role, "#777777")
        poly = Poly3DCollection(polygons, facecolor=color, edgecolor="#333333", linewidth=0.35, alpha=0.88)
        ax.add_collection3d(poly)

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return frame


def _write_frame(writer, base_env, title: str, roles: list[str], repeat: int = 1) -> None:
    frame = _frame_from_env(base_env, title, roles)
    for _ in range(repeat):
        writer.append_data(frame)


def _step_and_record(env, writer, action, steps: int, sample_every: int, title: str, roles: list[str]) -> None:
    for step in range(steps):
        env.step(action)
        if step % sample_every == 0:
            _write_frame(writer, env.unwrapped, title, roles)


def _scenario_offsets(rng: np.random.Generator, count: int, yaw_deg: float, offset_scale: float) -> list[tuple[np.ndarray, float]]:
    offsets = []
    for index in range(count):
        xy = rng.uniform([-0.018, -0.018], [0.018, 0.018]).astype(np.float32) * offset_scale
        z = np.float32(rng.uniform(-0.002, 0.006) * offset_scale)
        yaw = float(rng.uniform(-yaw_deg, yaw_deg))
        offsets.append((np.asarray([xy[0], xy[1], z], dtype=np.float32), yaw))
    return offsets


def record_scenario(
    *,
    env,
    name: str,
    roles: list[str],
    out_dir: Path,
    rng: np.random.Generator,
    show_steps: int,
    far_steps: int,
    near_steps: int,
    sample_every: int,
    fps: int,
    yaw_deg: float,
    offset_scale: float,
) -> dict[str, Any]:
    env.reset()
    base_env = env.unwrapped
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    original = {
        role: (
            _find_locked(base_env, role).position.astype(np.float32),
            _find_locked(base_env, role).quaternion.astype(np.float32),
        )
        for role in roles
    }
    out_path = out_dir / f"{name}.mp4"
    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, macro_block_size=8) as writer:
        _write_frame(writer, base_env, f"{name}: complete house", roles, repeat=fps)
        _step_and_record(env, writer, action, show_steps, sample_every, f"{name}: complete house", roles)

        for index, role in enumerate(roles):
            position, quaternion = original[role]
            far_offset = np.asarray([0.34 + 0.09 * index, -0.30 - 0.05 * index, 0.08 + 0.025 * index], dtype=np.float32)
            _set_role_pose(base_env, role, _pose_with_offset(position, quaternion, far_offset, yaw_deg * (index + 1)))
        _write_frame(writer, base_env, f"{name}: moved far", roles, repeat=fps // 2)
        _step_and_record(env, writer, action, far_steps, sample_every, f"{name}: moved far", roles)
        far_connections = _active_connections_for(base_env, set(roles))

        near_offsets = _scenario_offsets(rng, len(roles), yaw_deg, offset_scale)
        for role, (offset, yaw) in zip(roles, near_offsets):
            position, quaternion = original[role]
            _set_role_pose(base_env, role, _pose_with_offset(position, quaternion, offset, yaw))
        _write_frame(writer, base_env, f"{name}: returned with offset", roles, repeat=fps // 2)
        _step_and_record(env, writer, action, near_steps, sample_every, f"{name}: magnetic correction", roles)
        _write_frame(writer, base_env, f"{name}: final", roles, repeat=fps)
        near_connections = _active_connections_for(base_env, set(roles))

    return {
        "name": name,
        "roles": roles,
        "video": str(out_path),
        "far_connections": far_connections,
        "near_connections": near_connections,
        "near_offsets": [
            {"role": role, "offset": offset.round(4).tolist(), "yaw_deg": round(float(yaw), 3)}
            for role, (offset, yaw) in zip(roles, near_offsets)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "videos"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--show-steps", type=int, default=40)
    parser.add_argument("--far-steps", type=int, default=70)
    parser.add_argument("--near-steps", type=int, default=220)
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--yaw-deg", type=float, default=18.0)
    parser.add_argument("--offset-scale", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    fixed_scenarios = [
        ("restore_one_wall", ["right_wall"]),
        ("restore_two_walls", ["right_wall", "back_wall"]),
        ("restore_three_walls", ["right_wall", "back_wall", "front_wall"]),
    ]
    side_roles = ["right_wall", "left_wall", "back_wall", "front_wall"]
    random_count = int(rng.integers(1, 4))
    random_roles = rng.choice(side_roles, size=random_count, replace=False).tolist()
    scenarios = [*fixed_scenarios, ("restore_random_walls", random_roles)]

    env = gym.make(
        "JimuPickCube-v1",
        obs_mode="state",
        render_mode=None,
        control_mode="pd_joint_pos",
        robot_uids="panda",
        assembly_mode="house",
        magnet_mode="edge_pair_drive",
        num_envs=1,
        max_episode_steps=100000,
    )
    results = []
    try:
        for name, roles in scenarios:
            print(f"[record] {name}: {roles}")
            results.append(
                record_scenario(
                    env=env,
                    name=name,
                    roles=roles,
                    out_dir=out_dir,
                    rng=rng,
                    show_steps=args.show_steps,
                    far_steps=args.far_steps,
                    near_steps=args.near_steps,
                    sample_every=args.sample_every,
                    fps=args.fps,
                    yaw_deg=args.yaw_deg,
                    offset_scale=args.offset_scale,
                )
            )
    finally:
        env.close()

    summary_path = out_dir / "restore_video_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "summary": str(summary_path), "videos": [item["video"] for item in results]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
