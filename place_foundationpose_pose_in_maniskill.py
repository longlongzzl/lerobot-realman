#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
from transforms3d.quaternions import mat2quat

from mani_skill.utils.structs.pose import Pose


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a FoundationPose pose and place the ManiSkill target object at that pose."
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="RM75GraspCube_two_cameras-v1",
        help="Gym environment id.",
    )
    parser.add_argument(
        "--env-kwargs-json-path",
        type=str,
        default="/home/zhangzhao/Desktop/lerobot/lerobot-sim2real/so101_env_config.json",
        help="Environment kwargs JSON path.",
    )
    parser.add_argument(
        "--foundationpose-pose-path",
        type=str,
        required=True,
        help="Path to FoundationPose ob_in_cam txt file, e.g. debug/ob_in_cam/000000.txt",
    )
    parser.add_argument(
        "--camera-extrinsic-opencv-path",
        type=str,
        default="/home/zhangzhao/Desktop/lerobot-sim2real/results/realman/realman_home/base_camera/camera_extrinsic_opencv.npy",
        help="Path to camera_extrinsic_opencv.npy, interpreted as T_base_cam.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        help="Render mode. Use human to inspect placement.",
    )
    parser.add_argument(
        "--shader-pack",
        type=str,
        default="default",
        help="Viewer shader pack for human render mode.",
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="rgb+segmentation",
        help="Observation mode.",
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="none",
        help="Reward mode.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Sensor width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Sensor height.",
    )
    parser.add_argument(
        "--hold-open",
        action="store_true",
        default=True,
        help="Keep viewer open until Ctrl+C.",
    )
    parser.add_argument(
        "--idle-render-fps",
        type=float,
        default=10.0,
        help="Viewer refresh FPS while holding.",
    )
    parser.add_argument(
        "--position-offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help="Optional xyz offset added in base frame after conversion.",
    )
    parser.add_argument(
        "--lock-object-z-to-table",
        action="store_true",
        help="Override Z using cube half height, useful if FoundationPose Z is noisy.",
    )
    return parser.parse_args()


def load_env_kwargs(args) -> dict:
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        render_mode=args.render_mode,
        reward_mode=args.reward_mode,
        render_backend="cpu",
        sensor_configs=dict(width=args.width, height=args.height),
        domain_randomization=False,
    )
    if args.render_mode == "human":
        env_kwargs["viewer_camera_configs"] = dict(shader_pack=args.shader_pack)

    cfg_path = Path(args.env_kwargs_json_path)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            env_kwargs.update(json.load(f))
    return env_kwargs


def load_matrix(path: str) -> np.ndarray:
    path_obj = Path(path)
    if path_obj.suffix == ".npy":
        mat = np.load(path_obj)
    else:
        mat = np.loadtxt(path_obj)
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix from {path}, got {mat.shape}")
    return mat


def matrix_to_pose(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = mat[:3, 3].astype(np.float32)
    quat_wxyz = mat2quat(mat[:3, :3]).astype(np.float32)
    return pos, quat_wxyz


def main():
    args = parse_args()

    T_cam_obj = load_matrix(args.foundationpose_pose_path)
    T_base_cam = load_matrix(args.camera_extrinsic_opencv_path)
    T_base_obj = T_base_cam @ T_cam_obj
    T_base_obj[:3, 3] += np.asarray(args.position_offset, dtype=np.float64)

    pos, quat_wxyz = matrix_to_pose(T_base_obj)

    print("Loaded T_cam_obj from FoundationPose:")
    print(T_cam_obj)
    print("\nLoaded T_base_cam from hand-eye calibration:")
    print(T_base_cam)
    print("\nComputed T_base_obj = T_base_cam @ T_cam_obj:")
    print(T_base_obj)
    print("\nTarget object position in base frame:", pos.tolist())
    print("Target object quaternion [w, x, y, z]:", quat_wxyz.tolist())

    env_kwargs = load_env_kwargs(args)
    env = gym.make(args.env_id, **env_kwargs)
    try:
        obs, info = env.reset()

        sim_env = env.unwrapped
        cube = getattr(sim_env, "cube", None)
        if cube is None:
            raise AttributeError(
                f"Environment {args.env_id} does not expose env.unwrapped.cube; "
                "this helper currently supports cube-based tasks."
            )

        if args.lock_object_z_to_table:
            cube_half_sizes = getattr(sim_env, "cube_half_sizes", None)
            if cube_half_sizes is not None:
                if hasattr(cube_half_sizes, "detach"):
                    cube_half_sizes = cube_half_sizes.detach().cpu().numpy()
                cube_half_size = float(np.asarray(cube_half_sizes).reshape(-1)[0])
                pos[2] = cube_half_size
                print(f"Locked object z to cube half size: {cube_half_size}")

        cube_pose = Pose.create_from_pq(p=pos, q=quat_wxyz)
        cube.set_pose(cube_pose)
        print("Placed ManiSkill cube at FoundationPose-derived pose.")

        if args.render_mode == "human" and args.hold_open:
            print("Viewer is being kept open. Press Ctrl+C to exit.")
            try:
                while True:
                    env.render()
                    if args.idle_render_fps > 0:
                        time.sleep(1.0 / args.idle_render_fps)
            except KeyboardInterrupt:
                print("Interrupted, closing viewer.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
