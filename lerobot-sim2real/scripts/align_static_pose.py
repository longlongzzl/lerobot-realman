#!/usr/bin/env python3
"""
Align sim/real visuals at a fixed robot pose without sending actions.

Workflow:
- connect real robot and camera
- read real qpos
- set sim qpos to the same values
- grab sim/real camera frames and overlay for visual check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# ensure repo roots on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MONO_ROOT = REPO_ROOT.parent
for p in (REPO_ROOT, MONO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lerobot_sim2real.config.real_robot import create_real_robot  # noqa: E402
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent  # noqa: E402
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Overlay sim vs real at current real robot pose.")
    parser.add_argument("--env-id", type=str, default="RM75GraspCube_two_cameras-v1")
    parser.add_argument(
        "--env-kwargs-json-path",
        type=str,
        default="env_config.json",
        help="Path to env kwargs JSON (camera pose, fov, etc.)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save overlay image (e.g., overlay.png)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    env_kwargs = dict(
        obs_mode="rgb+segmentation",
        render_mode="human",
        reward_mode="none",
        render_backend="cpu",
        sensor_configs=dict(width=512, height=512),
        domain_randomization=False,
    )
    cfg_path = Path(args.env_kwargs_json_path)
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            env_kwargs.update(json.load(f))
        print("Loaded env_kwargs from JSON:", env_kwargs)

    sim_env = FlattenRGBDObservationWrapper(gym.make(args.env_id, **env_kwargs))
    sim_env.reset()

    real_robot, camera = create_real_robot()
    real_robot.connect(calibrate=False)
    camera.connect()
    real_agent = LeRobotRealAgent(real_robot)

    # sync sim qpos to real qpos (no command sent to real), pad for gripper DOFs
    
    """"""
    real_qpos = real_agent.get_qpos()
    print("Real qpos:", real_qpos)
    sim_qpos = sim_env.unwrapped.agent.robot.get_qpos()
    
    ## FIXME: 我在这里直接设置成了相同的位姿来做测试
    
    if real_qpos.shape[-1] <= sim_qpos.shape[-1]:
        sim_qpos[..., : real_qpos.shape[-1]] = real_qpos
    else:
        sim_qpos = real_qpos
    sim_env.unwrapped.agent.robot.set_qpos(sim_qpos)

    # capture sim and real images
    sim_obs = sim_env.get_obs()["sensor_data"]
    real_agent.capture_sensor_data(["base_camera"])
    real_obs = real_agent.get_sensor_data()["base_camera"]["rgb"][0] / 255
    sim_img = sim_obs["base_camera"]["rgb"][0].cpu() / 255

    real_img = torch.as_tensor(real_obs, dtype=torch.float32)
    sim_img = torch.as_tensor(sim_img, dtype=torch.float32)

    # ensure both are HWC
    if real_img.dim() == 3 and real_img.shape[0] == 3:
        real_img = real_img.permute(1, 2, 0)
    if sim_img.dim() == 3 and sim_img.shape[0] == 3:
        sim_img = sim_img.permute(1, 2, 0)

    # resize real image to sim resolution if needed
    if real_img.shape[-2:] != sim_img.shape[-2:]:
        real_img = (
            F.interpolate(
                real_img.permute(2, 0, 1).unsqueeze(0),
                size=sim_img.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )

    overlay = 0.5 * real_img + 0.5 * sim_img

    plt.figure()
    plt.imshow(overlay)
    plt.title("Static pose overlay")
    if args.save:
        plt.savefig(args.save, bbox_inches="tight")
        print(f"Saved overlay to {args.save}")
    plt.show()

    real_robot.disconnect()


if __name__ == "__main__":
    main()
