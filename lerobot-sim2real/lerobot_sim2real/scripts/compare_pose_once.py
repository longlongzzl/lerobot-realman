#!/usr/bin/env python3
"""
Send the same joint targets to sim and real robot, then overlay their camera views once.
Use this to visually check alignment for a single pose.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import torch

# ensure repo roots are on PYTHONPATH when running standalone
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]  # /.../lerobot-sim2real
MONO_ROOT = REPO_ROOT.parent       # /.../lerobot
for p in (REPO_ROOT, MONO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lerobot_sim2real.config.real_robot import create_real_robot
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.utils.visualization.misc import tile_images
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper


def overlay_envs(sim_env, real_env):
    real_obs = real_env.get_obs()["sensor_data"]
    sim_obs = sim_env.get_obs()["sensor_data"]
    assert sorted(real_obs.keys()) == sorted(
        sim_obs.keys()
    ), f"real camera names {real_obs.keys()} and sim camera names {sim_obs.keys()} differ"

    overlaid_dict = sim_env.get_obs()["sensor_data"]
    overlaid_imgs = []
    for name in overlaid_dict:
        real_imgs = real_obs[name]["rgb"][0] / 255
        sim_imgs = overlaid_dict[name]["rgb"][0].cpu() / 255
        overlaid_imgs.append(0.5 * real_imgs + 0.5 * sim_imgs)
    return tile_images(overlaid_imgs)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare sim vs real at a fixed qpos.")
    parser.add_argument(
        "--env-id",
        type=str,
        default="RM75GraspCube_two_cameras-v1",
        help="Gym env id to create.",
    )
    parser.add_argument(
        "--env-kwargs-json-path",
        type=str,
        default="env_config.json",
        help="Path to JSON for extra env kwargs (camera pose etc.).",
    )
    parser.add_argument(
        "--target-qpos",
        type=float,
        nargs="+",
        required=True,
        help="Joint targets (radians) to send to sim and real robot.",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip real robot calibration on connect.",
    )
    parser.add_argument(
        "--save-overlay",
        type=str,
        default=None,
        help="Optional path to save the overlay image (e.g., overlay.png).",
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
    else:
        print(f"Config {cfg_path} not found, using defaults.")

    sim_env = gym.make(args.env_id, **env_kwargs)
    sim_env = FlattenRGBDObservationWrapper(sim_env)

    real_robot, camera = create_real_robot()
    real_robot.connect(calibrate=not args.no_calibrate)
    camera.connect()
    real_agent = LeRobotRealAgent(real_robot)

    def real_reset_function(self: Sim2RealEnv, seed=None, options=None):
        # Keep real robot as-is; just reset sim and copy real qpos into sim.
        self.sim_env.reset(seed=seed, options=options)
        real_qpos = self.agent.qpos
        sim_qpos = self.base_sim_env.agent.robot.get_qpos()
        if real_qpos.shape[-1] <= sim_qpos.shape[-1]:
            sim_qpos[..., : real_qpos.shape[-1]] = real_qpos
        else:
            sim_qpos = real_qpos
        self.base_sim_env.agent.robot.set_qpos(sim_qpos)

    real_env = Sim2RealEnv(
        sim_env=sim_env,
        agent=real_agent,
        real_reset_function=real_reset_function,
    )

    real_env.reset()

    target_qpos = torch.tensor(args.target_qpos, dtype=torch.float32).unsqueeze(0)
    sim_env.unwrapped.agent.robot.set_qpos(target_qpos)
    real_agent.set_target_qpos(target_qpos)
    time.sleep(0.5)  # give real robot a moment to reach the target

    overlaid = overlay_envs(sim_env, real_env)
    plt.figure()
    plt.imshow(overlaid)
    plt.title("Sim vs Real Overlay")
    if args.save_overlay:
        plt.savefig(args.save_overlay, bbox_inches="tight")
        print(f"Saved overlay to {args.save_overlay}")
    plt.show()


if __name__ == "__main__":
    main()
