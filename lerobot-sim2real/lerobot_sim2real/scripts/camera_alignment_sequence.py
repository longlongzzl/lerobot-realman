import json
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
except (ImportError, NameError):
    # ensure lerobot modules are imported before retrying
    import importlib

    importlib.import_module("lerobot.common.cameras.camera")
    importlib.import_module("lerobot.common.motors.motors_bus")
    importlib.import_module("lerobot.common.robots.robot")
    importlib.import_module("lerobot.common.utils.robot_utils")
    from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent  # type: ignore
import tyro
from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.utils import common
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

from lerobot_sim2real.config.real_robot import create_real_robot
from lerobot_sim2real.utils.safety import setup_safe_exit


# 20-step reference motion captured from eval_ppo_rgb.py (action_lsit).
ACTION_SEQUENCE: Tuple[Tuple[float, ...], ...] = (
    (0.3694, -2.1347, -0.2578, -0.5482, 0.8199, 1.4473, 0.9142, 1.4230),
    (0.2729, -1.8749, -0.2790, -0.0202, 0.5486, 1.3514, 0.5968, 1.1328),
    (0.1735, -1.6763, -0.1769, 0.2643, 0.3212, 1.2976, 0.4252, 0.8556),
    (0.4015, -1.4455, 0.2063, 0.3638, 0.1499, 1.3051, 0.5904, 0.6095),
    (0.5604, -1.4042, 0.4363, 0.4370, -0.0959, 1.3361, 0.6856, 0.4122),
    (0.8529, -1.3798, 0.7073, 0.5414, -0.1896, 1.3681, 0.6420, 0.2823),
    (0.9163, -1.3272, 0.5194, 0.5924, -0.3772, 1.3840, 0.6747, 0.1633),
    (0.7646, -1.4068, 0.1492, 0.1771, -0.8133, 1.3787, 0.8231, 0.1108),
    (0.6540, -1.4609, 0.1093, -0.4046, -0.9498, 1.1138, 0.5919, 0.2605),
    (0.4291, -1.0191, 0.0922, -0.6198, -0.8163, 0.0358, 0.0718, 0.6289),
    (-0.0773, -0.0160, -0.1877, -0.1215, -0.7917, -0.9982, -0.8006, 0.8585),
    (-0.5488, 0.5095, -0.3742, -0.1388, -0.2364, -1.1817, -1.1208, 0.5683),
    (-0.9665, 0.5984, -0.5642, -0.2473, 0.3070, -1.0526, -1.2258, 0.1516),
    (-0.9894, 0.8853, -0.4710, -0.1186, 0.5582, -0.9050, -1.3184, -0.0149),
    (-0.9406, 1.0729, -0.2058, -0.0765, 0.5771, -0.8820, -1.2998, -0.0014),
    (-0.9473, 1.2386, -0.0388, 0.0878, 0.4498, -0.8686, -1.2264, 0.0121),
    (-0.8547, 1.2649, 0.0769, 0.1339, 0.2611, -0.8132, -0.8627, 0.0047),
    (-0.5816, 1.2804, -0.0369, 0.0939, 0.0568, -0.8481, -0.5981, 0.0016),
    (-0.2792, 1.3256, -0.0915, 0.0624, 0.1163, -0.8468, -0.2382, 0.0038),
    (-0.035463, 1.3151, -0.12931, -0.018436, 0.10788, -0.74636, 0.086766, -0.00024263),
)


@dataclass
class Args:
    env_id: str = "RM75GraspCube_two_cameras-v1"
    env_kwargs_json_path: Optional[str] = None
    pause_seconds: float = 2.0
    repeat: bool = False


def _format_action(values: Sequence[float], dim: int) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == dim:
        return flat
    action = np.zeros(dim, dtype=np.float32)
    length = min(dim, flat.size)
    action[:length] = flat[:length]
    return action


def _build_env(args: Args):
    env_kwargs = dict(
        obs_mode="rgb+segmentation",
        render_mode="none",
        reward_mode="none",
        render_backend="cpu",
        sensor_configs=dict(width=1280, height=1280),
        domain_randomization_config=dict(
            max_camera_offset=[0.0, 0.0, 0.0],
            camera_target_noise=0.0,
            camera_view_rot_noise=0.0,
            camera_fov_noise=0.0,
        ),
    )
    if args.env_kwargs_json_path:
        try:
            with open(args.env_kwargs_json_path, "r", encoding="utf-8") as f:
                env_kwargs.update(json.load(f))
        except FileNotFoundError:
            print(f"[WARN] env_kwargs_json_path '{args.env_kwargs_json_path}' not found. Using defaults.")
    env_kwargs.setdefault("domain_randomization_config", {})
    env_kwargs["domain_randomization_config"].update(
        max_camera_offset=[0.0, 0.0, 0.0],
        camera_target_noise=0.0,
        camera_view_rot_noise=0.0,
        camera_fov_noise=0.0,
    )

    sim_env = FlattenRGBDObservationWrapper(gym.make(args.env_id, **env_kwargs))
    real_robot, _ = create_real_robot()
    real_agent = LeRobotRealAgent(real_robot, use_cached_qpos=False)
    real_env = Sim2RealEnv(
        sim_env=sim_env,
        agent=real_agent,
        real_reset_function=lambda self, seed=None, options=None: self.sim_env.reset(
            seed=seed, options=options
        ),
    )
    setup_safe_exit(sim_env, real_env, real_agent)
    real_env.reset()
    return sim_env, real_env, real_agent


def _print_state(real_env, step_idx: int):
    sim_qpos = real_env.base_sim_env.agent.robot.get_qpos().detach().cpu().flatten()
    try:
        real_qpos = common.to_cpu_tensor(real_env.agent.robot.get_qpos()).flatten()
    except Exception:
        real_qpos = None
    sim_deg = torch_to_deg(sim_qpos)
    print(f"[Step {step_idx}] Sim qpos (deg, first 8): {sim_deg[:8].tolist()}")
    if real_qpos is not None:
        real_deg = torch_to_deg(real_qpos)
        print(f"[Step {step_idx}] Real qpos (deg, first 8): {real_deg[:8].tolist()}")
    else:
        print(f"[Step {step_idx}] Real qpos unavailable.")


def torch_to_deg(tensor):
    import torch

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    return torch.rad2deg(tensor)


def main(args: Args):
    sim_env, real_env, _ = _build_env(args)

    action_dim = int(np.prod(real_env.action_space.shape))
    formatted_actions = [_format_action(a, action_dim) for a in ACTION_SEQUENCE]

    loop_idx = 0
    try:
        while True:
            print(f"Starting playback loop {loop_idx + 1}")
            for idx, action in enumerate(formatted_actions):
                print(
                    f"  Executing action {idx + 1}/{len(formatted_actions)} "
                    f"(loop {loop_idx + 1})"
                )
                real_env.step(action)
                _print_state(real_env, idx + 1)
                if args.pause_seconds > 0:
                    time.sleep(args.pause_seconds)
            loop_idx += 1
            if not args.repeat:
                break
    finally:
        print("Motion sequence finished.")


if __name__ == "__main__":
    main(tyro.cli(Args))
