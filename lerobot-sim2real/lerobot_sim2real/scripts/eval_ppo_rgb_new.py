"""
Evaluation entrypoint with a step-level tracking gate.

This keeps the original eval_ppo_rgb.py intact and only replaces Sim2RealEnv
with a version that can hold the previous action when the real robot is still
far from the previously commanded target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
from typing import Optional

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "eval_ppo_rgb.py"

_spec = importlib.util.spec_from_file_location("eval_ppo_rgb_base", BASE_SCRIPT)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Failed to load base script: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

OriginalSim2RealEnv = base.Sim2RealEnv


class TrackingGateSim2RealEnv(OriginalSim2RealEnv):
    tracking_gate_deg_default: float = 4.0
    tracking_gate_gripper_rad_default: float = 0.03
    tracking_gate_max_hold_steps_default: int = 0
    tracking_gate_debug_default: bool = False

    @classmethod
    def configure_defaults(
        cls,
        tracking_gate_deg: float,
        tracking_gate_gripper_rad: float,
        tracking_gate_max_hold_steps: int,
        tracking_gate_debug: bool,
    ) -> None:
        cls.tracking_gate_deg_default = float(tracking_gate_deg)
        cls.tracking_gate_gripper_rad_default = float(tracking_gate_gripper_rad)
        cls.tracking_gate_max_hold_steps_default = int(tracking_gate_max_hold_steps)
        cls.tracking_gate_debug_default = bool(tracking_gate_debug)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gate_deg = self.tracking_gate_deg_default
        self._gate_gripper_rad = self.tracking_gate_gripper_rad_default
        self._gate_max_hold_steps = self.tracking_gate_max_hold_steps_default
        self._gate_debug = self.tracking_gate_debug_default
        self._last_action = None
        self._hold_steps = 0

    def _get_real_qpos_rad(self):
        robot = self.agent.real_robot
        joint_names = robot._joint_names
        joint_keys = [f"{jn}.pos" for jn in joint_names]
        obs = robot.get_observation()
        qpos_deg = np.array([obs[k] for k in joint_keys], dtype=np.float32)
        qpos_rad = np.deg2rad(qpos_deg)
        gripper = obs.get("gripper.pos", None)
        if gripper is not None and np.isfinite(gripper):
            qpos_rad = np.concatenate([qpos_rad, np.array([float(gripper)], dtype=np.float32)])
        return qpos_rad

    def _get_prev_target_rad(self):
        robot = self.agent.real_robot
        joint_count = len(robot._joint_names)
        drive_targets = self.agent.controller.articulation.drive_targets[0].detach().cpu().numpy()
        target = drive_targets[:joint_count].astype(np.float32, copy=True)
        if getattr(robot.config, "enable_gripper", False) and drive_targets.shape[0] > joint_count:
            target = np.concatenate([target, np.array([float(drive_targets[joint_count])], dtype=np.float32)])
        return target

    def _should_hold(self):
        if self._last_action is None:
            return False, 0.0, 0.0

        real_qpos = self._get_real_qpos_rad()
        prev_target = self._get_prev_target_rad()
        n = min(real_qpos.shape[0], prev_target.shape[0])
        if n <= 0:
            return False, 0.0, 0.0

        err = np.abs(prev_target[:n] - real_qpos[:n])
        if n >= 2:
            arm_err = err[:-1]
            gripper_err = float(err[-1])
        else:
            arm_err = err
            gripper_err = 0.0

        max_arm_err_deg = float(np.rad2deg(np.max(arm_err))) if arm_err.size > 0 else 0.0
        blocked = max_arm_err_deg > self._gate_deg or gripper_err > self._gate_gripper_rad

        if blocked and self._gate_max_hold_steps > 0 and self._hold_steps >= self._gate_max_hold_steps:
            blocked = False

        return blocked, max_arm_err_deg, gripper_err

    def step(self, action):
        action_arr = np.array(action, copy=True) if action is not None else None
        if action_arr is not None:
            blocked, max_arm_err_deg, gripper_err = self._should_hold()
            if blocked:
                action_arr = np.array(self._last_action, copy=True)
                self._hold_steps += 1
                if self._gate_debug:
                    print(
                        "[GateStep] hold previous action",
                        "hold_steps=",
                        self._hold_steps,
                        "max_arm_err_deg=",
                        max_arm_err_deg,
                        "max_gripper_err_rad=",
                        gripper_err,
                    )
            else:
                if self._gate_debug and self._hold_steps > 0:
                    print(f"[GateStep] release new action after {self._hold_steps} hold steps")
                self._hold_steps = 0
            self._last_action = np.array(action_arr, copy=True)

        return super().step(action_arr)


@dataclass
class Args:
    checkpoint: Optional[str] = None
    env_kwargs_json_path: Optional[str] = None
    debug: bool = False
    save_step_debug: bool = False
    debug_save_dir: Optional[str] = None
    continuous_eval: bool = True
    max_episode_steps: int = 100
    num_episodes: Optional[int] = None
    env_id: str = "SO100GraspCube-v1"
    seed: int = 1
    record_dir: Optional[str] = None
    control_freq: Optional[int] = 15
    use_real_robot: bool = True
    tracking_gate_deg: float = 4.0
    tracking_gate_gripper_rad: float = 0.03
    tracking_gate_max_hold_steps: int = 0
    tracking_gate_debug: bool = False


def main(args: Args) -> None:
    TrackingGateSim2RealEnv.configure_defaults(
        tracking_gate_deg=args.tracking_gate_deg,
        tracking_gate_gripper_rad=args.tracking_gate_gripper_rad,
        tracking_gate_max_hold_steps=args.tracking_gate_max_hold_steps,
        tracking_gate_debug=args.tracking_gate_debug,
    )
    base.Sim2RealEnv = TrackingGateSim2RealEnv
    base.main(args)


if __name__ == "__main__":
    args = base.tyro.cli(Args)
    main(args)
