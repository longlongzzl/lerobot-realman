from __future__ import annotations

import argparse
import math
import time

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np

from mani_skill.utils import gym_utils


def parse_targets(text: str) -> list[float]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("target sequence is empty")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure simulated gripper response under a target sequence."
    )
    parser.add_argument("--env-id", default="RM75GraspCube_two_cameras-v1")
    parser.add_argument("--control-mode", default="pd_joint_pos")
    # This digital-twin task expects image-style observations and unconditionally
    # reads obs["sensor_data"] during reset/step, so a state-only obs_mode will fail.
    parser.add_argument("--obs-mode", default="rgb+segmentation")
    parser.add_argument("--reward-mode", default="none")
    parser.add_argument("--render-mode", default="none")
    parser.add_argument("--targets", default="0.0,0.30,0.0")
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--sample-hz", type=float, default=20.0)
    parser.add_argument(
        "--mode",
        choices=("step", "ramp"),
        default="ramp",
        help="step: hold each target constant for a segment; ramp: linearly interpolate over the segment.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def make_env(args: argparse.Namespace):
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        render_mode=args.render_mode,
        control_mode=args.control_mode,
    )
    env.reset(seed=args.seed)
    return env


def get_gripper_handles(env):
    controller = env.unwrapped.agent.controller
    if not hasattr(controller, "controllers") or "gripper" not in controller.controllers:
        raise RuntimeError("Combined controller does not expose a 'gripper' sub-controller")
    gripper_ctrl = controller.controllers["gripper"]
    if not hasattr(gripper_ctrl, "control_joint_indices") or len(gripper_ctrl.control_joint_indices) != 1:
        raise RuntimeError("Expected a 1-DOF gripper mimic controller")
    gripper_idx = int(gripper_ctrl.control_joint_indices[0].item())
    start, end = controller.action_mapping["gripper"]
    return controller, gripper_ctrl, gripper_idx, start, end


def encode_gripper_target(gripper_ctrl, target: float) -> float:
    if not hasattr(gripper_ctrl, "action_space_low") or not hasattr(gripper_ctrl, "action_space_high"):
        raise RuntimeError("gripper controller does not expose normalized action bounds")
    low = gripper_ctrl.action_space_low.detach().cpu().numpy()
    high = gripper_ctrl.action_space_high.detach().cpu().numpy()
    norm = gym_utils.inv_scale_action(np.array([target], dtype=np.float32), low, high)
    return float(np.clip(norm[0], -1.0, 1.0))


def current_actual_target(env, gripper_ctrl, gripper_idx) -> tuple[float, float]:
    # For mimic controllers, control_joint_indices are local to the gripper sub-controller's
    # own qpos/_target_qpos tensors, not the full robot qpos indexing.
    actual = float(gripper_ctrl.qpos[0, gripper_idx].detach().cpu().item())
    target = float(gripper_ctrl._target_qpos[0, gripper_idx].detach().cpu().item())
    return actual, target


def build_action(controller, gripper_slice, gripper_norm: float) -> np.ndarray:
    action_dim = controller.single_action_space.shape[0]
    action = np.zeros((action_dim,), dtype=np.float32)
    start, end = gripper_slice
    action[start:end] = gripper_norm
    return action


def run_sequence(env, targets: list[float], hold_seconds: float, sample_hz: float, mode: str) -> None:
    controller, gripper_ctrl, gripper_idx, start, end = get_gripper_handles(env)
    initial_actual, initial_target = current_actual_target(env, gripper_ctrl, gripper_idx)
    print(
        f"[SimGripTest] start targets={targets} hold_seconds={hold_seconds} "
        f"sample_hz={sample_hz} mode={mode} control_mode={env.unwrapped.control_mode}"
    )
    print(
        f"[SimGripTest] initial actual={initial_actual:0.4f} "
        f"target={initial_target:0.4f} idx={gripper_idx}"
    )

    dt = 1.0 / sample_hz
    start_t = time.perf_counter()
    prev_target = initial_actual if mode == "ramp" else targets[0]

    for segment_idx, segment_target in enumerate(targets):
        steps = max(1, int(math.ceil(hold_seconds * sample_hz)))
        if mode == "step":
            sample_targets = [segment_target] * steps
        else:
            sample_targets = [
                prev_target + ((i + 1) / steps) * (segment_target - prev_target)
                for i in range(steps)
            ]

        for sample_idx, sample_target in enumerate(sample_targets):
            tick_start = time.perf_counter()
            gripper_norm = encode_gripper_target(gripper_ctrl, sample_target)
            action = build_action(controller, (start, end), gripper_norm)
            env.step(action)
            actual, internal_target = current_actual_target(env, gripper_ctrl, gripper_idx)
            err = internal_target - actual
            now = time.perf_counter() - start_t
            print(
                f"[SimGripTest] seg={segment_idx:02d} sample={sample_idx:03d} "
                f"t={now:7.3f}s cmd={sample_target:0.4f} "
                f"target={internal_target:0.4f} actual={actual:0.4f} err={err:+0.4f}"
            )
            sleep_s = dt - (time.perf_counter() - tick_start)
            if sleep_s > 0:
                time.sleep(sleep_s)

        prev_target = segment_target

    final_actual, final_target = current_actual_target(env, gripper_ctrl, gripper_idx)
    print(
        f"[SimGripTest] done final actual={final_actual:0.4f} "
        f"target={final_target:0.4f}"
    )


def main() -> None:
    args = build_parser().parse_args()
    targets = parse_targets(args.targets)
    for value in targets:
        if value < 0.0 or value > 0.91:
            raise ValueError(f"target {value} is outside expected gripper range [0.0, 0.91]")

    env = make_env(args)
    try:
        run_sequence(
            env=env,
            targets=targets,
            hold_seconds=args.hold_seconds,
            sample_hz=args.sample_hz,
            mode=args.mode,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
