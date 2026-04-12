from __future__ import annotations

import argparse
import math
import socket
import time
from typing import Iterable

from lerobot.common.robots.realman_lerobot import RealManArm, RealManArmConfig


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


def create_robot(ip: str, port: int) -> RealManArm:
    cfg = RealManArmConfig(
        id="rm1",
        ip=ip,
        port=port,
        use_degrees=True,
        v=20,
        r=0,
        connect=0,
        block=1,
        max_relative_target=7,
        enable_gripper=True,
        gripper_normed=True,
        validate_on_connect=True,
    )
    return RealManArm(cfg)


def read_gripper(robot: RealManArm) -> float:
    obs = robot.get_observation()
    grip = obs.get("gripper.pos")
    if grip is None:
        raise RuntimeError("gripper.pos missing from observation")
    return float(grip)


def maybe_set_hand_speed(robot: RealManArm, speed: int | None) -> None:
    if speed is None:
        return
    if robot.arm is None:
        raise RuntimeError("robot.arm is not available after connect")
    if speed < 1 or speed > 1000:
        raise ValueError(f"gripper speed must be in [1, 1000], got {speed}")
    rc = robot.arm.rm_set_hand_speed(int(speed))
    if int(rc) != 0:
        raise RuntimeError(f"rm_set_hand_speed failed, rc={rc}")
    print(f"[GripTest] rm_set_hand_speed applied speed={speed}")


def ensure_command_socket(robot: RealManArm) -> None:
    if robot.client is not None:
        return
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((robot.config.ip, robot.config.port))
    robot.client = sock


def sample_segment(
    robot: RealManArm,
    target: float,
    hold_seconds: float,
    sample_hz: float,
    resend_every_step: bool,
    segment_idx: int,
    start_t: float,
    mode: str,
    prev_target: float,
) -> None:
    dt = 1.0 / sample_hz
    steps = max(1, int(math.ceil(hold_seconds * sample_hz)))

    if mode == "step" and not resend_every_step:
        robot.send_action({"gripper.pos": target})

    for sample_idx in range(steps):
        tick_start = time.perf_counter()
        if mode == "ramp":
            alpha = (sample_idx + 1) / steps
            sample_target = prev_target + alpha * (target - prev_target)
            robot.send_action({"gripper.pos": sample_target})
        else:
            sample_target = target
            if resend_every_step:
                robot.send_action({"gripper.pos": sample_target})

        actual = read_gripper(robot)
        err = sample_target - actual
        now = time.perf_counter() - start_t
        print(
            f"[GripTest] seg={segment_idx:02d} sample={sample_idx:03d} "
            f"t={now:7.3f}s target={sample_target:0.4f} actual={actual:0.4f} err={err:+0.4f}"
        )

        sleep_s = dt - (time.perf_counter() - tick_start)
        if sleep_s > 0:
            time.sleep(sleep_s)


def run_sequence(
    robot: RealManArm,
    targets: Iterable[float],
    hold_seconds: float,
    sample_hz: float,
    warmup_seconds: float,
    resend_every_step: bool,
    mode: str,
) -> None:
    targets = list(targets)
    print(
        f"[GripTest] start targets={list(targets)} hold_seconds={hold_seconds} "
        f"sample_hz={sample_hz} resend_every_step={resend_every_step} mode={mode}"
    )
    initial = read_gripper(robot)
    print(f"[GripTest] initial actual={initial:0.4f}")

    if warmup_seconds > 0:
        time.sleep(warmup_seconds)

    start_t = time.perf_counter()
    prev_target = initial if mode == "ramp" else 0.0
    for segment_idx, target in enumerate(targets):
        sample_segment(
            robot=robot,
            target=target,
            hold_seconds=hold_seconds,
            sample_hz=sample_hz,
            resend_every_step=resend_every_step,
            segment_idx=segment_idx,
            start_t=start_t,
            mode=mode,
            prev_target=prev_target,
        )
        prev_target = target

    final_actual = read_gripper(robot)
    print(f"[GripTest] done final actual={final_actual:0.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure RealMan gripper response under a target sequence."
    )
    parser.add_argument("--ip", default="192.168.101.20")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--targets",
        default="0.0,0.05,0.10,0.0",
        help="Comma-separated gripper targets in the same unit used by gripper.pos (typically 0..0.91).",
    )
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--sample-hz", type=float, default=20.0)
    parser.add_argument("--warmup-seconds", type=float, default=0.5)
    parser.add_argument(
        "--mode",
        choices=("step", "ramp"),
        default="step",
        help="step: hold each target constant for a segment; ramp: linearly interpolate from previous target to the current one over a segment.",
    )
    parser.add_argument(
        "--gripper-speed",
        type=int,
        default=None,
        help="Optional rm_set_hand_speed value to apply after connect.",
    )
    parser.add_argument(
        "--send-once",
        action="store_true",
        help="Send each target once per segment instead of resending every sample.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    targets = parse_targets(args.targets)
    for value in targets:
        if value < 0.0 or value > 0.91:
            raise ValueError(
                f"target {value} is outside expected gripper.pos range [0.0, 0.91]"
            )

    robot = create_robot(args.ip, args.port)
    robot.connect()
    try:
        ensure_command_socket(robot)
        maybe_set_hand_speed(robot, args.gripper_speed)
        run_sequence(
            robot=robot,
            targets=targets,
            hold_seconds=args.hold_seconds,
            sample_hz=args.sample_hz,
            warmup_seconds=args.warmup_seconds,
            resend_every_step=not args.send_once,
            mode=args.mode,
        )
    finally:
        try:
            robot.disconnect()
        except Exception:
            if robot.client is not None:
                try:
                    robot.client.close()
                except Exception:
                    pass
            raise


if __name__ == "__main__":
    main()
