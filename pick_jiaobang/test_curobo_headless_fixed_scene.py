#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

DEFAULT_PYTHON = Path("/home/zhangzhao/anaconda3/envs/foundationpose310/bin/python")
DEFAULT_RUNNER = THIS_DIR / "test_targeted_fixed_scene.py"
DEFAULT_ENTRYPOINT = THIS_DIR / "rm75_curobo_direct_fullchain.py"
DEFAULT_SCENE = THIS_DIR / "test_scenes" / "gluestick_desk_regression.json"
DEFAULT_ROBOT_CFG = THIS_DIR / "curobo_rm75_config" / "rm75.yml"

DEFAULT_OBJECTS = [
    "hongshupian",
    "gluestick",
    "shuazi",
    "lvmukuai",
    "tennis",
    "carriot",
    "bi",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Headless fixed-scene cuRobo pick-place regression runner. "
            "This intentionally disables human rendering and runs each object in a child process."
        )
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON, help="Python executable for the runner and child planner.")
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER, help="Batch runner script.")
    parser.add_argument("--entrypoint", type=Path, default=DEFAULT_ENTRYPOINT, help="cuRobo pick-place entrypoint.")
    parser.add_argument("--scene-file", type=Path, default=DEFAULT_SCENE, help="Fixed scene JSON file.")
    parser.add_argument("--robot-cfg", type=Path, default=DEFAULT_ROBOT_CFG, help="cuRobo RM75 robot config YAML.")
    parser.add_argument(
        "--objects",
        nargs="*",
        default=DEFAULT_OBJECTS,
        help="Objects to test. Defaults to the full fixed-scene sequence.",
    )
    parser.add_argument("--per-object-timeout", type=float, default=360.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    parser.add_argument("--post-final-grace-seconds", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-child-logs", action="store_true", help="Stream planner logs from each child process.")
    parser.add_argument("--curobo-debug", action="store_true", help="Pass --curobo-debug to the child planner.")
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments appended to the child planner. Put this option last.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = args.python.expanduser().resolve()
    runner = args.runner.expanduser().resolve()
    entrypoint = args.entrypoint.expanduser().resolve()
    scene_file = args.scene_file.expanduser().resolve()
    robot_cfg = args.robot_cfg.expanduser().resolve()

    cmd = [
        str(python),
        str(runner),
        "--child-python",
        str(python),
        "--entrypoint",
        str(entrypoint),
        "--scene-file",
        str(scene_file),
        "--render-mode",
        "none",
        "--seed",
        str(int(args.seed)),
        "--per-object-timeout",
        str(float(args.per_object_timeout)),
        "--heartbeat-seconds",
        str(float(args.heartbeat_seconds)),
        "--post-final-grace-seconds",
        str(float(args.post_final_grace_seconds)),
        "--objects",
        *[str(obj) for obj in list(args.objects or [])],
        "--curobo-rm75-robot-cfg",
        str(robot_cfg),
        "--trajectory-preview-sleep",
        "0",
    ]
    if args.print_child_logs:
        cmd.append("--print-child-logs")
    if args.curobo_debug:
        cmd.append("--curobo-debug")
    cmd.extend(str(item) for item in list(args.extra or []))

    print("[headless-test] running:")
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
