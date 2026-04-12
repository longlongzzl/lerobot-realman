#!/usr/bin/env python3
"""
Read and print the current RealMan joint positions via RealManArm.get_observation().
Uses the existing config in lerobot_sim2real/config/real_robot.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent  # /.../lerobot-sim2real
MONO_ROOT = REPO_ROOT.parent   # /.../lerobot
for p in (REPO_ROOT, MONO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lerobot_sim2real.config.real_robot import create_real_robot  # noqa: E402


def main():
    robot, camera = create_real_robot(auto_connect=False)
    robot.connect()
    try:
        obs = robot.get_observation()
        print("Raw observation keys:", list(obs.keys()))
        joint_keys = [k for k in obs if k.endswith(".pos")]
        print("Joint positions:")
        for k in joint_keys:
            print(f"  {k}: {obs[k]}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
