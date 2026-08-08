#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    app_root = Path(__file__).resolve().parents[2]
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))
    legacy_script = app_root.parent / "pick_jiaobang" / "wrist_gripper_object_relation.py"
    if not legacy_script.exists():
        raise FileNotFoundError(f"legacy wrist relation script not found: {legacy_script}")
    legacy_dir = legacy_script.parent
    if str(legacy_dir) not in sys.path:
        sys.path.insert(0, str(legacy_dir))
    runpy.run_path(str(legacy_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
