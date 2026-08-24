from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rm75_app.paths import RUNTIME_DIR
from rm75_app.validation.three_gate import run_three_gates


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate an LLM manipulation plan through geometry, Curobo2 and ManiSkill")
    parser.add_argument("--plan", type=Path, required=True, help="manipulation_plan.json")
    parser.add_argument("--through", choices=["geometry", "curobo2", "maniskill"], default="maniskill")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--render-mode", choices=["rgb_array", "human"], default="rgb_array")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else RUNTIME_DIR / "task_validation" / time.strftime("%Y%m%d_%H%M%S")
    )
    report = run_three_gates(args.plan, output, through=args.through, render_mode=args.render_mode)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
