from __future__ import annotations

import argparse
import json
from pathlib import Path

from assembly_planner.full_build import FullBuildConfig, run_full_build
from assembly_planner.second_layer_matrix import CHECKPOINT, ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "v9_full_physical_build"))
    parser.add_argument("--from-scratch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--four-wall-state", default=str(CHECKPOINT))
    parser.add_argument("--four-wall-max-attempts-per-role", type=int, default=6)
    parser.add_argument("--four-wall-timeout-sec", type=float, default=600.0)
    parser.add_argument("--triangle-case-retries", type=int, default=2)
    parser.add_argument("--triangle-timeout-sec", type=float, default=900.0)
    parser.add_argument("--enable-triangle-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-every", type=int, default=4)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--use-success-profile-fast-path", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    config = FullBuildConfig(
        out_dir=Path(args.out_dir),
        from_scratch=bool(args.from_scratch),
        four_wall_state=Path(args.four_wall_state),
        four_wall_max_attempts_per_role=int(args.four_wall_max_attempts_per_role),
        four_wall_timeout_sec=float(args.four_wall_timeout_sec),
        triangle_case_retries=int(args.triangle_case_retries),
        triangle_timeout_sec=float(args.triangle_timeout_sec),
        enable_triangle_fallback=bool(args.enable_triangle_fallback),
        record_every=int(args.record_every),
        fps=int(args.fps),
        use_success_profile_fast_path=bool(args.use_success_profile_fast_path),
    )
    result = run_full_build(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
