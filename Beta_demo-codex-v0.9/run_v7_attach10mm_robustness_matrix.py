from __future__ import annotations

import argparse
import json
from pathlib import Path

from assembly_planner.second_layer_matrix import MatrixConfig, ROOT, run_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(ROOT / "v7_attach10mm_robustness_matrix"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case-retries", type=int, default=2)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--release-gap-mms", default="3")
    parser.add_argument("--release-yaw-degs", default="0,-1,1")
    parser.add_argument("--max-release-candidates", type=int, default=36)
    parser.add_argument("--release-ik-max-position-error", default="0.009")
    parser.add_argument("--release-ik-max-rotation-error", default="0.18")
    parser.add_argument("--edge-seating-attempts", type=int, default=0)
    parser.add_argument("--edge-seating-max-step", default="0.0")
    parser.add_argument("--edge-seating-max-angle-step-deg", default="2.0")
    parser.add_argument("--pre-magnet-geometric-capture-attempts", type=int, default=0)
    parser.add_argument("--magnetic-capture-nudge-attempts", type=int, default=0)
    parser.add_argument("--magnetic-capture-nudge-step", default="0.002")
    parser.add_argument("--magnetic-capture-nudge-steps", default="10")
    parser.add_argument("--magnetic-capture-hold-steps", default="10")
    parser.add_argument("--magnetic-capture-max-angle-step-deg", default="2.0")
    parser.add_argument("--magnetic-capture-max-joint-delta", default="1.4")
    parser.add_argument("--magnetic-capture-revert-tolerance", default="0.0007")
    args = parser.parse_args()

    config = MatrixConfig.from_namespace(args)
    payload = run_matrix(config)
    Path(config.out_root).mkdir(parents=True, exist_ok=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
