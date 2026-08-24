"""Measure cuRobo2's unmodified MotionPlanner fast path on this workstation."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch


DEFAULT_CUROBO_ROOT = Path("/home/zhangzhao/PycharmProjects/curobo2/curobo")


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.mean(ordered),
        "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max_ms": ordered[-1],
    }


def _measure(planner: Any, goals: Any, start: Any, *, repeats: int) -> dict[str, Any]:
    wall_ms: list[float] = []
    reported_ms: list[float] = []
    successes = 0
    for _ in range(repeats):
        planner.reset_seed()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = planner.plan_pose(
            goals,
            start,
            max_attempts=1,
            enable_graph_attempt=1,
        )
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - started) * 1000.0)
        if result is not None:
            reported_ms.append(float(result.total_time) * 1000.0)
            successes += int(bool(result.success.item()))
    return {
        "successes": successes,
        "repeats": repeats,
        "wall": _summary(wall_ms),
        "reported": _summary(reported_ms) if reported_ms else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curobo-root", type=Path, default=DEFAULT_CUROBO_ROOT)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.curobo_root.resolve()))
    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
    from curobo._src.motion import MotionPlanner, MotionPlannerCfg
    from curobo._src.state.state_joint import JointState
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.types.pose import Pose
    from curobo._src.types.tool_pose import GoalToolPose

    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        device_cfg=device_cfg,
        num_ik_seeds=32,
        num_trajopt_seeds=4,
        use_cuda_graph=bool(args.cuda_graph),
        max_goalset=4,
    )
    planner = MotionPlanner(config)
    planner.warmup(
        enable_graph=True,
        num_warmup_iterations=int(args.warmup_iterations),
    )

    start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )
    start_pose = planner.compute_kinematics(start).tool_poses.to_dict()[planner.tool_frames[0]]
    target_pose = Pose(
        position=start_pose.position + torch.tensor(
            [[0.0, 0.0, 0.05]], **device_cfg.as_torch_dict()
        ),
        quaternion=start_pose.quaternion.clone(),
    )
    goals = GoalToolPose.from_poses(
        {planner.tool_frames[0]: target_pose},
        ordered_tool_frames=planner.tool_frames,
    )

    for _ in range(3):
        planner.reset_seed()
        planner.plan_pose(goals, start, max_attempts=1, enable_graph_attempt=1)
    torch.cuda.synchronize()
    unconstrained = _measure(planner, goals, start, repeats=int(args.repeats))

    planner.update_tool_pose_criteria(
        {
            planner.tool_frames[0]: ToolPoseCriteria.linear_motion(
                axis="z",
                non_terminal_scale=0.5,
                project_distance_to_goal=False,
            )
        }
    )
    for _ in range(3):
        planner.reset_seed()
        planner.plan_pose(goals, start, max_attempts=1, enable_graph_attempt=1)
    torch.cuda.synchronize()
    constrained = _measure(planner, goals, start, repeats=int(args.repeats))

    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda_graph": bool(args.cuda_graph),
                "robot": "franka.yml",
                "scene": "empty",
                "motion": "world_z_+50mm",
                "unconstrained": unconstrained,
                "constrained_world_z": constrained,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
