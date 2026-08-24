#!/usr/bin/env python3
"""Compare one task IK target under cuRobo2 batch=4 and batch=64."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rm75_app.core.frames import robot_base_world_pose
from rm75_app.orchestration.multi_object_executor import load_task_scene
from rm75_app.pickplace.atom_task_builder import AtomTaskBuilderConfig, FixedSceneAtomTaskBuilder
from rm75_app.planning.backends.curobo2 import Curobo2Backend, Curobo2BackendConfig
from rm75_app.tasks.manipulation_plan import load_plan


def _goals(backend, planner, candidates):
    modules = backend._import_modules()
    positions = np.stack([item.pose.position for item in candidates])
    quaternions = np.stack([item.pose.quaternion_wxyz for item in candidates])
    return modules["GoalToolPose"](
        tool_frames=["gripper_tcp"],
        position=modules["torch"].as_tensor(
            positions, dtype=planner.device_cfg.dtype, device=planner.device_cfg.device
        )[:, None, None, None, :],
        quaternion=modules["torch"].as_tensor(
            quaternions, dtype=planner.device_cfg.dtype, device=planner.device_cfg.device
        )[:, None, None, None, :],
    )


def _summary(result, batch_size, target_index):
    success = result.success.reshape(batch_size, -1)
    feasible = result.feasible.reshape(batch_size, -1)
    position = result.position_error.reshape(batch_size, -1)
    rotation = result.rotation_error.reshape(batch_size, -1)
    return {
        "target_success": bool(success[target_index].any().item()),
        "target_success_seed_count": int(success[target_index].sum().item()),
        "target_feasible_seed_count": int(feasible[target_index].sum().item()),
        "target_min_position_error_m": float(position[target_index].min().item()),
        "target_min_rotation_error_rad": float(rotation[target_index].min().item()),
        "batch_success_rows": int(success.any(dim=-1).sum().item()),
        "batch_success_seed_count": int(success.sum().item()),
    }


def _run(plan_path: Path, batch_size: int, candidate_id: str, repeat_only: bool):
    plan = load_plan(plan_path)
    scene = load_task_scene(plan.scene_file)
    scene.commit_object_pose(plan.atoms[0].object_id, plan.atoms[0].target_pose)
    scene_payload = json.loads(Path(plan.scene_file).read_text(encoding="utf-8"))
    T_world_base, _ = robot_base_world_pose(scene_payload)
    builder = FixedSceneAtomTaskBuilder(
        config=AtomTaskBuilderConfig(
            robot_base_world_xyz_m=tuple(T_world_base[:3, 3].tolist())
        )
    )
    config = Curobo2BackendConfig(
        max_batch_size=batch_size,
        max_goalset=4,
        num_ik_seeds=64,
        num_trajopt_seeds=4,
        use_cuda_graph=False,
    )
    with Curobo2Backend(config) as backend:
        planner = backend._ensure_planner()
        default = planner.default_joint_state.position.reshape(-1)[:7].detach().cpu().numpy()
        scene.set_joints(tuple(planner.joint_names), default)
        task = builder(plan.atoms[1], scene)
        target = next(item for item in task.grasp_candidates if item.candidate_id == candidate_id)
        if repeat_only:
            candidates = [target] * batch_size
            target_index = 0
        else:
            all_candidates = list(task.grasp_candidates)
            if batch_size == 4:
                index = all_candidates.index(target)
                start = (index // batch_size) * batch_size
                candidates = all_candidates[start : start + batch_size]
                target_index = index - start
            else:
                candidates = all_candidates[:batch_size]
                target_index = candidates.index(target)
                candidates.extend([candidates[-1]] * (batch_size - len(candidates)))
        backend.update_scene(task.scene)
        backend._set_obstacle_enabled(task.object_name, False)
        goals = _goals(backend, planner, candidates)
        planner.ik_solver.reset_seed()
        result = planner.ik_solver.solve_pose(
            goals,
            current_state=None,
            return_seeds=planner.trajopt_solver.config.num_seeds,
        )
        return {
            "batch_size": batch_size,
            "repeat_only": repeat_only,
            "target_index": target_index,
            **_summary(result, batch_size, target_index),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--candidate-id", default="grasp_11_+0deg_tilt_+20deg_axis_-40mm"
    )
    args = parser.parse_args()
    report = [
        _run(args.plan.resolve(), batch_size, args.candidate_id, repeat_only)
        for batch_size in (4, 64)
        for repeat_only in (True, False)
    ]
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
