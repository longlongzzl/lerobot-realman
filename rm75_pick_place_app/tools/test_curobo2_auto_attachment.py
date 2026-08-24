#!/usr/bin/env python3
"""GPU smoke test for automatic cuRobo2 MORPHIT object attachment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rm75_app.orchestration.multi_object_executor import load_task_scene
from rm75_app.pickplace.atom_task_builder import FixedSceneAtomTaskBuilder
from rm75_app.planning.backends.curobo2 import Curobo2Backend, Curobo2BackendConfig
from rm75_app.planning.contracts import BatchPlanningRequest, JointConfiguration
from rm75_app.tasks.manipulation_plan import load_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()

    plan = load_plan(args.plan.expanduser().resolve())
    scene = load_task_scene(plan.scene_file)
    with Curobo2Backend(
        Curobo2BackendConfig(
            max_batch_size=4,
            max_goalset=4,
            num_ik_seeds=64,
            num_trajopt_seeds=4,
            attachment_num_spheres=64,
        )
    ) as backend:
        backend.update_scene(FixedSceneAtomTaskBuilder()._planning_scene(scene))
        planner = backend._ensure_planner()
        default = planner.default_joint_state.position.reshape(-1)[:7].detach().cpu().numpy()
        scene.set_joints(tuple(planner.joint_names), default)
        task = FixedSceneAtomTaskBuilder()(plan.atoms[0], scene)
        request = BatchPlanningRequest(
            current=task.current,
            candidates=task.grasp_candidates,
            scene=task.scene,
            tool_frame=task.tool_frame,
            max_attempts=task.max_attempts,
        )
        current, goals = backend._make_batch_inputs(request)
        for item in task.scene.objects:
            backend._set_obstacle_enabled(item.name, False)
        result = planner.ik_solver.solve_pose(
            goals,
            current_state=current,
            return_seeds=planner.trajopt_solver.config.num_seeds,
        )
        success = result.success.reshape(planner.batch_size, -1)
        candidate_idx = next(
            (i for i in range(len(task.grasp_candidates)) if bool(success[i].any().item())),
            None,
        )
        if candidate_idx is None:
            raise RuntimeError("no IK endpoint available for attachment smoke test")
        torch = backend._import_modules()["torch"]
        seed_idx = int(success[candidate_idx].to(dtype=torch.int32).argmax().item())
        q = result.solution[candidate_idx, seed_idx].reshape(-1).detach().cpu().numpy()
        grasp = JointConfiguration(tuple(planner.joint_names), q)
        backend.attach_object(task.object_name, grasp)
        indices = planner.kinematics.config.kinematics_config.get_sphere_index_from_link_name(
            "attached_object"
        )
        fitted = planner.kinematics.config.kinematics_config.link_spheres[0, indices]
        enabled = fitted[fitted[:, 3] > 0]
        payload = {
            "ok": True,
            "object": task.object_name,
            "candidate": task.grasp_candidates[candidate_idx].candidate_id,
            "fit": backend._attachment_fit_report,
            "enabled_spheres": int(enabled.shape[0]),
            "radius_min_m": float(enabled[:, 3].min().item()),
            "radius_max_m": float(enabled[:, 3].max().item()),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
