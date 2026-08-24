#!/usr/bin/env python3
"""Identify which collision class/obstacle rejects a task's grasp IK batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rm75_app.orchestration.multi_object_executor import load_task_scene
from rm75_app.pickplace.atom_task_builder import FixedSceneAtomTaskBuilder
from rm75_app.planning.backends.curobo2 import Curobo2Backend, Curobo2BackendConfig
from rm75_app.planning.contracts import BatchPlanningRequest
from rm75_app.tasks.manipulation_plan import load_plan


def _solve(planner, goals, current) -> list[bool]:
    planner.ik_solver.reset_seed()
    result = planner.ik_solver.solve_pose(
        goals,
        current_state=current,
        return_seeds=planner.trajopt_solver.config.num_seeds,
    )
    return [bool(value) for value in result.success.reshape(planner.batch_size, -1).any(dim=-1).tolist()]


def _endpoint_world_contacts(backend, planner, goals, current, names, target) -> dict:
    """Solve without world obstacles, then identify endpoint robot/object contacts."""
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer

    modules = backend._import_modules()
    planner.ik_solver.reset_seed()
    result = planner.ik_solver.solve_pose(
        goals,
        current_state=current,
        return_seeds=planner.trajopt_solver.config.num_seeds,
    )
    success = result.success.reshape(planner.batch_size, -1)
    if not bool(success[0].any().item()):
        return {"available": False}
    seed_idx = int(success[0].to(dtype=modules["torch"].int32).argmax().item())
    q = result.solution[0, seed_idx].reshape(1, -1)
    state = planner.kinematics.compute_kinematics(
        modules["JointState"].from_position(q, joint_names=list(planner.joint_names))
    )
    spheres = state.robot_spheres.reshape(1, 1, -1, 4)
    sphere_to_link = planner.kinematics.config.kinematics_config.link_sphere_idx_map
    idx_to_name = {
        value: key
        for key, value in planner.kinematics.config.kinematics_config.link_name_to_idx_map.items()
    }
    contacts = {}
    for name in names:
        if name == target:
            continue
        backend._set_obstacle_enabled(name, True)
        per_margin = {}
        for margin in (0.0, backend.config.collision_activation_distance):
            buffer = CollisionBuffer.from_shape(spheres.shape, planner.device_cfg)
            distance = planner.scene_collision_checker.get_sphere_distance_raw(
                spheres,
                buffer,
                modules["torch"].ones(1, device=spheres.device),
                modules["torch"].tensor([margin], device=spheres.device),
            )
            colliding_indices = (distance[0, 0] > 0).nonzero().reshape(-1)
            links = sorted(
                {
                    idx_to_name[int(sphere_to_link[int(index)].item())]
                    for index in colliding_indices.detach().cpu().tolist()
                }
            )
            per_margin[str(margin)] = {
                "links": links,
                "max_cost": float(distance.max().item()),
            }
        contacts[name] = per_margin
        backend._set_obstacle_enabled(name, False)
    return {"available": True, "candidate_index": 0, "contacts": contacts}


def _variant_goals(modules, goals, *, dz: float = 0.0, quaternion=None):
    position = goals.position.clone()
    position[..., 2] += dz
    output_quaternion = goals.quaternion.clone()
    if quaternion is not None:
        output_quaternion[:] = quaternion[:, None, None, None, :]
    return modules["GoalToolPose"](
        tool_frames=goals.tool_frames,
        position=position,
        quaternion=output_quaternion,
    )


def _set_all(backend: Curobo2Backend, names: list[str], enabled: bool) -> None:
    for name in names:
        backend._set_obstacle_enabled(name, enabled)


def _run(plan_path: Path, self_collision_check: bool) -> dict:
    plan = load_plan(plan_path)
    atom = plan.atoms[0]
    scene = load_task_scene(plan.scene_file)
    config = Curobo2BackendConfig(
        max_batch_size=4,
        max_goalset=4,
        num_ik_seeds=64,
        num_trajopt_seeds=4,
        self_collision_check=self_collision_check,
    )
    with Curobo2Backend(config) as backend:
        backend.update_scene(FixedSceneAtomTaskBuilder()._planning_scene(scene))
        planner = backend._ensure_planner()
        default = planner.default_joint_state.position.reshape(-1)[:7].detach().cpu().numpy()
        scene.set_joints(tuple(planner.joint_names), default)
        task = FixedSceneAtomTaskBuilder()(atom, scene)
        request = BatchPlanningRequest(
            current=task.current,
            candidates=task.grasp_candidates,
            scene=task.scene,
            tool_frame=task.tool_frame,
            max_attempts=task.max_attempts,
        )
        current, goals = backend._make_batch_inputs(request)
        names = [item.name for item in task.scene.objects]
        target = task.object_name

        baseline = _solve(planner, goals, current)
        backend._set_obstacle_enabled(target, False)
        target_disabled = _solve(planner, goals, current)

        leave_one_disabled = {}
        for name in names:
            if name == target:
                continue
            backend._set_obstacle_enabled(name, False)
            leave_one_disabled[name] = _solve(planner, goals, current)
            backend._set_obstacle_enabled(name, True)

        _set_all(backend, names, False)
        all_world_disabled = _solve(planner, goals, current)
        endpoint_world_contacts = _endpoint_world_contacts(
            backend, planner, goals, current, names, target
        )
        only_one_enabled = {}
        for name in names:
            if name == target:
                continue
            backend._set_obstacle_enabled(name, True)
            only_one_enabled[name] = _solve(planner, goals, current)
            backend._set_obstacle_enabled(name, False)
        collision_free_variants = None
        if not self_collision_check:
            fk = planner.kinematics.compute_kinematics(current)
            default_quaternion = fk.tool_poses.get_link_pose(task.tool_frame).quaternion.reshape(
                planner.batch_size, 4
            )
            collision_free_variants = {
                "contact_generated_orientation": all_world_disabled,
                "generated_orientation_z_plus_8cm": _solve(
                    planner, _variant_goals(backend._import_modules(), goals, dz=0.08), current
                ),
                "contact_default_orientation": _solve(
                    planner,
                    _variant_goals(
                        backend._import_modules(), goals, quaternion=default_quaternion
                    ),
                    current,
                ),
                "default_orientation_z_plus_8cm": _solve(
                    planner,
                    _variant_goals(
                        backend._import_modules(),
                        goals,
                        dz=0.08,
                        quaternion=default_quaternion,
                    ),
                    current,
                ),
            }
        _set_all(backend, names, True)

        return {
            "self_collision_check": self_collision_check,
            "object": target,
            "candidate_ids": [item.candidate_id for item in task.grasp_candidates],
            "world_objects": names,
            "target_enabled": baseline[: len(task.grasp_candidates)],
            "target_disabled": target_disabled[: len(task.grasp_candidates)],
            "target_and_one_other_disabled": {
                key: value[: len(task.grasp_candidates)]
                for key, value in leave_one_disabled.items()
            },
            "all_world_disabled": all_world_disabled[: len(task.grasp_candidates)],
            "only_one_world_obstacle_enabled": {
                key: value[: len(task.grasp_candidates)]
                for key, value in only_one_enabled.items()
            },
            "candidate_0_endpoint_world_contacts": endpoint_world_contacts,
            "collision_free_pose_variants": collision_free_variants,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "with_self_collision": _run(args.plan.resolve(), True),
        "without_self_collision": _run(args.plan.resolve(), False),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
