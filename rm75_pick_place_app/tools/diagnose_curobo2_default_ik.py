#!/usr/bin/env python3
"""Closed-loop FK -> IK smoke test for the RM75 cuRobo2 configuration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rm75_app.planning.backends.curobo2 import Curobo2Backend, Curobo2BackendConfig
from rm75_app.orchestration.multi_object_executor import load_task_scene
from rm75_app.pickplace.atom_task_builder import FixedSceneAtomTaskBuilder
from rm75_app.planning.contracts import PlanningScene


def _summary(result, reference, batch_size: int) -> dict:
    success = result.success.reshape(batch_size, -1)
    position_error = result.position_error.reshape(batch_size, -1)
    rotation_error = result.rotation_error.reshape(batch_size, -1)
    feasible = result.feasible.reshape(batch_size, -1)
    output = {
        "success": bool(success[0].any().item()),
        "constraint_feasible": bool(feasible[0].any().item()),
        "min_position_error_m": float(position_error[0].min().item()),
        "min_rotation_error_rad": float(rotation_error[0].min().item()),
    }
    if output["success"]:
        solutions = result.solution.reshape(batch_size, -1, result.solution.shape[-1])
        winner = int(success[0].to(dtype=solutions.dtype).argmax().item())
        output["max_joint_error_rad"] = float(
            (solutions[0, winner] - reference.position[0]).abs().max().item()
        )
    return output


def main() -> None:
    config = Curobo2BackendConfig(
        max_batch_size=4,
        max_goalset=4,
        num_ik_seeds=64,
        num_trajopt_seeds=4,
    )
    with Curobo2Backend(config) as backend:
        planner = backend._ensure_planner()
        modules = backend._import_modules()
        batch_size = planner.batch_size
        default_position = planner.default_joint_state.position.reshape(1, -1)
        current = modules["JointState"].from_position(
            default_position.repeat(batch_size, 1),
            joint_names=list(planner.joint_names),
        )
        state = planner.kinematics.compute_kinematics(current)
        tcp = state.tool_poses.get_link_pose("gripper_tcp")
        base_position = tcp.position.reshape(batch_size, 3)
        base_quaternion = tcp.quaternion.reshape(batch_size, 4)

        isolated_scene_path = (
            Path(__file__).resolve().parents[1]
            / "runtime_data/task_validation/20260811_164912/carrot_only_scene.json"
        )
        task_scene = load_task_scene(isolated_scene_path)
        backend.update_scene(FixedSceneAtomTaskBuilder()._planning_scene(task_scene))
        isolated_position = modules["torch"].tensor(
            [-0.2148664594, -0.2131349742, 0.1038346636],
            device=base_position.device,
        ).repeat(batch_size, 1)
        isolated_quaternion = modules["torch"].tensor(
            [
                [0.0, 0.9362358089, -0.3513723242, 0.0],
                [0.0, 0.8943786021, -0.4473107602, 0.0],
                [0.0, 0.9678354202, -0.2515841795, 0.0],
                [0.0, 0.9104764424, 0.4135609361, 0.0],
            ],
            device=base_position.device,
        )
        isolated_goals = modules["GoalToolPose"](
            tool_frames=["gripper_tcp"],
            position=isolated_position[:, None, None, None, :],
            quaternion=isolated_quaternion[:, None, None, None, :],
        )
        backend._set_obstacle_enabled("carriot", False)
        first_plan_pose = planner.plan_pose(
            isolated_goals,
            current,
            max_attempts=3,
            enable_graph_attempt=1,
        )
        planner.reset_seed()
        native_grasp = planner.plan_grasp(
            isolated_goals,
            current,
            grasp_approach_axis="z",
            grasp_approach_offset=-0.08,
            grasp_approach_in_tool_frame=True,
            plan_approach_to_grasp=True,
            plan_grasp_to_lift=False,
        )
        self_collision_report = {"has_collision": False, "pairs": []}
        if (
            native_grasp.approach_result is not None
            and native_grasp.goalset_result is not None
        ):
            from curobo._src.cost.cost_self_collision import SelfCollisionCost
            from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg

            approach_end = modules["get_joint_state_at_horizon_index"](
                native_grasp.approach_result.js_solution, -1
            ).squeeze(1)
            approach_end = planner.kinematics.get_active_js(approach_end)
            grasp_end = modules["get_joint_state_at_horizon_index"](
                native_grasp.goalset_result.js_solution, -1
            ).squeeze(1)
            grasp_end = planner.kinematics.get_active_js(grasp_end)
            alpha = modules["torch"].linspace(
                0.0, 1.0, 65, device=approach_end.position.device
            ).view(1, 65, 1)
            q_path = approach_end.position[:, None, :] + alpha * (
                grasp_end.position[:, None, :] - approach_end.position[:, None, :]
            )
            contact_links = list(
                planner.kinematics.config.kinematics_config.grasp_contact_link_names or []
            )
            planner.disable_link_collision(contact_links)
            path_state = planner.kinematics.compute_kinematics(
                modules["JointState"].from_position(
                    q_path.reshape(-1, q_path.shape[-1]),
                    joint_names=list(planner.joint_names),
                )
            )
            spheres = path_state.robot_spheres.reshape(
                batch_size, 65, -1, 4
            )
            self_cost = SelfCollisionCost(
                SelfCollisionCostCfg(
                    weight=1.0,
                    self_collision_kin_config=planner.kinematics.get_self_collision_config(),
                    store_pair_distance=True,
                )
            )
            self_cost.setup_batch_tensors(batch_size, 65)
            self_cost.forward(spheres)
            pair_distance = self_cost._pair_distance
            colliding = pair_distance > 0
            pair_cfg = planner.kinematics.get_self_collision_config().collision_pairs
            sphere_to_link = (
                planner.kinematics.config.kinematics_config.link_sphere_idx_map
            )
            idx_to_name = {
                value: key
                for key, value in planner.kinematics.config.kinematics_config.link_name_to_idx_map.items()
            }
            pairs = set()
            if bool(colliding.any().item()):
                pair_indices = colliding.any(dim=(0, 1))
                sphere_pairs = pair_cfg[pair_indices]
                link_pairs = sphere_to_link[sphere_pairs.to(dtype=modules["torch"].int32)]
                for left, right in link_pairs.detach().cpu().tolist():
                    pairs.add((idx_to_name[int(left)], idx_to_name[int(right)]))
            self_collision_report = {
                "has_collision": bool(colliding.any().item()),
                "pairs": sorted([list(pair) for pair in pairs]),
                "max_penetration_m": float(pair_distance.max().item()),
            }
            planner.enable_link_collision(contact_links)
        backend._set_obstacle_enabled("carriot", True)
        first_plan_pose_report = {
            "returned_none": first_plan_pose is None,
            "success": False
            if first_plan_pose is None
            else bool(first_plan_pose.success[0].any().item()),
        }
        native_grasp_report = {
            "status": native_grasp.status,
            "success": native_grasp.success.detach().cpu().tolist(),
            "approach_success": native_grasp.approach_success.detach().cpu().tolist(),
            "grasp_success": native_grasp.grasp_success.detach().cpu().tolist(),
            "joint_linear_self_collision": self_collision_report,
            "enabled_world_obstacles_during_grasp": [],
        }
        backend.update_scene(PlanningScene())
        planner.destroy()
        backend._planner = None
        planner = backend._ensure_planner()

        cases = {
            "fk_roundtrip": (base_position, base_quaternion),
            "world_z_plus_2cm": (
                base_position + modules["torch"].tensor(
                    [0.0, 0.0, 0.02], device=base_position.device
                ),
                base_quaternion,
            ),
            "world_x_plus_5cm": (
                base_position + modules["torch"].tensor(
                    [0.05, 0.0, 0.0], device=base_position.device
                ),
                base_quaternion,
            ),
            "carrot_pregrasp_default_orientation": (
                modules["torch"].tensor(
                    [-0.2148664594, -0.2131349742, 0.1038346636],
                    device=base_position.device,
                ).repeat(batch_size, 1),
                base_quaternion,
            ),
            "carrot_pregrasp_generated_orientation": (
                modules["torch"].tensor(
                    [-0.2148664594, -0.2131349742, 0.1038346636],
                    device=base_position.device,
                ).repeat(batch_size, 1),
                modules["torch"].tensor(
                    [0.0, 0.9362358089, -0.3513723242, 0.0],
                    device=base_position.device,
                ).repeat(batch_size, 1),
            ),
            "carrot_grasp_generated_orientation": (
                modules["torch"].tensor(
                    [-0.2148664594, -0.2131349742, 0.0238346636],
                    device=base_position.device,
                ).repeat(batch_size, 1),
                modules["torch"].tensor(
                    [0.0, 0.9362358089, -0.3513723242, 0.0],
                    device=base_position.device,
                ).repeat(batch_size, 1),
            ),
        }
        report = {
            "joint_names": list(planner.joint_names),
            "default_joints_rad": default_position[0].detach().cpu().tolist(),
            "default_tcp": {
                "position_m": base_position[0].detach().cpu().tolist(),
                "quaternion_wxyz": base_quaternion[0].detach().cpu().tolist(),
            },
            "first_call_isolated_plan_pose": first_plan_pose_report,
            "native_plan_grasp": native_grasp_report,
            "cases": {},
        }
        for name, (position, quaternion) in cases.items():
            goals = modules["GoalToolPose"](
                tool_frames=["gripper_tcp"],
                position=position[:, None, None, None, :],
                quaternion=quaternion[:, None, None, None, :],
            )
            biased = planner.ik_solver.solve_pose(
                goals,
                current_state=current,
                return_seeds=planner.trajopt_solver.config.num_seeds,
            )
            unbiased = planner.ik_solver.solve_pose(
                goals,
                current_state=None,
                return_seeds=planner.trajopt_solver.config.num_seeds,
            )
            report["cases"][name] = {
                "current_biased": _summary(biased, current, batch_size),
                "unbiased": _summary(unbiased, current, batch_size),
            }
        backend.update_scene(FixedSceneAtomTaskBuilder()._planning_scene(task_scene))
        position, quaternion = cases["carrot_pregrasp_generated_orientation"]
        goals = modules["GoalToolPose"](
            tool_frames=["gripper_tcp"],
            position=position[:, None, None, None, :],
            quaternion=quaternion[:, None, None, None, :],
        )
        mesh_store = planner.scene_collision_checker.data.meshes
        before = planner.ik_solver.solve_pose(
            goals,
            current_state=current,
            return_seeds=planner.trajopt_solver.config.num_seeds,
        )
        backend._set_obstacle_enabled("carriot", False)
        disabled_value = int(mesh_store.enable[0, 0].item())
        after = planner.ik_solver.solve_pose(
            goals,
            current_state=current,
            return_seeds=planner.trajopt_solver.config.num_seeds,
        )
        backend._set_obstacle_enabled("carriot", True)
        report["isolated_mesh_toggle"] = {
            "mesh_names": mesh_store.names,
            "disabled_tensor_value": disabled_value,
            "enabled": _summary(before, current, batch_size),
            "disabled": _summary(after, current, batch_size),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
