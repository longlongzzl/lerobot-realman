#!/usr/bin/env python3
"""Benchmark one native cuRobo2 plan_grasp call without external IK screening."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rm75_app.core.frames import robot_base_world_pose
from rm75_app.orchestration.multi_object_executor import load_task_scene
from rm75_app.pickplace.atom_task_builder import AtomTaskBuilderConfig, FixedSceneAtomTaskBuilder
from rm75_app.planning.backends.curobo2 import Curobo2Backend, Curobo2BackendConfig
from rm75_app.planning.contracts import BatchPlanningRequest, GraspPlanningRequest
from rm75_app.tasks.manipulation_plan import load_plan


def _success_count(value, batch_size: int) -> int:
    if value is None:
        return 0
    return int(value.reshape(batch_size, -1).any(dim=-1).sum().item())


def _success_ids(value, batch_size: int, candidates) -> list[str]:
    if value is None:
        return []
    rows = value.reshape(batch_size, -1).any(dim=-1).tolist()
    return [
        candidate.candidate_id
        for candidate, succeeded in zip(candidates, rows, strict=True)
        if bool(succeeded)
    ]


def run_once(
    plan_path: Path,
    atom_index: int,
    batch_size: int,
    candidate_offset: int,
    rm75_adapter_patches: bool,
) -> dict:
    plan = load_plan(plan_path)
    atom = plan.atoms[atom_index]
    scene = load_task_scene(plan.scene_file)
    scene_payload = json.loads(Path(plan.scene_file).read_text(encoding="utf-8"))
    world_from_base, _ = robot_base_world_pose(scene_payload)
    builder = FixedSceneAtomTaskBuilder(
        config=AtomTaskBuilderConfig(
            robot_base_world_xyz_m=tuple(world_from_base[:3, 3].tolist())
        )
    )
    config = Curobo2BackendConfig(
        max_batch_size=batch_size,
        max_goalset=batch_size,
        num_ik_seeds=64,
        num_trajopt_seeds=4,
        use_cuda_graph=False,
        multi_env=False,
    )
    construction_started = time.perf_counter()
    with Curobo2Backend(config) as backend:
        planner = backend._ensure_planner()
        modules = backend._import_modules()
        modules["torch"].cuda.synchronize()
        construction_s = time.perf_counter() - construction_started
        default = planner.default_joint_state.position.reshape(-1)[:7].detach().cpu().numpy()
        scene.set_joints(tuple(planner.joint_names), default)
        task = builder(atom, scene)
        candidates = tuple(
            task.grasp_candidates[candidate_offset : candidate_offset + batch_size]
        )
        if len(candidates) != batch_size:
            raise ValueError(
                f"requested candidates [{candidate_offset}:{candidate_offset + batch_size}], "
                f"but task only has {len(task.grasp_candidates)}"
            )
        backend.update_scene(task.scene)
        current, goals = backend._make_batch_inputs(
            BatchPlanningRequest(
                current=task.current,
                candidates=candidates,
                scene=task.scene,
                tool_frame=task.tool_frame,
                max_attempts=task.max_attempts,
            )
        )
        backend._set_obstacle_enabled(task.object_name, False)
        contact_links = list(
            planner.kinematics.config.kinematics_config.grasp_contact_link_names or []
        )
        restore_callbacks = []
        if rm75_adapter_patches:
            restore_callbacks = [
                backend._install_isolated_graph_seeding(planner),
                backend._install_grasp_stage_attempts(
                    planner, max_attempts=task.max_attempts
                ),
                backend._install_grasp_linear_scale(
                    planner, scale=config.grasp_linear_non_terminal_scale
                ),
            ]
        calls = []
        for label in ("cold", "warm"):
            planner.reset_seed()
            modules["torch"].cuda.synchronize()
            started = time.perf_counter()
            result = planner.plan_grasp(
                goals,
                current,
                grasp_approach_axis="z",
                grasp_approach_offset=abs(float(task.grasp_approach_offset)),
                grasp_approach_in_tool_frame=False,
                plan_approach_to_grasp=True,
                grasp_lift_axis="z",
                grasp_lift_offset=abs(float(task.lift_height)),
                grasp_lift_in_tool_frame=False,
                plan_grasp_to_lift=True,
                disable_collision_links=contact_links,
            )
            modules["torch"].cuda.synchronize()
            calls.append(
                {
                    "label": label,
                    "elapsed_s": time.perf_counter() - started,
                    "status": str(result.status),
                    "success_rows": _success_count(result.success, batch_size),
                    "success_candidate_ids": _success_ids(
                        result.success, batch_size, candidates
                    ),
                    "goalset_success_rows": _success_count(
                        None
                        if getattr(result, "goalset_result", None) is None
                        else result.goalset_result.success,
                        batch_size,
                    ),
                    "goalset_success_candidate_ids": _success_ids(
                        None
                        if getattr(result, "goalset_result", None) is None
                        else result.goalset_result.success,
                        batch_size,
                        candidates,
                    ),
                    "approach_success_rows": _success_count(
                        getattr(result, "approach_success", None), batch_size
                    ),
                    "approach_success_candidate_ids": _success_ids(
                        getattr(result, "approach_success", None), batch_size, candidates
                    ),
                    "grasp_success_rows": _success_count(
                        getattr(result, "grasp_success", None), batch_size
                    ),
                    "grasp_success_candidate_ids": _success_ids(
                        getattr(result, "grasp_success", None), batch_size, candidates
                    ),
                    "lift_success_rows": _success_count(
                        getattr(result, "lift_success", None), batch_size
                    ),
                    "lift_success_candidate_ids": _success_ids(
                        getattr(result, "lift_success", None), batch_size, candidates
                    ),
                }
            )
        for restore in reversed(restore_callbacks):
            restore()
        return {
            "atom_id": atom.atom_id,
            "object_id": atom.object_id,
            "batch_size": batch_size,
            "candidate_offset": candidate_offset,
            "rm75_adapter_patches": rm75_adapter_patches,
            "candidate_ids": [item.candidate_id for item in candidates],
            "planner_construction_s": construction_s,
            "calls": calls,
        }


def run_chunked_backend(
    plan_path: Path, atom_index: int, chunk_size: int, num_trajopt_seeds: int
) -> dict:
    plan = load_plan(plan_path)
    atom = plan.atoms[atom_index]
    scene = load_task_scene(plan.scene_file)
    scene_payload = json.loads(Path(plan.scene_file).read_text(encoding="utf-8"))
    world_from_base, _ = robot_base_world_pose(scene_payload)
    builder = FixedSceneAtomTaskBuilder(
        config=AtomTaskBuilderConfig(
            robot_base_world_xyz_m=tuple(world_from_base[:3, 3].tolist())
        )
    )
    config = Curobo2BackendConfig(
        max_batch_size=chunk_size,
        max_goalset=chunk_size,
        num_ik_seeds=32,
        num_trajopt_seeds=num_trajopt_seeds,
        use_cuda_graph=False,
        multi_env=False,
    )
    wall_started = time.perf_counter()
    with Curobo2Backend(config) as backend:
        planner = backend._ensure_planner()
        default = planner.default_joint_state.position.reshape(-1)[:7].detach().cpu().numpy()
        scene.set_joints(tuple(planner.joint_names), default)
        task = builder(atom, scene)
        chunks = []
        for start in range(0, len(task.grasp_candidates), chunk_size):
            candidates = tuple(task.grasp_candidates[start : start + chunk_size])
            started = time.perf_counter()
            result = backend.plan_grasps(
                GraspPlanningRequest(
                    planning=BatchPlanningRequest(
                        current=task.current,
                        candidates=candidates,
                        scene=task.scene,
                        tool_frame=task.tool_frame,
                        max_attempts=task.max_attempts,
                    ),
                    target_object_name=task.object_name,
                    approach_axis="z",
                    approach_offset=abs(float(task.grasp_approach_offset)),
                    approach_in_tool_frame=False,
                    phase="pick",
                    plan_lift=True,
                    lift_axis="z",
                    lift_offset=abs(float(task.lift_height)),
                    lift_in_tool_frame=False,
                )
            )
            chunks.append(
                {
                    "chunk_index": len(chunks),
                    "candidate_offset": start,
                    "candidate_count": len(candidates),
                    "wall_time_s": time.perf_counter() - started,
                    "planner_time_s": float(result.total_time),
                    "success_count": sum(bool(item.success) for item in result.plans),
                    "success_candidate_ids": [
                        item.candidate_id for item in result.plans if item.success
                    ],
                    "status_counts": {
                        status: sum(item.status == status for item in result.plans)
                        for status in sorted({str(item.status) for item in result.plans})
                    },
                }
            )
        return {
            "object_id": task.object_name,
            "candidate_count": len(task.grasp_candidates),
            "chunk_size": chunk_size,
            "num_trajopt_seeds": num_trajopt_seeds,
            "chunk_count": len(chunks),
            "wall_time_s": time.perf_counter() - wall_started,
            "planner_time_s": sum(item["planner_time_s"] for item in chunks),
            "chunks": chunks,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--atom-index", type=int, default=0)
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--rm75-adapter-patches", action="store_true")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 4, 8))
    parser.add_argument("--chunked-backend", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--chunked-num-trajopt-seeds", type=int, default=4)
    args = parser.parse_args()
    if args.chunked_backend:
        report = run_chunked_backend(
            args.plan.resolve(),
            args.atom_index,
            args.chunk_size,
            args.chunked_num_trajopt_seeds,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    report = [
        run_once(
            args.plan.resolve(),
            args.atom_index,
            batch_size,
            args.candidate_offset,
            args.rm75_adapter_patches,
        )
        for batch_size in args.batch_sizes
    ]
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
