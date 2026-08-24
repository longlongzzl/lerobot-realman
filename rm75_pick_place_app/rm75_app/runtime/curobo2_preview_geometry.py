"""Export cuRobo2 robot collision spheres for grasp-preview rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rm75_app.core.frames import robot_base_world_pose
from rm75_app.orchestration.multi_object_executor import load_task_scene
from rm75_app.pickplace.atom_task_builder import AtomTaskBuilderConfig, FixedSceneAtomTaskBuilder
from rm75_app.pickplace.coordinator import _offset_candidates
from rm75_app.planning.backends.curobo2 import Curobo2Backend, Curobo2BackendConfig
from rm75_app.planning.contracts import PlanningScene
from rm75_app.tasks.manipulation_plan import load_plan


def _sphere_model(
    backend: Curobo2Backend,
    joint_positions: np.ndarray,
    role: str,
    *,
    solution_source: str,
) -> dict:
    planner = backend._ensure_planner()
    modules = backend._import_modules()
    state = modules["JointState"].from_position(
        modules["torch"].as_tensor(
            np.asarray(joint_positions, dtype=np.float32).reshape(1, -1),
            dtype=planner.device_cfg.dtype,
            device=planner.device_cfg.device,
        ),
        joint_names=list(planner.joint_names),
    )
    active = planner.kinematics.get_active_js(state)
    kinematics = planner.kinematics.compute_kinematics(active)
    spheres = kinematics.robot_spheres.reshape(-1, 4).detach().cpu().numpy()
    config = planner.kinematics.config.kinematics_config
    sphere_to_link = config.link_sphere_idx_map.detach().cpu().numpy().reshape(-1)
    index_to_name = {
        int(index): str(name) for name, index in config.link_name_to_idx_map.items()
    }
    serialized = []
    for sphere_index, values in enumerate(spheres):
        radius = float(values[3])
        if not np.isfinite(radius) or radius <= 1e-6:
            continue
        serialized.append(
            {
                "sphere_index": sphere_index,
                "link_name": index_to_name.get(
                    int(sphere_to_link[sphere_index]),
                    f"link_index_{int(sphere_to_link[sphere_index])}",
                ),
                "center": np.asarray(values[:3], dtype=float).tolist(),
                "radius": radius,
            }
        )
    return {
        "role": role,
        "solution_source": solution_source,
        "joint_names": list(planner.joint_names),
        "joint_positions": np.asarray(joint_positions, dtype=float).tolist(),
        "spheres": serialized,
    }


def export_preview_geometry(
    plan_file: str | Path,
    output_file: str | Path,
    *,
    candidate_id: str | None = None,
) -> Path:
    plan = load_plan(plan_file)
    if not plan.atoms:
        raise ValueError("preview plan has no manipulation atoms")
    scene = load_task_scene(plan.scene_file)
    scene_payload = json.loads(
        Path(plan.scene_file).expanduser().resolve().read_text(encoding="utf-8")
    )
    T_world_robot_base, robot_base_pose_source = robot_base_world_pose(scene_payload)
    atom = plan.atoms[0]
    with Curobo2Backend(
        Curobo2BackendConfig(
            max_batch_size=4,
            max_goalset=4,
            num_ik_seeds=64,
            num_trajopt_seeds=4,
        )
    ) as backend:
        planner = backend._ensure_planner()
        default_q = (
            planner.default_joint_state.position.detach().cpu().numpy().reshape(-1)[:7]
        )
        scene.set_joints(tuple(planner.joint_names), default_q)
        task = FixedSceneAtomTaskBuilder(
            config=AtomTaskBuilderConfig(
                robot_base_world_xyz_m=tuple(T_world_robot_base[:3, 3].tolist())
            )
        )(atom, scene)
        candidate_pairs = [
            (
                _offset_candidates(
                    (grasp,), abs(float(task.grasp_approach_offset)), "pregrasp"
                )[0],
                grasp,
            )
            for grasp in task.grasp_candidates
        ]
        candidates = tuple(
            candidate
            for pregrasp, grasp in candidate_pairs
            for candidate in (pregrasp, grasp)
        )
        backend.prepare_pose_candidates(
            candidates,
            task.scene,
            tool_frame=task.tool_frame,
            ignore_object_name=task.object_name,
        )
        solutions: dict[tuple[float, ...], tuple[np.ndarray, str]] = {}
        for candidate in candidates:
            q = backend._pose_ik_cache.get(backend._pose_cache_key(candidate))
            if q is not None:
                solutions[backend._pose_cache_key(candidate)] = (
                    q,
                    "collision_checked_ik",
                )
        missing_candidates = tuple(
            candidate
            for candidate in candidates
            if backend._pose_cache_key(candidate) not in solutions
        )
        if missing_candidates:
            # A collision-rejected endpoint is precisely the pose users need
            # to inspect. Solve only its kinematics in an empty world so its
            # robot sphere model can still be overlaid against the real OBBs.
            backend.prepare_pose_candidates(
                missing_candidates,
                PlanningScene(revision="preview-kinematic-fallback"),
                tool_frame=task.tool_frame,
                ignore_object_name=None,
            )
            for candidate in candidates:
                key = backend._pose_cache_key(candidate)
                if key in solutions:
                    continue
                q = backend._pose_ik_cache.get(key)
                if q is not None:
                    solutions[key] = (q, "kinematic_fallback")

        def pair_rank(pair) -> tuple[int, int]:
            pair_solutions = [
                solutions.get(backend._pose_cache_key(candidate))
                for candidate in pair
            ]
            solved = sum(item is not None for item in pair_solutions)
            strict = sum(
                item is not None and item[1] == "collision_checked_ik"
                for item in pair_solutions
            )
            return solved, strict

        if candidate_id is None:
            selected_pregrasp, selected_grasp = max(candidate_pairs, key=pair_rank)
        else:
            try:
                selected_pregrasp, selected_grasp = next(
                    pair for pair in candidate_pairs if pair[1].candidate_id == candidate_id
                )
            except StopIteration as exc:
                available = ", ".join(grasp.candidate_id for _, grasp in candidate_pairs)
                raise ValueError(
                    f"unknown candidate_id {candidate_id!r}; available: {available}"
                ) from exc
        selected = (("pregrasp", selected_pregrasp), ("grasp", selected_grasp))
        models = []
        missing = []
        for role, candidate in selected:
            solution = solutions.get(backend._pose_cache_key(candidate))
            if solution is None:
                missing.append(role)
                continue
            q, source = solution
            models.append(
                _sphere_model(backend, q, role, solution_source=source)
            )
        if not models:
            models.append(
                _sphere_model(
                    backend,
                    default_q,
                    "default",
                    solution_source="default_joint_state",
                )
            )
        payload = {
            "schema_version": "rm75.curobo2_preview_geometry/v1",
            "sphere_coordinate_frame": "robot_base",
            "robot_base_world_pose": T_world_robot_base.tolist(),
            "robot_base_world_pose_source": robot_base_pose_source,
            "atom_id": atom.atom_id,
            "object_id": atom.object_id,
            "candidate_id": selected_grasp.candidate_id,
            "evaluated_candidate_ids": [
                grasp.candidate_id for _, grasp in candidate_pairs
            ],
            "models": models,
            "missing_ik_roles": missing,
            "kinematic_fallback_roles": [
                model["role"]
                for model in models
                if model["solution_source"] == "kinematic_fallback"
            ],
        }
    path = Path(output_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = export_preview_geometry(
        args.plan,
        args.output,
        candidate_id=args.candidate_id,
    )
    print(json.dumps({"ok": True, "output_file": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
