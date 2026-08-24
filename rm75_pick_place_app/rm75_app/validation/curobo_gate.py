"""Whole-plan Curobo2 feasibility gate without robot or physics execution."""

from __future__ import annotations

import json
from pathlib import Path

from rm75_app.core.frames import robot_base_world_pose
from rm75_app.execution.trajectory_executor import RecordingTrajectoryExecutor
from rm75_app.orchestration.multi_object_executor import MultiObjectTaskExecutor, load_task_scene
from rm75_app.pickplace.atom_task_builder import AtomTaskBuilderConfig, FixedSceneAtomTaskBuilder
from rm75_app.pickplace.coordinator import PickPlaceCoordinator
from rm75_app.pickplace.multi_object_adapter import PickPlaceAtomBackend
from rm75_app.planning.backends.curobo2 import Curobo2Backend, Curobo2BackendConfig
from rm75_app.tasks.manipulation_plan import ManipulationPlan
from rm75_app.validation.contracts import GateResult, GateStatus


def run_curobo_gate(
    plan: ManipulationPlan,
    output_dir: str | Path,
    *,
    backend_config: Curobo2BackendConfig | None = None,
    max_attempts_per_atom: int = 2,
) -> GateResult:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = load_task_scene(plan.scene_file)
    scene_payload = json.loads(
        Path(plan.scene_file).expanduser().resolve().read_text(encoding="utf-8")
    )
    T_world_robot_base, robot_base_pose_source = robot_base_world_pose(scene_payload)
    execution_file = output / "execution.json"
    recorder = RecordingTrajectoryExecutor(execution_file)
    config = backend_config or Curobo2BackendConfig(
        max_batch_size=4,
        max_goalset=4,
        num_ik_seeds=64,
        num_trajopt_seeds=4,
        attachment_num_spheres=64,
    )
    with Curobo2Backend(config) as planner:
        curobo = planner._ensure_planner()
        initial = curobo.default_joint_state.position.detach().cpu().numpy().reshape(-1)[:7]
        scene.set_joints(tuple(curobo.joint_names), initial)
        coordinator = PickPlaceCoordinator(planner, recorder)
        atom_backend = PickPlaceAtomBackend(
            coordinator,
            FixedSceneAtomTaskBuilder(
                config=AtomTaskBuilderConfig(
                    robot_base_world_xyz_m=tuple(T_world_robot_base[:3, 3].tolist())
                )
            ),
            initial_scene=scene,
        )
        result = MultiObjectTaskExecutor(
            atom_backend,
            max_attempts_per_atom=max_attempts_per_atom,
        ).run(plan)
    report_file = output / "curobo_gate_result.json"
    report_file.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    checks_list = []
    for record in result.records:
        attempts = []
        collision_map = {}
        for attempt in record.attempts:
            diagnostics = dict(attempt.execution.artifacts.get("diagnostics") or {})
            attempts.append(
                {
                    "attempt": attempt.attempt,
                    "failure_stage": attempt.execution.failure_stage,
                    "diagnostics": diagnostics,
                }
            )
            for candidate in diagnostics.get("candidates", []):
                candidate_id = candidate.get("candidate_id")
                for collision in candidate.get("collision_diagnostics", []):
                    item = {
                        "attempt": attempt.attempt,
                        "candidate_id": candidate_id,
                        **dict(collision),
                    }
                    other = item.get("world_object") or item.get("other_robot_link")
                    key = (
                        candidate_id,
                        item.get("collision_type"),
                        item.get("state"),
                        item.get("robot_link"),
                        other,
                    )
                    previous = collision_map.get(key)
                    if previous is None or float(item.get("penetration_m", 0.0)) > float(
                        previous.get("penetration_m", 0.0)
                    ):
                        collision_map[key] = item
        checks_list.append(
            {
                "atom_id": record.atom_id,
                "status": record.status.value,
                "attempt_count": len(record.attempts),
                "message": record.message,
                "attempts": attempts,
                "collisions": list(collision_map.values()),
                "batch_collision_sources": sorted(
                    {
                        str(item["candidate_id"])
                        for item in collision_map.values()
                        if item.get("source") == "curobo2_prm_terminal_check"
                    }
                ),
            }
        )
    checks = tuple(checks_list)
    return GateResult(
        "curobo2",
        GateStatus.PASSED if result.success else GateStatus.FAILED,
        "all atoms have feasible Curobo2 trajectories" if result.success else "Curobo2 rejected the task chain",
        checks,
        artifacts={
            "execution_file": str(execution_file),
            "result_file": str(report_file),
            "robot_base_world_pose": T_world_robot_base.tolist(),
            "robot_base_world_pose_source": robot_base_pose_source,
        },
    )
