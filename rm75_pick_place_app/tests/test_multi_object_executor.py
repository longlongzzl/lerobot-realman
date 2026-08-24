from __future__ import annotations

import json

import numpy as np

from rm75_app.orchestration.multi_object_executor import (
    AtomExecution,
    AtomStatus,
    AtomValidation,
    MultiObjectTaskExecutor,
    RecoveryAction,
    RecoveryDirective,
    SceneObjectState,
    TaskSceneState,
    load_task_scene,
    validate_target_pose,
)
from rm75_app.tasks.manipulation_plan import (
    ManipulationAtom,
    ManipulationPlan,
    ManipulationPrimitive,
)


def pose(x: float) -> np.ndarray:
    value = np.eye(4)
    value[0, 3] = x
    return value


def plan() -> ManipulationPlan:
    first = ManipulationAtom(
        "atom_01",
        ManipulationPrimitive.PICK_PLACE,
        "carrot_1",
        "carriot",
        pose(0.3),
    )
    second = ManipulationAtom(
        "atom_02",
        ManipulationPrimitive.PICK_PLACE,
        "ball_1",
        "tennis",
        pose(0.5),
        depends_on=("atom_01",),
    )
    return ManipulationPlan("demo", "/tmp/scene.json", (first, second))


class FakeBackend:
    def __init__(self, *, fail_atom: str | None = None):
        self.fail_atom = fail_atom
        self.calls: list[tuple] = []
        self.counts: dict[str, int] = {}

    def initialize(self, task_plan):
        self.calls.append(("initialize", task_plan.plan_id))
        return TaskSceneState(
            {
                "carrot_1": SceneObjectState("carrot_1", "carriot", pose(0.0)),
                "ball_1": SceneObjectState("ball_1", "tennis", pose(0.1)),
            },
            joint_names=("j1", "j2"),
            joint_positions=[0.0, 0.0],
        )

    def execute_atom(self, atom, scene, *, attempt):
        self.calls.append(("execute", atom.atom_id, attempt, scene.revision))
        self.counts[atom.atom_id] = self.counts.get(atom.atom_id, 0) + 1
        if atom.atom_id == "atom_01" and attempt == 1:
            return AtomExecution(False, failure_code="grasp_failed", message="mask stale")
        if atom.atom_id == self.fail_atom:
            return AtomExecution(False, failure_code="planning_failed", message="blocked")
        return AtomExecution(
            True,
            final_object_pose=atom.target_pose,
            joint_names=("j1", "j2"),
            joint_positions=[attempt, attempt],
        )

    def validate_atom(self, atom, execution, scene):
        self.calls.append(("validate", atom.atom_id, scene.revision))
        return validate_target_pose(atom, execution)

    def recover_atom(self, atom, execution, validation, scene, *, attempt):
        del validation, scene, attempt
        action = RecoveryAction.REFRESH_AND_RETRY if execution.failure_code == "grasp_failed" else RecoveryAction.RETRY
        return RecoveryDirective(action, f"recover {atom.atom_id}")

    def refresh_scene(self, scene, *, reason):
        self.calls.append(("refresh", reason, scene.revision))
        scene.backend_revision = "refreshed"
        return scene

    def synchronize_scene(self, scene):
        self.calls.append(("sync", scene.revision))


def test_executor_retries_updates_scene_and_preserves_dependencies() -> None:
    backend = FakeBackend()
    result = MultiObjectTaskExecutor(backend, max_attempts_per_atom=2).run(plan())
    assert result.success
    assert [record.status for record in result.records] == [AtomStatus.SUCCEEDED, AtomStatus.SUCCEEDED]
    assert len(result.records[0].attempts) == 2
    assert result.final_scene.revision == 2
    np.testing.assert_allclose(result.final_scene.objects["carrot_1"].pose, pose(0.3))
    np.testing.assert_allclose(result.final_scene.objects["ball_1"].pose, pose(0.5))
    assert result.final_scene.joint_positions.tolist() == [1.0, 1.0]
    assert ("execute", "atom_02", 1, 1) in backend.calls
    assert [call for call in backend.calls if call[0] == "sync"] == [("sync", 1), ("sync", 2)]


def test_executor_fails_fast_and_marks_remaining_atoms_skipped() -> None:
    backend = FakeBackend(fail_atom="atom_01")
    result = MultiObjectTaskExecutor(backend, max_attempts_per_atom=2).run(plan())
    assert not result.success
    assert result.failure_atom_id == "atom_01"
    assert result.records[0].status is AtomStatus.FAILED
    assert result.records[1].status is AtomStatus.SKIPPED
    assert result.final_scene.revision == 0


def test_validation_failure_is_recoverable() -> None:
    backend = FakeBackend()

    def reject(atom, execution, scene):
        del execution, scene
        return AtomValidation(False, atom.success.relation, message="object slipped")

    backend.validate_atom = reject
    result = MultiObjectTaskExecutor(backend, max_attempts_per_atom=2).run(plan())
    assert not result.success
    assert len(result.records[0].attempts) == 2
    assert result.final_scene.revision == 0


def test_scene_loader_reads_fixed_scene_objects(tmp_path) -> None:
    scene_file = tmp_path / "scene.json"
    scene_file.write_text(
        json.dumps(
            {
                "schema": "test_scene_v1",
                "objects": {
                    "carrot_1": {
                        "name": "carriot",
                        "fixed": False,
                        "T_world_obj": pose(0.2).tolist(),
                    },
                    "desk": {
                        "name": "desk",
                        "fixed": True,
                        "pose": [0, 0, 0, 1, 0, 0, 0],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    scene = load_task_scene(str(scene_file))
    assert set(scene.objects) == {"carrot_1", "desk"}
    assert scene.objects["carrot_1"].asset_name == "carriot"
    assert not scene.objects["desk"].movable
    assert scene.backend_revision == "test_scene_v1"


def test_target_validation_ignores_spherical_rotation() -> None:
    target = np.eye(4)
    actual = target.copy()
    actual[:3, :3] = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    atom = ManipulationAtom(
        "ball", ManipulationPrimitive.PICK_PLACE, "ball", "tennis", target
    )

    result = validate_target_pose(atom, AtomExecution(True, final_object_pose=actual))

    assert result.success
    assert result.orientation_error_deg == 0.0


def test_target_validation_ignores_only_axial_spin() -> None:
    target = np.eye(4)
    axial_spin = target.copy()
    axial_spin[:3, :3] = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    wrong_axis = target.copy()
    wrong_axis[:3, :3] = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    atom = ManipulationAtom(
        "pen", ManipulationPrimitive.PICK_PLACE, "pen", "bi", target
    )
    reversed_axis = target.copy()
    reversed_axis[:3, :3] = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    )

    spin_result = validate_target_pose(
        atom, AtomExecution(True, final_object_pose=axial_spin)
    )
    wrong_result = validate_target_pose(
        atom, AtomExecution(True, final_object_pose=wrong_axis)
    )
    reversed_result = validate_target_pose(
        atom, AtomExecution(True, final_object_pose=reversed_axis)
    )

    assert spin_result.success
    assert spin_result.orientation_error_deg == 0.0
    assert reversed_result.success
    assert reversed_result.orientation_error_deg == 0.0
    assert not wrong_result.success
