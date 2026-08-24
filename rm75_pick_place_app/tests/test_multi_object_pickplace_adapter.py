from __future__ import annotations

import json

import numpy as np

from rm75_app.orchestration.multi_object_executor import MultiObjectTaskExecutor, SceneObjectState, TaskSceneState
from rm75_app.pickplace.coordinator import ExecutedStage, PickPlaceRunResult
from rm75_app.pickplace.multi_object_adapter import PickPlaceAtomBackend
from rm75_app.planning.contracts import JointConfiguration
from rm75_app.tasks.manipulation_plan import ManipulationAtom, ManipulationPlan, ManipulationPrimitive


class FakeCoordinator:
    def __init__(self):
        self.tasks = []

    def run(self, task):
        self.tasks.append(task)
        return PickPlaceRunResult(
            True,
            (ExecutedStage("retreat", "place", JointConfiguration(("j1", "j2"), [0.4, 0.5])),),
            selected_grasp="grasp_1",
            selected_place="place_1",
        )


class InfeasibleCoordinator:
    def run(self, task):
        del task
        return PickPlaceRunResult(
            False,
            (),
            failure_stage="relation_screen",
            message="no complete relation",
        )


def test_pickplace_adapter_runs_one_atom_and_emits_json_result() -> None:
    target = np.eye(4)
    target[0, 3] = 0.3
    atom = ManipulationAtom(
        "atom_01",
        ManipulationPrimitive.PICK_PLACE,
        "carrot_1",
        "carriot",
        target,
    )
    plan = ManipulationPlan("demo", "/tmp/scene.json", (atom,))
    initial = TaskSceneState(
        {"carrot_1": SceneObjectState("carrot_1", "carriot", np.eye(4))},
        joint_names=("j1", "j2"),
        joint_positions=[0.0, 0.0],
    )
    coordinator = FakeCoordinator()
    synchronized = []
    backend = PickPlaceAtomBackend(
        coordinator,
        lambda current_atom, scene: (current_atom.atom_id, scene.revision),
        initial_scene=initial,
        observe_object_pose=lambda object_id: target,
        synchronize_scene=lambda scene: synchronized.append(scene.revision),
    )
    result = MultiObjectTaskExecutor(backend).run(plan)
    assert result.success
    assert coordinator.tasks == [("atom_01", 0)]
    assert result.final_scene.joint_positions.tolist() == [0.4, 0.5]
    assert synchronized == [0, 1]
    assert json.loads(json.dumps(result.as_dict()))["records"][0]["status"] == "succeeded"


def test_relation_screen_failure_is_not_retried() -> None:
    target = np.eye(4)
    atom = ManipulationAtom(
        "atom_01", ManipulationPrimitive.PICK_PLACE, "pen", "bi", target
    )
    plan = ManipulationPlan("demo", "/tmp/scene.json", (atom,))
    initial = TaskSceneState(
        {"pen": SceneObjectState("pen", "bi", np.eye(4))},
        joint_names=("j1", "j2"),
        joint_positions=[0.0, 0.0],
    )
    backend = PickPlaceAtomBackend(
        InfeasibleCoordinator(),
        lambda current_atom, scene: (current_atom.atom_id, scene.revision),
        initial_scene=initial,
    )

    result = MultiObjectTaskExecutor(backend, max_attempts_per_atom=3).run(plan)

    assert not result.success
    assert len(result.records[0].attempts) == 1
    assert result.records[0].attempts[0].execution.failure_code == "relation_infeasible"
