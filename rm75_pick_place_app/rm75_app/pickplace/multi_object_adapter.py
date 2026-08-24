"""Adapt the single-cycle PickPlaceCoordinator to task atoms."""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from rm75_app.orchestration.multi_object_executor import (
    AtomExecution,
    AtomValidation,
    RecoveryAction,
    RecoveryDirective,
    TaskSceneState,
    load_task_scene,
    validate_target_pose,
)
from rm75_app.pickplace.coordinator import PickPlaceCoordinator, PickPlaceTask
from rm75_app.tasks.manipulation_plan import ManipulationAtom, ManipulationPlan


class PickPlaceTaskBuilder(Protocol):
    def __call__(self, atom: ManipulationAtom, scene: TaskSceneState) -> PickPlaceTask: ...


ObjectPoseObserver = Callable[[str], np.ndarray]
SceneRefresh = Callable[[TaskSceneState, str], TaskSceneState]
SceneSynchronizer = Callable[[TaskSceneState], None]
AtomValidator = Callable[[ManipulationAtom, AtomExecution, TaskSceneState], AtomValidation]


class PickPlaceAtomBackend:
    """Run one manipulation atom through the existing coordinator.

    ManiSkill supplies ``observe_object_pose`` from actor state. A real robot
    supplies the same callback from RRTrack/re-perception. Without an observer,
    this adapter is suitable only for planner/headless tests and uses the
    commanded target as the reported final pose.
    """

    def __init__(
        self,
        coordinator: PickPlaceCoordinator,
        task_builder: PickPlaceTaskBuilder,
        *,
        initial_scene: TaskSceneState | None = None,
        observe_object_pose: ObjectPoseObserver | None = None,
        refresh_scene: SceneRefresh | None = None,
        synchronize_scene: SceneSynchronizer | None = None,
        validator: AtomValidator | None = None,
    ):
        self.coordinator = coordinator
        self.task_builder = task_builder
        self.initial_scene = initial_scene.copy() if initial_scene is not None else None
        self.observe_object_pose = observe_object_pose
        self._refresh_scene = refresh_scene
        self._synchronize_scene = synchronize_scene
        self._validator = validator

    def initialize(self, plan: ManipulationPlan) -> TaskSceneState:
        scene = self.initial_scene.copy() if self.initial_scene is not None else load_task_scene(plan.scene_file)
        self.synchronize_scene(scene.copy())
        return scene

    def execute_atom(
        self,
        atom: ManipulationAtom,
        scene: TaskSceneState,
        *,
        attempt: int,
    ) -> AtomExecution:
        task = self.task_builder(atom, scene)
        trajectory_executor = getattr(self.coordinator, "executor", None)
        begin_atom = getattr(trajectory_executor, "begin_atom", None)
        end_atom = getattr(trajectory_executor, "end_atom", None)
        if callable(begin_atom):
            begin_atom(atom.atom_id)
        result = self.coordinator.run(task)
        if callable(end_atom):
            end_atom(atom.atom_id, success=result.success)
        last = result.stages[-1].end if result.stages else None
        if not result.success:
            return AtomExecution(
                False,
                joint_names=() if last is None else last.names,
                joint_positions=None if last is None else last.positions,
                failure_code=self._failure_code(result.failure_stage),
                failure_stage=result.failure_stage,
                message=result.message or "pick-place cycle failed",
                recoverable=result.failure_stage not in {"retreat", "relation_screen"},
                artifacts={
                    "attempt": attempt,
                    "selected_grasp": result.selected_grasp,
                    "selected_place": result.selected_place,
                    "diagnostics": dict(result.diagnostics),
                },
            )
        pose = (
            np.asarray(self.observe_object_pose(atom.object_id), dtype=np.float64).reshape(4, 4)
            if self.observe_object_pose is not None
            else np.asarray(atom.target_pose, dtype=np.float64).reshape(4, 4)
        )
        return AtomExecution(
            True,
            final_object_pose=pose,
            joint_names=() if last is None else last.names,
            joint_positions=None if last is None else last.positions,
            artifacts={
                "attempt": attempt,
                "selected_grasp": result.selected_grasp,
                "selected_place": result.selected_place,
                "pose_source": "observer" if self.observe_object_pose is not None else "commanded_target",
                "diagnostics": dict(result.diagnostics),
            },
        )

    def validate_atom(
        self,
        atom: ManipulationAtom,
        execution: AtomExecution,
        scene: TaskSceneState,
    ) -> AtomValidation:
        if self._validator is not None:
            return self._validator(atom, execution, scene)
        return validate_target_pose(atom, execution)

    def recover_atom(
        self,
        atom: ManipulationAtom,
        execution: AtomExecution,
        validation: AtomValidation | None,
        scene: TaskSceneState,
        *,
        attempt: int,
    ) -> RecoveryDirective:
        del atom, scene, attempt
        if not execution.recoverable or execution.failure_code == "backend_exception":
            return RecoveryDirective(RecoveryAction.ABORT, execution.message)
        if execution.failure_code in {"grasp_failed", "object_lost"}:
            return RecoveryDirective(RecoveryAction.REFRESH_AND_RETRY, execution.message or "refresh object pose")
        if execution.success and validation is not None and not validation.success:
            return RecoveryDirective(RecoveryAction.REFRESH_AND_RETRY, validation.message)
        return RecoveryDirective(RecoveryAction.RETRY, execution.message or "retry with another candidate")

    def refresh_scene(self, scene: TaskSceneState, *, reason: str) -> TaskSceneState:
        if self._refresh_scene is None:
            return scene.copy()
        return self._refresh_scene(scene.copy(), reason)

    def synchronize_scene(self, scene: TaskSceneState) -> None:
        if self._synchronize_scene is not None:
            self._synchronize_scene(scene.copy())

    @staticmethod
    def _failure_code(stage: str | None) -> str:
        return {
            "grasp": "grasp_failed",
            "lift": "lift_planning_failed",
            "preplace": "transport_planning_failed",
            "place": "place_planning_failed",
            "retreat": "unsafe_retreat",
            "relation_screen": "relation_infeasible",
        }.get(str(stage), "pick_place_failed")
