"""Thin ManiSkill hooks for the shared multi-object task executor."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from rm75_app.orchestration.multi_object_executor import (
    AtomExecution,
    AtomValidation,
    TaskSceneState,
    validate_target_pose,
)
from rm75_app.tasks.manipulation_plan import ManipulationAtom


def _pose_matrix(actor: Any) -> np.ndarray:
    pose = getattr(actor, "pose", None)
    matrix = getattr(pose, "to_transformation_matrix", None)
    if not callable(matrix):
        raise TypeError("ManiSkill actor pose has no transformation matrix")
    value = matrix()
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 3:
        value = value[0]
    return value.reshape(4, 4)


class ManiSkillTaskBridge:
    """Observe actors and settle physics without coupling the scheduler to SAPIEN."""

    def __init__(
        self,
        env: Any,
        actors: Mapping[str, Any] | Callable[[str], Any],
        *,
        settle_steps: int | None = None,
        hold_action: Callable[[], np.ndarray] | None = None,
    ):
        self.env = env
        self.actors = actors
        self.settle_steps = settle_steps
        self.hold_action = hold_action

    def actor(self, object_id: str) -> Any:
        actor = self.actors(object_id) if callable(self.actors) else self.actors.get(object_id)
        if actor is None:
            raise KeyError(f"ManiSkill scene has no actor for {object_id!r}")
        return actor

    def observe_object_pose(self, object_id: str) -> np.ndarray:
        return _pose_matrix(self.actor(object_id))

    def validate_atom(
        self,
        atom: ManipulationAtom,
        execution: AtomExecution,
        scene: TaskSceneState,
    ) -> AtomValidation:
        del scene
        steps = atom.success.settle_steps if self.settle_steps is None else int(self.settle_steps)
        if steps > 0:
            self._settle(steps)
        observed = AtomExecution(
            True,
            final_object_pose=self.observe_object_pose(atom.object_id),
            joint_names=execution.joint_names,
            joint_positions=execution.joint_positions,
            artifacts=execution.artifacts,
        )
        return validate_target_pose(atom, observed)

    def synchronize_scene(self, scene: TaskSceneState) -> None:
        # ManiSkill is authoritative after physics execution. This hook checks
        # that every manipulated actor remains addressable; planner-world sync
        # is supplied separately by the task builder/backend.
        for object_id in scene.objects:
            self.actor(object_id)

    def _settle(self, steps: int) -> None:
        if self.hold_action is None:
            raise RuntimeError(
                "ManiSkillTaskBridge needs a controller-specific hold_action callback for settling"
            )
        for _ in range(int(steps)):
            self.env.step(np.asarray(self.hold_action(), dtype=np.float32))
