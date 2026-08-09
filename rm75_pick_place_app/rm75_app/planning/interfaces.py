"""Public planning backend interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .contracts import (
    BatchPlanningRequest,
    BatchPlanningResult,
    GraspPlanningRequest,
    GraspPlanningResult,
    JointConfiguration,
    PlanningScene,
    Pose,
)


@runtime_checkable
class PlanningBackend(Protocol):
    name: str

    def update_scene(self, scene: PlanningScene) -> None: ...

    def plan_candidates(self, request: BatchPlanningRequest) -> BatchPlanningResult: ...

    def plan_grasps(self, request: GraspPlanningRequest) -> GraspPlanningResult: ...

    def attach_object(self, object_name: str, grasp: JointConfiguration) -> None: ...

    def update_attached_object_pose(
        self,
        object_name: str,
        current: JointConfiguration,
        T_tcp_object: np.ndarray,
    ) -> None: ...

    def detach_object(self, object_name: str, released_pose: Pose | None = None) -> None: ...

    def enable_object_collision(self, object_name: str) -> None: ...

    def close(self) -> None: ...
