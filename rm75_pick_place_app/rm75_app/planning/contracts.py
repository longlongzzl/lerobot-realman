"""Planner-independent data contracts for manipulation planning.

The rest of the application must not exchange cuRobo tensors or result objects.
Keeping this boundary in NumPy makes cached replay, simulation and alternative
planners use the exact same coordinator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np


def _vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


@dataclass(frozen=True)
class Pose:
    """Rigid pose using metres and a scalar-first ``wxyz`` quaternion."""

    position: Sequence[float]
    quaternion_wxyz: Sequence[float]

    def __post_init__(self) -> None:
        position = _vector(self.position, 3, "position")
        quaternion = _vector(self.quaternion_wxyz, 4, "quaternion_wxyz")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-8:
            raise ValueError("quaternion_wxyz must not be zero")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion_wxyz", quaternion / norm)

    def as_curobo_list(self) -> list[float]:
        return [*self.position.tolist(), *self.quaternion_wxyz.tolist()]


@dataclass(frozen=True)
class JointConfiguration:
    names: tuple[str, ...]
    positions: Sequence[float]

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.names)
        positions = _vector(self.positions, len(names), "joint positions")
        if len(set(names)) != len(names):
            raise ValueError("joint names must be unique")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "positions", positions)


ObstacleKind = Literal["cuboid", "sphere", "mesh"]


@dataclass(frozen=True)
class CollisionObject:
    name: str
    kind: ObstacleKind
    pose: Pose
    dimensions: Sequence[float] | None = None
    radius: float | None = None
    mesh_path: str | Path | None = None
    scale: Sequence[float] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("collision object name must not be empty")
        if self.kind == "cuboid":
            dimensions = _vector(self.dimensions or (), 3, "cuboid dimensions")
            if np.any(dimensions <= 0.0):
                raise ValueError("cuboid dimensions must be positive")
            object.__setattr__(self, "dimensions", dimensions)
        elif self.kind == "sphere":
            if self.radius is None or not np.isfinite(self.radius) or self.radius <= 0.0:
                raise ValueError("sphere radius must be positive")
        elif self.kind == "mesh":
            if self.mesh_path is None:
                raise ValueError("mesh_path is required for a mesh obstacle")
            object.__setattr__(self, "mesh_path", Path(self.mesh_path).expanduser().resolve())
            if self.scale is not None:
                object.__setattr__(self, "scale", _vector(self.scale, 3, "mesh scale"))
        else:
            raise ValueError(f"unsupported collision object kind: {self.kind}")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class PlanningScene:
    objects: tuple[CollisionObject, ...] = ()
    revision: str | None = None

    def __post_init__(self) -> None:
        objects = tuple(self.objects)
        names = [item.name for item in objects]
        if len(set(names)) != len(names):
            raise ValueError("collision object names must be unique")
        object.__setattr__(self, "objects", objects)


@dataclass(frozen=True)
class PoseCandidate:
    candidate_id: str
    pose: Pose
    score: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if not np.isfinite(self.score):
            raise ValueError("candidate score must be finite")


@dataclass(frozen=True)
class BatchPlanningRequest:
    current: JointConfiguration
    candidates: tuple[PoseCandidate, ...]
    scene: PlanningScene = field(default_factory=PlanningScene)
    tool_frame: str = "gripper_tcp"
    max_attempts: int = 2
    prefer_unbiased_ik: bool = False

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("at least one pose candidate is required")
        identifiers = [item.candidate_id for item in candidates]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("candidate identifiers must be unique")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        object.__setattr__(self, "candidates", candidates)


@dataclass(frozen=True)
class JointTrajectory:
    joint_names: tuple[str, ...]
    positions: np.ndarray
    dt: float | np.ndarray | None = None

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != len(self.joint_names):
            raise ValueError(
                "trajectory positions must have shape [time, joints], "
                f"got {positions.shape} for {len(self.joint_names)} joints"
            )
        if not np.all(np.isfinite(positions)):
            raise ValueError("trajectory contains a non-finite value")
        object.__setattr__(self, "positions", positions)


@dataclass(frozen=True)
class CandidatePlan:
    candidate_id: str
    success: bool
    trajectory: JointTrajectory | None = None
    position_error: float | None = None
    orientation_error: float | None = None
    solve_time: float = 0.0
    status: str | None = None


@dataclass(frozen=True)
class BatchPlanningResult:
    plans: tuple[CandidatePlan, ...]
    backend: str
    total_time: float = 0.0

    @property
    def successful(self) -> tuple[CandidatePlan, ...]:
        return tuple(plan for plan in self.plans if plan.success)

    def best(self, candidates: Sequence[PoseCandidate]) -> CandidatePlan | None:
        scores = {candidate.candidate_id: candidate.score for candidate in candidates}
        feasible = self.successful
        if not feasible:
            return None
        return max(feasible, key=lambda plan: scores.get(plan.candidate_id, float("-inf")))


@dataclass(frozen=True)
class GraspPlanningRequest:
    planning: BatchPlanningRequest
    target_object_name: str | None = None
    approach_axis: Literal["x", "y", "z"] = "z"
    approach_offset: float = -0.10
    approach_in_tool_frame: bool = True


@dataclass(frozen=True)
class GraspCandidatePlan:
    candidate_id: str
    success: bool
    approach: JointTrajectory | None = None
    grasp: JointTrajectory | None = None
    status: str | None = None

    @property
    def grasp_configuration(self) -> JointConfiguration | None:
        if self.grasp is None or len(self.grasp.positions) == 0:
            return None
        return JointConfiguration(self.grasp.joint_names, self.grasp.positions[-1])


@dataclass(frozen=True)
class GraspPlanningResult:
    plans: tuple[GraspCandidatePlan, ...]
    backend: str
    total_time: float = 0.0

    def best(self, candidates: Sequence[PoseCandidate]) -> GraspCandidatePlan | None:
        scores = {candidate.candidate_id: candidate.score for candidate in candidates}
        feasible = tuple(plan for plan in self.plans if plan.success)
        if not feasible:
            return None
        return max(feasible, key=lambda plan: scores.get(plan.candidate_id, float("-inf")))
