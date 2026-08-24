"""Backend-neutral scheduler for multi-object manipulation plans."""

from __future__ import annotations

import copy
import json
import math
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from rm75_app.tasks.manipulation_plan import ManipulationAtom, ManipulationPlan


class ObjectLifecycle(str, Enum):
    AVAILABLE = "available"
    HELD = "held"
    PLACED = "placed"
    LOST = "lost"


class AtomStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class RecoveryAction(str, Enum):
    RETRY = "retry"
    REFRESH_AND_RETRY = "refresh_and_retry"
    ABORT = "abort"


def _matrix(value: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("object pose must be a finite 4x4 matrix")
    return matrix.copy()


@dataclass
class SceneObjectState:
    object_id: str
    asset_name: str
    pose: np.ndarray
    lifecycle: ObjectLifecycle = ObjectLifecycle.AVAILABLE
    movable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.object_id or not self.asset_name:
            raise ValueError("scene object id and asset name must not be empty")
        self.pose = _matrix(self.pose)
        self.lifecycle = ObjectLifecycle(self.lifecycle)
        self.metadata = dict(self.metadata)

    def copy(self) -> "SceneObjectState":
        return SceneObjectState(
            self.object_id,
            self.asset_name,
            self.pose.copy(),
            self.lifecycle,
            self.movable,
            copy.deepcopy(self.metadata),
        )


@dataclass
class TaskSceneState:
    objects: dict[str, SceneObjectState]
    revision: int = 0
    backend_revision: str | None = None
    joint_names: tuple[str, ...] = ()
    joint_positions: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.objects = {str(key): value.copy() for key, value in self.objects.items()}
        if set(self.objects) != {item.object_id for item in self.objects.values()}:
            raise ValueError("scene object mapping keys must match object ids")
        self.joint_names = tuple(str(name) for name in self.joint_names)
        if self.joint_positions is not None:
            positions = np.asarray(self.joint_positions, dtype=np.float64).reshape(-1)
            if self.joint_names and len(positions) != len(self.joint_names):
                raise ValueError("joint names and positions have different lengths")
            self.joint_positions = positions.copy()
        self.metadata = dict(self.metadata)

    def copy(self) -> "TaskSceneState":
        return TaskSceneState(
            self.objects,
            self.revision,
            self.backend_revision,
            self.joint_names,
            None if self.joint_positions is None else self.joint_positions.copy(),
            copy.deepcopy(self.metadata),
        )

    def commit_object_pose(
        self,
        object_id: str,
        pose: Sequence[Sequence[float]],
        *,
        lifecycle: ObjectLifecycle = ObjectLifecycle.PLACED,
    ) -> None:
        if object_id not in self.objects:
            raise KeyError(f"object {object_id!r} is not in the task scene")
        self.objects[object_id].pose = _matrix(pose)
        self.objects[object_id].lifecycle = lifecycle
        self.revision += 1

    def set_joints(self, names: Sequence[str], positions: Sequence[float]) -> None:
        names = tuple(str(name) for name in names)
        values = np.asarray(positions, dtype=np.float64).reshape(-1)
        if len(names) != len(values):
            raise ValueError("joint names and positions have different lengths")
        self.joint_names = names
        self.joint_positions = values.copy()

    def as_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "backend_revision": self.backend_revision,
            "joint_names": list(self.joint_names),
            "joint_positions": None if self.joint_positions is None else self.joint_positions.tolist(),
            "objects": {
                key: {
                    "object_id": value.object_id,
                    "asset_name": value.asset_name,
                    "pose": value.pose.tolist(),
                    "lifecycle": value.lifecycle.value,
                    "movable": value.movable,
                    "metadata": copy.deepcopy(value.metadata),
                }
                for key, value in self.objects.items()
            },
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class AtomExecution:
    success: bool
    final_object_pose: Sequence[Sequence[float]] | None = None
    joint_names: tuple[str, ...] = ()
    joint_positions: Sequence[float] | None = None
    failure_code: str | None = None
    failure_stage: str | None = None
    message: str = ""
    recoverable: bool = True
    artifacts: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.success and self.final_object_pose is None:
            raise ValueError("successful atom execution must report the final object pose")
        if self.final_object_pose is not None:
            object.__setattr__(self, "final_object_pose", _matrix(self.final_object_pose))
        names = tuple(str(name) for name in self.joint_names)
        object.__setattr__(self, "joint_names", names)
        if self.joint_positions is not None:
            positions = np.asarray(self.joint_positions, dtype=np.float64).reshape(-1)
            if names and len(names) != len(positions):
                raise ValueError("execution joint names and positions have different lengths")
            object.__setattr__(self, "joint_positions", positions)
        object.__setattr__(self, "artifacts", dict(self.artifacts))


@dataclass(frozen=True)
class AtomValidation:
    success: bool
    relation: str
    position_error_m: float | None = None
    orientation_error_deg: float | None = None
    message: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    observed_object_pose: Sequence[Sequence[float]] | None = None

    def __post_init__(self) -> None:
        if self.observed_object_pose is not None:
            object.__setattr__(self, "observed_object_pose", _matrix(self.observed_object_pose))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@dataclass(frozen=True)
class RecoveryDirective:
    action: RecoveryAction
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", RecoveryAction(self.action))


@dataclass(frozen=True)
class AtomAttemptRecord:
    attempt: int
    scene_revision: int
    execution: AtomExecution
    validation: AtomValidation | None = None
    recovery: RecoveryDirective | None = None


@dataclass(frozen=True)
class AtomRunRecord:
    atom_id: str
    status: AtomStatus
    attempts: tuple[AtomAttemptRecord, ...] = ()
    message: str = ""


@dataclass(frozen=True)
class MultiObjectRunResult:
    plan_id: str
    success: bool
    records: tuple[AtomRunRecord, ...]
    final_scene: TaskSceneState
    failure_atom_id: str | None = None
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "success": self.success,
            "failure_atom_id": self.failure_atom_id,
            "message": self.message,
            "records": [
                {
                    "atom_id": record.atom_id,
                    "status": record.status.value,
                    "message": record.message,
                    "attempts": [
                        {
                            "attempt": attempt.attempt,
                            "scene_revision": attempt.scene_revision,
                            "execution": {
                                "success": attempt.execution.success,
                                "failure_code": attempt.execution.failure_code,
                                "failure_stage": attempt.execution.failure_stage,
                                "message": attempt.execution.message,
                                "recoverable": attempt.execution.recoverable,
                                "final_object_pose": None
                                if attempt.execution.final_object_pose is None
                                else np.asarray(attempt.execution.final_object_pose).tolist(),
                                "artifacts": dict(attempt.execution.artifacts),
                            },
                            "validation": None
                            if attempt.validation is None
                            else {
                                "success": attempt.validation.success,
                                "relation": attempt.validation.relation,
                                "position_error_m": attempt.validation.position_error_m,
                                "orientation_error_deg": attempt.validation.orientation_error_deg,
                                "message": attempt.validation.message,
                                "diagnostics": dict(attempt.validation.diagnostics),
                                "observed_object_pose": None
                                if attempt.validation.observed_object_pose is None
                                else np.asarray(attempt.validation.observed_object_pose).tolist(),
                            },
                            "recovery": None
                            if attempt.recovery is None
                            else {
                                "action": attempt.recovery.action.value,
                                "reason": attempt.recovery.reason,
                            },
                        }
                        for attempt in record.attempts
                    ],
                }
                for record in self.records
            ],
            "final_scene": self.final_scene.as_dict(),
        }


class AtomicTaskBackend(Protocol):
    """Capabilities required from ManiSkill or a real-robot adapter."""

    def initialize(self, plan: ManipulationPlan) -> TaskSceneState: ...

    def execute_atom(
        self,
        atom: ManipulationAtom,
        scene: TaskSceneState,
        *,
        attempt: int,
    ) -> AtomExecution: ...

    def validate_atom(
        self,
        atom: ManipulationAtom,
        execution: AtomExecution,
        scene: TaskSceneState,
    ) -> AtomValidation: ...

    def recover_atom(
        self,
        atom: ManipulationAtom,
        execution: AtomExecution,
        validation: AtomValidation | None,
        scene: TaskSceneState,
        *,
        attempt: int,
    ) -> RecoveryDirective: ...

    def refresh_scene(self, scene: TaskSceneState, *, reason: str) -> TaskSceneState: ...

    def synchronize_scene(self, scene: TaskSceneState) -> None: ...


def pose_error(
    actual: Sequence[Sequence[float]], expected: Sequence[Sequence[float]]
) -> tuple[float, float]:
    actual_matrix = _matrix(actual)
    expected_matrix = _matrix(expected)
    position = float(np.linalg.norm(actual_matrix[:3, 3] - expected_matrix[:3, 3]))
    relative = expected_matrix[:3, :3].T @ actual_matrix[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    orientation = math.degrees(math.acos(cosine))
    return position, orientation


def symmetry_aware_pose_error(
    actual: Sequence[Sequence[float]],
    expected: Sequence[Sequence[float]],
    asset_name: str,
) -> tuple[float, float]:
    """Measure only orientation components that are meaningful for an asset."""

    from rm75_app.assets.object_specs import get_object_spec

    actual_matrix = _matrix(actual)
    expected_matrix = _matrix(expected)
    position = float(np.linalg.norm(actual_matrix[:3, 3] - expected_matrix[:3, 3]))
    spec = get_object_spec(asset_name)
    symmetry = "none" if spec is None else str(spec.orientation_symmetry)
    if symmetry == "spherical":
        return position, 0.0
    if symmetry in {"axial", "axial_bidirectional"}:
        local_axis = np.asarray(
            spec.symmetry_axis_local or (0.0, 1.0, 0.0), dtype=np.float64
        ).reshape(3)
        local_axis /= np.linalg.norm(local_axis)
        actual_axis = actual_matrix[:3, :3] @ local_axis
        expected_axis = expected_matrix[:3, :3] @ local_axis
        cosine = float(np.clip(np.dot(actual_axis, expected_axis), -1.0, 1.0))
        if symmetry == "axial_bidirectional":
            cosine = abs(cosine)
        return position, math.degrees(math.acos(cosine))
    return pose_error(actual_matrix, expected_matrix)


def validate_target_pose(atom: ManipulationAtom, execution: AtomExecution) -> AtomValidation:
    if execution.final_object_pose is None:
        return AtomValidation(False, atom.success.relation, message="execution did not report object pose")
    position, orientation = symmetry_aware_pose_error(
        execution.final_object_pose, atom.target_pose, atom.object_asset
    )
    position_ok = (
        atom.success.position_tolerance_m is None
        or position <= atom.success.position_tolerance_m
    )
    orientation_ok = (
        atom.success.orientation_tolerance_deg is None
        or orientation <= atom.success.orientation_tolerance_deg
    )
    return AtomValidation(
        position_ok and orientation_ok,
        atom.success.relation,
        position_error_m=position,
        orientation_error_deg=orientation,
        message="target condition satisfied" if position_ok and orientation_ok else "target pose outside tolerance",
        observed_object_pose=execution.final_object_pose,
    )


def load_task_scene(path: str) -> TaskSceneState:
    """Load the object subset shared by LLM fixed-scene JSON variants."""

    from pathlib import Path

    scene_path = Path(path).expanduser().resolve()
    payload = json.loads(scene_path.read_text(encoding="utf-8"))
    raw_objects = payload.get("objects") or payload.get("scene_objects") or {}
    if isinstance(raw_objects, list):
        entries = [(str(item.get("object_id") or item.get("name") or ""), item) for item in raw_objects]
    elif isinstance(raw_objects, dict):
        entries = [(str(key), value) for key, value in raw_objects.items()]
    else:
        raise ValueError(f"scene {scene_path} has no object collection")
    objects: dict[str, SceneObjectState] = {}
    for key, raw in entries:
        if not key or not isinstance(raw, dict):
            continue
        transform = raw.get("T_world_obj") or raw.get("T_base_obj") or raw.get("transform")
        if transform is None and raw.get("pose") is not None:
            pose = np.asarray(raw["pose"], dtype=np.float64).reshape(-1)
            if len(pose) >= 7:
                matrix = np.eye(4, dtype=np.float64)
                matrix[:3, 3] = pose[:3]
                quaternion = pose[3:7] / np.linalg.norm(pose[3:7])
                w, x, y, z = quaternion
                matrix[:3, :3] = np.asarray(
                    [
                        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    ]
                )
                transform = matrix
        if transform is None:
            continue
        asset = str(
            raw.get("spec_name")
            or raw.get("asset_name")
            or raw.get("name")
            or key
        )
        objects[key] = SceneObjectState(
            key,
            asset,
            transform,
            movable=not bool(raw.get("fixed", False)),
            metadata={"source_scene_entry": raw},
        )
    if not objects:
        raise ValueError(f"scene {scene_path} has no usable object poses")
    return TaskSceneState(
        objects,
        backend_revision=str(payload.get("revision") or payload.get("schema") or scene_path.stem),
        metadata={"scene_file": str(scene_path)},
    )


class MultiObjectTaskExecutor:
    """Execute dependency-ordered atoms while maintaining one authoritative scene."""

    def __init__(
        self,
        backend: AtomicTaskBackend,
        *,
        max_attempts_per_atom: int = 2,
        stop_on_failure: bool = True,
    ):
        if max_attempts_per_atom < 1:
            raise ValueError("max_attempts_per_atom must be at least one")
        self.backend = backend
        self.max_attempts_per_atom = int(max_attempts_per_atom)
        self.stop_on_failure = bool(stop_on_failure)

    def run(self, plan: ManipulationPlan) -> MultiObjectRunResult:
        scene = self.backend.initialize(plan)
        self._validate_scene(plan, scene)
        records: list[AtomRunRecord] = []
        status: dict[str, AtomStatus] = {}
        failure_atom_id: str | None = None

        for atom in plan.atoms:
            blocked = [dep for dep in atom.depends_on if status.get(dep) is not AtomStatus.SUCCEEDED]
            if blocked:
                record = AtomRunRecord(
                    atom.atom_id,
                    AtomStatus.SKIPPED,
                    message=f"dependencies did not succeed: {blocked}",
                )
                records.append(record)
                status[atom.atom_id] = record.status
                continue

            obj = scene.objects[atom.object_id]
            if obj.lifecycle is ObjectLifecycle.LOST:
                record = AtomRunRecord(atom.atom_id, AtomStatus.FAILED, message="object is lost")
                records.append(record)
                status[atom.atom_id] = record.status
                failure_atom_id = failure_atom_id or atom.atom_id
                if self.stop_on_failure:
                    break
                continue

            attempts: list[AtomAttemptRecord] = []
            succeeded = False
            final_message = ""
            for attempt in range(1, self.max_attempts_per_atom + 1):
                scene_revision = scene.revision
                try:
                    execution = self.backend.execute_atom(atom, scene.copy(), attempt=attempt)
                except Exception as exc:
                    execution = AtomExecution(
                        False,
                        failure_code="backend_exception",
                        failure_stage="execute_atom",
                        message=f"{type(exc).__name__}: {exc}",
                        recoverable=False,
                        artifacts={"traceback": traceback.format_exc(limit=12)},
                    )

                validation = None
                if execution.success:
                    try:
                        validation = self.backend.validate_atom(atom, execution, scene.copy())
                    except Exception as exc:
                        validation = AtomValidation(
                            False,
                            atom.success.relation,
                            message=f"validator exception: {type(exc).__name__}: {exc}",
                        )
                    if validation.success:
                        committed_pose = (
                            validation.observed_object_pose
                            if validation.observed_object_pose is not None
                            else execution.final_object_pose
                        )
                        scene.commit_object_pose(atom.object_id, committed_pose)
                        if execution.joint_positions is not None:
                            scene.set_joints(execution.joint_names, execution.joint_positions)
                        self.backend.synchronize_scene(scene.copy())
                        attempts.append(AtomAttemptRecord(attempt, scene_revision, execution, validation))
                        succeeded = True
                        final_message = validation.message
                        break

                directive = self.backend.recover_atom(
                    atom,
                    execution,
                    validation,
                    scene.copy(),
                    attempt=attempt,
                )
                attempts.append(AtomAttemptRecord(attempt, scene_revision, execution, validation, directive))
                final_message = directive.reason or execution.message or (
                    validation.message if validation is not None else "atom failed"
                )
                if (
                    directive.action is RecoveryAction.ABORT
                    or not execution.recoverable
                    or attempt >= self.max_attempts_per_atom
                ):
                    break
                if directive.action is RecoveryAction.REFRESH_AND_RETRY:
                    refreshed = self.backend.refresh_scene(scene.copy(), reason=directive.reason)
                    self._validate_scene(plan, refreshed)
                    scene = refreshed

            atom_status = AtomStatus.SUCCEEDED if succeeded else AtomStatus.FAILED
            records.append(AtomRunRecord(atom.atom_id, atom_status, tuple(attempts), final_message))
            status[atom.atom_id] = atom_status
            if not succeeded:
                failure_atom_id = failure_atom_id or atom.atom_id
                if self.stop_on_failure:
                    break

        # Preserve a complete per-atom result even when fail-fast stops early.
        recorded = {record.atom_id for record in records}
        for atom in plan.atoms:
            if atom.atom_id not in recorded:
                records.append(AtomRunRecord(atom.atom_id, AtomStatus.SKIPPED, message="stopped after previous failure"))

        success = all(record.status is AtomStatus.SUCCEEDED for record in records)
        return MultiObjectRunResult(
            plan.plan_id,
            success,
            tuple(records),
            scene.copy(),
            failure_atom_id=failure_atom_id,
            message="all atoms succeeded" if success else "task did not complete",
        )

    @staticmethod
    def _validate_scene(plan: ManipulationPlan, scene: TaskSceneState) -> None:
        missing = sorted({atom.object_id for atom in plan.atoms} - set(scene.objects))
        if missing:
            raise ValueError(f"task scene is missing manipulated objects: {missing}")
