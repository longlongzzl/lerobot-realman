"""Backend-neutral manipulation task protocol.

Natural-language operators such as ``exchange`` and ``collection`` are macros.
They are expanded before a planner, simulator, or real robot receives the
task. One atom describes one physical manipulation of one object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "rm75.manipulation_plan/v1"


class ManipulationPrimitive(str, Enum):
    """Physical capabilities exposed by an execution backend."""

    PICK_PLACE = "pick_place"
    INSERT = "insert"
    SNAP_PLACE = "snap_place"


class PlacementMode(str, Enum):
    AUTO = "auto"
    SURFACE = "surface_place"
    DROP = "drop_place"
    VERTICAL = "vertical_place"


def _pose_matrix(value: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"target_pose must have shape (4, 4), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("target_pose contains a non-finite value")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("target_pose is not a homogeneous transform")
    return tuple(tuple(float(item) for item in row) for row in matrix)


@dataclass(frozen=True)
class SuccessCriteria:
    """Observable condition used by simulation and real-world validation."""

    relation: str
    position_tolerance_m: float | None = None
    orientation_tolerance_deg: float | None = None
    settle_steps: int = 20

    def __post_init__(self) -> None:
        if not self.relation:
            raise ValueError("success relation must not be empty")
        if self.position_tolerance_m is not None and self.position_tolerance_m <= 0.0:
            raise ValueError("position_tolerance_m must be positive")
        if self.orientation_tolerance_deg is not None and self.orientation_tolerance_deg <= 0.0:
            raise ValueError("orientation_tolerance_deg must be positive")
        if self.settle_steps < 0:
            raise ValueError("settle_steps must not be negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation,
            "position_tolerance_m": self.position_tolerance_m,
            "orientation_tolerance_deg": self.orientation_tolerance_deg,
            "settle_steps": self.settle_steps,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SuccessCriteria":
        return cls(
            relation=str(value["relation"]),
            position_tolerance_m=value.get("position_tolerance_m"),
            orientation_tolerance_deg=value.get("orientation_tolerance_deg"),
            settle_steps=int(value.get("settle_steps", 20)),
        )


@dataclass(frozen=True)
class ManipulationAtom:
    """One indivisible object manipulation after macro expansion."""

    atom_id: str
    primitive: ManipulationPrimitive
    object_id: str
    object_asset: str
    target_pose: Sequence[Sequence[float]]
    support_object_id: str | None = None
    placement_mode: PlacementMode = PlacementMode.SURFACE
    semantic_operator: str = "pose_goal"
    success: SuccessCriteria = field(
        default_factory=lambda: SuccessCriteria("target_pose", 0.02, 20.0)
    )
    depends_on: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.atom_id or not self.object_id or not self.object_asset:
            raise ValueError("atom_id, object_id and object_asset must not be empty")
        object.__setattr__(self, "primitive", ManipulationPrimitive(self.primitive))
        object.__setattr__(self, "placement_mode", PlacementMode(self.placement_mode))
        object.__setattr__(self, "target_pose", _pose_matrix(self.target_pose))
        object.__setattr__(self, "depends_on", tuple(str(item) for item in self.depends_on))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "primitive": self.primitive.value,
            "object_id": self.object_id,
            "object_asset": self.object_asset,
            "support_object_id": self.support_object_id,
            "target_pose": [list(row) for row in self.target_pose],
            "placement_mode": self.placement_mode.value,
            "semantic_operator": self.semantic_operator,
            "success": self.success.as_dict(),
            "depends_on": list(self.depends_on),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ManipulationAtom":
        return cls(
            atom_id=str(value["atom_id"]),
            primitive=ManipulationPrimitive(str(value["primitive"])),
            object_id=str(value["object_id"]),
            object_asset=str(value["object_asset"]),
            support_object_id=value.get("support_object_id"),
            target_pose=value["target_pose"],
            placement_mode=PlacementMode(str(value.get("placement_mode", PlacementMode.SURFACE.value))),
            semantic_operator=str(value.get("semantic_operator", "pose_goal")),
            success=SuccessCriteria.from_dict(value.get("success") or {"relation": "target_pose"}),
            depends_on=tuple(value.get("depends_on") or ()),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ManipulationPlan:
    """The task package handed to ManiSkill, Curobo2, or a real executor."""

    plan_id: str
    scene_file: str
    atoms: tuple[ManipulationAtom, ...]
    user_command: str = ""
    schema_version: str = SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported manipulation plan schema {self.schema_version!r}")
        if not self.plan_id or not self.scene_file:
            raise ValueError("plan_id and scene_file must not be empty")
        atoms = tuple(self.atoms)
        ids = [atom.atom_id for atom in atoms]
        if len(set(ids)) != len(ids):
            raise ValueError("manipulation atom ids must be unique")
        known: set[str] = set()
        for atom in atoms:
            missing = set(atom.depends_on) - known
            if missing:
                raise ValueError(f"atom {atom.atom_id!r} has forward or missing dependencies: {sorted(missing)}")
            known.add(atom.atom_id)
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "user_command": self.user_command,
            "scene_file": self.scene_file,
            "atoms": [atom.as_dict() for atom in self.atoms],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ManipulationPlan":
        return cls(
            plan_id=str(value["plan_id"]),
            scene_file=str(value["scene_file"]),
            atoms=tuple(ManipulationAtom.from_dict(item) for item in value.get("atoms") or ()),
            user_command=str(value.get("user_command") or ""),
            schema_version=str(value.get("schema_version") or ""),
            metadata=dict(value.get("metadata") or {}),
        )


def _primitive(value: str) -> ManipulationPrimitive:
    # place_on_slots was also used for beside/on-top/free-space placement, so
    # it is a backend rule name, not a public task primitive.
    return {
        "insert_vertical": ManipulationPrimitive.INSERT,
        "roof_assembly": ManipulationPrimitive.SNAP_PLACE,
        "snap_place": ManipulationPrimitive.SNAP_PLACE,
    }.get(str(value), ManipulationPrimitive.PICK_PLACE)


def _relation(operator: str) -> str:
    return {
        "on": "on_top",
        "desk_slot": "slot",
        "collection_slot": "slot",
        "collection_right_half": "target_pose",
        "pose_goal": "target_pose",
        "buffer": "target_pose",
        "exchange_to_a_pose": "target_pose",
        "exchange_to_b_pose": "target_pose",
    }.get(str(operator), str(operator) or "target_pose")


def _success_for(primitive: ManipulationPrimitive, relation: str) -> SuccessCriteria:
    if primitive is ManipulationPrimitive.INSERT:
        return SuccessCriteria(relation, 0.008, 10.0, 30)
    if primitive is ManipulationPrimitive.SNAP_PLACE:
        return SuccessCriteria(relation, 0.005, 8.0, 40)
    if relation == "inside":
        return SuccessCriteria(relation, 0.04, None, 40)
    return SuccessCriteria(relation, 0.02, 20.0, 20)


def compile_resolved_steps(
    *,
    plan_id: str,
    scene_file: str | Path,
    steps: Iterable[Mapping[str, Any]],
    user_command: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> ManipulationPlan:
    """Compile LLM-resolved step dictionaries into the stable task protocol."""

    atoms: list[ManipulationAtom] = []
    previous_id: str | None = None
    for index, raw in enumerate(steps, start=1):
        atom_id = str(raw.get("atom_id") or f"atom_{index:02d}")
        primitive = _primitive(str(raw.get("primitive", "")))
        relation = _relation(str(raw.get("operator", "")))
        atom = ManipulationAtom(
            atom_id=atom_id,
            primitive=primitive,
            object_id=str(raw["source_id"]),
            object_asset=str(raw.get("source_spec") or raw["source_id"]),
            support_object_id=None if raw.get("target_object_id") is None else str(raw["target_object_id"]),
            target_pose=raw["target_pose"],
            placement_mode=PlacementMode(str(raw.get("place_mode", PlacementMode.SURFACE.value))),
            semantic_operator=str(raw.get("operator") or "pose_goal"),
            success=_success_for(primitive, relation),
            depends_on=() if previous_id is None else (previous_id,),
            metadata={
                "description": str(raw.get("description") or ""),
                "warnings": list(raw.get("warnings") or []),
                "legacy_rule_primitive": str(raw.get("primitive") or ""),
                "semantic_step": dict(raw.get("semantic_step") or {}),
            },
        )
        atoms.append(atom)
        previous_id = atom_id
    return ManipulationPlan(
        plan_id=plan_id,
        scene_file=str(Path(scene_file).expanduser().resolve()),
        atoms=tuple(atoms),
        user_command=user_command,
        metadata=dict(metadata or {}),
    )


def load_plan(path: str | Path) -> ManipulationPlan:
    import json

    plan_path = Path(path).expanduser().resolve()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manipulation plan must contain one JSON object")
    return ManipulationPlan.from_dict(payload)
