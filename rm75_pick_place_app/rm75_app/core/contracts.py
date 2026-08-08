from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class PipelineStage(str, Enum):
    """The stable lifecycle shared by pick-place and assembly tasks."""

    TASK_INPUT = "task_input"
    SCENE_CAPTURE = "scene_capture"
    PERCEPTION = "perception"
    CANDIDATE_GENERATION = "candidate_generation"
    PAIR_SELECTION = "pair_selection"
    MOTION_PLANNING = "motion_planning"
    EXECUTION = "execution"
    VALIDATION = "validation"


@dataclass(frozen=True)
class TaskDefinition:
    """Metadata and capabilities of a task family.

    This object intentionally contains no NumPy, simulator, camera, or robot
    SDK types.  It is safe to use from a CLI, web service, test, or planner.
    """

    key: str
    title: str
    family: str
    description: str
    capabilities: tuple[str, ...] = ()
    modes: tuple[str, ...] = ()
    default_mode: str = "default"
    aliases: tuple[str, ...] = ()
    stages: tuple[PipelineStage, ...] = ()
    status: str = "active"
    backend: str | None = None
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "family": self.family,
            "description": self.description,
            "capabilities": list(self.capabilities),
            "modes": list(self.modes),
            "default_mode": self.default_mode,
            "aliases": list(self.aliases),
            "stages": [stage.value for stage in self.stages],
            "status": self.status,
            "backend": self.backend,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class TaskRequest:
    """Normalized input passed from a task adapter into the pipeline."""

    task: str
    mode: str | None = None
    source: Path | None = None
    args: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandSpec:
    """A launchable command without coupling callers to shell syntax."""

    argv: tuple[str, ...]
    cwd: Path | None = None
    description: str = ""
    notes: tuple[str, ...] = ()

    def as_list(self) -> list[str]:
        return list(self.argv)

    def display(self) -> str:
        command = shlex.join([str(part) for part in self.argv])
        if self.cwd is not None:
            return f"(cd {shlex.quote(str(self.cwd))} && {command})"
        return command


@dataclass(frozen=True)
class TaskStep:
    key: str
    stage: PipelineStage
    description: str
    optional: bool = False


@dataclass(frozen=True)
class TaskPlan:
    definition: TaskDefinition
    request: TaskRequest
    steps: tuple[TaskStep, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.definition.key,
            "mode": self.request.mode or self.definition.default_mode,
            "steps": [
                {
                    "key": step.key,
                    "stage": step.stage.value,
                    "description": step.description,
                    "optional": step.optional,
                }
                for step in self.steps
            ],
            "metadata": dict(self.metadata),
        }


@dataclass
class PipelineContext:
    """Mutable runtime state shared by stage implementations."""

    request: TaskRequest
    payload: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageResult:
    stage: PipelineStage
    ok: bool
    status: str
    detail: str = ""
    artifacts: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    task: str
    mode: str
    ok: bool
    stages: tuple[StageResult, ...] = ()
    summary: str = ""

