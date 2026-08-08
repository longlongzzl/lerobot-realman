from __future__ import annotations

from typing import Any, Mapping, Protocol

from .contracts import (
    CommandSpec,
    PipelineContext,
    StageResult,
    TaskPlan,
    TaskRequest,
)


class TaskAdapter(Protocol):
    """Task-specific policy: schema, plan shape, and compatibility command."""

    definition: Any

    def normalize_request(self, request: TaskRequest) -> TaskRequest: ...

    def validate_request(self, request: TaskRequest) -> tuple[str, ...]: ...

    def build_plan(self, request: TaskRequest) -> TaskPlan: ...

    def command(self, request: TaskRequest, *, python: str = "python") -> CommandSpec: ...


class PerceptionProvider(Protocol):
    """Scene capture and object pose estimation."""

    def perceive(self, context: PipelineContext) -> Mapping[str, Any]: ...


class CandidateGenerator(Protocol):
    """Generate grasp/place candidates from task state and perception."""

    def generate(self, context: PipelineContext) -> Mapping[str, Any]: ...


class MotionPlanner(Protocol):
    """Screen candidate pairs and produce collision-free motion."""

    def plan(self, context: PipelineContext) -> Mapping[str, Any]: ...


class MotionExecutor(Protocol):
    """Execute a plan in simulation or on the real RM75."""

    def execute(self, context: PipelineContext) -> StageResult: ...


class ResultValidator(Protocol):
    """Validate object placement, assembly state, and safety diagnostics."""

    def validate(self, context: PipelineContext) -> StageResult: ...

