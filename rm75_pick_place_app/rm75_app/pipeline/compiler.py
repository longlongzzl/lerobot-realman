from __future__ import annotations

from dataclasses import dataclass

from rm75_app.core.contracts import CommandSpec, TaskPlan, TaskRequest
from rm75_app.tasks.registry import get_task_adapter


@dataclass(frozen=True)
class CompiledTask:
    """A task request lowered to a stable plan and a compatibility command."""

    plan: TaskPlan
    command: CommandSpec


class TaskPipeline:
    """Resolve task policy before touching perception, planning, or hardware."""

    def compile(self, request: TaskRequest, *, python: str = "python") -> CompiledTask:
        adapter = get_task_adapter(request.task)
        normalized = adapter.normalize_request(request)
        errors = adapter.validate_request(normalized)
        if errors:
            raise ValueError("; ".join(errors))
        return CompiledTask(
            plan=adapter.build_plan(normalized),
            command=adapter.command(normalized, python=python),
        )

