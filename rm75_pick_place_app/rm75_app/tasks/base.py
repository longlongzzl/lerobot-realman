from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Mapping

from rm75_app.core.contracts import (
    CommandSpec,
    PipelineStage,
    TaskDefinition,
    TaskPlan,
    TaskRequest,
    TaskStep,
)


def normalize_token(value: str) -> str:
    return "-".join(str(value).strip().lower().replace("_", "-").split())


_STAGE_DESCRIPTIONS: Mapping[PipelineStage, str] = {
    PipelineStage.TASK_INPUT: "解析任务定义和约束",
    PipelineStage.SCENE_CAPTURE: "采集并冻结任务场景",
    PipelineStage.PERCEPTION: "估计目标、障碍物和坐标变换",
    PipelineStage.CANDIDATE_GENERATION: "生成抓取/连接和放置候选",
    PipelineStage.PAIR_SELECTION: "做候选配对与可行性筛选",
    PipelineStage.MOTION_PLANNING: "生成并验证运动轨迹",
    PipelineStage.EXECUTION: "执行轨迹和末端动作",
    PipelineStage.VALIDATION: "验证最终放置或装配状态",
}


class TaskAdapterBase(ABC):
    """Small task-specific layer above the shared runtime capabilities."""

    definition: ClassVar[TaskDefinition]
    mode_aliases: ClassVar[Mapping[str, str]] = {}

    def normalize_mode(self, mode: str | None) -> str:
        requested = normalize_token(mode or self.definition.default_mode)
        return normalize_token(self.mode_aliases.get(requested, requested))

    def normalize_request(self, request: TaskRequest) -> TaskRequest:
        return TaskRequest(
            task=self.definition.key,
            mode=self.normalize_mode(request.mode),
            source=request.source,
            args=tuple(request.args),
            options=dict(request.options),
        )

    def validate_request(self, request: TaskRequest) -> tuple[str, ...]:
        normalized = self.normalize_request(request)
        if normalized.mode not in self.definition.modes:
            valid = ", ".join(self.definition.modes)
            return (f"task {self.definition.key!r} does not support mode {normalized.mode!r}; valid modes: {valid}",)
        return ()

    def build_plan(self, request: TaskRequest) -> TaskPlan:
        normalized = self.normalize_request(request)
        errors = self.validate_request(normalized)
        if errors:
            raise ValueError("; ".join(errors))
        steps = tuple(
            TaskStep(
                key=stage.value,
                stage=stage,
                description=_STAGE_DESCRIPTIONS[stage],
            )
            for stage in self.definition.stages
        )
        return TaskPlan(
            definition=self.definition,
            request=normalized,
            steps=steps,
            metadata={"backend": self.definition.backend, "status": self.definition.status},
        )

    @abstractmethod
    def command(self, request: TaskRequest, *, python: str = "python") -> CommandSpec:
        raise NotImplementedError
