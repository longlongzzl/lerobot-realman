from __future__ import annotations

from rm75_app.core.contracts import CommandSpec, PipelineStage, TaskDefinition, TaskRequest
from rm75_app.legacy.backends import JIMU_ROOT, jimu_script
from rm75_app.tasks.base import TaskAdapterBase


class JimuTask(TaskAdapterBase):
    definition = TaskDefinition(
        key="jimu",
        title="Jimu 搭积木",
        family="assembly",
        description="磁吸积木的取件、姿态对齐、连接/插接和装配状态验证。",
        capabilities=("assembly", "multi_step", "magnetic_snap", "apriltag", "sam6d", "curobo"),
        modes=("four-wall", "triangle-roof"),
        default_mode="four-wall",
        aliases=("jimu-builder", "builder", "stacking", "stack"),
        stages=(
            PipelineStage.TASK_INPUT,
            PipelineStage.SCENE_CAPTURE,
            PipelineStage.PERCEPTION,
            PipelineStage.CANDIDATE_GENERATION,
            PipelineStage.PAIR_SELECTION,
            PipelineStage.MOTION_PLANNING,
            PipelineStage.EXECUTION,
            PipelineStage.VALIDATION,
        ),
        status="compatibility",
        backend="Beta_demo-codex-v0.9",
        notes=(
            "当前适配器只负责统一任务契约和旧入口命令；Jimu 运动实现仍在 legacy backend。",
            "迁移后保留同一任务契约，不再由 Jimu 任务直接导入 pick-place 巨型脚本。",
        ),
    )
    mode_aliases = {
        "four-wall-portable": "four-wall",
        "four-walls": "four-wall",
        "triangle": "triangle-roof",
        "roof": "triangle-roof",
    }

    def command(self, request: TaskRequest, *, python: str = "python") -> CommandSpec:
        normalized = self.normalize_request(request)
        errors = self.validate_request(normalized)
        if errors:
            raise ValueError("; ".join(errors))
        script = jimu_script(normalized.mode or self.definition.default_mode)
        notes = list(self.definition.notes)
        if not JIMU_ROOT.exists():
            notes.append(f"legacy backend not found: {JIMU_ROOT}")
        return CommandSpec(
            argv=(python, str(script), *normalized.args),
            cwd=JIMU_ROOT,
            description=f"{self.definition.title} / {normalized.mode}",
            notes=tuple(notes),
        )

