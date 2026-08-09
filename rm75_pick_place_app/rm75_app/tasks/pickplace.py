from __future__ import annotations

from rm75_app.core.contracts import CommandSpec, PipelineStage, TaskDefinition, TaskRequest
from rm75_app.pickplace.backends import PICKPLACE_MODE_ALIASES, command_for_mode
from rm75_app.tasks.base import TaskAdapterBase, normalize_token


class PickPlaceTask(TaskAdapterBase):
    definition = TaskDefinition(
        key="pickplace",
        title="通用 Pick-Place",
        family="pick_place",
        description="以分层感知、动态几何和 Curobo2 批量规划为主流程的通用物体任务。",
        capabilities=("single_object", "multi_object_cycle", "sam3", "sam6d", "rrtrack", "cutie", "openworld_geometry", "dynamic_mesh", "curobo2", "batch_ik", "wrist_refine"),
        modes=("curobo2", "rrtrack", "openworld-geometry", "tabletop-refine"),
        default_mode="curobo2",
        aliases=("pick-place", "pick_place", "pick", "grasp-place", "direct", "rrtrack", "openworld", "unknown-object"),
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
        backend="rm75_app.pickplace",
    )
    mode_aliases = PICKPLACE_MODE_ALIASES

    def normalize_request(self, request: TaskRequest) -> TaskRequest:
        task_alias_modes = {"direct": "curobo2", "rrtrack": "rrtrack", "openworld": "openworld-geometry", "unknown-object": "openworld-geometry"}
        mode = request.mode
        if mode is None:
            mode = task_alias_modes.get(normalize_token(request.task))
        return super().normalize_request(
            TaskRequest(
                task=self.definition.key,
                mode=mode,
                source=request.source,
                args=request.args,
                options=request.options,
            )
        )

    def command(self, request: TaskRequest, *, python: str = "python") -> CommandSpec:
        normalized = self.normalize_request(request)
        errors = self.validate_request(normalized)
        if errors:
            raise ValueError("; ".join(errors))
        return command_for_mode(normalized.mode, normalized.args, python=python)
