from __future__ import annotations

import json

from rm75_app.core.contracts import CommandSpec, PipelineStage, TaskDefinition, TaskRequest
from rm75_app.legacy.backends import LEGO_ROOT
from rm75_app.tasks.base import TaskAdapterBase, normalize_token


class LegoTask(TaskAdapterBase):
    definition = TaskDefinition(
        key="lego",
        title="Lego Snap-Place",
        family="assembly",
        description="固定连接头驱动 Lego 连接、放置、按压/扭转脱离和结果验证。",
        capabilities=("assembly", "grid_task", "snap", "fixed_tool", "curobo"),
        modes=("real", "dry-run"),
        default_mode="real",
        aliases=("lego-snap", "lego-real", "snap-place", "snap"),
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
        backend="rm75_lego_snap_place_app",
        notes=(
            "任务 JSON 语义已在主线契约中固定；当前真机执行器仍保留在 Lego 兼容目录。",
            "后续只迁移 Lego simulation/runtime，复用本目录 perception/planning/execution 层。",
        ),
    )
    mode_aliases = {
        "lego-real": "real",
        "simulation": "dry-run",
        "dryrun": "dry-run",
    }

    def normalize_request(self, request: TaskRequest) -> TaskRequest:
        mode = request.mode
        if mode is None and normalize_token(request.task) == "lego-real":
            mode = "real"
        return super().normalize_request(
            TaskRequest(
                task=self.definition.key,
                mode=mode,
                source=request.source,
                args=request.args,
                options=request.options,
            )
        )

    def validate_request(self, request: TaskRequest) -> tuple[str, ...]:
        errors = list(super().validate_request(request))
        if request.source is not None:
            source = request.source.expanduser()
            try:
                data = json.loads(source.read_text(encoding="utf-8"))
            except OSError as exc:
                errors.append(f"cannot read Lego task JSON {source}: {exc}")
            except json.JSONDecodeError as exc:
                errors.append(f"invalid Lego task JSON {source}: {exc}")
            else:
                steps = data if isinstance(data, list) else data.get("steps") if isinstance(data, dict) else None
                if not isinstance(steps, list) or not steps:
                    errors.append("Lego task JSON must be a non-empty list or an object with a non-empty 'steps' list")
                else:
                    for index, step in enumerate(steps):
                        if not isinstance(step, dict):
                            errors.append(f"Lego task step {index} must be an object")
                            continue
                        missing = [name for name in ("action", "brick", "grid_x", "grid_y") if name not in step]
                        if missing:
                            errors.append(f"Lego task step {index} missing: {', '.join(missing)}")
                        if step.get("action") not in {"assemble", "disassemble"}:
                            errors.append(f"Lego task step {index} has invalid action: {step.get('action')!r}")
                        try:
                            if int(step.get("grid_x")) < 0 or int(step.get("grid_y")) < 0:
                                errors.append(f"Lego task step {index} grid coordinates must be non-negative")
                            if int(step.get("layer", 1)) < 1:
                                errors.append(f"Lego task step {index} layer must be >= 1")
                        except (TypeError, ValueError):
                            errors.append(f"Lego task step {index} grid_x/grid_y/layer must be integers")
        return tuple(errors)

    def command(self, request: TaskRequest, *, python: str = "python") -> CommandSpec:
        normalized = self.normalize_request(request)
        errors = self.validate_request(normalized)
        if errors:
            raise ValueError("; ".join(errors))
        args = list(normalized.args)
        if normalized.source is not None and "--task-json" not in args:
            args = ["--task-json", str(normalized.source.expanduser().resolve()), *args]
        if normalized.mode == "dry-run" and "--dry-run" not in args:
            args.append("--dry-run")
        notes = list(self.definition.notes)
        if not LEGO_ROOT.exists():
            notes.append(f"legacy backend not found: {LEGO_ROOT}")
        return CommandSpec(
            argv=(python, "-m", "rm75_app", "lego-real", *args),
            cwd=LEGO_ROOT,
            description=f"{self.definition.title} / {normalized.mode}",
            notes=tuple(notes),
        )
