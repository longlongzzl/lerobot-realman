from __future__ import annotations

from dataclasses import dataclass

from rm75_app.paths import DEFAULT_CUROBO_CFG


@dataclass(frozen=True)
class PickPlaceDefaults:
    """Stable user-facing defaults shared by direct and SAM6D pick-place."""

    cycle_object_names: tuple[str, ...] = (
        "lvmukuai",
        "carriot",
        "shuazi",
        "hongshupian",
        "gluestick",
        "bi",
        "tennis",
    )
    tracked_scene_object_names: tuple[str, ...] = ("desk", "bitong")
    trajectory_preview_sleep: float = 0.08
    render_mode: str = "human"
    real_control_hz: int = 30
    real_max_delta_per_step: float = 0.1


DEFAULT_PICKPLACE = PickPlaceDefaults()


def build_default_args(*, render_mode: str = "human", execute_real: bool = False) -> list[str]:
    """Build the safe default CLI arguments for the main pick-place cycle."""

    defaults = DEFAULT_PICKPLACE
    args = [
        "--curobo-rm75-robot-cfg",
        str(DEFAULT_CUROBO_CFG),
        "--trajectory-preview-sleep",
        str(defaults.trajectory_preview_sleep),
        "--cycle-object-names",
        *defaults.cycle_object_names,
        "--tracked-scene-object-names",
        *defaults.tracked_scene_object_names,
        "--render-mode",
        str(render_mode),
        "--auto-execute",
        "--dry-run-motion-window-scale",
        "1.0",
        "--real-control-hz",
        str(defaults.real_control_hz),
        "--real-max-delta-per-step",
        str(defaults.real_max_delta_per_step),
    ]
    if execute_real:
        args.append("--execute-real")
    return args

