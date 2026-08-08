from __future__ import annotations

from ..pickplace.config import DEFAULT_PICKPLACE, build_default_args


# Compatibility names for old command builders.  New pick-place code should
# import defaults from rm75_app.pickplace.config.
DEFAULT_OBJECTS = list(DEFAULT_PICKPLACE.cycle_object_names)
DEFAULT_TRACKED_OBJECTS = list(DEFAULT_PICKPLACE.tracked_scene_object_names)


def base_pick_args(*, render_mode: str = "human", execute_real: bool = False) -> list[str]:
    return build_default_args(render_mode=render_mode, execute_real=execute_real)
