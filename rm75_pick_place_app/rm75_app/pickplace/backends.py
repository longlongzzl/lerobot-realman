from __future__ import annotations

from dataclasses import dataclass

from rm75_app.core.contracts import CommandSpec


@dataclass(frozen=True)
class PickPlaceBackend:
    mode: str
    module: str
    description: str


PICKPLACE_BACKENDS: dict[str, PickPlaceBackend] = {
    "curobo2": PickPlaceBackend(
        mode="curobo2",
        module="rm75_app.runtime.curobo2_pick_place",
        description="Layered Curobo2 batch pick-place with explicit attachment boundaries",
    ),
    "rrtrack": PickPlaceBackend(
        mode="rrtrack",
        module="rm75_app.runtime.rrtrack_pose_tracking",
        description="Recoverable CUTIE + 6D closed-loop perception for pick-place",
    ),
    "openworld-geometry": PickPlaceBackend(
        mode="openworld-geometry",
        module="rm75_app.runtime.openworld_geometry",
        description="RaySt3R-style generated prior plus dynamic RGB-D collision geometry",
    ),
    "tabletop-refine": PickPlaceBackend(
        mode="tabletop-refine",
        module="rm75_app.runtime.tabletop_pose_refine",
        description="SAM6D tabletop x/y/yaw refinement",
    ),
}

PICKPLACE_MODE_ALIASES = {
    "pick": "curobo2",
    "pick-place": "curobo2",
    "pick-place-direct": "curobo2",
    "direct": "curobo2",
    "tabletop": "tabletop-refine",
    "openworld": "openworld-geometry",
    "unknown-object": "openworld-geometry",
    "v2": "curobo2",
}


def normalize_mode(mode: str | None) -> str:
    requested = str(mode or "curobo2").strip().lower().replace("_", "-")
    return PICKPLACE_MODE_ALIASES.get(requested, requested)


def get_backend(mode: str | None) -> PickPlaceBackend:
    normalized = normalize_mode(mode)
    try:
        return PICKPLACE_BACKENDS[normalized]
    except KeyError as exc:
        valid = ", ".join(PICKPLACE_BACKENDS)
        raise ValueError(f"unknown pick-place mode {mode!r}; valid modes: {valid}") from exc


def command_for_mode(mode: str | None, args: list[str] | tuple[str, ...] = (), *, python: str = "python") -> CommandSpec:
    backend = get_backend(mode)
    return CommandSpec(
        argv=(python, "-m", backend.module, *tuple(str(arg) for arg in args)),
        description=backend.description,
    )
