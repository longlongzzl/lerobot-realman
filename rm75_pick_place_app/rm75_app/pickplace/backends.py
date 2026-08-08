from __future__ import annotations

from dataclasses import dataclass

from rm75_app.core.contracts import CommandSpec


@dataclass(frozen=True)
class PickPlaceBackend:
    mode: str
    module: str
    description: str


PICKPLACE_BACKENDS: dict[str, PickPlaceBackend] = {
    "direct": PickPlaceBackend(
        mode="direct",
        module="rm75_app.runtime.direct_pre_place",
        description="FoundationPose/direct pick-place",
    ),
    "sam6d": PickPlaceBackend(
        mode="sam6d",
        module="rm75_app.runtime.sam6d_pick_place",
        description="SAM3/SAM6D scene-aware pick-place",
    ),
    "wrist": PickPlaceBackend(
        mode="wrist",
        module="rm75_app.runtime.wrist_refined_pick_place",
        description="Direct pick-place with held-object wrist refinement",
    ),
    "tabletop-refine": PickPlaceBackend(
        mode="tabletop-refine",
        module="rm75_app.runtime.tabletop_pose_refine",
        description="SAM6D tabletop x/y/yaw refinement",
    ),
}

PICKPLACE_MODE_ALIASES = {
    "pick": "direct",
    "pick-place": "direct",
    "pick-place-direct": "direct",
    "tabletop": "tabletop-refine",
}


def normalize_mode(mode: str | None) -> str:
    requested = str(mode or "direct").strip().lower().replace("_", "-")
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

