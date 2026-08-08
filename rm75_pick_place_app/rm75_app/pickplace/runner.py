from __future__ import annotations

from rm75_app.core.contracts import CommandSpec
from rm75_app.launch import run_app_module


def run_compiled(command: CommandSpec) -> int:
    """Run a compiled in-app pick-place command through the local runtime."""

    argv = command.as_list()
    if len(argv) < 3 or argv[1] != "-m":
        raise ValueError(f"pick-place backend must be an in-app module command: {argv!r}")
    return run_app_module(argv[2], argv[3:])


def run_mode(mode: str, args: list[str] | tuple[str, ...] = ()) -> int:
    from .backends import command_for_mode

    return run_compiled(command_for_mode(mode, args))

