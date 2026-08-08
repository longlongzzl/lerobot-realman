"""Canonical pick-place application boundary.

This package owns pick-place mode selection and defaults.  The heavy runtime
implementations stay below it in ``runtime/``, while perception, placement,
planning, and execution remain replaceable capability layers.
"""

from .backends import PICKPLACE_BACKENDS, PickPlaceBackend, command_for_mode, get_backend
from .config import DEFAULT_PICKPLACE, build_default_args
from .layers import PICKPLACE_LAYERS
from .runner import run_compiled, run_mode

__all__ = [
    "DEFAULT_PICKPLACE",
    "PICKPLACE_BACKENDS",
    "PICKPLACE_LAYERS",
    "PickPlaceBackend",
    "build_default_args",
    "command_for_mode",
    "get_backend",
    "run_compiled",
    "run_mode",
]
