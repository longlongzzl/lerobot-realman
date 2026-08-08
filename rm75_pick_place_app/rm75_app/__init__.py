"""Refactored RM75 pick-place application package.

This package owns the production entry points and module boundaries.  Code,
assets, configs, test scenes, and runtime logs are local to the refactor
folder.
"""

from .paths import APP_ROOT, ASSET_DIR, RUNTIME_DIR

__all__ = ["APP_ROOT", "ASSET_DIR", "RUNTIME_DIR"]
