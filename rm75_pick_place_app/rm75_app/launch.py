from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Iterable

from .paths import APP_ROOT, PACKAGE_DIR, ensure_runtime_dirs


@contextmanager
def local_runtime(argv: Iterable[str] | None = None):
    """Run RM75 modules with paths scoped to this refactor folder."""
    ensure_runtime_dirs()
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    old_path = sys.path[:]
    old_mplconfig = os.environ.get("MPLCONFIGDIR")
    try:
        os.chdir(APP_ROOT)
        mpl_dir = APP_ROOT / ".cache" / "matplotlib"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
        sys.path.insert(0, str(PACKAGE_DIR))
        sys.path.insert(0, str(APP_ROOT))
        if argv is not None:
            sys.argv = list(argv)
        yield
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
        if old_mplconfig is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = old_mplconfig
        os.chdir(old_cwd)


def run_app_module(module_name: str, args: list[str] | None = None) -> int:
    argv = [f"-m {module_name}", *(args or [])]
    with local_runtime(argv):
        module = import_module(module_name)
        main = getattr(module, "main", None)
        if not callable(main):
            raise AttributeError(f"{module_name} has no callable main()")
        result = main()
        return int(result or 0)
    return 0


def import_app_module(module_name: str):
    with local_runtime(None):
        return import_module(module_name)
