from __future__ import annotations

import sys

from rm75_app.launch import run_app_module


def main() -> int:
    return run_app_module("rm75_app.web.control_panel", sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
