from __future__ import annotations

import sys

from rm75_app.launch import run_app_module


DEFAULT_ARGS = [
    "--wrist-relation-refine-after-grasp",
]


def main(argv: list[str] | None = None) -> int:
    user_args = list(sys.argv[1:] if argv is None else argv)
    return run_app_module("rm75_app.runtime.direct_pre_place", [*DEFAULT_ARGS, *user_args])


if __name__ == "__main__":
    raise SystemExit(main())
