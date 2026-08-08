from __future__ import annotations

from pathlib import Path

from rm75_app.paths import APP_ROOT


REPO_ROOT = APP_ROOT.parent
JIMU_ROOT = REPO_ROOT / "Beta_demo-codex-v0.9"
LEGO_ROOT = REPO_ROOT / "rm75_lego_snap_place_app"


def jimu_script(mode: str) -> Path:
    scripts = {
        "four-wall": JIMU_ROOT / "rm75_jimu_four_wall_portable.py",
        "triangle-roof": JIMU_ROOT / "rm75_jimu_triangle_roof_apriltag_portable.py",
    }
    try:
        return scripts[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported Jimu compatibility mode: {mode}") from exc


def lego_command_root() -> Path:
    return LEGO_ROOT

