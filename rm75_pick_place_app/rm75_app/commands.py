from __future__ import annotations

import shlex

from .config import base_pick_args
from .pickplace.backends import command_for_mode, get_backend


DIRECT_MODULE = get_backend("direct").module
WRIST_REFINED_MODULE = get_backend("wrist").module
ROOF_ASSEMBLY_MODULE = "rm75_app.runtime.roof_assembly_pick_place"
SAM6D_MODULE = get_backend("sam6d").module
WEB_MODULE = "rm75_app.web.control_panel"
LLM_MODULE = "rm75_app.llm.orchestrator"
CALIB_BASE_MODULE = "rm75_app.calibration.base_camera_visual_calibration"
CALIB_BOARD_ANCHOR_MODULE = "rm75_app.calibration.global_board_anchor"
CALIB_WRIST_MODULE = "rm75_app.calibration.wrist_camera_board_calibration"
CALIB_WRIST_ANCHOR_MODULE = "rm75_app.calibration.wrist_camera_board_anchor_calibration"
CALIB_DUAL_BOARD_MODULE = "rm75_app.calibration.dual_camera_moving_board_calibration"
CALIB_WRIST_VISUAL_MODULE = "rm75_app.calibration.wrist_camera_visual_refine"
CALIB_CHECK_MODULE = "rm75_app.calibration.combined_calibration_check"
CALIB_BASE_POINT_CHECK_MODULE = "rm75_app.calibration.base_camera_point_accuracy_check"
CALIB_JOINT_BOARD_MODULE = "rm75_app.calibration.joint_board_optimization"


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def direct_pick_command(*, python: str = "python", render_mode: str = "human", execute_real: bool = False) -> list[str]:
    args = base_pick_args(render_mode=render_mode, execute_real=execute_real)
    return command_for_mode("direct", args, python=python).as_list()


def sam6d_pick_command(*, python: str = "python", render_mode: str = "human", execute_real: bool = False) -> list[str]:
    args = base_pick_args(render_mode=render_mode, execute_real=execute_real)
    return command_for_mode("sam6d", args, python=python).as_list()


def wrist_refined_pick_command(*, python: str = "python", render_mode: str = "human", execute_real: bool = False) -> list[str]:
    args = base_pick_args(render_mode=render_mode, execute_real=execute_real)
    return command_for_mode("wrist", args, python=python).as_list()


def _remove_nargs_star_option(args: list[str], option: str) -> list[str]:
    cleaned: list[str] = []
    idx = 0
    while idx < len(args):
        if str(args[idx]) == option:
            idx += 1
            while idx < len(args) and not str(args[idx]).startswith("--"):
                idx += 1
            continue
        cleaned.append(args[idx])
        idx += 1
    return cleaned


def roof_assembly_pick_command(*, python: str = "python", render_mode: str = "human", execute_real: bool = False) -> list[str]:
    args = base_pick_args(render_mode=render_mode, execute_real=execute_real)
    args = _remove_nargs_star_option(args, "--cycle-object-names")
    args = _remove_nargs_star_option(args, "--tracked-scene-object-names")
    return [python, "-m", ROOF_ASSEMBLY_MODULE, *args]


def web_command(*, python: str = "python", host: str = "127.0.0.1", port: int = 7860) -> list[str]:
    return [python, "-m", WEB_MODULE, "--host", str(host), "--port", str(int(port))]


def calibration_command(kind: str, *, python: str = "python") -> list[str]:
    modules = {
        "base": CALIB_BASE_MODULE,
        "board-anchor": CALIB_BOARD_ANCHOR_MODULE,
        "wrist": CALIB_WRIST_MODULE,
        "wrist-anchor": CALIB_WRIST_ANCHOR_MODULE,
        "dual-board": CALIB_DUAL_BOARD_MODULE,
        "wrist-visual": CALIB_WRIST_VISUAL_MODULE,
        "check": CALIB_CHECK_MODULE,
        "base-point-check": CALIB_BASE_POINT_CHECK_MODULE,
        "joint-board": CALIB_JOINT_BOARD_MODULE,
    }
    if kind not in modules:
        raise ValueError(f"unknown calibration command kind: {kind}")
    return [python, "-m", modules[kind]]
