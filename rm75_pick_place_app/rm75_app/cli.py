from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .commands import (
    calibration_command,
    direct_pick_command,
    shell_join,
    web_command,
)
from .launch import run_app_module


def _split_passthrough(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def _print_task_list() -> int:
    from .tasks import list_task_adapters

    for adapter in list_task_adapters():
        definition = adapter.definition
        modes = ", ".join(definition.modes)
        print(f"{definition.key:10s} {definition.status:13s} {definition.title}  modes={modes}")
    return 0


def _print_task_info(task_name: str) -> int:
    from .core.contracts import TaskRequest
    from .pipeline import TaskPipeline
    from .tasks import get_task_adapter

    try:
        adapter = get_task_adapter(task_name)
        definition = adapter.definition
        compiled = TaskPipeline().compile(TaskRequest(task=task_name))
    except (KeyError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 2

    print(f"key: {definition.key}")
    print(f"title: {definition.title}")
    print(f"family: {definition.family}")
    print(f"status: {definition.status}")
    print(f"description: {definition.description}")
    print(f"modes: {', '.join(definition.modes)}")
    print(f"aliases: {', '.join(definition.aliases) or '(none)'}")
    print(f"capabilities: {', '.join(definition.capabilities) or '(none)'}")
    print(f"stages: {' -> '.join(stage.value for stage in definition.stages)}")
    print(f"backend: {definition.backend or '(none)'}")
    print(f"default command: {compiled.command.display()}")
    for note in definition.notes:
        print(f"note: {note}")
    return 0


def _run_task_command(ns: argparse.Namespace) -> int:
    from .core.contracts import TaskRequest
    from .pipeline import TaskPipeline

    request = TaskRequest(
        task=ns.task,
        mode=ns.mode,
        source=Path(ns.source).expanduser() if ns.source else None,
        args=tuple(_split_passthrough(ns.args)),
    )
    try:
        compiled = TaskPipeline().compile(request, python=ns.python)
    except (KeyError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 2

    print(compiled.command.display())
    print("plan:")
    for step in compiled.plan.steps:
        print(f"  {step.stage.value:24s} {step.description}")
    for note in compiled.command.notes:
        print(f"note: {note}")
    return 0


def _run_registered_pickplace(mode: str, args: list[str]) -> int:
    """Keep the public direct/SAM6D CLI names while resolving through tasks."""
    from .core.contracts import TaskRequest
    from .pipeline import TaskPipeline

    compiled = TaskPipeline().compile(TaskRequest(task="pickplace", mode=mode, args=tuple(args)))
    from .pickplace.runner import run_compiled

    return run_compiled(compiled.command)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RM75 refactored launcher")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_direct = sub.add_parser("direct", help="Compatibility alias for the layered Curobo2 pick-place entrypoint")
    p_direct.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the Curobo2 module")

    p_sam6d = sub.add_parser("sam6d", help="Run SAM3/SAM6D pose perception only")
    p_sam6d.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the SAM6D pose provider")

    p_rrtrack = sub.add_parser("rrtrack", help="Run recoverable CUTIE + 6D object tracking")
    p_rrtrack.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the RRTrack perception module")

    p_rrtrack_bank = sub.add_parser("rrtrack-build-bank", help="Build a DINOv2 offline recovery bank from SAM6D templates")
    p_rrtrack_bank.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the bank builder")

    p_rrtrack_all_banks = sub.add_parser("rrtrack-build-all-banks", help="Build DINOv2 recovery banks for all known objects")
    p_rrtrack_all_banks.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the all-object bank builder")

    p_openworld = sub.add_parser("openworld-geometry", aliases=["openworld"], help="Build/update unseen-object collision geometry")
    p_openworld.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the dynamic geometry module")

    p_curobo2 = sub.add_parser("curobo2-pickplace", aliases=["curobo2"], help="Run the layered Curobo2 pick-place pipeline")
    p_curobo2.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the Curobo2 pipeline")

    p_curobo2_sim = sub.add_parser("curobo2-sim-replay", help="Replay a portable Curobo2 trajectory package in ManiSkill")
    p_curobo2_sim.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the ManiSkill replay module")
    p_task_validation = sub.add_parser("task-validate", help="Run geometry, Curobo2 and ManiSkill task gates")
    p_task_validation.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the three-gate validator")
    p_maniskill_preview = sub.add_parser("maniskill-preview", help="Open a task scene and target poses in ManiSkill")
    p_maniskill_preview.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the interactive preview")

    p_tabletop_refine = sub.add_parser(
        "tabletop-refine",
        help="Refine a SAM6D full-scene result for tabletop objects by optimizing x/y/yaw only",
    )
    p_tabletop_refine.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the tabletop refinement module")

    p_wrist_live = sub.add_parser("wrist-live-overlay", help="Live wrist RGB / simulated wrist view / overlay alignment viewer")
    p_wrist_live.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the wrist live overlay module")

    p_web = sub.add_parser("web", help="Run the web control panel")
    p_web.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the web module")

    p_llm = sub.add_parser("llm", help="Run the LLM pick-place orchestrator")
    p_llm.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the LLM module")

    p_pickplace = sub.add_parser("pickplace", help="Run the canonical pick-place mainline")
    p_pickplace.add_argument(
        "--mode",
        choices=["curobo2", "rrtrack", "openworld-geometry", "tabletop-refine"],
        default="curobo2",
        help="Pick-place backend selected by the mainline runner",
    )
    p_pickplace.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the selected pick-place backend")

    p_tasks = sub.add_parser("tasks", aliases=["task"], help="Inspect task adapters and compile a task command")
    p_tasks.add_argument("action", choices=["list", "info", "command"])
    p_tasks.add_argument("task", nargs="?", help="Canonical task name or alias")
    p_tasks.add_argument("--mode", help="Task-specific mode, for example four-wall or sam6d")
    p_tasks.add_argument("--source", help="Optional task definition file, currently used by Lego")
    p_tasks.add_argument("--python", default="python", help="Python executable used in the displayed command")

    p_calib_base = sub.add_parser("calib-base", help="Run fixed/base camera visual calibration")
    p_calib_base.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the calibration module")

    p_calib_board_anchor = sub.add_parser(
        "calib-board-anchor",
        help="Anchor the fixed ChArUco board in robot-base coordinates with the global camera",
    )
    p_calib_board_anchor.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the calibration module")

    p_calib_wrist = sub.add_parser("calib-wrist", help="Run wrist-camera ChArUco hand-eye calibration")
    p_calib_wrist.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the calibration module")

    p_calib_wrist_anchor = sub.add_parser(
        "calib-wrist-anchor",
        help="Run wrist-camera fixed-board-anchor reprojection calibration",
    )
    p_calib_wrist_anchor.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the calibration module")

    p_calib_dual_board = sub.add_parser(
        "calib-dual-board",
        help="Run paired global+wrist moving-board wrist-camera calibration",
    )
    p_calib_dual_board.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the calibration module")

    p_calib_wrist_visual = sub.add_parser(
        "calib-wrist-visual-refine",
        aliases=["calib-wrist-visual"],
        help="Refine wrist-camera calibration using multi-view robot body silhouettes",
    )
    p_calib_wrist_visual.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the calibration module")

    p_calib_check = sub.add_parser("calib-check", help="Run combined base/wrist camera calibration check")
    p_calib_check.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the calibration module")

    p_calib_base_point = sub.add_parser(
        "calib-base-point-check",
        aliases=["calib-global-point-check", "calib-base-accuracy"],
        help="Validate global/base camera calibration by pointing the robot at a detected board point",
    )
    p_calib_base_point.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the calibration module")

    p_calib_joint = sub.add_parser("calib-joint-board", help="Run two-camera fixed-board joint optimization")
    p_calib_joint.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the calibration module")

    p_cmd = sub.add_parser("print-command", help="Print common commands without running them")
    p_cmd.add_argument("kind", choices=["direct", "tabletop-refine", "web", "calib-base", "calib-board-anchor", "calib-wrist", "calib-wrist-anchor", "calib-dual-board", "calib-wrist-visual", "calib-check", "calib-base-point-check", "calib-joint-board"])
    p_cmd.add_argument("--execute-real", action="store_true")
    p_cmd.add_argument("--render-mode", default="human")
    p_cmd.add_argument("--host", default="127.0.0.1")
    p_cmd.add_argument("--port", type=int, default=7860)

    sub.add_parser("verify", help="Verify the refactored folder is locally runnable")

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    task_passthrough: list[str] = []
    parser_argv = raw_argv
    if raw_argv and raw_argv[0] in {"tasks", "task"} and "--" in raw_argv:
        separator = raw_argv.index("--")
        parser_argv = raw_argv[:separator]
        task_passthrough = raw_argv[separator + 1 :]

    ns = parser.parse_args(parser_argv)
    if ns.cmd in {"tasks", "task"}:
        ns.args = task_passthrough
        if ns.action == "list":
            return _print_task_list()
        if not ns.task:
            print(f"[ERROR] tasks {ns.action} requires a task name")
            return 2
        if ns.action == "info":
            return _print_task_info(ns.task)
        return _run_task_command(ns)
    if ns.cmd == "pickplace":
        return _run_registered_pickplace(ns.mode, _split_passthrough(ns.args))
    if ns.cmd == "direct":
        return _run_registered_pickplace("curobo2", _split_passthrough(ns.args))
    if ns.cmd == "sam6d":
        return run_app_module("rm75_app.perception.sam6d_pose_provider", _split_passthrough(ns.args))
    if ns.cmd == "rrtrack":
        return _run_registered_pickplace("rrtrack", _split_passthrough(ns.args))
    if ns.cmd == "rrtrack-build-bank":
        return run_app_module("rm75_app.runtime.rrtrack_build_bank", _split_passthrough(ns.args))
    if ns.cmd == "rrtrack-build-all-banks":
        return run_app_module("rm75_app.runtime.rrtrack_build_all_banks", _split_passthrough(ns.args))
    if ns.cmd in {"openworld-geometry", "openworld"}:
        return _run_registered_pickplace("openworld-geometry", _split_passthrough(ns.args))
    if ns.cmd in {"curobo2-pickplace", "curobo2"}:
        return run_app_module("rm75_app.runtime.curobo2_pick_place", _split_passthrough(ns.args))
    if ns.cmd == "curobo2-sim-replay":
        return run_app_module("rm75_app.runtime.curobo2_sim_replay", _split_passthrough(ns.args))
    if ns.cmd == "task-validate":
        return run_app_module("rm75_app.runtime.task_validation", _split_passthrough(ns.args))
    if ns.cmd == "maniskill-preview":
        return run_app_module("rm75_app.runtime.maniskill_scene_preview", _split_passthrough(ns.args))
    if ns.cmd == "tabletop-refine":
        return _run_registered_pickplace("tabletop-refine", _split_passthrough(ns.args))
    if ns.cmd == "wrist-live-overlay":
        return run_app_module("rm75_app.runtime.wrist_live_overlay", _split_passthrough(ns.args))
    if ns.cmd == "web":
        return run_app_module("rm75_app.web.control_panel", _split_passthrough(ns.args))
    if ns.cmd == "llm":
        return run_app_module("rm75_app.llm.orchestrator", _split_passthrough(ns.args))
    if ns.cmd == "calib-base":
        return run_app_module("rm75_app.calibration.base_camera_visual_calibration", _split_passthrough(ns.args))
    if ns.cmd == "calib-board-anchor":
        return run_app_module("rm75_app.calibration.global_board_anchor", _split_passthrough(ns.args))
    if ns.cmd == "calib-wrist":
        return run_app_module("rm75_app.calibration.wrist_camera_board_calibration", _split_passthrough(ns.args))
    if ns.cmd == "calib-wrist-anchor":
        return run_app_module("rm75_app.calibration.wrist_camera_board_anchor_calibration", _split_passthrough(ns.args))
    if ns.cmd == "calib-dual-board":
        return run_app_module("rm75_app.calibration.dual_camera_moving_board_calibration", _split_passthrough(ns.args))
    if ns.cmd in {"calib-wrist-visual-refine", "calib-wrist-visual"}:
        return run_app_module("rm75_app.calibration.wrist_camera_visual_refine", _split_passthrough(ns.args))
    if ns.cmd == "calib-check":
        return run_app_module("rm75_app.calibration.combined_calibration_check", _split_passthrough(ns.args))
    if ns.cmd in {"calib-base-point-check", "calib-global-point-check", "calib-base-accuracy"}:
        return run_app_module("rm75_app.calibration.base_camera_point_accuracy_check", _split_passthrough(ns.args))
    if ns.cmd == "calib-joint-board":
        return run_app_module("rm75_app.calibration.joint_board_optimization", _split_passthrough(ns.args))
    if ns.cmd == "print-command":
        if ns.kind == "direct":
            print(shell_join(direct_pick_command(render_mode=ns.render_mode, execute_real=ns.execute_real)))
        elif ns.kind == "tabletop-refine":
            print(shell_join(["python", "-m", "rm75_app.runtime.tabletop_pose_refine"]))
        elif ns.kind == "calib-base":
            print(shell_join(calibration_command("base")))
        elif ns.kind == "calib-board-anchor":
            print(shell_join(calibration_command("board-anchor")))
        elif ns.kind == "calib-wrist":
            print(shell_join(calibration_command("wrist")))
        elif ns.kind == "calib-wrist-anchor":
            print(shell_join(calibration_command("wrist-anchor")))
        elif ns.kind == "calib-dual-board":
            print(shell_join(calibration_command("dual-board")))
        elif ns.kind == "calib-wrist-visual":
            print(shell_join(calibration_command("wrist-visual")))
        elif ns.kind == "calib-check":
            print(shell_join(calibration_command("check")))
        elif ns.kind == "calib-base-point-check":
            print(shell_join(calibration_command("base-point-check")))
        elif ns.kind == "calib-joint-board":
            print(shell_join(calibration_command("joint-board")))
        else:
            print(shell_join(web_command(host=ns.host, port=ns.port)))
        return 0
    if ns.cmd == "verify":
        from .verify import main as verify_main

        return verify_main()
    parser.error(f"unknown command: {ns.cmd}")
    return 2
