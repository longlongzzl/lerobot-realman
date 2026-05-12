#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from datetime import datetime

from place_rules import list_place_rule_sources

DEFAULT_SCENE = Path(__file__).resolve().parent / "test_scenes" / "gluestick_desk_regression.json"
DEFAULT_ENTRYPOINT = Path(__file__).resolve().parent / "rm75_jiaobang_pick_place_targeted_curobo_direct_pre_place_sam6d.py"
DEFAULT_CHILD_PYTHON = Path(sys.executable).resolve()
RESULT_PREFIX = "TEST_RESULT_JSON:"
SUMMARY_PREFIX = "TEST_SUMMARY_JSON:"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run fixed-scene targeted planning tests for all source objects with place rules."
    )
    parser.add_argument("--scene-file", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--entrypoint", type=Path, default=DEFAULT_ENTRYPOINT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--child-python", type=Path, default=DEFAULT_CHILD_PYTHON)
    parser.add_argument("--objects", nargs="*", default=None, help="Optional subset of source objects to test.")
    parser.add_argument(
        "--obstacles",
        nargs="*",
        default=None,
        help=(
            "Optional fixed obstacle object names to pass to the child for every tested object. "
            "If omitted, all scene objects except the current target are used."
        ),
    )
    parser.add_argument("--render-mode", type=str, default="none")
    parser.add_argument("--repetitions", type=int, default=1, help="Run each selected object this many times.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory to store per-object child logs.")
    parser.add_argument("--per-object-timeout", type=float, default=300.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    parser.add_argument("--post-final-grace-seconds", type=float, default=2.0)
    parser.add_argument(
        "--print-child-logs",
        "--print",
        dest="print_child_logs",
        action="store_true",
        help="Stream child planner logs. Disabled by default to keep batch output compact.",
    )
    return parser.parse_known_args()


def _load_scene_object_names(scene_file: Path) -> list[str]:
    data = json.loads(scene_file.read_text())
    objects = data.get("objects", {})
    if not isinstance(objects, dict):
        return []
    return sorted(str(name) for name in objects.keys())


def _resolve_test_objects(scene_object_names: list[str], requested_objects: list[str] | None) -> list[str]:
    available_sources = [name for name in list_place_rule_sources() if name in scene_object_names]
    if requested_objects:
        requested = [str(name) for name in requested_objects]
        return [name for name in available_sources if name in requested]
    return available_sources


def _extract_failure_reason(stdout: str, stderr: str) -> tuple[str, str]:
    combined = "\n".join(part for part in (stdout, stderr) if part)
    lines = combined.splitlines()

    def _last_matching_line(needle: str) -> str | None:
        for line in reversed(lines):
            if needle in line:
                return line.strip()
        return None

    runtime_errors = [line.strip() for line in lines if line.strip().startswith("RuntimeError:")]
    if runtime_errors:
        detail = runtime_errors[-1]
        if "child planner timed out after" in detail:
            return "planner_timeout", detail
        if 'Failed to find a supported physical device "cpu"' in detail:
            return "render_device_init_failed", detail
        return "runtime_error", detail

    other_errors = [
        line.strip()
        for line in lines
        if line.strip().startswith(("ValueError:", "ImportError:", "ModuleNotFoundError:"))
    ]
    if other_errors:
        detail = other_errors[-1]
        if detail.startswith("ModuleNotFoundError:"):
            return "module_not_found", detail
        if detail.startswith("ImportError:"):
            return "import_error", detail
        return "value_error", detail

    checks = [
        ("[FAIL] no targeted-place rule is configured", "missing_targeted_place_rule"),
        ("[FAIL] no grasp candidate yielded a complete grasp->pre_place->release chain", "no_complete_grasp_place_release_chain"),
        ("[FAIL] direct cuRobo grasp planning failed", "direct_curobo_grasp_planning_failed"),
        ("[joint_search] no direct-place chain stayed feasible", "joint_search_no_feasible_direct_place_chain"),
        ("direct_grasp IK prefilter kept 0/", "direct_grasp_ik_prefilter_zero"),
        ("[two_step_grasp] no pregrasp candidates succeeded", "two_step_pregrasp_no_candidates"),
        ("two_step_grasp_pregrasp IK prefilter kept 0/", "two_step_pregrasp_ik_prefilter_zero"),
    ]
    for needle, code in checks:
        matched_line = _last_matching_line(needle)
        if matched_line is not None:
            return code, matched_line

    fail_lines = [line.strip() for line in lines if "[FAIL]" in line]
    if fail_lines:
        return "planner_fail", fail_lines[-1]
    if "final success = False" in combined:
        return "final_success_false", "planner reported final success = False without a more specific [FAIL] line"
    return "unknown_failure", "process failed without a recognized planner failure signature"


def _run_child(
    cmd: list[str],
    *,
    print_child_logs: bool,
    heartbeat_seconds: float,
    timeout_seconds: float,
    post_final_grace_seconds: float,
    object_name: str,
) -> tuple[int, str, str]:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    stdout_chunks: list[str] = []
    start_time = time.monotonic()
    last_output_time = start_time
    last_heartbeat_time = start_time
    timed_out = False
    forced_after_final = False
    final_status: bool | None = None
    final_status_time: float | None = None
    output_count = 0
    last_output_excerpt = "(no child output yet)"

    def _terminate_process_group() -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _maybe_capture_final_status(text: str) -> None:
        nonlocal final_status, final_status_time
        if final_status is not None:
            return
        if "final success = True" in text:
            final_status = True
            final_status_time = time.monotonic()
            return
        if "final success = False" in text:
            final_status = False
            final_status_time = time.monotonic()

    def _reader() -> None:
        nonlocal last_output_time, output_count, last_output_excerpt
        assert process.stdout is not None
        try:
            for line in process.stdout:
                stdout_chunks.append(line)
                output_count += 1
                last_output_time = time.monotonic()
                stripped = line.strip()
                if stripped:
                    last_output_excerpt = stripped[:200]
                _maybe_capture_final_status(line)
                if print_child_logs:
                    print(line, end="")
        except Exception:
            return

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    while True:
        now = time.monotonic()
        if timeout_seconds > 0 and (now - start_time) > timeout_seconds:
            timed_out = True
            _terminate_process_group()
            break
        if (
            final_status is not None
            and final_status_time is not None
            and post_final_grace_seconds >= 0
            and (now - final_status_time) > post_final_grace_seconds
        ):
            forced_after_final = True
            _terminate_process_group()
            break
        if process.poll() is not None and (not reader_thread.is_alive() or (now - last_output_time) > 0.5):
            break
        if heartbeat_seconds > 0 and (now - last_heartbeat_time) >= heartbeat_seconds:
            print(
                f"[test] heartbeat object={object_name} "
                f"elapsed={now - start_time:.1f}s idle_since_output={now - last_output_time:.1f}s "
                f"last_output={last_output_excerpt}"
            )
            last_heartbeat_time = now
        time.sleep(0.2)

    try:
        returncode = process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        _terminate_process_group()
        returncode = process.wait(timeout=5.0)
    if process.stdout is not None:
        try:
            process.stdout.close()
        except Exception:
            pass
    reader_thread.join(timeout=1.0)
    stderr = ""
    if timed_out:
        stderr = f"RuntimeError: child planner timed out after {timeout_seconds:.1f}s for object {object_name}"
        stdout_chunks.append("\n" + stderr + "\n")
    elif forced_after_final and final_status is not None:
        synthetic_returncode = 0 if final_status else 1
        returncode = synthetic_returncode
        stderr = (
            f"RuntimeError: child planner did not exit within {post_final_grace_seconds:.1f}s "
            f"after final success marker for object {object_name}; terminated after capturing final result"
        )
        stdout_chunks.append("\n" + stderr + "\n")
    return returncode, "".join(stdout_chunks), stderr


def _build_child_cmd(
    child_python: Path,
    entrypoint: Path,
    scene_file: Path,
    seed: int,
    render_mode: str,
    object_name: str,
    obstacle_names: list[str],
    extra_args: list[str],
) -> list[str]:
    cmd = [
        str(child_python),
        str(entrypoint),
        "--skip-foundationpose",
        "--fixed-scene-pose-file",
        str(scene_file),
        "--fixed-scene-strict",
        "--no-reselect-target-on-planning-failure",
        "--auto-execute",
        "--no-preview-trajectory-before-confirm",
        "--trajectory-preview-sleep",
        "0",
        "--render-mode",
        render_mode,
        "--seed",
        str(seed),
        "--object-name",
        object_name,
    ]
    if obstacle_names:
        cmd.extend(["--selected-obstacle-object-names", *obstacle_names])
    cmd.extend(extra_args)
    return cmd


def _emit_result(result: dict) -> None:
    print(f"{RESULT_PREFIX} {json.dumps(result, ensure_ascii=True, sort_keys=True)}")


def _safe_log_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def main() -> None:
    args, extra_args = parse_args()
    scene_file = args.scene_file.resolve()
    entrypoint = args.entrypoint.resolve()
    child_python = args.child_python.resolve()
    scene_object_names = _load_scene_object_names(scene_file)
    test_objects = _resolve_test_objects(scene_object_names, args.objects)
    repetitions = max(1, int(args.repetitions))
    log_dir = args.log_dir.resolve() if args.log_dir is not None else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    if not test_objects:
        result = {
            "scene_file": str(scene_file),
            "success": False,
            "failure_reason_code": "no_test_objects",
            "failure_reason_detail": "no source objects with place rules were found in the fixed scene",
        }
        print(f"{SUMMARY_PREFIX} {json.dumps(result, ensure_ascii=True, sort_keys=True)}")
        print(f"{SUMMARY_PREFIX} {json.dumps(result, ensure_ascii=True, sort_keys=True)}", file=sys.stderr)
        raise SystemExit(2)

    print(
        f"[test] fixed-scene batch planning: scene={scene_file} seed={args.seed} "
        f"objects={test_objects} render_mode={args.render_mode} repetitions={repetitions}"
    )

    results: list[dict] = []
    fixed_obstacles = None
    if args.obstacles is not None:
        fixed_obstacles = [str(name) for name in args.obstacles if str(name)]
    for repetition in range(1, repetitions + 1):
        for index, object_name in enumerate(test_objects, start=1):
            if fixed_obstacles is None:
                obstacle_names = [name for name in scene_object_names if name != object_name]
            else:
                obstacle_names = [name for name in fixed_obstacles if name != object_name]
            print(
                f"\n[test] repetition {repetition}/{repetitions} object {index}/{len(test_objects)}: "
                f"{object_name} obstacles={obstacle_names}"
            )
            cmd = _build_child_cmd(
                child_python=child_python,
                entrypoint=entrypoint,
                scene_file=scene_file,
                seed=args.seed,
                render_mode=args.render_mode,
                object_name=object_name,
                obstacle_names=obstacle_names,
                extra_args=extra_args,
            )
            returncode, stdout, stderr = _run_child(
                cmd,
                print_child_logs=bool(args.print_child_logs),
                heartbeat_seconds=float(args.heartbeat_seconds),
                timeout_seconds=float(args.per_object_timeout),
                post_final_grace_seconds=float(args.post_final_grace_seconds),
                object_name=object_name,
            )
            final_success_seen = "final success = True" in stdout
            benign_post_success_x_error = (
                final_success_seen
                and returncode != 0
                and "X Error of failed request:" in stdout
                and "BadWindow" in stdout
            )
            success = (returncode == 0 and final_success_seen) or benign_post_success_x_error
            effective_returncode = 0 if success else (returncode if returncode != 0 else 1)
            failure_code = "ok"
            failure_detail = ""
            if not success:
                failure_code, failure_detail = _extract_failure_reason(stdout, stderr)
            log_path = None
            if log_dir is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                scene_stem = _safe_log_name(scene_file.stem)
                object_stem = _safe_log_name(object_name)
                log_path = log_dir / f"{timestamp}_{scene_stem}_{object_stem}_rep{repetition}.log"
                header = {
                    "cmd": cmd,
                    "scene_file": str(scene_file),
                    "object_name": object_name,
                    "repetition": repetition,
                    "returncode": effective_returncode,
                    "raw_returncode": returncode,
                    "success": success,
                    "failure_reason_code": failure_code,
                    "failure_reason_detail": failure_detail,
                }
                log_path.write_text(
                    "# TEST_CHILD_METADATA "
                    + json.dumps(header, ensure_ascii=True, sort_keys=True)
                    + "\n"
                    + stdout,
                    encoding="utf-8",
                )
            result = {
                "object_name": object_name,
                "repetition": repetition,
                "success": success,
                "returncode": effective_returncode,
                "raw_returncode": returncode,
                "failure_reason_code": failure_code,
                "failure_reason_detail": failure_detail,
                "log_path": str(log_path) if log_path is not None else "",
            }
            results.append(result)
            _emit_result(result)

    passed = sum(1 for item in results if bool(item["success"]))
    total = len(results)
    summary = {
        "scene_file": str(scene_file),
        "seed": int(args.seed),
        "tested_objects": [item["object_name"] for item in results],
        "repetitions": repetitions,
        "passed": passed,
        "failed": total - passed,
        "success_rate": (float(passed) / float(total)) if total else 0.0,
        "success": passed == total,
    }
    print(f"\n{SUMMARY_PREFIX} {json.dumps(summary, ensure_ascii=True, sort_keys=True)}")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
