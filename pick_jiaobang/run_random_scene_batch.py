#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = THIS_DIR / "test_scenes" / "generated_random_batch_002" / "manifest.json"
DEFAULT_ENTRYPOINT = THIS_DIR / "rm75_jiaobang_pick_place_targeted_curobo_direct_pre_place_sam6d.py"
DEFAULT_TEST_RUNNER = THIS_DIR / "test_targeted_fixed_scene.py"
DEFAULT_ROBOT_CFG = THIS_DIR / "curobo_rm75_config" / "rm75.yml"
DEFAULT_LOG_ROOT = THIS_DIR / "logs" / "random_batch_002_full_64"
DEFAULT_OBJECTS = ("bi", "carriot", "gluestick", "hongshupian", "lvmukuai", "shuazi", "tennis")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


ENV_FAILURE_CODES = {
    "render_device_init_failed",
    "scene_load_failed",
    "fixed_scene_load_failed",
    "timeout",
    "scene_runner_failed",
}


def _safe_scene_stem(path: str | Path) -> str:
    return Path(str(path)).stem.replace("/", "_").replace(" ", "_")


def _read_text(path: str | Path | None) -> str:
    if not path:
        return ""
    try:
        return Path(path).read_text(errors="replace")
    except Exception:
        return ""


def classify_result(result: dict) -> str:
    if bool(result.get("success", False)):
        return "success"

    code = str(result.get("failure_reason_code", "") or "")
    detail = str(result.get("failure_reason_detail", "") or "")
    log_text = _read_text(result.get("log_path"))
    text = f"{code}\n{detail}\n{log_text}".lower()

    if code in ENV_FAILURE_CODES:
        return "environment_or_device_failure"
    if "failed to find a supported physical device" in text or "can't initialize nvml" in text:
        return "environment_or_device_failure"
    if "cuda device was not found" in text or "render_device_init_failed" in text:
        return "environment_or_device_failure"

    attach_markers = (
        "attaching object box",
        "attached long-axis object",
        "attached sphere bottom after attach",
        "transport attached sphere bottom after attach",
        "attached object collision spheres",
        "restored attached object spheres",
    )
    post_grasp_markers = (
        "post_grasp_lift",
        "transport-hover pair candidate count",
        "transport_to_hover",
        "final_contact",
        "place_state]",
    )
    if any(marker in text for marker in attach_markers) and any(marker in text for marker in post_grasp_markers):
        return "placement_failure_after_grasp"

    if "final_contact_approach failed" in text:
        return "placement_failure_after_grasp"
    if "transport-hover" in text and "attached" in text:
        return "placement_failure_after_grasp"

    return "grasp_blocked_or_unreachable"


def parse_test_output(stdout_path: Path) -> tuple[list[dict], dict | None]:
    results: list[dict] = []
    summary = None
    if not stdout_path.exists():
        return results, summary
    for line in stdout_path.read_text(errors="replace").splitlines():
        if line.startswith("TEST_RESULT_JSON:"):
            payload = line.split("TEST_RESULT_JSON:", 1)[1].strip()
            try:
                results.append(json.loads(payload))
            except json.JSONDecodeError:
                pass
        elif line.startswith("TEST_SUMMARY_JSON:"):
            payload = line.split("TEST_SUMMARY_JSON:", 1)[1].strip()
            try:
                summary = json.loads(payload)
            except json.JSONDecodeError:
                pass
    return results, summary


def _extract_failure_reason_from_text(text: str) -> tuple[str, str]:
    lines = text.splitlines()

    def _last_matching_line(needle: str) -> str | None:
        for line in reversed(lines):
            if needle in line:
                return line.strip()
        return None

    runtime_errors = [line.strip() for line in lines if line.strip().startswith("RuntimeError:")]
    if runtime_errors:
        detail = runtime_errors[-1]
        if "timed out" in detail:
            return "timeout", detail
        if 'Failed to find a supported physical device "cpu"' in detail:
            return "render_device_init_failed", detail
        return "runtime_error", detail

    checks = [
        ("[FAIL] no targeted-place rule is configured", "missing_targeted_place_rule"),
        ("[FAIL] final_contact_approach failed for all reachable hover candidates", "final_contact_approach_failed"),
        ("[FAIL] no grasp candidate yielded a complete grasp->pre_place->release chain", "no_complete_grasp_place_release_chain"),
        ("[FAIL] direct cuRobo grasp planning failed", "direct_curobo_grasp_planning_failed"),
        ("[joint_search] no direct-place chain stayed feasible", "joint_search_no_feasible_direct_place_chain"),
        ("direct_grasp IK prefilter kept 0/", "direct_grasp_ik_prefilter_zero"),
        ("[two_step_grasp] no pregrasp candidates succeeded", "two_step_pregrasp_no_candidates"),
        ("two_step_grasp_pregrasp IK prefilter kept 0/", "two_step_pregrasp_ik_prefilter_zero"),
    ]
    for needle, code in checks:
        matched = _last_matching_line(needle)
        if matched is not None:
            return code, matched

    fail_lines = [line.strip() for line in lines if "[FAIL]" in line]
    if fail_lines:
        return "planner_fail", fail_lines[-1]
    if "final success = false" in text.lower():
        return "final_success_false", "planner reported final success = False"
    return "unknown_failure", "process failed without a recognized planner failure signature"


def parse_single_process_output(stdout_path: Path, scene_log_dir: Path, objects: list[str]) -> tuple[list[dict], dict | None]:
    if not stdout_path.exists():
        return [], None
    lines = stdout_path.read_text(errors="replace").splitlines()
    object_logs = scene_log_dir / "object_logs"
    object_logs.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    current_obj: str | None = None
    current_cycle: str | None = None
    current_lines: list[str] = []
    selected_prefix = "randomly selected target object:"
    object_names = sorted((str(name) for name in objects), key=len, reverse=True)

    def _parse_selected_object(line: str) -> str | None:
        if selected_prefix in line:
            payload = line.split(selected_prefix, 1)[1].strip()
        elif "selected target object:" in line:
            payload = line.split("selected target object:", 1)[1].strip()
        else:
            return None
        for object_name in object_names:
            if not payload.startswith(object_name):
                continue
            tail = payload[len(object_name) : len(object_name) + 1]
            if not tail or tail.isspace() or tail == "\x1b":
                return object_name
        # Fall back to the first clean token only if it is an expected object.
        cleaned = ANSI_ESCAPE_RE.sub("", payload).strip()
        token = cleaned.split()[0] if cleaned.split() else ""
        return token if token in objects else None

    def _finish_segment(success: bool, cycle_label: str | None) -> None:
        nonlocal current_obj, current_cycle, current_lines
        if current_obj is None:
            return
        segment_text = "\n".join(current_lines) + "\n"
        scene_stem = _safe_scene_stem(stdout_path.parent.name)
        cycle_stem = _safe_scene_stem(cycle_label or current_cycle or "cycle")
        object_stem = _safe_scene_stem(current_obj)
        log_path = object_logs / f"{scene_stem}_{cycle_stem}_{object_stem}.log"
        log_path.write_text(segment_text, encoding="utf-8")
        failure_code = "ok"
        failure_detail = ""
        if not success:
            failure_code, failure_detail = _extract_failure_reason_from_text(segment_text)
        results.append(
            {
                "object_name": current_obj,
                "cycle_label": str(cycle_label or current_cycle or ""),
                "repetition": 1,
                "success": bool(success),
                "returncode": 0 if success else 1,
                "raw_returncode": 0 if success else 1,
                "failure_reason_code": failure_code,
                "failure_reason_detail": failure_detail,
                "log_path": str(log_path),
            }
        )
        current_obj = None
        current_cycle = None
        current_lines = []

    for line in lines:
        selected_object = _parse_selected_object(line)
        if selected_object is not None:
            if current_obj is not None:
                _finish_segment(False, current_cycle)
            current_obj = selected_object
            current_cycle = None
            current_lines = [line]
            continue
        if current_obj is not None:
            current_lines.append(line)
        if line.startswith("cycle ") and " success = " in line:
            parts = line.split()
            current_cycle = parts[1] if len(parts) > 1 else current_cycle
            success = "success = True" in line
            _finish_segment(success, current_cycle)

    if current_obj is not None:
        _finish_segment(False, current_cycle)

    attempts = list(results)
    collapsed_by_object: dict[str, dict] = {}
    for item in attempts:
        name = str(item.get("object_name"))
        current = collapsed_by_object.get(name)
        if current is None or bool(item.get("success", False)) or not bool(current.get("success", False)):
            merged = dict(item)
            object_attempts = [dict(attempt) for attempt in attempts if str(attempt.get("object_name")) == name]
            merged["attempt_count"] = len(object_attempts)
            merged["attempts"] = object_attempts
            collapsed_by_object[name] = merged
    results = [collapsed_by_object[name] for name in objects if name in collapsed_by_object]

    seen = {str(item.get("object_name")) for item in results}
    whole_text = "\n".join(lines)
    for object_name in objects:
        if object_name in seen:
            continue
        failure_code, failure_detail = _extract_failure_reason_from_text(whole_text)
        results.append(
            {
                "object_name": object_name,
                "repetition": 1,
                "success": False,
                "returncode": 1,
                "raw_returncode": 1,
                "failure_reason_code": failure_code,
                "failure_reason_detail": failure_detail,
                "log_path": str(stdout_path),
            }
        )

    passed = sum(1 for item in results if bool(item.get("success", False)))
    summary = {
        "tested_objects": [str(item.get("object_name")) for item in results],
        "passed": passed,
        "failed": len(results) - passed,
        "success_rate": float(passed) / float(len(results)) if results else 0.0,
        "success": bool(results) and passed == len(results),
    }
    return results, summary


def run_scene(scene_file: Path, args, scene_log_dir: Path) -> tuple[list[dict], dict | None, int, float]:
    scene_log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = scene_log_dir / "runner_stdout.log"
    cmd = [
        str(Path(args.child_python).expanduser()),
        str(DEFAULT_TEST_RUNNER),
        "--child-python",
        str(Path(args.child_python).expanduser()),
        "--entrypoint",
        str(Path(args.entrypoint).expanduser()),
        "--scene-file",
        str(scene_file),
        "--render-mode",
        str(args.render_mode),
        "--objects",
        *list(args.objects),
        "--repetitions",
        str(int(args.repetitions)),
        "--log-dir",
        str(scene_log_dir / "object_logs"),
        "--per-object-timeout",
        str(int(args.per_object_timeout)),
        "--heartbeat-seconds",
        str(int(args.heartbeat_seconds)),
        "--post-final-grace-seconds",
        str(int(args.post_final_grace_seconds)),
        "--curobo-rm75-robot-cfg",
        str(Path(args.curobo_rm75_robot_cfg).expanduser()),
        "--trajectory-preview-sleep",
        str(float(args.trajectory_preview_sleep)),
    ]
    if args.extra_arg:
        cmd.extend(list(args.extra_arg))

    started = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(THIS_DIR / "logs" / ".matplotlib_cache"))
    with stdout_path.open("w", encoding="utf-8") as out:
        out.write("# BATCH_SCENE_CMD " + json.dumps(cmd, ensure_ascii=False) + "\n")
        out.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(THIS_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            out.write(line)
            out.flush()
            if (
                line.startswith("TEST_RESULT_JSON:")
                or line.startswith("TEST_SUMMARY_JSON:")
                or "[test] heartbeat" in line
            ):
                print(line.rstrip(), flush=True)
        returncode = proc.wait()
    elapsed = time.perf_counter() - started
    results, summary = parse_test_output(stdout_path)
    return results, summary, int(returncode), elapsed


def run_scene_single_process(scene_file: Path, args, scene_log_dir: Path) -> tuple[list[dict], dict | None, int, float]:
    scene_log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = scene_log_dir / "runner_stdout.log"
    cmd = [
        str(Path(args.child_python).expanduser()),
        str(Path(args.entrypoint).expanduser()),
        "--curobo-rm75-robot-cfg",
        str(Path(args.curobo_rm75_robot_cfg).expanduser()),
        "--cycle-object-names",
        *list(args.objects),
        "--tracked-scene-object-names",
        *list(args.tracked_scene_object_names),
        "--skip-foundationpose",
        "--fixed-scene-pose-file",
        str(scene_file),
        "--fixed-scene-strict",
        "--render-mode",
        str(args.render_mode),
        "--auto-execute",
        "--trajectory-preview-sleep",
        str(float(args.trajectory_preview_sleep)),
        "--seed",
        str(int(args.seed)),
    ]
    if bool(getattr(args, "curobo_debug", False)):
        cmd.append("--curobo-debug")
    if str(args.target_selection_order) == "random":
        cmd.append("--random-cycle-targets")
    elif str(args.target_selection_order) == "risk_aware":
        cmd.append("--risk-aware-cycle-targets")
    if args.skip_return_to_cycle_start:
        cmd.append("--skip-return-to-cycle-start")
    if args.skip_post_place_clearance:
        cmd.append("--skip-post-place-clearance")
    if args.extra_arg:
        cmd.extend(list(args.extra_arg))

    started = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(THIS_DIR / "logs" / ".matplotlib_cache"))
    with stdout_path.open("w", encoding="utf-8") as out:
        out.write("# BATCH_SCENE_CMD " + json.dumps(cmd, ensure_ascii=False) + "\n")
        out.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(THIS_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        last_heartbeat = time.perf_counter()
        for line in proc.stdout:
            out.write(line)
            out.flush()
            now = time.perf_counter()
            if (
                line.startswith("[cycle ")
                or line.startswith("cycle ")
                or line.startswith("final success")
                or (float(args.heartbeat_seconds) > 0 and now - last_heartbeat >= float(args.heartbeat_seconds))
            ):
                print(line.rstrip(), flush=True)
                last_heartbeat = now
        returncode = proc.wait()
    elapsed = time.perf_counter() - started
    results, summary = parse_single_process_output(stdout_path, scene_log_dir, list(args.objects))
    return results, summary, int(returncode), elapsed


def write_summary(log_root: Path, manifest: dict, scene_records: list[dict]) -> None:
    flat_results = []
    flat_attempts = []
    for scene in scene_records:
        for result in scene.get("results", []):
            item = dict(result)
            item["scene_file"] = scene["scene_file"]
            item["scene_mode"] = scene.get("mode")
            item["classification"] = classify_result(result)
            flat_results.append(item)
            attempts = list(result.get("attempts") or [])
            if not attempts:
                attempts = [result]
            for attempt_index, attempt in enumerate(attempts, start=1):
                attempt_item = dict(attempt)
                attempt_item["attempt_index"] = attempt_index
                attempt_item["scene_file"] = scene["scene_file"]
                attempt_item["scene_mode"] = scene.get("mode")
                attempt_item["final_object_success"] = bool(result.get("success", False))
                attempt_item["classification"] = classify_result(attempt)
                flat_attempts.append(attempt_item)

    class_counts = Counter(item["classification"] for item in flat_results)
    attempt_failure_counts = Counter(
        item["classification"] for item in flat_attempts if not bool(item.get("success", False))
    )
    object_counts: dict[str, Counter] = defaultdict(Counter)
    mode_counts: dict[str, Counter] = defaultdict(Counter)
    for item in flat_results:
        object_counts[str(item.get("object_name", ""))][item["classification"]] += 1
        mode_counts[str(item.get("scene_mode", ""))][item["classification"]] += 1

    summary = {
        "manifest": manifest.get("output_dir"),
        "scene_count": len(scene_records),
        "object_result_count": len(flat_results),
        "classification_counts": dict(class_counts),
        "attempt_failure_counts": dict(attempt_failure_counts),
        "object_counts": {name: dict(counter) for name, counter in sorted(object_counts.items())},
        "mode_counts": {name: dict(counter) for name, counter in sorted(mode_counts.items())},
        "scenes": scene_records,
    }
    (log_root / "batch_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    lines = []
    lines.append("# Random Batch Summary\n")
    lines.append(f"- Scenes tested: {len(scene_records)}")
    lines.append(f"- Object results: {len(flat_results)}")
    lines.append(f"- Success: {class_counts.get('success', 0)}")
    lines.append(f"- Grasp blocked/unreachable: {class_counts.get('grasp_blocked_or_unreachable', 0)}")
    lines.append(f"- Placement failure after grasp: {class_counts.get('placement_failure_after_grasp', 0)}")
    lines.append(f"- Environment/device failure: {class_counts.get('environment_or_device_failure', 0)}")
    lines.append(
        "- Failed attempts before final per-object result: "
        + str(sum(attempt_failure_counts.values()))
    )
    lines.append(
        "- Failed-attempt placement_after_grasp: "
        + str(attempt_failure_counts.get("placement_failure_after_grasp", 0))
    )
    lines.append("")
    lines.append("## Per Object")
    lines.append("")
    lines.append("| object | success | grasp_blocked_or_unreachable | placement_failure_after_grasp | environment_or_device_failure |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in sorted(object_counts):
        counter = object_counts[name]
        lines.append(
            f"| {name} | {counter.get('success', 0)} | "
            f"{counter.get('grasp_blocked_or_unreachable', 0)} | "
            f"{counter.get('placement_failure_after_grasp', 0)} | "
            f"{counter.get('environment_or_device_failure', 0)} |"
        )
    lines.append("")
    lines.append("## Placement Failures After Grasp")
    placement_failures = [item for item in flat_results if item["classification"] == "placement_failure_after_grasp"]
    if not placement_failures:
        lines.append("")
        lines.append("None observed.")
    else:
        for item in placement_failures:
            lines.append(
                f"- `{item.get('object_name')}` in `{Path(str(item.get('scene_file'))).name}`: "
                f"{item.get('failure_reason_code')} | {item.get('failure_reason_detail')} | "
                f"log `{item.get('log_path')}`"
            )
    lines.append("")
    lines.append("## Failed Attempts Before Final Result")
    failed_attempts = [item for item in flat_attempts if not bool(item.get("success", False))]
    if not failed_attempts:
        lines.append("")
        lines.append("None observed.")
    else:
        for item in failed_attempts[:80]:
            final_note = "deferred_then_success" if bool(item.get("final_object_success", False)) else "final_failure"
            lines.append(
                f"- `{item.get('object_name')}` attempt {item.get('attempt_index')} in "
                f"`{Path(str(item.get('scene_file'))).name}` ({final_note}): "
                f"{item.get('classification')} | {item.get('failure_reason_code')} | "
                f"log `{item.get('log_path')}`"
            )
        if len(failed_attempts) > 80:
            lines.append(f"- ... {len(failed_attempts) - 80} more")
    lines.append("")
    lines.append("## Grasp Blocked Or Unreachable")
    grasp_failures = [item for item in flat_results if item["classification"] == "grasp_blocked_or_unreachable"]
    for item in grasp_failures[:80]:
        lines.append(
            f"- `{item.get('object_name')}` in `{Path(str(item.get('scene_file'))).name}`: "
            f"{item.get('failure_reason_code')} | log `{item.get('log_path')}`"
        )
    if len(grasp_failures) > 80:
        lines.append(f"- ... {len(grasp_failures) - 80} more")
    lines.append("")
    (log_root / "batch_summary.md").write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all generated random fixed scenes through the headless targeted-place test runner.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--child-python", default=sys.executable)
    parser.add_argument("--entrypoint", type=Path, default=DEFAULT_ENTRYPOINT)
    parser.add_argument("--curobo-rm75-robot-cfg", type=Path, default=DEFAULT_ROBOT_CFG)
    parser.add_argument("--objects", nargs="+", default=list(DEFAULT_OBJECTS))
    parser.add_argument("--tracked-scene-object-names", nargs="+", default=["desk", "bitong"])
    parser.add_argument("--render-mode", default="none")
    parser.add_argument("--runner-mode", choices=["single-process", "per-object"], default="single-process")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-selection-order", choices=["random", "risk_aware"], default="risk_aware")
    parser.add_argument("--curobo-debug", action="store_true")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--per-object-timeout", type=int, default=900)
    parser.add_argument("--heartbeat-seconds", type=int, default=90)
    parser.add_argument("--post-final-grace-seconds", type=int, default=5)
    parser.add_argument("--trajectory-preview-sleep", type=float, default=0.0)
    parser.add_argument("--skip-return-to-cycle-start", action="store_true")
    parser.add_argument("--skip-post-place-clearance", action="store_true")
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    scenes = list(manifest.get("scenes", []))
    if int(args.start_index) > 0:
        scenes = scenes[int(args.start_index) :]
    if int(args.max_scenes) > 0:
        scenes = scenes[: int(args.max_scenes)]

    log_root = Path(args.log_root).expanduser().resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    scene_records_path = log_root / "scene_results.jsonl"

    existing: dict[str, dict] = {}
    if scene_records_path.exists() and not bool(args.force):
        for line in scene_records_path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            existing[str(record.get("scene_file"))] = record

    scene_records: list[dict] = []
    if not bool(args.force):
        scene_records.extend(existing.values())

    print(f"[batch] manifest={manifest_path} scenes_to_consider={len(scenes)} log_root={log_root}", flush=True)
    for idx, scene in enumerate(scenes, start=1):
        scene_file = Path(scene["file"]).expanduser().resolve()
        scene_key = str(scene_file)
        if scene_key in existing and not bool(args.force):
            print(f"[batch] skip existing {idx}/{len(scenes)} {scene_file.name}", flush=True)
            continue
        scene_dir = log_root / f"{idx:03d}_{_safe_scene_stem(scene_file)}"
        print(f"[batch] scene {idx}/{len(scenes)} mode={scene.get('mode')} file={scene_file.name}", flush=True)
        if args.runner_mode == "per-object":
            results, test_summary, returncode, elapsed = run_scene(scene_file, args, scene_dir)
        else:
            results, test_summary, returncode, elapsed = run_scene_single_process(scene_file, args, scene_dir)
        record = {
            "scene_file": scene_key,
            "mode": scene.get("mode"),
            "base_scene": scene.get("base_scene"),
            "returncode": returncode,
            "elapsed_s": round(elapsed, 3),
            "results": results,
            "test_summary": test_summary,
            "scene_log_dir": str(scene_dir),
        }
        with scene_records_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        scene_records.append(record)
        counts = Counter(classify_result(item) for item in results)
        print(
            f"[batch] scene_done {scene_file.name}: "
            f"success={counts.get('success', 0)} "
            f"grasp_blocked={counts.get('grasp_blocked_or_unreachable', 0)} "
            f"placement_fail={counts.get('placement_failure_after_grasp', 0)} "
            f"env_fail={counts.get('environment_or_device_failure', 0)} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        write_summary(log_root, manifest, scene_records)

    write_summary(log_root, manifest, scene_records)
    print(f"[batch] summary: {log_root / 'batch_summary.md'}", flush=True)
    print(f"[batch] summary_json: {log_root / 'batch_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
