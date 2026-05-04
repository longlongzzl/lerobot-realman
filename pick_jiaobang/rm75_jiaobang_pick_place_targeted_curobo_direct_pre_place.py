#!/usr/bin/env python3
from __future__ import annotations

import re
import atexit
import copy
import gc
import json
import time
import os
import random
import sys
import threading
import traceback
import types
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import rm75_jiaobang_pick_place_targeted as targeted
import rm75_jiaobang_pick_place_targeted_curobo as curobo_wrapper

_PROFILE_RECORDS: list[dict] = []
_PROFILE_PATH: Path | None = None
_PROFILE_REGISTERED = False
_PROFILE_RUN_HEADER_WRITTEN = False
_PROFILE_COUNTERS: Counter = Counter()
_PROFILE_IK_BATCH_SIZE_HIST: Counter = Counter()
_PROFILE_THREAD_LOCAL = threading.local()
_CUROBO_GPU_LOCK = threading.RLock()
_PROFILE_COUNTER_FIELDS = (
    "ik_batch_call_count",
    "ik_goal_count",
    "ik_cuda_graph_solve_count",
    "ik_cuda_graph_requested_goal_count",
    "ik_cuda_graph_padded_goal_count",
    "motiongen_call_count",
    "constrained_linear_call_count",
    "graph_call_count",
    "prefilter_q_goal_motiongen_count",
    "prefilter_q_goal_success_count",
    "timeout_count",
    "fallback_count",
    "world_refresh_count",
    "attach_object_count",
)


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, SimpleNamespace):
        return _jsonable(vars(value))
    if isinstance(value, types.ModuleType):
        return f"<module {getattr(value, '__name__', 'unknown')}>"
    if callable(value):
        return f"<callable {getattr(value, '__name__', type(value).__name__)}>"
    return value


def _profile_enabled(args) -> bool:
    return bool(getattr(args, "planning_profile_enabled", True))


def _bump_profile_counter(name: str, amount: int = 1) -> None:
    if not name or amount == 0:
        return
    if bool(getattr(_PROFILE_THREAD_LOCAL, "suspend_counters", False)):
        return
    _PROFILE_COUNTERS[str(name)] += int(amount)


@contextmanager
def _suspend_profile_counters_for_thread():
    prev = bool(getattr(_PROFILE_THREAD_LOCAL, "suspend_counters", False))
    _PROFILE_THREAD_LOCAL.suspend_counters = True
    try:
        yield
    finally:
        _PROFILE_THREAD_LOCAL.suspend_counters = prev


def _snapshot_profile_counters() -> Counter:
    return Counter(_PROFILE_COUNTERS)


def _snapshot_profile_ik_batch_size_hist() -> Counter:
    return Counter(_PROFILE_IK_BATCH_SIZE_HIST)


def _profile_counter_delta(start: Counter) -> dict[str, int]:
    return {
        key: int(_PROFILE_COUNTERS.get(key, 0) - start.get(key, 0))
        for key in _PROFILE_COUNTER_FIELDS
    }


def _profile_ik_batch_size_hist_delta(start: Counter) -> dict[str, int]:
    delta = Counter(_PROFILE_IK_BATCH_SIZE_HIST)
    delta.subtract(start)
    return {str(key): int(value) for key, value in sorted(delta.items(), key=lambda kv: int(kv[0])) if int(value) > 0}


def _copy_last_candidate_counts_to_profile(prof: dict, planner) -> None:
    for attr, field in (
        ("_last_candidate_count_in", "candidate_count_in"),
        ("_last_candidate_count_after_ik", "candidate_count_after_ik"),
        ("_last_candidate_count_motiongen", "candidate_count_motiongen"),
    ):
        value = getattr(planner, attr, None)
        if value is not None:
            prof[field] = int(value)


def _status_indicates_timeout(status) -> bool:
    return "TIMEOUT" in str(status).upper()


def _count_motiongen_status(status) -> None:
    if _status_indicates_timeout(status):
        _bump_profile_counter("timeout_count")


def _count_motiongen_result(result) -> None:
    _count_motiongen_status(getattr(result, "status", ""))


def _graph_enabled_from_kwargs(kwargs: dict) -> bool:
    return bool(kwargs.get("enable_graph", False))


def _get_or_create_curobo_planner_serialized(args):
    # cuRobo CUDA graph capture is process-global enough that background planner
    # creation can trip over an active foreground capture. Keep GPU-facing cuRobo
    # entry points serialized; background work still runs during robot execution.
    with _CUROBO_GPU_LOCK:
        return curobo_wrapper._get_or_create_curobo_planner(args)


def _count_ik_batch_result_details(results) -> None:
    seen_cuda_graph_chunks: set[int] = set()
    for result in list(results or []):
        debug = getattr(result, "debug", None)
        if not isinstance(debug, dict) or not bool(debug.get("cuda_graph_batch", False)):
            continue
        chunk_index = int(debug.get("cuda_graph_chunk_index", 0) or 0)
        if chunk_index in seen_cuda_graph_chunks:
            continue
        seen_cuda_graph_chunks.add(chunk_index)
        _bump_profile_counter("ik_cuda_graph_solve_count")
        _bump_profile_counter("ik_cuda_graph_requested_goal_count", int(debug.get("requested_batch_size", 0) or 0))
        _bump_profile_counter("ik_cuda_graph_padded_goal_count", int(debug.get("fixed_batch_size", 0) or 0))


def _profile_solve_batch_start_goal_ik(planner, *call_args, **kwargs):
    _bump_profile_counter("ik_batch_call_count")
    goal_poses = call_args[1] if len(call_args) > 1 else kwargs.get("goal_poses", [])
    try:
        goal_count = len(goal_poses or [])
    except TypeError:
        goal_count = 0
    _bump_profile_counter("ik_goal_count", goal_count)
    if goal_count > 0:
        _PROFILE_IK_BATCH_SIZE_HIST[str(goal_count)] += 1
    with _CUROBO_GPU_LOCK:
        results = planner.solve_batch_start_goal_ik(*call_args, **kwargs)
    _count_ik_batch_result_details(results)
    return results


def _profile_solve_ik(planner, *args, **kwargs):
    _bump_profile_counter("ik_batch_call_count")
    _bump_profile_counter("ik_goal_count")
    _PROFILE_IK_BATCH_SIZE_HIST["1"] += 1
    with _CUROBO_GPU_LOCK:
        return planner.solve_ik(*args, **kwargs)


def _profile_fast_chain_solve_batch_start_goal_ik(args, planner, start_qs, goal_poses, *, num_seeds: int):
    return _profile_solve_batch_start_goal_ik(
        planner,
        start_qs,
        goal_poses,
        num_seeds=int(num_seeds),
        use_cuda_graph_batch=bool(getattr(args, "fast_chain_cuda_graph_ik", False)),
        cuda_graph_batch_size=int(getattr(args, "fast_chain_cuda_graph_ik_max_batch_size", 128) or 0),
        cuda_graph_fixed_batch_size=int(getattr(args, "fast_chain_cuda_graph_ik_fixed_batch_size", 16) or 0),
    )


def _profile_plan_to_pose(planner, *args, **kwargs):
    _bump_profile_counter("motiongen_call_count")
    if _graph_enabled_from_kwargs(kwargs):
        _bump_profile_counter("graph_call_count")
    with _CUROBO_GPU_LOCK:
        result = planner.plan_to_pose(*args, **kwargs)
    _count_motiongen_result(result)
    return result


def _profile_plan_constrained_linear_to_pose(planner, *args, **kwargs):
    """Profile cuRobo PoseCostMetric straight-line primitives separately.

    These calls still use cuRobo's MotionGen implementation internally, but they
    are not the free-space transport search. Keeping a separate counter makes the
    profile match the intended architecture: one full MotionGen transport plus
    short constrained-line primitives on the selected pair.
    """
    _bump_profile_counter("constrained_linear_call_count")
    with _CUROBO_GPU_LOCK:
        result = planner.plan_to_pose(*args, **kwargs)
    _count_motiongen_result(result)
    return result


def _profile_plan_to_joint_state(planner, *args, **kwargs):
    _bump_profile_counter("motiongen_call_count")
    if _graph_enabled_from_kwargs(kwargs):
        _bump_profile_counter("graph_call_count")
    with _CUROBO_GPU_LOCK:
        result = planner.plan_to_joint_state(*args, **kwargs)
    _count_motiongen_result(result)
    return result


def _profile_plan_goalset_to_poses(planner, *args, **kwargs):
    _bump_profile_counter("motiongen_call_count")
    if _graph_enabled_from_kwargs(kwargs):
        _bump_profile_counter("graph_call_count")
    with _CUROBO_GPU_LOCK:
        result = planner.plan_goalset_to_poses(*args, **kwargs)
    _count_motiongen_result(result)
    return result


def _profile_plan_batch_to_poses(planner, *args, **kwargs):
    _bump_profile_counter("motiongen_call_count")
    if _graph_enabled_from_kwargs(kwargs):
        _bump_profile_counter("graph_call_count")
    with _CUROBO_GPU_LOCK:
        results = planner.plan_batch_to_poses(*args, **kwargs)
    for result in list(results or []):
        _count_motiongen_result(result)
    return results


def _profile_plan_batch_start_goal_pairs(planner, *args, **kwargs):
    _bump_profile_counter("motiongen_call_count")
    if _graph_enabled_from_kwargs(kwargs):
        _bump_profile_counter("graph_call_count")
    with _CUROBO_GPU_LOCK:
        results = planner.plan_batch_start_goal_pairs(*args, **kwargs)
    for result in list(results or []):
        _count_motiongen_result(result)
    return results


def _profile_object_name(args) -> str:
    return str(getattr(args, "object_name", "") or "unknown")


def _safe_profile_filename_part(value, *, fallback: str = "run", max_len: int = 80) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    if not text:
        text = fallback
    return text[:max_len] or fallback


def _profile_run_name(args) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    target = curobo_wrapper.normalize_object_name(getattr(args, "object_name", None))
    if target is None:
        cycle_names = list(getattr(args, "cycle_object_names", None) or [])
        target = f"cycle{len(cycle_names)}" if cycle_names else "run"
    target_part = _safe_profile_filename_part(target, fallback="run", max_len=48)
    return f"{timestamp}_{target_part}_pid{os.getpid()}.jsonl"


def _profile_auto_path(args) -> Path:
    raw_jsonl = getattr(args, "planning_profile_jsonl", None)
    if raw_jsonl:
        requested = Path(str(raw_jsonl)).expanduser()
        raw_text = str(raw_jsonl)
        if raw_text.endswith(("/", os.sep)) or requested.suffix.lower() != ".jsonl":
            profile_dir = requested
        else:
            return requested
    else:
        profile_dir = Path(
            str(getattr(args, "planning_profile_dir", "planning_profile_logs") or "planning_profile_logs")
        ).expanduser()
    profile_dir.mkdir(parents=True, exist_ok=True)
    path = profile_dir / _profile_run_name(args)
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for idx in range(1, 1000):
        candidate = profile_dir / f"{stem}_{idx:03d}{suffix}"
        if not candidate.exists():
            return candidate
    return profile_dir / f"{stem}_{time.time_ns()}{suffix}"


def _profile_path(args) -> Path:
    return _profile_auto_path(args)


def _profile_args_snapshot(args) -> dict:
    raw = dict(vars(args)) if hasattr(args, "__dict__") else {}
    return {str(k): _jsonable(v) for k, v in sorted(raw.items(), key=lambda kv: str(kv[0]))}


def _write_profile_run_header(args) -> None:
    global _PROFILE_RUN_HEADER_WRITTEN
    if _PROFILE_RUN_HEADER_WRITTEN or _PROFILE_PATH is None:
        return
    record = {
        "ts": time.time(),
        "object_name": "__run__",
        "stage_name": "run_config",
        "success": True,
        "status": "STARTED",
        "argv": list(sys.argv),
        "cwd": str(Path.cwd()),
        "pid": os.getpid(),
        "profile_jsonl": str(_PROFILE_PATH),
        "args": _profile_args_snapshot(args),
    }
    with _PROFILE_PATH.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    _PROFILE_RECORDS.append(record)
    _PROFILE_RUN_HEADER_WRITTEN = True


def _ensure_profile(args) -> None:
    global _PROFILE_PATH, _PROFILE_REGISTERED
    if not _profile_enabled(args):
        return
    if _PROFILE_PATH is None:
        _PROFILE_PATH = _profile_path(args)
        if _PROFILE_PATH.parent != Path("."):
            _PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_profile_run_header(args)
    if not _PROFILE_REGISTERED:
        atexit.register(_print_profile_summary)
        _PROFILE_REGISTERED = True


@contextmanager
def _profile_record_context(**fields):
    prev = getattr(_PROFILE_THREAD_LOCAL, "record_context", None)
    merged = dict(prev or {})
    merged.update({str(k): v for k, v in fields.items() if v is not None})
    _PROFILE_THREAD_LOCAL.record_context = merged
    try:
        yield
    finally:
        if prev is None:
            try:
                delattr(_PROFILE_THREAD_LOCAL, "record_context")
            except AttributeError:
                pass
        else:
            _PROFILE_THREAD_LOCAL.record_context = prev


def _record_profile(args, stage_name: str, **fields) -> None:
    if not _profile_enabled(args):
        return
    _ensure_profile(args)
    record = {
        "ts": time.time(),
        "object_name": _profile_object_name(args),
        "stage_name": str(stage_name),
    }
    context_fields = getattr(_PROFILE_THREAD_LOCAL, "record_context", None)
    if isinstance(context_fields, dict):
        record.update({str(k): _jsonable(v) for k, v in context_fields.items() if v is not None})
    record.update({str(k): _jsonable(v) for k, v in fields.items() if v is not None})
    _PROFILE_RECORDS.append(record)
    if _PROFILE_PATH is not None:
        with _PROFILE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _record_prefetch_profile(base_args, object_name, stage_name: str, **fields) -> None:
    object_name = curobo_wrapper.normalize_object_name(object_name) or str(object_name or "prefetch")
    fields.setdefault("is_prefetch", True)
    profile_args = SimpleNamespace(
        object_name=object_name,
        planning_profile_enabled=bool(getattr(base_args, "planning_profile_enabled", True)),
        planning_profile_jsonl=getattr(base_args, "planning_profile_jsonl", None),
        planning_profile_dir=getattr(base_args, "planning_profile_dir", "planning_profile_logs"),
    )
    _record_profile(profile_args, stage_name, **fields)


@contextmanager
def _profile_stage(args, stage_name: str, **fields):
    start_t = time.perf_counter()
    counter_start = _snapshot_profile_counters()
    ik_batch_hist_start = _snapshot_profile_ik_batch_size_hist()
    rec = dict(fields)
    try:
        yield rec
        rec.setdefault("success", True)
    except Exception as exc:
        rec.setdefault("success", False)
        rec.setdefault("status", type(exc).__name__)
        raise
    finally:
        rec["elapsed_ms"] = round((time.perf_counter() - start_t) * 1000.0, 3)
        if not bool(getattr(_PROFILE_THREAD_LOCAL, "suspend_counters", False)):
            for key, value in _profile_counter_delta(counter_start).items():
                rec.setdefault(key, value)
            ik_batch_size_hist = _profile_ik_batch_size_hist_delta(ik_batch_hist_start)
            if ik_batch_size_hist:
                rec.setdefault("ik_batch_size_hist", ik_batch_size_hist)
        if "candidate_count" in rec and "candidate_count_in" not in rec:
            rec["candidate_count_in"] = rec.get("candidate_count")
        _record_profile(args, stage_name, **rec)


def _print_profile_summary() -> None:
    if not _PROFILE_RECORDS:
        return
    grouped: dict[tuple[str, str], list[dict]] = {}
    for rec in _PROFILE_RECORDS:
        if str(rec.get("stage_name", "")) == "run_config":
            continue
        grouped.setdefault((str(rec.get("object_name", "?")), str(rec.get("stage_name", "?"))), []).append(rec)
    rows = []
    for (obj, stage), items in grouped.items():
        elapsed = [float(x.get("elapsed_ms", 0.0) or 0.0) for x in items]
        success_count = sum(1 for x in items if bool(x.get("success", False)))
        total_ms = float(sum(elapsed))
        avg_ms = total_ms / max(len(elapsed), 1)
        max_ms = max(elapsed) if elapsed else 0.0
        rows.append((total_ms, obj, stage, len(items), success_count, avg_ms, max_ms))
    rows.sort(reverse=True)
    max_total = max((row[0] for row in rows), default=1.0)
    print("\n[planning_profile] summary")
    if _PROFILE_PATH is not None:
        print(f"[planning_profile] jsonl: {_PROFILE_PATH}")
    print("[planning_profile] object | stage | n | ok | total_ms | avg_ms | max_ms | bar")
    for total_ms, obj, stage, count, success_count, avg_ms, max_ms in rows:
        bar_len = int(round(32.0 * total_ms / max(max_total, 1e-6)))
        bar = "#" * max(1, bar_len)
        print(
            f"[planning_profile] {obj:12s} | {stage:28s} | {count:2d} | "
            f"{success_count:2d} | {total_ms:8.1f} | {avg_ms:7.1f} | {max_ms:7.1f} | {bar}"
        )


def _configure_curobo_torch_extensions(args) -> None:
    ext_dir = Path(
        getattr(
            args,
            "curobo_torch_extensions_dir",
            curobo_wrapper.DEFAULT_TORCH_EXTENSIONS_DIR,
        )
    ).expanduser().resolve()
    ext_dir.mkdir(parents=True, exist_ok=True)
    prev = os.environ.get("TORCH_EXTENSIONS_DIR")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(ext_dir)
    if prev is None:
        print(f"[curobo] set TORCH_EXTENSIONS_DIR={ext_dir}")
    elif Path(prev) != ext_dir:
        print(f"[curobo] override TORCH_EXTENSIONS_DIR={prev} -> {ext_dir}")


def build_arg_parser():
    parser = curobo_wrapper.build_arg_parser()
    parser.description = (
        "FoundationPose -> direct cuRobo grasp -> close gripper -> cuRobo transport-to-hover "
        "with contact-aware final placement."
    )
    parser.set_defaults(
        targeted_place_staging=False,
        curobo_plan_label_prefixes=[],
        curobo_ee_link="gripper_tcp",
        curobo_rm75_robot_cfg=Path(__file__).resolve().parent / "curobo_rm75_config" / "rm75.yml",
        carry_sim_arm_across_cycles=False,
        insert_vertical_axial_spin_deg=[
            0.0,
            -15.0,
            15.0,
            -30.0,
            30.0,
            -45.0,
            45.0,
            -60.0,
            60.0,
            -75.0,
            75.0,
            -90.0,
            90.0,
            -105.0,
            105.0,
            -120.0,
            120.0,
            -135.0,
            135.0,
            -150.0,
            150.0,
            -165.0,
            165.0,
            180.0,
        ],
        targeted_place_allow_insert_axis_flip=False,
        tabletop_place_yaw_variant_deg=[0.0, -45.0, 45.0, -90.0, 90.0, -135.0, 135.0, 180.0],
        carriot_tabletop_place_yaw_variant_deg=[0.0, -45.0, 45.0],
        tabletop_place_tilt_toward_robot_deg=[0.0],
        tabletop_place_axial_spin_deg=[0.0],
        targeted_place_expand_orientation_invariant=False,
        targeted_place_hover_extra_height_m=[0.0, 0.01],
        topdown_grasp_yaw_variant_deg=[0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0, 180.0],
        topdown_tilt_toward_robot_deg=[12.0, 20.0, 30.0, 45.0],
        topdown_tilt_toward_robot_shift_m=[0.0, 0.02],
    )
    parser.add_argument(
        "--direct-release-approach-distance",
        type=float,
        default=0.04,
        help="For insert_vertical rules, build a short pre-place this far from the release pose along the target long axis.",
    )
    parser.add_argument(
        "--planning-profile-jsonl",
        type=str,
        default=None,
        help=(
            "Write planning profile JSONL to this explicit file. "
            "If omitted, a timestamped file is created under --planning-profile-dir."
        ),
    )
    parser.add_argument(
        "--planning-profile-dir",
        type=str,
        default="planning_profile_logs",
        help="Directory for per-run timestamped planning profile JSONL files.",
    )
    parser.add_argument(
        "--planning-profile",
        dest="planning_profile_enabled",
        action="store_true",
        default=True,
        help="Enable planning stage profiling and final summary chart.",
    )
    parser.add_argument(
        "--no-planning-profile",
        dest="planning_profile_enabled",
        action="store_false",
        help="Disable planning stage profiling.",
    )
    parser.add_argument(
        "--random-cycle-targets",
        dest="target_selection_order",
        action="store_const",
        const="random",
        help=(
            "Choose each cycle target randomly from the remaining unplaced "
            "--cycle-object-names pool instead of using the risk-aware priority order."
        ),
    )
    parser.add_argument(
        "--risk-aware-cycle-targets",
        dest="target_selection_order",
        action="store_const",
        const="risk_aware",
        help="Use the current risk-aware cycle target priority order.",
    )
    parser.add_argument(
        "--cycle-target-random-seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible random cycle target selection.",
    )
    parser.add_argument(
        "--direct-release-approach-distances",
        type=float,
        nargs="*",
        default=[0.04, 0.05],
        help="For insert_vertical rules, also try these short pre-place approach distances along the target long axis.",
    )
    parser.add_argument(
        "--bi-insert-release-height-offsets-m",
        type=float,
        nargs="*",
        default=[0.0],
        help=(
            "For bi insert placement, try these extra release offsets along the holder opening axis. "
            "Default keeps the original target depth; use the fallback option to retry shallower releases only after failure."
        ),
    )
    parser.add_argument(
        "--bi-insert-release-fallback-height-offsets-m",
        type=float,
        nargs="*",
        default=[0.01],
        help=(
            "For bi insert placement, retry these shallower release offsets only if the original release depth "
            "does not produce a complete grasp->place chain."
        ),
    )
    parser.add_argument(
        "--bi-fast-insert-spin-deg",
        type=float,
        nargs="*",
        default=[0.0, -15.0, 15.0, -90.0, 90.0, -120.0, -135.0, -150.0],
        help=(
            "Small first-pass axial spin set for bi insert placement. The full "
            "--insert-vertical-axial-spin-deg set remains fallback if this lane fails."
        ),
    )
    parser.add_argument(
        "--bi-fast-insert-approach-distances",
        type=float,
        nargs="*",
        default=[0.05],
        help=(
            "Small first-pass approach distances for bi insert placement. "
            "Default 50mm matches the most reliable straight insertion contact."
        ),
    )
    parser.add_argument(
        "--bi-fast-insert-hover-extra-heights-m",
        type=float,
        nargs="*",
        default=[0.0, 0.03],
        help="Small first-pass hover-extra heights for bi insert placement.",
    )
    parser.add_argument(
        "--direct-pre-place-z-offsets",
        type=float,
        nargs="*",
        default=[0.0, 0.01],
        help="Also lift the short pre-place by these extra world-z offsets. This helps avoid under-table or desk-edge sweeps without changing the final release pose.",
    )
    parser.add_argument(
        "--direct-grasp-object-axis-shifts-m",
        type=float,
        nargs="*",
        default=[0.0, -0.012, 0.012, -0.024, 0.024],
        help="Also try shifting the grasp TCP along the source object's longest axis by these distances to search for grasp placements that yield a better downstream place chain.",
    )
    parser.add_argument(
        "--direct-elongated-grasp-tilt-toward-robot-deg",
        type=float,
        nargs="*",
        default=[0.0, 12.0, 20.0],
        help="For elongated-object grasp modes, also try these tilt-toward-robot angles when searching for grasp branches that better support downstream place.",
    )
    parser.add_argument(
        "--direct-elongated-grasp-tilt-toward-robot-shift-m",
        type=float,
        nargs="*",
        default=[0.0, 0.01, 0.02],
        help="For elongated-object tilt grasp variants, also try these small XY shifts toward the robot.",
    )
    parser.add_argument(
        "--direct-grasp-z-lifts-m",
        type=float,
        nargs="*",
        default=[0.0, 0.01, 0.02],
        help="Also try lifting grasp TCP in world-z by these offsets to improve IK reachability screening.",
    )
    parser.add_argument(
        "--sphere-grasp-yaw-variant-deg",
        type=float,
        nargs="*",
        default=[0.0, -45.0, 45.0, -90.0, 90.0, 180.0],
        help="For spherical objects, keep the same center grasp but try these equivalent wrist yaw angles.",
    )
    parser.add_argument(
        "--bi-direct-grasp-approach-roll-degs",
        type=float,
        nargs="*",
        default=[0.0, 180.0],
        help=(
            "For the pen, try equivalent TCP approach-axis rolls. With the long-axis aligned "
            "grasp frame only 0/180 preserve the gripper opening direction perpendicular to the pen."
        ),
    )
    parser.add_argument(
        "--shuazi-direct-grasp-approach-roll-degs",
        type=float,
        nargs="*",
        default=[0.0, 90.0, 180.0, 270.0],
        help=(
            "For shuazi, try equivalent TCP approach-axis rolls. This changes the gripper/object relation "
            "without changing the fixed final object pose."
        ),
    )
    parser.add_argument(
        "--direct-grasp-diagnose-topk",
        type=int,
        default=0,
        help="When direct_grasp IK prefilter keeps zero candidates, run detailed diagnose_pose_goal for top-K closest IK candidates.",
    )
    parser.add_argument(
        "--direct-grasp-axis-shift-penalty-per-cm",
        type=float,
        default=0.10,
        help="Additional score penalty per 1cm absolute grasp-axis shift. Larger values prefer center-aligned grasps.",
    )
    parser.add_argument(
        "--direct-grasp-z-lift-penalty-per-cm",
        type=float,
        default=0.08,
        help="Additional score penalty per 1cm grasp TCP lift. Larger values prefer lower-lift grasps.",
    )
    parser.add_argument(
        "--direct-grasp-max-axis-shift-ratio",
        type=float,
        default=0.35,
        help="Max |axis_shift| as a fraction of the object's half-length along its longest axis. "
        "E.g. 0.35 on a 60mm object = max 10.5mm shift. Prevents grasp candidates that would "
        "overshoot the object edge.",
    )
    parser.add_argument(
        "--direct-min-place-tcp-z",
        type=float,
        default=0.005,
        help="Reject any direct pre-place/place candidate whose TCP world z falls below this threshold.",
    )
    parser.add_argument(
        "--direct-place-max-pad-tilt",
        type=float,
        default=0.25,
        help="Max |tcp_y_world_z| for place candidates — how much the pad opening axis can deviate "
        "from horizontal. 0.0 = pads must be perfectly level; 0.25 ≈ max ~14° tilt "
        "(~12mm height difference for 50mm pad span). Rejects poses where one pad is much "
        "lower than the other, causing unstable release.",
    )
    parser.add_argument(
        "--direct-place-max-abs-yaw-deg",
        type=float,
        default=0.0,
        help="Optional absolute-yaw filter for direct place candidates. "
        "Set <= 0 to disable and keep the full tabletop yaw variant set.",
    )
    parser.add_argument(
        "--direct-pre-place-verticality-band",
        type=float,
        default=0.08,
        help="Among successful insert_vertical pre-place candidates, keep only those within this tcp_verticality band from the best vertical candidate before selecting the lowest-cost cuRobo path.",
    )
    parser.add_argument(
        "--place-orientation-preference-weight",
        type=float,
        default=0.25,
        help="Additional penalty weight used to prefer place candidates whose object facing direction is oriented toward the robot. Set <=0 to disable this preference.",
    )
    parser.add_argument(
        "--direct-terminal-max-realized-pos-error",
        type=float,
        default=0.010,
        help="Reject a cuRobo candidate if the realized sim TCP position error after terminal snap exceeds this threshold.",
    )
    parser.add_argument(
        "--direct-terminal-max-realized-rot-error-deg",
        type=float,
        default=12.0,
        help="Reject a cuRobo candidate if the realized sim TCP orientation error after terminal snap exceeds this threshold.",
    )
    parser.add_argument(
        "--direct-terminal-snap-max-joint-delta",
        type=float,
        default=0.10,
        help="Maximum per-joint interpolation step for the short collision-checked terminal snap from the cuRobo endpoint to the sim terminal IK.",
    )
    parser.add_argument(
        "--direct-release-cartesian-step-m",
        type=float,
        default=0.005,
        help="Cartesian step size for the pure-cuRobo short constrained release descent.",
    )
    parser.add_argument(
        "--joint-search-max-grasp-candidates",
        type=int,
        default=2,
        help="During joint grasp->place search, only expand this many top-ranked grasp candidates with downstream place feasibility checks. Set <=0 to search all grasp candidates.",
    )
    parser.add_argument(
        "--joint-search-max-pre-place-candidates",
        type=int,
        default=0,
        help="During joint search, only expand this many heuristic-ranked pre-place candidates per grasp before running cuRobo. Set <=0 to search all pre-place candidates.",
    )
    parser.add_argument(
        "--joint-search-fallback-max-pre-place-candidates",
        type=int,
        default=0,
        help="If the first diverse pre-place/hover subset fails for a grasp, expand to this many candidates for a generic fallback pass. Set <=0 to disable this fallback expansion.",
    )
    parser.add_argument(
        "--joint-search-fallback-enable-graph",
        dest="joint_search_fallback_enable_graph",
        action="store_true",
        default=False,
        help="Enable cuRobo graph planning during the generic joint-search fallback pass after trajopt-only transport screening fails.",
    )
    parser.add_argument(
        "--no-joint-search-fallback-enable-graph",
        dest="joint_search_fallback_enable_graph",
        action="store_false",
        help="Disable graph planning during the generic joint-search fallback pass.",
    )
    parser.add_argument(
        "--joint-search-fallback-max-attempts",
        type=int,
        default=4,
        help="Max MotionGen attempts for the generic joint-search fallback pass.",
    )
    parser.add_argument(
        "--joint-search-fallback-num-graph-seeds",
        type=int,
        default=4,
        help="Graph seeds for the generic joint-search fallback pass.",
    )
    parser.add_argument(
        "--joint-search-fallback-num-ik-seeds",
        type=int,
        default=128,
        help="IK seeds for the generic joint-search fallback pass.",
    )
    parser.add_argument(
        "--joint-search-fallback-num-trajopt-seeds",
        type=int,
        default=4,
        help="Trajectory optimization seeds for the generic joint-search fallback pass.",
    )
    parser.add_argument(
        "--joint-search-fallback-timeout",
        type=float,
        default=8.0,
        help="Timeout in seconds for the generic joint-search fallback pass.",
    )
    parser.add_argument(
        "--transport-use-prefilter-q-goal",
        dest="transport_use_prefilter_q_goal",
        action="store_true",
        default=True,
        help="Before pose MotionGen in transport-hover, try the already-screened IK q_hover/q_goal with joint-space MotionGen.",
    )
    parser.add_argument(
        "--no-transport-use-prefilter-q-goal",
        dest="transport_use_prefilter_q_goal",
        action="store_false",
        help="Disable the transport-hover joint-space trial that reuses the IK prefilter q goal.",
    )
    parser.add_argument(
        "--transport-prefilter-q-goal-max-trials",
        type=int,
        default=1,
        help="Maximum number of IK-prefiltered q goals to try with joint-space MotionGen before falling back to pose MotionGen.",
    )
    parser.add_argument(
        "--transport-prefilter-q-goal-timeout",
        type=float,
        default=2.0,
        help="Timeout in seconds for each transport-hover joint-space q_goal trial.",
    )
    parser.add_argument(
        "--transport-prefilter-q-goal-max-attempts",
        type=int,
        default=1,
        help="MotionGen attempts for each transport-hover joint-space q_goal trial.",
    )
    parser.add_argument(
        "--transport-prefilter-q-goal-num-trajopt-seeds",
        type=int,
        default=1,
        help="Trajectory optimization seeds for each transport-hover joint-space q_goal trial.",
    )
    parser.add_argument(
        "--fast-chain-screening",
        dest="fast_chain_screening",
        action="store_true",
        default=True,
        help=(
            "Use the integrated fast path: batched IK ranks grasp->place pairs, reuses "
            "the IK results in transport prefilter, and tries the top pair before the "
            "full fallback search."
        ),
    )
    parser.add_argument(
        "--no-fast-chain-screening",
        dest="fast_chain_screening",
        action="store_false",
        help="Disable the integrated fast-chain IK ranking path.",
    )
    parser.add_argument(
        "--fast-chain-top-pairs",
        type=int,
        default=3,
        help=(
            "Number of IK-ranked grasp->place pairs allowed to reach the transport stage. "
            "Each pair still uses only the middle transport MotionGen; short primitives stay IK/constrained-line."
        ),
    )
    parser.add_argument(
        "--joint-search-primary-fallback-after-fast-ik-fail",
        dest="joint_search_primary_fallback_after_fast_ik_fail",
        action="store_true",
        default=True,
        help=(
            "After the IK-ranked fast transport candidates fail, run the broader primary transport "
            "candidate pass for the same grasp. This preserves fast success while improving random-order robustness."
        ),
    )
    parser.add_argument(
        "--no-joint-search-primary-fallback-after-fast-ik-fail",
        dest="joint_search_primary_fallback_after_fast_ik_fail",
        action="store_false",
        help="Keep the old fast-chain gate behavior and skip primary transport fallback after IK-ranked candidates.",
    )
    parser.add_argument(
        "--fast-chain-initial-grasp-winners",
        type=int,
        default=1,
        help=(
            "Legacy compatibility knob. In pair-first mode the IK-preselected top pair "
            "is used directly and candidate-stage pregrasp MotionGen is skipped."
        ),
    )
    parser.add_argument(
        "--fast-chain-preselect-grasp-candidates",
        type=int,
        default=6,
        help=(
            "Maximum grasp candidates to check in the cheap winner-chain IK preselect. "
            "Set <=0 to check every generated grasp candidate."
        ),
    )
    parser.add_argument(
        "--fast-chain-preselect-first-valid",
        dest="fast_chain_preselect_first_valid",
        action="store_true",
        default=False,
        help="Stop cheap winner-chain preselect at the first ordered grasp with a valid place IK pair.",
    )
    parser.add_argument(
        "--no-fast-chain-preselect-first-valid",
        dest="fast_chain_preselect_first_valid",
        action="store_false",
        help="Rank all cheap winner-chain preselect pairs instead of stopping at the first valid pair.",
    )
    parser.add_argument(
        "--fast-chain-max-ik-candidates",
        type=int,
        default=0,
        help=(
            "Maximum place candidates to IK-rank in the fast-chain pass. "
            "Set <=0 to rank every generated place candidate."
        ),
    )
    parser.add_argument(
        "--fast-chain-place-rank-grasp-limit",
        type=int,
        default=3,
        help=(
            "Maximum cheap-IK grasp candidates allowed to build/rank place candidates in winner-chain preselect. "
            "Set <=0 to rank place candidates for every grasp with valid pregrasp/grasp IK."
        ),
    )
    parser.add_argument(
        "--fast-chain-place-ik-chunk-candidates",
        type=int,
        default=8,
        help=(
            "Number of place candidates to IK-rank per fast-chain batch. "
            "Each candidate contributes hover+release goals, so the default 8 maps to a fixed CUDA graph IK batch of 16."
        ),
    )
    parser.add_argument(
        "--fast-chain-place-ik-early-stop",
        dest="fast_chain_place_ik_early_stop",
        action="store_true",
        default=True,
        help="Stop fast-chain place IK ranking once the current grasp has enough valid ranked place candidates.",
    )
    parser.add_argument(
        "--no-fast-chain-place-ik-early-stop",
        dest="fast_chain_place_ik_early_stop",
        action="store_false",
        help="Rank all fast-chain place IK candidates even after enough valid candidates are found.",
    )
    parser.add_argument(
        "--fast-chain-ik-seeds",
        type=int,
        default=32,
        help="IK seeds used by the fast-chain ranking pass.",
    )
    parser.add_argument(
        "--fast-chain-cuda-graph-ik",
        dest="fast_chain_cuda_graph_ik",
        action="store_true",
        default=True,
        help=(
            "Use fixed-size cuRobo CUDA graph batch IK for fast-chain IK screening. "
            "Enabled by default; use --no-fast-chain-cuda-graph-ik to fall back to eager cuRobo batch IK."
        ),
    )
    parser.add_argument(
        "--no-fast-chain-cuda-graph-ik",
        dest="fast_chain_cuda_graph_ik",
        action="store_false",
        help="Disable CUDA graph batch IK for fast-chain screening and use eager cuRobo batch IK.",
    )
    parser.add_argument(
        "--fast-chain-cuda-graph-ik-max-batch-size",
        type=int,
        default=128,
        help="Maximum fixed CUDA graph IK batch bucket used by fast-chain screening.",
    )
    parser.add_argument(
        "--fast-chain-cuda-graph-ik-fixed-batch-size",
        type=int,
        default=16,
        help=(
            "When --fast-chain-cuda-graph-ik is enabled, pad every fast-chain IK request to this single fixed batch size. "
            "Set <=0 to use power-of-two buckets instead."
        ),
    )
    parser.add_argument(
        "--next-cycle-plan-prefetch",
        dest="next_cycle_plan_prefetch",
        action="store_true",
        default=True,
        help=(
            "Speculatively plan the next cycle in a background demo/planner once the current "
            "cycle's place target is known. The cached plan is reused only if the next cycle "
            "selects the same object and the start state/scene pose still match."
        ),
    )
    parser.add_argument(
        "--no-next-cycle-plan-prefetch",
        dest="next_cycle_plan_prefetch",
        action="store_false",
        help="Disable background next-cycle planning prefetch.",
    )
    parser.add_argument(
        "--next-cycle-prefetch-start-q-tolerance",
        type=float,
        default=0.08,
        help="Maximum per-joint start-q mismatch allowed when reusing a prefetched next-cycle plan.",
    )
    parser.add_argument(
        "--next-cycle-prefetch-scene-pos-tolerance-m",
        type=float,
        default=0.05,
        help="Maximum placed-object translation mismatch allowed when reusing a prefetched next-cycle plan.",
    )
    parser.add_argument(
        "--next-cycle-prefetch-wait-timeout",
        type=float,
        default=30.0,
        help=(
            "Maximum seconds to wait at the next cycle for a reserved background plan to finish "
            "before falling back to live planning. The default waits for the already-running "
            "background plan instead of duplicating the same target in the foreground."
        ),
    )
    parser.add_argument(
        "--dry-run-motion-window-scale",
        type=float,
        default=0.0,
        help=(
            "In dry-run mode, sleep after each simulated path execution for this fraction of the "
            "estimated real waypoint-stream duration. Use 1.0 to preserve a realistic execution "
            "window so background next-cycle prefetch can overlap with simulated motion."
        ),
    )
    parser.add_argument(
        "--dry-run-motion-window-max-s",
        type=float,
        default=30.0,
        help="Maximum dry-run motion-window sleep per executed path. Set <=0 for no cap.",
    )
    parser.add_argument(
        "--dry-run-motion-window-min-s",
        type=float,
        default=0.0,
        help="Minimum dry-run motion-window sleep for non-empty executed paths when the scale is enabled.",
    )
    parser.add_argument(
        "--joint-search-start-collision-lift-m",
        type=float,
        default=0.080,
        help="If joint-search transport starts in world collision, try a straight world-Z lift by this distance before replanning transport.",
    )
    parser.add_argument(
        "--skip-joint-search-object-names",
        type=str,
        nargs="*",
        default=[],
        help="Object names that should skip conservative joint grasp->place pre-screening and plan place only after the real/sim grasp and lift.",
    )
    parser.add_argument(
        "--joint-search-max-feasible-chains",
        type=int,
        default=1,
        help="Stop joint search after collecting this many feasible grasp->pre_place->release chains. Set <=0 to keep searching all candidates.",
    )
    parser.add_argument(
        "--reuse-joint-search-chain",
        dest="reuse_joint_search_chain",
        action="store_true",
        default=True,
        help="After executing the selected grasp, reuse the transport path found during joint-search when the executed start q still matches.",
    )
    parser.add_argument(
        "--no-reuse-joint-search-chain",
        dest="reuse_joint_search_chain",
        action="store_false",
        help="Disable reuse of the joint-search transport path and always replan transport from the executed state.",
    )
    parser.add_argument(
        "--joint-search-reuse-start-q-tolerance",
        type=float,
        default=0.03,
        help="Max per-joint absolute delta allowed when reusing the selected joint-search transport path.",
    )
    parser.add_argument(
        "--pair-first-reuse-start-q-tolerance",
        type=float,
        default=0.12,
        help=(
            "Max per-joint delta allowed when reusing a pair-first transport path after the selected "
            "straight final approach lands on a different IK branch than the cheap q_grasp."
        ),
    )
    parser.add_argument(
        "--joint-search-screen-timeout",
        type=float,
        default=0.0,
        help="Use this shorter cuRobo timeout during joint-search screening before the final selected chain is executed. Set <=0 to reuse the normal cuRobo timeout.",
    )
    parser.add_argument(
        "--joint-search-screen-max-attempts",
        type=int,
        default=0,
        help="Use this smaller max_attempts during joint-search screening. Set <=0 to reuse the normal cuRobo value.",
    )
    parser.add_argument(
        "--joint-search-screen-num-ik-seeds",
        type=int,
        default=0,
        help="Use this smaller IK seed count during joint-search screening. Set <=0 to reuse the normal cuRobo value.",
    )
    parser.add_argument(
        "--joint-search-screen-num-trajopt-seeds",
        type=int,
        default=0,
        help="Use this smaller trajopt seed count during joint-search screening. Set <=0 to reuse the normal cuRobo value.",
    )
    parser.add_argument(
        "--joint-search-goalset-max-winners",
        type=int,
        default=1,
        help="When using cuRobo goalset screening for pre-place during joint search, keep extracting at most this many successful winners before moving on to release checks. Set <=0 to exhaust the goalset.",
    )
    parser.add_argument(
        "--direct-grasp-goalset-max-winners",
        type=int,
        default=1,
        help="When screening grasp candidates with cuRobo goalset, keep extracting at most this many successful winners. Set <=0 to exhaust the goalset.",
    )
    parser.add_argument(
        "--curobo-batch-chunk-size",
        type=int,
        default=64,
        help="Evaluate multi-candidate cuRobo batches in chunks of this size to reduce peak VRAM use. Set <=0 to run all candidates in one batch.",
    )
    parser.add_argument(
        "--curobo-table-collision",
        dest="curobo_table_collision",
        action="store_true",
        default=True,
        help="Inject a virtual table cuboid at z=0 into cuRobo world for ground-plane collision avoidance.",
    )
    parser.add_argument(
        "--no-curobo-table-collision",
        dest="curobo_table_collision",
        action="store_false",
        help="Disable the virtual table cuboid in cuRobo world.",
    )
    parser.add_argument(
        "--curobo-table-z-offset",
        type=float,
        default=-0.01,
        help="Z offset of the virtual table cuboid top surface relative to z=0. "
        "Negative values lower the table below z=0, giving clearance for gripper approach. "
        "Default -0.01 means the table top is at z=-1cm.",
    )
    parser.add_argument(
        "--curobo-table-thickness",
        type=float,
        default=0.02,
        help="Thickness of the virtual table cuboid.",
    )
    parser.add_argument(
        "--curobo-table-size-x",
        type=float,
        default=1.2,
        help="X extent of the virtual table cuboid.",
    )
    parser.add_argument(
        "--curobo-table-size-y",
        type=float,
        default=1.2,
        help="Y extent of the virtual table cuboid.",
    )
    parser.add_argument(
        "--curobo-attach-object",
        dest="curobo_attach_object",
        action="store_true",
        default=True,
        help="After grasping, attach the target object to the robot in cuRobo so its collision spheres participate in transport planning.",
    )
    parser.add_argument(
        "--no-curobo-attach-object",
        dest="curobo_attach_object",
        action="store_false",
        help="Disable cuRobo attached object collision during transport.",
    )
    parser.add_argument(
        "--curobo-attach-world-z-offset-m",
        type=float,
        default=0.002,
        help="When attaching the grasped object into cuRobo, first apply this small world-frame +Z offset. "
        "Useful for objects that are still lightly touching the table at the instant of attach.",
    )
    parser.add_argument(
        "--curobo-attach-min-start-clearance-m",
        type=float,
        default=0.003,
        help="Minimum cuRobo attached-sphere clearance above the table immediately after attach. "
        "If the conservative payload model starts below this, the attach z-offset is increased up to the auto limit.",
    )
    parser.add_argument(
        "--curobo-attach-max-auto-z-offset-m",
        type=float,
        default=0.030,
        help="Maximum automatic world-z offset applied to the attached payload collision model to avoid invalid "
        "start-state table penetration.",
    )
    parser.add_argument(
        "--post-grasp-lift",
        dest="post_grasp_lift_enabled",
        action="store_true",
        default=False,
        help="Enable the post-grasp safety lift before transport.",
    )
    parser.add_argument(
        "--post-grasp-lift-height-m",
        type=float,
        default=0.080,
        help="After attaching the grasped payload, lift the TCP by this world-z distance before transport planning.",
    )
    parser.add_argument(
        "--post-grasp-lift-min-tcp-z-m",
        type=float,
        default=0.100,
        help="Minimum TCP world-z target after post-grasp lift.",
    )
    parser.add_argument(
        "--return-start-clearance-lift-m",
        type=float,
        default=0.060,
        help=(
            "Fallback/fixed world-Z lift distance before returning to the cycle-start joint state after place. "
            "By default the return prelift uses the placed object's world-Z half-height plus "
            "--return-start-clearance-lift-extra-m; disable that with "
            "--no-return-start-clearance-lift-from-placed-object."
        ),
    )
    parser.add_argument(
        "--return-start-clearance-lift-from-placed-object",
        dest="return_start_clearance_lift_from_placed_object",
        action="store_true",
        default=True,
        help="Set return prelift distance from the newly placed object's oriented world-Z half-height plus a small margin.",
    )
    parser.add_argument(
        "--no-return-start-clearance-lift-from-placed-object",
        dest="return_start_clearance_lift_from_placed_object",
        action="store_false",
        help="Use --return-start-clearance-lift-m as a fixed return prelift distance.",
    )
    parser.add_argument(
        "--return-start-clearance-lift-extra-m",
        type=float,
        default=0.015,
        help="Extra clearance added to the placed object's world-Z half-height for dynamic return prelift.",
    )
    parser.add_argument(
        "--return-start-clearance-lift-min-m",
        type=float,
        default=0.030,
        help="Minimum dynamic return prelift distance when using the placed object's world-Z half-height.",
    )
    parser.add_argument(
        "--return-start-clearance-lift-max-m",
        type=float,
        default=0.090,
        help="Maximum dynamic return prelift distance when using the placed object's world-Z half-height.",
    )
    parser.add_argument(
        "--return-to-start-preplan",
        dest="return_to_start_preplan",
        action="store_true",
        default=True,
        help=(
            "Preplan the empty-gripper return_to_start joint path in a background thread once "
            "post-place clearance is known."
        ),
    )
    parser.add_argument(
        "--no-return-to-start-preplan",
        dest="return_to_start_preplan",
        action="store_false",
        help="Disable background preplanning for the empty-gripper return_to_start path.",
    )
    parser.add_argument(
        "--return-to-start-preplan-wait-timeout",
        type=float,
        default=30.0,
        help="Maximum seconds to wait for an already-started return_to_start preplan before falling back to live planning.",
    )
    parser.add_argument(
        "--return-to-start-preplan-start-q-tolerance",
        type=float,
        default=0.05,
        help="Maximum per-joint mismatch allowed when reusing a preplanned return_to_start path.",
    )
    parser.add_argument(
        "--return-to-start-preplan-prelift",
        dest="return_to_start_preplan_prelift",
        action="store_true",
        default=True,
        help="If direct return_to_start preplanning fails, preplan a short prelift then return from the lifted joint state.",
    )
    parser.add_argument(
        "--no-return-to-start-preplan-prelift",
        dest="return_to_start_preplan_prelift",
        action="store_false",
        help="Disable the prelift rescue branch inside return_to_start preplanning.",
    )
    parser.add_argument(
        "--return-to-start-preplan-prelift-first",
        dest="return_to_start_preplan_prelift_first",
        action="store_true",
        default=True,
        help=(
            "Start return_to_start preplanning from a fixed world-Z prelift pose instead of first trying "
            "direct return from the release/clearance pose. This avoids repeated start-state collision "
            "failures against the newly placed object."
        ),
    )
    parser.add_argument(
        "--no-return-to-start-preplan-prelift-first",
        dest="return_to_start_preplan_prelift_first",
        action="store_false",
        help="Try direct return_to_start preplanning before the prelift rescue branch.",
    )
    parser.add_argument(
        "--return-to-start-self-collision-audit",
        dest="return_to_start_self_collision_audit",
        action="store_true",
        default=True,
        help="Audit every return_to_start path for cuRobo self-collision clearance before execution.",
    )
    parser.add_argument(
        "--no-return-to-start-self-collision-audit",
        dest="return_to_start_self_collision_audit",
        action="store_false",
        help="Disable return_to_start self-collision clearance audit.",
    )
    parser.add_argument(
        "--return-to-start-self-collision-warning-clearance-m",
        type=float,
        default=0.003,
        help="Warn in the return self-collision audit when minimum clearance drops below this distance.",
    )
    parser.add_argument(
        "--return-to-start-self-collision-audit-stride",
        type=int,
        default=1,
        help="Check every Nth waypoint in return_to_start self-collision audit. The first and last waypoints are always checked.",
    )
    parser.add_argument(
        "--no-post-grasp-lift",
        dest="post_grasp_lift_enabled",
        action="store_false",
        help="Disable the post-grasp safety lift before transport.",
    )
    parser.add_argument(
        "--hongshupian-attached-long-axis-scale",
        type=float,
        default=1.12,
        help="Object-specific scale for hongshupian attached collision length.",
    )
    parser.add_argument(
        "--hongshupian-attached-short-axis-scale",
        type=float,
        default=1.00,
        help="Object-specific scale for hongshupian attached collision short axes before sphere fitting.",
    )
    parser.add_argument(
        "--hongshupian-attached-sphere-radius-scale",
        type=float,
        default=0.48,
        help="Object-specific radius scale for hongshupian long-axis attached spheres.",
    )
    parser.add_argument(
        "--hongshupian-attached-sphere-count",
        type=int,
        default=6,
        help="Number of long-axis attached spheres for hongshupian.",
    )
    parser.add_argument(
        "--bi-attached-short-axis-scale",
        type=float,
        default=0.75,
        help="Object-specific scale for bi attached collision short axes before sphere fitting.",
    )
    parser.add_argument(
        "--bi-attached-long-axis-scale",
        type=float,
        default=0.95,
        help="Object-specific scale for bi attached collision length before sphere fitting.",
    )
    parser.add_argument(
        "--bi-attached-sphere-radius-scale",
        type=float,
        default=0.35,
        help="Object-specific radius scale for bi attached collision long-axis spheres.",
    )
    parser.add_argument(
        "--bi-attached-sphere-count",
        type=int,
        default=5,
        help="Number of long-axis attached spheres for bi.",
    )
    parser.add_argument(
        "--tennis-direct-place-release-lift-m",
        type=float,
        default=0.025,
        help="Deprecated alias for --sphere-place-release-lift-m.",
    )
    parser.add_argument(
        "--sphere-place-release-lift-m",
        type=float,
        default=0.030,
        help="World-z release lift for spherical/orientation-invariant objects; object orientation is ignored.",
    )
    parser.add_argument(
        "--tennis-direct-place-tilt-toward-robot-deg",
        type=float,
        default=None,
        help="Deprecated single-angle alias for tennis release tilt toward robot.",
    )
    parser.add_argument(
        "--tennis-direct-place-tilt-toward-robot-degs",
        type=float,
        nargs="*",
        default=[0.0, 15.0, -15.0, 30.0, -30.0],
        help="Multi-level tennis release TCP-body tilt candidates in degrees. "
        "Positive values shift/lean the TCP body toward the robot, negative values shift it away.",
    )
    parser.add_argument(
        "--tennis-direct-place-axial-roll-degs",
        type=float,
        nargs="*",
        default=[0.0, -45.0, 45.0, -90.0, 90.0, 180.0],
        help="Extra wrist-roll candidates around the tennis release approach axis. "
        "This does not change the target ball center, but gives cuRobo more IK/collision branches.",
    )
    parser.add_argument(
        "--direct-place-contact-lift-m",
        type=float,
        default=0.0,
        help="Legacy alias for tabletop release lift. Default 0 keeps the final object height fixed.",
    )
    parser.add_argument(
        "--place-mode",
        choices=["auto", "drop_place", "surface_place", "vertical_place", "insert_place"],
        default="auto",
        help="Override automatic place primitive classification.",
    )
    parser.add_argument(
        "--drop-place-release-lift-m",
        type=float,
        default=0.030,
        help="World-z lift used as the safe release height for drop_place primitives.",
    )
    parser.add_argument(
        "--surface-place-hover-height-m",
        type=float,
        default=0.020,
        help="World-z hover height above the final release pose for surface_place primitives.",
    )
    parser.add_argument(
        "--vertical-place-hover-height-m",
        type=float,
        default=0.040,
        help="World-z hover height above the final release pose for vertical_place primitives.",
    )
    parser.add_argument(
        "--transport-hover-extra-heights-m",
        type=float,
        nargs="*",
        default=[0.0, 0.03, 0.06],
        help="Additional world-z lifts applied to non-drop transport hover candidates.",
    )
    parser.add_argument(
        "--final-contact-clearance-m",
        type=float,
        default=0.0,
        help="Small world-z clearance added to tabletop contact release poses. Default 0 keeps the final object height fixed.",
    )
    parser.add_argument(
        "--place-transport-max-winners",
        type=int,
        default=2,
        help="Keep this many successful transport-to-hover candidates before trying final contact approach.",
    )
    parser.add_argument(
        "--joint-search-max-final-contact-checks",
        type=int,
        default=4,
        help=(
            "Maximum sequential final-contact MotionGen validations per joint-search "
            "transport batch when --joint-search-validate-final-contact is enabled. "
            "Set <=0 to validate every transport winner."
        ),
    )
    parser.add_argument(
        "--joint-search-validate-final-contact",
        action="store_true",
        default=False,
        help=(
            "Validate hover->release final-contact inside joint-search before accepting "
            "a transport winner. Default is off: final contact is planned once during "
            "execution of the selected chain."
        ),
    )
    parser.add_argument(
        "--no-joint-search-validate-final-contact",
        dest="joint_search_validate_final_contact",
        action="store_false",
        help="Disable joint-search final-contact prevalidation.",
    )
    parser.add_argument(
        "--insert-joint-search-max-final-contact-checks",
        type=int,
        default=2,
        help=(
            "Maximum sequential final-contact validations for insert_place batches "
            "such as bi->bitong. Set <=0 to use --joint-search-max-final-contact-checks."
        ),
    )
    parser.add_argument(
        "--skip-return-to-cycle-start",
        action="store_true",
        default=False,
        help="After a successful place, do not plan the empty-gripper return-to-start segment. Useful for headless batch regression speed tests.",
    )
    parser.add_argument(
        "--strict-return-to-cycle-start",
        action="store_true",
        default=False,
        help=(
            "Treat empty-gripper return_to_cycle_start failure after a completed place as fatal. "
            "By default the object placement remains successful and the next cycle starts from the current arm pose."
        ),
    )
    parser.add_argument(
        "--skip-post-place-clearance",
        action="store_true",
        default=False,
        help="After opening the gripper, skip the empty-gripper post-place clearance plan. Useful for headless batch regression speed tests.",
    )
    parser.add_argument(
        "--final-contact-segmented-ik-fallback",
        dest="final_contact_segmented_ik_fallback",
        action="store_true",
        default=False,
        help="If constrained final contact approach fails, allow short segmented IK fallback for the last few centimeters.",
    )
    parser.add_argument(
        "--final-contact-segmented-ik-first",
        dest="final_contact_segmented_ik_first",
        action="store_true",
        default=False,
        help="For short final-contact descents, try segmented IK before MotionGen and fall back to MotionGen if needed.",
    )
    parser.add_argument(
        "--no-final-contact-segmented-ik-first",
        dest="final_contact_segmented_ik_first",
        action="store_false",
        help="Use MotionGen first for final-contact descents.",
    )
    parser.add_argument(
        "--strict-final-contact-linear",
        dest="strict_final_contact_linear",
        action="store_true",
        default=True,
        help=(
            "Require hover->release final contact to use cuRobo PoseCostMetric constrained approach "
            "and reject returned paths whose realized TCP waypoints deviate too much from the straight line."
        ),
    )
    parser.add_argument(
        "--no-strict-final-contact-linear",
        dest="strict_final_contact_linear",
        action="store_false",
        help="Allow final contact paths without straight-line waypoint validation.",
    )
    parser.add_argument(
        "--strict-final-contact-waypoint-pos-tol-m",
        type=float,
        default=0.012,
        help="Maximum realized TCP distance from the commanded straight line for short constrained segments.",
    )
    parser.add_argument(
        "--strict-final-contact-waypoint-backtrack-tol-m",
        type=float,
        default=0.008,
        help=(
            "Maximum extra movement opposite the commanded hover->release line before descending. "
            "This allows small same-line cuRobo pre-lifts without allowing lateral drift."
        ),
    )
    parser.add_argument(
        "--strict-short-linear-waypoint-pos-tol-m",
        type=float,
        default=0.015,
        help=(
            "Maximum realized TCP line error for non-contact short constrained segments "
            "(grasp approach, post-grasp lift, post-place retreat). Final contact keeps "
            "--strict-final-contact-waypoint-pos-tol-m."
        ),
    )
    parser.add_argument(
        "--short-linear-joint-step-rad",
        type=float,
        default=0.035,
        help="Max per-joint interpolation step for IK-endpoint short straight primitives.",
    )
    parser.add_argument(
        "--short-linear-ik-seeds",
        type=int,
        default=64,
        help="IK seeds for endpoint IK used by short straight primitives.",
    )
    parser.add_argument(
        "--short-linear-endpoint-ik-first",
        dest="short_linear_endpoint_ik_first",
        action="store_true",
        default=True,
        help="Try endpoint IK + validated joint interpolation before cuRobo PoseCostMetric for short straight primitives.",
    )
    parser.add_argument(
        "--no-short-linear-endpoint-ik-first",
        dest="short_linear_endpoint_ik_first",
        action="store_false",
        help="Disable endpoint IK fast path for short straight primitives.",
    )
    parser.add_argument(
        "--strict-final-contact-waypoint-rot-tol-deg",
        type=float,
        default=8.0,
        help="Maximum realized TCP rotation error for short constrained-segment waypoints.",
    )
    parser.add_argument(
        "--final-contact-approach-metric-tstep-fraction",
        type=float,
        default=0.0,
        help=(
            "cuRobo PoseCostMetric activation fraction for hover->release final contact. "
            "Default constrains the whole short contact segment so release descents stay straight."
        ),
    )
    parser.add_argument(
        "--curobo-approach-metric-tstep-fraction",
        type=float,
        default=0.0,
        help=(
            "cuRobo PoseCostMetric activation fraction for non-contact short straight segments "
            "(grasp approach, post-grasp lift, post-place retreat). Default 0 constrains the whole segment."
        ),
    )
    parser.add_argument(
        "--curobo-approach-metric-locked-axis-tol-m",
        type=float,
        default=0.012,
        help=(
            "Maximum non-free-axis translation allowed when constructing cuRobo PoseCostMetric "
            "straight-line approach constraints. Strict waypoint validation still rejects visibly curved paths."
        ),
    )
    parser.add_argument(
        "--strict-short-linear-segments",
        dest="strict_short_linear_segments",
        action="store_true",
        default=True,
        help="Apply straight-line waypoint validation to grasp approach, post-grasp lift, final contact, and post-place retreat.",
    )
    parser.add_argument(
        "--no-strict-short-linear-segments",
        dest="strict_short_linear_segments",
        action="store_false",
        help="Disable straight-line waypoint validation for short constrained segments.",
    )
    parser.add_argument(
        "--no-final-contact-segmented-ik-fallback",
        dest="final_contact_segmented_ik_fallback",
        action="store_false",
        help="Disable segmented IK fallback for final contact approach.",
    )
    parser.add_argument(
        "--allow-segmented-ik-rescue",
        action="store_true",
        default=False,
        help="Allow slow segmented-IK rescue after cuRobo MotionGen fails for short Cartesian moves.",
    )
    parser.add_argument(
        "--two-step-final-approach-segmented-fallback",
        dest="two_step_final_approach_segmented_fallback",
        action="store_true",
        default=False,
        help="If the constrained straight primitive fails for the short pregrasp->grasp descent, "
        "try bounded segmented IK before rejecting the grasp candidate.",
    )
    parser.add_argument(
        "--no-two-step-final-approach-segmented-fallback",
        dest="two_step_final_approach_segmented_fallback",
        action="store_false",
        help="Disable segmented IK fallback for the two-step grasp final approach.",
    )
    parser.add_argument(
        "--two-step-final-approach-segmented-first",
        dest="two_step_final_approach_segmented_first",
        action="store_true",
        default=False,
        help="For short pregrasp->grasp descents, try bounded segmented IK before MotionGen.",
    )
    parser.add_argument(
        "--no-two-step-final-approach-segmented-first",
        dest="two_step_final_approach_segmented_first",
        action="store_false",
        help="Use MotionGen first for pregrasp->grasp final approach.",
    )
    parser.add_argument(
        "--allow-demo-planner-rescue",
        action="store_true",
        default=False,
        help="Allow legacy demo-planner/MPLib rescue paths after cuRobo planning fails.",
    )
    parser.add_argument(
        "--curobo-ik-prefilter-position-threshold",
        type=float,
        default=0.01,
        help="Soft position threshold used by batch IK prefilter before MotionGen. "
        "Candidates that miss strict IK but stay within this bound are still sent to trajectory planning.",
    )
    parser.add_argument(
        "--curobo-ik-prefilter-rotation-threshold",
        type=float,
        default=0.25,
        help="Soft rotation threshold (rad) used by batch IK prefilter before MotionGen. "
        "Raise this when direct-place candidates are close in position but fail due to orientation error.",
    )
    parser.add_argument(
        "--curobo-multistart-stage-size",
        type=int,
        default=128,
        help="For large multi-start goal sets with a small winner count, evaluate candidates in ordered stages of this size; set <=0 to prefilter all candidates at once.",
    )
    parser.add_argument(
        "--transport-attached-box-scale-xy",
        type=float,
        default=None,
        help="Optional XY-only scale for the cuRobo attached-object collision box. "
        "If omitted, reuse --transport-attached-box-scale.",
    )
    parser.add_argument(
        "--transport-attached-box-scale-z",
        type=float,
        default=1.0,
        help="Optional Z-only scale for the cuRobo attached-object collision box during transport. "
        "Useful when the grasped object is scraping tabletop clutter even though the XY footprint is covered.",
    )
    parser.add_argument(
        "--curobo-demo-path-validation",
        dest="curobo_demo_path_validation",
        action="store_true",
        default=False,
        help="Run the extra demo-planner dense collision validation after a cuRobo plan succeeds. "
        "Disabled by default so candidate acceptance follows cuRobo directly.",
    )
    parser.add_argument(
        "--no-curobo-demo-path-validation",
        dest="curobo_demo_path_validation",
        action="store_false",
        help="Disable the extra demo-planner dense collision validation after cuRobo success.",
    )
    return parser


def parse_args():
    args = build_arg_parser().parse_args()
    _configure_curobo_torch_extensions(args)
    return args


def _skip_post_grasp_escape(demo, bridge_mod, real_exec, args, label: str, *, use_attach: bool) -> bool:
    force_lift = _current_source_object_name(args) == "bi"
    if not force_lift and not bool(getattr(args, "post_grasp_lift_enabled", True)):
        print(f"[post_grasp_lift] skipped {label}; disabled")
        return True

    current_tcp_p = targeted.base.flatten_np(demo.tcp.pose.p)[:3].astype(np.float32)
    lift_height = float(max(getattr(args, "post_grasp_lift_height_m", 0.080), 0.0))
    min_tcp_z = float(max(getattr(args, "post_grasp_lift_min_tcp_z_m", 0.100), 0.0))
    if force_lift:
        lift_height = max(lift_height, 0.060)
        min_tcp_z = max(min_tcp_z, 0.140)
    target_z = max(float(current_tcp_p[2]) + lift_height, min_tcp_z)
    lift_delta = max(0.0, target_z - float(current_tcp_p[2]))
    if lift_delta <= 1e-4:
        print(f"[post_grasp_lift] skipped {label}; current tcp z={current_tcp_p[2]:.4f} already safe")
        return True

    print(f"\n[{label}] lifting grasped payload by {lift_delta:.3f} m before transport")
    lift_pose = targeted.base.make_pose_with_position(
        demo.tcp.pose,
        (current_tcp_p + np.array([0.0, 0.0, lift_delta], dtype=np.float32)).astype(np.float32),
    )

    planner = _get_or_create_curobo_planner_serialized(args)
    q_current = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    q_path = None
    if planner is not None:
        _refresh_curobo_world(
            planner,
            demo,
            args,
            label=label,
            include_active_object=False,
            include_table=bool(getattr(args, "curobo_table_collision", True)),
        )
        disabled_links = (
            _post_grasp_lift_disabled_world_links(planner, _direct_place_contact_tolerant_disabled_links(planner))
            if use_attach
            else _direct_place_contact_tolerant_disabled_links(planner)
        )
        if "attached_object" in set(disabled_links):
            _cache_attached_spheres_for_contact(planner)
        disabled = _set_world_collision_for_links(
            planner,
            disabled_links,
            enabled=False,
            label=label,
        )
        try:
            q_path = _plan_constrained_linear_segment(
                planner,
                demo,
                args,
                q_current,
                demo.tcp.pose,
                lift_pose,
                label=label,
                validation_pos_tol_m=float(max(getattr(args, "strict_short_linear_waypoint_pos_tol_m", 0.010), 0.0)),
            )
        finally:
            _set_world_collision_for_links(
                planner,
                disabled,
                enabled=True,
                label=label,
            )
            if "attached_object" in set(disabled):
                _restore_attached_spheres_after_contact(planner)
        if q_path is None:
            print(
                "[post_grasp_lift] cuRobo constrained lift failed; "
                "demo planner rescue is "
                f"{'enabled' if bool(getattr(args, 'allow_demo_planner_rescue', False)) else 'disabled'}"
            )

    if q_path is None and bool(getattr(args, "allow_demo_planner_rescue", False)):
        try:
            q_path = targeted.base.plan_lift_path(
                demo,
                lift_pose,
                variant_name=args.variant,
                use_attach=use_attach,
                label=label,
                planning_time=min(float(args.fixed_goal_planning_time), 3.0),
                rrt_range=float(args.fixed_goal_rrt_range),
                start_q=q_current,
                allow_pose_rrt_fallback=True,
                max_segment_joint_delta=0.35,
                max_segment_joint7_delta=0.80,
                max_segment_norm_delta=0.60,
            )
        except Exception as exc:
            print(f"[post_grasp_lift] demo planner fallback raised: {exc}")
            q_path = None
    elif q_path is None:
        print("[post_grasp_lift] skipped legacy demo planner fallback; enable --allow-demo-planner-rescue to use it")
    if q_path is None:
        print(f"[warn] {label} planning failed; continuing to transport from current low pose")
        return False

    ok, _ = targeted.base.execute_pose_path_stage(
        demo,
        bridge_mod,
        real_exec,
        label,
        lift_pose,
        q_path,
        args.real_gripper_close,
        args,
        use_attach=use_attach,
    )
    if not ok:
        return False
    targeted.base.lift_active_object_above_table_if_needed(
        demo,
        args,
        min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
    )
    return True


def _stabilize_post_grasp_attached_state(demo, args, grasp_choice) -> None:
    q_path = list(grasp_choice.get("q_path") or [])
    grasp_terminal_q = None
    if q_path:
        grasp_terminal_q = np.asarray(q_path[-1], dtype=np.float32).reshape(-1)[:7]
        targeted.base.sync_demo_arm_qpos(demo, grasp_terminal_q)
    T_tcp_obj = grasp_choice.get("T_tcp_obj")
    if T_tcp_obj is None:
        demo._transport_attached_T_tcp_obj = None
        print("[direct_pre_place] grasp choice has no saved T_tcp_obj; attached-state stabilization skipped")
        return
    demo._transport_attached_T_tcp_obj = np.asarray(T_tcp_obj, dtype=np.float32).reshape(4, 4)
    targeted._register_transport_attached_box(
        demo,
        args,
        show_visual=False,
        activate_payload_visual=False,
        T_tcp_obj_override=demo._transport_attached_T_tcp_obj,
    )
    if targeted.base.force_active_object_to_attached_pose(demo):
        print("[direct_pre_place] stabilized attached sim state from grasp-time TCP<->object transform")
    else:
        print("[direct_pre_place] failed to force active object to attached pose; continuing with current sim object state")


_MISSING_ATTR = object()


def _snapshot_transport_payload_state(demo) -> dict[str, object]:
    names = (
        "_transport_attached_T_tcp_obj",
        "attached_box_size",
        "attached_box_pose_tcp",
        "_attached_object_visual_active",
        "_attached_box_visual_visible",
    )
    return {name: getattr(demo, name, _MISSING_ATTR) for name in names}


def _restore_transport_payload_state(demo, snapshot: dict[str, object]) -> None:
    for name, value in snapshot.items():
        if value is _MISSING_ATTR:
            try:
                delattr(demo, name)
            except Exception:
                pass
        else:
            setattr(demo, name, value)


def _set_active_object_pose_quiet(demo, obj_p, obj_q) -> bool:
    obj = getattr(getattr(demo, "base_env", None), "obj", None)
    if obj is None:
        return False
    try:
        obj.set_pose(
            targeted.Pose.create_from_pq(
                p=np.asarray(obj_p, dtype=np.float32).reshape(3),
                q=np.asarray(obj_q, dtype=np.float32).reshape(4),
            )
        )
        zero_vel = np.zeros(3, dtype=np.float32)
        for method_name in ("set_linear_velocity", "set_velocity"):
            method = getattr(obj, method_name, None)
            if callable(method):
                method(zero_vel)
        ang_method = getattr(obj, "set_angular_velocity", None)
        if callable(ang_method):
            ang_method(zero_vel)
        return True
    except Exception:
        return False


def _build_transport_attach_model(args):
    attach_box_dims = targeted.base.get_asset_box_size(args.sim_asset_file, args.sim_asset_scale)
    raw_dims = np.asarray(attach_box_dims, dtype=np.float32).reshape(3)
    attach_box_scale = float(np.clip(getattr(args, "transport_attached_box_scale", 1.0), 0.5, 2.0))

    extent_ratios = raw_dims / (np.min(raw_dims) + 1e-6)
    is_spherical = bool(np.max(extent_ratios) < 1.3)

    scale_xy_override = getattr(args, "transport_attached_box_scale_xy", None)
    if scale_xy_override is None:
        scale_xy = attach_box_scale
    else:
        scale_xy = float(np.clip(scale_xy_override, 0.5, 2.0))
    if is_spherical:
        scale_z = scale_xy
    else:
        scale_z = float(np.clip(getattr(args, "transport_attached_box_scale_z", 1.0), 0.5, 2.0))

    scaled_dims = raw_dims.copy()
    scaled_dims[0] *= scale_xy
    scaled_dims[1] *= scale_xy
    scaled_dims[2] *= scale_z
    scaled_dims = np.maximum(scaled_dims, 1e-4)

    attach_kwargs = {}
    source_name = _current_source_object_name(args)
    object_category = _current_object_category(args)
    if source_name == "hongshupian":
        long_axis_idx = int(np.argmax(scaled_dims))
        short_scale = float(np.clip(getattr(args, "hongshupian_attached_short_axis_scale", 1.00), 0.5, 1.2))
        long_scale = float(np.clip(getattr(args, "hongshupian_attached_long_axis_scale", 1.12), 0.8, 1.5))
        for dim_idx in range(3):
            scaled_dims[dim_idx] *= long_scale if dim_idx == long_axis_idx else short_scale
        scaled_dims = np.maximum(scaled_dims, 1e-4)
        attach_kwargs = {
            "linear_sphere_count": int(max(getattr(args, "hongshupian_attached_sphere_count", 6), 1)),
            "linear_sphere_radius_scale": float(
                np.clip(getattr(args, "hongshupian_attached_sphere_radius_scale", 0.48), 0.20, 0.60)
            ),
            "linear_end_cover_margin_scale": 0.14,
            "linear_length_scale": 1.0,
        }
        print(
            f"[curobo] hongshupian attached model tuned: long_axis={long_axis_idx}, "
            f"long_scale={long_scale:.2f}, short_scale={short_scale:.2f}, "
            f"sphere_count={attach_kwargs['linear_sphere_count']}, "
            f"radius_scale={attach_kwargs['linear_sphere_radius_scale']:.2f}"
        )
    elif source_name == "bi":
        long_axis_idx = int(np.argmax(scaled_dims))
        short_scale = float(np.clip(getattr(args, "bi_attached_short_axis_scale", 0.75), 0.4, 1.1))
        long_scale = float(np.clip(getattr(args, "bi_attached_long_axis_scale", 0.95), 0.7, 1.2))
        for dim_idx in range(3):
            scaled_dims[dim_idx] *= long_scale if dim_idx == long_axis_idx else short_scale
        scaled_dims = np.maximum(scaled_dims, 1e-4)
        attach_kwargs = {
            "linear_sphere_count": int(max(getattr(args, "bi_attached_sphere_count", 5), 1)),
            "linear_sphere_radius_scale": float(
                np.clip(getattr(args, "bi_attached_sphere_radius_scale", 0.35), 0.20, 0.55)
            ),
            "linear_end_cover_margin_scale": 0.10,
            "linear_length_scale": 1.0,
        }
        print(
            f"[curobo] bi attached model tuned: long_axis={long_axis_idx}, "
            f"long_scale={long_scale:.2f}, short_scale={short_scale:.2f}, "
            f"sphere_count={attach_kwargs['linear_sphere_count']}, "
            f"radius_scale={attach_kwargs['linear_sphere_radius_scale']:.2f}"
        )
    elif is_spherical or object_category == "sphere":
        single_radius = 0.5 * float(np.max(scaled_dims))
        attach_kwargs = {
            "single_sphere_radius": single_radius,
        }
        print(
            f"[curobo] detected spherical object for transport, using single sphere "
            f"radius={single_radius * 1000.0:.1f}mm"
        )

    return scaled_dims.astype(np.float32), raw_dims, scale_xy, scale_z, attach_kwargs


def _attached_sphere_bottom_z(planner, q) -> float | None:
    try:
        spheres = planner.get_attached_spheres_world(np.asarray(q, dtype=np.float32).reshape(-1)[:7])
        if not spheres:
            return None
        return float(min(float(np.asarray(s["center"], dtype=np.float32)[2]) - float(s["radius"]) for s in spheres))
    except Exception:
        return None


def _attach_transport_payload_to_curobo(planner, demo, args, *, label: str) -> bool:
    start_t = time.perf_counter()
    counter_start = _snapshot_profile_counters()
    profile_success = False
    profile_status = "disabled"
    if not bool(getattr(args, "curobo_attach_object", True)):
        _record_profile(
            args,
            "transport_attach",
            success=False,
            status=profile_status,
            elapsed_ms=round((time.perf_counter() - start_t) * 1000.0, 3),
            **_profile_counter_delta(counter_start),
        )
        return False
    try:
        if getattr(planner, "attached_object_active", False):
            planner.detach_object_from_robot()
        attach_box_dims, raw_dims, scale_xy, scale_z, attach_kwargs = _build_transport_attach_model(args)
        print(
            f"[curobo] attaching object box for {label}: "
            f"original_dims={np.round(raw_dims * 1000, 1).tolist()}mm, "
            f"scaled_dims={np.round(attach_box_dims * 1000, 1).tolist()}mm, "
            f"scale_xy={scale_xy:.2f}, scale_z={scale_z:.2f}"
        )
        current_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
        obj_p_attach, obj_q_attach = demo.get_obj_pose()
        obj_pose_attach = np.concatenate(
            [
                np.asarray(obj_p_attach, dtype=np.float32).reshape(3),
                np.asarray(obj_q_attach, dtype=np.float32).reshape(4),
            ]
        )
        T_world_obj_attach = targeted.base.pose_to_matrix(
            np.asarray(obj_p_attach, dtype=np.float32).reshape(3),
            np.asarray(obj_q_attach, dtype=np.float32).reshape(4),
        )
        # cuRobo robot kinematics and collision world are expressed in the robot base frame.
        # Convert the attached payload pose to the same frame before building attached spheres.
        T_base_obj_attach = np.linalg.inv(_get_robot_base_world_transform(demo)) @ T_world_obj_attach
        obj_pose_attach_for_curobo = np.concatenate(
            [
                T_base_obj_attach[:3, 3].astype(np.float32),
                targeted.base.bridge_mod_mat2quat(T_base_obj_attach[:3, :3]).astype(np.float32),
            ]
        ).astype(np.float32)
        if bool(getattr(args, "curobo_debug", False)):
            print(
                f"[curobo] {label} attach pose converted world->base: "
                f"world_p={np.round(obj_pose_attach[:3], 4).tolist()}, "
                f"base_p={np.round(obj_pose_attach_for_curobo[:3], 4).tolist()}"
            )
        base_z_offset = float(getattr(args, "curobo_attach_world_z_offset_m", 0.002))
        _bump_profile_counter("attach_object_count")
        ok = planner.attach_object_box_to_robot(
            current_q,
            attach_box_dims,
            object_pose_world=obj_pose_attach_for_curobo,
            world_z_offset=base_z_offset,
            **attach_kwargs,
        )
        if ok and getattr(planner, "attached_object_active", False):
            bottom_z = _attached_sphere_bottom_z(planner, current_q)
            min_clearance = float(max(getattr(args, "curobo_attach_min_start_clearance_m", 0.003), 0.0))
            max_auto_offset = float(max(getattr(args, "curobo_attach_max_auto_z_offset_m", 0.030), base_z_offset))
            if bottom_z is None:
                print(f"[curobo] {label} attached sphere bottom unavailable after attach")
            else:
                virtual_table_top = float(getattr(args, "curobo_table_z_offset", -0.01))
                print(
                    f"[curobo] {label} attached sphere bottom after attach: "
                    f"z={bottom_z:.4f}m, virtual_table_top={virtual_table_top:.4f}m, "
                    f"min_clearance={min_clearance:.4f}m, world_z_offset={base_z_offset:.4f}m"
                )
            if bottom_z is not None and bottom_z < min_clearance - 1e-5 and base_z_offset < max_auto_offset:
                adjusted_offset = min(max_auto_offset, base_z_offset + (min_clearance - bottom_z))
                print(
                    f"[curobo] {label} attached payload starts too low "
                    f"(sphere_bottom_z={bottom_z:.4f}m, min_clearance={min_clearance:.4f}m); "
                    f"reattaching with world_z_offset={adjusted_offset:.4f}m"
                )
                planner.detach_object_from_robot()
                _bump_profile_counter("attach_object_count")
                ok = planner.attach_object_box_to_robot(
                    current_q,
                    attach_box_dims,
                    object_pose_world=obj_pose_attach_for_curobo,
                    world_z_offset=adjusted_offset,
                    **attach_kwargs,
                )
                if ok and getattr(planner, "attached_object_active", False):
                    adjusted_bottom_z = _attached_sphere_bottom_z(planner, current_q)
                    if adjusted_bottom_z is not None:
                        print(
                            f"[curobo] {label} attached sphere bottom after reattach: "
                            f"z={adjusted_bottom_z:.4f}m"
                        )
        if ok and getattr(planner, "attached_object_active", False):
            if bool(getattr(args, "curobo_debug", False)):
                _visualize_attached_spheres(planner, demo, args)
            profile_success = True
            profile_status = "Success"
            return True
        print(f"[curobo] failed to attach object for {label}: planner returned {ok}")
        profile_status = str(ok)
        return False
    except Exception as exc:
        print(f"[curobo] failed to attach object for {label}: {exc}")
        profile_status = type(exc).__name__
        return False
    finally:
        _record_profile(
            args,
            "transport_attach",
            success=profile_success,
            status=profile_status,
            elapsed_ms=round((time.perf_counter() - start_t) * 1000.0, 3),
            **_profile_counter_delta(counter_start),
        )


def _apply_deferred_two_step_final_approach(planner, demo, args, grasp_choice, pregrasp_lookup) -> dict:
    if bool(grasp_choice.get("two_step_grasp", False)):
        return grasp_choice
    if not pregrasp_lookup:
        return grasp_choice
    pregrasp_success = pregrasp_lookup.get(str(grasp_choice.get("label", "")))
    if pregrasp_success is None:
        return grasp_choice
    pregrasp_q = np.asarray(pregrasp_success.get("deferred_pregrasp_q"), dtype=np.float32).reshape(-1)[:7]
    pregrasp_pose = pregrasp_success.get("deferred_pregrasp_pose")
    grasp_pose = pregrasp_success.get("deferred_grasp_pose", grasp_choice.get("pose"))
    if pregrasp_pose is None or grasp_pose is None:
        return grasp_choice
    final_approach_label = f"{grasp_choice.get('label', '?')}_final_approach"
    with _profile_stage(
        args,
        "two_step_final_approach",
        candidate_count=1,
        max_attempts=int(getattr(args, "curobo_max_attempts", 2)),
        num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
        num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
        enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
    ) as prof:
        _refresh_curobo_world(
            planner,
            demo,
            args,
            label=final_approach_label,
            include_active_object=False,
            include_table=False,
        )
        source_name = _current_source_object_name(args)
        segmented_fallback_enabled = bool(getattr(args, "two_step_final_approach_segmented_fallback", False))
        final_approach_path = None
        approach_distance_m = float(pregrasp_success.get("approach_distance_m", 0.0) or 0.0)
        pair_first_q_grasp = _q7_or_none(
            pregrasp_success.get("q_grasp", grasp_choice.get("q_grasp"))
        )
        pair_first_ik_only = bool(pregrasp_success.get("pair_first_ik_only", False))
        if pair_first_ik_only and pair_first_q_grasp is not None:
            stored_approach_path = [
                np.asarray(q, dtype=np.float32).reshape(-1)[:7]
                for q in list(pregrasp_success.get("deferred_final_approach_q_path") or [])
            ]
            if stored_approach_path:
                ik_joint_path = stored_approach_path
            else:
                ik_joint_path = _linear_joint_path(pregrasp_q, pair_first_q_grasp, max_step_rad=0.035)
            if (
                len(ik_joint_path) >= 2
                and _validate_strict_linear_waypoints(
                    demo,
                    args,
                    ik_joint_path,
                    pregrasp_pose,
                    grasp_pose,
                    label=f"{final_approach_label}_pair_first_ik",
                    max_pos_err_m=float(max(getattr(args, "strict_short_linear_waypoint_pos_tol_m", 0.012), 0.0)),
                )
                and _validate_candidate_joint_path_with_demo_planner(
                    demo,
                    pregrasp_q,
                    ik_joint_path,
                    use_attach=False,
                    label=f"{final_approach_label}_pair_first_ik",
                )
            ):
                final_approach_path = ik_joint_path
                print(
                    f"[two_step_grasp] {final_approach_label}: using pair-first IK q_grasp "
                    "for the constrained final approach, preserving transport start consistency"
                )
            else:
                print(
                    f"[two_step_grasp] {final_approach_label}: pair-first q_grasp cannot be reached "
                    "by a validated straight primitive; rejecting this pair instead of switching IK branch"
                )
        if (
            final_approach_path is None
            and not pair_first_ik_only
            and
            bool(getattr(args, "two_step_final_approach_segmented_first", False))
            and segmented_fallback_enabled
            and approach_distance_m <= 0.080 + 1e-6
        ):
            final_approach_path = _plan_short_curobo_cartesian_descent(
                planner,
                demo,
                args,
                pregrasp_q,
                pregrasp_pose,
                grasp_pose,
                label=f"{final_approach_label}_segmented_first",
                force_segmented=True,
            )
        if final_approach_path is None and not pair_first_ik_only:
            final_approach_path = _plan_constrained_linear_segment(
                planner,
                demo,
                args,
                pregrasp_q,
                pregrasp_pose,
                grasp_pose,
                label=final_approach_label,
                validation_pos_tol_m=float(max(getattr(args, "strict_short_linear_waypoint_pos_tol_m", 0.010), 0.0)),
            )
        if final_approach_path is None and not pair_first_ik_only:
            # The last 7 cm of grasping starts from a near-contact configuration.
            # Keep cuRobo's constrained straight approach, but relax gripper/world
            # collision for this contact segment only; the active target remains
            # excluded from the world and full collision is restored immediately.
            disabled = _set_world_collision_for_links(
                planner,
                _direct_grasp_target_contact_only_disabled_links(planner),
                enabled=False,
                label=f"{final_approach_label}_gripper_world_relaxed",
            )
            try:
                print(
                    f"[two_step_grasp] constrained final approach failed for {grasp_choice.get('label', '?')}; "
                    "retrying with gripper world collision relaxed"
                )
                _bump_profile_counter("fallback_count")
                final_approach_path = _plan_constrained_linear_segment(
                    planner,
                    demo,
                    args,
                    pregrasp_q,
                    pregrasp_pose,
                    grasp_pose,
                    label=f"{final_approach_label}_gripper_world_relaxed",
                    validation_pos_tol_m=float(max(getattr(args, "strict_short_linear_waypoint_pos_tol_m", 0.010), 0.0)),
                )
            finally:
                _set_world_collision_for_links(
                    planner,
                    disabled,
                    enabled=True,
                    label=f"{final_approach_label}_gripper_world_relaxed",
                )
        if final_approach_path is None and not pair_first_ik_only and source_name == "bi" and segmented_fallback_enabled:
            disabled = _set_world_collision_for_links(
                planner,
                _direct_place_contact_tolerant_disabled_links(planner),
                enabled=False,
                label=f"{final_approach_label}_support_relaxed_segmented_ik",
            )
            try:
                final_approach_path = _plan_short_curobo_cartesian_descent(
                    planner,
                    demo,
                    args,
                    pregrasp_q,
                    pregrasp_pose,
                    grasp_pose,
                    label=f"{final_approach_label}_support_relaxed_segmented_ik",
                    force_segmented=True,
                    allow_motiongen_fallback=False,
                )
            finally:
                _set_world_collision_for_links(
                    planner,
                    disabled,
                    enabled=True,
                    label=f"{final_approach_label}_support_relaxed_segmented_ik",
                )
        if final_approach_path is None and not pair_first_ik_only and source_name != "bi" and segmented_fallback_enabled:
            print(
                f"[two_step_grasp] constrained final approach failed for {grasp_choice.get('label', '?')}; "
                "trying bounded segmented IK fallback"
            )
            _bump_profile_counter("fallback_count")
            final_approach_path = _plan_short_curobo_cartesian_descent(
                planner,
                demo,
                args,
                pregrasp_q,
                pregrasp_pose,
                grasp_pose,
                label=f"{final_approach_label}_segmented_ik",
                force_segmented=True,
                allow_motiongen_fallback=False,
            )
        _refresh_curobo_world(
            planner,
            demo,
            args,
            label=final_approach_label,
            include_active_object=True,
            include_table=False,
        )
        prof["success"] = bool(final_approach_path)
        prof["status"] = "Success" if final_approach_path else "PLAN_FAIL"
        prof["winner_count"] = 1 if final_approach_path else 0
        prof["world_changed"] = bool(getattr(planner, "_last_world_changed", False))
        prof["cache_hit"] = bool(getattr(planner, "_last_world_cache_hit", False))
        if final_approach_path:
            prof["path_waypoints"] = len(final_approach_path)
    if not final_approach_path:
        print(
            f"[two_step_grasp] deferred constrained final approach failed for "
            f"{grasp_choice.get('label', '?')}; candidate rejected"
        )
        return grasp_choice
    upgraded = dict(grasp_choice)
    upgraded["pose"] = grasp_pose
    upgraded["q_path"] = list(pregrasp_success["q_path"]) + list(final_approach_path[1:])
    upgraded["two_step_grasp"] = True
    upgraded["q_pregrasp"] = pregrasp_q
    upgraded["q_grasp"] = np.asarray(final_approach_path[-1], dtype=np.float32).reshape(-1)[:7]
    upgraded["pregrasp_waypoints"] = len(pregrasp_success["q_path"])
    upgraded["approach_waypoints"] = len(final_approach_path)
    upgraded["approach_distance_m"] = float(pregrasp_success.get("approach_distance_m", 0.0))
    print(
        f"[two_step_grasp] using deferred constrained final approach for {grasp_choice.get('label', '?')}: "
        f"pregrasp={len(pregrasp_success['q_path'])} + approach={len(final_approach_path)} waypoints, "
        f"distance={upgraded['approach_distance_m']*1000:.1f}mm"
    )
    return upgraded


def _normalize(vec):
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return None
    return (arr / norm).astype(np.float32)


def _unique_finite_float_list(values, *, min_value: float | None = None):
    cleaned = []
    seen = set()
    for value in list(values or []):
        try:
            f = float(value)
        except Exception:
            continue
        if not np.isfinite(f):
            continue
        if min_value is not None and f < float(min_value):
            continue
        key = round(f, 6)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(f)
    return cleaned


def _build_active_object_curobo_world(demo, args):
    asset_file = getattr(args, "sim_asset_file", None) or getattr(args, "mesh_file", None)
    asset_scale = getattr(args, "sim_asset_scale", None)
    if asset_scale is None:
        asset_scale = getattr(args, "mesh_scale", 1.0)
    try:
        obj_p, obj_q = demo.get_obj_pose()
    except Exception:
        return [], []
    pose = np.concatenate(
        [
            np.asarray(obj_p, dtype=np.float32).reshape(3),
            _normalize_quat_wxyz(obj_q),
        ]
    ).astype(np.float32)
    if asset_file is not None:
        asset_path = str(Path(str(asset_file)).expanduser())
        if Path(asset_path).exists():
            scale = float(asset_scale or 1.0)
            return [], [
                {
                    "name": "active_target_object",
                    "file_path": asset_path,
                    "scale": [scale, scale, scale],
                    "pose": pose.tolist(),
                }
            ]
    try:
        dims = targeted.base.get_asset_box_size(asset_file, float(asset_scale or 1.0))
    except Exception:
        return [], []
    return [
        {
            "name": "active_target_object",
            "dims": np.asarray(dims, dtype=np.float32).reshape(3).tolist(),
            "pose": pose.tolist(),
        }
    ], []


def _copy_scene_obstacle_entries(entries) -> list[dict]:
    copied = []
    for item in list(entries or []):
        if not isinstance(item, dict):
            continue
        out = {}
        for key, value in item.items():
            if isinstance(value, np.ndarray):
                out[key] = value.copy()
            elif isinstance(value, (list, tuple)):
                out[key] = [v.copy() if isinstance(v, np.ndarray) else copy.deepcopy(v) for v in value]
            else:
                try:
                    out[key] = copy.deepcopy(value)
                except Exception:
                    out[key] = value
        copied.append(out)
    return copied


def _build_current_target_placed_obstacle(args, T_world_obj) -> dict | None:
    object_name = curobo_wrapper.normalize_object_name(getattr(args, "object_name", None))
    if object_name is None or T_world_obj is None:
        return None
    try:
        T_world_obj = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4).copy()
    except Exception:
        return None
    asset_file = getattr(args, "sim_asset_file", None) or getattr(args, "mesh_file", None)
    if asset_file is None:
        return None
    asset_file = str(Path(str(asset_file)).expanduser())
    try:
        asset_scale = float(getattr(args, "sim_asset_scale", None) or getattr(args, "mesh_scale", 1.0) or 1.0)
        visual_box_size = np.asarray(targeted.base.get_asset_box_size(asset_file, asset_scale), dtype=np.float32).reshape(3)
    except Exception as exc:
        if bool(getattr(args, "curobo_debug", False)):
            print(f"[return_preplan] failed to build placed-object obstacle for {object_name}: {exc}")
        return None
    try:
        box_scale = float(_single_scene_obstacle_box_scale(args, object_name, placed=True))
    except Exception:
        box_scale = float(max(getattr(args, "scene_obstacle_box_scale", 1.0), 1e-3))
    planner_box_size = (visual_box_size * max(box_scale, 1e-3)).astype(np.float32)
    return {
        "object_name": object_name,
        "actor_name": f"scene_obstacle_{object_name}",
        "label": object_name,
        "score": 1.0,
        "T_world_obj": T_world_obj,
        "placed": True,
        "planner_collision": True,
        "asset_file": asset_file,
        "asset_scale": asset_scale,
        "visual_box_size": visual_box_size,
        "planner_box_size": planner_box_size,
    }


def _return_to_start_placed_obstacles(demo, args, *, place_choice=None) -> list[dict]:
    T_world_obj = None
    if place_choice is not None:
        T_world_obj = _predicted_place_T_world_obj(place_choice)
    if T_world_obj is None:
        T_world_obj = getattr(demo, "_scene_cache_placed_object_world_pose", None)
    if T_world_obj is None:
        cached = _copy_scene_obstacle_entries(getattr(args, "_return_to_start_extra_scene_obstacles", []) or [])
        return cached
    obstacle = _build_current_target_placed_obstacle(args, T_world_obj)
    if obstacle is None:
        return []
    return [obstacle]


def _placed_obstacle_world_z_height(obstacle: dict | None) -> float | None:
    if not isinstance(obstacle, dict):
        return None
    T_world_obj = obstacle.get("T_world_obj")
    dims = obstacle.get("visual_box_size", obstacle.get("planner_box_size"))
    if T_world_obj is None or dims is None:
        return None
    try:
        T_world_obj = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4)
        dims = np.asarray(dims, dtype=np.float32).reshape(3)
    except Exception:
        return None
    if not np.all(np.isfinite(T_world_obj)) or not np.all(np.isfinite(dims)):
        return None
    R = T_world_obj[:3, :3]
    height = float(np.sum(np.abs(R[2, :]) * np.maximum(dims, 0.0)))
    return height if height > 1e-6 else None


def _return_start_clearance_lift_m(args, placed_obstacles=None) -> tuple[float, dict]:
    fixed_lift = float(max(getattr(args, "return_start_clearance_lift_m", 0.060), 0.0))
    if not bool(getattr(args, "return_start_clearance_lift_from_placed_object", True)):
        return fixed_lift, {"source": "fixed", "fixed_lift_m": fixed_lift}

    heights = [
        h
        for h in (
            _placed_obstacle_world_z_height(item)
            for item in list(placed_obstacles or [])
        )
        if h is not None
    ]
    if not heights:
        return fixed_lift, {"source": "fixed_no_placed_object", "fixed_lift_m": fixed_lift}

    object_height = float(max(heights))
    extra = float(max(getattr(args, "return_start_clearance_lift_extra_m", 0.015), 0.0))
    min_lift = float(max(getattr(args, "return_start_clearance_lift_min_m", 0.030), 0.0))
    max_lift = float(max(getattr(args, "return_start_clearance_lift_max_m", 0.090), min_lift))
    raw_lift = 0.5 * object_height + extra
    lift_m = float(np.clip(raw_lift, min_lift, max_lift))
    return lift_m, {
        "source": "placed_object_half_height",
        "object_world_z_height_m": object_height,
        "raw_lift_m": raw_lift,
        "extra_m": extra,
        "min_lift_m": min_lift,
        "max_lift_m": max_lift,
    }


def _build_virtual_table_cuboid(args) -> dict:
    table_x = float(getattr(args, "curobo_table_center_x", 0.0))
    table_y = float(getattr(args, "curobo_table_center_y", 0.0))
    table_thickness = float(max(getattr(args, "curobo_table_thickness", 0.02), 1e-3))
    table_size_x = float(max(getattr(args, "curobo_table_size_x", 1.2), 0.1))
    table_size_y = float(max(getattr(args, "curobo_table_size_y", 1.2), 0.1))
    z_offset = float(getattr(args, "curobo_table_z_offset", -0.01))
    table_z = z_offset - 0.5 * table_thickness

    if bool(getattr(args, "curobo_debug", False)):
        print(
            f"[curobo] virtual table: center=({table_x:.3f}, {table_y:.3f}, {table_z:.3f}), "
            f"size=({table_size_x:.3f}, {table_size_y:.3f}, {table_thickness:.3f}), "
            f"top_surface_z={z_offset:.3f}"
        )

    return {
        "name": "virtual_table_plane",
        "dims": [table_size_x, table_size_y, table_thickness],
        "pose": [table_x, table_y, table_z, 1.0, 0.0, 0.0, 0.0],
    }


def _round_signature_values(values):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return tuple(float(np.round(v, 6)) for v in arr.tolist())


def _world_obstacle_signature(cuboids, meshes):
    cuboid_sig = []
    for item in list(cuboids or []):
        cuboid_sig.append(
            (
                str(item.get("name", "")),
                _round_signature_values(item.get("dims", [])),
                _round_signature_values(item.get("pose", item.get("xyz_quat", []))),
            )
        )
    mesh_sig = []
    for item in list(meshes or []):
        mesh_sig.append(
            (
                str(item.get("name", "")),
                str(item.get("file_path", item.get("asset_file", item.get("mesh_file", "")))),
                _round_signature_values(item.get("scale", item.get("asset_scale", [1.0, 1.0, 1.0]))),
                _round_signature_values(item.get("pose", item.get("xyz_quat", []))),
            )
        )
    cuboid_sig.sort()
    mesh_sig.sort()
    return (tuple(cuboid_sig), tuple(mesh_sig))


def _visualize_attached_spheres(planner, demo, args):
    """Print and optionally visualize the attached object collision spheres."""
    if not bool(getattr(args, "curobo_debug", False)):
        return
    current_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    spheres = planner.get_attached_spheres_world(current_q)
    if not spheres:
        print("[curobo] no attached spheres to visualize")
        return
    T_world_base = _get_robot_base_world_transform(demo)
    print(f"[curobo] attached object collision spheres ({len(spheres)} total):")
    for i, s in enumerate(spheres):
        center_base = np.asarray(s["center"], dtype=np.float32).reshape(3)
        world_center = (T_world_base @ np.array([center_base[0], center_base[1], center_base[2], 1.0], dtype=np.float32))[:3]
        print(
            f"  sphere[{i}]: world_center=[{world_center[0]:.4f}, {world_center[1]:.4f}, {world_center[2]:.4f}], "
            f"radius={s['radius']*1000:.1f}mm"
        )
    if str(getattr(args, "render_mode", "none")) == "none":
        return
    if not bool(getattr(args, "curobo_show_attached_spheres", True)):
        return
    try:
        from sapien.core import Pose
        env = demo.env
        sphere_actors = list(getattr(env.unwrapped, "_attached_sphere_actors", []) or [])
        for actor in sphere_actors:
            try:
                actor.remove_from_scene()
            except Exception:
                pass
        sphere_actors = []
        visual_seq = int(getattr(env.unwrapped, "_attached_sphere_visual_seq", 0)) + 1
        env.unwrapped._attached_sphere_visual_seq = visual_seq
        for i, s in enumerate(spheres):
            center_base = np.asarray(s["center"], dtype=np.float32).reshape(3)
            world_center = (T_world_base @ np.array([center_base[0], center_base[1], center_base[2], 1.0], dtype=np.float32))[:3]
            radius = float(s["radius"])
            builder = env.unwrapped.scene.create_actor_builder()
            builder.add_sphere_visual(radius=radius, material=None)
            actor = builder.build_static(name=f"attached_sphere_{visual_seq}_{i}")
            actor.set_pose(Pose(p=world_center.tolist()))
            try:
                for body in actor.get_visual_bodies():
                    for shape in body.get_render_shapes():
                        mat = shape.material
                        mat.set_base_color([1.0, 0.2, 0.2, 0.4])
                        shape.set_material(mat)
            except Exception:
                pass
            sphere_actors.append(actor)
        env.unwrapped._attached_sphere_actors = sphere_actors
        print(f"[curobo] visualized {len(sphere_actors)} attached collision spheres (red, semi-transparent)")
    except Exception as exc:
        print(f"[curobo] sphere visualization failed (non-critical): {exc}")


def _print_attached_sphere_clearance(planner, demo, args, *, label: str) -> None:
    if planner is None or not getattr(planner, "attached_object_active", False):
        return
    try:
        current_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
        spheres = planner.get_attached_spheres_world(current_q)
        if not spheres:
            return
        sphere_bottoms = [float(np.asarray(s["center"], dtype=np.float32)[2]) - float(s["radius"]) for s in spheres]
        bottom_z = float(min(sphere_bottoms))
        print(f"[collision] {label}: cuRobo attached sphere bottom z: {bottom_z:.4f} m (table plane approx z=0.0000)")
        if bottom_z < 0.0:
            print(f"[collision] {label}: cuRobo attached spheres appear to penetrate the table by {-bottom_z:.4f} m")
    except Exception as exc:
        if bool(getattr(args, "curobo_debug", False)):
            print(f"[collision] {label}: failed to compute cuRobo attached sphere clearance: {exc}")


def _clear_visualized_attached_spheres(demo) -> None:
    env = getattr(demo, "env", None)
    if env is None:
        return
    sphere_actors = list(getattr(env.unwrapped, "_attached_sphere_actors", []) or [])
    for actor in sphere_actors:
        try:
            actor.remove_from_scene()
        except Exception:
            pass
    env.unwrapped._attached_sphere_actors = []


def _refresh_curobo_world(
    planner, demo, args, *, label: str, include_active_object: bool = False, include_table: bool = False,
    exclude_object_names: set[str] | None = None,
    extra_scene_obstacles: list[dict] | None = None,
) -> None:
    _bump_profile_counter("world_refresh_count")
    if planner.collision_enabled:
        requested_excludes = {
            curobo_wrapper.normalize_object_name(x)
            for x in list(exclude_object_names or [])
            if curobo_wrapper.normalize_object_name(x) is not None
        }
        cuboids, meshes = curobo_wrapper._scene_obstacles_to_curobo_world(
            demo, args, exclude_object_names=requested_excludes or None,
        )
        if extra_scene_obstacles:
            extra_demo = SimpleNamespace(scene_obstacles=_copy_scene_obstacle_entries(extra_scene_obstacles))
            extra_cuboids, extra_meshes = curobo_wrapper._scene_obstacles_to_curobo_world(
                extra_demo,
                args,
                exclude_object_names=requested_excludes or None,
            )
            cuboids.extend(extra_cuboids)
            meshes.extend(extra_meshes)
        if include_active_object:
            active_cuboids, active_meshes = _build_active_object_curobo_world(demo, args)
            cuboids.extend(active_cuboids)
            meshes.extend(active_meshes)
        if include_table and bool(getattr(args, "curobo_table_collision", True)):
            cuboids.append(_build_virtual_table_cuboid(args))
        world_signature = _world_obstacle_signature(cuboids, meshes)
        world_changed = world_signature != getattr(planner, "_persistent_world_signature", None)
        planner._last_world_changed = bool(world_changed)
        planner._last_world_cache_hit = not bool(world_changed)
        if world_changed:
            targeted.base.sync_curobo_collision_world_visuals(
                demo.env,
                args,
                cuboids=cuboids,
                meshes=meshes,
                label=label,
            )
        if planner.attached_object_active and bool(getattr(args, "curobo_debug", False)):
            _visualize_attached_spheres(planner, demo, args)
        else:
            _clear_visualized_attached_spheres(demo)
        if world_changed:
            cuboids_in_base, meshes_in_base = curobo_wrapper._transform_curobo_world_to_robot_base(
                cuboids,
                meshes,
                demo,
            )
            with _CUROBO_GPU_LOCK:
                planner.set_world_from_obstacles(cuboids=cuboids_in_base, meshes=meshes_in_base)
            planner._persistent_world_signature = world_signature
        if requested_excludes:
            print(
                f"[curobo] {label} world: excluded scene obstacle(s): {sorted(requested_excludes)}"
            )
            remaining_names = [str(item.get("name", "")) for item in list(cuboids or []) + list(meshes or [])]
            leaked = []
            for name in remaining_names:
                normalized_name = curobo_wrapper.normalize_object_name(name)
                stripped_name = name
                if stripped_name.startswith("scene_obstacle_"):
                    stripped_name = stripped_name[len("scene_obstacle_") :]
                normalized_stripped = curobo_wrapper.normalize_object_name(stripped_name)
                if normalized_name in requested_excludes or normalized_stripped in requested_excludes:
                    leaked.append(name)
            if leaked:
                print(
                    f"[curobo] WARNING: {label} exclude request did not remove obstacle(s): {leaked}"
                )
        if bool(getattr(args, "curobo_debug", False)):
            if world_changed:
                print(
                    f"[curobo] updated world with {len(cuboids_in_base)} cuboid and "
                    f"{len(meshes_in_base)} mesh obstacles for {label}"
                )
            else:
                print(f"[curobo] reused persistent world for {label}")
            cuboid_names = [str(item.get("name", "")) for item in list(cuboids or [])]
            mesh_names = [str(item.get("name", "")) for item in list(meshes or [])]
            print(
                f"[curobo] {label} world members: "
                f"cuboids={cuboid_names}, meshes={mesh_names}"
            )
    elif bool(getattr(args, "curobo_debug", False)):
        print(f"[curobo] planner is in embedded free-space mode for {label}")
    else:
        _clear_visualized_attached_spheres(demo)


def _direct_grasp_target_contact_only_disabled_links(planner) -> list[str]:
    configured_links = set(getattr(planner, "configured_collision_links", []) or [])
    if not configured_links:
        return []
    candidate_disable_links = {
        "gripper_base_link",
        "gripper_Left_1_Link",
        "gripper_Left_Support_Link",
        "gripper_Left_2_Link",
        "gripper_Right_1_Link",
        "gripper_Right_Support_Link",
        "gripper_Right_2_Link",
        "left_pad",
        "right_pad",
    }
    return sorted(candidate_disable_links & configured_links)


def _direct_place_contact_tolerant_disabled_links(planner) -> list[str]:
    """
    放置时禁用的链路：允许夹爪的support link接触桌面，
    但保留pad的碰撞检测以确保不会夹到目标物体。
    """
    configured_links = set(getattr(planner, "configured_collision_links", []) or [])
    if not configured_links:
        return []
    candidate_disable_links = {
        "gripper_Left_Support_Link",
        "gripper_Right_Support_Link",
    }
    return sorted(candidate_disable_links & configured_links)


def _post_grasp_lift_disabled_world_links(planner, base_links: list[str] | None = None) -> list[str]:
    """
    抓住后的第一段直线上提用于把物体从接触面上解耦。
    这段从夹爪/物体/桌面的接触状态出发，允许夹爪和 attached payload
    暂时忽略世界碰撞；后续 transport 会重新启用完整碰撞。
    """
    configured_links = set(getattr(planner, "configured_collision_links", []) or [])
    links = set(base_links or [])
    links.update(_direct_grasp_target_contact_only_disabled_links(planner))
    # attached_object is a dynamic cuRobo collision link created by attach_spheres_to_robot().
    # It is not guaranteed to appear in the static robot yaml collision_link_names.
    if getattr(planner, "attached_object_active", False) or "attached_object" in configured_links:
        links.add("attached_object")
    if configured_links:
        return sorted((links & configured_links) | ({"attached_object"} if "attached_object" in links else set()))
    return sorted(links)


def _normalize_disabled_world_collision_links(planner, disabled_world_collision_links) -> list[str]:
    if not bool(getattr(planner, "collision_enabled", False)):
        return []
    configured_links = set(getattr(planner, "configured_collision_links", []) or [])
    normalized = []
    seen = set()
    for link_name in list(disabled_world_collision_links or []):
        name = str(link_name)
        if not name or name in seen:
            continue
        if configured_links and name not in configured_links and name != "attached_object":
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


def _set_world_collision_for_links(planner, link_names, *, enabled: bool, label: str) -> list[str]:
    normalized = _normalize_disabled_world_collision_links(planner, link_names)
    if normalized:
        planner.set_world_collision_for_links(normalized, enabled=enabled)
        action = "re-enabled" if enabled else "temporarily disabled"
        print(f"[curobo] {label} {action} world collision for links: {normalized}")
    return normalized


def _path_metrics_and_score(q_start, q_path):
    metrics = targeted.base._joint_path_quality_metrics(q_start, q_path)
    score = targeted.base._joint_path_quality_score(metrics)
    return metrics, float(score)


def _linear_joint_path(q_start, q_goal, *, max_step_rad: float = 0.045) -> list[np.ndarray]:
    q0 = np.asarray(q_start, dtype=np.float32).reshape(-1)[:7]
    q1 = np.asarray(q_goal, dtype=np.float32).reshape(-1)[:7]
    if q0.shape[0] < 7 or q1.shape[0] < 7:
        return []
    if not np.all(np.isfinite(q0)) or not np.all(np.isfinite(q1)):
        return []
    max_step = float(max(max_step_rad, 1e-3))
    n_steps = int(max(2, np.ceil(float(np.max(np.abs(q1 - q0))) / max_step) + 1))
    return [
        ((1.0 - alpha) * q0 + alpha * q1).astype(np.float32)
        for alpha in np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    ]


def _candidate_selection_penalty(candidate, args) -> float:
    axis_shift_m = float(candidate.get("grasp_axis_shift_m", 0.0))
    z_lift_m = float(candidate.get("grasp_z_lift_m", 0.0))
    axis_penalty_per_cm = float(max(getattr(args, "direct_grasp_axis_shift_penalty_per_cm", 0.10), 0.0))
    z_lift_penalty_per_cm = float(max(getattr(args, "direct_grasp_z_lift_penalty_per_cm", 0.08), 0.0))
    axis_cm = abs(axis_shift_m) * 100.0
    z_lift_cm = max(z_lift_m, 0.0) * 100.0
    return float(axis_penalty_per_cm * axis_cm + z_lift_penalty_per_cm * z_lift_cm)


def _candidate_target_up_axis_world(candidate, demo) -> np.ndarray | None:
    place_plan = candidate.get("place_plan")
    target_name = None if place_plan is None else getattr(place_plan, "target_name", None)
    if target_name:
        scene_entry = targeted._find_scene_object_entry(demo, target_name)
        if scene_entry is not None and scene_entry.get("T_world_obj") is not None:
            T_world_target = np.asarray(scene_entry["T_world_obj"], dtype=np.float32).reshape(4, 4)
            return _normalize(T_world_target[:3, 1])
    return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)


def _candidate_place_orientation_penalty(candidate, demo, args) -> float:
    weight = float(max(getattr(args, "place_orientation_preference_weight", 0.25), 0.0))
    if weight <= 0.0:
        return 0.0
    place_plan = candidate.get("place_plan")
    if place_plan is None:
        return 0.0
    T_world_obj_desired = getattr(place_plan, "T_world_obj_desired", None)
    if T_world_obj_desired is None:
        return 0.0
    rule = getattr(place_plan, "rule", None)
    T_world_obj_desired = np.asarray(T_world_obj_desired, dtype=np.float32).reshape(4, 4)
    robot_pos = targeted.base.flatten_np(demo.robot.pose.p)[:3].astype(np.float32)
    obj_pos = T_world_obj_desired[:3, 3].astype(np.float32)
    up_axis = _candidate_target_up_axis_world(candidate, demo)
    if up_axis is None:
        up_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    toward_robot = robot_pos - obj_pos
    toward_robot = toward_robot - float(np.dot(toward_robot, up_axis)) * up_axis
    toward_robot = _normalize(toward_robot)
    if toward_robot is None:
        return 0.0

    R_world_obj = T_world_obj_desired[:3, :3].astype(np.float32)
    face_axis_local = None if rule is None else getattr(rule, "face_robot_axis_local", None)
    if face_axis_local is not None:
        face_axis_world = (R_world_obj @ np.asarray(face_axis_local, dtype=np.float32).reshape(3)).astype(np.float32)
        face_axis_world = face_axis_world - float(np.dot(face_axis_world, up_axis)) * up_axis
        face_axis_world = _normalize(face_axis_world)
        if face_axis_world is None:
            return 0.0
        alignment = float(np.clip(np.dot(face_axis_world, toward_robot), -1.0, 1.0))
    else:
        alignment = -1.0
        for axis_local in (
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        ):
            axis_world = (R_world_obj @ axis_local).astype(np.float32)
            axis_world = axis_world - float(np.dot(axis_world, up_axis)) * up_axis
            axis_world = _normalize(axis_world)
            if axis_world is None:
                continue
            alignment = max(alignment, float(np.clip(np.dot(axis_world, toward_robot), -1.0, 1.0)))
        if alignment < -0.5:
            return 0.0
    return float((1.0 - alignment) * 0.5 * weight)


def _candidate_sort_key(item):
    metrics = item["metrics"]
    return (
        float(item["score"]),
        float(metrics["total_motion"]),
        float(metrics["joint7_total_motion"]),
        int(metrics["waypoint_count"]),
    )


def _variant_yaw_deg_from_labels(variant_label: str | None, label: str | None = None) -> float:
    """Parse tabletop yaw or insert axial spin from labels. No match -> 0."""
    text = " ".join(str(x) for x in (variant_label, label) if x is not None)
    m = re.search(r"(?:yaw|spin)_([-+]?\d+(?:\.\d+)?)deg", text)
    if not m:
        return 0.0
    yaw = float(m.group(1))
    yaw = ((yaw + 180.0) % 360.0) - 180.0
    if abs(yaw + 180.0) <= 1e-6:
        yaw = 180.0
    return yaw


def _variant_abs_yaw_deg_from_labels(variant_label: str | None, label: str | None = None) -> float:
    """Parse tabletop yaw magnitude from variant/label, e.g. yaw_-60deg -> 60. No match -> 0."""
    return abs(_variant_yaw_deg_from_labels(variant_label, label))


def _pre_place_screen_sort_key(item):
    label = "" if item.get("label") is None else str(item.get("label"))
    variant = "" if item.get("variant_label") is None else str(item.get("variant_label"))
    pose_z = float(_get_pose_position(item["pose"])[2]) if "pose" in item else 0.0
    place_z = float(_get_pose_position(item["place_pose"])[2]) if "place_pose" in item else 0.0
    yaw_abs = _variant_abs_yaw_deg_from_labels(variant, label)
    hover_extra = float(item.get("hover_extra_height_m", 0.0) or 0.0)
    verticality_target = _candidate_tcp_verticality_target(item)
    axis_vertical_target = _candidate_tcp_axis_vertical_target(item)
    if verticality_target is not None or axis_vertical_target is not None:
        axes_z = _tcp_axis_world_z_components(item["place_pose"]) if "place_pose" in item else {}
        axis_delta = 0.0
        if axis_vertical_target is not None:
            axis_delta = abs(float(axes_z.get(f"abs_{axis_vertical_target}", 0.0)) - 1.0)
        verticality_delta = 0.0
        if verticality_target is not None:
            verticality_delta = abs(float(item.get("tcp_verticality", 0.0)) - verticality_target)
        return (
            axis_delta,
            verticality_delta,
            yaw_abs,
            hover_extra,
            -place_z,
            -pose_z,
            variant,
            label,
        )
    return (
        yaw_abs,
        hover_extra,
        -float(item.get("tcp_verticality", 0.0)),
        -place_z,
        -pose_z,
        variant,
        label,
    )


def _candidate_tcp_verticality_target(item) -> float | None:
    place_plan = item.get("place_plan") if isinstance(item, dict) else None
    rule = getattr(place_plan, "rule", None)
    if rule is None:
        return None
    target = getattr(rule, "tabletop_place_tcp_verticality_target", None)
    if target is None:
        return None
    try:
        return float(np.clip(float(target), 0.0, 1.0))
    except Exception:
        return None


def _candidate_tcp_axis_vertical_target(item) -> str | None:
    place_plan = item.get("place_plan") if isinstance(item, dict) else None
    rule = getattr(place_plan, "rule", None)
    if rule is None:
        return None
    axis_name = getattr(rule, "tabletop_place_tcp_axis_vertical", None)
    if axis_name is None:
        return None
    axis_name = str(axis_name).strip().lower()
    if axis_name not in {"x", "y", "z"}:
        return None
    return axis_name


def _filter_place_candidates_by_tcp_verticality_target(candidates, args, *, label: str) -> list:
    remaining = list(candidates or [])
    if not remaining:
        return remaining
    band = float(max(getattr(args, "direct_pre_place_verticality_band", 0.08), 0.0))
    verticality_values = [
        float(v)
        for v in (_candidate_tcp_verticality_target(item) for item in remaining)
        if v is not None and np.isfinite(float(v))
    ]
    axis_values = [v for v in (_candidate_tcp_axis_vertical_target(item) for item in remaining) if v is not None]
    if not verticality_values and not axis_values:
        return remaining
    filtered = remaining
    if axis_values:
        axis_name = axis_values[0]
        best_axis_delta = min(
            abs(float(_tcp_axis_world_z_components(item["place_pose"]).get(f"abs_{axis_name}", 0.0)) - 1.0)
            for item in filtered
            if "place_pose" in item
        )
        axis_filtered = [
            item
            for item in filtered
            if "place_pose" in item
            and abs(float(_tcp_axis_world_z_components(item["place_pose"]).get(f"abs_{axis_name}", 0.0)) - 1.0)
            <= best_axis_delta + band
        ]
        if axis_filtered:
            filtered = axis_filtered
            print(
                f"[direct_pre_place] {label}: kept {len(filtered)}/{len(remaining)} candidate(s) "
                f"near tcp_{axis_name}_vertical (best_delta={best_axis_delta:.3f}, band={band:.3f})"
            )
    if verticality_values:
        target = float(verticality_values[0])
        best_delta = min(abs(float(item.get("tcp_verticality", 0.0)) - target) for item in filtered)
        verticality_filtered = [
            item
            for item in filtered
            if abs(float(item.get("tcp_verticality", 0.0)) - target) <= best_delta + band
        ]
        if verticality_filtered:
            filtered = verticality_filtered
    if filtered is not remaining:
        print(
            f"[direct_pre_place] {label}: kept {len(filtered)}/{len(remaining)} candidate(s) "
            f"near tcp_verticality_target={float(verticality_values[0]) if verticality_values else float('nan'):.3f}"
        )
        return filtered
    return remaining


def _yaw_distance_deg(a: float, b: float) -> float:
    return abs(((float(a) - float(b) + 180.0) % 360.0) - 180.0)


def _select_diverse_place_candidates(candidates, max_count: int, *, label: str) -> list:
    ordered = sorted(list(candidates or []), key=_pre_place_screen_sort_key)
    max_count = int(max_count)
    if max_count <= 0 or len(ordered) <= max_count:
        return ordered

    groups: dict[int, list] = {}
    for item in ordered:
        yaw = _variant_yaw_deg_from_labels(item.get("variant_label"), item.get("label"))
        yaw_bucket = int(round(yaw / 5.0) * 5)
        if yaw_bucket == -180:
            yaw_bucket = 180
        groups.setdefault(yaw_bucket, []).append(item)

    available = set(groups.keys())
    selected_yaws: list[int] = []
    while available and len(selected_yaws) < min(max_count, len(groups)):
        if not selected_yaws:
            best_yaw = min(available, key=lambda y: (abs(y), y))
        else:
            best_yaw = max(
                available,
                key=lambda y: (
                    min(_yaw_distance_deg(y, s) for s in selected_yaws),
                    -abs(abs(y) - 90.0),
                    -abs(y),
                    -y,
                ),
            )
        selected_yaws.append(best_yaw)
        available.remove(best_yaw)

    selected = []
    used_ids: set[int] = set()

    def _take_from_yaw(yaw: int) -> None:
        if len(selected) >= max_count:
            return
        bucket = groups.get(yaw, [])
        while bucket:
            item = bucket.pop(0)
            key = id(item)
            if key in used_ids:
                continue
            selected.append(item)
            used_ids.add(key)
            return

    for yaw in selected_yaws:
        _take_from_yaw(yaw)
    yaw_order = selected_yaws + [y for y in sorted(groups.keys(), key=lambda y: (abs(y), y)) if y not in selected_yaws]
    while len(selected) < max_count:
        before = len(selected)
        for yaw in yaw_order:
            _take_from_yaw(yaw)
            if len(selected) >= max_count:
                break
        if len(selected) == before:
            break

    selected_yaw_values = [
        int(round(_variant_yaw_deg_from_labels(item.get("variant_label"), item.get("label"))))
        for item in selected
    ]
    selected_hover_mm = [int(round(float(item.get("hover_extra_height_m", 0.0) or 0.0) * 1000.0)) for item in selected]
    print(
        f"[joint_search] {label} diverse candidate selection kept {len(selected)}/{len(ordered)} "
        f"(yaw_deg={selected_yaw_values}, hover_extra_mm={selected_hover_mm})"
    )
    return selected


def _bi_fast_insert_values(args):
    spin_degs = _unique_finite_float_list(
        getattr(args, "bi_fast_insert_spin_deg", [0.0, -15.0, 15.0, -90.0, 90.0, -120.0, -135.0, -150.0]),
    )
    if not spin_degs:
        spin_degs = [0.0]
    approach_distances = _unique_finite_float_list(
        getattr(args, "bi_fast_insert_approach_distances", [0.05]),
        min_value=0.005,
    )
    if not approach_distances:
        approach_distances = [0.05]
    hover_extra_heights = _unique_finite_float_list(
        getattr(args, "bi_fast_insert_hover_extra_heights_m", [0.0, 0.03]),
        min_value=0.0,
    )
    if not hover_extra_heights:
        hover_extra_heights = [0.0]
    return spin_degs, approach_distances, hover_extra_heights


def _bi_fast_insert_screen_args(args):
    spin_degs, approach_distances, hover_extra_heights = _bi_fast_insert_values(args)
    adjusted = SimpleNamespace(**vars(args))
    adjusted.insert_vertical_axial_spin_deg = list(spin_degs)
    adjusted.direct_release_approach_distances = list(approach_distances)
    adjusted.direct_release_approach_distance = float(approach_distances[0])
    adjusted.transport_hover_extra_heights_m = list(hover_extra_heights)
    adjusted.bi_insert_release_height_offsets_m = [0.0]
    adjusted.targeted_place_allow_insert_axis_flip = False
    return adjusted


def _bi_insert_small_lane_candidates(candidates, args, *, label: str) -> list:
    ordered = sorted(list(candidates or []), key=_pre_place_screen_sort_key)
    if not ordered:
        return []
    spin_degs, approach_distances, hover_extra_heights = _bi_fast_insert_values(args)
    max_count = max(1, len(spin_degs) * len(approach_distances) * len(hover_extra_heights))
    fast = []
    for item in ordered:
        if str(item.get("place_mode", "")) != "insert_place":
            continue
        item_label = str(item.get("label", ""))
        approach_match = re.search(r"approach_([-+]?\d+(?:\.\d+)?)mm", item_label.lower())
        approach_m = float(approach_match.group(1)) / 1000.0 if approach_match else 0.0
        if all(abs(approach_m - float(target)) > 0.0015 for target in approach_distances):
            continue
        spin_deg = _variant_yaw_deg_from_labels(item.get("variant_label"), item.get("label"))
        if all(_yaw_distance_deg(spin_deg, preferred) > 1.0 for preferred in spin_degs):
            continue
        hover_extra = float(item.get("hover_extra_height_m", 0.0) or 0.0)
        if all(abs(hover_extra - float(target)) > 0.0015 for target in hover_extra_heights):
            continue
        release_offset = float(item.get("insert_release_height_offset_m", 0.0) or 0.0)
        if release_offset > 1e-6:
            continue
        fast.append(item)
    if not fast:
        return []
    fast = fast[:max_count]
    selected_spins = [
        int(round(_variant_yaw_deg_from_labels(item.get("variant_label"), item.get("label"))))
        for item in fast
    ]
    selected_approach = []
    for item in fast:
        m = re.search(r"approach_([-+]?\d+(?:\.\d+)?)mm", str(item.get("label", "")).lower())
        selected_approach.append(int(round(float(m.group(1)))) if m else 0)
    selected_hover_mm = [int(round(float(item.get("hover_extra_height_m", 0.0) or 0.0) * 1000.0)) for item in fast]
    print(
        f"[joint_search] {label} bi small insert lane kept {len(fast)}/{len(ordered)} "
        f"candidate(s): spin_deg={selected_spins}, approach_mm={selected_approach}, "
        f"hover_extra_mm={selected_hover_mm}; full set remains fallback"
    )
    return fast


def _bi_insert_fast_lane_candidates(candidates, args=None, *, label: str) -> list:
    ordered = sorted(list(candidates or []), key=_pre_place_screen_sort_key)
    if not ordered:
        return []
    if args is not None:
        small = _bi_insert_small_lane_candidates(ordered, args, label=label)
        if small:
            return small
    preferred_spins = (0.0, -15.0, 15.0, -90.0, 90.0, -105.0, -120.0, -135.0, -150.0)
    fast = []
    for item in ordered:
        item_label = str(item.get("label", ""))
        if str(item.get("place_mode", "")) != "insert_place":
            continue
        if "approach_50mm" not in item_label:
            continue
        spin_deg = _variant_yaw_deg_from_labels(item.get("variant_label"), item.get("label"))
        if all(_yaw_distance_deg(spin_deg, preferred) > 1.0 for preferred in preferred_spins):
            continue
        hover_extra = float(item.get("hover_extra_height_m", 0.0) or 0.0)
        if hover_extra > 0.031:
            continue
        fast.append(item)
    if not fast or len(fast) >= len(ordered):
        return ordered
    selected_spins = sorted(
        {int(round(_variant_yaw_deg_from_labels(item.get("variant_label"), item.get("label")))) for item in fast}
    )
    print(
        f"[joint_search] {label} bi insert fast lane kept {len(fast)}/{len(ordered)} "
        f"candidate(s): approach_50mm, hover_extra<=30mm, preferred raw/no-spin first, spin_deg={selected_spins}; "
        "full set remains fallback"
    )
    return fast


def _vertical_long_axis_transport_fast_lane_candidates(candidates, source_name: str | None, args, *, label: str) -> list:
    """High-probability transport candidates for vertical long-axis tabletop objects.

    This does not remove any candidates from the full search.  It only probes a
    small, repeatedly successful subset before falling back to the complete
    candidate set.  The final object pose remains the original place-rule pose.
    """
    source = str(source_name or "").lower()
    if source not in {"gluestick", "hongshupian"}:
        return []

    max_count = int(getattr(args, "vertical_long_axis_transport_fast_lane_max_candidates", 0) or 0)
    if max_count <= 0:
        return []

    ordered = sorted(list(candidates or []), key=_pre_place_screen_sort_key)
    if not ordered:
        return []

    def _slot_index(item) -> int:
        m = re.search(r"slot_(\d+)", str(item.get("slot_name", "")))
        return int(m.group(1)) if m else 999

    preferred = []
    fallback = []
    for item in ordered:
        label_text = " ".join(str(item.get(k, "")) for k in ("label", "variant_label", "slot_name")).lower()
        yaw = _variant_yaw_deg_from_labels(item.get("variant_label"), item.get("label"))
        hover_extra = float(item.get("hover_extra_height_m", 0.0) or 0.0)
        slot_idx = _slot_index(item)

        if source == "gluestick":
            # Glue-stick vertical placements have consistently accepted yaw_135
            # once the transport branch is otherwise feasible.  Include the
            # nearby opposite-wrap spelling in case label normalization changes.
            yaw_rank = min(_yaw_distance_deg(yaw, 135.0), _yaw_distance_deg(yaw, -135.0))
            is_preferred = yaw_rank <= 1.0 and hover_extra <= 0.061
            rank = (yaw_rank, hover_extra, float(item.get("score", 0.0) or 0.0), label_text)
        else:
            # Chip-can vertical placements vary by slot.  Keep a small slot-aware
            # yaw set that covers the successful fixed/random-scene branches.
            if slot_idx >= 5:
                target_yaws = (45.0, 0.0, -90.0)
            elif slot_idx == 4:
                target_yaws = (0.0, -90.0, 45.0)
            else:
                target_yaws = (0.0, 45.0, -90.0)
            yaw_rank = min(_yaw_distance_deg(yaw, target) for target in target_yaws)
            is_preferred = yaw_rank <= 1.0 and hover_extra <= 0.061
            rank = (yaw_rank, hover_extra, float(item.get("score", 0.0) or 0.0), label_text)

        bucket_item = (rank, item)
        if is_preferred:
            preferred.append(bucket_item)
        else:
            fallback.append(bucket_item)

    preferred.sort(key=lambda x: x[0])
    fallback.sort(key=lambda x: x[0])
    fast = [item for _, item in (preferred + fallback)[:max_count]]
    if not fast or len(fast) >= len(ordered):
        return []
    selected = [
        str(item.get("label", "?"))
        for item in fast[: min(len(fast), 8)]
    ]
    print(
        f"[joint_search] {label} {source} vertical-long-axis fast lane kept "
        f"{len(fast)}/{len(ordered)} candidate(s): {selected}; full set remains fallback"
    )
    return fast


def _final_contact_validation_sort_key(item) -> tuple:
    hover_extra = float(item.get("hover_extra_height_m", 0.0) or 0.0)
    spin_deg = _variant_yaw_deg_from_labels(item.get("variant_label"), item.get("label"))
    metrics = item.get("metrics") or {}
    return (
        1 if hover_extra > 1e-6 else 0,
        hover_extra,
        float(item.get("score", 0.0) or 0.0),
        float(metrics.get("total_motion", 0.0) or 0.0),
        _yaw_distance_deg(spin_deg, -90.0),
        spin_deg,
    )


def _bi_final_contact_validation_sort_key(item) -> tuple:
    """Prefer insert descents that have repeatedly validated for the pen.

    This is deliberately ordering-only: if the preferred spin fails, the full
    set still falls through to the generic validation order.
    """
    label = "" if item.get("label") is None else str(item.get("label"))
    variant = "" if item.get("variant_label") is None else str(item.get("variant_label"))
    text = f"{variant} {label}".lower()
    spin_deg = _variant_yaw_deg_from_labels(variant, label)
    hover_extra = float(item.get("hover_extra_height_m", 0.0) or 0.0)
    approach_match = re.search(r"approach_([-+]?\d+(?:\.\d+)?)mm", text)
    approach_mm = float(approach_match.group(1)) if approach_match else 0.0
    release_up_match = re.search(r"release_up_([-+]?\d+(?:\.\d+)?)mm", text)
    release_up_mm = float(release_up_match.group(1)) if release_up_match else 0.0
    is_insert = str(item.get("place_mode", "")) == "insert_place"
    return (
        0 if is_insert else 1,
        0 if abs(approach_mm - 50.0) <= 1.0 else 1,
        _yaw_distance_deg(spin_deg, 0.0),
        release_up_mm,
        _yaw_distance_deg(spin_deg, -15.0),
        0 if hover_extra <= 1e-6 else 1,
        hover_extra,
        *_final_contact_validation_sort_key(item),
    )


def _vertical_long_axis_final_contact_sort_key(item, source_name: str | None) -> tuple:
    source = str(source_name or "").lower()
    label = "" if item.get("label") is None else str(item.get("label"))
    variant = "" if item.get("variant_label") is None else str(item.get("variant_label"))
    text = f"{variant} {label}".lower()
    yaw = _variant_yaw_deg_from_labels(variant, label)
    hover_extra = float(item.get("hover_extra_height_m", 0.0) or 0.0)
    slot_text = str(item.get("slot_name", ""))
    slot_match = re.search(r"slot_(\d+)", slot_text)
    slot_idx = int(slot_match.group(1)) if slot_match else 999
    if source == "gluestick":
        yaw_pref = min(_yaw_distance_deg(yaw, 135.0), _yaw_distance_deg(yaw, -135.0))
    elif source == "hongshupian":
        if slot_idx >= 5:
            targets = (45.0, 0.0, -90.0)
        elif slot_idx == 4:
            targets = (0.0, -90.0, 45.0)
        else:
            targets = (0.0, 45.0, -90.0)
        yaw_pref = min(_yaw_distance_deg(yaw, target) for target in targets)
    else:
        yaw_pref = _yaw_distance_deg(yaw, 0.0)
    return (
        yaw_pref,
        hover_extra,
        1 if "hover_plus_60mm" in text else 0,
        float(item.get("score", 0.0) or 0.0),
        *_final_contact_validation_sort_key(item),
    )


def _joint_search_final_contact_check_limit(args, source_name: str | None, place_modes=None) -> int | None:
    """Limit expensive sequential final-contact validations after transport wins.

    Transport candidates are still generated and screened by cuRobo. This limit
    only prevents a single joint-search batch from spending minutes validating
    many hover variants one-by-one after each final contact fails.
    """
    base_limit = int(getattr(args, "joint_search_max_final_contact_checks", 4))
    modes = {str(mode) for mode in (place_modes or [])}
    if source_name == "bi" or "insert_place" in modes:
        insert_limit = int(getattr(args, "insert_joint_search_max_final_contact_checks", 2))
        if insert_limit > 0:
            return insert_limit
    if base_limit > 0:
        return base_limit
    return None


def _print_ik_error_summary(label: str, ik_records) -> None:
    records = list(ik_records or [])
    if not records:
        return
    pos = np.asarray(
        [float(x["position_error"]) for x in records if np.isfinite(float(x.get("position_error", np.nan)))],
        dtype=np.float32,
    )
    rot = np.asarray(
        [float(x["rotation_error"]) for x in records if np.isfinite(float(x.get("rotation_error", np.nan)))],
        dtype=np.float32,
    )
    if pos.size == 0 and rot.size == 0:
        return
    pos_txt = "n/a"
    rot_txt = "n/a"
    if pos.size > 0:
        pos_txt = f"min={float(np.min(pos)):.5f}, median={float(np.median(pos)):.5f}"
    if rot.size > 0:
        rot_txt = f"min={float(np.min(rot)):.5f}, median={float(np.median(rot)):.5f}"
    print(f"[curobo][diag] {label} IK error summary: pos({pos_txt}) rot({rot_txt})")


def _diagnose_failed_ik_candidates(planner, demo, args, label: str, start_q, ranked_candidates, *, topk: int = 3):
    use_topk = int(max(topk, 0))
    if use_topk <= 0:
        return
    for idx, item in enumerate(list(ranked_candidates or [])[:use_topk], start=1):
        candidate = item["candidate"]
        candidate_label = str(candidate["label"])
        planner_pose = item["planner_pose"]
        diag = planner.diagnose_pose_goal(
            start_q,
            planner_pose,
            num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
            num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
        )
        print(
            f"[curobo][diag] {label} top{idx}={candidate_label}: "
            f"standalone_ok={diag['standalone_ik_success']}, "
            f"pos_err={diag['standalone_ik_position_error']:.6f}, "
            f"rot_err={diag['standalone_ik_rotation_error']:.6f}, "
            f"goal_delta={diag['goal_translation_delta_norm']:.6f}, "
            f"motiongen_internal_ik={diag['motiongen_internal_ik_successes']}"
        )


def _print_start_state_self_collision_diagnostics(planner, start_q, *, label: str) -> None:
    start_diag = planner.diagnose_start_state_self_collision(start_q, top_k=10)
    if start_diag.get("error"):
        print(f"[curobo] {label} self-collision diagnosis failed: {start_diag['error']}")
        return
    for item in list(start_diag.get("link_pairs", []) or []):
        print(
            f"[curobo] {label} self-collision link_pair="
            f"{item['link_a']} <-> {item['link_b']} "
            f"overlap={float(item['overlap']) * 1000.0:.2f}mm"
        )
    for item in list(start_diag.get("pairs", []) or []):
        print(
            f"[curobo] {label} self-collision sphere_pair="
            f"{item['sphere_i']}({item['link_i']}) <-> {item['sphere_j']}({item['link_j']}) "
            f"overlap={float(item['overlap']) * 1000.0:.2f}mm "
            f"center_dist={float(item['center_distance']) * 1000.0:.2f}mm "
            f"threshold={float(item['threshold']) * 1000.0:.2f}mm"
        )


def _curobo_self_collision_clearance_for_q(planner, q, *, top_k: int = 5) -> dict:
    q_np = np.asarray(q, dtype=np.float32).reshape(-1)[:7]
    try:
        spheres = planner._compute_world_link_spheres(q_np)
        sphere_link_names = planner._collision_sphere_link_names()
    except Exception as exc:
        return {"success": False, "status": "ERROR", "error": str(exc)}
    spheres = np.asarray(spheres, dtype=np.float32).reshape(-1, 4)
    if len(sphere_link_names) != spheres.shape[0]:
        return {
            "success": False,
            "status": "SPHERE_LINK_COUNT_MISMATCH",
            "n_spheres": int(spheres.shape[0]),
            "n_link_names": int(len(sphere_link_names)),
        }

    ignore_pairs = planner._self_collision_ignore_pairs()
    buffer_by_link = planner._self_collision_buffers()
    records: list[dict] = []
    link_pair_best: dict[tuple[str, str], dict] = {}
    min_record = None
    for i in range(spheres.shape[0]):
        c_i = spheres[i, :3]
        r_i = float(spheres[i, 3])
        link_i = str(sphere_link_names[i])
        for j in range(i + 1, spheres.shape[0]):
            link_j = str(sphere_link_names[j])
            if link_i == link_j:
                continue
            pair_key = tuple(sorted((link_i, link_j)))
            if pair_key in ignore_pairs:
                continue
            c_j = spheres[j, :3]
            r_j = float(spheres[j, 3])
            center_dist = float(np.linalg.norm(c_i - c_j))
            threshold = (
                r_i
                + r_j
                + float(buffer_by_link.get(link_i, 0.0))
                + float(buffer_by_link.get(link_j, 0.0))
            )
            clearance = float(center_dist - threshold)
            record = {
                "sphere_i": int(i),
                "sphere_j": int(j),
                "link_i": link_i,
                "link_j": link_j,
                "clearance_m": clearance,
                "overlap_m": float(max(0.0, -clearance)),
                "center_distance_m": center_dist,
                "threshold_m": float(threshold),
            }
            if min_record is None or clearance < float(min_record["clearance_m"]):
                min_record = record
            current_best = link_pair_best.get(pair_key)
            if current_best is None or clearance < float(current_best["clearance_m"]):
                link_pair_best[pair_key] = {
                    "link_a": pair_key[0],
                    "link_b": pair_key[1],
                    "clearance_m": clearance,
                    "overlap_m": float(max(0.0, -clearance)),
                    "sphere_i": int(i),
                    "sphere_j": int(j),
                }
            records.append(record)

    if min_record is None:
        return {"success": False, "status": "NO_CHECKABLE_SPHERE_PAIRS"}
    records.sort(key=lambda x: (float(x["clearance_m"]), str(x["link_i"]), str(x["link_j"])))
    link_pair_records = sorted(
        link_pair_best.values(),
        key=lambda x: (float(x["clearance_m"]), str(x["link_a"]), str(x["link_b"])),
    )
    use_top_k = max(int(top_k), 0)
    return {
        "success": True,
        "status": "Success",
        "min_clearance_m": float(min_record["clearance_m"]),
        "min_pair": min_record,
        "pairs": records if use_top_k <= 0 else records[:use_top_k],
        "link_pairs": link_pair_records if use_top_k <= 0 else link_pair_records[:use_top_k],
        "pair_count": int(len(records)),
    }


def _audit_return_to_start_self_collision_path(planner, args, q_path, *, label: str, mode: str) -> dict:
    if planner is None or not q_path:
        return {"status": "SKIPPED_NO_PATH", "success": True}
    path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in list(q_path or [])]
    stride = int(max(getattr(args, "return_to_start_self_collision_audit_stride", 1), 1))
    indices = list(range(0, len(path), stride))
    if 0 not in indices:
        indices.insert(0, 0)
    if (len(path) - 1) not in indices:
        indices.append(len(path) - 1)
    indices = sorted(set(int(i) for i in indices if 0 <= int(i) < len(path)))

    warning_clearance = float(getattr(args, "return_to_start_self_collision_warning_clearance_m", 0.003))
    top_k = 5
    min_diag = None
    min_index = None
    overlap_count = 0
    low_clearance_count = 0
    checked_count = 0
    errors: list[str] = []
    with _CUROBO_GPU_LOCK:
        for idx in indices:
            checked_count += 1
            diag = _curobo_self_collision_clearance_for_q(planner, path[idx], top_k=top_k)
            if not bool(diag.get("success", False)):
                errors.append(str(diag.get("status", "ERROR")) + ":" + str(diag.get("error", "")))
                continue
            clearance = float(diag.get("min_clearance_m", float("inf")))
            if clearance < 0.0:
                overlap_count += 1
            if clearance < warning_clearance:
                low_clearance_count += 1
            if min_diag is None or clearance < float(min_diag.get("min_clearance_m", float("inf"))):
                min_diag = diag
                min_index = int(idx)

    if min_diag is None:
        return {
            "success": False,
            "status": "ERROR",
            "checked_waypoints": int(checked_count),
            "path_waypoints": int(len(path)),
            "stride": int(stride),
            "errors": errors[:5],
        }
    min_clearance = float(min_diag.get("min_clearance_m", 0.0))
    if min_clearance < 0.0:
        status = "OVERLAP"
        success = False
    elif min_clearance < warning_clearance:
        status = "LOW_CLEARANCE"
        success = True
    else:
        status = "CLEAR"
        success = True

    min_pair = dict(min_diag.get("min_pair") or {})
    link_pair = "?"
    if min_pair:
        link_pair = f"{min_pair.get('link_i')}<->{min_pair.get('link_j')}"
    print(
        f"[return_self_collision] {label}: status={status}, "
        f"min_clearance={min_clearance * 1000.0:.2f}mm at waypoint {min_index}, "
        f"pair={link_pair}, checked={checked_count}/{len(path)}, mode={mode}"
    )
    return {
        "success": bool(success),
        "status": status,
        "warning": bool(status == "LOW_CLEARANCE"),
        "mode": str(mode),
        "path_waypoints": int(len(path)),
        "checked_waypoints": int(checked_count),
        "stride": int(stride),
        "warning_clearance_m": warning_clearance,
        "min_clearance_m": min_clearance,
        "min_clearance_mm": float(min_clearance * 1000.0),
        "min_clearance_waypoint": min_index,
        "overlap_waypoint_count": int(overlap_count),
        "low_clearance_waypoint_count": int(low_clearance_count),
        "min_pair": min_pair,
        "top_link_pairs": list(min_diag.get("link_pairs") or []),
        "top_sphere_pairs": list(min_diag.get("pairs") or []),
        "errors": errors[:5],
    }


def _normalize_quat_wxyz(quat):
    arr = np.asarray(quat, dtype=np.float32).reshape(-1)[:4]
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (arr / norm).astype(np.float32)


def _quat_angle_rad_wxyz(q_a, q_b) -> float:
    q_a = _normalize_quat_wxyz(q_a)
    q_b = _normalize_quat_wxyz(q_b)
    dot = float(np.clip(np.abs(np.dot(q_a, q_b)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def _measure_realized_tcp_error(demo, q_arm, pose):
    q_saved = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    try:
        targeted.base.sync_demo_arm_qpos(demo, q_arm)
        actual_p = targeted.base.flatten_np(demo.tcp.pose.p)[:3].astype(np.float32)
        actual_q = _normalize_quat_wxyz(targeted.base.flatten_np(demo.tcp.pose.q)[:4])
    finally:
        targeted.base.sync_demo_arm_qpos(demo, q_saved)
    target_p = targeted.base.flatten_np(pose.p)[:3].astype(np.float32)
    target_q = _normalize_quat_wxyz(targeted.base.flatten_np(pose.q)[:4])
    pos_err = float(np.linalg.norm(actual_p - target_p))
    rot_err = _quat_angle_rad_wxyz(actual_q, target_q)
    return {
        "actual_p": actual_p,
        "actual_q": actual_q,
        "target_p": target_p,
        "target_q": target_q,
        "pos_err": pos_err,
        "rot_err_rad": rot_err,
        "rot_err_deg": float(np.degrees(rot_err)),
    }


def _quat_wxyz_to_rotmat(q_wxyz) -> np.ndarray:
    w, x, y, z = float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _tcp_pad_tilt_z(pose) -> float:
    """Return |tcp_y_axis_world[2]| — how tilted the pad opening direction is from horizontal.

    Pads open along TCP y-axis (URDF: gripper_tcp has 90-deg yaw from gripper_base_link,
    mapping the pad opening axis to TCP y).

    0.0 = pads perfectly level (both at the same height) — safe to release.
    1.0 = one pad directly above the other — object will slide off on release.
    """
    q_wxyz = _normalize_quat_wxyz(targeted.base.flatten_np(pose.q)[:4])
    R = _quat_wxyz_to_rotmat(q_wxyz)
    tcp_y_in_world = R[:, 1]
    return float(abs(tcp_y_in_world[2]))


def _tcp_axis_world_z_components(pose) -> dict[str, float]:
    q_wxyz = _normalize_quat_wxyz(targeted.base.flatten_np(pose.q)[:4])
    R = _quat_wxyz_to_rotmat(q_wxyz)
    return _rotation_axis_world_z_components(R)


def _rotation_axis_world_z_components(R) -> dict[str, float]:
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    return {
        "x": float(R[2, 0]),
        "y": float(R[2, 1]),
        "z": float(R[2, 2]),
        "abs_x": float(abs(R[2, 0])),
        "abs_y": float(abs(R[2, 1])),
        "abs_z": float(abs(R[2, 2])),
    }


def _get_pose_position(pose) -> np.ndarray:
    return targeted.base.flatten_np(pose.p)[:3].astype(np.float32)


def _copy_pose_obj(pose):
    return targeted.Pose.create_from_pq(
        p=targeted.base.flatten_np(pose.p)[:3].astype(np.float32),
        q=_normalize_quat_wxyz(targeted.base.flatten_np(pose.q)[:4]),
    )


def _get_demo_tcp_pose_for_joint_q(demo, q_arm):
    q_saved = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    try:
        targeted.base.sync_demo_arm_qpos(demo, q_arm)
        return _copy_pose_obj(demo.tcp.pose)
    finally:
        targeted.base.sync_demo_arm_qpos(demo, q_saved)


def _terminal_error_within_limits(diag, args) -> bool:
    max_pos_err = float(max(getattr(args, "direct_terminal_max_realized_pos_error", 0.010), 0.0))
    max_rot_err_deg = float(max(getattr(args, "direct_terminal_max_realized_rot_error_deg", 12.0), 0.0))
    return float(diag["pos_err"]) <= max_pos_err and float(diag["rot_err_deg"]) <= max_rot_err_deg


def _get_robot_link_by_name(demo, link_name: str):
    links_map = getattr(demo.robot, "links_map", None)
    if isinstance(links_map, dict) and link_name in links_map:
        return links_map[link_name]
    for link in list(demo.robot.get_links()):
        get_name = getattr(link, "get_name", None)
        name = str(get_name()) if callable(get_name) else str(getattr(link, "name", ""))
        if name == link_name:
            return link
    return None


def _pose_to_matrix_from_pose_obj(pose) -> np.ndarray:
    p = targeted.base.flatten_np(pose.p)[:3].astype(np.float32)
    q = targeted.base.flatten_np(pose.q)[:4].astype(np.float32)
    return targeted.base.pose_to_matrix(p, q)


def _pose_from_world_matrix(T_world_pose: np.ndarray):
    T_world_pose = np.asarray(T_world_pose, dtype=np.float32).reshape(4, 4)
    return targeted.Pose.create_from_pq(
        p=T_world_pose[:3, 3].astype(np.float32),
        q=targeted.base.bridge_mod_mat2quat(T_world_pose[:3, :3]).astype(np.float32),
    )


def _hidden_visual_pose():
    return targeted.Pose.create_from_pq(
        p=np.asarray([0.0, 0.0, -10.0], dtype=np.float32),
        q=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def _hide_actor_quiet(actor) -> None:
    if actor is None:
        return
    try:
        actor.set_pose(_hidden_visual_pose())
    except Exception:
        pass


def _ensure_target_object_goal_visual_actor(demo, args, *, source_name: str):
    asset_file = getattr(args, "sim_asset_file", None) or getattr(args, "mesh_file", None)
    asset_scale = float(getattr(args, "sim_asset_scale", None) or getattr(args, "mesh_scale", None) or 1.0)
    if not asset_file:
        raise RuntimeError("missing sim_asset_file/mesh_file for target goal ghost")
    key = (str(Path(str(asset_file)).expanduser()), float(asset_scale))
    actor = getattr(demo, "_target_object_goal_visual_actor", None)
    actor_key = getattr(demo, "_target_object_goal_visual_key", None)
    actor_mode = getattr(demo, "_target_object_goal_visual_mode", "mesh")
    if actor is not None and actor_key == key:
        return actor, actor_mode

    _hide_actor_quiet(actor)
    counter = int(getattr(demo, "_target_object_goal_visual_counter", 0)) + 1
    demo._target_object_goal_visual_counter = counter
    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(source_name or "object")).strip("_") or "object"
    actor_name = f"target_goal_ghost_{safe_name}_{counter}"
    color = (0.05, 0.8, 1.0, 0.55)
    try:
        actor = targeted.base.build_visual_obstacle_actor(
            demo.env,
            str(asset_file),
            asset_scale,
            actor_name,
            box_size=None,
            color=color,
        )
        actor_mode = "mesh"
    except Exception as exc:
        box_size = targeted.base.get_asset_box_size(str(asset_file), asset_scale)
        actor = targeted.base.build_visual_box_actor(
            demo.env,
            box_size,
            f"{actor_name}_box",
            color=color,
        )
        actor_mode = "box"
        print(f"[target_ghost] mesh visual failed for {source_name}, using box ghost: {exc}")

    demo._target_object_goal_visual_actor = actor
    demo._target_object_goal_visual_key = key
    demo._target_object_goal_visual_mode = actor_mode
    return actor, actor_mode


def _render_current_target_object_goal_visual(
    demo,
    bridge_mod,
    scene_capture_cache,
    place_state_cache,
    rule,
    args,
) -> None:
    if rule is None:
        _hide_actor_quiet(getattr(demo, "_target_object_goal_visual_actor", None))
        return
    try:
        # Identity TCP override asks the targeted-place planner only for the
        # desired object pose; it does not consume slots or affect planning.
        place_plans = targeted.build_targeted_place_plan_variants(
            demo,
            bridge_mod,
            scene_capture_cache,
            rule,
            place_state_cache,
            args,
            T_tcp_obj_override=np.eye(4, dtype=np.float32),
        )
        if not place_plans:
            raise RuntimeError("targeted place planner returned no target pose")
        plan = place_plans[0]
        actor, actor_mode = _ensure_target_object_goal_visual_actor(
            demo,
            args,
            source_name=str(getattr(rule, "source_object_name", None) or getattr(args, "object_name", "object")),
        )
        actor.set_pose(_pose_from_world_matrix(plan.T_world_obj_desired))
        target_p = np.asarray(plan.T_world_obj_desired[:3, 3], dtype=np.float32).reshape(3)
        label = str(getattr(plan, "variant_label", "") or getattr(plan, "slot_name", "") or "default")
        print(
            f"[target_ghost] rendered current target pose for {args.object_name}: "
            f"mode={actor_mode}, variant={label}, p={np.round(target_p, 4).tolist()}, alpha=0.55"
        )
    except Exception as exc:
        _hide_actor_quiet(getattr(demo, "_target_object_goal_visual_actor", None))
        print(f"[target_ghost] failed to render target pose for {getattr(args, 'object_name', 'object')}: {exc}")


def _get_robot_base_world_transform(demo) -> np.ndarray:
    return _pose_to_matrix_from_pose_obj(demo.robot.pose)


def _get_link_name(link) -> str:
    get_name = getattr(link, "get_name", None)
    if callable(get_name):
        try:
            return str(get_name())
        except Exception:
            pass
    return str(getattr(link, "name", ""))


def _convert_demo_tcp_pose_to_curobo_ee_pose(demo, pose, *, ee_link_name: str):
    ee_link = _get_robot_link_by_name(demo, ee_link_name)
    T_world_base = _get_robot_base_world_transform(demo)
    T_base_world = np.linalg.inv(T_world_base)
    T_world_demo_tcp = _pose_to_matrix_from_pose_obj(demo.tcp.pose)
    T_world_ee_link = T_world_demo_tcp if ee_link is None else _pose_to_matrix_from_pose_obj(ee_link.pose)
    T_demo_tcp_to_ee_link = np.linalg.inv(T_world_demo_tcp) @ T_world_ee_link
    T_goal_demo_tcp = _pose_to_matrix_from_pose_obj(pose)
    T_world_goal_ee_link = T_goal_demo_tcp @ T_demo_tcp_to_ee_link
    T_base_goal_ee_link = T_base_world @ T_world_goal_ee_link
    return targeted._pose_from_matrix(T_base_goal_ee_link.astype(np.float32))


def _convert_demo_tcp_pose_to_curobo_ee_pose_for_joint_q(demo, q_arm, pose, *, ee_link_name: str):
    q_saved = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    try:
        targeted.base.sync_demo_arm_qpos(demo, q_arm)
        return _convert_demo_tcp_pose_to_curobo_ee_pose(demo, pose, ee_link_name=ee_link_name)
    finally:
        targeted.base.sync_demo_arm_qpos(demo, q_saved)


def _nlerp_quat_wxyz(q0, q1, alpha: float) -> np.ndarray:
    q0 = _normalize_quat_wxyz(q0)
    q1 = _normalize_quat_wxyz(q1)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    q = (1.0 - float(alpha)) * q0 + float(alpha) * q1
    return _normalize_quat_wxyz(q)


def _interpolate_demo_tcp_pose(pose_start, pose_goal, alpha: float):
    p0 = targeted.base.flatten_np(pose_start.p)[:3].astype(np.float32)
    p1 = targeted.base.flatten_np(pose_goal.p)[:3].astype(np.float32)
    q0 = targeted.base.flatten_np(pose_start.q)[:4].astype(np.float32)
    q1 = targeted.base.flatten_np(pose_goal.q)[:4].astype(np.float32)
    p = ((1.0 - float(alpha)) * p0 + float(alpha) * p1).astype(np.float32)
    q = _nlerp_quat_wxyz(q0, q1, alpha)
    return targeted.Pose.create_from_pq(p=p, q=q)


def _infer_goal_frame_free_linear_axis(pose_start, pose_goal):
    T_world_start = targeted.base.pose_to_matrix(
        targeted.base.flatten_np(pose_start.p)[:3],
        targeted.base.flatten_np(pose_start.q)[:4],
    ).astype(np.float32)
    T_world_goal = targeted.base.pose_to_matrix(
        targeted.base.flatten_np(pose_goal.p)[:3],
        targeted.base.flatten_np(pose_goal.q)[:4],
    ).astype(np.float32)
    T_goal_start = np.linalg.inv(T_world_goal) @ T_world_start
    delta_goal = np.asarray(T_goal_start[:3, 3], dtype=np.float32).reshape(3)
    free_axis = int(np.argmax(np.abs(delta_goal)))
    locked_delta = np.delete(delta_goal, free_axis)
    rot_trace = float(np.trace(T_goal_start[:3, :3]))
    cos_theta = float(np.clip((rot_trace - 1.0) * 0.5, -1.0, 1.0))
    rot_err_deg = float(np.degrees(np.arccos(cos_theta)))
    return free_axis, delta_goal, locked_delta, rot_err_deg


def _build_official_approach_metric(
    planner,
    args,
    pose_start,
    pose_goal,
    *,
    label: str,
    tstep_fraction: float | None = None,
):
    free_axis, delta_goal, locked_delta, rot_err_deg = _infer_goal_frame_free_linear_axis(pose_start, pose_goal)
    locked_axis_tol_m = float(max(getattr(args, "curobo_approach_metric_locked_axis_tol_m", 0.012), 0.0))
    if rot_err_deg > 5.0 or float(np.max(np.abs(locked_delta))) > locked_axis_tol_m:
        print(
            f"[curobo] {label} cannot use official approach metric cleanly "
            f"(goal-frame rot_err={rot_err_deg:.2f} deg, locked_delta={np.round(locked_delta, 6)}, "
            f"tol={locked_axis_tol_m:.4f})"
        )
        return None
    offset = float(abs(delta_goal[free_axis]))
    if offset <= 1e-6:
        return None
    metric = planner.mods["PoseCostMetric"].create_grasp_approach_metric(
        offset_position=offset,
        linear_axis=free_axis,
        tstep_fraction=float(
            getattr(args, "curobo_approach_metric_tstep_fraction", 0.0)
            if tstep_fraction is None
            else tstep_fraction
        ),
        tensor_args=planner.tensor_args,
    )
    metric.project_to_goal_frame = True
    return metric, free_axis, delta_goal


def _clip_arm_q_to_joint_limits(demo, q, *, label: str, margin: float = 1e-4, max_adjust: float = 0.005) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(-1)[:7].copy()
    joint_limits = np.asarray(getattr(getattr(demo, "planner", None), "joint_limits", []), dtype=np.float32)
    if joint_limits.ndim != 2 or joint_limits.shape[0] < q.shape[0] or joint_limits.shape[1] < 2:
        return q
    q_min = joint_limits[: q.shape[0], 0] + float(margin)
    q_max = joint_limits[: q.shape[0], 1] - float(margin)
    clipped = np.minimum(np.maximum(q, q_min), q_max).astype(np.float32)
    max_delta = float(np.max(np.abs(clipped - q))) if clipped.size else 0.0
    if max_delta <= 1e-8:
        return q
    if max_delta > float(max_adjust):
        print(
            f"[curobo] {label} start_q exceeds joint limits by {max_delta:.6f} rad; "
            "not clipping because the adjustment is not a near-limit numerical correction"
        )
        return q
    print(f"[curobo] {label} clipped near-limit start_q by max {max_delta:.6f} rad before MotionGen")
    return clipped


def _plan_with_official_approach_metric(
    planner,
    demo,
    args,
    start_q,
    pose_start,
    pose_goal,
    *,
    label: str,
    tstep_fraction: float | None = None,
):
    metric_info = _build_official_approach_metric(
        planner,
        args,
        pose_start,
        pose_goal,
        label=label,
        tstep_fraction=tstep_fraction,
    )
    if metric_info is None:
        return None
    metric, free_axis, delta_goal = metric_info
    start_q = _clip_arm_q_to_joint_limits(demo, start_q, label=label)
    planner_pose = _convert_demo_tcp_pose_to_curobo_ee_pose(
        demo,
        pose_goal,
        ee_link_name=str(getattr(planner.config, "ee_link", "gripper_tcp")),
    )
    print(
        f"[curobo] {label} trying cuRobo constrained straight-line metric "
        f"(free_goal_axis={free_axis}, goal_frame_delta={np.round(delta_goal, 6)})"
    )
    result = _profile_plan_constrained_linear_to_pose(
        planner,
        start_q,
        planner_pose,
        enable_graph=False,
        max_attempts=int(getattr(args, "curobo_max_attempts", 2)),
        timeout=float(getattr(args, "curobo_timeout", 5.0)),
        num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
        num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
        num_graph_seeds=int(getattr(args, "curobo_num_graph_seeds", 1)),
        pose_cost_metric=metric,
    )
    if not result.success or result.joint_path is None:
        print(f"[curobo] {label} constrained straight-line metric failed with status={result.status}")
        return None
    q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in result.joint_path]
    final_diag = _measure_realized_tcp_error(demo, q_path[-1], pose_goal)
    print(
        f"[curobo] {label} constrained straight-line realized tcp: "
        f"pos_err={final_diag['pos_err']:.4f} m, rot_err={final_diag['rot_err_deg']:.2f} deg"
    )
    if not _terminal_error_within_limits(final_diag, args):
        print(f"[curobo] {label} constrained straight-line segment misses the target pose too much")
        return None
    return q_path


def _plan_release_with_motiongen_constraint(
    planner,
    demo,
    args,
    start_q,
    pose_start,
    pose_goal,
    *,
    label: str,
):
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    return _plan_with_official_approach_metric(
        planner,
        demo,
        args,
        start_q,
        pose_start,
        pose_goal,
        label=label,
        tstep_fraction=float(getattr(args, "final_contact_approach_metric_tstep_fraction", 0.0)),
    )


def _build_validated_linear_path_to_q(
    demo,
    args,
    start_q,
    q_goal,
    pose_start,
    pose_goal,
    *,
    label: str,
    use_attach: bool,
    max_step_rad: float | None = None,
    validation_pos_tol_m: float | None = None,
    validation_rot_tol_deg: float | None = None,
    max_backtrack_m: float | None = None,
) -> list[np.ndarray] | None:
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    q_goal = _q7_or_none(q_goal)
    if q_goal is None:
        return None
    step = float(max_step_rad if max_step_rad is not None else getattr(args, "short_linear_joint_step_rad", 0.035))
    q_path = _linear_joint_path(start_q, q_goal, max_step_rad=step)
    if len(q_path) < 2:
        return None
    if bool(getattr(args, "strict_short_linear_segments", True)) and not _validate_strict_linear_waypoints(
        demo,
        args,
        q_path,
        pose_start,
        pose_goal,
        label=label,
        max_pos_err_m=validation_pos_tol_m,
        max_rot_err_deg=validation_rot_tol_deg,
        max_backtrack_m=max_backtrack_m,
    ):
        return None
    if not _validate_candidate_joint_path_with_demo_planner(
        demo,
        start_q,
        q_path,
        use_attach=use_attach,
        label=label,
    ):
        return None
    final_diag = _measure_realized_tcp_error(demo, q_path[-1], pose_goal)
    if not _terminal_error_within_limits(final_diag, args):
        print(
            f"[curobo] {label} endpoint linear path misses target: "
            f"pos_err={final_diag['pos_err']:.4f}m, rot_err={final_diag['rot_err_deg']:.2f}deg"
        )
        return None
    return q_path


def _plan_short_linear_segment_via_goal_ik(
    planner,
    demo,
    args,
    start_q,
    pose_start,
    pose_goal,
    *,
    label: str,
    use_attach: bool,
    validation_pos_tol_m: float | None = None,
    validation_rot_tol_deg: float | None = None,
    max_backtrack_m: float | None = None,
) -> list[np.ndarray] | None:
    planner_pose = _convert_demo_tcp_pose_to_curobo_ee_pose(
        demo,
        pose_goal,
        ee_link_name=str(getattr(planner.config, "ee_link", "gripper_tcp")),
    )
    ik_result = _profile_solve_ik(
        planner,
        np.asarray(start_q, dtype=np.float32).reshape(-1)[:7],
        planner_pose,
        num_seeds=int(getattr(args, "short_linear_ik_seeds", getattr(args, "curobo_num_ik_seeds", 64))),
    )
    print(f"[curobo] {label} endpoint IK for straight primitive success={ik_result.success}")
    if not bool(ik_result.success) or ik_result.goal_joint is None:
        return None
    return _build_validated_linear_path_to_q(
        demo,
        args,
        start_q,
        ik_result.goal_joint,
        pose_start,
        pose_goal,
        label=f"{label}_endpoint_ik",
        use_attach=use_attach,
        validation_pos_tol_m=validation_pos_tol_m,
        validation_rot_tol_deg=validation_rot_tol_deg,
        max_backtrack_m=max_backtrack_m,
    )


def _plan_short_curobo_cartesian_descent(
    planner,
    demo,
    args,
    start_q,
    pose_start,
    pose_goal,
    *,
    label: str,
    force_segmented: bool = False,
    allow_motiongen_fallback: bool = True,
):
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    segmented_allowed = (
        bool(force_segmented)
        or
        bool(getattr(args, "final_contact_segmented_ik_first", False))
        or
        bool(getattr(args, "allow_segmented_ik_rescue", False))
        or bool(getattr(args, "final_contact_segmented_ik_fallback", False))
    )
    p0 = targeted.base.flatten_np(pose_start.p)[:3].astype(np.float32)
    p1 = targeted.base.flatten_np(pose_goal.p)[:3].astype(np.float32)
    distance = float(np.linalg.norm(p1 - p0))
    step_m = float(max(getattr(args, "direct_release_cartesian_step_m", 0.005), 1e-3))
    num_segments = max(2, int(np.ceil(distance / step_m)))

    def _try_segmented_descent(reason: str):
        print(
            f"[curobo] {label} constrained cartesian descent {reason} with {num_segments} segments "
            f"(distance={distance:.4f} m)"
        )
        q_prev = start_q.copy()
        q_path = [q_prev.copy()]
        max_joint_delta = 0.20
        max_joint7_delta = 0.25
        max_norm_delta = 0.35
        for seg_idx, alpha in enumerate(np.linspace(0.0, 1.0, num_segments + 1)[1:], start=1):
            interp_pose = _interpolate_demo_tcp_pose(pose_start, pose_goal, float(alpha))
            planner_pose = _convert_demo_tcp_pose_to_curobo_ee_pose(
                demo,
                interp_pose,
                ee_link_name=str(getattr(planner.config, "ee_link", "gripper_tcp")),
            )
            ik_result = _profile_solve_ik(
                planner,
                q_prev,
                planner_pose,
                num_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
            )
            print(f"[curobo] {label} segment={seg_idx}/{num_segments} ik_success={ik_result.success}")
            if not ik_result.success or ik_result.goal_joint is None:
                return None
            q_next = np.asarray(ik_result.goal_joint, dtype=np.float32).reshape(-1)[:7]
            dq = np.abs(q_next - q_prev)
            if (
                float(np.max(dq)) > max_joint_delta
                or float(dq[6]) > max_joint7_delta
                or float(np.linalg.norm(q_next - q_prev)) > max_norm_delta
            ):
                print(
                    f"[curobo] {label} segment={seg_idx}/{num_segments} rejected non-local IK step "
                    f"(max_joint_delta={float(np.max(dq)):.3f}, "
                    f"joint7_delta={float(dq[6]):.3f}, "
                    f"norm_delta={float(np.linalg.norm(q_next - q_prev)):.3f})"
                )
                return None
            q_path.append(q_next)
            q_prev = q_next

        final_diag = _measure_realized_tcp_error(demo, q_path[-1], pose_goal)
        print(
            f"[curobo] {label} realized tcp after constrained descent: "
            f"pos_err={final_diag['pos_err']:.4f} m, rot_err={final_diag['rot_err_deg']:.2f} deg"
        )
        if not _terminal_error_within_limits(final_diag, args):
            print(f"[curobo] {label} constrained descent misses the release pose too much")
            return None
        return q_path

    segmented_first = bool(force_segmented) or (
        segmented_allowed and bool(getattr(args, "final_contact_segmented_ik_first", False))
    )
    if segmented_first:
        segmented_path = _try_segmented_descent("first")
        if segmented_path is not None:
            return segmented_path
        if not bool(allow_motiongen_fallback):
            return None

    motiongen_path = _plan_release_with_motiongen_constraint(
        planner,
        demo,
        args,
        start_q,
        pose_start,
        pose_goal,
        label=label,
    )
    if motiongen_path is not None:
        return motiongen_path

    if not segmented_allowed:
        print(
            f"[curobo] {label} MotionGen descent failed; segmented IK rescue disabled "
            "(enable --allow-segmented-ik-rescue to use it)"
        )
        return None
    if not segmented_first:
        return _try_segmented_descent("fallback")
    return None


def _validate_strict_linear_waypoints(
    demo,
    args,
    q_path,
    pose_start,
    pose_goal,
    *,
    label: str,
    max_pos_err_m: float | None = None,
    max_rot_err_deg: float | None = None,
    max_backtrack_m: float | None = None,
) -> bool:
    q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in list(q_path or [])]
    if len(q_path) < 2:
        print(f"[curobo] {label} strict linear validation failed: empty waypoint path")
        return False
    p0 = _get_pose_position(pose_start)
    p1 = _get_pose_position(pose_goal)
    delta = (p1 - p0).astype(np.float32)
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-6:
        return True
    axis = delta / distance
    if max_pos_err_m is None:
        max_pos_err = float(max(getattr(args, "strict_final_contact_waypoint_pos_tol_m", 0.006), 0.0))
    else:
        max_pos_err = float(max(max_pos_err_m, 0.0))
    if max_rot_err_deg is None:
        max_rot_err = float(max(getattr(args, "strict_final_contact_waypoint_rot_tol_deg", 8.0), 0.0))
    else:
        max_rot_err = float(max(max_rot_err_deg, 0.0))
    # cuRobo's approach metric can add a tiny retreat along the commanded line
    # before descending.  That is still a straight-line contact approach and is
    # safer than lateral drift; keep the allowance small and tied to the line
    # tolerance so visibly curved paths are still rejected.
    if max_backtrack_m is None:
        max_backtrack = max(0.5 * max_pos_err, 0.003)
    else:
        max_backtrack = float(max(max_backtrack_m, 0.0))
    worst_pos = 0.0
    worst_rot = 0.0
    worst_i = 0
    last_progress = -max_backtrack
    for idx, q in enumerate(q_path):
        diag = _measure_realized_tcp_error(demo, q, pose_goal)
        actual_p = np.asarray(diag["actual_p"], dtype=np.float32).reshape(3)
        progress = float(np.dot(actual_p - p0, axis))
        clipped_progress = float(np.clip(progress, 0.0, distance))
        closest = p0 + axis * clipped_progress
        pos_err = float(np.linalg.norm(actual_p - closest))
        rot_err = float(diag["rot_err_deg"])
        if pos_err > worst_pos or rot_err > worst_rot:
            worst_pos = max(worst_pos, pos_err)
            worst_rot = max(worst_rot, rot_err)
            worst_i = idx
        outside = max(0.0, -progress, progress - distance)
        if outside > max_pos_err or progress + max_backtrack < last_progress:
            print(
                f"[curobo] {label} strict linear validation failed at waypoint {idx}/{len(q_path) - 1}: "
                f"progress={progress:.4f}m, outside_segment={outside:.4f}m, "
                f"last_progress={last_progress:.4f}m"
            )
            return False
        if pos_err > max_pos_err or rot_err > max_rot_err:
            print(
                f"[curobo] {label} strict linear validation failed at waypoint {idx}/{len(q_path) - 1}: "
                f"line_err={pos_err:.4f}m (limit={max_pos_err:.4f}), "
                f"rot_err={rot_err:.2f}deg (limit={max_rot_err:.2f})"
            )
            return False
        last_progress = max(last_progress, progress)
    print(
        f"[curobo] {label} strict linear validation passed: "
        f"waypoints={len(q_path)}, distance={distance:.4f}m, worst_line_err={worst_pos:.4f}m, "
        f"worst_rot_err={worst_rot:.2f}deg at waypoint={worst_i}"
    )
    return True


def _plan_constrained_linear_segment(
    planner,
    demo,
    args,
    start_q,
    pose_start,
    pose_goal,
    *,
    label: str,
    validate: bool = True,
    validation_pos_tol_m: float | None = None,
    validation_rot_tol_deg: float | None = None,
):
    q_path = None
    if bool(getattr(args, "short_linear_endpoint_ik_first", True)):
        q_path = _plan_short_linear_segment_via_goal_ik(
            planner,
            demo,
            args,
            start_q,
            pose_start,
            pose_goal,
            label=f"{label}_straight_ik",
            use_attach=False,
            validation_pos_tol_m=validation_pos_tol_m,
            validation_rot_tol_deg=validation_rot_tol_deg,
        )
        if q_path is not None:
            print(f"[curobo] {label} using endpoint-IK straight primitive ({len(q_path)} waypoint(s))")
            return q_path

    q_path = _plan_with_official_approach_metric(
        planner,
        demo,
        args,
        start_q,
        pose_start,
        pose_goal,
        label=label,
    )
    if (
        q_path is not None
        and bool(validate)
        and bool(getattr(args, "strict_short_linear_segments", True))
        and not _validate_strict_linear_waypoints(
            demo,
            args,
            q_path,
            pose_start,
            pose_goal,
            label=label,
            max_pos_err_m=validation_pos_tol_m,
            max_rot_err_deg=validation_rot_tol_deg,
        )
    ):
        return None
    return q_path


def _plan_short_world_z_lift_ik(
    planner,
    demo,
    args,
    start_q,
    *,
    lift_m: float,
    label: str,
    include_table: bool = True,
    exclude_object_names: set[str] | None = None,
    extra_scene_obstacles: list[dict] | None = None,
    disabled_world_collision_links: list[str] | None = None,
):
    lift_m = float(max(lift_m, 0.0))
    if lift_m <= 1e-5:
        return None
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    targeted.base.sync_demo_arm_qpos(demo, start_q)
    pose_start = demo.tcp.pose
    pose_goal = _lift_pose_world_z(pose_start, lift_m)
    _refresh_curobo_world(
        planner,
        demo,
        args,
        label=f"{label}_world",
        include_active_object=False,
        include_table=include_table,
        exclude_object_names=exclude_object_names,
        extra_scene_obstacles=extra_scene_obstacles,
    )
    requested_disabled_links = _normalize_disabled_world_collision_links(planner, disabled_world_collision_links)
    restore_attached_spheres = "attached_object" in set(requested_disabled_links or [])
    if restore_attached_spheres:
        _cache_attached_spheres_for_contact(planner)
    disabled_world_collision_links = _set_world_collision_for_links(
        planner,
        requested_disabled_links,
        enabled=False,
        label=label,
    )
    q_path = None
    try:
        q_path = _plan_constrained_linear_segment(
            planner,
            demo,
            args,
            start_q,
            pose_start,
            pose_goal,
            label=label,
            validation_pos_tol_m=float(max(getattr(args, "strict_short_linear_waypoint_pos_tol_m", 0.010), 0.0)),
        )
        if q_path is None and not bool(getattr(args, "strict_short_linear_segments", True)):
            planner_pose = _convert_demo_tcp_pose_to_curobo_ee_pose(
                demo,
                pose_goal,
                ee_link_name=str(getattr(planner.config, "ee_link", "gripper_tcp")),
            )
            result = _profile_plan_to_pose(
                planner,
                start_q,
                planner_pose,
                enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
                max_attempts=max(int(getattr(args, "curobo_max_attempts", 2)), 2),
                timeout=max(float(getattr(args, "curobo_timeout", 5.0)), 3.0),
                num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
                num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
                num_graph_seeds=int(getattr(args, "curobo_num_graph_seeds", 1)),
            )
            if result.success and result.joint_path is not None:
                q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in result.joint_path]
                print(f"[joint_search] {label}: cuRobo lift MotionGen success with {len(q_path)} waypoint(s)")
            else:
                print(f"[joint_search] {label}: cuRobo lift MotionGen failed with status={getattr(result, 'status', None)}")
        elif q_path is None:
            print(f"[joint_search] {label}: constrained cuRobo lift failed; unconstrained MotionGen fallback disabled")

        if q_path is None:
            source_name = _current_source_object_name(args)
            allow_segmented_lift = bool(getattr(args, "allow_segmented_ik_rescue", False))
            if not allow_segmented_lift:
                print(
                    f"[joint_search] {label}: skipped segmented IK lift rescue "
                    "(enable --allow-segmented-ik-rescue to use it)"
                )
                return None
            p0 = targeted.base.flatten_np(pose_start.p)[:3].astype(np.float32)
            p1 = targeted.base.flatten_np(pose_goal.p)[:3].astype(np.float32)
            distance = float(np.linalg.norm(p1 - p0))
            step_m = float(max(getattr(args, "direct_release_cartesian_step_m", 0.005), 1e-3))
            if source_name == "bi" and str(label).startswith("joint_start_lift_"):
                step_m = max(step_m, 0.010)
            num_segments = max(2, int(np.ceil(distance / step_m)))
            print(
                f"[joint_search] {label}: trying segmented IK lift rescue "
                f"dz={lift_m:.3f}m with {num_segments} segment(s)"
            )
            q_prev = start_q.copy()
            q_path = [q_prev.copy()]
            max_joint_delta = 0.20
            max_joint7_delta = 0.30
            max_norm_delta = 0.40
            for seg_idx, alpha in enumerate(np.linspace(0.0, 1.0, num_segments + 1)[1:], start=1):
                interp_p = p0 + (p1 - p0) * float(alpha)
                interp_pose = targeted.base.make_pose_with_position(pose_start, interp_p.astype(np.float32))
                planner_pose = _convert_demo_tcp_pose_to_curobo_ee_pose(
                    demo,
                    interp_pose,
                    ee_link_name=str(getattr(planner.config, "ee_link", "gripper_tcp")),
                )
                ik_result = _profile_solve_ik(
                    planner,
                    q_prev,
                    planner_pose,
                    num_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
                )
                print(f"[joint_search] {label}: lift segment={seg_idx}/{num_segments} ik_success={ik_result.success}")
                if not ik_result.success or ik_result.goal_joint is None:
                    if seg_idx == 1:
                        goal_planner_pose = _convert_demo_tcp_pose_to_curobo_ee_pose(
                            demo,
                            pose_goal,
                            ee_link_name=str(getattr(planner.config, "ee_link", "gripper_tcp")),
                        )
                        goal_ik = _profile_solve_ik(
                            planner,
                            start_q,
                            goal_planner_pose,
                            num_seeds=max(int(getattr(args, "curobo_num_ik_seeds", 64)), 128),
                        )
                        print(
                            f"[joint_search] {label}: first lift segment failed; "
                            f"direct lifted-goal ik_success={goal_ik.success}"
                        )
                        if goal_ik.success and goal_ik.goal_joint is not None:
                            q_goal = np.asarray(goal_ik.goal_joint, dtype=np.float32).reshape(-1)[:7]
                            q_path = [start_q.copy(), q_goal]
                            break
                    return None
                q_next = np.asarray(ik_result.goal_joint, dtype=np.float32).reshape(-1)[:7]
                dq = np.abs(q_next - q_prev)
                if (
                    float(np.max(dq)) > max_joint_delta
                    or float(dq[6]) > max_joint7_delta
                    or float(np.linalg.norm(q_next - q_prev)) > max_norm_delta
                ):
                    print(
                        f"[joint_search] {label}: lift segment={seg_idx}/{num_segments} rejected non-local IK step "
                        f"(max_joint_delta={float(np.max(dq)):.3f}, "
                        f"joint7_delta={float(dq[6]):.3f}, norm_delta={float(np.linalg.norm(q_next - q_prev)):.3f})"
                    )
                    return None
                q_path.append(q_next)
                q_prev = q_next
    finally:
        _set_world_collision_for_links(
            planner,
            disabled_world_collision_links,
            enabled=True,
            label=label,
        )
        if restore_attached_spheres:
            _restore_attached_spheres_after_contact(planner)

    if q_path is None:
        return None
    if not _validate_candidate_joint_path_with_demo_planner(
        demo,
        start_q,
        q_path,
        use_attach=True,
        label=f"{label}_lift",
    ):
        return None
    return q_path


def _plan_short_grasp_approach_retreat_ik(
    planner,
    demo,
    args,
    start_q,
    grasp_choice,
    *,
    label: str,
    include_table: bool = True,
    exclude_object_names: set[str] | None = None,
    disabled_world_collision_links: list[str] | None = None,
):
    """Retreat from the grasp pose back along the validated pregrasp->grasp line."""

    if grasp_choice is None:
        return None
    pose_start = grasp_choice.get("deferred_grasp_pose", grasp_choice.get("pose"))
    pose_goal = grasp_choice.get("deferred_pregrasp_pose", grasp_choice.get("pregrasp_pose"))
    if pose_start is None or pose_goal is None:
        print(f"[joint_search] {label}: no deferred pregrasp pose available for approach-line retreat")
        return None

    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    targeted.base.sync_demo_arm_qpos(demo, start_q)
    p0 = _get_pose_position(pose_start)
    p1 = _get_pose_position(pose_goal)
    distance = float(np.linalg.norm(p1 - p0))
    if distance <= 1e-5:
        print(f"[joint_search] {label}: approach-line retreat distance is too small")
        return None

    _refresh_curobo_world(
        planner,
        demo,
        args,
        label=f"{label}_world",
        include_active_object=False,
        include_table=include_table,
        exclude_object_names=exclude_object_names,
    )
    requested_disabled_links = _normalize_disabled_world_collision_links(planner, disabled_world_collision_links)
    restore_attached_spheres = "attached_object" in set(requested_disabled_links or [])
    if restore_attached_spheres:
        _cache_attached_spheres_for_contact(planner)
    disabled_world_collision_links = _set_world_collision_for_links(
        planner,
        requested_disabled_links,
        enabled=False,
        label=label,
    )
    q_path = None
    try:
        print(
            f"[joint_search] {label}: trying cuRobo constrained retreat along grasp approach line "
            f"(distance={distance:.3f}m)"
        )
        q_path = _plan_constrained_linear_segment(
            planner,
            demo,
            args,
            start_q,
            pose_start,
            pose_goal,
            label=label,
            validation_pos_tol_m=float(max(getattr(args, "strict_short_linear_waypoint_pos_tol_m", 0.010), 0.0)),
        )
    finally:
        _set_world_collision_for_links(
            planner,
            disabled_world_collision_links,
            enabled=True,
            label=label,
        )
        if restore_attached_spheres:
            _restore_attached_spheres_after_contact(planner)

    if q_path is None:
        print(f"[joint_search] {label}: constrained approach-line retreat failed")
        return None
    if not _validate_candidate_joint_path_with_demo_planner(
        demo,
        start_q,
        q_path,
        use_attach=True,
        label=f"{label}_approach_retreat",
    ):
        return None
    return q_path


def _plan_short_tcp_up_axis_lift_ik(
    planner,
    demo,
    args,
    start_q,
    grasp_choice,
    *,
    lift_m: float,
    label: str,
    include_table: bool = True,
    exclude_object_names: set[str] | None = None,
    disabled_world_collision_links: list[str] | None = None,
):
    """Lift along the TCP axis that points most upward, keeping PoseCostMetric single-axis."""

    pose_start = None if grasp_choice is None else grasp_choice.get("deferred_grasp_pose", grasp_choice.get("pose"))
    if pose_start is None:
        targeted.base.sync_demo_arm_qpos(demo, np.asarray(start_q, dtype=np.float32).reshape(-1)[:7])
        pose_start = demo.tcp.pose

    lift_m = float(max(lift_m, 0.0))
    if lift_m <= 1e-5:
        return None
    T_start = targeted.base.pose_to_matrix(
        targeted.base.flatten_np(pose_start.p)[:3],
        targeted.base.flatten_np(pose_start.q)[:4],
    ).astype(np.float32)
    axes = T_start[:3, :3]
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dots = axes.T @ world_up
    axis_idx = int(np.argmax(np.abs(dots)))
    signed_axis = axes[:, axis_idx].astype(np.float32)
    axis_dot = float(dots[axis_idx])
    if axis_dot < 0.0:
        signed_axis = -signed_axis
        axis_dot = -axis_dot
    if axis_dot <= 1e-4:
        print(f"[joint_search] {label}: no TCP axis has upward component for single-axis lift")
        return None
    distance = lift_m / max(axis_dot, 0.35)
    distance = float(min(max(distance, lift_m), max(lift_m + 0.03, 0.12)))
    p_start = _get_pose_position(pose_start)
    pose_goal = targeted.base.make_pose_with_position(
        pose_start,
        (p_start + signed_axis * distance).astype(np.float32),
    )

    _refresh_curobo_world(
        planner,
        demo,
        args,
        label=f"{label}_world",
        include_active_object=False,
        include_table=include_table,
        exclude_object_names=exclude_object_names,
    )
    requested_disabled_links = _normalize_disabled_world_collision_links(planner, disabled_world_collision_links)
    restore_attached_spheres = "attached_object" in set(requested_disabled_links or [])
    if restore_attached_spheres:
        _cache_attached_spheres_for_contact(planner)
    disabled_world_collision_links = _set_world_collision_for_links(
        planner,
        requested_disabled_links,
        enabled=False,
        label=label,
    )
    q_path = None
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    try:
        print(
            f"[joint_search] {label}: trying cuRobo constrained TCP-up lift "
            f"(axis={axis_idx}, axis_world_z={axis_dot:.3f}, distance={distance:.3f}m, "
            f"world_z_gain={distance * axis_dot:.3f}m)"
        )
        q_path = _plan_constrained_linear_segment(
            planner,
            demo,
            args,
            start_q,
            pose_start,
            pose_goal,
            label=label,
            validation_pos_tol_m=float(max(getattr(args, "strict_short_linear_waypoint_pos_tol_m", 0.010), 0.0)),
        )
    finally:
        _set_world_collision_for_links(
            planner,
            disabled_world_collision_links,
            enabled=True,
            label=label,
        )
        if restore_attached_spheres:
            _restore_attached_spheres_after_contact(planner)

    if q_path is None:
        print(f"[joint_search] {label}: constrained TCP-up lift failed")
        return None
    if not _validate_candidate_joint_path_with_demo_planner(
        demo,
        start_q,
        q_path,
        use_attach=True,
        label=f"{label}_tcp_up_lift",
    ):
        return None
    return q_path


def _single_obstacle_start_collision_relief(planner, start_q, *, already_excluded: set[str] | None = None) -> str | None:
    try:
        start_diag = planner.diagnose_start_state_world_collision(np.asarray(start_q, dtype=np.float32).reshape(-1)[:7])
    except Exception:
        return None
    already_excluded = set(already_excluded or [])
    candidates = []
    for item in list(start_diag.get("ablation", []) or []):
        if not bool(item.get("valid", False)):
            continue
        removed = str(item.get("removed", "") or "")
        if not removed.startswith("scene_obstacle_"):
            continue
        obstacle_name = removed.removeprefix("scene_obstacle_")
        if obstacle_name in already_excluded:
            continue
        candidates.append(obstacle_name)
    if not candidates:
        return None
    return sorted(candidates)[0]


def _validate_candidate_joint_path_with_demo_planner(demo, start_q, q_path, *, use_attach: bool, label: str) -> bool:
    if not bool(getattr(demo.args, "curobo_demo_path_validation", False)):
        return True
    dense_validate_delta = 0.01 if use_attach else 0.03
    ok = targeted.base.validate_joint_path_segments(
        demo,
        start_q,
        q_path,
        use_attach=use_attach,
        label=f"{label}_demo_validate",
        max_delta=dense_validate_delta,
    )
    if not ok:
        validation_mode = "attached-box" if use_attach else "scene"
        print(f"[curobo] {label} rejected by demo planner {validation_mode} collision validation")
    return bool(ok)


def _start_state_is_world_collision(
    planner,
    demo,
    args,
    start_q,
    *,
    label: str,
    include_table: bool,
    exclude_object_names: set[str] | None,
    disabled_world_collision_links: list[str] | None,
) -> bool:
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    _refresh_curobo_world(
        planner,
        demo,
        args,
        label=f"{label}_start_precheck_world",
        include_active_object=False,
        include_table=include_table,
        exclude_object_names=exclude_object_names,
    )
    disabled = _set_world_collision_for_links(
        planner,
        disabled_world_collision_links,
        enabled=False,
        label=f"{label}_start_precheck",
    )
    try:
        diag = planner.diagnose_start_state_world_collision(start_q)
    except Exception as exc:
        print(f"[curobo] {label}: start-state world-collision precheck failed: {exc}")
        return False
    finally:
        _set_world_collision_for_links(
            planner,
            disabled,
            enabled=True,
            label=f"{label}_start_precheck",
        )
    valid = bool(diag.get("valid", False))
    status = str(diag.get("status", ""))
    if (not valid) and status == "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION":
        print(f"[curobo] {label}: start state is already in world collision; skipping direct transport batch")
        return True
    return False


def _plan_return_to_start_joint_curobo(
    planner,
    demo,
    args,
    start_q,
    goal_q,
    *,
    label: str,
    extra_scene_obstacles: list[dict] | None = None,
) -> dict:
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    goal_q = np.asarray(goal_q, dtype=np.float32).reshape(-1)[:7]
    if planner is None:
        return {"success": False, "status": "NO_PLANNER", "q_path": None}
    _refresh_curobo_world(planner, demo, args, label=label, extra_scene_obstacles=extra_scene_obstacles)
    result = _profile_plan_to_joint_state(
        planner,
        start_q,
        goal_q,
        enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
        max_attempts=int(getattr(args, "curobo_max_attempts", 2)),
        timeout=float(getattr(args, "curobo_timeout", 5.0)),
        num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
        num_graph_seeds=int(getattr(args, "curobo_num_graph_seeds", 1)),
    )
    payload = {
        "success": False,
        "status": str(getattr(result, "status", "")),
        "q_path": None,
        "solve_time": float(getattr(result, "solve_time", 0.0) or 0.0),
        "trajopt_time": float(getattr(result, "trajopt_time", 0.0) or 0.0),
    }
    if result.success and result.joint_path is not None:
        q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in result.joint_path]
        q_final = q_path[-1]
        joint_errors = np.abs(q_final - goal_q)
        max_joint_error_per_joint = float(getattr(args, "return_to_start_max_joint_error_rad", 0.1))
        max_error = float(np.max(joint_errors))
        payload.update(
            {
                "path_waypoints": len(q_path),
                "max_joint_error": max_error,
                "q_path": q_path if max_error <= max_joint_error_per_joint else None,
                "success": max_error <= max_joint_error_per_joint,
                "status": "Success" if max_error <= max_joint_error_per_joint else "JOINT_ERROR_TOO_LARGE",
            }
        )
        if max_error > max_joint_error_per_joint:
            worst_joint_idx = int(np.argmax(joint_errors))
            payload["worst_joint_idx"] = worst_joint_idx
            payload["worst_joint_error"] = float(joint_errors[worst_joint_idx])
    return payload


def _concat_joint_paths(*paths) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for path in paths:
        for q in list(path or []):
            q_arr = np.asarray(q, dtype=np.float32).reshape(-1)[:7].copy()
            if out and np.allclose(out[-1], q_arr, atol=1e-6, rtol=0.0):
                continue
            out.append(q_arr)
    return out


def _plan_return_to_start_prelift_rescue_curobo(
    planner,
    demo,
    args,
    start_q,
    goal_q,
    *,
    prelift_planner_pose,
    lift_m: float,
    label: str,
    extra_scene_obstacles: list[dict] | None = None,
) -> dict:
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    goal_q = np.asarray(goal_q, dtype=np.float32).reshape(-1)[:7]
    if planner is None:
        return {"success": False, "status": "NO_PLANNER", "q_path": None, "mode": "prelift_rescue"}
    if prelift_planner_pose is None:
        return {"success": False, "status": "NO_PRELIFT_POSE", "q_path": None, "mode": "prelift_rescue"}

    prelift_path = None
    q_lift = None
    with _profile_stage(
        args,
        "return_to_start_preplan_prelift_ik",
        lift_m=float(lift_m),
        num_ik_seeds=int(getattr(args, "short_linear_ik_seeds", getattr(args, "curobo_num_ik_seeds", 64))),
    ) as prof:
        _refresh_curobo_world(
            planner,
            demo,
            args,
            label=f"{label}_prelift_ik_world",
            include_active_object=False,
            include_table=bool(getattr(args, "curobo_table_collision", True)),
            extra_scene_obstacles=extra_scene_obstacles,
        )
        disabled = _set_world_collision_for_links(
            planner,
            _direct_place_contact_tolerant_disabled_links(planner),
            enabled=False,
            label=f"{label}_prelift_ik",
        )
        try:
            ik_result = _profile_solve_ik(
                planner,
                start_q,
                prelift_planner_pose,
                num_seeds=int(getattr(args, "short_linear_ik_seeds", getattr(args, "curobo_num_ik_seeds", 64))),
            )
        finally:
            _set_world_collision_for_links(
                planner,
                disabled,
                enabled=True,
                label=f"{label}_prelift_ik",
            )
        prof["raw_status"] = str(getattr(ik_result, "status", ""))
        prof["solve_time"] = float(getattr(ik_result, "solve_time", 0.0) or 0.0)
        prof["ik_success_count"] = (
            getattr(getattr(ik_result, "debug", None), "get", lambda *_: None)("ik_success_count")
            if isinstance(getattr(ik_result, "debug", None), dict)
            else None
        )
        if not bool(ik_result.success) or ik_result.goal_joint is None:
            prof["success"] = False
            prof["status"] = str(getattr(ik_result, "status", "IK_FAIL"))
            prof["path_waypoints"] = 0
            return {
                "success": False,
                "status": str(getattr(ik_result, "status", "IK_FAIL")),
                "q_path": None,
                "mode": "prelift_rescue",
            }
        q_lift = np.asarray(ik_result.goal_joint, dtype=np.float32).reshape(-1)[:7]
        prelift_path = _linear_joint_path(
            start_q,
            q_lift,
            max_step_rad=float(getattr(args, "short_linear_joint_step_rad", 0.035)),
        )
        prof["success"] = bool(prelift_path)
        prof["status"] = "Success" if prelift_path else "EMPTY_PRELIFT_PATH"
        prof["path_waypoints"] = len(prelift_path or [])
        prof["max_joint_delta"] = float(np.max(np.abs(q_lift - start_q)))
    if not prelift_path or q_lift is None:
        return {"success": False, "status": "EMPTY_PRELIFT_PATH", "q_path": None, "mode": "prelift_rescue"}

    with _profile_stage(
        args,
        "return_to_start_preplan_after_prelift_joint_plan",
        enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
        num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
    ) as prof:
        return_payload = _plan_return_to_start_joint_curobo(
            planner,
            demo,
            args,
            q_lift,
            goal_q,
            label=f"{label}_after_prelift",
            extra_scene_obstacles=extra_scene_obstacles,
        )
        prof["success"] = bool(return_payload.get("success", False))
        prof["status"] = str(return_payload.get("status", "FAILED"))
        prof["path_waypoints"] = int(return_payload.get("path_waypoints", 0) or 0)
        prof["max_joint_error"] = return_payload.get("max_joint_error")
        prof["solve_time"] = return_payload.get("solve_time")
        prof["trajopt_time"] = return_payload.get("trajopt_time")
    if not bool(return_payload.get("success", False)) or not return_payload.get("q_path"):
        return {
            "success": False,
            "status": str(return_payload.get("status", "FAILED")),
            "q_path": None,
            "mode": "prelift_rescue",
            "prelift_waypoints": len(prelift_path or []),
            "solve_time": return_payload.get("solve_time"),
            "trajopt_time": return_payload.get("trajopt_time"),
        }
    q_path = _concat_joint_paths(prelift_path, return_payload.get("q_path"))
    return {
        "success": True,
        "status": "Success",
        "q_path": q_path,
        "mode": "prelift_rescue",
        "path_waypoints": len(q_path),
        "prelift_waypoints": len(prelift_path or []),
        "return_waypoints": int(return_payload.get("path_waypoints", 0) or 0),
        "max_joint_error": return_payload.get("max_joint_error"),
        "solve_time": return_payload.get("solve_time"),
        "trajopt_time": return_payload.get("trajopt_time"),
    }


def _start_return_to_start_preplan(
    demo,
    planner,
    args,
    start_q,
    goal_q,
    *,
    prelift_planner_pose=None,
    prelift_lift_m: float = 0.0,
    extra_scene_obstacles: list[dict] | None = None,
) -> None:
    if not bool(getattr(args, "return_to_start_preplan", True)):
        return
    if bool(getattr(args, "_planning_prefetch_capture_only", False)):
        return
    if isinstance(getattr(args, "_return_to_start_preplan_state", None), dict):
        _record_profile(
            args,
            "return_to_start_preplan_start",
            success=True,
            status="ALREADY_STARTED",
        )
        return
    preplan_start_q = _q7_or_none(start_q)
    goal_q = _q7_or_none(goal_q)
    if preplan_start_q is None or goal_q is None or planner is None:
        _record_profile(
            args,
            "return_to_start_preplan_start",
            success=False,
            status="INVALID_INPUT",
        )
        return
    state = {
        "lock": threading.Lock(),
        "result": None,
        "start_q": preplan_start_q.copy(),
        "goal_q": goal_q.copy(),
        "prelift_enabled": bool(prelift_planner_pose is not None and prelift_lift_m > 1e-5),
        "extra_scene_obstacles": _copy_scene_obstacle_entries(extra_scene_obstacles),
        "started_ts": time.time(),
    }
    prelift_first = (
        bool(getattr(args, "return_to_start_preplan_prelift_first", True))
        and bool(state["prelift_enabled"])
        and bool(getattr(args, "return_to_start_preplan_prelift", True))
    )
    args._return_to_start_preplan_state = state
    _record_profile(
        args,
        "return_to_start_preplan_start",
        success=True,
        status="STARTED",
        start_goal_delta=float(np.max(np.abs(preplan_start_q - goal_q))),
        prelift_enabled=bool(prelift_planner_pose is not None and prelift_lift_m > 1e-5),
        prelift_first=bool(prelift_first),
        prelift_lift_m=float(prelift_lift_m),
        extra_scene_obstacle_count=len(state["extra_scene_obstacles"]),
        extra_scene_obstacle_names=[
            str(item.get("object_name") or item.get("actor_name") or "")
            for item in list(state["extra_scene_obstacles"] or [])
        ],
    )

    def _worker() -> None:
        worker_start = time.perf_counter()
        status = "FAILED"
        error_text = None
        result_payload = None
        direct_status = None
        try:
            with _profile_record_context(is_return_to_start_preplan=True):
                if prelift_first:
                    direct_status = "SKIPPED_PRELIFT_FIRST"
                    result_payload = _plan_return_to_start_prelift_rescue_curobo(
                        planner,
                        demo,
                        args,
                        preplan_start_q,
                        goal_q,
                        prelift_planner_pose=prelift_planner_pose,
                        lift_m=float(prelift_lift_m),
                        label="return_to_cycle_start_preplan",
                        extra_scene_obstacles=state["extra_scene_obstacles"],
                    )
                    status = str(result_payload.get("status", "FAILED"))
                else:
                    with _profile_stage(
                        args,
                        "return_to_start_preplan_joint_plan",
                        start_goal_delta=float(np.max(np.abs(preplan_start_q - goal_q))),
                        enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
                        num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
                    ) as prof:
                        result_payload = _plan_return_to_start_joint_curobo(
                            planner,
                            demo,
                            args,
                            preplan_start_q,
                            goal_q,
                            label="return_to_cycle_start_preplan",
                            extra_scene_obstacles=state["extra_scene_obstacles"],
                        )
                        status = str(result_payload.get("status", "FAILED"))
                        direct_status = status
                        prof["success"] = bool(result_payload.get("success", False))
                        prof["status"] = status
                        prof["path_waypoints"] = int(result_payload.get("path_waypoints", 0) or 0)
                        prof["max_joint_error"] = result_payload.get("max_joint_error")
                        prof["solve_time"] = result_payload.get("solve_time")
                        prof["trajopt_time"] = result_payload.get("trajopt_time")
                if (
                    not prelift_first
                    and not bool(result_payload and result_payload.get("success", False))
                    and bool(getattr(args, "return_to_start_preplan_prelift", True))
                    and prelift_planner_pose is not None
                    and float(prelift_lift_m) > 1e-5
                ):
                    rescue_payload = _plan_return_to_start_prelift_rescue_curobo(
                        planner,
                        demo,
                        args,
                        preplan_start_q,
                        goal_q,
                        prelift_planner_pose=prelift_planner_pose,
                        lift_m=float(prelift_lift_m),
                        label="return_to_cycle_start_preplan",
                        extra_scene_obstacles=state["extra_scene_obstacles"],
                    )
                    if bool(rescue_payload.get("success", False)):
                        result_payload = rescue_payload
                        status = str(rescue_payload.get("status", "Success"))
                    elif result_payload is None:
                        result_payload = rescue_payload
        except Exception:
            status = "EXCEPTION"
            error_text = traceback.format_exc()
            print("[return_preplan] worker failed:\n" + error_text)
        elapsed_ms = round((time.perf_counter() - worker_start) * 1000.0, 3)
        payload = {
            "status": status,
            "success": bool(result_payload and result_payload.get("success", False)),
            "elapsed_ms": elapsed_ms,
            "q_path": None if result_payload is None else result_payload.get("q_path"),
            "start_q": preplan_start_q.copy(),
            "goal_q": goal_q.copy(),
            "mode": None if result_payload is None else result_payload.get("mode", "direct"),
            "direct_status": direct_status,
            "prelift_waypoints": None if result_payload is None else result_payload.get("prelift_waypoints"),
            "return_waypoints": None if result_payload is None else result_payload.get("return_waypoints"),
            "extra_scene_obstacle_count": len(state["extra_scene_obstacles"]),
            "error_text": error_text,
        }
        with state["lock"]:
            state["result"] = payload
        _record_profile(
            args,
            "return_to_start_preplan_worker",
            success=bool(payload["success"]),
            status=status,
            elapsed_ms=elapsed_ms,
            path_waypoints=len(payload["q_path"] or []),
            mode=payload.get("mode"),
            direct_status=direct_status,
            prelift_waypoints=payload.get("prelift_waypoints"),
            return_waypoints=payload.get("return_waypoints"),
            extra_scene_obstacle_count=payload.get("extra_scene_obstacle_count"),
            error_text=None if error_text is None else str(error_text)[-6000:],
        )

    thread = threading.Thread(
        target=_worker,
        name=f"return-to-start-preplan-{getattr(args, 'object_name', 'unknown')}",
        daemon=True,
    )
    state["thread"] = thread
    thread.start()


def _consume_return_to_start_preplan(demo, args, current_q, goal_q, *, use_attach: bool) -> list[np.ndarray] | None:
    state = getattr(args, "_return_to_start_preplan_state", None)
    if not isinstance(state, dict):
        return None
    current_q = _q7_or_none(current_q)
    goal_q = _q7_or_none(goal_q)
    if current_q is None or goal_q is None:
        return None
    thread = state.get("thread")
    wait_timeout = float(max(getattr(args, "return_to_start_preplan_wait_timeout", 30.0), 0.0))
    if thread is not None and thread.is_alive() and wait_timeout > 0.0:
        print(f"[return_preplan] waiting up to {wait_timeout:.2f}s for return_to_start preplan")
        thread.join(wait_timeout)
    thread_alive = bool(thread is not None and thread.is_alive())
    with state["lock"]:
        payload = dict(state.get("result") or {})
    if not payload:
        _record_profile(
            args,
            "return_to_start_preplan_consume",
            success=False,
            status="NOT_READY",
            wait_timeout=wait_timeout,
            thread_alive=thread_alive,
        )
        return None
    if not bool(payload.get("success", False)) or not payload.get("q_path"):
        _record_profile(
            args,
            "return_to_start_preplan_consume",
            success=False,
            status=str(payload.get("status", "FAILED")),
            wait_timeout=wait_timeout,
            thread_alive=thread_alive,
            worker_elapsed_ms=payload.get("elapsed_ms"),
            mode=payload.get("mode"),
            direct_status=payload.get("direct_status"),
            prelift_waypoints=payload.get("prelift_waypoints"),
            return_waypoints=payload.get("return_waypoints"),
            worker_error_text=None if payload.get("error_text") is None else str(payload.get("error_text"))[-6000:],
        )
        return None
    planned_start_q = _q7_or_none(payload.get("start_q"))
    planned_goal_q = _q7_or_none(payload.get("goal_q"))
    if planned_start_q is None or planned_goal_q is None:
        return None
    start_delta = float(np.max(np.abs(current_q - planned_start_q)))
    goal_delta = float(np.max(np.abs(goal_q - planned_goal_q)))
    q_tol = float(max(getattr(args, "return_to_start_preplan_start_q_tolerance", 0.05), 0.0))
    if start_delta > q_tol or goal_delta > q_tol:
        _record_profile(
            args,
            "return_to_start_preplan_consume",
            success=False,
            status="Q_MISMATCH",
            wait_timeout=wait_timeout,
            start_delta=start_delta,
            goal_delta=goal_delta,
            q_tol=q_tol,
            worker_elapsed_ms=payload.get("elapsed_ms"),
        )
        return None
    q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7].copy() for q in list(payload.get("q_path") or [])]
    if not q_path:
        return None
    q_path[0] = current_q.copy()
    if not _validate_candidate_joint_path_with_demo_planner(
        demo,
        current_q,
        q_path,
        use_attach=use_attach,
        label="return_to_start_preplan_cached",
    ):
        _record_profile(
            args,
            "return_to_start_preplan_consume",
            success=False,
            status="DEMO_VALIDATE_FAIL",
            wait_timeout=wait_timeout,
            start_delta=start_delta,
            goal_delta=goal_delta,
            q_tol=q_tol,
            worker_elapsed_ms=payload.get("elapsed_ms"),
            path_waypoints=len(q_path),
            mode=payload.get("mode"),
            direct_status=payload.get("direct_status"),
            prelift_waypoints=payload.get("prelift_waypoints"),
            return_waypoints=payload.get("return_waypoints"),
        )
        return None
    args._return_to_start_preplan_last_mode = str(payload.get("mode", "direct") or "direct")
    _record_profile(
        args,
        "return_to_start_preplan_consume",
        success=True,
        status="HIT",
        wait_timeout=wait_timeout,
        start_delta=start_delta,
        goal_delta=goal_delta,
        q_tol=q_tol,
        worker_elapsed_ms=payload.get("elapsed_ms"),
        path_waypoints=len(q_path),
        mode=payload.get("mode"),
        direct_status=payload.get("direct_status"),
        prelift_waypoints=payload.get("prelift_waypoints"),
        return_waypoints=payload.get("return_waypoints"),
    )
    return q_path


def _prepare_return_preplan_prelift_pose(demo, planner, args, return_start_q, *, placed_obstacles=None):
    if not bool(getattr(args, "return_to_start_preplan_prelift", True)) or planner is None:
        return None, 0.0
    prelift_lift_m, lift_debug = _return_start_clearance_lift_m(args, placed_obstacles)
    if prelift_lift_m <= 1e-5:
        return None, prelift_lift_m
    try:
        prelift_start_pose = _get_demo_tcp_pose_for_joint_q(demo, return_start_q)
        prelift_goal_pose = _lift_pose_world_z(prelift_start_pose, prelift_lift_m)
        prelift_planner_pose = _convert_demo_tcp_pose_to_curobo_ee_pose_for_joint_q(
            demo,
            return_start_q,
            prelift_goal_pose,
            ee_link_name=str(getattr(planner.config, "ee_link", "gripper_tcp")),
        )
        if bool(getattr(args, "curobo_debug", False)):
            print(
                f"[return_preplan] prelift distance={prelift_lift_m:.4f}m "
                f"({lift_debug.get('source')}, object_world_z_height_m={lift_debug.get('object_world_z_height_m')})"
            )
        return prelift_planner_pose, prelift_lift_m
    except Exception as exc:
        print(f"[return_preplan] failed to prepare prelift rescue pose: {exc}")
        return None, prelift_lift_m


def _plan_and_execute_return_to_cycle_start(
    demo,
    bridge_mod,
    real_exec,
    args,
    start_q,
    *,
    use_attach: bool = False,
    gripper_pos: float | None = None,
) -> bool:
    profile_start_t = time.perf_counter()
    label = "return_to_cycle_start"
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    q_current = _clip_arm_q_to_joint_limits(
        demo,
        np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7],
        label=label,
    )
    target_pose = _get_demo_tcp_pose_for_joint_q(demo, start_q)

    planner = _get_or_create_curobo_planner_serialized(args)
    q_path = None
    return_mode = "live"
    return_extra_obstacles = _return_to_start_placed_obstacles(demo, args)
    if return_extra_obstacles:
        args._return_to_start_extra_scene_obstacles = _copy_scene_obstacle_entries(return_extra_obstacles)
    cached_path = _consume_return_to_start_preplan(demo, args, q_current, start_q, use_attach=use_attach)
    if cached_path:
        q_path = cached_path
        cached_plan_mode = str(getattr(args, "_return_to_start_preplan_last_mode", "direct") or "direct")
        return_mode = f"preplan_cached_{cached_plan_mode}"
        print(f"[return_preplan] using cached return_to_start path ({len(q_path)} waypoint(s))")

    clearance_lift_m, clearance_lift_debug = _return_start_clearance_lift_m(args, return_extra_obstacles)
    if q_path is None and clearance_lift_m > 1e-5:
        with _profile_stage(
            args,
            "return_to_start_prelift",
            lift_m=clearance_lift_m,
            lift_source=clearance_lift_debug.get("source"),
            object_world_z_height_m=clearance_lift_debug.get("object_world_z_height_m"),
        ) as prof:
            lift_path = _plan_short_world_z_lift_ik(
                planner,
                demo,
                args,
                q_current,
                lift_m=clearance_lift_m,
                label=f"{label}_prelift",
                include_table=bool(getattr(args, "curobo_table_collision", True)),
                exclude_object_names=None,
                extra_scene_obstacles=return_extra_obstacles,
                disabled_world_collision_links=_direct_place_contact_tolerant_disabled_links(planner),
            ) if planner is not None else None
            prof["path_waypoints"] = len(lift_path or [])
            if lift_path is not None and len(lift_path) >= 2:
                lift_pose = _lift_pose_world_z(demo.tcp.pose, clearance_lift_m)
                ok, _ = targeted.base.execute_pose_path_stage(
                    demo,
                    bridge_mod,
                    real_exec,
                    f"{label}_prelift",
                    lift_pose,
                    lift_path,
                    args.real_gripper_open if gripper_pos is None else float(gripper_pos),
                    args,
                    use_attach=False,
                )
                prof["success"] = bool(ok)
                prof["status"] = "Success" if ok else "EXEC_FAIL_CONTINUE"
                if ok:
                    q_current = _clip_arm_q_to_joint_limits(
                        demo,
                        np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7],
                        label=label,
                    )
                    print(f"[planner] {label}: lifted empty gripper by {clearance_lift_m:.3f}m before return planning")
                else:
                    print(f"[warn] {label}: prelift execution failed; trying direct return anyway")
            else:
                prof["success"] = False
                prof["status"] = "PLAN_FAIL_CONTINUE"
                print(f"[warn] {label}: prelift planning failed; trying direct return anyway")
    if q_path is None and planner is not None:
        with _profile_stage(
            args,
            "return_to_start_joint_plan",
            enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
            num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
        ) as prof:
            joint_payload = _plan_return_to_start_joint_curobo(
                planner,
                demo,
                args,
                q_current,
                start_q,
                label=label,
                extra_scene_obstacles=return_extra_obstacles,
            )
            q_path = joint_payload.get("q_path")
            prof["success"] = bool(joint_payload.get("success", False))
            prof["status"] = str(joint_payload.get("status", "FAILED"))
            prof["path_waypoints"] = int(joint_payload.get("path_waypoints", 0) or 0)
            prof["max_joint_error"] = joint_payload.get("max_joint_error")
            prof["solve_time"] = joint_payload.get("solve_time")
            prof["trajopt_time"] = joint_payload.get("trajopt_time")
        if q_path is not None:
            print(
                f"[curobo] {label} joint-space plan succeeded ({len(q_path)} waypoints), "
                f"max_joint_error={float(joint_payload.get('max_joint_error', 0.0) or 0.0):.4f} rad"
            )
        else:
            print(
                f"[curobo] {label} joint-space cuRobo failed "
                f"(status={joint_payload.get('status')}), falling back to TCP-pose cuRobo"
            )
            goal_pose = _get_demo_tcp_pose_for_joint_q(demo, start_q)
            planner_pose = _convert_demo_tcp_pose_to_curobo_ee_pose(
                demo,
                goal_pose,
                ee_link_name=str(getattr(planner.config, "ee_link", "gripper_tcp")),
            )
            with _profile_stage(
                args,
                "return_to_start_pose_fallback",
                enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
                num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
                num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
            ) as prof:
                pose_result = _profile_plan_to_pose(
                    planner,
                    q_current,
                    planner_pose,
                    enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
                    max_attempts=int(getattr(args, "curobo_max_attempts", 2)),
                    timeout=float(getattr(args, "curobo_timeout", 5.0)),
                    num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
                    num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
                    num_graph_seeds=int(getattr(args, "curobo_num_graph_seeds", 1)),
                )
                prof["raw_status"] = str(getattr(pose_result, "status", ""))
                if pose_result.success and pose_result.joint_path is not None:
                    pose_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in pose_result.joint_path]
                    q_final = pose_path[-1]
                    joint_errors = np.abs(q_final - start_q)
                    max_joint_error_per_joint = float(getattr(args, "return_to_start_max_joint_error_rad", 0.1))
                    max_error = float(np.max(joint_errors))
                    prof["path_waypoints"] = len(pose_path)
                    prof["max_joint_error"] = max_error
                    if max_error <= max_joint_error_per_joint:
                        q_path = pose_path
                        prof["success"] = True
                        prof["status"] = "Success"
                        print(
                            f"[curobo] {label} TCP-pose fallback succeeded ({len(q_path)} waypoints), "
                            f"max_joint_error={max_error:.4f} rad"
                        )
                    else:
                        worst_joint_idx = int(np.argmax(joint_errors))
                        prof["success"] = False
                        prof["status"] = "JOINT_ERROR_TOO_LARGE"
                        prof["worst_joint_idx"] = worst_joint_idx
                        prof["worst_joint_error"] = float(joint_errors[worst_joint_idx])
                        print(
                            f"[curobo] {label} TCP-pose fallback joint[{worst_joint_idx}] error="
                            f"{joint_errors[worst_joint_idx]:.4f} rad exceeds "
                            f"max_allowed={max_joint_error_per_joint:.4f} rad"
                        )
                        print(f"[curobo] {label} all joint errors (rad): {np.round(joint_errors, 4).tolist()}")
                else:
                    prof["success"] = False
                    prof["status"] = str(getattr(pose_result, "status", "FAILED"))
                    print(f"[curobo] {label} TCP-pose cuRobo failed (status={pose_result.status}), falling back to MPLib (no RRT)")

    if q_path is None:
        with _profile_stage(args, "return_to_start_mplib_fallback") as prof:
            q_path = targeted.base.plan_joint_path(
                demo,
                start_q,
                use_attach=use_attach,
                label=label,
                start_q=q_current,
                allow_reverse_rrt_fallback=False,
            )
            prof["success"] = bool(q_path)
            prof["status"] = "Success" if q_path else "PLAN_FAIL"
            prof["path_waypoints"] = len(q_path or [])
    if q_path is None:
        print(f"[FAIL] {label} planning failed")
        targeted.base.print_failure_diagnostics(
            demo,
            args,
            label,
            q_target=start_q,
            use_attach=use_attach,
        )
        _record_profile(
            args,
            "return_to_start",
            success=False,
            status="PLAN_FAIL",
            elapsed_ms=round((time.perf_counter() - profile_start_t) * 1000.0, 3),
        )
        return False
    if bool(getattr(args, "return_to_start_self_collision_audit", True)) and planner is not None:
        with _profile_stage(
            args,
            "return_to_start_self_collision_audit",
            mode=return_mode,
            path_waypoints=len(q_path or []),
        ) as prof:
            audit = _audit_return_to_start_self_collision_path(
                planner,
                args,
                q_path,
                label=label,
                mode=return_mode,
            )
            prof.update(audit)
    with _profile_stage(args, "return_to_start_execute", mode=return_mode, path_waypoints=len(q_path or [])) as prof:
        ok, _ = targeted.base.execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            label,
            target_pose,
            q_path,
            args.real_gripper_open if gripper_pos is None else float(gripper_pos),
            args,
            use_attach=use_attach,
        )
        prof["success"] = bool(ok)
        prof["status"] = "Success" if ok else "EXEC_FAIL"
    if not ok:
        print(f"[FAIL] {label} execution failed")
        _record_profile(
            args,
            "return_to_start",
            success=False,
            status="EXEC_FAIL",
            elapsed_ms=round((time.perf_counter() - profile_start_t) * 1000.0, 3),
            path_waypoints=len(q_path or []),
        )
        return False
    _record_profile(
        args,
        "return_to_start",
        success=True,
        status="Success",
        elapsed_ms=round((time.perf_counter() - profile_start_t) * 1000.0, 3),
        path_waypoints=len(q_path or []),
        mode=return_mode,
        enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
        num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
        num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
    )
    return True


def _validate_curobo_terminal_pose(demo, args, q_path, pose, *, label: str):
    q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in list(q_path or [])]
    if not q_path:
        return None

    realized = _measure_realized_tcp_error(demo, q_path[-1], pose)
    print(
        f"[curobo] {label} realized tcp: "
        f"pos_err={realized['pos_err']:.4f} m, "
        f"rot_err={realized['rot_err_deg']:.2f} deg"
    )
    if not _terminal_error_within_limits(realized, args):
        print(f"[curobo] {label} terminal pose misses the target pose too much; rejecting this candidate")
        return None

    return {
        "q_path": q_path,
        "realized_before": realized,
        "realized_after": realized,
        "snap_waypoints": 0,
        "terminal_q": q_path[-1],
    }


def _evaluate_curobo_pose_candidates(
    planner,
    demo,
    args,
    start_q,
    candidates,
    *,
    label: str,
    prefer_verticality: bool = False,
    use_attach: bool = False,
    timeout: float | None = None,
    max_attempts: int | None = None,
    num_ik_seeds: int | None = None,
    num_trajopt_seeds: int | None = None,
    num_graph_seeds: int | None = None,
    include_active_object: bool = False,
    include_table: bool = False,
    exclude_object_names: set[str] | None = None,
    disabled_world_collision_links: list[str] | None = None,
):
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    _refresh_curobo_world(planner, demo, args, label=label, include_active_object=include_active_object, include_table=include_table, exclude_object_names=exclude_object_names)
    ee_link_name = str(getattr(planner.config, "ee_link", "gripper_tcp"))
    if bool(getattr(args, "curobo_debug", False)):
        ee_link = _get_robot_link_by_name(demo, ee_link_name)
        base_pose = targeted.base.flatten_np(demo.robot.pose.p)[:3]
        print(
            f"[curobo] runtime frames for {label}: "
            f"planner_ee_link={ee_link_name}, "
            f"demo_tcp_link={_get_link_name(demo.tcp)}, "
            f"robot_base_world_p={np.round(base_pose, 6)}"
        )
        if ee_link is not None:
            ee_world = targeted.base.flatten_np(ee_link.pose.p)[:3].astype(np.float32)
            tcp_world = targeted.base.flatten_np(demo.tcp.pose.p)[:3].astype(np.float32)
            print(
                f"[curobo] runtime link check for {label}: "
                f"ee_link_world_p={np.round(ee_world, 6)}, "
                f"tcp_world_p={np.round(tcp_world, 6)}, "
                f"ee_tcp_delta_norm={float(np.linalg.norm(ee_world - tcp_world)):.6f}"
            )
    successes = []
    disabled_world_collision_links = _set_world_collision_for_links(
        planner,
        disabled_world_collision_links,
        enabled=False,
        label=label,
    )
    try:
        for candidate in list(candidates or []):
            candidate_label = str(candidate["label"])
            pose = candidate["pose"]
            planner_pose = _convert_demo_tcp_pose_to_curobo_ee_pose(
                demo,
                pose,
                ee_link_name=ee_link_name,
            )
            print(f"[curobo] trying {candidate_label}")
            print(f"[curobo] {candidate_label} pose p: {np.round(targeted.base.flatten_np(pose.p)[:3], 6)}")
            print(f"[curobo] {candidate_label} pose q: {np.round(targeted.base.flatten_np(pose.q)[:4], 6)}")
            if bool(getattr(args, "curobo_debug", False)):
                print(f"[curobo] {candidate_label} ee_link(base) pose p: {np.round(targeted.base.flatten_np(planner_pose.p)[:3], 6)}")
                print(f"[curobo] {candidate_label} ee_link(base) pose q: {np.round(targeted.base.flatten_np(planner_pose.q)[:4], 6)}")
            result = _profile_plan_to_pose(
                planner,
                start_q,
                planner_pose,
                enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
                max_attempts=int(getattr(args, "curobo_max_attempts", 2) if max_attempts is None else max_attempts),
                timeout=float(getattr(args, "curobo_timeout", 5.0) if timeout is None else timeout),
                num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64) if num_ik_seeds is None else num_ik_seeds),
                num_trajopt_seeds=int(
                    getattr(args, "curobo_num_trajopt_seeds", 1) if num_trajopt_seeds is None else num_trajopt_seeds
                ),
                num_graph_seeds=int(
                    getattr(args, "curobo_num_graph_seeds", 1) if num_graph_seeds is None else num_graph_seeds
                ),
            )
            if not result.success or result.joint_path is None:
                print(
                    f"[curobo] {candidate_label} failed with status={result.status}, "
                    f"internal_ik={result.debug.get('motion_gen_internal_ik_successes')}"
                )
                if str(result.status) == "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION":
                    targeted.base.print_failure_diagnostics(
                        demo,
                        args,
                        f"{candidate_label}_start_state",
                        q_target=start_q,
                        use_attach=use_attach,
                    )
                    start_diag = planner.diagnose_start_state_world_collision(start_q)
                    for item in list(start_diag.get("ablation", []) or []):
                        print(
                            f"[curobo] start-state ablation remove={item['removed']}: "
                            f"valid={item['valid']}, status={item['status']}"
                        )
                elif str(result.status) == "MotionGenStatus.INVALID_START_STATE_SELF_COLLISION":
                    _print_start_state_self_collision_diagnostics(
                        planner,
                        start_q,
                        label=f"{candidate_label}_start_state",
                    )
                continue
            q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in result.joint_path]
            terminal_align = _validate_curobo_terminal_pose(
                demo,
                args,
                q_path,
                pose,
                label=candidate_label,
            )
            if terminal_align is None:
                continue
            q_path = terminal_align["q_path"]
            if not _validate_candidate_joint_path_with_demo_planner(
                demo,
                start_q,
                q_path,
                use_attach=use_attach,
                label=candidate_label,
            ):
                continue
            metrics, score = _path_metrics_and_score(start_q, q_path)
            selection_penalty = _candidate_selection_penalty(candidate, args)
            place_pref_penalty = _candidate_place_orientation_penalty(candidate, demo, args)
            score = float(score) + float(selection_penalty) + float(place_pref_penalty)
            item = dict(candidate)
            item["result"] = result
            item["q_path"] = q_path
            item["metrics"] = metrics
            item["score"] = score
            item["selection_penalty"] = selection_penalty
            item["place_preference_penalty"] = place_pref_penalty
            item["terminal_align"] = terminal_align
            item["planner_pose"] = planner_pose
            print(
                f"[curobo] {candidate_label} success: score={score:.3f}, "
                f"selection_penalty={selection_penalty:.3f}, "
                f"place_pref_penalty={place_pref_penalty:.3f}, "
                f"total_motion={metrics['total_motion']:.3f} rad, "
                f"joint7_total={metrics['joint7_total_motion']:.3f} rad, "
                f"joint7_excursion={metrics['joint7_max_excursion']:.3f} rad, "
                f"waypoints={metrics['waypoint_count']}, "
                f"realized_pos_err={terminal_align['realized_after']['pos_err']:.4f} m, "
                f"realized_rot_err={terminal_align['realized_after']['rot_err_deg']:.2f} deg"
            )
            successes.append(item)
    finally:
        _set_world_collision_for_links(
            planner,
            disabled_world_collision_links,
            enabled=True,
            label=label,
        )

    if not successes:
        return []

    if prefer_verticality:
        best_verticality = max(float(item.get("tcp_verticality", 0.0)) for item in successes)
        band = float(max(getattr(args, "direct_pre_place_verticality_band", 0.08), 0.0))
        filtered = [item for item in successes if float(item.get("tcp_verticality", 0.0)) >= best_verticality - band]
        if filtered:
            successes = filtered
        print(
            f"[direct_pre_place] kept {len(successes)} pre-place candidate(s) within tcp_verticality "
            f"{best_verticality - band:.3f}..{best_verticality:.3f}"
        )

    successes.sort(key=_candidate_sort_key)
    best = successes[0]
    print(
        f"[curobo] selected {best['label']}: score={best['score']:.3f}, "
        f"tcp_verticality={float(best.get('tcp_verticality', 0.0)):.3f}, "
        f"waypoints={best['metrics']['waypoint_count']}"
    )
    return successes


def _evaluate_two_step_grasp_candidates(
    planner,
    demo,
    args,
    start_q,
    candidates,
    *,
    label: str,
    max_winners: int | None = None,
    include_active_object: bool = False,
    disabled_world_collision_links: list[str] | None = None,
):
    """
    实现两步抓取：
    1. 第一步：规划到pregrasp pose（带碰撞检测）
    2. 第二步：从pregrasp沿夹爪轴向直线下降到grasp pose（简化碰撞检测）

    注意：pregrasp必须是通过demo.build_pregrasp_pose从grasp沿轴向回退得到的，
    这样第二步才能保证是直线下降而不是横向移动。
    """
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]

    # 准备pregrasp candidates
    pregrasp_candidates = []
    for cand in candidates:
        if "pregrasp_pose" not in cand:
            continue
        pregrasp_cand = dict(cand)
        pregrasp_cand["pose"] = cand["pregrasp_pose"]
        pregrasp_cand["original_grasp_pose"] = cand["pose"]
        pregrasp_candidates.append(pregrasp_cand)

    if not pregrasp_candidates:
        print(f"[{label}] no candidates have pregrasp_pose; explicit two-step grasp cannot proceed")
        return []

    print(f"[{label}] evaluating {len(pregrasp_candidates)} two-step grasp candidates")

    # 第一步：规划到pregrasp pose
    with _profile_stage(
        args,
        "grasp_goalset",
        candidate_count=len(pregrasp_candidates),
        max_attempts=int(getattr(args, "curobo_max_attempts", 2)),
        num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
        num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
        enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
    ) as prof:
        pregrasp_successes = _evaluate_curobo_pose_candidates_goalset(
            planner,
            demo,
            args,
            start_q,
            pregrasp_candidates,
            label=f"{label}_pregrasp",
            prefer_verticality=False,
            use_attach=False,
            max_winners=max_winners,
            include_active_object=include_active_object,
            disabled_world_collision_links=disabled_world_collision_links,
        )
        _copy_last_candidate_counts_to_profile(prof, planner)
        prof["winner_count"] = len(pregrasp_successes)
        prof["success"] = bool(pregrasp_successes)
        prof["status"] = "Success" if pregrasp_successes else "NO_WINNERS"
        prof["world_changed"] = bool(getattr(planner, "_last_world_changed", False))
        prof["cache_hit"] = bool(getattr(planner, "_last_world_cache_hit", False))
        if pregrasp_successes:
            prof["path_waypoints"] = int((pregrasp_successes[0].get("metrics") or {}).get("waypoint_count", 0) or 0)
            prof["path_score"] = float(pregrasp_successes[0].get("score", 0.0) or 0.0)

    if not pregrasp_successes:
        print(f"[{label}] no pregrasp candidates succeeded")
        return []

    for pregrasp_success in pregrasp_successes:
        pregrasp_q = np.asarray(pregrasp_success["q_path"][-1], dtype=np.float32).reshape(-1)[:7]
        pregrasp_pose = pregrasp_success["pose"]
        grasp_pose = pregrasp_success["original_grasp_pose"]
        pregrasp_p = targeted.base.flatten_np(pregrasp_pose.p)[:3]
        grasp_p = targeted.base.flatten_np(grasp_pose.p)[:3]
        pregrasp_success["deferred_two_step_grasp"] = True
        pregrasp_success["deferred_pregrasp_q"] = pregrasp_q
        pregrasp_success["deferred_pregrasp_pose"] = pregrasp_pose
        pregrasp_success["deferred_grasp_pose"] = grasp_pose
        pregrasp_success["approach_distance_m"] = float(np.linalg.norm(grasp_p - pregrasp_p))

    print(
        f"[{label}] {len(pregrasp_successes)} pregrasp candidates succeeded; "
        "constrained final approach will be attempted only once for the final selected grasp"
    )
    return pregrasp_successes


def _make_ik_preselected_grasp_success(args, start_q, grasp_candidate: dict) -> dict | None:
    """Use fast-chain IK results as the candidate-stage grasp record.

    This deliberately does not call MotionGen.  The selected candidate's short
    contact descent is planned later, once, with the constrained straight-line
    planner after transport has accepted the pair.
    """
    q_pregrasp = _q7_or_none(
        grasp_candidate.get("_winner_preselect_q_pregrasp", grasp_candidate.get("q_pregrasp"))
    )
    q_grasp = _q7_or_none(
        grasp_candidate.get("_winner_preselect_q_grasp", grasp_candidate.get("q_grasp"))
    )
    pregrasp_pose = grasp_candidate.get("pregrasp_pose")
    grasp_pose = grasp_candidate.get("pose")
    if q_pregrasp is None or q_grasp is None or pregrasp_pose is None or grasp_pose is None:
        return None
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    q_path = _linear_joint_path(start_q, q_pregrasp)
    if not q_path:
        return None
    metrics, path_score = _path_metrics_and_score(start_q, q_path)
    pair_score = grasp_candidate.get("_winner_chain_preselect_score")
    try:
        pair_score = float(pair_score)
    except Exception:
        pair_score = float("nan")
    item = dict(grasp_candidate)
    item["q_path"] = q_path
    item["metrics"] = metrics
    item["score"] = (
        float(pair_score)
        if np.isfinite(pair_score)
        else float(path_score) + float(_candidate_selection_penalty(item, args))
    )
    item["pair_first_pair_score"] = float(item["score"])
    item["pose"] = pregrasp_pose
    item["original_grasp_pose"] = grasp_pose
    item["deferred_two_step_grasp"] = True
    item["deferred_pregrasp_q"] = q_pregrasp
    item["deferred_pregrasp_pose"] = pregrasp_pose
    item["deferred_grasp_pose"] = grasp_pose
    item["q_pregrasp"] = q_pregrasp
    item["q_grasp"] = q_grasp
    approach_q_path = [
        np.asarray(q, dtype=np.float32).reshape(-1)[:7]
        for q in list(grasp_candidate.get("_winner_preselect_grasp_approach_q_path") or [])
    ]
    if approach_q_path:
        item["deferred_final_approach_q_path"] = approach_q_path
    item["pair_first_ik_only"] = True
    item["two_step_grasp"] = False
    item["pregrasp_waypoints"] = len(q_path)
    pregrasp_p = targeted.base.flatten_np(pregrasp_pose.p)[:3]
    grasp_p = targeted.base.flatten_np(grasp_pose.p)[:3]
    item["approach_distance_m"] = float(np.linalg.norm(grasp_p - pregrasp_p))
    return item


def _q7_or_none(q) -> np.ndarray | None:
    if q is None:
        return None
    try:
        arr = np.asarray(q, dtype=np.float32).reshape(-1)[:7]
    except Exception:
        return None
    if arr.shape[0] < 7 or not np.all(np.isfinite(arr)):
        return None
    return arr.astype(np.float32, copy=True)


def _ik_debug_errors(ik_result) -> tuple[float, float]:
    dbg = getattr(ik_result, "debug", {}) or {}
    pos_err = float(dbg.get("position_error", np.nan))
    rot_err = float(dbg.get("rotation_error", np.nan))
    return pos_err, rot_err


def _ik_score(pos_err: float, rot_err: float) -> float:
    pos = pos_err if np.isfinite(pos_err) else 1.0
    rot = rot_err if np.isfinite(rot_err) else 1.0
    return float(pos + 0.05 * rot)


def _candidate_start_matches_prefilter(candidate: dict, start_q=None) -> bool:
    if start_q is None or "_prefilter_start_q" not in candidate:
        return True
    stored = _q7_or_none(candidate.get("_prefilter_start_q"))
    current = _q7_or_none(start_q)
    if stored is None or current is None:
        return False
    return bool(np.allclose(stored, current, atol=1e-5, rtol=0.0))


def _candidate_reusable_prefilter_q(candidate: dict, *, start_q=None) -> np.ndarray | None:
    if not _candidate_start_matches_prefilter(candidate, start_q=start_q):
        return None
    q = _q7_or_none(candidate.get("_prefilter_q_goal"))
    if q is None:
        q = _q7_or_none(candidate.get("q_hover"))
    return q


def _store_candidate_prefilter_record(
    candidate: dict,
    *,
    q_goal,
    start_q=None,
    pos_err: float = np.nan,
    rot_err: float = np.nan,
    ik_score: float | None = None,
    role: str | None = None,
) -> None:
    q = _q7_or_none(q_goal)
    if q is None:
        return
    candidate["_prefilter_q_goal"] = q
    if start_q is not None:
        q_start = _q7_or_none(start_q)
        if q_start is not None:
            candidate["_prefilter_start_q"] = q_start
    candidate["_prefilter_ik_pos_error"] = float(pos_err)
    candidate["_prefilter_ik_rot_error"] = float(rot_err)
    candidate["_prefilter_ik_score"] = float(_ik_score(pos_err, rot_err) if ik_score is None else ik_score)
    if role:
        candidate[f"q_{role}"] = q


def _evaluate_curobo_pose_candidates_goalset(
    planner,
    demo,
    args,
    start_q,
    candidates,
    *,
    label: str,
    prefer_verticality: bool = False,
    use_attach: bool = False,
    timeout: float | None = None,
    max_attempts: int | None = None,
    num_ik_seeds: int | None = None,
    num_trajopt_seeds: int | None = None,
    num_graph_seeds: int | None = None,
    max_winners: int | None = None,
    include_active_object: bool = False,
    include_table: bool = False,
    exclude_object_names: set[str] | None = None,
    disabled_world_collision_links: list[str] | None = None,
):
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    _refresh_curobo_world(planner, demo, args, label=label, include_active_object=include_active_object, include_table=include_table, exclude_object_names=exclude_object_names)
    remaining = list(candidates or [])
    if not remaining:
        return []

    ee_link_name = str(getattr(planner.config, "ee_link", "gripper_tcp"))
    if bool(getattr(args, "curobo_debug", False)):
        ee_link = _get_robot_link_by_name(demo, ee_link_name)
        base_pose = targeted.base.flatten_np(demo.robot.pose.p)[:3]
        print(
            f"[curobo] runtime frames for {label}: "
            f"planner_ee_link={ee_link_name}, "
            f"demo_tcp_link={_get_link_name(demo.tcp)}, "
            f"robot_base_world_p={np.round(base_pose, 6)}"
        )
        if ee_link is not None:
            print(
                f"[curobo] runtime link check for {label}: "
                f"ee_link_world_p={np.round(targeted.base.flatten_np(ee_link.pose.p)[:3], 6)}, "
                f"tcp_world_p={np.round(targeted.base.flatten_np(demo.tcp.pose.p)[:3], 6)}"
            )

    if prefer_verticality:
        best_verticality = max(float(item.get("tcp_verticality", 0.0)) for item in remaining)
        band = float(max(getattr(args, "direct_pre_place_verticality_band", 0.08), 0.0))
        filtered = [item for item in remaining if float(item.get("tcp_verticality", 0.0)) >= best_verticality - band]
        if filtered:
            remaining = filtered
        print(
            f"[direct_pre_place] prefiltered {len(remaining)} goalset candidate(s) within tcp_verticality "
            f"{best_verticality - band:.3f}..{best_verticality:.3f}"
        )

    winners = []
    use_max_winners = int(len(remaining) if max_winners is None else max_winners)
    if use_max_winners <= 0:
        use_max_winners = len(remaining)
    saw_invalid_start = False
    saw_invalid_start_self_collision = False
    chunk_size = int(getattr(args, "curobo_batch_chunk_size", 64))
    if chunk_size <= 0:
        chunk_size = len(remaining)

    disabled_world_collision_links = _set_world_collision_for_links(
        planner,
        disabled_world_collision_links,
        enabled=False,
        label=f"{label}_ik+plan",
    )
    try:
        ik_screened = []
        ik_error_records = []
        ik_status_counts = {}
        ik_screen_total = len(remaining)
        planner._last_candidate_count_in = int(ik_screen_total)
        planner._last_candidate_count_after_ik = 0
        planner._last_candidate_count_motiongen = 0
        ranked_failed_candidates = []
        reused_prefilter_count = 0
        strict_pass_count = 0
        soft_pass_count = 0
        ik_prefilter_pos_thresh = float(getattr(args, "curobo_ik_prefilter_position_threshold", 0.01) or 0.0)
        ik_prefilter_rot_thresh = float(getattr(args, "curobo_ik_prefilter_rotation_threshold", 0.25) or 0.0)
        use_soft_prefilter = ik_prefilter_pos_thresh > 0.0 and ik_prefilter_rot_thresh > 0.0
        for start_idx in range(0, len(remaining), chunk_size):
            chunk = remaining[start_idx : start_idx + chunk_size]
            need_ik = []
            for candidate in chunk:
                reused_q = _candidate_reusable_prefilter_q(candidate, start_q=start_q)
                if reused_q is None:
                    need_ik.append(candidate)
                    continue
                reused_prefilter_count += 1
                pos_err = float(candidate.get("_prefilter_ik_pos_error", np.nan))
                rot_err = float(candidate.get("_prefilter_ik_rot_error", np.nan))
                ik_error_records.append(
                    {
                        "label": str(candidate["label"]),
                        "position_error": pos_err,
                        "rotation_error": rot_err,
                        "success": True,
                        "reused": True,
                    }
                )
                ik_screened.append(candidate)
                strict_pass_count += 1
            if not need_ik:
                continue
            planner_poses = [
                _convert_demo_tcp_pose_to_curobo_ee_pose(
                    demo,
                    item["pose"],
                    ee_link_name=ee_link_name,
                )
                for item in need_ik
            ]
            start_qs = [start_q for _ in need_ik]
            ik_results = _profile_solve_batch_start_goal_ik(
                planner,
                start_qs,
                planner_poses,
                num_seeds=int(getattr(args, "curobo_num_ik_seeds", 64) if num_ik_seeds is None else num_ik_seeds),
            )
            for candidate, planner_pose, ik_result in zip(need_ik, planner_poses, ik_results):
                status_key = str(ik_result.status)
                ik_status_counts[status_key] = int(ik_status_counts.get(status_key, 0)) + 1
                pos_err, rot_err = _ik_debug_errors(ik_result)
                ik_error_records.append(
                    {
                        "label": str(candidate["label"]),
                        "position_error": pos_err,
                        "rotation_error": rot_err,
                        "success": bool(ik_result.success),
                    }
                )
                if bool(getattr(args, "curobo_debug", False)) and "lvmukuai_vertical_gripper" in str(candidate.get("label", "")).lower():
                    print(
                        f"[curobo][diag] {label} {candidate['label']} IK "
                        f"success={bool(ik_result.success)} status={ik_result.status} "
                        f"pos_err={pos_err:.5f} rot_err={rot_err:.5f}"
                    )
                if ik_result.success:
                    _store_candidate_prefilter_record(
                        candidate,
                        q_goal=getattr(ik_result, "goal_joint", None),
                        start_q=start_q,
                        pos_err=pos_err,
                        rot_err=rot_err,
                    )
                    ik_screened.append(candidate)
                    strict_pass_count += 1
                    continue
                if (
                    use_soft_prefilter
                    and pos_err is not None
                    and rot_err is not None
                    and float(pos_err) <= ik_prefilter_pos_thresh
                    and float(rot_err) <= ik_prefilter_rot_thresh
                ):
                    _store_candidate_prefilter_record(
                        candidate,
                        q_goal=getattr(ik_result, "goal_joint", None),
                        start_q=start_q,
                        pos_err=pos_err,
                        rot_err=rot_err,
                    )
                    ik_screened.append(candidate)
                    soft_pass_count += 1
                    continue
                rank_key = (
                    pos_err if np.isfinite(pos_err) else np.inf,
                    rot_err if np.isfinite(rot_err) else np.inf,
                    str(candidate["label"]),
                )
                ranked_failed_candidates.append(
                    {
                        "candidate": candidate,
                        "planner_pose": planner_pose,
                        "rank_key": rank_key,
                    }
                )
        remaining = ik_screened
        planner._last_candidate_count_after_ik = int(len(remaining))
        status_summary = ", ".join(f"{k}={v}" for k, v in sorted(ik_status_counts.items()))
        soft_info = ""
        if use_soft_prefilter:
            soft_info = (
                f" (strict={strict_pass_count}, soft={soft_pass_count}, "
                f"soft_thresholds: pos<={ik_prefilter_pos_thresh:.4f} rot<={ik_prefilter_rot_thresh:.4f})"
            )
        print(
            f"[curobo][diag] {label} IK prefilter kept {len(remaining)}/{ik_screen_total} candidate(s); "
            f"reused_prefilter={reused_prefilter_count}; statuses: {status_summary or 'none'}{soft_info}"
        )
        if bool(getattr(args, "curobo_debug", False)) or not remaining:
            _print_ik_error_summary(label, ik_error_records)
        if not remaining:
            diagnose_topk = int(getattr(args, "direct_grasp_diagnose_topk", 0))
            if diagnose_topk > 0:
                ranked_failed_candidates.sort(key=lambda x: x["rank_key"])
                _diagnose_failed_ik_candidates(
                    planner,
                    demo,
                    args,
                    label,
                    start_q,
                    ranked_failed_candidates,
                    topk=diagnose_topk,
                )
            return []
        num_chunks = max(1, (len(remaining) + chunk_size - 1) // chunk_size)
        print(
            f"[curobo] {label} evaluating {len(remaining)} candidate(s) "
            f"in {num_chunks} batch chunk(s) of size <= {chunk_size}"
        )
        planner._last_candidate_count_motiongen = int(len(remaining))
        if use_max_winners == 1 and len(remaining) > 1:
            planner_poses = [
                _convert_demo_tcp_pose_to_curobo_ee_pose(
                    demo,
                    item["pose"],
                    ee_link_name=ee_link_name,
                )
                for item in remaining
            ]
            print(f"[curobo] {label} using official goalset fast-path for {len(remaining)} candidate(s)")
            goalset_result = _profile_plan_goalset_to_poses(
                planner,
                start_q,
                planner_poses,
                enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
                max_attempts=int(getattr(args, "curobo_max_attempts", 2) if max_attempts is None else max_attempts),
                timeout=float(getattr(args, "curobo_timeout", 5.0) if timeout is None else timeout),
                num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64) if num_ik_seeds is None else num_ik_seeds),
                num_trajopt_seeds=int(
                    getattr(args, "curobo_num_trajopt_seeds", 1) if num_trajopt_seeds is None else num_trajopt_seeds
                ),
                num_graph_seeds=int(
                    getattr(args, "curobo_num_graph_seeds", 1) if num_graph_seeds is None else num_graph_seeds
                ),
            )
            goal_idx = int((goalset_result.debug or {}).get("goalset_index", -1))
            if goalset_result.success and goalset_result.joint_path is not None and 0 <= goal_idx < len(remaining):
                candidate = remaining[goal_idx]
                candidate_label = str(candidate["label"])
                q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in goalset_result.joint_path]
                terminal_align = _validate_curobo_terminal_pose(
                    demo,
                    args,
                    q_path,
                    candidate["pose"],
                    label=candidate_label,
                )
                if terminal_align is not None:
                    q_path = terminal_align["q_path"]
                    if _validate_candidate_joint_path_with_demo_planner(
                        demo,
                        start_q,
                        q_path,
                        use_attach=use_attach,
                        label=candidate_label,
                    ):
                        metrics, score = _path_metrics_and_score(start_q, q_path)
                        selection_penalty = _candidate_selection_penalty(candidate, args)
                        place_pref_penalty = _candidate_place_orientation_penalty(candidate, demo, args)
                        score = float(score) + float(selection_penalty) + float(place_pref_penalty)
                        item = dict(candidate)
                        item["result"] = goalset_result
                        item["q_path"] = q_path
                        item["metrics"] = metrics
                        item["score"] = score
                        item["selection_penalty"] = selection_penalty
                        item["place_preference_penalty"] = place_pref_penalty
                        item["terminal_align"] = terminal_align
                        item["planner_pose"] = planner_poses[goal_idx]
                        item["q_goal"] = q_path[-1]
                        if "pregrasp" in str(label).lower():
                            item["q_pregrasp"] = q_path[-1]
                        elif "hover" in str(label).lower() or "transport" in str(label).lower():
                            item["q_hover"] = q_path[-1]
                        winners.append(item)
                        print(
                            f"[curobo] {candidate_label} goalset success: score={score:.3f}, "
                            f"selection_penalty={selection_penalty:.3f}, "
                            f"place_pref_penalty={place_pref_penalty:.3f}, "
                            f"waypoints={metrics['waypoint_count']}"
                        )
                        return winners
        ee_link_name = str(getattr(planner.config, "ee_link", "gripper_tcp"))
        for chunk_idx, start_idx in enumerate(range(0, len(remaining), chunk_size), start=1):
            chunk = remaining[start_idx : start_idx + chunk_size]
            planner_poses = [
                _convert_demo_tcp_pose_to_curobo_ee_pose(
                    demo,
                    item["pose"],
                    ee_link_name=ee_link_name,
                )
                for item in chunk
            ]
            if num_chunks > 1:
                print(
                    f"[curobo] {label} batch chunk {chunk_idx}/{num_chunks}: "
                    f"{len(chunk)} candidate(s)"
                )
            batch_results = _profile_plan_batch_to_poses(
                planner,
                start_q,
                planner_poses,
                enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
                max_attempts=int(getattr(args, "curobo_max_attempts", 2) if max_attempts is None else max_attempts),
                timeout=float(getattr(args, "curobo_timeout", 5.0) if timeout is None else timeout),
                num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64) if num_ik_seeds is None else num_ik_seeds),
                num_trajopt_seeds=int(
                    getattr(args, "curobo_num_trajopt_seeds", 1) if num_trajopt_seeds is None else num_trajopt_seeds
                ),
                num_graph_seeds=int(
                    getattr(args, "curobo_num_graph_seeds", 1) if num_graph_seeds is None else num_graph_seeds
                ),
            )
            for candidate, planner_pose, result in zip(chunk, planner_poses, batch_results):
                candidate_label = str(candidate["label"])
                if not result.success or result.joint_path is None:
                    print(f"[curobo] {candidate_label} batch failed with status={result.status}")
                    if str(result.status) == "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION":
                        saw_invalid_start = True
                    continue
                q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in result.joint_path]
                terminal_align = _validate_curobo_terminal_pose(
                    demo,
                    args,
                    q_path,
                    candidate["pose"],
                    label=candidate_label,
                )
                if terminal_align is None:
                    continue
                q_path = terminal_align["q_path"]
                if not _validate_candidate_joint_path_with_demo_planner(
                    demo,
                    start_q,
                    q_path,
                    use_attach=use_attach,
                    label=candidate_label,
                ):
                    continue
                metrics, score = _path_metrics_and_score(start_q, q_path)
                selection_penalty = _candidate_selection_penalty(candidate, args)
                place_pref_penalty = _candidate_place_orientation_penalty(candidate, demo, args)
                score = float(score) + float(selection_penalty) + float(place_pref_penalty)
                item = dict(candidate)
                item["result"] = result
                item["q_path"] = q_path
                item["metrics"] = metrics
                item["score"] = score
                item["selection_penalty"] = selection_penalty
                item["place_preference_penalty"] = place_pref_penalty
                item["terminal_align"] = terminal_align
                item["planner_pose"] = planner_pose
                item["q_goal"] = q_path[-1]
                if "pregrasp" in str(label).lower():
                    item["q_pregrasp"] = q_path[-1]
                elif "hover" in str(label).lower() or "transport" in str(label).lower():
                    item["q_hover"] = q_path[-1]
                print(
                    f"[curobo] {candidate_label} success: score={score:.3f}, "
                    f"selection_penalty={selection_penalty:.3f}, "
                    f"place_pref_penalty={place_pref_penalty:.3f}, "
                    f"total_motion={metrics['total_motion']:.3f} rad, "
                    f"joint7_total={metrics['joint7_total_motion']:.3f} rad, "
                    f"joint7_excursion={metrics['joint7_max_excursion']:.3f} rad, "
                    f"waypoints={metrics['waypoint_count']}, "
                    f"realized_pos_err={terminal_align['realized_after']['pos_err']:.4f} m, "
                    f"realized_rot_err={terminal_align['realized_after']['rot_err_deg']:.2f} deg"
                )
                winners.append(item)
                if len(winners) >= use_max_winners:
                    break
            if len(winners) >= use_max_winners:
                break
    finally:
        _set_world_collision_for_links(
            planner,
            disabled_world_collision_links,
            enabled=True,
            label=label,
        )

    if not winners and saw_invalid_start:
        targeted.base.print_failure_diagnostics(
            demo,
            args,
            f"{label}_start_state",
            q_target=start_q,
            use_attach=use_attach,
        )

    if not winners:
        return []
    winners.sort(key=_candidate_sort_key)
    best = winners[0]
    print(
        f"[curobo] selected {best['label']}: score={best['score']:.3f}, "
        f"tcp_verticality={float(best.get('tcp_verticality', 0.0)):.3f}, "
        f"waypoints={best['metrics']['waypoint_count']}"
    )
    return winners


def _filter_pre_place_candidates_by_verticality(candidates, args):
    remaining = list(candidates or [])
    if not remaining:
        return []
    best_verticality = max(float(item.get("tcp_verticality", 0.0)) for item in remaining)
    band = float(max(getattr(args, "direct_pre_place_verticality_band", 0.08), 0.0))
    filtered = [item for item in remaining if float(item.get("tcp_verticality", 0.0)) >= best_verticality - band]
    if filtered:
        remaining = filtered
    print(
        f"[direct_pre_place] prefiltered {len(remaining)} candidate(s) within tcp_verticality "
        f"{best_verticality - band:.3f}..{best_verticality:.3f}"
    )
    return remaining


def _fast_chain_rank_place_candidates(
    planner,
    demo,
    args,
    candidates,
    start_q,
    *,
    label: str,
    disabled_world_collision_links: list[str] | None = None,
):
    """Cheaply rank place candidates before the expensive transport MotionGen pass.

    This is intentionally only a pre-screen: it uses batched IK for hover and
    release poses, then returns a small ordered subset. The normal transport and
    final-contact planners still validate the chain, and the caller falls back to
    the full search if this subset fails.
    """
    top_pairs = int(getattr(args, "fast_chain_top_pairs", 3) or 0)
    if top_pairs <= 0:
        return []
    remaining = sorted(list(candidates or []), key=_pre_place_screen_sort_key)
    if not remaining:
        return []
    source_name = _current_source_object_name(args)
    if source_name == "bi":
        small_remaining = _bi_insert_small_lane_candidates(remaining, args, label=label)
        if small_remaining:
            remaining = small_remaining
    max_ik_candidates = int(getattr(args, "fast_chain_max_ik_candidates", 0) or 0)
    if max_ik_candidates > 0:
        remaining = remaining[:max_ik_candidates]

    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    chunk_size = int(getattr(args, "fast_chain_place_ik_chunk_candidates", 8) or 0)
    if chunk_size <= 0:
        chunk_size = int(getattr(args, "curobo_batch_chunk_size", 64) or 0)
    if chunk_size <= 0:
        chunk_size = len(remaining)
    early_stop = bool(getattr(args, "fast_chain_place_ik_early_stop", True))
    ik_seeds = int(getattr(args, "fast_chain_ik_seeds", 32) or 0)
    if ik_seeds <= 0:
        ik_seeds = int(getattr(args, "curobo_num_ik_seeds", 64) or 64)
    ee_link_name = str(getattr(planner.config, "ee_link", "gripper_tcp"))

    ranked: list[dict] = []
    hover_status_counts: dict[str, int] = {}
    release_status_counts: dict[str, int] = {}
    disabled_world_collision_links = _set_world_collision_for_links(
        planner,
        disabled_world_collision_links,
        enabled=False,
        label=f"{label}_fast_chain_ik",
    )
    try:
        for start_idx in range(0, len(remaining), chunk_size):
            chunk = remaining[start_idx : start_idx + chunk_size]
            start_qs = [start_q for _ in chunk]
            hover_poses = [
                _convert_demo_tcp_pose_to_curobo_ee_pose(
                    demo,
                    item["pose"],
                    ee_link_name=ee_link_name,
                )
                for item in chunk
            ]
            release_poses = [
                _convert_demo_tcp_pose_to_curobo_ee_pose(
                    demo,
                    item.get("release_pose", item.get("place_pose", item["pose"])),
                    ee_link_name=ee_link_name,
                )
                for item in chunk
            ]
            combined_results = _profile_fast_chain_solve_batch_start_goal_ik(
                args,
                planner,
                start_qs + start_qs,
                hover_poses + release_poses,
                num_seeds=ik_seeds,
            )
            hover_results = combined_results[: len(chunk)]
            release_results = combined_results[len(chunk) :]
            for item, hover_result, release_result in zip(chunk, hover_results, release_results):
                hover_status = str(hover_result.status)
                release_status = str(release_result.status)
                hover_status_counts[hover_status] = int(hover_status_counts.get(hover_status, 0)) + 1
                release_status_counts[release_status] = int(release_status_counts.get(release_status, 0)) + 1
                hover_pos_err, hover_rot_err = _ik_debug_errors(hover_result)
                release_pos_err, release_rot_err = _ik_debug_errors(release_result)
                if (
                    not bool(hover_result.success)
                    or hover_result.goal_joint is None
                    or not bool(release_result.success)
                    or release_result.goal_joint is None
                ):
                    continue
                q_hover = np.asarray(hover_result.goal_joint, dtype=np.float32).reshape(-1)[:7]
                q_release = np.asarray(release_result.goal_joint, dtype=np.float32).reshape(-1)[:7]
                joint_delta = q_hover - start_q
                release_delta = q_release - q_hover
                transport_score = float(np.linalg.norm(joint_delta))
                release_score = float(np.linalg.norm(release_delta))
                joint7_score = float(abs(joint_delta[6])) if joint_delta.shape[0] >= 7 else 0.0
                selection_penalty = float(_candidate_selection_penalty(item, args))
                place_pref_penalty = float(_candidate_place_orientation_penalty(item, demo, args))
                heuristic_score = (
                    transport_score
                    + 0.35 * release_score
                    + 0.15 * joint7_score
                    + selection_penalty
                    + place_pref_penalty
                )
                ranked_item = dict(item)
                max_pos_err = float(np.nanmax([hover_pos_err, release_pos_err]))
                max_rot_err = float(np.nanmax([hover_rot_err, release_rot_err]))
                if not np.isfinite(max_pos_err):
                    max_pos_err = float("nan")
                if not np.isfinite(max_rot_err):
                    max_rot_err = float("nan")
                pair_ik_score = float(_ik_score(max_pos_err, max_rot_err))
                pair_score = float(heuristic_score + pair_ik_score)
                ranked_item["fast_chain_score"] = heuristic_score
                ranked_item["fast_chain_hover_q"] = q_hover
                ranked_item["fast_chain_release_q"] = q_release
                ranked_item["fast_chain_transport_joint_norm"] = transport_score
                ranked_item["fast_chain_release_joint_norm"] = release_score
                ranked_item["q_hover"] = q_hover
                ranked_item["q_release"] = q_release
                ranked_item["ik_pos_error"] = max_pos_err
                ranked_item["ik_rot_error"] = max_rot_err
                ranked_item["ik_score"] = pair_ik_score
                ranked_item["pair_score"] = pair_score
                ranked_item["fast_chain_ik_record"] = {
                    "hover_pos_error": hover_pos_err,
                    "hover_rot_error": hover_rot_err,
                    "release_pos_error": release_pos_err,
                    "release_rot_error": release_rot_err,
                    "hover_status": hover_status,
                    "release_status": release_status,
                }
                _store_candidate_prefilter_record(
                    ranked_item,
                    q_goal=q_hover,
                    start_q=start_q,
                    pos_err=hover_pos_err,
                    rot_err=hover_rot_err,
                    ik_score=pair_ik_score,
                    role="hover",
                )
                ranked.append(ranked_item)
            if early_stop and len(ranked) >= top_pairs:
                print(
                    f"[joint_search] {label} fast-chain IK early stop after "
                    f"{min(start_idx + len(chunk), len(remaining))}/{len(remaining)} place candidate(s); "
                    f"kept {len(ranked)} valid candidate(s)"
                )
                break
    finally:
        _set_world_collision_for_links(
            planner,
            disabled_world_collision_links,
            enabled=True,
            label=f"{label}_fast_chain_ik",
        )

    def _fast_chain_select_key(item) -> tuple:
        pair_score = float(item.get("pair_score", item.get("fast_chain_score", 0.0)) or 0.0)
        label_text = " ".join(str(item.get(k, "")) for k in ("variant_label", "label")).lower()
        if source_name == "bi":
            # For insertion, the cheap IK distance often prefers a small axial
            # spin that later fails the straight contact. Prefer the validated
            # raw 50mm approach family first, then use IK distance inside that tier.
            return (_bi_final_contact_validation_sort_key(item), pair_score, *_pre_place_screen_sort_key(item))
        return (pair_score, *_pre_place_screen_sort_key(item))

    ranked.sort(key=_fast_chain_select_key)
    planner._last_fast_chain_prefilter_records = list(ranked)
    selected = ranked[:top_pairs]
    hover_status_str = ", ".join(f"{k}={v}" for k, v in sorted(hover_status_counts.items()))
    release_status_str = ", ".join(f"{k}={v}" for k, v in sorted(release_status_counts.items()))
    print(
        f"[joint_search] {label} fast-chain IK rank kept {len(ranked)}/{len(remaining)} "
        f"candidate(s), selected {len(selected)}; "
        f"hover_statuses: {hover_status_str}; release_statuses: {release_status_str}"
    )
    if selected:
        print(
            f"[joint_search] {label} fast-chain top labels: "
            f"{[(item.get('label'), round(float(item.get('pair_score', item.get('fast_chain_score', 0.0))), 3)) for item in selected]}"
        )
    return selected


def _evaluate_curobo_pose_candidates_multi_start(
    planner,
    demo,
    args,
    candidates,
    *,
    label: str,
    use_attach: bool = False,
    timeout: float | None = None,
    max_attempts: int | None = None,
    num_ik_seeds: int | None = None,
    num_trajopt_seeds: int | None = None,
    num_graph_seeds: int | None = None,
    enable_graph: bool | None = None,
    max_winners: int | None = None,
    include_active_object: bool = False,
    include_table: bool = False,
    exclude_object_names: set[str] | None = None,
    disabled_world_collision_links: list[str] | None = None,
):
    _refresh_curobo_world(planner, demo, args, label=label, include_active_object=include_active_object, include_table=include_table, exclude_object_names=exclude_object_names)
    remaining = list(candidates or [])
    remaining.sort(key=_pre_place_screen_sort_key)
    if not remaining:
        return []

    ee_link_name = str(getattr(planner.config, "ee_link", "gripper_tcp"))
    if bool(getattr(args, "curobo_debug", False)):
        ee_link = _get_robot_link_by_name(demo, ee_link_name)
        base_pose = targeted.base.flatten_np(demo.robot.pose.p)[:3]
        print(
            f"[curobo] runtime frames for {label}: "
            f"planner_ee_link={ee_link_name}, "
            f"demo_tcp_link={_get_link_name(demo.tcp)}, "
            f"robot_base_world_p={np.round(base_pose, 6)}"
        )
        if ee_link is not None:
            print(
                f"[curobo] runtime link check for {label}: "
                f"ee_link_world_p={np.round(targeted.base.flatten_np(ee_link.pose.p)[:3], 6)}, "
                f"tcp_world_p={np.round(targeted.base.flatten_np(demo.tcp.pose.p)[:3], 6)}"
            )

    winners = []
    use_max_winners = int(len(remaining) if max_winners is None else max_winners)
    if use_max_winners <= 0:
        use_max_winners = len(remaining)
    saw_invalid_start = False
    saw_invalid_start_self_collision = False
    chunk_size = int(getattr(args, "curobo_batch_chunk_size", 64))
    if chunk_size <= 0:
        chunk_size = len(remaining)
    stage_size = int(getattr(args, "curobo_multistart_stage_size", 128) or 0)
    if (
        stage_size > 0
        and max_winners is not None
        and use_max_winners > 0
        and use_max_winners < len(remaining)
        and len(remaining) > stage_size
    ):
        staged_args = SimpleNamespace(**vars(args))
        staged_args.curobo_multistart_stage_size = 0
        staged_winners = []
        total_stages = int(np.ceil(float(len(remaining)) / float(stage_size)))
        for stage_idx, start_idx in enumerate(range(0, len(remaining), stage_size), start=1):
            if len(staged_winners) >= use_max_winners:
                break
            stage_candidates = remaining[start_idx : start_idx + stage_size]
            needed = max(1, use_max_winners - len(staged_winners))
            print(
                f"[curobo] {label} staged multi-start {stage_idx}/{total_stages}: "
                f"testing {len(stage_candidates)} candidate(s), need {needed} more winner(s)"
            )
            stage_winners = _evaluate_curobo_pose_candidates_multi_start(
                planner,
                demo,
                staged_args,
                stage_candidates,
                label=f"{label}_stage{stage_idx}",
                use_attach=use_attach,
                timeout=timeout,
                max_attempts=max_attempts,
                num_ik_seeds=num_ik_seeds,
                num_trajopt_seeds=num_trajopt_seeds,
                num_graph_seeds=num_graph_seeds,
                enable_graph=enable_graph,
                max_winners=needed,
                include_active_object=include_active_object,
                include_table=include_table,
                exclude_object_names=exclude_object_names,
                disabled_world_collision_links=disabled_world_collision_links,
            )
            staged_winners.extend(stage_winners)
        if staged_winners:
            staged_winners.sort(key=_candidate_sort_key)
            best = staged_winners[0]
            print(
                f"[curobo] selected {best['label']} from staged {label}: "
                f"score={best['score']:.3f}, tcp_verticality={float(best.get('tcp_verticality', 0.0)):.3f}, "
                f"waypoints={best['metrics']['waypoint_count']}"
            )
        else:
            print(f"[curobo] {label} staged multi-start found no winner after {total_stages} stage(s)")
        return staged_winners[:use_max_winners]
    ik_prefilter_pos_thresh = float(getattr(args, "curobo_ik_prefilter_position_threshold", 0.01) or 0.0)
    ik_prefilter_rot_thresh = float(getattr(args, "curobo_ik_prefilter_rotation_threshold", 0.25) or 0.0)
    use_soft_prefilter = ik_prefilter_pos_thresh > 0 and ik_prefilter_rot_thresh > 0
    ik_screened = []
    ik_status_counts: dict[str, int] = {}
    ik_pos_errors: list[float] = []
    ik_rot_errors: list[float] = []
    strict_pass_count = 0
    soft_pass_count = 0
    ik_screen_total = len(remaining)
    planner._last_candidate_count_in = int(ik_screen_total)
    planner._last_candidate_count_after_ik = 0
    planner._last_candidate_count_motiongen = 0
    reused_prefilter_count = 0
    for start_idx in range(0, len(remaining), chunk_size):
        chunk = remaining[start_idx : start_idx + chunk_size]
        need_ik = []
        for candidate in chunk:
            candidate_start_q = np.asarray(candidate["start_q"], dtype=np.float32).reshape(-1)[:7]
            reused_q = _candidate_reusable_prefilter_q(candidate, start_q=candidate_start_q)
            if reused_q is None:
                need_ik.append(candidate)
                continue
            reused_prefilter_count += 1
            pos_err = candidate.get("_prefilter_ik_pos_error")
            rot_err = candidate.get("_prefilter_ik_rot_error")
            if pos_err is not None and np.isfinite(pos_err):
                ik_pos_errors.append(float(pos_err))
            if rot_err is not None and np.isfinite(rot_err):
                ik_rot_errors.append(float(rot_err))
            ik_screened.append(candidate)
            strict_pass_count += 1
        if not need_ik:
            continue
        planner_poses = [
            _convert_demo_tcp_pose_to_curobo_ee_pose(
                demo,
                item["pose"],
                ee_link_name=ee_link_name,
            )
            for item in need_ik
        ]
        start_qs = [np.asarray(item["start_q"], dtype=np.float32).reshape(-1)[:7] for item in need_ik]
        ik_results = _profile_solve_batch_start_goal_ik(
            planner,
            start_qs,
            planner_poses,
            num_seeds=int(getattr(args, "curobo_num_ik_seeds", 64) if num_ik_seeds is None else num_ik_seeds),
        )
        for candidate, ik_result in zip(need_ik, ik_results):
            status_key = str(ik_result.status)
            ik_status_counts[status_key] = ik_status_counts.get(status_key, 0) + 1
            pos_err, rot_err = _ik_debug_errors(ik_result)
            if pos_err is not None and np.isfinite(pos_err):
                ik_pos_errors.append(float(pos_err))
            if rot_err is not None and np.isfinite(rot_err):
                ik_rot_errors.append(float(rot_err))
            if ik_result.success:
                _store_candidate_prefilter_record(
                    candidate,
                    q_goal=getattr(ik_result, "goal_joint", None),
                    start_q=np.asarray(candidate["start_q"], dtype=np.float32).reshape(-1)[:7],
                    pos_err=float(pos_err),
                    rot_err=float(rot_err),
                    role="hover",
                )
                ik_screened.append(candidate)
                strict_pass_count += 1
            elif use_soft_prefilter and pos_err is not None and rot_err is not None:
                if float(pos_err) <= ik_prefilter_pos_thresh and float(rot_err) <= ik_prefilter_rot_thresh:
                    _store_candidate_prefilter_record(
                        candidate,
                        q_goal=getattr(ik_result, "goal_joint", None),
                        start_q=np.asarray(candidate["start_q"], dtype=np.float32).reshape(-1)[:7],
                        pos_err=float(pos_err),
                        rot_err=float(rot_err),
                        role="hover",
                    )
                    ik_screened.append(candidate)
                    soft_pass_count += 1
    remaining = ik_screened
    remaining.sort(key=_pre_place_screen_sort_key)
    planner._last_candidate_count_after_ik = int(len(remaining))
    status_str = ", ".join(f"{k}={v}" for k, v in sorted(ik_status_counts.items()))
    soft_info = ""
    if use_soft_prefilter:
        soft_info = (
            f" (strict={strict_pass_count}, soft={soft_pass_count}, "
            f"soft_thresholds: pos<={ik_prefilter_pos_thresh:.4f} rot<={ik_prefilter_rot_thresh:.4f})"
        )
    print(
        f"[curobo] {label} IK prefilter kept {len(remaining)}/{ik_screen_total} "
        f"start-goal pair(s); reused_prefilter={reused_prefilter_count}; statuses: {status_str}{soft_info}"
    )
    if not remaining:
        if ik_pos_errors:
            pos_arr = np.array(ik_pos_errors)
            rot_arr = np.array(ik_rot_errors) if ik_rot_errors else np.array([float("nan")])
            print(
                f"[curobo][diag] {label} IK error summary across {len(ik_pos_errors)} failed pair(s): "
                f"pos_err min={float(pos_arr.min()):.6f} median={float(np.median(pos_arr)):.6f} max={float(pos_arr.max()):.6f}  "
                f"rot_err min={float(rot_arr.min()):.6f} median={float(np.median(rot_arr)):.6f} max={float(rot_arr.max()):.6f}"
            )
            pos_thresh = float(planner.config.position_threshold)
            rot_thresh = float(planner.config.rotation_threshold)
            near_pass = sum(1 for p, r in zip(ik_pos_errors, ik_rot_errors) if p < pos_thresh * 2.0 and r < rot_thresh * 2.0)
            print(
                f"[curobo][diag] {label} IK thresholds: position={pos_thresh:.4f} rotation={rot_thresh:.4f}  "
                f"candidates within 2x threshold: {near_pass}/{len(ik_pos_errors)}"
            )
        return []

    num_chunks = max(1, (len(remaining) + chunk_size - 1) // chunk_size)
    print(
        f"[curobo] {label} evaluating {len(remaining)} start-goal pair(s) "
        f"in {num_chunks} batch chunk(s) of size <= {chunk_size}"
    )
    planner._last_candidate_count_motiongen = int(len(remaining))

    if include_table and bool(getattr(args, "curobo_table_collision", True)):
        _refresh_curobo_world(
            planner, demo, args, label=f"{label}_trajopt",
            include_active_object=include_active_object,
            include_table=True,
            exclude_object_names=exclude_object_names,
        )

    def _build_transport_success_item(candidate, result, planner_pose, *, mode_label: str):
        candidate_label = str(candidate["label"])
        if not result.success or result.joint_path is None:
            return None
        start_q = np.asarray(candidate["start_q"], dtype=np.float32).reshape(-1)[:7]
        q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in result.joint_path]
        terminal_align = _validate_curobo_terminal_pose(
            demo,
            args,
            q_path,
            candidate["pose"],
            label=candidate_label,
        )
        if terminal_align is None:
            return None
        q_path = terminal_align["q_path"]
        if not _validate_candidate_joint_path_with_demo_planner(
            demo,
            start_q,
            q_path,
            use_attach=use_attach,
            label=candidate_label,
        ):
            return None
        metrics, score = _path_metrics_and_score(start_q, q_path)
        selection_penalty = _candidate_selection_penalty(candidate, args)
        place_pref_penalty = _candidate_place_orientation_penalty(candidate, demo, args)
        score = float(score) + float(selection_penalty) + float(place_pref_penalty)
        item = dict(candidate)
        item["result"] = result
        item["q_path"] = q_path
        item["metrics"] = metrics
        item["score"] = score
        item["selection_penalty"] = selection_penalty
        item["place_preference_penalty"] = place_pref_penalty
        item["terminal_align"] = terminal_align
        item["planner_pose"] = planner_pose
        item.setdefault("q_hover", q_path[-1])
        print(
            f"[curobo] {candidate_label} {mode_label} success: score={score:.3f}, "
            f"selection_penalty={selection_penalty:.3f}, "
            f"place_pref_penalty={place_pref_penalty:.3f}, "
            f"total_motion={metrics['total_motion']:.3f} rad, "
            f"joint7_total={metrics['joint7_total_motion']:.3f} rad, "
            f"joint7_excursion={metrics['joint7_max_excursion']:.3f} rad, "
            f"waypoints={metrics['waypoint_count']}, "
            f"realized_pos_err={terminal_align['realized_after']['pos_err']:.4f} m, "
            f"realized_rot_err={terminal_align['realized_after']['rot_err_deg']:.2f} deg"
        )
        return item

    disabled_world_collision_links = _set_world_collision_for_links(
        planner,
        disabled_world_collision_links,
        enabled=False,
        label=label,
    )
    try:
        q_goal_trial_cap = int(getattr(args, "transport_prefilter_q_goal_max_trials", 1) or 0)
        if bool(getattr(args, "transport_use_prefilter_q_goal", True)) and q_goal_trial_cap > 0:
            q_goal_trials = 0
            q_goal_winner_ids: set[int] = set()
            q_goal_failures: list[tuple[str, str]] = []
            for candidate in remaining:
                if q_goal_trials >= q_goal_trial_cap or len(winners) >= use_max_winners:
                    break
                candidate_label = str(candidate["label"])
                start_q = np.asarray(candidate["start_q"], dtype=np.float32).reshape(-1)[:7]
                q_goal = _candidate_reusable_prefilter_q(candidate, start_q=start_q)
                if q_goal is None:
                    continue
                q_goal_trials += 1
                planner_pose = _convert_demo_tcp_pose_to_curobo_ee_pose(
                    demo,
                    candidate["pose"],
                    ee_link_name=ee_link_name,
                )
                _bump_profile_counter("prefilter_q_goal_motiongen_count")
                js_result = _profile_plan_to_joint_state(
                    planner,
                    start_q,
                    q_goal,
                    enable_graph=False,
                    max_attempts=int(getattr(args, "transport_prefilter_q_goal_max_attempts", 1) or 1),
                    timeout=float(getattr(args, "transport_prefilter_q_goal_timeout", 2.0) or 2.0),
                    num_trajopt_seeds=int(getattr(args, "transport_prefilter_q_goal_num_trajopt_seeds", 1) or 1),
                    num_graph_seeds=1,
                )
                if not js_result.success or js_result.joint_path is None:
                    q_goal_failures.append((candidate_label, str(js_result.status)))
                    if str(js_result.status) == "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION":
                        saw_invalid_start = True
                    elif str(js_result.status) == "MotionGenStatus.INVALID_START_STATE_SELF_COLLISION":
                        saw_invalid_start_self_collision = True
                    continue
                item = _build_transport_success_item(
                    candidate,
                    js_result,
                    planner_pose,
                    mode_label="prefilter-q-goal",
                )
                if item is None:
                    q_goal_failures.append((candidate_label, "TERMINAL_OR_DEMO_VALIDATION_FAIL"))
                    continue
                _bump_profile_counter("prefilter_q_goal_success_count")
                winners.append(item)
                q_goal_winner_ids.add(id(candidate))
            if q_goal_trials:
                print(
                    f"[curobo] {label} prefilter-q-goal joint-space trial: "
                    f"{len(winners)} winner(s) from {q_goal_trials} trial(s)"
                )
            if q_goal_failures:
                by_status = Counter(st for _, st in q_goal_failures)
                by_label = Counter(lbl for lbl, _ in q_goal_failures)
                print(
                    f"[curobo] {label} prefilter-q-goal failures: "
                    f"{len(q_goal_failures)}/{q_goal_trials} trial(s) failed | "
                    f"statuses={dict(by_status)} | top_labels={by_label.most_common(6)}"
                )
            if len(winners) >= use_max_winners:
                winners.sort(key=_candidate_sort_key)
                return winners[:use_max_winners]
            if q_goal_winner_ids:
                remaining = [item for item in remaining if id(item) not in q_goal_winner_ids]
                if not remaining:
                    winners.sort(key=_candidate_sort_key)
                    return winners[:use_max_winners]
                num_chunks = max(1, (len(remaining) + chunk_size - 1) // chunk_size)
        if use_max_winners == 1 and len(remaining) > 1:
            start_q0 = np.asarray(remaining[0]["start_q"], dtype=np.float32).reshape(-1)[:7]
            same_start = all(
                np.allclose(
                    start_q0,
                    np.asarray(item["start_q"], dtype=np.float32).reshape(-1)[:7],
                    atol=1e-5,
                    rtol=0.0,
                )
                for item in remaining[1:]
            )
            if same_start:
                planner_poses = [
                    _convert_demo_tcp_pose_to_curobo_ee_pose(
                        demo,
                        item["pose"],
                        ee_link_name=ee_link_name,
                    )
                    for item in remaining
                ]
                print(f"[curobo] {label} using goalset fast-path for {len(remaining)} same-start target(s)")
                goalset_result = _profile_plan_goalset_to_poses(
                    planner,
                    start_q0,
                    planner_poses,
                    enable_graph=bool(getattr(args, "curobo_enable_graph", False) if enable_graph is None else enable_graph),
                    max_attempts=int(getattr(args, "curobo_max_attempts", 2) if max_attempts is None else max_attempts),
                    timeout=float(getattr(args, "curobo_timeout", 5.0) if timeout is None else timeout),
                    num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64) if num_ik_seeds is None else num_ik_seeds),
                    num_trajopt_seeds=int(
                        getattr(args, "curobo_num_trajopt_seeds", 1) if num_trajopt_seeds is None else num_trajopt_seeds
                    ),
                    num_graph_seeds=int(
                        getattr(args, "curobo_num_graph_seeds", 1) if num_graph_seeds is None else num_graph_seeds
                    ),
                )
                goal_idx = int((goalset_result.debug or {}).get("goalset_index", -1))
                if goalset_result.success and goalset_result.joint_path is not None and 0 <= goal_idx < len(remaining):
                    candidate = remaining[goal_idx]
                    candidate_label = str(candidate["label"])
                    q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in goalset_result.joint_path]
                    terminal_align = _validate_curobo_terminal_pose(
                        demo,
                        args,
                        q_path,
                        candidate["pose"],
                        label=candidate_label,
                    )
                    if terminal_align is not None:
                        q_path = terminal_align["q_path"]
                        if _validate_candidate_joint_path_with_demo_planner(
                            demo,
                            start_q0,
                            q_path,
                            use_attach=use_attach,
                            label=candidate_label,
                        ):
                            metrics, score = _path_metrics_and_score(start_q0, q_path)
                            selection_penalty = _candidate_selection_penalty(candidate, args)
                            place_pref_penalty = _candidate_place_orientation_penalty(candidate, demo, args)
                            score = float(score) + float(selection_penalty) + float(place_pref_penalty)
                            item = dict(candidate)
                            item["result"] = goalset_result
                            item["q_path"] = q_path
                            item["metrics"] = metrics
                            item["score"] = score
                            item["selection_penalty"] = selection_penalty
                            item["place_preference_penalty"] = place_pref_penalty
                            item["terminal_align"] = terminal_align
                            item["planner_pose"] = planner_poses[goal_idx]
                            item.setdefault("q_hover", q_path[-1])
                            winners.append(item)
                            print(
                                f"[curobo] {candidate_label} goalset success: score={score:.3f}, "
                                f"selection_penalty={selection_penalty:.3f}, "
                                f"place_pref_penalty={place_pref_penalty:.3f}, "
                                f"waypoints={metrics['waypoint_count']}"
                            )
                            return winners
                else:
                    print(f"[curobo] {label} goalset fast-path failed with status={goalset_result.status}; falling back to pair batch")
                    _bump_profile_counter("fallback_count")
                    if str(goalset_result.status) == "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION":
                        saw_invalid_start = True
                    elif str(goalset_result.status) == "MotionGenStatus.INVALID_START_STATE_SELF_COLLISION":
                        saw_invalid_start_self_collision = True
        for chunk_idx, start_idx in enumerate(range(0, len(remaining), chunk_size), start=1):
            chunk = remaining[start_idx : start_idx + chunk_size]
            planner_poses = [
                _convert_demo_tcp_pose_to_curobo_ee_pose(
                    demo,
                    item["pose"],
                    ee_link_name=ee_link_name,
                )
                for item in chunk
            ]
            start_qs = [np.asarray(item["start_q"], dtype=np.float32).reshape(-1)[:7] for item in chunk]
            if num_chunks > 1:
                print(
                    f"[curobo] {label} batch chunk {chunk_idx}/{num_chunks}: "
                    f"{len(chunk)} pair(s)"
                )
            batch_results = _profile_plan_batch_start_goal_pairs(
                planner,
                start_qs,
                planner_poses,
                enable_graph=bool(getattr(args, "curobo_enable_graph", False) if enable_graph is None else enable_graph),
                max_attempts=int(getattr(args, "curobo_max_attempts", 2) if max_attempts is None else max_attempts),
                timeout=float(getattr(args, "curobo_timeout", 5.0) if timeout is None else timeout),
                num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64) if num_ik_seeds is None else num_ik_seeds),
                num_trajopt_seeds=int(
                    getattr(args, "curobo_num_trajopt_seeds", 1) if num_trajopt_seeds is None else num_trajopt_seeds
                ),
                num_graph_seeds=int(
                    getattr(args, "curobo_num_graph_seeds", 1) if num_graph_seeds is None else num_graph_seeds
                ),
            )
            batch_failures: list[tuple[str, str]] = []
            for candidate, planner_pose, result in zip(chunk, planner_poses, batch_results):
                candidate_label = str(candidate["label"])
                if not result.success or result.joint_path is None:
                    batch_failures.append((candidate_label, str(result.status)))
                    if str(result.status) == "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION":
                        saw_invalid_start = True
                    elif str(result.status) == "MotionGenStatus.INVALID_START_STATE_SELF_COLLISION":
                        saw_invalid_start_self_collision = True
                    continue
                start_q = np.asarray(candidate["start_q"], dtype=np.float32).reshape(-1)[:7]
                q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in result.joint_path]
                terminal_align = _validate_curobo_terminal_pose(
                    demo,
                    args,
                    q_path,
                    candidate["pose"],
                    label=candidate_label,
                )
                if terminal_align is None:
                    continue
                q_path = terminal_align["q_path"]
                if not _validate_candidate_joint_path_with_demo_planner(
                    demo,
                    start_q,
                    q_path,
                    use_attach=use_attach,
                    label=candidate_label,
                ):
                    continue
                metrics, score = _path_metrics_and_score(start_q, q_path)
                selection_penalty = _candidate_selection_penalty(candidate, args)
                place_pref_penalty = _candidate_place_orientation_penalty(candidate, demo, args)
                score = float(score) + float(selection_penalty) + float(place_pref_penalty)
                item = dict(candidate)
                item["result"] = result
                item["q_path"] = q_path
                item["metrics"] = metrics
                item["score"] = score
                item["selection_penalty"] = selection_penalty
                item["place_preference_penalty"] = place_pref_penalty
                item["terminal_align"] = terminal_align
                item["planner_pose"] = planner_pose
                item.setdefault("q_hover", q_path[-1])
                print(
                    f"[curobo] {candidate_label} success: score={score:.3f}, "
                    f"selection_penalty={selection_penalty:.3f}, "
                    f"place_pref_penalty={place_pref_penalty:.3f}, "
                    f"total_motion={metrics['total_motion']:.3f} rad, "
                    f"joint7_total={metrics['joint7_total_motion']:.3f} rad, "
                    f"joint7_excursion={metrics['joint7_max_excursion']:.3f} rad, "
                    f"waypoints={metrics['waypoint_count']}, "
                    f"realized_pos_err={terminal_align['realized_after']['pos_err']:.4f} m, "
                    f"realized_rot_err={terminal_align['realized_after']['rot_err_deg']:.2f} deg"
                )
                winners.append(item)
                if len(winners) >= use_max_winners:
                    break
            if batch_failures:
                by_status = Counter(st for _, st in batch_failures)
                by_label = Counter(lbl for lbl, _ in batch_failures)
                print(
                    f"[curobo] {label} batch chunk {chunk_idx}/{num_chunks}: "
                    f"{len(batch_failures)}/{len(chunk)} pair(s) failed | "
                    f"statuses={dict(by_status)} | top_labels={by_label.most_common(6)}"
                )
            if len(winners) >= use_max_winners:
                break
    finally:
        _set_world_collision_for_links(
            planner,
            disabled_world_collision_links,
            enabled=True,
            label=label,
        )

    if not winners and saw_invalid_start and remaining:
        start_q0 = np.asarray(remaining[0]["start_q"], dtype=np.float32).reshape(-1)[:7]
        targeted.base.print_failure_diagnostics(
            demo,
            args,
            f"{label}_start_state",
            q_target=start_q0,
            use_attach=use_attach,
        )
        start_diag = planner.diagnose_start_state_world_collision(start_q0)
        ablations = list(start_diag.get("ablation", []) or [])
        if ablations:
            for item in ablations:
                print(
                    f"[curobo] {label}_start_state ablation remove={item['removed']}: "
                    f"valid={item['valid']}, status={item['status']}"
                )
        else:
            print(
                f"[curobo] {label}_start_state cuRobo world diagnosis: "
                f"valid={start_diag.get('valid')}, status={start_diag.get('status')}, "
                f"world_obstacles={start_diag.get('world_obstacle_names')}"
            )
    if not winners and saw_invalid_start_self_collision and remaining:
        _print_start_state_self_collision_diagnostics(
            planner,
            np.asarray(remaining[0]["start_q"], dtype=np.float32).reshape(-1)[:7],
            label=f"{label}_start_state",
        )

    winners.sort(key=_candidate_sort_key)
    return winners


def _make_long_axis_perpendicular_topdown_grasp_pose(
    base_pose,
    long_axis_world: np.ndarray | None,
    *,
    label: str,
):
    if long_axis_world is None:
        return None
    long_axis = np.asarray(long_axis_world, dtype=np.float32).reshape(3).copy()
    long_axis[2] = 0.0
    long_axis = _normalize(long_axis)
    if long_axis is None:
        return None

    # TCP +Y is the gripper pad opening axis.  For pen-like objects it must be
    # perpendicular to the object's long axis; TCP +X is aligned with the long
    # axis and TCP +Z points down into the object.
    approach_axis = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    pad_open_axis = _normalize(np.cross(approach_axis, long_axis))
    if pad_open_axis is None:
        return None
    long_axis = _normalize(np.cross(pad_open_axis, approach_axis))
    if long_axis is None:
        return None
    R_tcp = np.stack([long_axis, pad_open_axis, approach_axis], axis=1).astype(np.float32)
    q_tcp = targeted.base.bridge_mod_mat2quat(R_tcp).astype(np.float32)
    pose = targeted.Pose.create_from_pq(
        p=targeted.base.flatten_np(base_pose.p)[:3].astype(np.float32),
        q=q_tcp,
    )
    print(
        f"[direct_grasp] {label}: long-axis perpendicular frame "
        f"tcp_x_dot_long={float(np.dot(R_tcp[:, 0], long_axis)):.3f}, "
        f"tcp_y_dot_long={float(np.dot(R_tcp[:, 1], long_axis)):.3f}, "
        f"tcp_z_world_z={float(R_tcp[2, 2]):.3f}, "
        f"long_axis_xy={np.round(long_axis, 4).tolist()}"
    )
    return pose


def _build_direct_grasp_candidates(
    demo,
    args,
    *,
    bridge_mod=None,
    scene_capture_cache=None,
    place_state_cache=None,
):
    raw_grasp_pose = demo.build_topdown_grasp_pose()
    obj_p0, obj_q0 = demo.get_obj_pose()
    T_world_obj0 = targeted.base.pose_to_matrix(obj_p0, obj_q0)
    grasp_mode = str(getattr(args, "grasp_mode", "object_normal") or "object_normal").strip().lower()
    place_rule = targeted.get_place_rule(getattr(args, "object_name", None))
    source_name = _current_source_object_name(args)
    grasp_variant_args = SimpleNamespace(**vars(args))
    grasp_variant_args.topdown_grasp_yaw_variant_deg = [0.0]
    orientation_invariant_place = bool(getattr(place_rule, "orientation_invariant", False))
    object_category = _current_object_category(args, place_rule)
    sphere_category = object_category == "sphere"
    if sphere_category:
        grasp_variant_args.topdown_grasp_yaw_variant_deg = _unique_finite_float_list(
            getattr(args, "sphere_grasp_yaw_variant_deg", [0.0, -45.0, 45.0, -90.0, 90.0, 180.0])
        )
        grasp_variant_args.topdown_tilt_toward_robot_deg = []
        grasp_variant_args.topdown_tilt_toward_robot_shift_m = [0.0]
    object_axis_world = None
    object_long_axis_idx = None
    object_long_axis_half = None
    object_extents = None
    try:
        extents = np.asarray(
            targeted.base.get_asset_box_size(args.sim_asset_file, args.sim_asset_scale),
            dtype=np.float32,
        ).reshape(3)
        object_extents = extents
        axis_idx = int(np.argmax(extents))
        object_long_axis_idx = axis_idx
        object_long_axis_half = float(extents[axis_idx]) * 0.5

        is_spherical = sphere_category
        if sphere_category:
            reason = "sphere category"
            print(
                f"[direct_grasp] detected {reason} (extents={np.round(extents * 1000, 1).tolist()}mm), "
                "disabling axis shifts, z lifts, and tilt variants; keeping equivalent yaw variants"
            )
            object_axis_world = None
            object_long_axis_half = None
        elif float(np.max(extents)) >= 1.5 * float(max(np.sort(extents)[1], 1e-6)):
            T_world_obj = targeted.base.pose_to_matrix(*demo.get_obj_pose())
            axis_local = np.zeros(3, dtype=np.float32)
            axis_local[axis_idx] = 1.0
            object_axis_world = _normalize(T_world_obj[:3, :3] @ axis_local)
    except Exception:
        object_axis_world = None
        is_spherical = False

    if source_name == "bi":
        pen_long_axis_world = None
        try:
            pen_long_axis_world = _fixed_tabletop_horizontal_long_axis(
                args,
                T_world_obj0,
                object_dims=object_extents,
                fallback_axis=object_axis_world,
            )
        except Exception:
            pen_long_axis_world = None
        if pen_long_axis_world is None and object_axis_world is not None:
            pen_long_axis_world = object_axis_world
        pen_grasp_pose = _make_long_axis_perpendicular_topdown_grasp_pose(
            raw_grasp_pose,
            pen_long_axis_world,
            label="bi",
        )
        if pen_grasp_pose is not None:
            raw_grasp_pose = pen_grasp_pose

    fixed_tabletop_place_first_candidates: list[dict] = []
    if _is_fixed_tabletop_source(source_name):
        fixed_tabletop_place_first_candidates = _build_fixed_tabletop_place_first_grasp_candidates(
            demo,
            args,
            place_rule,
            bridge_mod,
            scene_capture_cache,
            place_state_cache,
            T_world_obj0,
            object_dims=object_extents,
        )
        return fixed_tabletop_place_first_candidates

    rule_grasp_bias_variants = list(getattr(place_rule, "grasp_bias_variants", ()) or []) if place_rule is not None else []
    use_rule_bias_variants = bool(rule_grasp_bias_variants) and object_axis_world is not None
    rule_top_bias_axis_sign = None
    rule_top_bias_dot_up = None

    if (
        use_rule_bias_variants
        and place_rule is not None
        and bool(getattr(place_rule, "preserve_long_axis_vertical", False))
        and object_long_axis_idx is not None
    ):
        try:
            target_name = curobo_wrapper.normalize_object_name(getattr(place_rule, "target_object_name", None))
            T_world_target = targeted._get_scene_object_world_transform(demo, bridge_mod, scene_capture_cache, target_name)
            target_up_axis = None
            if T_world_target is not None:
                target_up_axis = targeted._target_place_up_axis(place_rule, T_world_target)
            target_up_axis = _normalize(np.array([0.0, 0.0, 1.0], dtype=np.float32) if target_up_axis is None else target_up_axis)
            pose_spec = None
            if getattr(place_rule, "primitive", "") == "place_on_slots":
                slots = list(getattr(place_rule, "slots", ()) or [])
                if slots:
                    pose_spec = getattr(slots[0], "object_pose_local", None)
            else:
                pose_spec = getattr(place_rule, "object_pose_local", None)
            if T_world_target is not None and target_up_axis is not None and pose_spec is not None:
                axis_local = np.zeros(3, dtype=np.float32)
                axis_local[int(object_long_axis_idx)] = 1.0
                T_target_obj = targeted._local_pose_spec_to_matrix(pose_spec)
                T_world_obj_desired = np.asarray(T_world_target, dtype=np.float32).reshape(4, 4) @ T_target_obj
                target_axis_world = _normalize(T_world_obj_desired[:3, :3] @ axis_local)
                if target_axis_world is not None:
                    rule_top_bias_dot_up = float(np.dot(target_axis_world, target_up_axis))
                    rule_top_bias_axis_sign = 1.0 if rule_top_bias_dot_up >= 0.0 else -1.0
        except Exception as exc:
            print(f"[direct_grasp] vertical top-bias axis remap unavailable: {exc}")

    def _rule_bias_actual_axis_shift(rule_bias) -> float:
        axis_shift = float(getattr(rule_bias, "axis_shift_m", 0.0))
        if rule_top_bias_axis_sign is None or abs(axis_shift) <= 1e-6:
            return axis_shift
        label = str(getattr(rule_bias, "label", "") or "").lower()
        if "top_bias" not in label:
            return axis_shift
        return float(rule_top_bias_axis_sign) * abs(axis_shift)

    def _rule_bias_display_label(label: str, actual_axis_shift: float) -> str:
        text = str(label or "grasp")
        if "top_bias" not in text.lower() or abs(float(actual_axis_shift)) <= 1e-6:
            return text
        sign_label = "pos" if float(actual_axis_shift) > 0.0 else "neg"
        magnitude_mm = int(round(abs(float(actual_axis_shift)) * 1000.0))
        return re.sub(r"top_bias_(?:neg|pos)\d+", f"top_bias_{sign_label}{magnitude_mm}", text)

    if use_rule_bias_variants:
        grasp_axis_shifts = [_rule_bias_actual_axis_shift(v) for v in rule_grasp_bias_variants]
    else:
        grasp_axis_shifts = _unique_finite_float_list(
            getattr(args, "direct_grasp_object_axis_shifts_m", [0.0]),
        )
    if not grasp_axis_shifts:
        grasp_axis_shifts = [0.0]

    # 球形物体：强制只使用中心抓取
    if sphere_category:
        grasp_axis_shifts = [0.0]
    max_axis_shift_ratio = float(max(getattr(args, "direct_grasp_max_axis_shift_ratio", 0.35), 0.0))
    if object_long_axis_half is not None and max_axis_shift_ratio > 0:
        max_abs_shift = object_long_axis_half * max_axis_shift_ratio
        before_count = len(grasp_axis_shifts)
        grasp_axis_shifts = [s for s in grasp_axis_shifts if abs(float(s)) <= max_abs_shift + 1e-6]
        if not grasp_axis_shifts:
            grasp_axis_shifts = [0.0]
        if len(grasp_axis_shifts) < before_count:
            print(
                f"[direct_grasp] axis_shift filter: object half-length={object_long_axis_half * 1000:.1f}mm, "
                f"max_ratio={max_axis_shift_ratio:.2f} -> max_shift={max_abs_shift * 1000:.1f}mm, "
                f"kept {len(grasp_axis_shifts)}/{before_count} shift value(s)"
            )
    if use_rule_bias_variants:
        grasp_z_lifts = [float(v.z_lift_m) for v in rule_grasp_bias_variants]
    else:
        grasp_z_lifts = _unique_finite_float_list(
            getattr(args, "direct_grasp_z_lifts_m", [0.0]),
            min_value=0.0,
        )
    if not grasp_z_lifts:
        grasp_z_lifts = [0.0]

    # 球形物体：禁用z lift
    if sphere_category:
        grasp_z_lifts = [0.0]

    elongated_modes = {"topdown_long_axis", "pen_topdown_insert_ready", "long_axis_adaptive"}

    def _pose_key(pose):
        return (
            tuple(np.round(targeted.base.flatten_np(pose.p)[:3], 5).tolist()),
            tuple(np.round(targeted.base.flatten_np(pose.q)[:4], 5).tolist()),
        )

    def _build_pose_variants():
        variants = []
        seen = set()

        def _append(label, pose):
            if pose is None:
                return
            key = (str(label), _pose_key(pose)) if use_rule_bias_variants else _pose_key(pose)
            if key in seen:
                return
            seen.add(key)
            variants.append((str(label), pose))

        base_orientation_variants: list[tuple[str, object]] = []
        if use_rule_bias_variants:
            for bias in rule_grasp_bias_variants:
                pose = raw_grasp_pose
                tilt_deg = float(bias.tilt_toward_robot_deg)
                if abs(tilt_deg) > 1e-6:
                    pose = targeted.base.tilt_pose_toward_robot(
                        demo,
                        pose,
                        tilt_deg,
                        direction=str(getattr(bias, "tilt_direction", "toward_robot")),
                    )
                    if pose is None:
                        continue
                shift_d = float(bias.tilt_shift_m)
                if abs(shift_d) > 1e-6:
                    pose = targeted.base.shift_pose_toward_robot_xy(demo, pose, shift_d)
                    if pose is None:
                        continue
                label = str(bias.label) if getattr(bias, "label", None) else "grasp"
                _append(label, pose)
        else:
            for label, pose in targeted.base.build_grasp_pose_variants(demo, raw_grasp_pose, grasp_variant_args):
                _append(label, pose)
            base_orientation_variants = list(variants)

        if use_rule_bias_variants:
            tilt_degs = []
        else:
            tilt_degs = _unique_finite_float_list(getattr(args, "direct_grasp_tilt_toward_robot_deg", [12.0, 20.0, 30.0, 45.0]))
        # 球形物体：禁用tilt variants，因为倾斜抓取容易滑脱
        if sphere_category:
            tilt_degs = []

        if use_rule_bias_variants:
            shift_ds = [0.0]
        else:
            shift_ds = _unique_finite_float_list(
                getattr(args, "direct_grasp_tilt_toward_robot_shift_m", [0.0, 0.02]),
                min_value=0.0,
            )
        if not shift_ds:
            shift_ds = [0.0]

        for tilt_deg in tilt_degs:
            if abs(float(tilt_deg)) <= 1e-6:
                continue
            tilted_pose = targeted.base.tilt_pose_toward_robot(demo, raw_grasp_pose, float(tilt_deg))
            if tilted_pose is None:
                continue
            for shift_d in shift_ds:
                shifted_pose = tilted_pose if float(shift_d) <= 1e-6 else targeted.base.shift_pose_toward_robot_xy(
                    demo,
                    tilted_pose,
                    float(shift_d),
                )
                if shifted_pose is None:
                    continue
                base_label = f"grasp_tilt_{int(round(abs(float(tilt_deg))))}deg"
                if float(shift_d) > 1e-6:
                    base_label += f"_shift_{int(round(1000.0 * float(shift_d)))}mm"
                _append(base_label, shifted_pose)

        if source_name == "shuazi" and not use_rule_bias_variants:
            shuazi_tilt_degs = [
                float(v)
                for v in _unique_finite_float_list(getattr(args, "shuazi_tilted_yaw_grasp_deg", [30.0]))
                if abs(float(v)) > 1e-6
            ]
            for yaw_label, yaw_pose in base_orientation_variants:
                yaw_label_text = str(yaw_label or "")
                if "yaw" not in yaw_label_text.lower():
                    continue
                for tilt_deg in shuazi_tilt_degs:
                    tilted_pose = targeted.base.tilt_pose_toward_robot(demo, yaw_pose, float(tilt_deg))
                    if tilted_pose is None:
                        continue
                    _append(f"{yaw_label_text}_tilt_{int(round(abs(float(tilt_deg))))}deg", tilted_pose)

        return variants

    grasp_variants = _build_pose_variants()
    candidates = []
    seen = set()
    if use_rule_bias_variants:
        bias_iter = []
        for grasp_variant_label, grasp_variant_pose in grasp_variants:
            matched_bias = next(
                (bias for bias in rule_grasp_bias_variants if str(getattr(bias, "label", "")) == str(grasp_variant_label)),
                None,
            )
            if matched_bias is None:
                continue
            bias_iter.append(((grasp_variant_label, grasp_variant_pose), matched_bias))
    else:
        bias_iter = [((grasp_variant_label, grasp_variant_pose), None) for grasp_variant_label, grasp_variant_pose in grasp_variants]

    for grasp_variant_item, rule_bias in bias_iter:
        grasp_variant_label, grasp_variant_pose = grasp_variant_item
        axis_shift_values = [_rule_bias_actual_axis_shift(rule_bias)] if rule_bias is not None else grasp_axis_shifts
        z_lift_values = [float(rule_bias.z_lift_m)] if rule_bias is not None else grasp_z_lifts
        for axis_shift in axis_shift_values:
            for z_lift in z_lift_values:
                base_grasp_pose = grasp_variant_pose
                if object_axis_world is not None and abs(float(axis_shift)) > 1e-6:
                    shifted_p = (_get_pose_position(base_grasp_pose) + object_axis_world * float(axis_shift)).astype(np.float32)
                    base_grasp_pose = targeted.base.make_pose_with_position(base_grasp_pose, shifted_p)
                if float(z_lift) > 1e-6:
                    shifted_p = (_get_pose_position(base_grasp_pose) + np.array([0.0, 0.0, float(z_lift)], dtype=np.float32)).astype(
                        np.float32
                    )
                    base_grasp_pose = targeted.base.make_pose_with_position(base_grasp_pose, shifted_p)
                approach_roll_values = [0.0]
                if source_name == "bi":
                    approach_roll_values = _bi_direct_grasp_approach_roll_degs(args)
                elif source_name == "shuazi":
                    approach_roll_values = _shuazi_direct_grasp_approach_roll_degs(args)
                for approach_roll_deg in approach_roll_values:
                    current_grasp_pose = _roll_pose_about_tcp_approach(base_grasp_pose, float(approach_roll_deg))
                    current_pregrasp_pose = demo.build_pregrasp_pose(current_grasp_pose)
                    current_grasp_pose, current_pregrasp_pose, geometry_grasp_raise = targeted.base.enforce_topdown_grasp_insertion_limit(
                        demo,
                        args,
                        current_grasp_pose,
                        current_pregrasp_pose,
                    )
                    if geometry_grasp_raise > 0:
                        print(
                            f"[safety] raised grasp TCP z by {geometry_grasp_raise:.4f} m "
                            f"to satisfy topdown_grasp_max_insertion_depth={args.topdown_grasp_max_insertion_depth:.4f}"
                        )
                    current_grasp_pose, _, grasp_tcp_raise = targeted.base.enforce_min_grasp_tcp_z(
                        current_grasp_pose,
                        current_pregrasp_pose,
                        args.min_grasp_tcp_z,
                    )
                    if grasp_tcp_raise > 0:
                        print(
                            f"[safety] raised grasp TCP z by {grasp_tcp_raise:.4f} m "
                            f"to satisfy min_grasp_tcp_z={args.min_grasp_tcp_z:.4f}"
                        )
                    key = (
                        tuple(np.round(targeted.base.flatten_np(current_grasp_pose.p)[:3], 5).tolist()),
                        tuple(np.round(targeted.base.flatten_np(current_grasp_pose.q)[:4], 5).tolist()),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    display_grasp_variant_label = (
                        _rule_bias_display_label(grasp_variant_label, float(axis_shift))
                        if rule_bias is not None
                        else grasp_variant_label
                    )
                    label = (
                        "grasp_direct"
                        if display_grasp_variant_label in (None, "grasp")
                        else f"grasp_direct_{display_grasp_variant_label}"
                    )
                    if abs(float(axis_shift)) > 1e-6:
                        label = f"{label}_axis_{int(round(float(axis_shift) * 1000.0))}mm"
                    if float(z_lift) > 1e-6:
                        label = f"{label}_lift_{int(round(float(z_lift) * 1000.0))}mm"
                    if abs(float(approach_roll_deg)) > 1e-6:
                        label = f"{label}_roll_{int(round(float(approach_roll_deg)))}deg"
                    candidates.append(
                        {
                            "label": label,
                            "pose": current_grasp_pose,
                            "pregrasp_pose": current_pregrasp_pose,  # 保存pregrasp pose用于两步抓取
                            "T_tcp_obj": np.linalg.inv(
                                targeted.base.pose_to_matrix(
                                    targeted.base.flatten_np(current_grasp_pose.p)[:3],
                                    targeted.base.flatten_np(current_grasp_pose.q)[:4],
                                )
                            ) @ T_world_obj0,
                            "grasp_axis_shift_m": float(axis_shift),
                            "grasp_z_lift_m": float(z_lift),
                            "grasp_approach_roll_deg": float(approach_roll_deg),
                        }
                    )
    def _display_variant_label_for_print(label):
        if not use_rule_bias_variants:
            return label
        matched_bias = next(
            (bias for bias in rule_grasp_bias_variants if str(getattr(bias, "label", "")) == str(label)),
            None,
        )
        if matched_bias is None:
            return label
        return _rule_bias_display_label(label, _rule_bias_actual_axis_shift(matched_bias))

    variant_labels = [_display_variant_label_for_print(label) for label, _ in grasp_variants]
    tilt_variants = [l for l in variant_labels if "tilt" in l.lower()]
    if sphere_category and tilt_variants:
        before = len(candidates)
        candidates = [c for c in candidates if "tilt" not in str(c.get("label", "")).lower()]
        grasp_variants = [(label, pose) for label, pose in grasp_variants if "tilt" not in str(label).lower()]
        print(
            f"[direct_grasp] sphere category filtered tilt grasp candidates: "
            f"{before}->{len(candidates)}"
        )
        variant_labels = [_display_variant_label_for_print(label) for label, _ in grasp_variants]
        tilt_variants = []
    if fixed_tabletop_place_first_candidates:
        merged = []
        merged_seen = set()
        for item in list(candidates or []) + list(fixed_tabletop_place_first_candidates or []):
            pose = item.get("pose")
            try:
                pose_key = (
                    tuple(np.round(targeted.base.flatten_np(pose.p)[:3], 5).tolist()),
                    tuple(np.round(targeted.base.flatten_np(pose.q)[:4], 5).tolist()),
                )
            except Exception:
                pose_key = str(item.get("label", ""))
            tcp_obj = item.get("T_tcp_obj")
            try:
                tcp_key = tuple(np.round(np.asarray(tcp_obj, dtype=np.float32).reshape(4, 4), 5).reshape(-1).tolist())
            except Exception:
                tcp_key = ()
            key = (pose_key, tcp_key)
            if key in merged_seen:
                continue
            merged_seen.add(key)
            merged.append(item)
        print(
            f"[direct_grasp] {source_name}: merged {len(candidates)} raw/current-pose candidate(s) "
            f"with {len(fixed_tabletop_place_first_candidates)} fixed-place candidate(s) -> {len(merged)} total"
        )
        candidates = merged
    non_tilt_cands = [c for c in candidates if "tilt" not in str(c["label"]).lower()]
    tilt_cands = [c for c in candidates if "tilt" in str(c["label"]).lower()]
    if tilt_cands and non_tilt_cands:
        interleaved = []
        ratio = max(1, len(non_tilt_cands) // max(len(tilt_cands), 1))
        ti = 0
        ni = 0
        while ni < len(non_tilt_cands) or ti < len(tilt_cands):
            for _ in range(min(ratio, len(non_tilt_cands) - ni)):
                interleaved.append(non_tilt_cands[ni])
                ni += 1
            if ti < len(tilt_cands):
                interleaved.append(tilt_cands[ti])
                ti += 1
        candidates = interleaved
    if source_name == "lvmukuai":
        def _lvmukuai_candidate_order(item):
            label = str(item.get("label", "")).lower()
            is_vertical_helper = _is_fixed_tabletop_vertical_helper_label(label, source_name)
            is_tilted = "tilt" in label
            z_lift = max(float(item.get("grasp_z_lift_m", 0.0) or 0.0), 0.0)
            tilt_mag = 0.0
            tilt_match = re.search(r"tilt_(?:toward|away)_robot_([-+]?\d+(?:\.\d+)?)deg", label)
            if tilt_match:
                tilt_mag = abs(float(tilt_match.group(1)))
            is_plain_direct = (not is_tilted) and z_lift <= 1e-4 and "lift" not in label
            if is_vertical_helper:
                relation_class = 0
            elif is_tilted and abs(tilt_mag - 20.0) <= 1.0:
                relation_class = 1
            elif is_plain_direct:
                relation_class = 2
            elif is_tilted:
                relation_class = 3
            elif z_lift >= 0.005 or "lift_" in label:
                relation_class = 4
            else:
                relation_class = 5
            return (
                relation_class,
                0 if "toward_robot" in label else 1,
                abs(tilt_mag - 20.0) if is_tilted else 999.0,
                z_lift,
                str(item.get("label", "")),
            )

        candidates.sort(
            key=_lvmukuai_candidate_order
        )
        print(
            "[direct_grasp] lvmukuai priority candidate labels: "
            + ", ".join(str(c.get("label", "?")) for c in candidates[:12])
        )
    if source_name == "shuazi":
        candidates.sort(key=_shuazi_grasp_label_priority)
        print(
            "[direct_grasp] shuazi priority candidate labels: "
            + ", ".join(str(c.get("label", "?")) for c in candidates[:12])
        )
    print(
        f"[direct_grasp] built {len(candidates)} grasp candidate(s) "
        f"from {len(grasp_variants)} pose variant(s)"
        f"{'' if not use_rule_bias_variants else ' (rule-paired bias variants)'}"
        f"{'' if use_rule_bias_variants else f' x {len(grasp_axis_shifts)} axis shift(s) x {len(grasp_z_lifts)} z lift(s)'}; "
        f"grasp_mode={grasp_mode}, tilt={len(tilt_cands)}, non_tilt={len(non_tilt_cands)}"
    )
    if use_rule_bias_variants:
        if rule_top_bias_axis_sign is not None:
            sign_text = "+" if float(rule_top_bias_axis_sign) > 0.0 else "-"
            print(
                "[direct_grasp] vertical top-bias axis remap: "
                f"target_long_axis_dot_up={float(rule_top_bias_dot_up):.3f}; "
                f"top_bias shifts use {sign_text}object-long-axis in the current grasp frame"
            )
        print(
            "[direct_grasp] place-rule grasp bias variants: "
            + ", ".join(
                f"{_rule_bias_display_label(getattr(v, 'label', 'bias'), _rule_bias_actual_axis_shift(v))}"
                f"[axis={_rule_bias_actual_axis_shift(v):.3f}, rule_axis={float(v.axis_shift_m):.3f}, "
                f"tilt={float(v.tilt_toward_robot_deg):.1f}, tilt_shift={float(v.tilt_shift_m):.3f}, z_lift={float(v.z_lift_m):.3f}]"
                f"/dir={getattr(v, 'tilt_direction', 'toward_robot')}"
                for v in rule_grasp_bias_variants
            )
        )
    if tilt_variants:
        print(f"[direct_grasp] tilt variant labels: {tilt_variants[:8]}")
    return candidates


def _compute_object_lowest_point_offset(pose, object_dims):
    """
    计算物体在给定姿态下，最低点相对于中心点的Z偏移。
    当物体倾斜时，最低点会比中心点更低。

    Args:
        pose: 物体的放置姿态
        object_dims: 物体的尺寸 [x, y, z]

    Returns:
        最低点的Z偏移（负值表示低于中心）
    """
    try:
        # 获取旋转矩阵
        q = targeted.base.flatten_np(pose.q)[:4]
        from scipy.spatial.transform import Rotation
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()  # xyzw -> wxyz

        # 物体的8个顶点（在物体坐标系中）
        dims = np.asarray(object_dims, dtype=np.float32) / 2.0
        corners = np.array([
            [-dims[0], -dims[1], -dims[2]],
            [-dims[0], -dims[1], +dims[2]],
            [-dims[0], +dims[1], -dims[2]],
            [-dims[0], +dims[1], +dims[2]],
            [+dims[0], -dims[1], -dims[2]],
            [+dims[0], -dims[1], +dims[2]],
            [+dims[0], +dims[1], -dims[2]],
            [+dims[0], +dims[1], +dims[2]],
        ], dtype=np.float32)

        # 转换到世界坐标系
        corners_world = (R @ corners.T).T

        # 找到最低点的Z偏移
        min_z_offset = float(np.min(corners_world[:, 2]))
        return min_z_offset
    except Exception as e:
        print(f"[warning] failed to compute lowest point offset: {e}")
        return 0.0


def _make_short_pre_place_pose(candidate, target_axis: np.ndarray | None, approach_distance: float):
    if target_axis is None:
        return candidate.pre_place_pose
    place_p = targeted.base.flatten_np(candidate.place_pose.p)[:3].astype(np.float32)
    pre_place_p = (place_p + target_axis * float(approach_distance)).astype(np.float32)
    return targeted.base.make_pose_with_position(candidate.place_pose, pre_place_p)


def _dedupe_place_candidates(candidates):
    deduped = []
    seen = set()
    for item in list(candidates or []):
        pose = item["pose"]
        place_pose = item["place_pose"]
        key = (
            tuple(np.round(targeted.base.flatten_np(pose.p)[:3], 5).tolist()),
            tuple(np.round(targeted.base.flatten_np(pose.q)[:4], 5).tolist()),
            tuple(np.round(targeted.base.flatten_np(place_pose.p)[:3], 5).tolist()),
            tuple(np.round(targeted.base.flatten_np(place_pose.q)[:4], 5).tolist()),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _place_candidate_release_pose(candidate):
    if not isinstance(candidate, dict):
        return None
    return candidate.get("release_pose", candidate.get("place_pose", candidate.get("pose")))


def _pose_dedupe_key(pose):
    if pose is None:
        return None
    try:
        p = targeted.base.flatten_np(pose.p)[:3].astype(np.float32)
        q = _normalize_quat_wxyz(targeted.base.flatten_np(pose.q)[:4])
        if float(q[0]) < 0.0:
            q = -q
        return (
            tuple(np.round(p, 5).tolist()),
            tuple(np.round(q, 5).tolist()),
        )
    except Exception:
        return None


def _record_joint_chain_failed_place_candidates(
    demo,
    candidates,
    *,
    grasp_label: str,
    status: str,
) -> None:
    """Accumulate release poses across all tried grasp/tilt relations for failure renders."""
    keys = set(getattr(demo, "_last_joint_chain_failed_candidate_pose_keys", set()) or set())
    poses = list(getattr(demo, "_last_joint_chain_failed_candidate_poses", []) or [])
    records = list(getattr(demo, "_last_joint_chain_failed_candidate_records", []) or [])
    for candidate in list(candidates or []):
        pose = _place_candidate_release_pose(candidate)
        key = _pose_dedupe_key(pose)
        if key is None or key in keys:
            continue
        keys.add(key)
        poses.append(pose)
        records.append(
            {
                "grasp_label": str(grasp_label),
                "place_label": str(candidate.get("label", "")),
                "variant_label": str(candidate.get("variant_label", "")),
                "status": str(status),
            }
        )
    demo._last_joint_chain_failed_candidate_pose_keys = keys
    demo._last_joint_chain_failed_candidate_poses = poses
    demo._last_joint_chain_failed_candidate_records = records


def _print_joint_chain_failed_candidate_summary(demo, *, limit: int = 24) -> None:
    records = list(getattr(demo, "_last_joint_chain_failed_candidate_records", []) or [])
    if not records:
        return
    print(
        f"[inspect] accumulated {len(records)} distinct release candidate pose(s) "
        "across attempted grasp/tilt relations:"
    )
    for idx, record in enumerate(records[: max(0, int(limit))], 1):
        print(
            f"  [{idx:02d}] grasp={record.get('grasp_label', '')} "
            f"place={record.get('place_label', '')} status={record.get('status', '')}"
        )
    if len(records) > int(limit):
        print(f"  ... {len(records) - int(limit)} more candidate pose(s) omitted from text summary")


def _current_source_object_name(args) -> str | None:
    return curobo_wrapper.normalize_object_name(getattr(args, "object_name", None))


def _attached_source_exclude_names(args, rule=None) -> set[str]:
    names: set[str] = set()
    for raw_name in (
        getattr(args, "object_name", None),
        getattr(rule, "source_object_name", None) if rule is not None else None,
    ):
        normalized = curobo_wrapper.normalize_object_name(raw_name)
        if normalized:
            names.add(normalized)
    return names


def _lift_pose_world_z(pose, dz: float):
    dz = float(dz)
    if abs(dz) <= 1e-8:
        return pose
    p = _get_pose_position(pose)
    p[2] += dz
    return targeted.base.make_pose_with_position(pose, p.astype(np.float32))


def _extend_hover_pose_along_release_approach(release_pose, hover_pose, extra_distance: float):
    extra_distance = float(extra_distance)
    if extra_distance <= 1e-8:
        return hover_pose
    release_p = _get_pose_position(release_pose)
    hover_p = _get_pose_position(hover_pose)
    approach_dir = _normalize((hover_p - release_p).astype(np.float32))
    if approach_dir is None:
        return _lift_pose_world_z(hover_pose, extra_distance)
    return targeted.base.make_pose_with_position(
        hover_pose,
        (hover_p + approach_dir * extra_distance).astype(np.float32),
    )


def classify_object_category(args, rule, object_dims=None) -> str:
    """Coarse object category used to choose grasp/place primitives."""
    primitive = str(getattr(rule, "primitive", "") or "")
    if primitive == "insert_vertical":
        return "insert"
    if bool(getattr(rule, "orientation_invariant", False)):
        return "sphere"

    if object_dims is not None:
        try:
            dims = np.asarray(object_dims, dtype=np.float32).reshape(3)
            min_dim = max(float(np.min(dims)), 1e-6)
            extent_ratio = float(np.max(dims) / min_dim)
            if extent_ratio < 1.3:
                return "sphere"
            if extent_ratio >= 2.0:
                return "long_axis"
        except Exception:
            pass

    if bool(getattr(rule, "preserve_long_axis_vertical", False)):
        return "long_axis"
    return "ordinary"


def _current_object_category(args, rule=None, object_dims=None) -> str:
    if rule is None:
        rule = targeted.get_place_rule(getattr(args, "object_name", None))
    if object_dims is None:
        try:
            object_dims = targeted.base.get_asset_box_size(args.sim_asset_file, args.sim_asset_scale)
        except Exception:
            object_dims = None
    return classify_object_category(args, rule, object_dims)


def _is_sphere_category(args, rule=None, object_dims=None) -> bool:
    return _current_object_category(args, rule, object_dims) == "sphere"


def _skip_joint_search_for_current_object(args) -> bool:
    source_name = _current_source_object_name(args)
    skip_names = {
        curobo_wrapper.normalize_object_name(name)
        for name in list(getattr(args, "skip_joint_search_object_names", []) or [])
    }
    return source_name is not None and source_name in skip_names


def _bi_direct_grasp_approach_roll_degs(args) -> list[float]:
    values = _unique_finite_float_list(
        getattr(args, "bi_direct_grasp_approach_roll_degs", [0.0, 180.0]),
    )
    # The pen grasp frame is already aligned to the object's long axis.  Rolls
    # near 90/270 degrees would rotate TCP +Y, the pad opening axis, onto the
    # pen long axis, which is physically the wrong grasp direction.
    filtered = []
    for value in values:
        folded = abs((((float(value) + 180.0) % 360.0) - 180.0))
        if folded <= 1.0 or abs(folded - 180.0) <= 1.0:
            filtered.append(float(value))
    return filtered or [0.0, 180.0]


def _shuazi_direct_grasp_approach_roll_degs(args) -> list[float]:
    values = _unique_finite_float_list(
        getattr(args, "shuazi_direct_grasp_approach_roll_degs", [0.0, 90.0, 180.0, 270.0]),
    )
    return values or [0.0]


def _tennis_release_tilt_toward_robot_degs(args) -> list[float]:
    values = _unique_finite_float_list(
        getattr(args, "tennis_direct_place_tilt_toward_robot_degs", [0.0, 15.0, -15.0, 30.0, -30.0]),
    )
    legacy_value = getattr(args, "tennis_direct_place_tilt_toward_robot_deg", None)
    if legacy_value is not None:
        values = _unique_finite_float_list([float(legacy_value)] + list(values or []))
    return values or [0.0]


def _tennis_release_axial_roll_degs(args) -> list[float]:
    values = _unique_finite_float_list(
        getattr(args, "tennis_direct_place_axial_roll_degs", [0.0, -45.0, 45.0, -90.0, 90.0, 180.0]),
    )
    return values or [0.0]


def _transport_screen_args_for_object(args, source_name: str | None):
    adjusted = SimpleNamespace(**vars(args))
    if source_name == "bi":
        adjusted.place_transport_max_winners = max(int(getattr(args, "place_transport_max_winners", 2)), 12)
    if source_name != "tennis":
        return adjusted
    adjusted.curobo_ik_prefilter_position_threshold = max(
        float(getattr(args, "curobo_ik_prefilter_position_threshold", 0.01) or 0.0),
        0.012,
    )
    adjusted.curobo_ik_prefilter_rotation_threshold = max(
        float(getattr(args, "curobo_ik_prefilter_rotation_threshold", 0.25) or 0.0),
        0.50,
    )
    return adjusted


def _make_base_referenced_tennis_release_pose(demo, release_p: np.ndarray, tilt_deg: float):
    base_p = _get_robot_base_world_transform(demo)[:3, 3].astype(np.float32)
    to_base_xy = (base_p - release_p.astype(np.float32)).copy()
    to_base_xy[2] = 0.0
    to_base_xy = _normalize(to_base_xy)
    if to_base_xy is None:
        return None

    down = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    tilt_sign = 1.0 if float(tilt_deg) >= 0.0 else -1.0
    tilt_rad = np.deg2rad(abs(float(tilt_deg)))
    body_shift_dir = (to_base_xy * tilt_sign).astype(np.float32)
    # TCP local +Z points from the TCP toward the ball.  To make the rendered
    # gripper body lean/shift toward the robot, the approach axis must point
    # horizontally away from the robot.
    approach = _normalize(np.cos(tilt_rad) * down - np.sin(tilt_rad) * body_shift_dir)
    if approach is None:
        return None

    # Build the TCP frame directly from the robot-base direction and world Z,
    # so the tilt plane is fixed in world space rather than inherited from the
    # current gripper local axes.
    pad_axis = _normalize(np.cross(body_shift_dir, down))
    if pad_axis is None:
        return None
    ortho_axis = _normalize(np.cross(pad_axis, approach))
    if ortho_axis is None:
        return None
    pad_axis = _normalize(np.cross(approach, ortho_axis))
    if pad_axis is None:
        return None

    R_tcp = np.stack([ortho_axis, pad_axis, approach], axis=1).astype(np.float32)
    q_tcp = targeted.base.bridge_mod_mat2quat(R_tcp).astype(np.float32)
    return targeted.Pose.create_from_pq(p=release_p.astype(np.float32), q=q_tcp)


def _sphere_tcp_object_distance_m(T_tcp_obj_override, args) -> float:
    """Return the centered TCP->sphere-center distance, ignoring lateral/orientation offsets."""
    fallback = float(max(getattr(args, "grasp_z_offset", 0.0), 0.0))
    if T_tcp_obj_override is None:
        return fallback
    try:
        T_tcp_obj = np.asarray(T_tcp_obj_override, dtype=np.float32).reshape(4, 4)
        tcp_to_obj = np.asarray(T_tcp_obj[:3, 3], dtype=np.float32).reshape(3)
        axial = abs(float(tcp_to_obj[2]))
        norm = float(np.linalg.norm(tcp_to_obj))
        if np.isfinite(axial) and axial > 1e-4:
            return float(np.clip(axial, 0.0, 0.150))
        if np.isfinite(norm) and norm > 1e-4:
            return float(np.clip(norm, 0.0, 0.150))
    except Exception:
        pass
    return fallback


def _sphere_release_tcp_pose_from_center(demo, args, center_p: np.ndarray, tilt_deg: float, tcp_object_distance_m: float):
    center_p = np.asarray(center_p, dtype=np.float32).reshape(3)
    if _current_source_object_name(args) == "tennis":
        pose = _make_base_referenced_tennis_release_pose(demo, center_p, float(tilt_deg))
    else:
        pose = None
    if pose is None:
        topdown_pose = demo.build_topdown_grasp_pose()
        pose = targeted.base.make_pose_with_position(topdown_pose, center_p.astype(np.float32))
    try:
        T_world_tcp = targeted.base.pose_to_matrix(
            targeted.base.flatten_np(pose.p)[:3],
            targeted.base.flatten_np(pose.q)[:4],
        )
        approach_axis = np.asarray(T_world_tcp[:3, 2], dtype=np.float32).reshape(3)
        tcp_p = (center_p - approach_axis * float(max(tcp_object_distance_m, 0.0))).astype(np.float32)
        return targeted.base.make_pose_with_position(pose, tcp_p)
    except Exception:
        return targeted.base.make_pose_with_position(pose, center_p.astype(np.float32))


def _roll_pose_about_tcp_approach(pose, roll_deg: float):
    if abs(float(roll_deg)) <= 1e-6:
        return pose
    try:
        T_world_tcp = targeted.base.pose_to_matrix(
            targeted.base.flatten_np(pose.p)[:3],
            targeted.base.flatten_np(pose.q)[:4],
        )
        rad = np.deg2rad(float(roll_deg))
        c = float(np.cos(rad))
        s = float(np.sin(rad))
        R_roll_local = np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        R_world_tcp = (T_world_tcp[:3, :3].astype(np.float32) @ R_roll_local).astype(np.float32)
        q_tcp = targeted.base.bridge_mod_mat2quat(R_world_tcp).astype(np.float32)
        return targeted.Pose.create_from_pq(
            p=targeted.base.flatten_np(pose.p)[:3].astype(np.float32),
            q=q_tcp,
        )
    except Exception:
        return pose


def _horizontal_direction_variants(
    demo,
    obj_position: np.ndarray,
    *,
    T_world_obj_goal: np.ndarray | None = None,
    T_world_obj_current: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray]]:
    obj_position = np.asarray(obj_position, dtype=np.float32).reshape(3)
    base_p = _get_robot_base_world_transform(demo)[:3, 3].astype(np.float32)
    variants: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[float, float, float]] = set()

    def add(label: str, vec) -> None:
        arr = np.asarray(vec, dtype=np.float32).reshape(3).copy()
        arr[2] = 0.0
        unit = _normalize(arr)
        if unit is None:
            return
        key = tuple(np.round(unit, 4).tolist())
        if key in seen:
            return
        seen.add(key)
        variants.append((label, unit))

    to_robot_goal = (base_p - obj_position).astype(np.float32)
    add("face_robot", to_robot_goal)
    add("away_robot", -to_robot_goal)
    add("world_pos_x", np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    add("world_neg_x", np.asarray([-1.0, 0.0, 0.0], dtype=np.float32))
    add("world_pos_y", np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    add("world_neg_y", np.asarray([0.0, -1.0, 0.0], dtype=np.float32))

    if T_world_obj_current is not None:
        T_current = np.asarray(T_world_obj_current, dtype=np.float32).reshape(4, 4)
        to_robot_current = (base_p - T_current[:3, 3]).astype(np.float32)
        add("current_face_robot", to_robot_current)
        add("current_away_robot", -to_robot_current)

    for prefix, T_obj in (("goal", T_world_obj_goal), ("current", T_world_obj_current)):
        if T_obj is None:
            continue
        T_obj = np.asarray(T_obj, dtype=np.float32).reshape(4, 4)
        for axis_idx in range(3):
            axis = T_obj[:3, axis_idx].astype(np.float32)
            if float(np.linalg.norm(axis[:2])) < 0.20:
                continue
            add(f"{prefix}_axis{axis_idx}_pos", axis)
            add(f"{prefix}_axis{axis_idx}_neg", -axis)

    return variants


def _fixed_tabletop_horizontal_long_axis(
    args,
    T_world_obj: np.ndarray,
    *,
    object_dims: np.ndarray | None = None,
    fallback_axis: np.ndarray | None = None,
) -> np.ndarray | None:
    """Estimate a tabletop object's horizontal long axis without trusting local Z.

    Cylindrical/near-symmetric meshes can arrive with arbitrary local frame
    roll.  For fixed tabletop place/grasp relations, world Z is the only stable
    vertical reference; the horizontal long direction is estimated from the
    transformed asset geometry and then projected onto the world XY plane.
    """
    T_world_obj = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4)
    up_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def _canonical_sign(axis: np.ndarray) -> np.ndarray:
        arr = np.asarray(axis, dtype=np.float32).reshape(3)
        if fallback_axis is not None:
            fb = np.asarray(fallback_axis, dtype=np.float32).reshape(3).copy()
            fb[2] = 0.0
            fb = _normalize(fb)
            if fb is not None and float(np.dot(arr, fb)) < 0.0:
                return (-arr).astype(np.float32)
        if abs(float(arr[0])) >= abs(float(arr[1])):
            return ((-arr) if float(arr[0]) < 0.0 else arr).astype(np.float32)
        return ((-arr) if float(arr[1]) < 0.0 else arr).astype(np.float32)

    try:
        local_points = targeted.base.get_asset_local_points(args.sim_asset_file, args.sim_asset_scale)
        pts = np.asarray(local_points, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] >= 3:
            world_pts = (T_world_obj[:3, :3] @ pts.T).T + T_world_obj[:3, 3]
            xy = world_pts[:, :2] - np.mean(world_pts[:, :2], axis=0, keepdims=True)
            cov = (xy.T @ xy) / max(float(xy.shape[0]), 1.0)
            eigvals, eigvecs = np.linalg.eigh(cov)
            axis_xy = eigvecs[:, int(np.argmax(eigvals))]
            axis = np.array([float(axis_xy[0]), float(axis_xy[1]), 0.0], dtype=np.float32)
            axis = _normalize(axis)
            if axis is not None:
                return _canonical_sign(axis)
    except Exception:
        pass

    if object_dims is not None:
        try:
            dims = np.asarray(object_dims, dtype=np.float32).reshape(3)
            for axis_idx in np.argsort(dims)[::-1]:
                axis = T_world_obj[:3, int(axis_idx)].astype(np.float32)
                axis = axis - float(np.dot(axis, up_axis)) * up_axis
                axis = _normalize(axis)
                if axis is not None:
                    return _canonical_sign(axis)
        except Exception:
            pass

    if fallback_axis is not None:
        axis = np.asarray(fallback_axis, dtype=np.float32).reshape(3).copy()
        axis[2] = 0.0
        axis = _normalize(axis)
        if axis is not None:
            return _canonical_sign(axis)
    return _normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))


def _fixed_tabletop_relation_object_frame(
    args,
    T_world_obj: np.ndarray,
    *,
    object_dims: np.ndarray | None = None,
    fallback_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Build a tabletop relation frame that never trusts the object's local Z.

    The true object pose from place_rules / perception is still used as the
    final object pose.  This helper is only for deriving TCP<->object grasp
    relations: X follows the object's horizontal long direction, Z is world up.
    """
    T_world_obj = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4)
    up_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    x_axis = _fixed_tabletop_horizontal_long_axis(
        args,
        T_world_obj,
        object_dims=object_dims,
        fallback_axis=fallback_axis,
    )
    if x_axis is None:
        x_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    x_axis = np.asarray(x_axis, dtype=np.float32).reshape(3)
    x_axis[2] = 0.0
    x_axis = _normalize(x_axis)
    if x_axis is None:
        x_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    y_axis = _normalize(np.cross(up_axis, x_axis))
    if y_axis is None:
        y_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = _normalize(np.cross(y_axis, up_axis))
    if x_axis is None:
        x_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    T_relation = np.eye(4, dtype=np.float32)
    T_relation[:3, :3] = np.stack([x_axis, y_axis, up_axis], axis=1).astype(np.float32)
    T_relation[:3, 3] = T_world_obj[:3, 3].astype(np.float32)
    return T_relation


def _fixed_tabletop_release_tilt_degs(args, source_name: str | None) -> list[float]:
    values = getattr(args, "fixed_tabletop_release_tilt_deg", None)
    if values is None:
        # Three magnitude levels for ordinary tabletop objects: 0/30/60 deg.
        # Keep both signs for non-zero tilt because the reachable direction
        # depends on the slot/object side, while the final object pose remains fixed.
        values = [0.0, 30.0, -30.0, 60.0, -60.0]
    result = _unique_finite_float_list(values)
    return result or [0.0]


def _fixed_tabletop_signed_tilt_from_label(label: str) -> float:
    label_l = str(label or "").lower()
    m = re.search(r"world_z_tilt_(toward_robot|away_robot)_([-+]?\d+(?:\.\d+)?)deg", label_l)
    if m:
        sign = 1.0 if m.group(1) == "toward_robot" else -1.0
        return sign * abs(float(m.group(2)))
    m = re.search(r"tilt_along_long_(pos|neg)_([-+]?\d+(?:\.\d+)?)deg", label_l)
    if m:
        sign = 1.0 if m.group(1) == "pos" else -1.0
        return sign * abs(float(m.group(2)))
    return 0.0


def _fixed_tabletop_has_nonzero_tilt(label: str) -> bool:
    return abs(_fixed_tabletop_signed_tilt_from_label(label)) > 1.0


def _make_vertical_gripper_rotation_candidates(
    demo,
    obj_position: np.ndarray,
    *,
    T_world_obj_goal: np.ndarray | None = None,
    T_world_obj_current: np.ndarray | None = None,
    object_dims: np.ndarray | None = None,
    args=None,
    source_name: str | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Build top-down release TCP frames for fixed flat tabletop placement.

    The final object pose is fixed first.  Ordinary tabletop objects do not get
    yaw/spin/contact-point variants here: one canonical TCP frame is built from
    the target long axis and world Z, then only small tilt variants are tried.
    Grasp/place consistency is handled later through fixed tabletop relation
    frames, not through arbitrary object-local Z axes.
    """
    variants: list[tuple[str, np.ndarray]] = []
    if T_world_obj_goal is None or object_dims is None:
        return variants

    T_world_obj_goal = np.asarray(T_world_obj_goal, dtype=np.float32).reshape(4, 4)
    up_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    # The rendered gripper and final-contact metric both expect TCP +Z to be
    # the approach direction into the object/table for these fixed flat poses.
    approach_axis = -up_axis

    base_p = _get_robot_base_world_transform(demo)[:3, 3].astype(np.float32)
    to_robot = base_p - T_world_obj_goal[:3, 3].astype(np.float32)
    to_robot[2] = 0.0
    long_axis = _fixed_tabletop_horizontal_long_axis(
        args,
        T_world_obj_goal,
        object_dims=object_dims,
        fallback_axis=to_robot,
    )
    if long_axis is None:
        return variants

    seen: set[tuple[float, ...]] = set()
    for long_sign in (1.0,):
        ortho_axis = long_axis.astype(np.float32)
        closing_axis = _normalize(np.cross(approach_axis, ortho_axis))
        if closing_axis is None:
            continue
        ortho_axis = _normalize(np.cross(closing_axis, approach_axis))
        if ortho_axis is None:
            continue
        if float(np.dot(ortho_axis, long_axis * float(long_sign))) < 0.0:
            ortho_axis = -ortho_axis
            closing_axis = -closing_axis
        R_base = np.stack([ortho_axis, closing_axis, approach_axis], axis=1).astype(np.float32)
        side_label = "long"
        for tilt_deg in _fixed_tabletop_release_tilt_degs(args, source_name):
            theta = np.deg2rad(float(tilt_deg))
            c = float(np.cos(theta))
            s = float(np.sin(theta))
            R_tilt_local_y = np.array(
                [
                    [c, 0.0, s],
                    [0.0, 1.0, 0.0],
                    [-s, 0.0, c],
                ],
                dtype=np.float32,
            )
            R_tcp = (R_base @ R_tilt_local_y).astype(np.float32)
            key = tuple(np.round(R_tcp.reshape(-1), 5).tolist())
            if key in seen:
                continue
            seen.add(key)
            if abs(float(tilt_deg)) <= 1e-6:
                tilt_label = "tilt_0deg"
            elif float(tilt_deg) > 0.0:
                tilt_label = f"tilt_along_long_pos_{int(round(abs(float(tilt_deg))))}deg"
            else:
                tilt_label = f"tilt_along_long_neg_{int(round(abs(float(tilt_deg))))}deg"
            variants.append((f"flat_topdown_{side_label}_{tilt_label}", R_tcp))
    return variants


FIXED_TABLETOP_SOURCE_NAMES = {"carriot", "shuazi", "lvmukuai"}


def _is_fixed_tabletop_source(source_name: str | None) -> bool:
    return str(source_name or "").strip().lower() in FIXED_TABLETOP_SOURCE_NAMES


def _is_fixed_tabletop_vertical_helper_label(label: str, source_name: str | None = None) -> bool:
    label_l = str(label or "").lower()
    if "vertical_gripper" not in label_l:
        return False
    source = str(source_name or "").strip().lower()
    return not source or source in label_l


def _filter_fixed_tabletop_release_candidates(candidates, source_name: str | None, *, label: str) -> list:
    return list(candidates or [])


def _fixed_tabletop_contact_axis_shifts(args, object_dims: np.ndarray | None) -> list[float]:
    # Ordinary tabletop objects keep a fixed grasp/place relation.  Do not move
    # the contact point along the object axis; only tilt variants are allowed.
    return [0.0]


def _build_fixed_tabletop_place_first_grasp_candidates(
    demo,
    args,
    rule,
    bridge_mod,
    scene_capture_cache,
    place_state_cache,
    T_world_obj_current: np.ndarray,
    *,
    object_dims: np.ndarray | None,
) -> list[dict]:
    """Build fixed-tabletop grasp candidates from the final object pose first.

    For these objects the final object pose is non-negotiable.  We therefore
    generate legal release TCP frames at the configured target pose, then map
    that gripper/object relation through a tabletop relation frame whose Z axis
    is always world Z.  This keeps the true place_rules object pose intact while
    preventing arbitrary object local frames from tilting the grasp TCP.
    """
    source_name = _current_source_object_name(args)
    if not _is_fixed_tabletop_source(source_name) or rule is None:
        return []
    if bridge_mod is None or scene_capture_cache is None or place_state_cache is None:
        print(f"[direct_grasp] {source_name}: place-first generation missing scene context")
        return []
    try:
        place_plans = targeted.build_targeted_place_plan_variants(
            demo,
            bridge_mod,
            scene_capture_cache,
            rule,
            place_state_cache,
            args,
            T_tcp_obj_override=None,
        )
    except Exception as exc:
        print(f"[direct_grasp] {source_name}: place-first target lookup failed: {exc}")
        return []
    if not place_plans:
        return []

    T_world_obj_current = np.asarray(T_world_obj_current, dtype=np.float32).reshape(4, 4)
    try:
        current_center_world = np.asarray(demo.get_object_world_aabb_center(), dtype=np.float32).reshape(3)
    except Exception:
        current_center_world = T_world_obj_current[:3, 3].astype(np.float32)
    current_center_h = np.concatenate([current_center_world.astype(np.float32), np.array([1.0], dtype=np.float32)])
    contact_local_base = (np.linalg.inv(T_world_obj_current) @ current_center_h)[:3].astype(np.float32)

    object_axis_local = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    if object_dims is not None:
        try:
            dims = np.asarray(object_dims, dtype=np.float32).reshape(3)
            object_axis_local = np.zeros(3, dtype=np.float32)
            object_axis_local[int(np.argmax(dims))] = 1.0
        except Exception:
            object_axis_local = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    object_axis_world = _normalize(T_world_obj_current[:3, :3] @ object_axis_local)

    axis_shifts = _fixed_tabletop_contact_axis_shifts(args, object_dims)
    z_lifts = _unique_finite_float_list(getattr(args, "direct_grasp_z_lifts_m", [0.0]), min_value=0.0)
    if not z_lifts:
        z_lifts = [0.0]

    base_grasp_pose = demo.build_topdown_grasp_pose()
    base_grasp_p = _get_pose_position(base_grasp_pose).astype(np.float32)

    unique_goals: list[tuple[str | None, np.ndarray]] = []
    seen_goals = set()
    for plan in place_plans:
        T_goal = np.asarray(plan.T_world_obj_desired, dtype=np.float32).reshape(4, 4)
        key = tuple(np.round(T_goal.reshape(-1), 5).tolist())
        if key in seen_goals:
            continue
        seen_goals.add(key)
        unique_goals.append((getattr(plan, "variant_label", None), T_goal))

    candidates: list[dict] = []
    seen = set()
    rotation_variant_count = 0
    for plan_label, T_world_obj_goal in unique_goals:
        to_robot_goal = (_get_robot_base_world_transform(demo)[:3, 3] - T_world_obj_goal[:3, 3]).astype(np.float32)
        to_robot_goal[2] = 0.0
        T_world_obj_goal_relation = _fixed_tabletop_relation_object_frame(
            args,
            T_world_obj_goal,
            object_dims=object_dims,
            fallback_axis=to_robot_goal,
        )
        T_world_obj_current_relation = _fixed_tabletop_relation_object_frame(
            args,
            T_world_obj_current,
            object_dims=object_dims,
            fallback_axis=T_world_obj_goal_relation[:3, 0],
        )
        release_rotation_variants = _make_vertical_gripper_rotation_candidates(
            demo,
            T_world_obj_goal[:3, 3].astype(np.float32),
            T_world_obj_goal=T_world_obj_goal,
            T_world_obj_current=T_world_obj_current,
            object_dims=object_dims,
            args=args,
            source_name=source_name,
        )
        rotation_variant_count = max(rotation_variant_count, len(release_rotation_variants))
        grasp_rotation_variants: list[tuple[str, np.ndarray]] = []
        for release_label, R_release in release_rotation_variants:
            # Map release->grasp through relation frames, not raw object frames.
            # Raw local Z can be arbitrary for these meshes and must not tilt the
            # gripper away from the world-Z top-down direction.
            R_grasp_from_release = (
                T_world_obj_current_relation[:3, :3]
                @ T_world_obj_goal_relation[:3, :3].T
                @ np.asarray(R_release, dtype=np.float32).reshape(3, 3)
            ).astype(np.float32)
            grasp_rotation_variants.append((f"target_release_{release_label}", R_grasp_from_release))
        for grasp_label, R_grasp in grasp_rotation_variants:
            for axis_shift in axis_shifts:
                for z_lift in z_lifts:
                    T_world_tcp_grasp = np.eye(4, dtype=np.float32)
                    T_world_tcp_grasp[:3, :3] = np.asarray(R_grasp, dtype=np.float32).reshape(3, 3)
                    T_world_tcp_grasp[:3, 3] = base_grasp_p
                    grasp_pose = _pose_from_world_matrix(T_world_tcp_grasp)
                    if object_axis_world is not None and abs(float(axis_shift)) > 1e-6:
                        shifted_p = (
                            _get_pose_position(grasp_pose) + object_axis_world * float(axis_shift)
                        ).astype(np.float32)
                        grasp_pose = targeted.base.make_pose_with_position(grasp_pose, shifted_p)
                    if float(z_lift) > 1e-6:
                        shifted_p = (
                            _get_pose_position(grasp_pose) + np.array([0.0, 0.0, float(z_lift)], dtype=np.float32)
                        ).astype(np.float32)
                        grasp_pose = targeted.base.make_pose_with_position(grasp_pose, shifted_p)
                    T_world_tcp_grasp = _pose_to_matrix_from_pose_obj(grasp_pose)
                    T_tcp_relation = (
                        np.linalg.inv(T_world_tcp_grasp) @ T_world_obj_current_relation
                    ).astype(np.float32)
                    grasp_pose = _pose_from_world_matrix(T_world_tcp_grasp)
                    pregrasp_pose = demo.build_pregrasp_pose(grasp_pose)
                    grasp_pose, pregrasp_pose, geometry_grasp_raise = targeted.base.enforce_topdown_grasp_insertion_limit(
                        demo,
                        args,
                        grasp_pose,
                        pregrasp_pose,
                    )
                    if geometry_grasp_raise > 0:
                        T_world_tcp_grasp = _pose_to_matrix_from_pose_obj(grasp_pose)
                        T_tcp_relation = (
                            np.linalg.inv(T_world_tcp_grasp) @ T_world_obj_current_relation
                        ).astype(np.float32)
                    grasp_pose, pregrasp_pose, grasp_tcp_raise = targeted.base.enforce_min_grasp_tcp_z(
                        grasp_pose,
                        pregrasp_pose,
                        args.min_grasp_tcp_z,
                    )
                    if grasp_tcp_raise > 0:
                        T_world_tcp_grasp = _pose_to_matrix_from_pose_obj(grasp_pose)
                        T_tcp_relation = (
                            np.linalg.inv(T_world_tcp_grasp) @ T_world_obj_current_relation
                        ).astype(np.float32)
                    T_release_check = (
                        T_world_obj_goal_relation @ np.linalg.inv(T_tcp_relation)
                    ).astype(np.float32)
                    # Keep targeted place object pose exactly from place_rules, but
                    # choose a TCP<->object transform that reproduces the world-Z
                    # relation-frame release pose when build_targeted_place_plan_variants
                    # computes T_world_tcp = T_world_obj_desired @ inv(T_tcp_obj).
                    T_tcp_obj = (
                        np.linalg.inv(T_release_check) @ T_world_obj_goal
                    ).astype(np.float32)
                    release_check_pose = _pose_from_world_matrix(T_release_check)
                    label_parts = [str(source_name), "place_first", grasp_label]
                    if plan_label:
                        label_parts.append(str(plan_label))
                    if abs(float(axis_shift)) > 1e-6:
                        label_parts.append(f"axis_{int(round(float(axis_shift) * 1000.0))}mm")
                    if float(z_lift) > 1e-6:
                        label_parts.append(f"lift_{int(round(float(z_lift) * 1000.0))}mm")
                    label = "grasp_direct_" + "_".join(label_parts)
                    key = (
                        tuple(np.round(targeted.base.flatten_np(grasp_pose.p)[:3], 5).tolist()),
                        tuple(np.round(targeted.base.flatten_np(grasp_pose.q)[:4], 5).tolist()),
                        tuple(np.round(T_tcp_obj.reshape(-1), 5).tolist()),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        {
                            "label": label,
                            "pose": grasp_pose,
                            "pregrasp_pose": pregrasp_pose,
                            "T_tcp_obj": T_tcp_obj.astype(np.float32),
                            "grasp_axis_shift_m": float(axis_shift),
                            "grasp_z_lift_m": float(z_lift),
                            "grasp_approach_roll_deg": 0.0,
                            "place_first": True,
                            "fixed_tabletop_release_pose": release_check_pose,
                        }
                    )
    candidates.sort(
        key=lambda item: (
            max(float(item.get("grasp_z_lift_m", 0.0) or 0.0), 0.0),
            abs(float(item.get("grasp_axis_shift_m", 0.0) or 0.0)),
            0 if "face_robot" in str(item.get("label", "")).lower() else 1,
            0 if "x_down" in str(item.get("label", "")).lower() else 1,
            str(item.get("label", "")),
        )
    )
    print(
        f"[direct_grasp] {source_name} place-first fixed-tabletop generated {len(candidates)} "
        f"candidate(s) from {len(unique_goals)} fixed target pose(s), "
        f"{rotation_variant_count} target-legal world-Z release tilt variant(s), "
        f"{len(axis_shifts)} contact shift(s), {len(z_lifts)} contact z lift(s)"
    )
    if candidates:
        print(
            f"[direct_grasp] {source_name} place-first priority labels: "
            + ", ".join(str(c.get("label", "?")) for c in candidates[:12])
        )
    return candidates


def _make_sphere_free_release_pose_variants(
    demo,
    raw_release_pose,
    variant_label: str | None,
    args,
    *,
    object_center_p: np.ndarray | None = None,
    tcp_object_distance_m: float | None = None,
    ):
    """For balls, target the sphere center directly and generate a small TCP orientation set."""
    topdown_pose = demo.build_topdown_grasp_pose()
    if object_center_p is None:
        # Backward-compatible fallback: the caller already supplied a TCP target.
        release_p = _get_pose_position(raw_release_pose)
        release_pose = targeted.base.make_pose_with_position(topdown_pose, release_p.astype(np.float32))
        center_p = release_p.astype(np.float32)
        tcp_object_distance = 0.0
    else:
        center_p = np.asarray(object_center_p, dtype=np.float32).reshape(3)
        tcp_object_distance = float(max(tcp_object_distance_m or 0.0, 0.0))
        release_pose = _sphere_release_tcp_pose_from_center(demo, args, center_p, 0.0, tcp_object_distance)
    source_name = _current_source_object_name(args)
    if source_name != "tennis":
        return [(None, release_pose)]
    variants = []
    seen = set()
    for tilt_deg in _tennis_release_tilt_toward_robot_degs(args):
        base_pose = release_pose
        tilt_suffix = None
        if abs(float(tilt_deg)) > 1e-6:
            base_pose = _sphere_release_tcp_pose_from_center(demo, args, center_p, float(tilt_deg), tcp_object_distance)
            if base_pose is None:
                continue
            dir_label = "body_toward_robot" if float(tilt_deg) >= 0.0 else "body_away_robot"
            tilt_suffix = f"tilt_{dir_label}_{int(round(abs(float(tilt_deg))))}deg"
        else:
            candidate_pose = _sphere_release_tcp_pose_from_center(demo, args, center_p, 0.0, tcp_object_distance)
            if candidate_pose is not None:
                base_pose = candidate_pose
        for roll_deg in _tennis_release_axial_roll_degs(args):
            pose = _roll_pose_about_tcp_approach(base_pose, float(roll_deg))
            roll_suffix = None if abs(float(roll_deg)) <= 1e-6 else f"roll_{int(round(float(roll_deg)))}deg"
            suffix_parts = [part for part in (tilt_suffix, roll_suffix) if part]
            suffix = "+".join(suffix_parts) if suffix_parts else None
            key = (
                tuple(np.round(targeted.base.flatten_np(pose.p)[:3], 5).tolist()),
                tuple(np.round(targeted.base.flatten_np(pose.q)[:4], 5).tolist()),
            )
            if key in seen:
                continue
            seen.add(key)
            variants.append((suffix, pose))
    return variants or [(None, release_pose)]


def classify_place_mode(args, rule, object_dims=None) -> str:
    override = str(getattr(args, "place_mode", "auto") or "auto")
    if override != "auto":
        return override

    category = classify_object_category(args, rule, object_dims)
    if category == "insert":
        return "insert_place"
    if category == "sphere":
        # Spherical objects are orientation-free, but they still need a real
        # hover->release contact motion.  drop_place releases from a lifted pose
        # and skips final_contact, which leaves tennis visibly high.
        return "surface_place"

    if bool(getattr(rule, "preserve_long_axis_vertical", False)):
        return "vertical_place"

    if category == "long_axis":
        return "surface_place"

    if str(getattr(rule, "primitive", "")) == "place_on_slots":
        return "surface_place"
    return "drop_place"


def _place_mode_release_lift_m(args, rule, place_mode: str) -> float:
    if place_mode == "drop_place":
        lift = float(max(getattr(args, "drop_place_release_lift_m", 0.030), 0.0))
        if _is_sphere_category(args, rule):
            lift = max(
                lift,
                float(max(getattr(args, "sphere_place_release_lift_m", 0.030), 0.0)),
                float(max(getattr(args, "tennis_direct_place_release_lift_m", 0.0), 0.0)),
            )
        return lift
    if place_mode in {"surface_place", "vertical_place"}:
        clearance = float(max(getattr(args, "final_contact_clearance_m", 0.003), 0.0))
        legacy_clearance = float(max(getattr(args, "direct_place_contact_lift_m", clearance), 0.0))
        return max(clearance, legacy_clearance)
    return 0.0


def build_hover_pose(release_pose, place_mode: str, args, rule, *, candidate_pre_place_pose=None):
    if place_mode == "drop_place":
        return release_pose
    if place_mode == "insert_place" and candidate_pre_place_pose is not None:
        return candidate_pre_place_pose

    if place_mode == "vertical_place":
        hover_height = float(max(getattr(args, "vertical_place_hover_height_m", 0.040), 0.0))
    elif place_mode == "surface_place":
        hover_height = float(max(getattr(args, "surface_place_hover_height_m", 0.050), 0.0))
    else:
        hover_height = float(max(getattr(rule, "hover_height", 0.05), 0.0))
    if hover_height <= 1e-8:
        return release_pose
    try:
        T_world_tcp = targeted.base.pose_to_matrix(
            targeted.base.flatten_np(release_pose.p)[:3],
            targeted.base.flatten_np(release_pose.q)[:4],
        )
        axes_world = T_world_tcp[:3, :3]
        z_dots = axes_world.T @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        axis_idx = int(np.argmax(np.abs(z_dots)))
        sign = 1.0 if float(z_dots[axis_idx]) >= 0.0 else -1.0
        offset = (axes_world[:, axis_idx] * sign * hover_height).astype(np.float32)
        p = targeted.base.flatten_np(release_pose.p)[:3].astype(np.float32)
        return targeted.base.make_pose_with_position(release_pose, (p + offset).astype(np.float32))
    except Exception:
        return _lift_pose_world_z(release_pose, hover_height)


def _final_contact_disabled_world_links(planner, place_mode: str) -> list[str]:
    links = set(_direct_place_contact_tolerant_disabled_links(planner))
    configured_links = set(getattr(planner, "configured_collision_links", []) or [])
    if place_mode in {"surface_place", "vertical_place", "insert_place"} and "attached_object" in configured_links:
        links.add("attached_object")
    return sorted(links & configured_links) if configured_links else []


def _cache_attached_spheres_for_contact(planner) -> None:
    try:
        planner._contact_mode_attached_sphere_tensor = (
            planner.motion_gen.robot_cfg.kinematics.kinematics_config.get_link_spheres("attached_object")
            .clone()
        )
    except Exception:
        planner._contact_mode_attached_sphere_tensor = None


def _restore_attached_spheres_after_contact(planner) -> None:
    sphere_tensor = getattr(planner, "_contact_mode_attached_sphere_tensor", None)
    if sphere_tensor is None:
        return
    try:
        planner.motion_gen.attach_spheres_to_robot(
            sphere_radius=0.0,
            sphere_tensor=sphere_tensor,
            link_name="attached_object",
        )
        try:
            planner.ik_solver.attach_object_to_robot(
                sphere_radius=0.0,
                sphere_tensor=sphere_tensor.clone(),
                link_name="attached_object",
            )
        except Exception as exc:
            print(f"[curobo] failed to restore attached spheres to ik_solver after contact mode: {exc}")
    finally:
        planner._contact_mode_attached_sphere_tensor = None


def set_payload_collision_mode(
    planner,
    mode: str,
    place_mode: str,
    *,
    disabled_world_collision_links: list[str] | None = None,
    label: str,
) -> list[str]:
    if mode == "disabled_for_contact":
        links = set(disabled_world_collision_links or [])
        links.update(_final_contact_disabled_world_links(planner, place_mode))
        if "attached_object" in links:
            _cache_attached_spheres_for_contact(planner)
        return _set_world_collision_for_links(planner, sorted(links), enabled=False, label=label)
    if mode in {"transport", "detached"}:
        restored = _set_world_collision_for_links(
            planner,
            disabled_world_collision_links,
            enabled=True,
            label=label,
        )
        if mode == "transport" and "attached_object" in set(disabled_world_collision_links or []):
            _restore_attached_spheres_after_contact(planner)
        return restored
    return []


def _retreat_pose_for_place_mode(demo, place_mode: str, args, place_choice=None):
    if isinstance(place_choice, dict):
        hover_pose = place_choice.get("hover_pose") or place_choice.get("pre_place_pose")
        if hover_pose is not None:
            return hover_pose
    if place_mode in {"surface_place", "vertical_place"}:
        distance = 0.08 if place_mode == "vertical_place" else 0.06
        p = targeted.base.flatten_np(demo.tcp.pose.p)[:3].astype(np.float32)
        return targeted.base.make_pose_with_position(
            demo.tcp.pose,
            (p + np.array([0.0, 0.0, distance], dtype=np.float32)).astype(np.float32),
        )
    return targeted.base.make_tcp_axis_retreat_pose(demo, 0.05)


def _build_direct_pre_place_candidates(demo, bridge_mod, scene_capture_cache, rule, place_state_cache, args, *, T_tcp_obj_override=None):
    place_plan_candidates = targeted.build_targeted_place_plan_variants(
        demo,
        bridge_mod,
        scene_capture_cache,
        rule,
        place_state_cache,
        args,
        T_tcp_obj_override=T_tcp_obj_override,
    )
    if not place_plan_candidates:
        return []

    target_axis = None
    if rule.primitive == "insert_vertical":
        T_world_target = targeted._get_scene_object_world_transform(
            demo,
            bridge_mod,
            scene_capture_cache,
            rule.target_object_name,
        )
        if T_world_target is not None:
            target_axis = _normalize(np.asarray(T_world_target[:3, 1], dtype=np.float32).reshape(3))
            sign = 1.0 if float(getattr(rule.object_pose_local, "position", (0.0, 1.0, 0.0))[1]) >= 0.0 else -1.0
            if target_axis is not None:
                target_axis = (target_axis * sign).astype(np.float32)

    if rule.primitive == "insert_vertical":
        approach_distances = [float(max(getattr(args, "direct_release_approach_distance", 0.04), 0.005))]
        z_offsets = [0.0]
    else:
        approach_distances = _unique_finite_float_list(
            getattr(args, "direct_release_approach_distances", []),
            min_value=0.005,
        )
        if not approach_distances:
            approach_distances = [float(max(getattr(args, "direct_release_approach_distance", 0.04), 0.005))]
        z_offsets = _unique_finite_float_list(getattr(args, "direct_pre_place_z_offsets", [0.0]))
        if not z_offsets:
            z_offsets = [0.0]
    min_place_tcp_z = float(getattr(args, "direct_min_place_tcp_z", 0.005))
    max_pad_tilt = float(max(getattr(args, "direct_place_max_pad_tilt", 0.25), 0.0))
    if getattr(rule, "tabletop_place_tcp_verticality_target", None) is not None:
        max_pad_tilt = 1.0

    # 获取物体尺寸用于计算倾斜时的最低点偏移
    try:
        object_dims = targeted.base.get_asset_box_size(args.sim_asset_file, args.sim_asset_scale)
    except Exception:
        object_dims = None

    all_candidates = []
    level_candidates = []
    for candidate in place_plan_candidates:
        pad_tilt = _tcp_pad_tilt_z(candidate.place_pose)
        place_z = float(_get_pose_position(candidate.place_pose)[2])

        # 计算物体倾斜时的最低点偏移
        lowest_point_offset = 0.0
        if object_dims is not None:
            lowest_point_offset = _compute_object_lowest_point_offset(candidate.place_pose, object_dims)

        # 调整最小高度检查：考虑物体最低点
        effective_min_z = min_place_tcp_z - lowest_point_offset

        vlabel = candidate.variant_label or "base"
        if place_z < effective_min_z:
            if object_dims is not None:
                print(
                    f"[direct_pre_place] rejected {vlabel}: place_z={place_z:.4f}, "
                    f"lowest_offset={lowest_point_offset:.4f}, effective_min_z={effective_min_z:.4f}"
                )
            continue
        is_level = pad_tilt <= max_pad_tilt
        for approach_distance in approach_distances:
            short_pre_place_pose = _make_short_pre_place_pose(candidate, target_axis, float(approach_distance))
            for z_offset in z_offsets:
                pose = short_pre_place_pose
                if abs(float(z_offset)) > 1e-6:
                    pose = targeted.base.make_pose_with_position(
                        pose,
                        (_get_pose_position(pose) + np.array([0.0, 0.0, float(z_offset)], dtype=np.float32)).astype(np.float32),
                    )
                pose_z = float(_get_pose_position(pose)[2])
                # 同样考虑物体最低点偏移
                if pose_z < effective_min_z:
                    continue
                label = "pre_place" if candidate.variant_label is None else f"pre_place_{candidate.variant_label}"
                if len(approach_distances) > 1:
                    label = f"{label}_approach_{int(round(float(approach_distance) * 1000.0))}mm"
                if abs(float(z_offset)) > 1e-6:
                    label = f"{label}_lift_{int(round(float(z_offset) * 1000.0))}mm"
                item = {
                    "label": label,
                    "pose": pose,
                    "place_pose": candidate.place_pose,
                    "retreat_pose": pose,
                    "target_name": candidate.target_name,
                    "slot_name": candidate.slot_name,
                    "variant_label": candidate.variant_label,
                    "tcp_verticality": float(candidate.tcp_verticality),
                    "pad_tilt": float(pad_tilt),
                    "place_plan": candidate,
                }
                all_candidates.append(item)
                if is_level:
                    level_candidates.append(item)
    if level_candidates:
        candidates = _dedupe_place_candidates(level_candidates)
    elif all_candidates:
        all_candidates.sort(key=lambda c: float(c["pad_tilt"]))
        candidates = _dedupe_place_candidates(all_candidates)
        print(
            f"[direct_pre_place] WARNING: no candidate passed pad-level filter "
            f"(max_pad_tilt={max_pad_tilt:.2f}), keeping all {len(candidates)} sorted by pad_tilt"
        )
    else:
        candidates = []
    candidates = _filter_fixed_tabletop_release_candidates(
        candidates,
        _current_source_object_name(args),
        label="short_pre_place",
    )
    candidates = _filter_place_candidates_by_tcp_verticality_target(candidates, args, label="short_pre_place")
    print(
        f"[direct_pre_place] built {len(candidates)} short pre-place candidate(s) "
        f"with approach_distances={np.round(np.asarray(approach_distances, dtype=np.float32), 4).tolist()} "
        f"and z_offsets={np.round(np.asarray(z_offsets, dtype=np.float32), 4).tolist()}"
    )
    return candidates


def _build_direct_place_candidates(demo, bridge_mod, scene_capture_cache, rule, place_state_cache, args, *, T_tcp_obj_override=None):
    place_plan_candidates = targeted.build_targeted_place_plan_variants(
        demo,
        bridge_mod,
        scene_capture_cache,
        rule,
        place_state_cache,
        args,
        T_tcp_obj_override=T_tcp_obj_override,
    )
    if not place_plan_candidates:
        return []

    min_place_tcp_z = float(getattr(args, "direct_min_place_tcp_z", 0.005))
    max_pad_tilt = float(max(getattr(args, "direct_place_max_pad_tilt", 0.25), 0.0))
    if getattr(rule, "tabletop_place_tcp_verticality_target", None) is not None:
        max_pad_tilt = 1.0
    max_abs_yaw = float(getattr(args, "direct_place_max_abs_yaw_deg", 0.0) or 0.0)
    source_name = _current_source_object_name(args)
    try:
        object_dims = targeted.base.get_asset_box_size(args.sim_asset_file, args.sim_asset_scale)
    except Exception:
        object_dims = None
    object_category = classify_object_category(args, rule, object_dims)
    sphere_category = object_category == "sphere"
    # Carriot is elongated, not orientation-free: its final object pose must
    # follow the tabletop rule instead of targeting only the center point.
    center_only_orientation_free = False
    if sphere_category and len(place_plan_candidates) > 1:
        place_plan_candidates = [
            min(
                place_plan_candidates,
                key=lambda c: (
                    _variant_abs_yaw_deg_from_labels(c.variant_label, None),
                    str(c.variant_label or ""),
                ),
            )
        ]
    place_mode = classify_place_mode(args, rule, object_dims)
    release_lift = _place_mode_release_lift_m(args, rule, place_mode)
    if release_lift > 1e-6:
        print(
            f"[place_state] category={object_category}, mode={place_mode}, release lift for {source_name or 'object'}: "
            f"+{release_lift:.3f} m"
        )
    else:
        print(f"[place_state] category={object_category}, mode={place_mode}, release lift disabled")
    center_tcp_object_distance = 0.0
    if sphere_category or center_only_orientation_free:
        center_tcp_object_distance = _sphere_tcp_object_distance_m(T_tcp_obj_override, args)
        print(
            f"[place_state] center-only orientation-free release: target object center directly, "
            f"tcp_object_distance={center_tcp_object_distance:.4f} m"
        )
    all_candidates = []
    level_candidates = []
    target_axis = None
    insert_approach_distances: list[float] = []
    if place_mode == "insert_place":
        T_world_target = targeted._get_scene_object_world_transform(
            demo,
            bridge_mod,
            scene_capture_cache,
            rule.target_object_name,
        )
        if T_world_target is not None:
            target_axis = _normalize(np.asarray(T_world_target[:3, 1], dtype=np.float32).reshape(3))
            sign = 1.0 if float(getattr(rule.object_pose_local, "position", (0.0, 1.0, 0.0))[1]) >= 0.0 else -1.0
            if target_axis is not None:
                target_axis = (target_axis * sign).astype(np.float32)
        insert_approach_distances = _unique_finite_float_list(
            list(getattr(args, "direct_release_approach_distances", []) or [])
            + [float(max(getattr(args, "direct_release_approach_distance", 0.04), 0.005))],
            min_value=0.005,
        )
        if not insert_approach_distances:
            insert_approach_distances = [0.04]
    hover_extra_heights = [0.0]
    if place_mode != "drop_place":
        hover_extra_heights = _unique_finite_float_list(
            getattr(args, "transport_hover_extra_heights_m", [0.0, 0.03, 0.06]),
            min_value=0.0,
        )
        if not hover_extra_heights:
            hover_extra_heights = [0.0]
        if sphere_category:
            # The base surface hover already gives balls the required release
            # descent. Extra hover heights multiply tennis candidates without
            # improving the final object pose.
            hover_extra_heights = [0.0]
    insert_release_height_offsets = [0.0]
    if place_mode == "insert_place" and source_name == "bi":
        insert_release_height_offsets = _unique_finite_float_list(
            getattr(args, "bi_insert_release_height_offsets_m", [0.0]),
            min_value=0.0,
        )
        if not insert_release_height_offsets:
            insert_release_height_offsets = [0.0]
    for candidate in place_plan_candidates:
        raw_release_pose = candidate.place_pose
        object_center_p = None
        if (sphere_category or center_only_orientation_free) and getattr(candidate, "T_world_obj_desired", None) is not None:
            try:
                object_center_p = np.asarray(candidate.T_world_obj_desired, dtype=np.float32).reshape(4, 4)[:3, 3]
            except Exception:
                object_center_p = None
        release_pose_variants = (
            _make_sphere_free_release_pose_variants(
                demo,
                raw_release_pose,
                candidate.variant_label,
                args,
                object_center_p=object_center_p,
                tcp_object_distance_m=center_tcp_object_distance,
            )
            if (sphere_category or center_only_orientation_free)
            else [(None, raw_release_pose)]
        )
        for release_variant_label, base_release_pose in release_pose_variants:
            for insert_release_offset in insert_release_height_offsets:
                release_pose = base_release_pose
                release_height_label = None
                if float(insert_release_offset) > 1e-6:
                    offset_axis = target_axis
                    if offset_axis is None:
                        offset_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    release_pose = targeted.base.make_pose_with_position(
                        release_pose,
                        (_get_pose_position(release_pose) + offset_axis * float(insert_release_offset)).astype(np.float32),
                    )
                    release_height_label = f"release_up_{int(round(float(insert_release_offset) * 1000.0))}mm"
                combined_variant_parts = [
                    part for part in (candidate.variant_label, release_variant_label, release_height_label) if part
                ]
                combined_variant_label = "+".join(combined_variant_parts) if combined_variant_parts else None
                if release_lift > 1e-6:
                    release_pose = _lift_pose_world_z(release_pose, release_lift)
                if place_mode == "insert_place" and target_axis is not None:
                    base_hover_entries = []
                    release_p = _get_pose_position(release_pose)
                    for approach_distance in insert_approach_distances:
                        approach_pose = targeted.base.make_pose_with_position(
                            release_pose,
                            (release_p + target_axis * float(approach_distance)).astype(np.float32),
                        )
                        base_hover_entries.append(
                            (
                                f"approach_{int(round(float(approach_distance) * 1000.0))}mm",
                                approach_pose,
                            )
                        )
                else:
                    base_hover_entries = [
                        (
                            None,
                            build_hover_pose(
                                release_pose,
                                place_mode,
                                args,
                                rule,
                                candidate_pre_place_pose=candidate.pre_place_pose,
                            ),
                        )
                    ]
                pad_tilt = _tcp_pad_tilt_z(release_pose)
                place_z = float(_get_pose_position(release_pose)[2])
                if place_z < min_place_tcp_z:
                    continue
                for hover_variant_label, base_hover_pose in base_hover_entries:
                    label_parts = [combined_variant_label, hover_variant_label]
                    hover_variant = "+".join([part for part in label_parts if part]) or None
                    label = "transport_hover" if hover_variant is None else f"transport_hover_{hover_variant}"
                    if max_abs_yaw > 0.0:
                        ydeg = _variant_abs_yaw_deg_from_labels(hover_variant, label)
                        if ydeg > max_abs_yaw + 1e-4:
                            continue
                    for hover_extra in hover_extra_heights:
                        hover_pose = base_hover_pose
                        if float(hover_extra) > 1e-6:
                            # Keep hover_plus on the same release->hover approach line.  Adding
                            # world-z here makes the final-contact delta multi-axis in the goal
                            # frame, which defeats cuRobo's constrained approach metric.
                            hover_pose = _extend_hover_pose_along_release_approach(
                                release_pose,
                                base_hover_pose,
                                float(hover_extra),
                            )
                        hover_label = label
                        if float(hover_extra) > 1e-6:
                            hover_label = f"{hover_label}_hover_plus_{int(round(float(hover_extra) * 1000.0))}mm"
                        item = {
                            "label": hover_label,
                            "pose": hover_pose,
                            "hover_pose": hover_pose,
                            "pre_place_pose": hover_pose,
                            "place_pose": release_pose,
                            "release_pose": release_pose,
                            "raw_release_pose": raw_release_pose,
                            "retreat_pose": hover_pose,
                            "target_name": candidate.target_name,
                            "slot_name": candidate.slot_name,
                            "variant_label": hover_variant,
                            "tcp_verticality": float(candidate.tcp_verticality),
                            "pad_tilt": float(pad_tilt),
                            "place_plan": candidate,
                            "place_mode": place_mode,
                            "object_category": object_category,
                            "direct_place_mode": False,
                            "transport_to_hover": True,
                            "release_lift_m": float(release_lift),
                            "insert_release_height_offset_m": float(insert_release_offset),
                            "hover_extra_height_m": float(hover_extra),
                        }
                        all_candidates.append(item)
                        if pad_tilt <= max_pad_tilt:
                            level_candidates.append(item)
    if level_candidates:
        candidates = _dedupe_place_candidates(all_candidates)
        candidates.sort(key=_pre_place_screen_sort_key)
        print(
            f"[place_state] built {len(candidates)} transport hover candidate(s) "
            f"(pad-level filter would keep {len(level_candidates)}/{len(all_candidates)}, "
            f"but retaining tilted variants too; max_pad_tilt={max_pad_tilt:.2f})"
        )
    elif all_candidates:
        all_candidates.sort(key=lambda c: float(c["pad_tilt"]))
        candidates = _dedupe_place_candidates(all_candidates)
        candidates.sort(key=_pre_place_screen_sort_key)
        best_tilt = float(candidates[0]["pad_tilt"]) if candidates else float("nan")
        print(
            f"[place_state] WARNING: no candidate passed pad-level filter (max_pad_tilt={max_pad_tilt:.2f}), "
            f"keeping all {len(candidates)} candidate(s) sorted by pad_tilt (best={best_tilt:.3f}). "
            f"This place task may require a tilted gripper."
        )
    else:
        candidates = []
        print("[place_state] built 0 transport hover candidate(s) (all rejected by tcp_z threshold)")
    candidates = _filter_fixed_tabletop_release_candidates(candidates, source_name, label="transport_hover")
    candidates = _filter_place_candidates_by_tcp_verticality_target(candidates, args, label="transport_hover")
    return candidates


def _fast_chain_preselect_grasp_place_pair(
    planner,
    demo,
    bridge_mod,
    args,
    scene_capture_cache,
    place_state_cache,
    rule,
    grasp_candidates,
    start_q,
    *,
    disabled_world_collision_links: list[str] | None,
) -> dict | None:
    """Choose one grasp-place chain with cheap IK before any candidate MotionGen.

    This is the integrated fast path: it does not prove the whole trajectory is
    safe, but it picks one grasp candidate whose pregrasp/grasp and matching
    place hover/release poses all have IK.  The expensive MotionGen stages then
    validate only that winner chain first.
    """
    if not bool(getattr(args, "fast_chain_screening", False)):
        return None
    candidates = [dict(item) for item in list(grasp_candidates or []) if item.get("pregrasp_pose") is not None]
    if not candidates:
        return None
    source_name = _current_source_object_name(args)
    max_grasps = int(getattr(args, "fast_chain_preselect_grasp_candidates", 6) or 0)
    if _is_fixed_tabletop_source(source_name):
        # Fixed tabletop objects rely on the lifted duplicates and both tilt
        # signs. This stage is IK-only, so keep all generated relations here
        # instead of truncating away the eventual successful pair.
        max_grasps = 0
    if max_grasps > 0:
        candidates = candidates[:max_grasps]
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    ee_link_name = str(getattr(planner.config, "ee_link", "gripper_tcp"))
    place_screen_args = _transport_screen_args_for_object(args, source_name)
    with _profile_stage(
        args,
        "winner_chain_ik_preselect",
        candidate_count=len(candidates),
        num_ik_seeds=int(getattr(args, "fast_chain_ik_seeds", 32) or getattr(args, "curobo_num_ik_seeds", 64)),
        fast_chain_cuda_graph_ik=bool(getattr(args, "fast_chain_cuda_graph_ik", False)),
        fast_chain_cuda_graph_fixed_batch_size=int(getattr(args, "fast_chain_cuda_graph_ik_fixed_batch_size", 16) or 0),
    ) as prof:
        _refresh_curobo_world(
            planner,
            demo,
            args,
            label="winner_chain_ik_preselect_grasp",
            include_active_object=True,
            include_table=False,
        )
        disabled = _set_world_collision_for_links(
            planner,
            disabled_world_collision_links,
            enabled=False,
            label="winner_chain_ik_preselect_grasp",
        )
        grasp_ik_candidates: list[dict] = []
        try:
            pregrasp_poses = [
                _convert_demo_tcp_pose_to_curobo_ee_pose(
                    demo,
                    item["pregrasp_pose"],
                    ee_link_name=ee_link_name,
                )
                for item in candidates
            ]
            pregrasp_results = _profile_fast_chain_solve_batch_start_goal_ik(
                args,
                planner,
                [start_q for _ in candidates],
                pregrasp_poses,
                num_seeds=int(getattr(args, "fast_chain_ik_seeds", 32) or getattr(args, "curobo_num_ik_seeds", 64)),
            )
            pregrasp_ok: list[dict] = []
            for candidate, ik_result in zip(candidates, pregrasp_results):
                if not bool(ik_result.success) or ik_result.goal_joint is None:
                    continue
                pos_err, rot_err = _ik_debug_errors(ik_result)
                q_pregrasp = np.asarray(ik_result.goal_joint, dtype=np.float32).reshape(-1)[:7]
                candidate["_winner_preselect_q_pregrasp"] = q_pregrasp
                candidate["q_pregrasp"] = q_pregrasp
                candidate["_winner_preselect_pregrasp_ik_score"] = _ik_score(pos_err, rot_err)
                pregrasp_ok.append(candidate)
            if pregrasp_ok:
                _refresh_curobo_world(
                    planner,
                    demo,
                    args,
                    label="winner_chain_ik_preselect_grasp_contact",
                    include_active_object=False,
                    include_table=False,
                )
                grasp_poses = [
                    _convert_demo_tcp_pose_to_curobo_ee_pose(
                        demo,
                        item["pose"],
                        ee_link_name=ee_link_name,
                    )
                    for item in pregrasp_ok
                ]
                grasp_results = _profile_fast_chain_solve_batch_start_goal_ik(
                    args,
                    planner,
                    [np.asarray(item["_winner_preselect_q_pregrasp"], dtype=np.float32).reshape(-1)[:7] for item in pregrasp_ok],
                    grasp_poses,
                    num_seeds=int(getattr(args, "fast_chain_ik_seeds", 32) or getattr(args, "curobo_num_ik_seeds", 64)),
                )
                for candidate, ik_result in zip(pregrasp_ok, grasp_results):
                    pos_err, rot_err = _ik_debug_errors(ik_result)
                    if not bool(ik_result.success) or ik_result.goal_joint is None:
                        continue
                    q_grasp = np.asarray(ik_result.goal_joint, dtype=np.float32).reshape(-1)[:7]
                    approach_q_path = _build_validated_linear_path_to_q(
                        demo,
                        args,
                        candidate["_winner_preselect_q_pregrasp"],
                        q_grasp,
                        candidate["pregrasp_pose"],
                        candidate["pose"],
                        label=f"{candidate.get('label', 'grasp')}_winner_chain_grasp_approach",
                        use_attach=False,
                        validation_pos_tol_m=float(
                            max(getattr(args, "strict_short_linear_waypoint_pos_tol_m", 0.015), 0.0)
                        ),
                    )
                    if approach_q_path is None:
                        print(
                            "[winner_chain] rejected grasp candidate before transport: "
                            f"{candidate.get('label')} has IK but no validated straight pregrasp->grasp primitive"
                        )
                        continue
                    grasp_score = _ik_score(pos_err, rot_err)
                    candidate["_winner_preselect_q_grasp"] = q_grasp
                    candidate["q_grasp"] = q_grasp
                    candidate["_winner_preselect_grasp_approach_q_path"] = [
                        np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in approach_q_path
                    ]
                    candidate["_winner_preselect_grasp_ik_score"] = float(grasp_score)
                    grasp_ik_candidates.append(candidate)
        finally:
            _set_world_collision_for_links(
                planner,
                disabled,
                enabled=True,
                label="winner_chain_ik_preselect_grasp",
            )

        prof["candidate_count_after_ik"] = len(grasp_ik_candidates)
        if not grasp_ik_candidates:
            prof["success"] = False
            prof["status"] = "NO_GRASP_IK"
            print(
                "[winner_chain] IK preselect found no candidate with both pregrasp and grasp IK "
                f"(pregrasp_ok={len(pregrasp_ok)}/{len(candidates)}); falling back only if caller allows it"
            )
            return None

        prof["grasp_candidate_count_before_place_rank"] = len(grasp_ik_candidates)
        place_rank_grasp_limit = int(getattr(args, "fast_chain_place_rank_grasp_limit", 3) or 0)
        if place_rank_grasp_limit > 0 and len(grasp_ik_candidates) > place_rank_grasp_limit:
            print(
                "[winner_chain] limiting place IK ranking to "
                f"{place_rank_grasp_limit}/{len(grasp_ik_candidates)} grasp candidate(s) "
                "in generated priority order"
            )
            grasp_ik_candidates = grasp_ik_candidates[:place_rank_grasp_limit]
        prof["grasp_candidate_count_place_ranked"] = len(grasp_ik_candidates)

        pair_records: list[dict] = []
        top_pair_count = max(1, int(getattr(args, "fast_chain_top_pairs", 1) or 1))
        first_valid_pair = bool(getattr(args, "fast_chain_preselect_first_valid", False))
        for grasp_candidate in grasp_ik_candidates:
            place_build_args = _bi_fast_insert_screen_args(args) if source_name == "bi" else args
            place_candidates = _build_direct_place_candidates(
                demo,
                bridge_mod,
                scene_capture_cache,
                rule,
                place_state_cache,
                place_build_args,
                T_tcp_obj_override=grasp_candidate.get("T_tcp_obj"),
            )
            if getattr(rule, "primitive", None) == "insert_vertical":
                place_candidates = _filter_pre_place_candidates_by_verticality(place_candidates, place_build_args)
            place_candidates = sorted(list(place_candidates or []), key=_pre_place_screen_sort_key)
            if source_name == "bi":
                small_place_candidates = _bi_insert_small_lane_candidates(
                    place_candidates,
                    args,
                    label=f"{grasp_candidate.get('label', 'grasp')}_winner_chain",
                )
                if small_place_candidates:
                    place_candidates = small_place_candidates
            if not place_candidates:
                continue
            q_grasp = np.asarray(grasp_candidate["_winner_preselect_q_grasp"], dtype=np.float32).reshape(-1)[:7]
            ranked_places = _fast_chain_rank_place_candidates(
                planner,
                demo,
                place_screen_args,
                place_candidates,
                q_grasp,
                label=f"{grasp_candidate.get('label', 'grasp')}_winner_chain",
                disabled_world_collision_links=_direct_place_contact_tolerant_disabled_links(planner),
            )
            if not ranked_places:
                continue
            grasp_joint_score = float(np.linalg.norm(grasp_candidate["_winner_preselect_q_pregrasp"] - start_q))
            grasp_joint_score += 0.35 * float(
                np.linalg.norm(grasp_candidate["_winner_preselect_q_grasp"] - grasp_candidate["_winner_preselect_q_pregrasp"])
            )
            for ranked_place in ranked_places[:top_pair_count]:
                place_candidate = dict(ranked_place)
                # q_hover/q_release are target IK solutions. They remain useful even
                # if the final MotionGen start q differs slightly from this cheap q_grasp.
                place_candidate.pop("_prefilter_start_q", None)
                score = (
                    float(place_candidate.get("pair_score", place_candidate.get("fast_chain_score", 0.0)) or 0.0)
                    + grasp_joint_score
                    + float(grasp_candidate.get("_winner_preselect_pregrasp_ik_score", 0.0) or 0.0)
                    + float(grasp_candidate.get("_winner_preselect_grasp_ik_score", 0.0) or 0.0)
                )
                pair_records.append(
                    {
                        "score": float(score),
                        "grasp_candidate": grasp_candidate,
                        "place_candidate": place_candidate,
                    }
                )
                if first_valid_pair:
                    print(
                        "[winner_chain] first-valid preselect accepted "
                        f"grasp={grasp_candidate.get('label')} place={place_candidate.get('label')}"
                    )
                    break
            if first_valid_pair and pair_records:
                break
        prof["candidate_count_motiongen"] = 0
        prof["winner_count"] = 1 if pair_records else 0
        if not pair_records:
            prof["success"] = False
            prof["status"] = "NO_PLACE_IK_PAIR"
            print(
                "[winner_chain] IK preselect found grasp IK candidates but no hover/release IK pair "
                f"(grasp_ik={len(grasp_ik_candidates)}/{len(candidates)})"
            )
            return None
        pair_records.sort(key=lambda item: float(item["score"]))
        top_pair_records = pair_records[:top_pair_count]
        top_pair_grasps: list[dict] = []
        grouped_grasps: dict[str, dict] = {}
        for rec in top_pair_records:
            source_grasp = rec["grasp_candidate"]
            grasp_key = str(source_grasp.get("label", f"grasp_{len(grouped_grasps)}"))
            p = dict(rec["place_candidate"])
            p["_winner_chain_pair_score"] = float(rec["score"])
            if grasp_key not in grouped_grasps:
                g = dict(source_grasp)
                g["_preselected_fast_place_candidates"] = []
                g["_winner_chain_preselected_place_labels"] = []
                g["_winner_chain_preselect_score"] = float(rec["score"])
                grouped_grasps[grasp_key] = g
                top_pair_grasps.append(g)
            g = grouped_grasps[grasp_key]
            g["_preselected_fast_place_candidates"].append(p)
            g["_winner_chain_preselected_place_labels"].append(str(p.get("label", "")))
            g["_winner_chain_preselect_score"] = min(
                float(g.get("_winner_chain_preselect_score", float("inf"))),
                float(rec["score"]),
            )
            if not g.get("_winner_chain_preselected_place_label"):
                g["_winner_chain_preselected_place_label"] = str(p.get("label", ""))
        best = top_pair_records[0]
        selected_grasp = top_pair_grasps[0]
        selected_place = dict((selected_grasp.get("_preselected_fast_place_candidates") or [best["place_candidate"]])[0])
        selected_grasp["_winner_chain_top_pair_grasps"] = top_pair_grasps
        prof["success"] = True
        prof["status"] = "Success"
        prof["selected_grasp_label"] = str(selected_grasp.get("label", ""))
        prof["selected_place_label"] = str(selected_place.get("label", ""))
        prof["path_score"] = float(best["score"])
        prof["winner_count"] = len(top_pair_records)
        prof["grasp_winner_count"] = len(top_pair_grasps)
        print(
            "[winner_chain] IK preselected grasp-place pair: "
            f"grasp={selected_grasp.get('label')} place={selected_place.get('label')} "
            f"score={float(best['score']):.3f}; top_pairs={len(top_pair_records)}, "
            f"top_grasps={len(top_pair_grasps)}"
        )
        if len(top_pair_records) > 1:
            print(
                "[winner_chain] top pair fallback order: "
                f"{[(g.get('label'), list(g.get('_winner_chain_preselected_place_labels') or [])) for g in top_pair_grasps]}"
            )
        return selected_grasp


def plan_transport_to_hover(
    planner,
    demo,
    args,
    start_q,
    hover_candidates,
    *,
    include_table: bool,
    exclude_object_names: set[str] | None,
    disabled_world_collision_links: list[str] | None,
):
    max_winners = int(max(getattr(args, "place_transport_max_winners", 3), 1))
    hover_candidates = list(hover_candidates or [])
    candidate_count = len(hover_candidates)
    print(
        f"[place_state] transport_to_hover evaluating {candidate_count} "
        f"candidate(s), max_winners={max_winners}"
    )
    with _profile_stage(
        args,
        "transport_to_hover",
        candidate_count=candidate_count,
        max_attempts=int(getattr(args, "curobo_max_attempts", 2)),
        num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
        num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
        enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
    ) as prof:
        winners = _evaluate_curobo_pose_candidates_goalset(
            planner,
            demo,
            args,
            start_q,
            hover_candidates,
            label="transport_to_hover",
            use_attach=True,
            max_winners=max_winners,
            include_table=include_table,
            exclude_object_names=exclude_object_names,
            disabled_world_collision_links=disabled_world_collision_links,
        )
        _copy_last_candidate_counts_to_profile(prof, planner)
        prof["winner_count"] = len(winners)
        prof["success"] = bool(winners)
        prof["status"] = "Success" if winners else "NO_WINNERS"
        prof["world_changed"] = bool(getattr(planner, "_last_world_changed", False))
        prof["cache_hit"] = bool(getattr(planner, "_last_world_cache_hit", False))
        if winners:
            prof["path_waypoints"] = int((winners[0].get("metrics") or {}).get("waypoint_count", 0) or 0)
            prof["path_score"] = float(winners[0].get("score", 0.0) or 0.0)
        return winners


def plan_final_contact_approach(
    planner,
    demo,
    args,
    start_q,
    transport_choice,
    *,
    disabled_world_collision_links: list[str] | None,
):
    start_t = time.perf_counter()
    counter_start = _snapshot_profile_counters()
    place_mode = str(transport_choice.get("place_mode", "drop_place"))
    pre_q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in transport_choice["q_path"]]
    hover_pose = transport_choice.get("hover_pose", transport_choice["pose"])
    release_pose = transport_choice.get("release_pose", transport_choice.get("place_pose", transport_choice["pose"]))
    if place_mode == "drop_place":
        item = dict(transport_choice)
        item["q_pre_place_path"] = pre_q_path
        item["q_place_path"] = [np.asarray(pre_q_path[-1], dtype=np.float32).reshape(-1)[:7]]
        item["pose"] = release_pose
        item["place_pose"] = release_pose
        item["two_stage_place"] = False
        item["final_contact_policy"] = "drop_release"
        print("[place_state] drop_place: transport hover is the release pose; no final contact descent")
        _record_profile(
            args,
            "final_contact",
            success=True,
            status="drop_release",
            elapsed_ms=round((time.perf_counter() - start_t) * 1000.0, 3),
            path_waypoints=len(pre_q_path),
            path_score=float(item.get("score", 0.0) or 0.0),
            **_profile_counter_delta(counter_start),
        )
        return item

    final_start_q = np.asarray(pre_q_path[-1], dtype=np.float32).reshape(-1)[:7]
    final_label = f"{transport_choice['label']}_final_contact"
    verticality_target = _candidate_tcp_verticality_target(transport_choice)
    axis_vertical_target = _candidate_tcp_axis_vertical_target(transport_choice)
    if bool(getattr(args, "curobo_debug", False)):
        axes_z = _tcp_axis_world_z_components(release_pose)
        object_axes_text = ""
        place_plan = transport_choice.get("place_plan")
        T_world_obj_desired = getattr(place_plan, "T_world_obj_desired", None)
        if T_world_obj_desired is not None:
            obj_axes_z = _rotation_axis_world_z_components(np.asarray(T_world_obj_desired, dtype=np.float32).reshape(4, 4)[:3, :3])
            object_axes_text = (
                f", object_axes_world_z: x={obj_axes_z['x']:.3f}, y={obj_axes_z['y']:.3f}, z={obj_axes_z['z']:.3f} "
                f"(abs: x={obj_axes_z['abs_x']:.3f}, y={obj_axes_z['abs_y']:.3f}, z={obj_axes_z['abs_z']:.3f})"
            )
        print(
            f"[place_state] {final_label} tcp_axes_world_z: "
            f"x={axes_z['x']:.3f}, y={axes_z['y']:.3f}, z={axes_z['z']:.3f} "
            f"(abs: x={axes_z['abs_x']:.3f}, y={axes_z['abs_y']:.3f}, z={axes_z['abs_z']:.3f}), "
            f"tcp_verticality={float(transport_choice.get('tcp_verticality', 0.0)):.3f}, "
            f"verticality_target={verticality_target}, axis_vertical_target={axis_vertical_target}"
            f"{object_axes_text}"
        )
    disabled = set_payload_collision_mode(
        planner,
        "disabled_for_contact",
        place_mode,
        disabled_world_collision_links=disabled_world_collision_links,
        label=final_label,
    )
    source_name = _current_source_object_name(args)
    strict_linear_final_contact = bool(getattr(args, "strict_final_contact_linear", True))
    use_segmented_final_contact = (
        bool(getattr(args, "final_contact_segmented_ik_first", False))
        or bool(getattr(args, "final_contact_segmented_ik_fallback", False))
        or bool(getattr(args, "allow_segmented_ik_rescue", False))
    )
    force_segmented_final_contact = bool(getattr(args, "final_contact_segmented_ik_first", False))
    if (
        bool(getattr(args, "curobo_debug", False))
        and not use_segmented_final_contact
        and (place_mode == "insert_place" or (verticality_target is not None and verticality_target < 0.5))
    ):
        print(
            f"[place_state] {final_label}: using selected-pair straight final-contact primitive "
            "(cached q_release / endpoint IK first; constrained metric only as fallback)"
        )
    if bool(getattr(args, "curobo_debug", False)) and strict_linear_final_contact:
        start_p = _get_pose_position(hover_pose)
        goal_p = _get_pose_position(release_pose)
        print(
            f"[place_state] {final_label}: using cuRobo PoseCostMetric constrained final contact "
            f"(distance={float(np.linalg.norm(goal_p - start_p)):.4f} m, "
            "segmented_ik=disabled, curved_path_rejected=yes)"
        )
    try:
        release_q_path = None
        cached_release_q = _q7_or_none(
            transport_choice.get("q_release", transport_choice.get("fast_chain_release_q"))
        )
        final_contact_backtrack_tol_m = float(
            max(getattr(args, "strict_final_contact_waypoint_backtrack_tol_m", 0.008), 0.0)
        )
        if cached_release_q is not None:
            release_q_path = _build_validated_linear_path_to_q(
                demo,
                args,
                final_start_q,
                cached_release_q,
                hover_pose,
                release_pose,
                label=f"{final_label}_cached_q_release",
                use_attach=False,
                validation_pos_tol_m=float(max(getattr(args, "strict_final_contact_waypoint_pos_tol_m", 0.012), 0.0)),
                max_backtrack_m=final_contact_backtrack_tol_m,
            )
            if release_q_path is not None:
                print(
                    f"[place_state] {final_label}: using pair-first cached q_release "
                    f"as the straight final-contact primitive ({len(release_q_path)} waypoint(s))"
                )
        if release_q_path is None:
            release_q_path = _plan_short_linear_segment_via_goal_ik(
                planner,
                demo,
                args,
                final_start_q,
                hover_pose,
                release_pose,
                label=f"{final_label}_release_ik",
                use_attach=False,
                validation_pos_tol_m=float(max(getattr(args, "strict_final_contact_waypoint_pos_tol_m", 0.012), 0.0)),
                max_backtrack_m=final_contact_backtrack_tol_m,
            )
            if release_q_path is not None:
                print(
                    f"[place_state] {final_label}: using endpoint-IK straight final contact "
                    f"({len(release_q_path)} waypoint(s))"
                )
        if release_q_path is None and use_segmented_final_contact:
            release_q_path = _plan_short_curobo_cartesian_descent(
                planner,
                demo,
                args,
                final_start_q,
                hover_pose,
                release_pose,
                label=final_label,
                force_segmented=force_segmented_final_contact,
            )
        elif release_q_path is None:
            release_q_path = _plan_release_with_motiongen_constraint(
                planner,
                demo,
                args,
                final_start_q,
                hover_pose,
                release_pose,
                label=final_label,
            )
    finally:
        set_payload_collision_mode(
            planner,
            "transport",
            place_mode,
            disabled_world_collision_links=disabled,
            label=final_label,
        )
    if release_q_path is None:
        print(f"[place_state] {final_label} failed")
        _record_profile(
            args,
            "final_contact",
            success=False,
            status="PLAN_FAIL",
            elapsed_ms=round((time.perf_counter() - start_t) * 1000.0, 3),
            enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
            num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
            num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
            **_profile_counter_delta(counter_start),
        )
        return None
    release_q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in release_q_path]
    if strict_linear_final_contact and not _validate_strict_linear_waypoints(
        demo,
        args,
        release_q_path,
        hover_pose,
        release_pose,
        label=final_label,
        max_backtrack_m=final_contact_backtrack_tol_m,
    ):
        _record_profile(
            args,
            "final_contact",
            success=False,
            status="STRICT_LINEAR_VALIDATE_FAIL",
            elapsed_ms=round((time.perf_counter() - start_t) * 1000.0, 3),
            path_waypoints=len(release_q_path),
            **_profile_counter_delta(counter_start),
        )
        return None
    if not _validate_candidate_joint_path_with_demo_planner(
        demo,
        final_start_q,
        release_q_path,
        use_attach=False,
        label=final_label,
    ):
        _record_profile(
            args,
            "final_contact",
            success=False,
            status="DEMO_VALIDATE_FAIL",
            elapsed_ms=round((time.perf_counter() - start_t) * 1000.0, 3),
            path_waypoints=len(release_q_path),
            **_profile_counter_delta(counter_start),
        )
        return None
    combined_path = pre_q_path + release_q_path[1:]
    metrics, score = _path_metrics_and_score(np.asarray(start_q, dtype=np.float32).reshape(-1)[:7], combined_path)
    item = dict(transport_choice)
    item["label"] = final_label
    item["pose"] = release_pose
    item["hover_pose"] = hover_pose
    item["pre_place_pose"] = hover_pose
    item["place_pose"] = release_pose
    item["release_pose"] = release_pose
    item["q_path"] = combined_path
    item["q_pre_place_path"] = pre_q_path
    item["q_place_path"] = release_q_path
    item["two_stage_place"] = True
    item["final_contact_policy"] = "payload_world_collision_disabled"
    item["metrics"] = metrics
    item["score"] = float(score)
    print(
        f"[place_state] final_contact success: mode={place_mode}, label={final_label}, "
        f"total_motion={metrics['total_motion']:.3f} rad, waypoints={metrics['waypoint_count']}"
    )
    _record_profile(
        args,
        "final_contact",
        success=True,
        status="Success",
        elapsed_ms=round((time.perf_counter() - start_t) * 1000.0, 3),
        path_waypoints=int(metrics.get("waypoint_count", 0) or 0),
        path_score=float(score),
        enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
        num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
        num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
        **_profile_counter_delta(counter_start),
    )
    return item


def _joint_chain_sort_key(item):
    return (
        float(item["total_score"]),
        float(item["grasp_choice"]["score"]),
        float(item["pre_place_choice"]["score"]),
        float(item["release_score"]),
    )


def _grasp_chain_eval_sort_key(item, rule):
    base_key = _candidate_sort_key(item)
    if not bool(getattr(rule, "preserve_long_axis_vertical", False)):
        return base_key
    label = str(item.get("label", "")).lower()
    is_vertical_relation = "vertical" in label
    is_tilted_relation = "tilt" in label
    axis_shift = abs(float(item.get("grasp_axis_shift_m", 0.0) or 0.0))
    z_lift = max(float(item.get("grasp_z_lift_m", 0.0) or 0.0), 0.0)
    return (
        0 if is_vertical_relation else 1,
        1 if is_tilted_relation else 0,
        axis_shift,
        z_lift,
        base_key,
    )


def _fixed_tabletop_grasp_chain_eval_sort_key(item, args, source_name: str | None) -> tuple:
    """Prefer fixed-pose relations that are likely to survive transport.

    For ordinary tabletop objects the placement pose is fixed and the
    grasp/place relation is the meaningful choice.  The +10mm grasp-lift
    variants have consistently avoided start-state and final-contact false
    negatives without changing the final object pose, so they are tried before
    raw contact-height duplicates.
    """
    base_key = _candidate_sort_key(item)
    label = str(item.get("label", "")).lower()
    configured_tilts = _fixed_tabletop_release_tilt_degs(args, source_name)
    signed_tilt = _fixed_tabletop_signed_tilt_from_label(label)
    tilt_rank = len(configured_tilts)
    for idx, value in enumerate(configured_tilts):
        if abs(float(value) - float(signed_tilt)) <= 1.0:
            tilt_rank = idx
            break
    place_first_rank = 0 if bool(item.get("place_first", False)) or "place_first" in label else 1
    z_lift = max(float(item.get("grasp_z_lift_m", 0.0) or 0.0), 0.0)
    axis_shift = abs(float(item.get("grasp_axis_shift_m", 0.0) or 0.0))
    lift_rank = 0 if z_lift >= 0.005 else 1
    return (
        place_first_rank,
        lift_rank,
        tilt_rank,
        z_lift,
        axis_shift,
        str(item.get("label", "")),
        base_key,
    )


def _lvmukuai_grasp_chain_eval_sort_key(item) -> tuple:
    base_key = _candidate_sort_key(item)
    label = str(item.get("label", "")).lower()
    is_vertical_helper = _is_fixed_tabletop_vertical_helper_label(label, "lvmukuai")
    is_tilted = "tilt" in label
    z_lift = max(float(item.get("grasp_z_lift_m", 0.0) or 0.0), 0.0)
    is_plain_direct = (not is_tilted) and z_lift <= 1e-4 and "lift" not in label
    tilt_mag = 0.0
    tilt_match = re.search(r"tilt_(?:toward|away)_robot_([-+]?\d+(?:\.\d+)?)deg", label)
    if tilt_match:
        tilt_mag = abs(float(tilt_match.group(1)))
    if is_vertical_helper:
        relation_class = 0
    elif is_tilted and abs(tilt_mag - 20.0) <= 1.0:
        relation_class = 1
    elif is_plain_direct:
        relation_class = 2
    elif is_tilted:
        relation_class = 3
    elif z_lift >= 0.005 or "lift_10mm" in label:
        relation_class = 4
    else:
        relation_class = 5
    toward_pref = 0 if "toward_robot" in label else 1
    lift_pref = 0 if z_lift >= 0.005 or "lift_10mm" in label else 1
    return (
        relation_class,
        lift_pref,
        toward_pref,
        abs(tilt_mag - 20.0) if is_tilted else 999.0,
        abs(float(item.get("grasp_axis_shift_m", 0.0) or 0.0)),
        base_key,
    )


def _hongshupian_grasp_chain_eval_sort_key(item) -> tuple:
    """Order chip-box grasps without changing the candidate set.

    Random-scene logs show the center vertical branch is often the fastest
    winner, but when it fails the shallow top-bias tilt branch is the next
    reliable branch.  Keeping all other vertical offsets as fallback avoids
    trading success rate for speed.
    """
    base_key = _candidate_sort_key(item)
    label = str(item.get("label", "")).lower()
    def _has_top_bias_magnitude(mm: int) -> bool:
        return f"top_bias_neg{int(mm)}" in label or f"top_bias_pos{int(mm)}" in label

    def _has_axis_magnitude(mm: int) -> bool:
        return f"axis_-{int(mm)}mm" in label or f"axis_{int(mm)}mm" in label

    is_center_vertical = "top_bias_center_vertical" in label
    is_neg10_tilt15_away = (
        _has_top_bias_magnitude(10)
        and "tilt15_away" in label
        and _has_axis_magnitude(10)
    )
    is_neg2_vertical = _has_top_bias_magnitude(2) and "vertical" in label
    is_neg6_vertical = _has_top_bias_magnitude(6) and "vertical" in label
    is_vertical = "vertical" in label
    is_tilt = "tilt" in label
    if is_center_vertical:
        relation_class = 0
    elif is_neg10_tilt15_away:
        relation_class = 1
    elif is_neg2_vertical:
        relation_class = 2
    elif is_neg6_vertical:
        relation_class = 3
    elif is_vertical:
        relation_class = 4
    elif is_tilt:
        relation_class = 5
    else:
        relation_class = 6
    return (
        relation_class,
        abs(float(item.get("grasp_axis_shift_m", 0.0) or 0.0)),
        max(float(item.get("grasp_z_lift_m", 0.0) or 0.0), 0.0),
        base_key,
    )


def _bi_grasp_chain_eval_sort_key(item) -> tuple:
    """Prefer pen grasp branches that consistently validate insertion fastest."""
    base_key = _candidate_sort_key(item)
    label = str(item.get("label", "")).lower()
    is_tilted = "tilt" in label
    roll_deg = abs(float(item.get("grasp_approach_roll_deg", 0.0) or 0.0))
    if not is_tilted and abs(roll_deg) <= 1.0:
        relation_class = 0
    elif not is_tilted and abs(roll_deg - 180.0) <= 1.0:
        relation_class = 1
    elif not is_tilted:
        relation_class = 2
    elif abs(roll_deg - 180.0) <= 1.0:
        relation_class = 3
    else:
        relation_class = 4
    return (
        relation_class,
        roll_deg,
        base_key,
    )


def _shuazi_grasp_label_priority(item) -> tuple:
    label = str(item.get("label", "")).lower()
    is_vertical_helper = _is_fixed_tabletop_vertical_helper_label(label, "shuazi")
    tilt_deg = 0.0
    tilt_match = re.search(r"tilt_([-+]?\d+(?:\.\d+)?)deg", label)
    if tilt_match:
        tilt_deg = abs(float(tilt_match.group(1)))
    has_yaw = "yaw" in label
    roll_deg = abs(float(item.get("grasp_approach_roll_deg", 0.0) or 0.0))
    has_roll = roll_deg > 1e-5 or "roll_" in label
    has_shift_20 = "shift_20mm" in label
    axis_shift = abs(float(item.get("grasp_axis_shift_m", 0.0) or 0.0))
    z_lift = max(float(item.get("grasp_z_lift_m", 0.0) or 0.0), 0.0)
    has_extra_axis_or_lift = axis_shift > 1e-5 or z_lift > 1e-5
    is_direct_axis_lift20 = "grasp_tilt" not in label and "axis_12mm_lift_20mm" in label
    is_direct_axis_lift10 = "grasp_tilt" not in label and "axis_12mm_lift_10mm" in label
    is_direct_axis = "grasp_tilt" not in label and "axis_12mm" in label

    is_tilt30_roll180 = (
        (has_yaw or has_roll)
        and abs(tilt_deg - 30.0) <= 1.0
        and abs(roll_deg - 180.0) <= 1.0
        and not has_extra_axis_or_lift
    )
    is_tilt30_roll_other = (
        (has_yaw or has_roll)
        and abs(tilt_deg - 30.0) <= 1.0
        and not is_tilt30_roll180
        and not has_extra_axis_or_lift
    )

    if is_vertical_helper:
        relation_class = 0
    elif is_tilt30_roll180:
        relation_class = 1
    elif is_direct_axis_lift20:
        # Crowded desk scenes can block the shallow tilted brush transport
        # branches after nearby objects are placed. Keep the proven 30deg/180
        # branch first, but try lifted direct-axis 90/270 before spending all
        # downstream checks on other tilted rolls.
        relation_class = 2
    elif has_shift_20 and abs(tilt_deg - 30.0) <= 1.0 and not has_extra_axis_or_lift:
        relation_class = 3
    elif is_direct_axis_lift10:
        relation_class = 4
    elif is_tilt30_roll_other:
        relation_class = 5
    elif is_direct_axis:
        relation_class = 6
    elif has_shift_20 and abs(tilt_deg - 30.0) <= 1.0:
        relation_class = 7
    elif has_shift_20 and abs(tilt_deg - 20.0) <= 1.0 and not has_extra_axis_or_lift:
        relation_class = 8
    elif "tilt" in label:
        relation_class = 9
    else:
        relation_class = 10

    if relation_class in {1, 3, 5, 7, 8, 9}:
        roll_pref = 0 if abs(roll_deg - 180.0) <= 1.0 else 1
    else:
        # For direct-axis lifted shuazi grasps, 90/270 have remained the faster
        # reachable branches in the gluestick-regression jitter scenes.
        roll_pref = 0 if abs(roll_deg - 90.0) <= 1.0 or abs(roll_deg - 270.0) <= 1.0 else 1

    return (
        relation_class,
        abs(tilt_deg - 30.0) if tilt_deg > 0.0 else 999.0,
        roll_pref,
        roll_deg,
        axis_shift,
        z_lift,
        str(item.get("label", "")),
    )


def _shuazi_grasp_chain_eval_sort_key(item) -> tuple:
    base_key = _candidate_sort_key(item)
    return (
        *_shuazi_grasp_label_priority(item),
        base_key,
    )


def _evaluate_joint_grasp_place_chains(
    planner,
    demo,
    bridge_mod,
    args,
    scene_capture_cache,
    place_state_cache,
    rule,
    grasp_successes,
):
    saved_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    try:
        saved_obj_pose = tuple(np.asarray(v, dtype=np.float32).copy() for v in demo.get_obj_pose())
    except Exception:
        saved_obj_pose = None
    saved_payload_state = _snapshot_transport_payload_state(demo)
    demo._last_joint_chain_failed_pose = None
    demo._last_joint_chain_failed_label = None
    demo._last_joint_chain_failed_candidate_poses = []
    demo._last_joint_chain_failed_candidate_pose_keys = set()
    demo._last_joint_chain_failed_candidate_records = []
    demo._last_joint_chain_failed_start_q = None
    chains = []
    grasp_candidates = list(grasp_successes or [])
    max_grasp_candidates = int(getattr(args, "joint_search_max_grasp_candidates", 8))
    source_name = _current_source_object_name(args)
    if (
        rule is not None
        and bool(getattr(rule, "preserve_long_axis_vertical", False))
        and max_grasp_candidates > 0
        and max_grasp_candidates < 5
    ):
        print(
            "[joint_search] vertical long-axis placement: increasing downstream grasp expansion "
            f"{max_grasp_candidates}->5 so final-approach can test deeper axis biases"
        )
        max_grasp_candidates = 5
    bi_min_grasp_candidates = max(1, min(4, len(_bi_direct_grasp_approach_roll_degs(args))))
    if source_name == "bi" and max_grasp_candidates > 0 and max_grasp_candidates < bi_min_grasp_candidates:
        print(
            "[joint_search] bi insert: increasing downstream grasp expansion "
            f"{max_grasp_candidates}->{bi_min_grasp_candidates} to test equivalent wrist-roll branches"
        )
        max_grasp_candidates = bi_min_grasp_candidates
    fixed_tabletop_min_grasp_candidates = max(1, len(_fixed_tabletop_release_tilt_degs(args, source_name)))
    if (
        _is_fixed_tabletop_source(source_name)
        and max_grasp_candidates > 0
        and max_grasp_candidates < fixed_tabletop_min_grasp_candidates
    ):
        print(
            "[joint_search] fixed-tabletop: increasing downstream grasp expansion "
            f"{max_grasp_candidates}->{fixed_tabletop_min_grasp_candidates} so fixed-pose vertical-gripper relations are evaluated"
        )
        max_grasp_candidates = fixed_tabletop_min_grasp_candidates
    # Some jittered desk poses need the 180deg approach-roll branch for a
    # release pose whose final straight contact is IK-feasible.  The first four
    # prioritized winners can be only 90/270deg roll variants, so keep a small
    # six-branch floor instead of falling back to an expensive full search.
    shuazi_min_grasp_candidates = 6
    if source_name == "shuazi" and max_grasp_candidates > 0 and max_grasp_candidates < shuazi_min_grasp_candidates:
        print(
            "[joint_search] shuazi: increasing downstream grasp expansion "
            f"{max_grasp_candidates}->{shuazi_min_grasp_candidates} so approach-roll grasp relations are evaluated"
        )
        max_grasp_candidates = shuazi_min_grasp_candidates
    if bool(getattr(rule, "preserve_long_axis_vertical", False)):
        before_labels = [str(c.get("label", "?")) for c in grasp_candidates]
        grasp_candidates.sort(key=lambda item: _grasp_chain_eval_sort_key(item, rule))
        after_labels = [str(c.get("label", "?")) for c in grasp_candidates]
        if after_labels != before_labels:
            print(
                "[joint_search] vertical long-axis placement: trying vertical grasp relations first "
                f"(order={after_labels})"
            )
    if _is_fixed_tabletop_source(source_name):
        before_labels = [str(c.get("label", "?")) for c in grasp_candidates]
        grasp_candidates.sort(key=lambda item: _fixed_tabletop_grasp_chain_eval_sort_key(item, args, source_name))
        after_labels = [str(c.get("label", "?")) for c in grasp_candidates]
        if after_labels != before_labels:
            print(
                "[joint_search] fixed-tabletop: trying configured world-Z tilt relations before lifted duplicates "
                f"(order={after_labels})"
            )
    elif source_name == "bi":
        before_labels = [str(c.get("label", "?")) for c in grasp_candidates]
        grasp_candidates.sort(key=_bi_grasp_chain_eval_sort_key)
        after_labels = [str(c.get("label", "?")) for c in grasp_candidates]
        if after_labels != before_labels:
            print(
                "[joint_search] bi: trying raw roll insertion-proven grasp relations first "
                f"(order={after_labels})"
            )
    elif source_name == "lvmukuai":
        before_labels = [str(c.get("label", "?")) for c in grasp_candidates]
        grasp_candidates.sort(key=_lvmukuai_grasp_chain_eval_sort_key)
        after_labels = [str(c.get("label", "?")) for c in grasp_candidates]
        if after_labels != before_labels:
            print(
                "[joint_search] lvmukuai: trying fixed-pose vertical-gripper relations first "
                f"(order={after_labels})"
            )
    elif source_name == "hongshupian":
        before_labels = [str(c.get("label", "?")) for c in grasp_candidates]
        grasp_candidates.sort(key=_hongshupian_grasp_chain_eval_sort_key)
        after_labels = [str(c.get("label", "?")) for c in grasp_candidates]
        if after_labels != before_labels:
            print(
                f"[joint_search] {source_name}: trying center vertical then shallow top-bias tilt before other offsets "
                f"(order={after_labels})"
            )
    elif source_name == "shuazi":
        before_labels = [str(c.get("label", "?")) for c in grasp_candidates]
        grasp_candidates.sort(key=_shuazi_grasp_chain_eval_sort_key)
        after_labels = [str(c.get("label", "?")) for c in grasp_candidates]
        if after_labels != before_labels:
            print(
                "[joint_search] shuazi: trying transport-proven tilted/shifted grasp relations first "
                f"(order={after_labels})"
            )
    if max_grasp_candidates > 0 and len(grasp_candidates) > max_grasp_candidates:
        if _is_fixed_tabletop_source(source_name) or source_name == "hongshupian":
            grasp_candidates = grasp_candidates[:max_grasp_candidates]
        else:
            non_tilt = [c for c in grasp_candidates if "tilt" not in str(c.get("label", "")).lower()]
            tilted = [c for c in grasp_candidates if "tilt" in str(c.get("label", "")).lower()]
            min_tilt_slots = min(max(max_grasp_candidates // 3, 1), len(tilted))
            non_tilt_slots = max_grasp_candidates - min_tilt_slots
            selected = non_tilt[:non_tilt_slots] + tilted[:min_tilt_slots]
            if len(selected) < max_grasp_candidates:
                remaining_pool = [c for c in grasp_candidates if c not in selected]
                selected.extend(remaining_pool[: max_grasp_candidates - len(selected)])
            grasp_candidates = selected
    all_labels = [str(c.get("label", "?")) for c in grasp_candidates]
    if _is_fixed_tabletop_source(source_name):
        tilt_count = sum(1 for l in all_labels if _fixed_tabletop_has_nonzero_tilt(l))
    else:
        tilt_count = sum(1 for l in all_labels if "tilt" in l.lower())
    print(
        f"[joint_search] grasp winners for chain evaluation: {len(grasp_candidates)} "
        f"(tilted={tilt_count}): {all_labels}"
    )
    max_feasible_chains = int(getattr(args, "joint_search_max_feasible_chains", 1))
    screen_timeout = float(getattr(args, "joint_search_screen_timeout", 0.0))
    screen_max_attempts = int(getattr(args, "joint_search_screen_max_attempts", 0))
    screen_num_ik_seeds = int(getattr(args, "joint_search_screen_num_ik_seeds", 0))
    screen_num_trajopt_seeds = int(getattr(args, "joint_search_screen_num_trajopt_seeds", 0))
    screen_timeout_arg = None if screen_timeout <= 0.0 else screen_timeout
    screen_max_attempts_arg = None if screen_max_attempts <= 0 else screen_max_attempts
    screen_num_ik_seeds_arg = None if screen_num_ik_seeds <= 0 else screen_num_ik_seeds
    screen_num_trajopt_seeds_arg = None if screen_num_trajopt_seeds <= 0 else screen_num_trajopt_seeds
    target_obj_name = curobo_wrapper.normalize_object_name(
        getattr(rule, "target_object_name", None)
    )
    transport_screen_args = _transport_screen_args_for_object(args, _current_source_object_name(args))
    exclude_names = _attached_source_exclude_names(args, rule)
    # Once the source object is attached, keeping its world obstacle creates a duplicate
    # collision model and can invalidate the transport start state.
    if rule.primitive == "insert_vertical" and target_obj_name:
        exclude_names.add(target_obj_name)
    exclude_names = exclude_names or None
    direct_place_disabled_links = _direct_place_contact_tolerant_disabled_links(planner)
    try:
        object_dims = targeted.base.get_asset_box_size(args.sim_asset_file, args.sim_asset_scale)
    except Exception:
        object_dims = None
    sphere_category = classify_object_category(args, rule, object_dims) == "sphere"
    sphere_place_candidate_cache = None
    if sphere_category:
        with _profile_stage(args, "joint_search_build_candidates") as prof:
            sphere_place_candidate_cache = _build_direct_place_candidates(
                demo,
                bridge_mod,
                scene_capture_cache,
                rule,
                place_state_cache,
                args,
                T_tcp_obj_override=None,
            )
            prof["candidate_count"] = len(sphere_place_candidate_cache or [])
            prof["success"] = bool(sphere_place_candidate_cache)
            prof["status"] = "Success" if sphere_place_candidate_cache else "NO_CANDIDATES"
        sphere_place_candidate_cache.sort(key=_pre_place_screen_sort_key)
        print(
            f"[joint_search] sphere place candidates are geometry-decoupled from grasp: "
            f"{len(sphere_place_candidate_cache)} reusable candidate(s)"
        )
    try:
        for grasp_choice in grasp_candidates:
            grasp_label = str(grasp_choice["label"])
            pregrasp_terminal_q = np.asarray(grasp_choice["q_path"][-1], dtype=np.float32).reshape(-1)[:7]
            targeted.base.sync_demo_arm_qpos(demo, pregrasp_terminal_q)
            print(
                f"[joint_search] evaluating downstream place feasibility for {grasp_label} "
                f"(grasp_score={grasp_choice['score']:.3f})"
            )
            pair_first_ik_only = bool(grasp_choice.get("pair_first_ik_only", False))
            if pair_first_ik_only:
                grasp_terminal_q = _q7_or_none(grasp_choice.get("q_grasp"))
                if grasp_terminal_q is None:
                    print(f"[joint_search] skipped {grasp_label}: IK-preselected grasp has no q_grasp")
                    continue
                print(
                    f"[joint_search] {grasp_label}: using IK q_grasp for candidate transport; "
                    "grasp final-approach MotionGen is deferred until after transport selection"
                )
            else:
                grasp_choice = _apply_deferred_two_step_final_approach(
                    planner,
                    demo,
                    args,
                    grasp_choice,
                    {grasp_label: grasp_choice},
                )
                if not bool(grasp_choice.get("two_step_grasp", False)):
                    print(f"[joint_search] skipped {grasp_label}: final approach to real grasp pose failed")
                    continue
                grasp_terminal_q = np.asarray(grasp_choice["q_path"][-1], dtype=np.float32).reshape(-1)[:7]
            targeted.base.sync_demo_arm_qpos(demo, grasp_terminal_q)
            demo._last_joint_chain_failed_start_q = grasp_terminal_q.copy()
            max_place_candidates = int(getattr(args, "joint_search_max_pre_place_candidates", 16))
            if sphere_category and max_place_candidates > 0:
                max_place_candidates = max(max_place_candidates, 12)
            per_grasp_payload_state = _snapshot_transport_payload_state(demo)
            try:
                if grasp_choice.get("T_tcp_obj") is None:
                    print(f"[joint_search] skipped {grasp_label}: missing grasp-time T_tcp_obj")
                    continue
                demo._transport_attached_T_tcp_obj = np.asarray(
                    grasp_choice.get("T_tcp_obj"),
                    dtype=np.float32,
                ).reshape(4, 4)
                targeted._register_transport_attached_box(
                    demo,
                    args,
                    show_visual=False,
                    activate_payload_visual=False,
                    T_tcp_obj_override=demo._transport_attached_T_tcp_obj,
                )
                forced = targeted.base.force_active_object_to_attached_pose(demo)
                if bool(getattr(args, "curobo_attach_object", True)):
                    attached_ok = _attach_transport_payload_to_curobo(
                        planner,
                        demo,
                        args,
                        label=f"joint_search_{grasp_label}",
                    )
                    if not attached_ok:
                        print(f"[joint_search] skipped {grasp_label}: cuRobo payload attach failed")
                        continue
                if not forced:
                    print(f"[joint_search] warning: could not force active object to attached pose for {grasp_label}")

                if sphere_category:
                    all_direct_place_candidates = [dict(item) for item in list(sphere_place_candidate_cache or [])]
                else:
                    with _profile_stage(args, "joint_search_build_candidates") as prof:
                        all_direct_place_candidates = _build_direct_place_candidates(
                            demo,
                            bridge_mod,
                            scene_capture_cache,
                            rule,
                            place_state_cache,
                            args,
                            T_tcp_obj_override=grasp_choice.get("T_tcp_obj"),
                        )
                        prof["candidate_count"] = len(all_direct_place_candidates)
                        prof["success"] = bool(all_direct_place_candidates)
                        prof["status"] = "Success" if all_direct_place_candidates else "NO_CANDIDATES"
                if rule.primitive == "insert_vertical":
                    all_direct_place_candidates = _filter_pre_place_candidates_by_verticality(all_direct_place_candidates, args)
                all_direct_place_candidates.sort(key=_pre_place_screen_sort_key)
                _record_joint_chain_failed_place_candidates(
                    demo,
                    all_direct_place_candidates,
                    grasp_label=grasp_label,
                    status="built",
                )
                if all_direct_place_candidates and demo._last_joint_chain_failed_pose is None:
                    first_failed_candidate = all_direct_place_candidates[0]
                    first_failed_pose = _place_candidate_release_pose(first_failed_candidate)
                    if first_failed_pose is None:
                        first_failed_pose = first_failed_candidate["pose"]
                    demo._last_joint_chain_failed_pose = first_failed_pose
                    demo._last_joint_chain_failed_label = str(first_failed_candidate["label"])
                if not all_direct_place_candidates:
                    continue

                fixed_tabletop_fast_gate = (
                    _is_fixed_tabletop_source(source_name)
                    and bool(getattr(args, "fast_chain_screening", False))
                )
                place_modes_all = {
                    str(item.get("place_mode", "drop_place"))
                    for item in all_direct_place_candidates
                }

                candidate_passes = []
                fast_candidates = []
                fast_records = []
                preselected_fast_candidates = [
                    dict(item) for item in list(grasp_choice.get("_preselected_fast_place_candidates") or [])
                ]
                if preselected_fast_candidates:
                    fast_records = list(preselected_fast_candidates)
                    fast_candidates = fast_records[: max(1, int(getattr(args, "fast_chain_top_pairs", 1) or 1))]
                    print(
                        f"[joint_search] {grasp_label}: using {len(fast_candidates)} "
                        "IK-preselected place candidate(s) for winner-chain fast pass"
                    )
                    candidate_passes.append(("fast_ik", fast_candidates))
                elif bool(getattr(args, "fast_chain_screening", False)):
                    with _profile_stage(
                        args,
                        "joint_search_fast_chain_ik_screen",
                        candidate_count=len(all_direct_place_candidates),
                    ) as prof:
                        fast_candidates = _fast_chain_rank_place_candidates(
                            planner,
                            demo,
                            transport_screen_args,
                            all_direct_place_candidates,
                            grasp_terminal_q,
                            label=f"{grasp_label}_fast_chain",
                            disabled_world_collision_links=direct_place_disabled_links,
                        )
                        fast_records = list(getattr(planner, "_last_fast_chain_prefilter_records", []) or [])
                        prof["candidate_count_after_ik"] = len(fast_records)
                        prof["candidate_count_motiongen"] = len(fast_candidates)
                        prof["winner_count"] = len(fast_candidates)
                        prof["success"] = bool(fast_candidates)
                        prof["status"] = "Success" if fast_candidates else "NO_IK_RANKED_PAIRS"
                    if fast_candidates:
                        candidate_passes.append(("fast_ik", fast_candidates))

                primary_candidates = _select_diverse_place_candidates(
                    all_direct_place_candidates,
                    max_place_candidates,
                    label=f"{grasp_label}_primary",
                )
                preselected_pair_fast_gate = bool(preselected_fast_candidates) and bool(fast_candidates)
                fixed_tabletop_fast_gate_blocks_primary = fixed_tabletop_fast_gate and bool(fast_candidates)
                primary_after_fast_fail = (
                    preselected_pair_fast_gate
                    and bool(getattr(args, "joint_search_primary_fallback_after_fast_ik_fail", True))
                )
                fast_gate_blocks_primary = (
                    fixed_tabletop_fast_gate_blocks_primary
                    or (preselected_pair_fast_gate and not primary_after_fast_fail)
                )
                if primary_candidates:
                    if fast_gate_blocks_primary:
                        gate_reason = (
                            "IK-preselected top pair"
                            if preselected_pair_fast_gate
                            else "fixed-tabletop fast-chain gate"
                        )
                        print(
                            f"[joint_search] {grasp_label}: {gate_reason} "
                            "will not run primary transport fallback for this grasp"
                        )
                    else:
                        if primary_after_fast_fail:
                            print(
                                f"[joint_search] {grasp_label}: IK-preselected top pair will run first; "
                                "primary transport fallback remains available if fast IK transport fails"
                            )
                        if fixed_tabletop_fast_gate:
                            print(
                                f"[joint_search] {grasp_label}: fixed-tabletop fast-chain found no "
                                "IK-ranked pair; falling back to primary transport MotionGen"
                            )
                        candidate_passes.append(("primary", primary_candidates))
                if (
                    not fast_gate_blocks_primary
                    and max_place_candidates > 0
                    and len(primary_candidates) < len(all_direct_place_candidates)
                ):
                    fallback_max = int(getattr(args, "joint_search_fallback_max_pre_place_candidates", 16))
                    fallback_cap = fallback_max if fallback_max > 0 else 0
                    if fallback_cap > len(primary_candidates):
                        expanded_candidates = _select_diverse_place_candidates(
                            all_direct_place_candidates,
                            fallback_cap,
                            label=f"{grasp_label}_fallback",
                        )
                        primary_ids = {id(item) for item in primary_candidates}
                        fallback_candidates = [item for item in expanded_candidates if id(item) not in primary_ids]
                        if fallback_candidates:
                            candidate_passes.append(("fallback", fallback_candidates))

                safe_label = re.sub(r"[^A-Za-z0-9_]+", "_", grasp_label)[:48]
                for pass_label, direct_place_candidates in candidate_passes:
                    pass_is_fallback = pass_label == "fallback"
                    pass_enable_graph = None
                    pass_timeout = screen_timeout_arg
                    pass_max_attempts = screen_max_attempts_arg
                    pass_num_ik_seeds = screen_num_ik_seeds_arg
                    pass_num_trajopt_seeds = screen_num_trajopt_seeds_arg
                    pass_num_graph_seeds = None
                    if pass_is_fallback:
                        _bump_profile_counter("fallback_count")
                        pass_enable_graph = bool(getattr(args, "joint_search_fallback_enable_graph", False))
                        fallback_timeout = float(getattr(args, "joint_search_fallback_timeout", 8.0))
                        if fallback_timeout > 0.0:
                            pass_timeout = max(float(pass_timeout or 0.0), fallback_timeout)
                        fallback_attempts = int(getattr(args, "joint_search_fallback_max_attempts", 4))
                        if fallback_attempts > 0:
                            pass_max_attempts = max(int(pass_max_attempts or 0), fallback_attempts)
                        fallback_graph_seeds = int(getattr(args, "joint_search_fallback_num_graph_seeds", 4))
                        if fallback_graph_seeds > 0:
                            pass_num_graph_seeds = max(int(getattr(args, "curobo_num_graph_seeds", 1)), fallback_graph_seeds)
                        fallback_ik_seeds = int(getattr(args, "joint_search_fallback_num_ik_seeds", 128))
                        if fallback_ik_seeds > 0:
                            pass_num_ik_seeds = max(int(pass_num_ik_seeds or 0), fallback_ik_seeds)
                        fallback_trajopt_seeds = int(getattr(args, "joint_search_fallback_num_trajopt_seeds", 4))
                        if fallback_trajopt_seeds > 0:
                            pass_num_trajopt_seeds = max(
                                int(pass_num_trajopt_seeds or 0),
                                fallback_trajopt_seeds,
                            )
                        print(
                            f"[joint_search] {grasp_label} fallback pass: "
                            f"enable_graph={pass_enable_graph}, max_attempts={pass_max_attempts}, "
                            f"num_ik_seeds={pass_num_ik_seeds}, "
                            f"num_trajopt_seeds={pass_num_trajopt_seeds}, "
                            f"num_graph_seeds={pass_num_graph_seeds}, timeout={pass_timeout}"
                        )
                    direct_pair_candidates = []
                    _record_joint_chain_failed_place_candidates(
                        demo,
                        direct_place_candidates,
                        grasp_label=grasp_label,
                        status=f"pass:{pass_label}",
                    )
                    for candidate in direct_place_candidates:
                        pair_item = dict(candidate)
                        pair_item["start_q"] = grasp_terminal_q
                        pair_item["grasp_choice"] = grasp_choice
                        direct_pair_candidates.append(pair_item)
                    if not direct_pair_candidates:
                        continue
                    needed_winners = None
                    if max_feasible_chains > 0:
                        needed_winners = max(1, max_feasible_chains - len(chains))
                    place_modes_in_pass = {
                        str(item.get("place_mode", "drop_place"))
                        for item in direct_place_candidates
                    }
                    validate_final_contact_in_joint_search = (
                        bool(getattr(args, "joint_search_validate_final_contact", False))
                        and any(mode != "drop_place" for mode in place_modes_in_pass)
                    )
                    final_contact_check_limit = None
                    transport_max_winners = needed_winners
                    if validate_final_contact_in_joint_search:
                        final_contact_check_limit = _joint_search_final_contact_check_limit(
                            args,
                            source_name,
                            place_modes_in_pass,
                        )
                        winner_floor = int(final_contact_check_limit or 6)
                        transport_max_winners = max(int(transport_max_winners or 1), winner_floor)
                    if pass_label == "fast_ik":
                        # The expensive middle transport planner should see only the
                        # IK-ranked pair set, but it must be allowed to try every
                        # preselected pair in the batch before falling back to the next
                        # grasp. Returning one winner is enough because the first
                        # successful transport chain stops the object search.
                        transport_max_winners = 1
                        if validate_final_contact_in_joint_search:
                            final_contact_check_limit = 1
                        print(
                            f"[joint_search] {grasp_label} fast_ik winner-chain mode: "
                            f"{len(direct_pair_candidates)} IK-ranked pair(s) are sent to the single "
                            "transport MotionGen batch; first successful transport wins"
                        )
                    print(
                        f"[joint_search] transport-hover pair candidate count for {grasp_label} "
                        f"({pass_label}): {len(direct_pair_candidates)}"
                        + (f", needed_winners={needed_winners}" if needed_winners is not None else "")
                        + (
                            f", transport_max_winners={transport_max_winners}"
                            if transport_max_winners is not None
                            else ""
                        )
                        + (
                            ", final_contact_precheck=yes"
                            if validate_final_contact_in_joint_search
                            else ""
                        )
                    )

                    def _accept_direct_place_successes(successes) -> bool:
                        accepted_any = False
                        ordered_successes = list(successes or [])
                        if validate_final_contact_in_joint_search:
                            final_contact_sort_key = _final_contact_validation_sort_key
                            if source_name == "bi" or any(
                                str(item.get("place_mode", "")) == "insert_place"
                                for item in ordered_successes
                            ):
                                final_contact_sort_key = _bi_final_contact_validation_sort_key
                            elif source_name in {"gluestick", "hongshupian"}:
                                final_contact_sort_key = lambda item: _vertical_long_axis_final_contact_sort_key(
                                    item,
                                    source_name,
                                )
                            if source_name == "bi" or source_name in {"gluestick", "hongshupian"} or any(
                                str(item.get("place_mode", "")) == "insert_place"
                                for item in ordered_successes
                            ):
                                ordered_successes = sorted(
                                    ordered_successes,
                                    key=final_contact_sort_key,
                                )
                            if final_contact_check_limit is not None and len(ordered_successes) > final_contact_check_limit:
                                print(
                                    "[joint_search] final-contact precheck limited to "
                                    f"{final_contact_check_limit}/{len(ordered_successes)} ordered transport winner(s); "
                                    "remaining winners deferred to fallback search"
                                )
                                ordered_successes = ordered_successes[:final_contact_check_limit]
                        for candidate in ordered_successes:
                            q_direct_place_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in candidate["q_path"]]
                            pre_transport_lift_path = [
                                np.asarray(q, dtype=np.float32).reshape(-1)[:7]
                                for q in list(candidate.get("pre_transport_lift_path") or [])
                            ]
                            if pre_transport_lift_path:
                                q_direct_place_path = pre_transport_lift_path + q_direct_place_path[1:]
                            validated_place_choice = None
                            if validate_final_contact_in_joint_search:
                                contact_candidate = dict(candidate)
                                contact_candidate["q_path"] = q_direct_place_path
                                print(
                                    f"[joint_search] validating final contact before accepting chain: "
                                    f"hover={candidate['label']}"
                                )
                                validated_place_choice = plan_final_contact_approach(
                                    planner,
                                    demo,
                                    args,
                                    grasp_terminal_q,
                                    contact_candidate,
                                    disabled_world_collision_links=direct_place_disabled_links,
                                )
                                if validated_place_choice is None:
                                    print(
                                        f"[joint_search] rejected {candidate['label']}: "
                                        "transport succeeded but final contact failed"
                                    )
                                    continue
                                q_direct_place_path = [
                                    np.asarray(q, dtype=np.float32).reshape(-1)[:7]
                                    for q in validated_place_choice.get("q_pre_place_path", q_direct_place_path)
                                ]
                            release_metrics, release_score = _path_metrics_and_score(
                                grasp_terminal_q,
                                q_direct_place_path,
                            )
                            total_score = float(grasp_choice["score"]) + float(candidate["score"]) + float(release_score)
                            chain = {
                                "grasp_choice": grasp_choice,
                                "pre_place_choice": candidate,
                                "q_pre_place_path": q_direct_place_path,
                                "q_place_path": [
                                    np.asarray(q, dtype=np.float32).reshape(-1)[:7]
                                    for q in (
                                        list(validated_place_choice.get("q_place_path") or [])
                                        if validated_place_choice is not None
                                        else [np.asarray(q_direct_place_path[-1], dtype=np.float32).reshape(-1)[:7]]
                                    )
                                ],
                                "release_metrics": release_metrics,
                                "release_score": float(release_score),
                                "total_score": total_score,
                                "direct_place_mode": True,
                                "joint_search_validated_final_contact": bool(validated_place_choice is not None),
                            }
                            print(
                                f"[joint_search] feasible transport-hover chain: grasp={grasp_label}, "
                                f"hover={candidate['label']}, total_score={total_score:.3f}, "
                                f"waypoints={len(q_direct_place_path)}"
                            )
                            chains.append(chain)
                            accepted_any = True
                            if max_feasible_chains > 0 and len(chains) >= max_feasible_chains:
                                return True
                        return accepted_any

                    direct_place_successes = []
                    accepted_chain_this_pass = False
                    skip_direct_transport_batch = False
                    skip_direct_transport_reason = "SKIPPED_INVALID_START_WORLD_COLLISION"
                    pair_first_top_pair_pass = bool(pair_first_ik_only and pass_label == "fast_ik")
                    if pair_first_top_pair_pass:
                        # Pair-first mode has already proven q_grasp/q_hover/q_release by IK.
                        # Do the mandatory short grasp lift as a constrained straight segment
                        # first, then run exactly one transport MotionGen from the lifted state.
                        skip_direct_transport_batch = True
                        skip_direct_transport_reason = "SKIPPED_PAIR_FIRST_STRAIGHT_LIFT_FIRST"
                        print(
                            f"[joint_search] {grasp_label} {pass_label}: pair-first chain will "
                            "run constrained straight lift before the single transport MotionGen"
                        )
                    if (
                        bool(getattr(args, "curobo_attach_object", True))
                        and (pass_label == "fast_ik" or (source_name == "bi" and pass_label == "primary"))
                        and not pair_first_top_pair_pass
                    ):
                        skip_direct_transport_batch = _start_state_is_world_collision(
                            planner,
                            demo,
                            transport_screen_args,
                            grasp_terminal_q,
                            label=f"joint_transport_hover_pairs_{safe_label}_{pass_label}",
                            include_table=True,
                            exclude_object_names=exclude_names,
                            disabled_world_collision_links=direct_place_disabled_links,
                        )
                    if (
                        source_name == "lvmukuai"
                        and pass_label == "primary"
                        and not skip_direct_transport_batch
                        and max(float(grasp_choice.get("grasp_z_lift_m", 0.0) or 0.0), 0.0) >= 0.005
                    ):
                        skip_direct_transport_batch = True
                        skip_direct_transport_reason = "SKIPPED_LVMUKUAI_LIFTED_GRASP_START_LIFT_FIRST"
                        print(
                            f"[joint_search] {grasp_label} {pass_label}: "
                            "skipping direct transport and using cuRobo straight-lift first "
                            "for lifted lvmukuai grasp"
                        )
                    if (
                        source_name in {"gluestick", "hongshupian"}
                        and pass_label == "primary"
                        and validate_final_contact_in_joint_search
                        and not skip_direct_transport_batch
                    ):
                        fast_lane_candidates = _vertical_long_axis_transport_fast_lane_candidates(
                            direct_pair_candidates,
                            source_name,
                            args,
                            label=f"{grasp_label}_{pass_label}",
                        )
                        if fast_lane_candidates:
                            fast_lane_successes = []
                            fast_lane_max_winners = max(
                                int(transport_max_winners or 1),
                                int(final_contact_check_limit or 4),
                            )
                            with _profile_stage(
                                args,
                                "joint_search_transport_hover",
                                candidate_count=len(fast_lane_candidates),
                                status="vertical_long_axis_fast_lane",
                                max_attempts=int(getattr(args, "curobo_max_attempts", 2) if pass_max_attempts is None else pass_max_attempts),
                                num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64) if pass_num_ik_seeds is None else pass_num_ik_seeds),
                                num_trajopt_seeds=int(
                                    getattr(args, "curobo_num_trajopt_seeds", 1)
                                    if pass_num_trajopt_seeds is None
                                    else pass_num_trajopt_seeds
                                ),
                                enable_graph=bool(getattr(args, "curobo_enable_graph", False) if pass_enable_graph is None else pass_enable_graph),
                            ) as prof:
                                fast_lane_successes = _evaluate_curobo_pose_candidates_multi_start(
                                    planner,
                                    demo,
                                    transport_screen_args,
                                    fast_lane_candidates,
                                    label=f"joint_transport_hover_pairs_{safe_label}_{pass_label}_vertical_fast_lane",
                                    use_attach=True,
                                    timeout=pass_timeout,
                                    max_attempts=pass_max_attempts,
                                    num_ik_seeds=pass_num_ik_seeds,
                                    num_trajopt_seeds=pass_num_trajopt_seeds,
                                    num_graph_seeds=pass_num_graph_seeds,
                                    enable_graph=pass_enable_graph,
                                    max_winners=fast_lane_max_winners,
                                    include_table=True,
                                    exclude_object_names=exclude_names,
                                    disabled_world_collision_links=direct_place_disabled_links,
                                )
                                _copy_last_candidate_counts_to_profile(prof, planner)
                                prof["winner_count"] = len(fast_lane_successes)
                                prof["success"] = bool(fast_lane_successes)
                                prof["status"] = "Success" if fast_lane_successes else "NO_WINNERS_VERTICAL_FAST_LANE"
                                prof["world_changed"] = bool(getattr(planner, "_last_world_changed", False))
                                prof["cache_hit"] = bool(getattr(planner, "_last_world_cache_hit", False))
                                if fast_lane_successes:
                                    prof["path_waypoints"] = int((fast_lane_successes[0].get("metrics") or {}).get("waypoint_count", 0) or 0)
                                    prof["path_score"] = float(fast_lane_successes[0].get("score", 0.0) or 0.0)
                            if fast_lane_successes:
                                accepted_chain_this_pass = _accept_direct_place_successes(fast_lane_successes)
                                if accepted_chain_this_pass:
                                    print(
                                        f"[joint_search] {grasp_label} {pass_label}: "
                                        f"accepted {source_name} vertical-long-axis fast-lane transport/final-contact chain"
                                    )
                    if accepted_chain_this_pass:
                        if max_feasible_chains > 0 and len(chains) >= max_feasible_chains:
                            break
                        continue
                    if pair_first_top_pair_pass:
                        print(
                            f"[joint_search] {grasp_label} {pass_label}: skipped contact-state "
                            "transport profile entry; the only transport MotionGen will run after lift"
                        )
                    else:
                        with _profile_stage(
                            args,
                            "joint_search_transport_hover",
                            candidate_count=len(direct_pair_candidates),
                            max_attempts=int(getattr(args, "curobo_max_attempts", 2) if pass_max_attempts is None else pass_max_attempts),
                            num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64) if pass_num_ik_seeds is None else pass_num_ik_seeds),
                            num_trajopt_seeds=int(
                                getattr(args, "curobo_num_trajopt_seeds", 1)
                                if pass_num_trajopt_seeds is None
                                else pass_num_trajopt_seeds
                            ),
                            enable_graph=bool(getattr(args, "curobo_enable_graph", False) if pass_enable_graph is None else pass_enable_graph),
                            skipped_invalid_start=bool(skip_direct_transport_batch),
                        ) as prof:
                            if skip_direct_transport_batch:
                                prof["winner_count"] = 0
                                prof["success"] = False
                                prof["status"] = skip_direct_transport_reason
                                prof["world_changed"] = bool(getattr(planner, "_last_world_changed", False))
                                prof["cache_hit"] = bool(getattr(planner, "_last_world_cache_hit", False))
                            else:
                                direct_place_successes = _evaluate_curobo_pose_candidates_multi_start(
                                    planner,
                                    demo,
                                    transport_screen_args,
                                    direct_pair_candidates,
                                    label=f"joint_transport_hover_pairs_{safe_label}_{pass_label}",
                                    use_attach=True,
                                    timeout=pass_timeout,
                                    max_attempts=pass_max_attempts,
                                    num_ik_seeds=pass_num_ik_seeds,
                                    num_trajopt_seeds=pass_num_trajopt_seeds,
                                    num_graph_seeds=pass_num_graph_seeds,
                                    enable_graph=pass_enable_graph,
                                    max_winners=transport_max_winners,
                                    include_table=True,
                                    exclude_object_names=exclude_names,
                                    disabled_world_collision_links=direct_place_disabled_links,
                                )
                                _copy_last_candidate_counts_to_profile(prof, planner)
                                prof["winner_count"] = len(direct_place_successes)
                                prof["success"] = bool(direct_place_successes)
                                prof["status"] = "Success" if direct_place_successes else "NO_WINNERS"
                                prof["world_changed"] = bool(getattr(planner, "_last_world_changed", False))
                                prof["cache_hit"] = bool(getattr(planner, "_last_world_cache_hit", False))
                                if direct_place_successes:
                                    prof["path_waypoints"] = int((direct_place_successes[0].get("metrics") or {}).get("waypoint_count", 0) or 0)
                                    prof["path_score"] = float(direct_place_successes[0].get("score", 0.0) or 0.0)
                    if direct_place_successes and validate_final_contact_in_joint_search:
                        accepted_chain_this_pass = (
                            _accept_direct_place_successes(direct_place_successes)
                            or accepted_chain_this_pass
                        )
                        direct_place_successes = []
                    if not direct_place_successes and not accepted_chain_this_pass:
                        lift_m = float(max(getattr(args, "joint_search_start_collision_lift_m", 0.030), 0.0))
                        lift_exclude_names = set(exclude_names or set())
                        lift_disabled_links = _post_grasp_lift_disabled_world_links(
                            planner,
                            direct_place_disabled_links,
                        )
                        relief_name = _single_obstacle_start_collision_relief(
                            planner,
                            grasp_terminal_q,
                            already_excluded=lift_exclude_names,
                        )
                        if relief_name:
                            lift_exclude_names.add(relief_name)
                            print(
                                f"[joint_search] {grasp_label} {pass_label}: "
                                f"straight-lift fallback will temporarily exclude start-collision obstacle {relief_name!r}"
                            )
                        lift_path_kind = "TCP-up straight lift"
                        with _profile_stage(args, "joint_search_start_lift", candidate_count=1) as prof:
                            lift_path = _plan_short_tcp_up_axis_lift_ik(
                                planner,
                                demo,
                                args,
                                grasp_terminal_q,
                                grasp_choice,
                                lift_m=lift_m,
                                label=f"joint_start_tcp_up_lift_{safe_label}_{pass_label}",
                                include_table=True,
                                exclude_object_names=lift_exclude_names,
                                disabled_world_collision_links=lift_disabled_links,
                            )
                            if lift_path is None:
                                lift_path_kind = "approach-line retreat"
                                lift_path = _plan_short_grasp_approach_retreat_ik(
                                    planner,
                                    demo,
                                    args,
                                    grasp_terminal_q,
                                    grasp_choice,
                                    label=f"joint_start_retreat_{safe_label}_{pass_label}",
                                    include_table=True,
                                    exclude_object_names=lift_exclude_names,
                                    disabled_world_collision_links=lift_disabled_links,
                                )
                            if lift_path is None:
                                lift_path_kind = f"{lift_m:.3f}m straight world-Z lift"
                                lift_path = _plan_short_world_z_lift_ik(
                                    planner,
                                    demo,
                                    args,
                                    grasp_terminal_q,
                                    lift_m=lift_m,
                                    label=f"joint_start_lift_{safe_label}_{pass_label}",
                                    include_table=True,
                                    exclude_object_names=lift_exclude_names,
                                    disabled_world_collision_links=lift_disabled_links,
                                )
                            prof["success"] = lift_path is not None
                            prof["status"] = "Success" if lift_path is not None else "PLAN_FAIL"
                            prof["lift_kind"] = lift_path_kind
                            prof["path_waypoints"] = len(lift_path or [])
                        if lift_path is None and skip_direct_transport_batch:
                            if pair_first_top_pair_pass:
                                print(
                                    f"[joint_search] {grasp_label} {pass_label}: pair-first straight lift failed; "
                                    "not running skipped direct transport fallback"
                                )
                            else:
                                print(
                                    f"[joint_search] {grasp_label} {pass_label}: straight lift failed after "
                                    "start-collision skip; running the skipped direct transport batch as fallback"
                                )
                                _bump_profile_counter("fallback_count")
                                with _profile_stage(
                                    args,
                                    "joint_search_transport_hover",
                                    candidate_count=len(direct_pair_candidates),
                                    status="fallback_after_lift_fail",
                                    max_attempts=int(getattr(args, "curobo_max_attempts", 2) if pass_max_attempts is None else pass_max_attempts),
                                    num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64) if pass_num_ik_seeds is None else pass_num_ik_seeds),
                                    num_trajopt_seeds=int(
                                        getattr(args, "curobo_num_trajopt_seeds", 1)
                                        if pass_num_trajopt_seeds is None
                                        else pass_num_trajopt_seeds
                                    ),
                                    enable_graph=bool(getattr(args, "curobo_enable_graph", False) if pass_enable_graph is None else pass_enable_graph),
                                ) as prof:
                                    direct_place_successes = _evaluate_curobo_pose_candidates_multi_start(
                                        planner,
                                        demo,
                                        transport_screen_args,
                                        direct_pair_candidates,
                                        label=f"joint_transport_hover_pairs_{safe_label}_{pass_label}_fallback_after_lift_fail",
                                        use_attach=True,
                                        timeout=pass_timeout,
                                        max_attempts=pass_max_attempts,
                                        num_ik_seeds=pass_num_ik_seeds,
                                        num_trajopt_seeds=pass_num_trajopt_seeds,
                                        num_graph_seeds=pass_num_graph_seeds,
                                        enable_graph=pass_enable_graph,
                                        max_winners=transport_max_winners,
                                        include_table=True,
                                        exclude_object_names=exclude_names,
                                        disabled_world_collision_links=direct_place_disabled_links,
                                    )
                                    _copy_last_candidate_counts_to_profile(prof, planner)
                                    prof["winner_count"] = len(direct_place_successes)
                                    prof["success"] = bool(direct_place_successes)
                                    prof["status"] = "Success" if direct_place_successes else "NO_WINNERS_FALLBACK_AFTER_LIFT_FAIL"
                                    prof["world_changed"] = bool(getattr(planner, "_last_world_changed", False))
                                    prof["cache_hit"] = bool(getattr(planner, "_last_world_cache_hit", False))
                                    if direct_place_successes:
                                        prof["path_waypoints"] = int((direct_place_successes[0].get("metrics") or {}).get("waypoint_count", 0) or 0)
                                        prof["path_score"] = float(direct_place_successes[0].get("score", 0.0) or 0.0)
                                if not direct_place_successes:
                                    with _profile_stage(
                                        args,
                                        "joint_search_start_lift",
                                        candidate_count=1,
                                        status="retry_after_direct_fallback",
                                    ) as prof:
                                        lift_path_kind = "TCP-up straight lift"
                                        lift_path = _plan_short_tcp_up_axis_lift_ik(
                                            planner,
                                            demo,
                                            args,
                                            grasp_terminal_q,
                                            grasp_choice,
                                            lift_m=lift_m,
                                            label=f"joint_start_tcp_up_lift_{safe_label}_{pass_label}_retry_after_direct_fallback",
                                            include_table=True,
                                            exclude_object_names=lift_exclude_names,
                                            disabled_world_collision_links=lift_disabled_links,
                                        )
                                        if lift_path is None:
                                            lift_path_kind = "approach-line retreat"
                                            lift_path = _plan_short_grasp_approach_retreat_ik(
                                                planner,
                                                demo,
                                                args,
                                                grasp_terminal_q,
                                                grasp_choice,
                                                label=f"joint_start_retreat_{safe_label}_{pass_label}_retry_after_direct_fallback",
                                                include_table=True,
                                                exclude_object_names=lift_exclude_names,
                                                disabled_world_collision_links=lift_disabled_links,
                                            )
                                        if lift_path is None:
                                            lift_path_kind = f"{lift_m:.3f}m straight world-Z lift"
                                            lift_path = _plan_short_world_z_lift_ik(
                                                planner,
                                                demo,
                                                args,
                                                grasp_terminal_q,
                                                lift_m=lift_m,
                                                label=f"joint_start_lift_{safe_label}_{pass_label}_retry_after_direct_fallback",
                                                include_table=True,
                                                exclude_object_names=lift_exclude_names,
                                                disabled_world_collision_links=lift_disabled_links,
                                            )
                                        prof["success"] = lift_path is not None
                                        prof["status"] = "Success" if lift_path is not None else "PLAN_FAIL_RETRY_AFTER_DIRECT_FALLBACK"
                                        prof["path_waypoints"] = len(lift_path or [])
                        if lift_path is not None and len(lift_path) >= 2:
                            lifted_start_q = np.asarray(lift_path[-1], dtype=np.float32).reshape(-1)[:7]
                            lifted_pair_candidates = []
                            for candidate in direct_place_candidates:
                                pair_item = dict(candidate)
                                pair_item["start_q"] = lifted_start_q
                                pair_item["grasp_choice"] = grasp_choice
                                pair_item["pre_transport_lift_path"] = [
                                    np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in lift_path
                                ]
                                lifted_pair_candidates.append(pair_item)
                            print(
                                f"[joint_search] retrying transport-hover after {lift_path_kind} "
                                f"for {grasp_label} ({pass_label})"
                            )
                            lifted_candidate_passes = [("after_lift", lifted_pair_candidates)]
                            if source_name == "bi" and pass_label == "primary":
                                fast_lane = _bi_insert_fast_lane_candidates(
                                    lifted_pair_candidates,
                                    args,
                                    label=f"{grasp_label}_{pass_label}_after_lift",
                                )
                                if fast_lane and len(fast_lane) < len(lifted_pair_candidates):
                                    fast_ids = {id(item) for item in fast_lane}
                                    remaining_candidates = [
                                        item for item in lifted_pair_candidates if id(item) not in fast_ids
                                    ]
                                    lifted_candidate_passes = [("after_lift_fast_lane", fast_lane)]
                                    if remaining_candidates:
                                        lifted_candidate_passes.append(
                                            ("after_lift_remaining", remaining_candidates)
                                        )
                            for lifted_pass_label, lifted_eval_candidates in lifted_candidate_passes:
                                lifted_max_winners = transport_max_winners
                                if validate_final_contact_in_joint_search:
                                    # Only collect as many transport winners as we are willing to
                                    # validate with expensive sequential final-contact MotionGen.
                                    lifted_max_winners = max(
                                        int(lifted_max_winners or 1),
                                        int(final_contact_check_limit or 6),
                                    )
                                with _profile_stage(
                                    args,
                                    "joint_search_transport_hover",
                                    candidate_count=len(lifted_eval_candidates),
                                    status=lifted_pass_label,
                                    max_attempts=int(getattr(args, "curobo_max_attempts", 2) if pass_max_attempts is None else pass_max_attempts),
                                    num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64) if pass_num_ik_seeds is None else pass_num_ik_seeds),
                                    num_trajopt_seeds=int(
                                        getattr(args, "curobo_num_trajopt_seeds", 1)
                                        if pass_num_trajopt_seeds is None
                                        else pass_num_trajopt_seeds
                                    ),
                                    enable_graph=bool(getattr(args, "curobo_enable_graph", False) if pass_enable_graph is None else pass_enable_graph),
                                ) as prof:
                                    direct_place_successes = _evaluate_curobo_pose_candidates_multi_start(
                                        planner,
                                        demo,
                                        transport_screen_args,
                                        lifted_eval_candidates,
                                        label=f"joint_transport_hover_pairs_{safe_label}_{pass_label}_{lifted_pass_label}",
                                        use_attach=True,
                                        timeout=pass_timeout,
                                        max_attempts=pass_max_attempts,
                                        num_ik_seeds=pass_num_ik_seeds,
                                        num_trajopt_seeds=pass_num_trajopt_seeds,
                                        num_graph_seeds=pass_num_graph_seeds,
                                        enable_graph=pass_enable_graph,
                                        max_winners=lifted_max_winners,
                                        include_table=True,
                                        exclude_object_names=lift_exclude_names,
                                        disabled_world_collision_links=direct_place_disabled_links,
                                    )
                                    _copy_last_candidate_counts_to_profile(prof, planner)
                                    prof["winner_count"] = len(direct_place_successes)
                                    prof["success"] = bool(direct_place_successes)
                                    prof["status"] = "Success" if direct_place_successes else f"NO_WINNERS_{lifted_pass_label.upper()}"
                                    prof["world_changed"] = bool(getattr(planner, "_last_world_changed", False))
                                    prof["cache_hit"] = bool(getattr(planner, "_last_world_cache_hit", False))
                                    if direct_place_successes:
                                        prof["path_waypoints"] = int((direct_place_successes[0].get("metrics") or {}).get("waypoint_count", 0) or 0)
                                        prof["path_score"] = float(direct_place_successes[0].get("score", 0.0) or 0.0)
                                if direct_place_successes:
                                    if validate_final_contact_in_joint_search:
                                        accepted_chain_this_pass = (
                                            _accept_direct_place_successes(direct_place_successes)
                                            or accepted_chain_this_pass
                                        )
                                        direct_place_successes = []
                                        if accepted_chain_this_pass or (
                                            max_feasible_chains > 0 and len(chains) >= max_feasible_chains
                                        ):
                                            break
                                        continue
                                    break
                    if direct_place_successes:
                        accepted_chain_this_pass = (
                            _accept_direct_place_successes(direct_place_successes)
                            or accepted_chain_this_pass
                        )
                    if max_feasible_chains > 0 and len(chains) >= max_feasible_chains:
                        break
                if max_feasible_chains > 0 and len(chains) >= max_feasible_chains:
                    break
            finally:
                if getattr(planner, "attached_object_active", False):
                    planner.detach_object_from_robot()
                _restore_transport_payload_state(demo, per_grasp_payload_state)
                if saved_obj_pose is not None:
                    _set_active_object_pose_quiet(demo, saved_obj_pose[0], saved_obj_pose[1])
        if not chains:
            print("[joint_search] no transport-hover chain stayed feasible")
    finally:
        if getattr(planner, "attached_object_active", False):
            planner.detach_object_from_robot()
        _restore_transport_payload_state(demo, saved_payload_state)
        if saved_obj_pose is not None:
            _set_active_object_pose_quiet(demo, saved_obj_pose[0], saved_obj_pose[1])
        targeted.base.sync_demo_arm_qpos(demo, saved_q)

    if not chains:
        return []

    demo._last_joint_chain_failed_pose = None
    demo._last_joint_chain_failed_label = None
    demo._last_joint_chain_failed_candidate_poses = []
    demo._last_joint_chain_failed_start_q = None
    chains.sort(key=_joint_chain_sort_key)
    best = chains[0]
    print(
        f"[joint_search] selected chain: grasp={best['grasp_choice']['label']}, "
        f"pre_place={best['pre_place_choice']['label']}, total_score={best['total_score']:.3f}"
    )
    return chains


def run_targeted_place_episode_curobo_direct(
    demo,
    bridge_mod,
    real_exec,
    args,
    scene_capture_cache,
    place_state_cache,
) -> bool:
    args._skip_remaining_step_confirms_in_object = False

    rule = None
    selected_joint_chain = None
    two_step_pregrasp_lookup = {}
    if not args.skip_goal_motion:
        rule = targeted.get_place_rule(args.object_name)
        if rule is None:
            print(f"[FAIL] no targeted-place rule is configured for source object {args.object_name}")
            _hide_actor_quiet(getattr(demo, "_target_object_goal_visual_actor", None))
            return False
        _render_current_target_object_goal_visual(
            demo,
            bridge_mod,
            scene_capture_cache,
            place_state_cache,
            rule,
            args,
        )
        if getattr(args, "render_mode", None) == "human":
            try:
                bridge_mod.render_preview(demo.env, repeats=1)
            except Exception:
                pass
    else:
        _hide_actor_quiet(getattr(demo, "_target_object_goal_visual_actor", None))

    print("\n[episode] planning from FoundationPose-initialized object pose")

    start_q = demo.current_arm_qpos()
    if real_exec is not None and bool(getattr(args, "single_confirm_per_object", False)):
        if not targeted.base.begin_single_confirm_window_for_object(demo, bridge_mod, args):
            print("[abort] user cancelled before executing this object's pick-place sequence")
            return False
    if real_exec is not None:
        print("\n[real robot setup]")
        if args.render_mode == "human":
            bridge_mod.render_preview(demo.env, repeats=5)
        ok, q_sent = targeted.base.align_real_robot_to_sim_start(demo, bridge_mod, real_exec, start_q, args)
        if not ok:
            print("[abort] failed to align the real robot to the simulation start pose")
            return False
        targeted.base.sync_demo_arm_qpos(demo, q_sent if q_sent is not None else start_q)
    else:
        print("\n[dry-run] --execute-real was not provided, so motions will only be planned and previewed")

    planner = _get_or_create_curobo_planner_serialized(args)
    if getattr(planner, "attached_object_active", False):
        print("[curobo] clearing stale attached payload before grasp planning")
        planner.detach_object_from_robot()
    demo._attached_box_visual_visible = False
    demo._attached_object_visual_active = False
    targeted.base.update_attached_box_visual(demo, visible=False)
    _clear_visualized_attached_spheres(demo)

    targeted.base.lift_active_object_above_table_if_needed(
        demo,
        args,
        min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
    )

    print("\n[move to grasp with two-step approach]")
    with _profile_stage(args, "build_grasp_candidates") as prof:
        grasp_candidates = _build_direct_grasp_candidates(
            demo,
            args,
            bridge_mod=bridge_mod,
            scene_capture_cache=scene_capture_cache,
            place_state_cache=place_state_cache,
        )
        prof["candidate_count"] = len(grasp_candidates)
        prof["success"] = bool(grasp_candidates)
        prof["status"] = "Success" if grasp_candidates else "NO_CANDIDATES"
    print("\n[poses]")
    print("object p:", np.round(demo.get_obj_pose()[0], 6), "object q:", np.round(demo.get_obj_pose()[1], 6))
    direct_grasp_disabled_links: list[str] = []
    direct_grasp_max_winners = int(getattr(args, "direct_grasp_goalset_max_winners", 2))
    requested_direct_grasp_max_winners = int(direct_grasp_max_winners)
    source_name = _current_source_object_name(args)
    if (
        rule is not None
        and bool(getattr(rule, "preserve_long_axis_vertical", False))
        and direct_grasp_max_winners > 0
        and direct_grasp_max_winners < 5
    ):
        print(
            "[direct_grasp] vertical long-axis placement: increasing pregrasp winners "
            f"{direct_grasp_max_winners}->5 so downstream final-approach can test deeper axis biases"
        )
        direct_grasp_max_winners = 5
    bi_min_pregrasp_winners = max(1, min(4, len(_bi_direct_grasp_approach_roll_degs(args))))
    if source_name == "bi" and direct_grasp_max_winners > 0 and direct_grasp_max_winners < bi_min_pregrasp_winners:
        print(
            "[direct_grasp] bi insert: increasing pregrasp winners "
            f"{direct_grasp_max_winners}->{bi_min_pregrasp_winners} to test equivalent wrist-roll branches"
        )
        direct_grasp_max_winners = bi_min_pregrasp_winners
    fixed_tabletop_min_pregrasp_winners = max(1, len(_fixed_tabletop_release_tilt_degs(args, source_name)))
    if (
        _is_fixed_tabletop_source(source_name)
        and direct_grasp_max_winners > 0
        and direct_grasp_max_winners < fixed_tabletop_min_pregrasp_winners
    ):
        print(
            "[direct_grasp] fixed-tabletop: increasing pregrasp winners "
            f"{direct_grasp_max_winners}->{fixed_tabletop_min_pregrasp_winners} to test fixed-pose vertical-gripper relations"
        )
        direct_grasp_max_winners = fixed_tabletop_min_pregrasp_winners
    expanded_direct_grasp_max_winners = int(direct_grasp_max_winners)
    if bool(getattr(args, "fast_chain_screening", False)):
        initial_fast_winners = int(getattr(args, "fast_chain_initial_grasp_winners", 1) or 0)
        if initial_fast_winners > 0 and direct_grasp_max_winners > initial_fast_winners:
            print(
                "[direct_grasp] integrated fast-chain: first planning only "
                f"{initial_fast_winners}/{expanded_direct_grasp_max_winners} pregrasp winner(s); "
                "pair-first IK preselect will choose the actual top pair"
            )
            direct_grasp_max_winners = initial_fast_winners
    grasp_start_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    all_grasp_candidates_for_fallback = list(grasp_candidates or [])
    initial_grasp_candidates = list(grasp_candidates or [])
    prefetched_plan = None
    prefetch_manager = getattr(args, "_next_cycle_prefetch_manager", None)
    if prefetch_manager is not None:
        try:
            prefetched_plan = prefetch_manager.consume_for_current_cycle(
                args,
                getattr(args, "object_name", None),
                grasp_start_q,
                scene_capture_cache,
            )
        except Exception as exc:
            print(f"[prefetch] failed to consume cached plan for {args.object_name}: {exc}")
            prefetched_plan = None
    preselected_grasp = None
    two_step_pregrasp_successes = []
    if isinstance(prefetched_plan, dict):
        prefetched_chain = prefetched_plan.get("selected_joint_chain")
        prefetched_grasp = prefetched_plan.get("grasp_choice")
        if isinstance(prefetched_chain, dict) and isinstance(prefetched_grasp, dict):
            selected_joint_chain = dict(prefetched_chain)
            selected_joint_chain["grasp_choice"] = dict(prefetched_grasp)
            two_step_pregrasp_successes = [dict(prefetched_grasp)]
            direct_grasp_max_winners = 1
            _record_prefetch_profile(
                args,
                getattr(args, "object_name", None),
                "next_cycle_prefetch_apply",
                success=True,
                status="APPLIED",
                selected_grasp_label=str(prefetched_grasp.get("label", "")),
                selected_place_label=str((selected_joint_chain.get("place_choice") or {}).get("label", "")),
            )
            print(
                f"[prefetch] applying cached grasp/place plan for {args.object_name}; "
                "winner_chain IK and transport MotionGen will be skipped if validation stays within tolerance"
            )
        else:
            _record_prefetch_profile(
                args,
                getattr(args, "object_name", None),
                "next_cycle_prefetch_apply",
                success=False,
                status="INVALID_PLAN_SHAPE",
                has_selected_joint_chain=isinstance(prefetched_chain, dict),
                has_grasp_choice=isinstance(prefetched_grasp, dict),
            )
            prefetched_plan = None
    if prefetched_plan is None:
        preselected_grasp = _fast_chain_preselect_grasp_place_pair(
            planner,
            demo,
            bridge_mod,
            args,
            scene_capture_cache,
            place_state_cache,
            rule,
            initial_grasp_candidates,
            grasp_start_q,
            disabled_world_collision_links=direct_grasp_disabled_links,
        )
    if (
        rule is not None
        and bool(getattr(args, "fast_chain_screening", False))
        and preselected_grasp is None
        and prefetched_plan is None
    ):
        print(
            "[FAIL] pair-first IK preselect found no complete grasp/place pair; "
            "legacy candidate-stage MotionGen fallback is disabled"
        )
        selected_pose = (
            grasp_candidates[0].get("pregrasp_pose", grasp_candidates[0]["pose"])
            if grasp_candidates
            else demo.build_topdown_grasp_pose()
        )
        targeted.base.inspect_failed_pose(
            demo,
            bridge_mod,
            "pair_first_ik_preselect",
            args,
            pose=selected_pose,
            gripper_closed=False,
            candidate_poses=[
                item.get("pregrasp_pose", item.get("pose"))
                for item in list(grasp_candidates or [])
                if item.get("pregrasp_pose", item.get("pose")) is not None
            ],
        )
        return False
    if preselected_grasp is not None:
        selected_label = str(preselected_grasp.get("label", ""))
        preselected_pair_grasps = [
            dict(item)
            for item in list(preselected_grasp.get("_winner_chain_top_pair_grasps") or [preselected_grasp])
        ]
        initial_grasp_candidates = preselected_pair_grasps
        direct_grasp_max_winners = 1
        print(
            "[winner_chain] transport MotionGen restricted to IK-preselected top pair(s) "
            f"first={selected_label!r}, count={len(preselected_pair_grasps)}; "
            "candidate-stage grasp MotionGen is skipped"
        )
    if preselected_grasp is not None:
        preselected_pair_grasps = [
            dict(item)
            for item in list(preselected_grasp.get("_winner_chain_top_pair_grasps") or [preselected_grasp])
        ]
        for pair_grasp in preselected_pair_grasps:
            preselected_success = _make_ik_preselected_grasp_success(args, grasp_start_q, pair_grasp)
            if preselected_success is not None:
                two_step_pregrasp_successes.append(preselected_success)
        if two_step_pregrasp_successes:
            print(
                f"[grasp] using {len(two_step_pregrasp_successes)} IK-preselected pregrasp/grasp pair(s); "
                "final approach will be planned once for the selected transport pair"
            )
        else:
            print("[FAIL] IK-preselected grasp list missing q_pregrasp/q_grasp; legacy pregrasp planner is disabled")
            return False
    if not two_step_pregrasp_successes:
        two_step_pregrasp_successes = _evaluate_two_step_grasp_candidates(
            planner,
            demo,
            args,
            grasp_start_q,
            initial_grasp_candidates,
            label="two_step_grasp",
            max_winners=direct_grasp_max_winners,
            include_active_object=True,
            disabled_world_collision_links=direct_grasp_disabled_links,
        )
    if two_step_pregrasp_successes:
        two_step_pregrasp_lookup = {
            str(item.get("label", "")): item for item in list(two_step_pregrasp_successes or [])
        }
        print(f"[grasp] explicit two-step grasp pregrasp succeeded for {len(two_step_pregrasp_lookup)} candidate(s)")
    elif preselected_grasp is not None and len(all_grasp_candidates_for_fallback) > 1:
        fallback_winners = min(
            max(2, expanded_direct_grasp_max_winners, requested_direct_grasp_max_winners),
            len(all_grasp_candidates_for_fallback),
        )
        print(
            "[winner_chain] IK-preselected grasp failed pregrasp MotionGen; "
            f"falling back to expanded grasp goalset winners={fallback_winners}"
        )
        targeted.base.sync_demo_arm_qpos(demo, grasp_start_q)
        two_step_pregrasp_successes = _evaluate_two_step_grasp_candidates(
            planner,
            demo,
            args,
            grasp_start_q,
            all_grasp_candidates_for_fallback,
            label="two_step_grasp_preselect_fallback",
            max_winners=fallback_winners,
            include_active_object=True,
            disabled_world_collision_links=direct_grasp_disabled_links,
        )
        two_step_pregrasp_lookup = {
            str(item.get("label", "")): item for item in list(two_step_pregrasp_successes or [])
        }
        if two_step_pregrasp_successes:
            print(f"[grasp] fallback pregrasp succeeded for {len(two_step_pregrasp_lookup)} candidate(s)")
    else:
        print("[FAIL] two-step grasp pregrasp planning failed")
        selected_pose = all_grasp_candidates_for_fallback[0].get("pregrasp_pose", all_grasp_candidates_for_fallback[0]["pose"]) if all_grasp_candidates_for_fallback else demo.build_topdown_grasp_pose()
        targeted.base.inspect_failed_pose(
            demo,
            bridge_mod,
            "grasp_pregrasp",
            args,
            pose=selected_pose,
            gripper_closed=False,
            candidate_poses=[item.get("pregrasp_pose", item["pose"]) for item in all_grasp_candidates_for_fallback],
        )
        return False

    grasp_successes = list(two_step_pregrasp_successes or [])
    grasp_successes.sort(key=_candidate_sort_key)
    if not grasp_successes:
        selected_pose = grasp_candidates[0]["pose"] if grasp_candidates else demo.build_topdown_grasp_pose()
        print("[FAIL] two-step grasp pregrasp produced no selectable candidates")
        targeted.base.inspect_failed_pose(
            demo,
            bridge_mod,
            "grasp",
            args,
            pose=selected_pose,
            gripper_closed=False,
            candidate_poses=[item["pose"] for item in grasp_candidates],
        )
        return False

    grasp_choice = grasp_successes[0]
    if rule is not None and _skip_joint_search_for_current_object(args):
        print(
            f"[joint_search] skipped for {args.object_name}; "
            "will plan targeted place after executed grasp and post-grasp lift"
        )
    elif rule is not None and selected_joint_chain is not None:
        selected_joint_chain["grasp_choice"] = grasp_choice
        print(
            f"[prefetch] reusing cached joint_search chain for {args.object_name}; "
            "live transport MotionGen is skipped unless later start-state validation rejects reuse"
        )
    elif rule is not None:
        joint_chains = _evaluate_joint_grasp_place_chains(
            planner,
            demo,
            bridge_mod,
            args,
            scene_capture_cache,
            place_state_cache,
            rule,
            grasp_successes,
        )
        pair_first_chain_attempt = any(bool(item.get("pair_first_ik_only", False)) for item in grasp_successes)
        if not joint_chains:
            fallback_grasp_successes = []
            if pair_first_chain_attempt:
                print(
                    "[joint_search] pair-first top pair produced no complete chain; "
                    "legacy expanded-grasp MotionGen fallback is disabled"
                )
            elif direct_grasp_max_winners == 1 and len(grasp_candidates) > 1:
                fallback_winners = min(
                    max(2, expanded_direct_grasp_max_winners, requested_direct_grasp_max_winners),
                    len(grasp_candidates),
                )
                print(
                    "[joint_search] first grasp winner produced no complete chain; "
                    f"expanding grasp goalset winners 1->{fallback_winners} before failing"
                )
                targeted.base.sync_demo_arm_qpos(demo, grasp_start_q)
                fallback_grasp_successes = _evaluate_two_step_grasp_candidates(
                    planner,
                    demo,
                    args,
                    grasp_start_q,
                    all_grasp_candidates_for_fallback,
                    label="two_step_grasp_fallback",
                    max_winners=fallback_winners,
                    include_active_object=True,
                    disabled_world_collision_links=direct_grasp_disabled_links,
                )
                merged_by_label = {str(item.get("label", "")): item for item in grasp_successes}
                for item in list(fallback_grasp_successes or []):
                    merged_by_label[str(item.get("label", ""))] = item
                if len(merged_by_label) > len(grasp_successes):
                    grasp_successes = list(merged_by_label.values())
                    grasp_successes.sort(key=_candidate_sort_key)
                    two_step_pregrasp_lookup.update(
                        {str(item.get("label", "")): item for item in list(fallback_grasp_successes or [])}
                    )
                    joint_chains = _evaluate_joint_grasp_place_chains(
                        planner,
                        demo,
                        bridge_mod,
                        args,
                        scene_capture_cache,
                        place_state_cache,
                        rule,
                        grasp_successes,
                    )
            if (
                not joint_chains
                and not pair_first_chain_attempt
                and source_name == "bi"
                and rule is not None
                and getattr(rule, "primitive", None) == "insert_vertical"
            ):
                base_offsets = _unique_finite_float_list(
                    getattr(args, "bi_insert_release_height_offsets_m", [0.0]),
                    min_value=0.0,
                )
                fallback_offsets = _unique_finite_float_list(
                    getattr(args, "bi_insert_release_fallback_height_offsets_m", [0.01]),
                    min_value=0.0,
                )
                fallback_offsets = [
                    float(v)
                    for v in fallback_offsets
                    if float(v) > 1e-6 and all(abs(float(v) - float(base)) > 1e-6 for base in base_offsets)
                ]
                if fallback_offsets:
                    print(
                        "[joint_search] bi insert: original release depth failed; "
                        "retrying with shallower release offset(s) "
                        f"{[round(v, 4) for v in fallback_offsets]}"
                    )
                    fallback_args = SimpleNamespace(**vars(args))
                    fallback_args.bi_insert_release_height_offsets_m = fallback_offsets
                    targeted.base.sync_demo_arm_qpos(demo, grasp_start_q)
                    with _profile_stage(
                        args,
                        "joint_search_bi_release_up_fallback",
                        candidate_count=len(fallback_offsets),
                    ) as prof:
                        joint_chains = _evaluate_joint_grasp_place_chains(
                            planner,
                            demo,
                            bridge_mod,
                            fallback_args,
                            scene_capture_cache,
                            place_state_cache,
                            rule,
                            grasp_successes,
                        )
                        prof["success"] = bool(joint_chains)
                        prof["status"] = "Success" if joint_chains else "NO_CHAIN"
            if not joint_chains:
                print("[FAIL] no grasp candidate yielded a complete grasp->pre_place->release chain")
                failed_place_pose = getattr(demo, "_last_joint_chain_failed_pose", None)
                failed_place_label = str(getattr(demo, "_last_joint_chain_failed_label", "") or "place")
                failed_place_candidates = list(getattr(demo, "_last_joint_chain_failed_candidate_poses", []) or [])
                failed_start_q = getattr(demo, "_last_joint_chain_failed_start_q", None)
                _print_joint_chain_failed_candidate_summary(demo)
                targeted.base.inspect_failed_pose(
                    demo,
                    bridge_mod,
                    failed_place_label if failed_place_pose is not None else "grasp",
                    args,
                    pose=failed_place_pose if failed_place_pose is not None else grasp_choice["pose"],
                    q_target=failed_start_q if failed_place_pose is not None else None,
                    gripper_closed=True if failed_place_pose is not None else False,
                    use_attach=True if failed_place_pose is not None else False,
                    candidate_poses=failed_place_candidates if failed_place_pose is not None else [item["pose"] for item in grasp_candidates],
                )
                return False
        selected_joint_chain = joint_chains[0]
        grasp_choice = selected_joint_chain["grasp_choice"]

    grasp_choice = _apply_deferred_two_step_final_approach(
        planner,
        demo,
        args,
        grasp_choice,
        two_step_pregrasp_lookup,
    )
    if not bool(grasp_choice.get("two_step_grasp", False)):
        failed_pose = grasp_choice.get("deferred_grasp_pose", grasp_choice.get("original_grasp_pose", grasp_choice.get("pose")))
        print("[FAIL] explicit two-step final approach failed for the selected grasp candidate")
        targeted.base.inspect_failed_pose(
            demo,
            bridge_mod,
            "grasp",
            args,
            pose=failed_pose,
            gripper_closed=False,
            candidate_poses=[item.get("deferred_grasp_pose", item.get("original_grasp_pose", item["pose"])) for item in grasp_successes],
        )
        return False
    if selected_joint_chain is not None:
        selected_joint_chain["grasp_choice"] = grasp_choice

    pregrasp_waypoints = int(max(grasp_choice.get("pregrasp_waypoints", 0), 0))
    if pregrasp_waypoints > 0:
        q_pregrasp_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in grasp_choice["q_path"][:pregrasp_waypoints]]
        pregrasp_pose = grasp_choice.get("deferred_pregrasp_pose", grasp_choice.get("pregrasp_pose", grasp_choice["pose"]))
        ok, _ = targeted.base.execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            f"{grasp_choice['label']}_pregrasp",
            pregrasp_pose,
            q_pregrasp_path,
            args.real_gripper_open,
            args,
        )
        if not ok:
            return False

        q_approach_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in grasp_choice["q_path"][pregrasp_waypoints - 1 :]]
        ok, _ = targeted.base.execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            f"{grasp_choice['label']}_final_approach",
            grasp_choice["pose"],
            q_approach_path,
            args.real_gripper_open,
            args,
        )
        if not ok:
            return False
    else:
        ok, _ = targeted.base.execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            grasp_choice["label"],
            grasp_choice["pose"],
            grasp_choice["q_path"],
            args.real_gripper_open,
            args,
        )
        if not ok:
            return False

    print("\n[close gripper]")
    if not targeted.base.confirm_simple_action("close the real gripper", args, bridge_mod=bridge_mod, env=demo.env, repeats=6):
        print("[abort] user cancelled before closing the real gripper")
        return False
    if real_exec is not None:
        real_exec.set_gripper(args.real_gripper_close)
        targeted.base.sync_demo_gripper_state(demo, closed=True, steps=4)
        targeted.base.set_pregrasp_object_freeze(demo, False)
        gripper_pos, blocked = targeted.base.real_gripper_blocked_after_close(
            real_exec,
            close_cmd=args.real_gripper_close,
            blocked_margin=args.real_gripper_blocked_margin,
        )
        if gripper_pos is not None and blocked is not None:
            print(
                f"[real] gripper.pos after close: {gripper_pos:.4f} "
                f"blocked_before_full_close={blocked}"
            )
    else:
        print("[dry-run] skipped real gripper close")
        targeted.base.sync_demo_gripper_state(demo, closed=True, steps=4)
        targeted.base.set_pregrasp_object_freeze(demo, False)

    _stabilize_post_grasp_attached_state(demo, args, grasp_choice)

    targeted.base.lift_active_object_above_table_if_needed(
        demo,
        args,
        min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
    )

    if bool(getattr(args, "curobo_attach_object", True)):
        _attach_transport_payload_to_curobo(planner, demo, args, label="transport")

    with _profile_stage(args, "post_grasp_lift") as prof:
        post_lift_ok = True
        planned_transport_path = []
        if selected_joint_chain is not None and bool(getattr(args, "reuse_joint_search_chain", True)):
            planned_transport_path = [
                np.asarray(q, dtype=np.float32).reshape(-1)[:7]
                for q in list(selected_joint_chain.get("q_pre_place_path") or [])
            ]
        current_after_grasp_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
        skip_independent_post_lift = False
        if len(planned_transport_path) >= 2:
            reuse_tol = float(max(getattr(args, "joint_search_reuse_start_q_tolerance", 0.03), 0.0))
            if bool((selected_joint_chain.get("grasp_choice") or {}).get("pair_first_ik_only", False)):
                reuse_tol = max(
                    reuse_tol,
                    float(max(getattr(args, "pair_first_reuse_start_q_tolerance", 0.12), 0.0)),
                )
            start_delta = float(np.max(np.abs(current_after_grasp_q - planned_transport_path[0])))
            if start_delta <= reuse_tol:
                skip_independent_post_lift = True
                print(
                    "[post_grasp_lift] skipped independent lift; "
                    "joint_search transport chain already starts at the executed grasp state "
                    f"(start_delta={start_delta:.4f} <= tol={reuse_tol:.4f})"
                )
            else:
                print(
                    "[post_grasp_lift] cannot skip independent lift; "
                    f"joint_search chain start_delta={start_delta:.4f} > tol={reuse_tol:.4f}"
                )
        if skip_independent_post_lift:
            prof["success"] = True
            prof["status"] = "SKIPPED_REUSE_JOINT_SEARCH_CHAIN"
            prof["path_waypoints"] = 0
        else:
            post_lift_ok = _skip_post_grasp_escape(demo, bridge_mod, real_exec, args, "post_grasp_lift", use_attach=True)
            prof["success"] = bool(post_lift_ok)
            prof["status"] = "Success" if post_lift_ok else "PLAN_OR_EXEC_FAIL"
    if not post_lift_ok:
        print("[warn] post-grasp lift failed; continuing to transport from current pose")

    if args.skip_goal_motion:
        print("[place] skipped place motion as requested; returning to the cycle start pose before finishing")
        targeted._register_transport_attached_box(
            demo,
            args,
            show_visual=False,
            activate_payload_visual=False,
            T_tcp_obj_override=getattr(demo, "_transport_attached_T_tcp_obj", None),
        )
        return _plan_and_execute_return_to_cycle_start(
            demo,
            bridge_mod,
            real_exec,
            args,
            start_q,
            use_attach=True,
            gripper_pos=args.real_gripper_close,
        )

    print("\n[move to targeted place]")
    try:
        targeted._ensure_target_registered_for_place(demo, rule.target_object_name)
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return False
    targeted._register_transport_attached_box(
        demo,
        args,
        T_tcp_obj_override=getattr(demo, "_transport_attached_T_tcp_obj", None),
    )

    relaxed_target_collision = False
    if selected_joint_chain is not None:
        print(
            "[place] joint_search selected a grasp-place pair after IK screening and attached-payload "
            "transport planning"
        )

    T_tcp_obj_transport = getattr(demo, "_transport_attached_T_tcp_obj", None)
    if T_tcp_obj_transport is None:
        T_tcp_obj_transport = grasp_choice.get("T_tcp_obj")
    direct_place_candidates = _build_direct_place_candidates(
        demo,
        bridge_mod,
        scene_capture_cache,
        rule,
        place_state_cache,
        args,
        T_tcp_obj_override=T_tcp_obj_transport,
    )
    if not direct_place_candidates:
        print("[FAIL] no targeted hover/release candidate could be built")
        return False

    if rule.primitive == "insert_vertical":
        direct_place_candidates = _filter_pre_place_candidates_by_verticality(direct_place_candidates, args)
    if not direct_place_candidates:
        print("[FAIL] hover/release candidates became empty after filtering")
        return False

    target_obj_name_place = curobo_wrapper.normalize_object_name(
        getattr(rule, "target_object_name", None)
    )
    exclude_names_place = _attached_source_exclude_names(args, rule)
    # During transport/final place, the grasped source is represented by attached
    # payload spheres, not by its original world obstacle.
    if rule.primitive == "insert_vertical" and target_obj_name_place:
        exclude_names_place.add(target_obj_name_place)
    exclude_names_place = exclude_names_place or None
    direct_place_disabled_links = _direct_place_contact_tolerant_disabled_links(planner)
    place_start_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    transport_screen_args = _transport_screen_args_for_object(args, _current_source_object_name(args))
    transport_successes = None
    joint_search_validated_place_choice = None
    pair_first_selected_chain = bool(
        selected_joint_chain is not None
        and bool((selected_joint_chain.get("grasp_choice") or {}).get("pair_first_ik_only", False))
    )
    if selected_joint_chain is not None and bool(getattr(args, "reuse_joint_search_chain", True)):
        planned_path = [
            np.asarray(q, dtype=np.float32).reshape(-1)[:7]
            for q in list(selected_joint_chain.get("q_pre_place_path") or [])
        ]
        reuse_tol = float(max(getattr(args, "joint_search_reuse_start_q_tolerance", 0.03), 0.0))
        if bool((selected_joint_chain.get("grasp_choice") or {}).get("pair_first_ik_only", False)):
            reuse_tol = max(
                reuse_tol,
                float(max(getattr(args, "pair_first_reuse_start_q_tolerance", 0.12), 0.0)),
            )
        if planned_path:
            start_delta = float(np.max(np.abs(place_start_q - planned_path[0])))
            if start_delta <= reuse_tol:
                reused_choice = dict(selected_joint_chain.get("pre_place_choice") or {})
                reused_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in planned_path]
                reused_path[0] = place_start_q.copy()
                metrics, score = _path_metrics_and_score(place_start_q, reused_path)
                reused_choice["q_path"] = reused_path
                reused_choice["metrics"] = metrics
                reused_choice["score"] = float(score) + float(_candidate_selection_penalty(reused_choice, args))
                reused_choice["start_q"] = place_start_q.copy()
                reused_choice["reused_joint_search_chain"] = True
                transport_successes = [reused_choice]
                validated_q_place_path = [
                    np.asarray(q, dtype=np.float32).reshape(-1)[:7]
                    for q in list(selected_joint_chain.get("q_place_path") or [])
                ]
                if (
                    bool(selected_joint_chain.get("joint_search_validated_final_contact", False))
                    and len(validated_q_place_path) >= 2
                    and np.allclose(reused_path[-1], validated_q_place_path[0], atol=0.03, rtol=0.0)
                ):
                    combined_path = reused_path + validated_q_place_path[1:]
                    combined_metrics, combined_score = _path_metrics_and_score(place_start_q, combined_path)
                    joint_search_validated_place_choice = dict(reused_choice)
                    joint_search_validated_place_choice["label"] = f"{reused_choice['label']}_final_contact"
                    joint_search_validated_place_choice["q_path"] = combined_path
                    joint_search_validated_place_choice["q_pre_place_path"] = reused_path
                    joint_search_validated_place_choice["q_place_path"] = validated_q_place_path
                    joint_search_validated_place_choice["two_stage_place"] = True
                    joint_search_validated_place_choice["final_contact_policy"] = "joint_search_validated_reuse"
                    joint_search_validated_place_choice["metrics"] = combined_metrics
                    joint_search_validated_place_choice["score"] = float(combined_score)
                    print(
                        "[place] reusing joint_search-validated final contact path "
                        f"(place_waypoints={len(validated_q_place_path)}, total_waypoints={len(combined_path)})"
                    )
                print(
                    f"[place] reusing joint_search transport path "
                    f"(start_delta={start_delta:.4f} <= tol={reuse_tol:.4f}, waypoints={len(reused_path)})"
                )
            else:
                action_text = (
                    "failing pair-first chain"
                    if pair_first_selected_chain
                    else "replanning"
                )
                print(
                    f"[place] joint_search transport path not reused: "
                    f"executed start q delta {start_delta:.4f} > tol {reuse_tol:.4f}; {action_text}"
                )
    if transport_successes is None:
        if pair_first_selected_chain:
            print(
                "[place] pair-first selected transport path was not reusable; "
                "not replanning transport_to_hover with the full candidate set"
            )
            transport_successes = []
        else:
            transport_successes = plan_transport_to_hover(
                planner,
                demo,
                transport_screen_args,
                place_start_q,
                direct_place_candidates,
                include_table=bool(getattr(args, "curobo_table_collision", True)),
                exclude_object_names=exclude_names_place,
                disabled_world_collision_links=direct_place_disabled_links,
            )
    if not transport_successes:
        if pair_first_selected_chain:
            print("[FAIL] pair-first transport path unavailable; legacy transport replanning is disabled")
            failed_pose = direct_place_candidates[0]["pose"] if direct_place_candidates else demo.tcp.pose
            _print_attached_sphere_clearance(planner, demo, args, label="pair_first_transport_reuse")
            targeted.base.inspect_failed_pose(
                demo,
                bridge_mod,
                "pair_first_transport_reuse",
                args,
                pose=failed_pose,
                gripper_closed=True,
                use_attach=True,
                candidate_poses=[item["pose"] for item in direct_place_candidates],
            )
            return False
        lift_m = float(max(getattr(args, "joint_search_start_collision_lift_m", 0.030), 0.0))
        lift_exclude_names = set(exclude_names_place or set())
        relief_name = _single_obstacle_start_collision_relief(
            planner,
            place_start_q,
            already_excluded=lift_exclude_names,
        )
        if relief_name:
            lift_exclude_names.add(relief_name)
            print(
                f"[place] transport_to_hover fallback: temporarily excluding start-collision "
                f"obstacle {relief_name!r} for {lift_m:.3f}m straight lift"
            )
        lift_path = _plan_short_world_z_lift_ik(
            planner,
            demo,
            args,
            place_start_q,
            lift_m=lift_m,
            label="transport_start_lift",
            include_table=bool(getattr(args, "curobo_table_collision", True)),
            exclude_object_names=lift_exclude_names,
            disabled_world_collision_links=_post_grasp_lift_disabled_world_links(
                planner,
                direct_place_disabled_links,
            ),
        )
        if lift_path is not None and len(lift_path) >= 2:
            lift_pose = _lift_pose_world_z(demo.tcp.pose, lift_m)
            ok, _ = targeted.base.execute_pose_path_stage(
                demo,
                bridge_mod,
                real_exec,
                "transport_start_lift",
                lift_pose,
                lift_path,
                args.real_gripper_close,
                args,
                use_attach=True,
            )
            if not ok:
                return False
            targeted.base.lift_active_object_above_table_if_needed(
                demo,
                args,
                min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
            )
            targeted._register_transport_attached_box(
                demo,
                args,
                T_tcp_obj_override=getattr(demo, "_transport_attached_T_tcp_obj", None),
            )
            if bool(getattr(args, "curobo_attach_object", True)):
                _attach_transport_payload_to_curobo(planner, demo, args, label="transport_after_start_lift")
            place_start_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
            print("[place] retrying transport_to_hover after straight start lift")
            transport_successes = plan_transport_to_hover(
                planner,
                demo,
                transport_screen_args,
                place_start_q,
                direct_place_candidates,
                include_table=bool(getattr(args, "curobo_table_collision", True)),
                exclude_object_names=lift_exclude_names,
                disabled_world_collision_links=direct_place_disabled_links,
            )
    if not transport_successes:
        print("[FAIL] cuRobo transport_to_hover planning failed")
        failed_pose = direct_place_candidates[0]["pose"] if direct_place_candidates else demo.tcp.pose
        _print_attached_sphere_clearance(planner, demo, args, label="transport_to_hover")
        targeted.base.inspect_failed_pose(
            demo,
            bridge_mod,
            "transport_to_hover",
            args,
            pose=failed_pose,
            gripper_closed=True,
            use_attach=True,
            candidate_poses=[item["pose"] for item in direct_place_candidates],
        )
        return False

    place_choice = joint_search_validated_place_choice
    if place_choice is None:
        for transport_choice in transport_successes:
            place_choice = plan_final_contact_approach(
                planner,
                demo,
                args,
                place_start_q,
                transport_choice,
                disabled_world_collision_links=direct_place_disabled_links,
            )
            if place_choice is not None:
                break
    if place_choice is None and any(bool(item.get("reused_joint_search_chain", False)) for item in transport_successes):
        if pair_first_selected_chain:
            print(
                "[place] pair-first selected hover failed final_contact; "
                "not running full transport_to_hover fallback"
            )
        else:
            print(
                "[place] reused joint_search hover failed final_contact; "
                "replanning transport_to_hover to collect alternate hover candidates"
            )
            transport_successes = plan_transport_to_hover(
                planner,
                demo,
                transport_screen_args,
                place_start_q,
                direct_place_candidates,
                include_table=bool(getattr(args, "curobo_table_collision", True)),
                exclude_object_names=exclude_names_place,
                disabled_world_collision_links=direct_place_disabled_links,
            )
            for transport_choice in transport_successes:
                place_choice = plan_final_contact_approach(
                    planner,
                    demo,
                    args,
                    place_start_q,
                    transport_choice,
                    disabled_world_collision_links=direct_place_disabled_links,
                )
                if place_choice is not None:
                    break
    if place_choice is None:
        print("[FAIL] final_contact_approach failed for all reachable hover candidates")
        failure_candidates = list(transport_successes or direct_place_candidates or [])
        if failure_candidates:
            failed_pose = failure_candidates[0].get("release_pose", failure_candidates[0]["pose"])
            failed_candidate_poses = [item.get("release_pose", item["pose"]) for item in failure_candidates]
        else:
            failed_pose = demo.tcp.pose
            failed_candidate_poses = []
        _print_attached_sphere_clearance(planner, demo, args, label="final_contact")
        targeted.base.inspect_failed_pose(
            demo,
            bridge_mod,
            "final_contact",
            args,
            pose=failed_pose,
            gripper_closed=True,
            use_attach=True,
            candidate_poses=failed_candidate_poses,
        )
        return False

    q_pre_place_path = [
        np.asarray(q, dtype=np.float32).reshape(-1)[:7]
        for q in place_choice.get("q_pre_place_path", place_choice["q_path"])
    ]
    q_place_path = [
        np.asarray(q, dtype=np.float32).reshape(-1)[:7]
        for q in place_choice.get("q_place_path", [np.asarray(q_pre_place_path[-1], dtype=np.float32).reshape(-1)[:7]])
    ]
    if bool(getattr(args, "_planning_prefetch_capture_only", False)):
        args._planning_prefetch_result = _build_prefetch_capture_result(
            args,
            start_q=start_q,
            grasp_start_q=grasp_start_q,
            selected_joint_chain=selected_joint_chain,
            grasp_choice=grasp_choice,
            place_choice=place_choice,
            q_pre_place_path=q_pre_place_path,
            q_place_path=q_place_path,
        )
        print(
            f"[prefetch] captured reusable plan for target={args.object_name}; "
            f"transport_waypoints={len(q_pre_place_path)}, place_waypoints={len(q_place_path)}"
        )
        return True
    if (
        not bool(getattr(args, "skip_return_to_cycle_start", False))
        and bool(getattr(args, "return_to_start_preplan", True))
    ):
        predicted_return_start_q = None
        place_mode_for_return_preplan = str(place_choice.get("place_mode", "drop_place"))
        skip_clearance_for_return_preplan = bool(getattr(args, "skip_post_place_clearance", False))
        force_clearance_for_return_preplan = (
            skip_clearance_for_return_preplan
            and place_mode_for_return_preplan == "insert_place"
        )
        if (
            place_mode_for_return_preplan != "insert_place"
            and not skip_clearance_for_return_preplan
            and len(q_place_path) >= 2
        ):
            predicted_return_start_q = np.asarray(q_place_path[0], dtype=np.float32).reshape(-1)[:7]
        elif skip_clearance_for_return_preplan and not force_clearance_for_return_preplan:
            predicted_return_start_q = np.asarray(q_place_path[-1], dtype=np.float32).reshape(-1)[:7]
        if predicted_return_start_q is not None:
            return_extra_obstacles = _return_to_start_placed_obstacles(demo, args, place_choice=place_choice)
            if return_extra_obstacles:
                args._return_to_start_extra_scene_obstacles = _copy_scene_obstacle_entries(return_extra_obstacles)
            if planner.attached_object_active:
                planner.detach_object_from_robot()
            prelift_planner_pose, prelift_lift_m = _prepare_return_preplan_prelift_pose(
                demo,
                planner,
                args,
                predicted_return_start_q,
                placed_obstacles=return_extra_obstacles,
            )
            _start_return_to_start_preplan(
                demo,
                planner,
                args,
                predicted_return_start_q,
                start_q,
                prelift_planner_pose=prelift_planner_pose,
                prelift_lift_m=prelift_lift_m,
                extra_scene_obstacles=return_extra_obstacles,
            )
    prefetch_manager = getattr(args, "_next_cycle_prefetch_manager", None)
    if prefetch_manager is not None:
        try:
            prefetch_manager.start_after_current_plan(
                current_args=args,
                scene_capture_cache=scene_capture_cache,
                place_state_cache=place_state_cache,
                rule=rule,
                place_choice=place_choice,
                predicted_next_start_q=start_q,
                next_cycle_idx=int(getattr(args, "_single_scene_cycle_idx", 0) or 0) + 1,
                failed_targets_this_cycle=getattr(args, "_next_cycle_prefetch_failed_targets_this_cycle", set()),
                deferred_failed_targets=getattr(args, "_next_cycle_prefetch_deferred_failed_targets", set()),
            )
        except Exception as exc:
            print(f"[prefetch] failed to start next-cycle plan prefetch: {exc}")
    slot_suffix = f", slot={place_choice['slot_name']}" if place_choice.get("slot_name") else ""
    variant_suffix = f", variant={place_choice['variant_label']}" if place_choice.get("variant_label") else ""
    print(
        f"[place] source={args.object_name}, primitive={rule.primitive}, "
        f"mode={place_choice.get('place_mode', 'unknown')}, "
        f"target={place_choice['target_name']}{slot_suffix}{variant_suffix}, "
        f"tcp_verticality={place_choice['tcp_verticality']:.3f}"
    )
    place_plan = place_choice.get("place_plan")
    T_world_obj_desired = getattr(place_plan, "T_world_obj_desired", None)
    if T_world_obj_desired is not None:
        obj_axes_z = _rotation_axis_world_z_components(np.asarray(T_world_obj_desired, dtype=np.float32).reshape(4, 4)[:3, :3])
        print(
            "[place] target object_axes_world_z: "
            f"x={obj_axes_z['x']:.3f}, y={obj_axes_z['y']:.3f}, z={obj_axes_z['z']:.3f} "
            f"(abs: x={obj_axes_z['abs_x']:.3f}, y={obj_axes_z['abs_y']:.3f}, z={obj_axes_z['abs_z']:.3f})"
        )
    print("[place] hover p:", np.round(targeted.base.flatten_np(place_choice["pre_place_pose"].p)[:3], 6), "q:", np.round(targeted.base.flatten_np(place_choice["pre_place_pose"].q)[:4], 6))
    print("[place] release p:", np.round(targeted.base.flatten_np(place_choice["place_pose"].p)[:3], 6), "q:", np.round(targeted.base.flatten_np(place_choice["place_pose"].q)[:4], 6))
    if rule.primitive == "insert_vertical":
        relaxed_target_collision = targeted._set_scene_obstacle_planner_box_scale(
            demo,
            place_choice["target_name"],
            float(args.place_insert_target_collision_scale),
        )

    if bool(place_choice.get("two_stage_place", False)):
        ok, _ = targeted.base.execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            f"{place_choice['label']}_hover",
            place_choice["pre_place_pose"],
            q_pre_place_path,
            args.real_gripper_close,
            args,
            use_attach=True,
        )
        if ok:
            ok, _ = targeted.base.execute_pose_path_stage(
                demo,
                bridge_mod,
                real_exec,
                place_choice["label"],
                place_choice["place_pose"],
                q_place_path,
                args.real_gripper_close,
                args,
                use_attach=True,
            )
    else:
        ok, _ = targeted.base.execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            place_choice["label"],
            place_choice["pose"],
            q_pre_place_path,
            args.real_gripper_close,
            args,
            use_attach=True,
        )
    if not ok:
        if planner.attached_object_active:
            planner.detach_object_from_robot()
        if relaxed_target_collision:
            targeted._set_scene_obstacle_planner_box_scale(demo, place_choice["target_name"], 1.0)
        return False

    if targeted.base.force_active_object_to_attached_pose(demo):
        print("[direct_pre_place] synchronized active object to attached release pose before opening gripper")
    else:
        print("[direct_pre_place] release-pose active object sync skipped; no attached TCP transform available")

    print("\n[open gripper at place]")
    if not targeted.base.confirm_simple_action("open the real gripper at the targeted place", args, bridge_mod=bridge_mod, env=demo.env, repeats=6):
        print("[abort] user cancelled before opening the real gripper at the targeted place")
        if planner.attached_object_active:
            planner.detach_object_from_robot()
        if relaxed_target_collision:
            targeted._set_scene_obstacle_planner_box_scale(demo, place_choice["target_name"], 1.0)
        return False
    if real_exec is not None:
        real_exec.set_gripper(args.real_gripper_open)
        targeted.base.sync_demo_gripper_state(demo, closed=False, steps=4)
    else:
        print("[dry-run] skipped real gripper open at the targeted place")
        targeted.base.sync_demo_gripper_state(demo, closed=False, steps=4)

    if planner.attached_object_active:
        planner.detach_object_from_robot()

    demo._attached_box_visual_visible = False
    demo._attached_object_visual_active = False
    targeted.base.update_attached_box_visual(demo, visible=False)
    place_mode_name = str(place_choice.get("place_mode", "drop_place"))
    settled_before_clearance = False
    if place_mode_name == "insert_place":
        # For insertion tasks the retreat path can pull the released item out of
        # the holder in dry-run sim. Cache the released pose before moving the
        # empty gripper away so the scene state represents the intended place.
        targeted.base.settle_released_active_object_for_scene_cache(demo, args)
        settled_before_clearance = True
    clearance_pose = _retreat_pose_for_place_mode(
        demo,
        place_mode_name,
        args,
        place_choice,
    )
    q_clearance_path = None
    skip_clearance_requested = bool(getattr(args, "skip_post_place_clearance", False))
    placed_object_obstacles = _return_to_start_placed_obstacles(demo, args, place_choice=place_choice)
    if placed_object_obstacles:
        args._return_to_start_extra_scene_obstacles = _copy_scene_obstacle_entries(placed_object_obstacles)
    force_clearance_after_insert = (
        skip_clearance_requested
        and place_mode_name == "insert_place"
    )
    if force_clearance_after_insert:
        print(
            "[place] post_place_clearance requested to skip, but insert_place requires "
            "a short retreat so the next object does not start from the insertion pose"
        )
    with _profile_stage(
        args,
        "post_place_clearance",
        max_attempts=int(getattr(args, "curobo_max_attempts", 2)),
        num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
        num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
        enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
    ) as prof:
        if skip_clearance_requested and not force_clearance_after_insert:
            print("[place] skipped post_place_clearance by request")
            prof["success"] = True
            prof["status"] = "SKIPPED_BY_REQUEST"
            prof["path_waypoints"] = 0
        else:
            current_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
            current_pose = demo.tcp.pose
            clearance_delta_m = float(np.linalg.norm(_get_pose_position(clearance_pose) - _get_pose_position(current_pose)))
            clearance_rot_delta_deg = float(
                np.degrees(
                    _quat_angle_rad_wxyz(
                        targeted.base.flatten_np(current_pose.q)[:4],
                        targeted.base.flatten_np(clearance_pose.q)[:4],
                    )
                )
            )
            reverse_place_path = [
                np.asarray(q, dtype=np.float32).reshape(-1)[:7]
                for q in list(place_choice.get("q_place_path") or [])
            ] if place_mode_name != "insert_place" else []
            if clearance_delta_m <= 1e-5 and clearance_rot_delta_deg <= 0.05:
                q_clearance_path = [current_q.copy()]
                print("[place] post_place_clearance is already at clearance pose; using zero-length path")
                prof["success"] = True
                prof["status"] = "ZERO_LENGTH"
                prof["path_waypoints"] = 1
                prof["world_changed"] = False
                prof["cache_hit"] = True
            elif len(reverse_place_path) >= 2:
                q_clearance_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7].copy() for q in reversed(reverse_place_path)]
                q_clearance_path[0] = current_q.copy()
                print(
                    "[place] post_place_clearance reusing reversed final-contact path "
                    f"({len(q_clearance_path)} waypoint(s))"
                )
                prof["success"] = True
                prof["status"] = "REUSED_FINAL_CONTACT_REVERSE"
                prof["path_waypoints"] = len(q_clearance_path)
                prof["world_changed"] = False
                prof["cache_hit"] = True
            else:
                _refresh_curobo_world(
                    planner,
                    demo,
                    args,
                    label="post_place_clearance",
                    include_active_object=False,
                    include_table=bool(getattr(args, "curobo_table_collision", True)),
                    extra_scene_obstacles=placed_object_obstacles,
                )
                q_clearance_path = _plan_constrained_linear_segment(
                    planner,
                    demo,
                    args,
                    np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7],
                    demo.tcp.pose,
                    clearance_pose,
                    label="post_place_clearance",
                    validation_pos_tol_m=float(max(getattr(args, "strict_short_linear_waypoint_pos_tol_m", 0.010), 0.0)),
                )
            if not q_clearance_path and bool(getattr(args, "allow_demo_planner_rescue", False)):
                _, q_clearance_path = targeted.base.plan_post_place_clearance_path(
                    demo,
                    retreat_distance=0.05,
                    label="post_place_clearance",
                )
            if str(prof.get("status", "")) not in {"REUSED_FINAL_CONTACT_REVERSE", "ZERO_LENGTH"}:
                prof["success"] = bool(q_clearance_path)
                prof["status"] = "Success" if q_clearance_path else "PLAN_FAIL"
                prof["path_waypoints"] = len(q_clearance_path or [])
                prof["world_changed"] = bool(getattr(planner, "_last_world_changed", False))
                prof["cache_hit"] = bool(getattr(planner, "_last_world_cache_hit", False))
    skipped_clearance_by_request = skip_clearance_requested and not force_clearance_after_insert
    if not q_clearance_path and not skipped_clearance_by_request:
        print("[place] skipped legacy post_place_clearance planner; enable --allow-demo-planner-rescue to use it")
    if (
        not bool(getattr(args, "skip_return_to_cycle_start", False))
        and bool(getattr(args, "return_to_start_preplan", True))
    ):
        return_preplan_start_q = None
        if q_clearance_path:
            return_preplan_start_q = np.asarray(q_clearance_path[-1], dtype=np.float32).reshape(-1)[:7]
        elif skipped_clearance_by_request:
            return_preplan_start_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
        if return_preplan_start_q is not None:
            return_extra_obstacles = placed_object_obstacles or _return_to_start_placed_obstacles(
                demo,
                args,
                place_choice=place_choice,
            )
            if return_extra_obstacles:
                args._return_to_start_extra_scene_obstacles = _copy_scene_obstacle_entries(return_extra_obstacles)
            prelift_planner_pose, prelift_lift_m = _prepare_return_preplan_prelift_pose(
                demo,
                planner,
                args,
                return_preplan_start_q,
                placed_obstacles=return_extra_obstacles,
            )
            _start_return_to_start_preplan(
                demo,
                planner,
                args,
                return_preplan_start_q,
                start_q,
                prelift_planner_pose=prelift_planner_pose,
                prelift_lift_m=prelift_lift_m,
                extra_scene_obstacles=return_extra_obstacles,
            )
    clearance_executed = False
    if q_clearance_path:
        ok, _ = targeted.base.execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            "post_place_clearance",
            clearance_pose,
            q_clearance_path,
            args.real_gripper_open,
            args,
            use_attach=False,
            skip_confirmation=True,
        )
        if not ok:
            print("[warn] post-place clearance execution failed after release; continuing to settle the object in place")
        else:
            clearance_executed = True
            print(
                "[place] post_place_clearance final q:",
                np.round(np.asarray(q_clearance_path[-1], dtype=np.float32).reshape(-1)[:7], 5).tolist(),
            )
    elif not skipped_clearance_by_request:
        print("[warn] post-place clearance planning failed after release; settling the object without moving the arm away first")

    if not settled_before_clearance:
        targeted.base.settle_released_active_object_for_scene_cache(demo, args)
    targeted._mark_place_rule_success(rule, place_state_cache, place_choice["slot_name"])

    if relaxed_target_collision:
        targeted._set_scene_obstacle_planner_box_scale(demo, place_choice["target_name"], 1.0)
        relaxed_target_collision = False
    if bool(getattr(args, "skip_return_to_cycle_start", False)):
        if force_clearance_after_insert and not clearance_executed:
            print(
                "[place] insert post_place_clearance failed; falling back to return_to_cycle_start "
                "despite --skip-return-to-cycle-start so the next object does not start from insertion"
            )
            return_ok = _plan_and_execute_return_to_cycle_start(
                demo,
                bridge_mod,
                real_exec,
                args,
                start_q,
            )
            if not return_ok and not bool(getattr(args, "strict_return_to_cycle_start", False)):
                print(
                    "[warn] return_to_cycle_start failed after completed place; "
                    "treating pick-place as successful and continuing from the current arm pose"
                )
                return True
            return return_ok
        if clearance_executed:
            print("[place] completed targeted place and clearance; skipping return_to_cycle_start by request")
        else:
            print("[place] completed targeted place without clearance; skipping return_to_cycle_start by request")
        return True
    if clearance_executed:
        print("[place] completed targeted place and clearance; returning to the cycle start pose")
    else:
        print("[place] completed targeted place without clearance; returning to the cycle start pose from the current release state")
    return_ok = _plan_and_execute_return_to_cycle_start(
        demo,
        bridge_mod,
        real_exec,
        args,
        start_q,
    )
    if not return_ok and not bool(getattr(args, "strict_return_to_cycle_start", False)):
        print(
            "[warn] return_to_cycle_start failed after completed place; "
            "treating pick-place as successful and continuing from the current arm pose"
        )
        return True
    return return_ok


def _single_scene_actor_name(actor, fallback: str = "") -> str:
    try:
        name = getattr(actor, "name", None)
        if name:
            return str(name)
    except Exception:
        pass
    try:
        name = actor.get_name()
        if name:
            return str(name)
    except Exception:
        pass
    return str(fallback)


def _single_scene_actor_pose_matrix(actor) -> np.ndarray | None:
    if actor is None:
        return None
    try:
        pose = actor.pose
        p = targeted.base.flatten_np(pose.p)[:3].astype(np.float32)
        q = targeted.base.flatten_np(pose.q)[:4].astype(np.float32)
        return targeted.base.pose_to_matrix(p, q).astype(np.float32)
    except Exception:
        return None


def _single_scene_cache_entry(scene_capture_cache, object_name: str | None):
    normalized = curobo_wrapper.normalize_object_name(object_name)
    if normalized is None or not isinstance(scene_capture_cache, dict):
        return None
    objects = scene_capture_cache.get("objects")
    if not isinstance(objects, dict):
        return None
    entry = objects.get(normalized)
    return entry if isinstance(entry, dict) else None


def _single_scene_get_object_args(base_args, object_name: str):
    try:
        cycle_args, _ = targeted.base.make_cycle_args(base_args, object_name)
        return cycle_args
    except Exception:
        return None


def _single_scene_build_actor_registry(demo, base_args, scene_capture_cache, active_name: str, active_args) -> dict:
    registry = {}
    active_name = curobo_wrapper.normalize_object_name(active_name)
    if active_name is not None:
        registry[active_name] = {
            "object_name": active_name,
            "actor": getattr(getattr(demo, "base_env", None), "obj", None),
            "actor_name": f"scene_obstacle_{active_name}",
            "planner_actor_name": f"scene_obstacle_{active_name}",
            "object_args": active_args,
        }

    actor_by_name = {}
    env_unwrapped = getattr(getattr(demo, "env", None), "unwrapped", None)
    for actor in list(getattr(env_unwrapped, "_scene_obstacle_actors", []) or []):
        actor_name = _single_scene_actor_name(actor)
        if actor_name:
            actor_by_name[actor_name] = actor

    for item in list(getattr(demo, "scene_obstacles", []) or []):
        object_name = curobo_wrapper.normalize_object_name(item.get("object_name"))
        if object_name is None:
            continue
        if object_name.startswith("virtual_") and _single_scene_cache_entry(scene_capture_cache, object_name) is None:
            continue
        actor_name = str(item.get("actor_name") or f"scene_obstacle_{object_name}")
        object_args = _single_scene_get_object_args(base_args, object_name)
        registry[object_name] = {
            "object_name": object_name,
            "actor": actor_by_name.get(actor_name),
            "actor_name": actor_name,
            "planner_actor_name": actor_name,
            "object_args": object_args,
            "seed_entry": dict(item),
        }

    if isinstance(scene_capture_cache, dict):
        objects = scene_capture_cache.get("objects")
        if isinstance(objects, dict):
            for raw_name in objects.keys():
                object_name = curobo_wrapper.normalize_object_name(raw_name)
                if object_name is None or object_name in registry:
                    continue
                object_args = _single_scene_get_object_args(base_args, object_name)
                registry[object_name] = {
                    "object_name": object_name,
                    "actor": None,
                    "actor_name": f"scene_obstacle_{object_name}",
                    "planner_actor_name": f"scene_obstacle_{object_name}",
                    "object_args": object_args,
                }

    demo._single_scene_object_registry = registry
    return registry


def _single_scene_registry(demo) -> dict:
    registry = getattr(demo, "_single_scene_object_registry", None)
    return registry if isinstance(registry, dict) else {}


def _single_scene_to_env_tensor(demo, values):
    torch_mod = getattr(targeted.base, "torch", None)
    if torch_mod is None:
        return np.asarray(values, dtype=np.float32)
    device = getattr(getattr(demo, "base_env", None), "device", None)
    return torch_mod.as_tensor(values, dtype=torch_mod.float32, device=device)


def _single_scene_update_active_asset_metadata(demo, args) -> None:
    base_env = getattr(demo, "base_env", None)
    if base_env is None:
        return
    asset_file = str(Path(str(args.sim_asset_file or args.mesh_file)).expanduser())
    asset_scale = float(args.sim_asset_scale or args.mesh_scale or 1.0)
    points = targeted.base.get_asset_local_points(asset_file, asset_scale)
    lo = np.asarray(points.min(axis=0), dtype=np.float32).reshape(1, 3)
    hi = np.asarray(points.max(axis=0), dtype=np.float32).reshape(1, 3)
    base_env.obj_local_aabb_min = _single_scene_to_env_tensor(demo, lo)
    base_env.obj_local_aabb_max = _single_scene_to_env_tensor(demo, hi)
    base_env.object_zs = _single_scene_to_env_tensor(demo, np.asarray([-float(lo.reshape(3)[2])], dtype=np.float32))
    base_env.object_initial_height = _single_scene_to_env_tensor(demo, np.asarray([-1.0], dtype=np.float32))
    try:
        base_env.object_asset_path = asset_file
        base_env.object_scale_vec = np.asarray([asset_scale, asset_scale, asset_scale], dtype=float)
    except Exception:
        pass
    if hasattr(base_env, "get_obj_xy_shortest_edge_vector"):
        try:
            torch_mod = getattr(targeted.base, "torch", None)
            if torch_mod is not None:
                with torch_mod.no_grad():
                    base_env.obj_xy_shortest_edge_vector = base_env.get_obj_xy_shortest_edge_vector().detach()
            else:
                base_env.obj_xy_shortest_edge_vector = np.zeros((1, 3), dtype=np.float32)
        except Exception:
            pass


def _single_scene_remove_planner_object(demo, object_name: str, actor_name: str | None = None) -> None:
    names = {
        str(actor_name or ""),
        f"scene_obstacle_{object_name}",
        str(object_name),
    }
    if object_name == curobo_wrapper.normalize_object_name(getattr(getattr(demo, "args", None), "object_name", None)):
        names.add("jiaobang")
    for name in [x for x in names if x]:
        try:
            demo.planner.remove_normal_object(name)
        except Exception:
            pass


def _single_scene_set_planner_obstacle(demo, entry: dict) -> None:
    if not bool(entry.get("planner_collision", False)):
        return
    actor_name = str(entry.get("actor_name") or entry.get("object_name") or "")
    T_world_obj = entry.get("T_world_obj")
    planner_box_size = entry.get("planner_box_size", entry.get("visual_box_size"))
    if not actor_name or T_world_obj is None or planner_box_size is None:
        return
    try:
        from mplib import collision_detection as mplib_cd

        T_world_obj = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4)
        pos = T_world_obj[:3, 3].astype(np.float32)
        quat = targeted.base.bridge_mod_mat2quat(T_world_obj[:3, :3]).astype(np.float32)
        dims = np.asarray(planner_box_size, dtype=np.float32).reshape(3)
        collision_object = mplib_cd.fcl.CollisionObject(
            mplib_cd.fcl.Box(dims.tolist()),
            pos.tolist(),
            quat.tolist(),
        )
        demo.planner.set_normal_object(actor_name, collision_object)
    except Exception as exc:
        print(f"[single_scene] warning: failed to update planner obstacle {actor_name}: {exc}")


def _single_scene_obstacle_box_scale(args, object_name: str, placed: bool) -> float:
    object_spec = targeted.base.get_object_spec(object_name)
    fixed_scene_names = {
        curobo_wrapper.normalize_object_name(name)
        for name in (
            list(getattr(args, "selected_obstacle_object_names", []) or [])
            + list(getattr(args, "tracked_scene_object_names", []) or [])
        )
    }
    fixed_scene_names.discard(None)
    global_box_scale = float(max(getattr(args, "scene_obstacle_box_scale", 1.0), 1e-3))
    object_box_scale = float(max(getattr(object_spec, "scene_obstacle_box_scale", 1.0) or 1.0, 1e-3))
    non_fixed_object_box_scale = 1.2 if object_name not in fixed_scene_names else 1.0
    placed_box_scale = (
        float(max(getattr(args, "placed_scene_obstacle_box_scale", 1.0), 1e-3))
        if placed
        else 1.0
    )
    return float(global_box_scale * object_box_scale * non_fixed_object_box_scale * placed_box_scale)


def _single_scene_build_obstacle_entry(demo, args, object_name: str, meta: dict, scene_capture_cache) -> dict | None:
    object_name = curobo_wrapper.normalize_object_name(object_name)
    if object_name is None:
        return None
    object_args = meta.get("object_args") or args
    asset_file = str(Path(str(getattr(object_args, "sim_asset_file", None) or getattr(object_args, "mesh_file", ""))).expanduser())
    asset_scale = float(getattr(object_args, "sim_asset_scale", None) or getattr(object_args, "mesh_scale", 1.0) or 1.0)
    actor = meta.get("actor")
    T_world_obj = _single_scene_actor_pose_matrix(actor)
    cache_entry = _single_scene_cache_entry(scene_capture_cache, object_name)
    if T_world_obj is None and cache_entry is not None and cache_entry.get("T_world_obj") is not None:
        T_world_obj = np.asarray(cache_entry["T_world_obj"], dtype=np.float32).reshape(4, 4)
    if T_world_obj is None:
        return None
    placed = bool(cache_entry.get("placed", False)) if isinstance(cache_entry, dict) else False
    visual_box_size = targeted.base.get_asset_box_size(asset_file, asset_scale)
    box_scale = _single_scene_obstacle_box_scale(args, object_name, placed)
    planner_box_size = (np.asarray(visual_box_size, dtype=np.float32) * box_scale).astype(np.float32)
    actor_name = str(meta.get("planner_actor_name") or meta.get("actor_name") or f"scene_obstacle_{object_name}")
    return {
        "object_name": object_name,
        "actor_name": actor_name,
        "label": str((cache_entry or {}).get("label", object_name)),
        "score": float((cache_entry or {}).get("score", 1.0)),
        "T_world_obj": T_world_obj,
        "placed": placed,
        "planner_collision": True,
        "asset_file": asset_file,
        "asset_scale": float(asset_scale),
        "visual_box_size": np.asarray(visual_box_size, dtype=np.float32).copy(),
        "planner_box_size": planner_box_size,
        "planner_box_actor_name": str((meta.get("seed_entry") or {}).get("planner_box_actor_name", "")),
    }


def _single_scene_sync_obstacles(demo, args, selected_name: str, obstacle_names, scene_capture_cache) -> None:
    selected_name = curobo_wrapper.normalize_object_name(selected_name)
    registry = _single_scene_registry(demo)
    registry_names = set(registry.keys())
    desired_names = {
        curobo_wrapper.normalize_object_name(name)
        for name in list(obstacle_names or [])
        if curobo_wrapper.normalize_object_name(name) is not None
    }
    if selected_name is not None:
        desired_names.discard(selected_name)

    existing_virtual_entries = []
    for item in list(getattr(demo, "scene_obstacles", []) or []):
        item_name = curobo_wrapper.normalize_object_name(item.get("object_name"))
        if item_name not in registry_names:
            existing_virtual_entries.append(item)

    for object_name, meta in registry.items():
        _single_scene_remove_planner_object(demo, object_name, meta.get("planner_actor_name") or meta.get("actor_name"))

    obstacle_entries = []
    obstacle_actors = []
    for object_name in sorted(desired_names):
        meta = registry.get(object_name)
        if meta is None:
            print(f"[single_scene] warning: no actor registered for obstacle {object_name}; skipping")
            continue
        entry = _single_scene_build_obstacle_entry(demo, args, object_name, meta, scene_capture_cache)
        if entry is None:
            print(f"[single_scene] warning: no pose available for obstacle {object_name}; skipping")
            continue
        obstacle_entries.append(entry)
        _single_scene_set_planner_obstacle(demo, entry)
        actor = meta.get("actor")
        if actor is not None:
            obstacle_actors.append(actor)

    demo.scene_obstacles = obstacle_entries + existing_virtual_entries
    try:
        env_unwrapped = demo.env.unwrapped
        virtual_actor_names = {
            str(item.get("actor_name") or "")
            for item in existing_virtual_entries
            if item.get("actor_name")
        }
        virtual_actors = [
            actor
            for actor in list(getattr(env_unwrapped, "_scene_obstacle_actors", []) or [])
            if _single_scene_actor_name(actor) in virtual_actor_names
        ]
        env_unwrapped._scene_obstacle_actors = obstacle_actors + virtual_actors
    except Exception:
        pass
    print(
        f"[single_scene] active={selected_name}, scene obstacle(s)="
        f"{[item.get('object_name') for item in obstacle_entries]}"
    )


def _single_scene_activate_object(
    demo,
    bridge_mod,
    base_args,
    cycle_args,
    selected_name: str,
    obstacle_names,
    scene_capture_cache,
) -> bool:
    selected_name = curobo_wrapper.normalize_object_name(selected_name)
    registry = _single_scene_registry(demo)
    meta = registry.get(selected_name)
    if meta is None or meta.get("actor") is None:
        print(f"[single_scene][FAIL] object {selected_name!r} is not present in the current ManiSkill scene")
        return False

    actor = meta["actor"]
    base_env = demo.env.unwrapped
    base_env.obj = actor
    try:
        base_env._objs = [actor]
    except Exception:
        pass
    demo.base_env = base_env
    demo.args = cycle_args
    cycle_args._single_scene_active_object_name = selected_name
    _single_scene_update_active_asset_metadata(demo, cycle_args)
    apply_physics_profile = getattr(bridge_mod, "apply_pick_object_physics_profile", None)
    if callable(apply_physics_profile):
        apply_physics_profile(demo.env, cycle_args)
    _single_scene_sync_obstacles(demo, cycle_args, selected_name, obstacle_names, scene_capture_cache)
    try:
        demo.refresh_runtime_handles(rebuild_visual=False)
    except Exception:
        pass
    if bool(getattr(cycle_args, "freeze_active_object_before_grasp", True)):
        demo._freeze_active_object_before_grasp = False
        demo._frozen_active_object_pose = None
        targeted.base.set_pregrasp_object_freeze(demo, True)
        targeted.base.refresh_frozen_active_object_pose(demo)
    print(f"[single_scene] switched active object to {selected_name} without recreating env")
    return True


def _single_scene_restore_after_failed_attempt(
    demo,
    args,
    scene_capture_cache,
    selected_name: str,
    cycle_start_q,
) -> None:
    if demo is None:
        return
    selected_name = curobo_wrapper.normalize_object_name(selected_name)
    restored_fields = []
    try:
        planner = _get_or_create_curobo_planner_serialized(args)
        if getattr(planner, "attached_object_active", False):
            planner.detach_object_from_robot()
            restored_fields.append("curobo_attached")
    except Exception:
        pass
    try:
        targeted.base.sync_demo_gripper_state(demo, closed=False, steps=4)
        restored_fields.append("gripper")
    except Exception:
        pass
    try:
        targeted.base.set_pregrasp_object_freeze(demo, False)
    except Exception:
        pass
    try:
        demo._attached_box_visual_visible = False
        demo._attached_object_visual_active = False
        targeted.base.update_attached_box_visual(demo, visible=False)
        restored_fields.append("attached_visual")
    except Exception:
        pass
    _clear_visualized_attached_spheres(demo)
    _restore_transport_payload_state(
        demo,
        {
            "_transport_attached_T_tcp_obj": _MISSING_ATTR,
            "attached_box_size": _MISSING_ATTR,
            "attached_box_pose_tcp": _MISSING_ATTR,
            "_attached_object_visual_active": False,
            "_attached_box_visual_visible": False,
        },
    )
    entry = _single_scene_cache_entry(scene_capture_cache, selected_name)
    T_world_obj = None if not isinstance(entry, dict) else entry.get("T_world_obj")
    if T_world_obj is not None and not bool((entry or {}).get("placed", False)):
        try:
            pose = _pose_from_world_matrix(np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4))
            registry = _single_scene_registry(demo)
            actor = (registry.get(selected_name) or {}).get("actor")
            if actor is not None:
                actor.set_pose(pose)
                zero_vel = np.zeros(3, dtype=np.float32)
                for method_name in ("set_linear_velocity", "set_velocity"):
                    method = getattr(actor, method_name, None)
                    if callable(method):
                        method(zero_vel)
                ang_method = getattr(actor, "set_angular_velocity", None)
                if callable(ang_method):
                    ang_method(zero_vel)
            elif selected_name == curobo_wrapper.normalize_object_name(getattr(args, "object_name", None)):
                _set_active_object_pose_quiet(demo, pose.p, pose.q)
            restored_fields.append("object_pose")
        except Exception as exc:
            print(f"[single_scene] warning: failed to restore object pose for {selected_name}: {exc}")
    q_restore = _q7_or_none(cycle_start_q)
    if q_restore is not None:
        try:
            targeted.base.sync_demo_arm_qpos(demo, q_restore)
            restored_fields.append("arm_q")
        except Exception as exc:
            print(f"[single_scene] warning: failed to restore arm q after failed {selected_name}: {exc}")
    try:
        if bool(getattr(args, "freeze_active_object_before_grasp", True)):
            targeted.base.set_pregrasp_object_freeze(demo, True)
            targeted.base.refresh_frozen_active_object_pose(demo)
    except Exception:
        pass
    print(
        f"[single_scene] restored simulation state after failed target={selected_name}: "
        f"{sorted(set(restored_fields))}"
    )
    _record_profile(
        args,
        "failed_attempt_restore",
        success=True,
        status="RESTORED",
        target_name=selected_name,
        restored_fields=sorted(set(restored_fields)),
        restored_arm_q=q_restore is not None,
        restored_object_pose=T_world_obj is not None,
    )


_PREFETCH_DROP_KEYS = {
    "result",
    "raw_result",
    "planner_pose",
    "terminal_align",
    "place_plan",
}
_PREFETCH_UNSAFE = object()


def _prefetch_safe_copy_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Path):
        return Path(value)
    if isinstance(value, types.ModuleType) or callable(value):
        return _PREFETCH_UNSAFE
    if isinstance(value, dict):
        copied = {}
        for key, item in value.items():
            copied_key = _prefetch_safe_copy_value(key)
            copied_item = _prefetch_safe_copy_value(item)
            if copied_key is _PREFETCH_UNSAFE or copied_item is _PREFETCH_UNSAFE:
                continue
            copied[copied_key] = copied_item
        return copied
    if isinstance(value, list):
        copied = []
        for item in value:
            copied_item = _prefetch_safe_copy_value(item)
            if copied_item is not _PREFETCH_UNSAFE:
                copied.append(copied_item)
        return copied
    if isinstance(value, tuple):
        copied = []
        for item in value:
            copied_item = _prefetch_safe_copy_value(item)
            if copied_item is not _PREFETCH_UNSAFE:
                copied.append(copied_item)
        return tuple(copied)
    if hasattr(value, "__dict__"):
        return _copy_namespace_like(value)
    try:
        return copy.deepcopy(value)
    except Exception:
        return _PREFETCH_UNSAFE


def _copy_namespace_like(value):
    if hasattr(value, "__dict__"):
        try:
            copied = {}
            for key, item in vars(value).items():
                copied_item = _prefetch_safe_copy_value(item)
                if copied_item is not _PREFETCH_UNSAFE:
                    copied[str(key)] = copied_item
            return targeted.argparse.Namespace(**copied)
        except Exception:
            pass
    return value


def _copy_scene_capture_cache_for_prefetch(scene_capture_cache) -> dict:
    if not isinstance(scene_capture_cache, dict):
        return {}
    copied: dict = {}
    for key, value in scene_capture_cache.items():
        if key == "objects" and isinstance(value, dict):
            objects = {}
            for raw_name, raw_entry in value.items():
                if not isinstance(raw_entry, dict):
                    continue
                entry = {}
                for entry_key, entry_value in raw_entry.items():
                    if isinstance(entry_value, np.ndarray):
                        entry[entry_key] = entry_value.copy()
                    elif entry_key == "object_args":
                        entry[entry_key] = _copy_namespace_like(entry_value)
                    else:
                        copied_value = _prefetch_safe_copy_value(entry_value)
                        if copied_value is not _PREFETCH_UNSAFE:
                            entry[entry_key] = copied_value
                objects[raw_name] = entry
            copied[key] = objects
        elif key == "scene_obstacles" and isinstance(value, list):
            obstacles = []
            for item in value:
                if isinstance(item, dict):
                    copied_item = _prefetch_safe_copy_value(item)
                    if copied_item is not _PREFETCH_UNSAFE:
                        obstacles.append(copied_item)
            copied[key] = obstacles
        elif isinstance(value, np.ndarray):
            copied[key] = value.copy()
        elif key == "fp_rt":
            # FoundationPose runtime is not deep-copyable; the prefetch path only
            # reuses cached poses, so sharing the reference is sufficient.
            copied[key] = value
        else:
            copied_value = _prefetch_safe_copy_value(value)
            if copied_value is not _PREFETCH_UNSAFE:
                copied[key] = copied_value
    return copied


def _predicted_place_T_world_obj(place_choice) -> np.ndarray | None:
    if not isinstance(place_choice, dict):
        return None
    for value in (
        place_choice.get("T_world_obj_desired"),
        getattr(place_choice.get("place_plan", None), "T_world_obj_desired", None),
    ):
        if value is None:
            continue
        try:
            return np.asarray(value, dtype=np.float32).reshape(4, 4).copy()
        except Exception:
            continue
    return None


def _apply_predicted_place_to_scene_cache(scene_capture_cache: dict, object_name: str | None, object_args, place_choice) -> np.ndarray | None:
    normalized = curobo_wrapper.normalize_object_name(object_name)
    T_world_obj = _predicted_place_T_world_obj(place_choice)
    if normalized is None or T_world_obj is None or not isinstance(scene_capture_cache, dict):
        return None
    objects = scene_capture_cache.setdefault("objects", {})
    if not isinstance(objects, dict):
        return None
    old_entry = objects.get(normalized, {}) if isinstance(objects.get(normalized), dict) else {}
    objects[normalized] = {
        "object_name": normalized,
        "label": str(old_entry.get("label", getattr(object_args, "target_object_name", "") or normalized)),
        "score": float(old_entry.get("score", 1.0)),
        "box": np.asarray(old_entry.get("box", np.zeros(4, dtype=np.float32)), dtype=np.float32).reshape(4),
        "T_cam_obj": np.asarray(old_entry.get("T_cam_obj", np.eye(4, dtype=np.float32)), dtype=np.float32).reshape(4, 4),
        "T_world_obj": T_world_obj,
        "object_args": _copy_namespace_like(object_args),
        "placed": True,
    }
    return T_world_obj


def _copy_place_state_cache_for_prefetch(place_state_cache) -> dict:
    if isinstance(place_state_cache, dict):
        try:
            return copy.deepcopy(place_state_cache)
        except Exception:
            return {"used_slots_by_target": dict(place_state_cache.get("used_slots_by_target", {}) or {})}
    return {"used_slots_by_target": {}}


def _sanitize_prefetch_value(value, _memo: set[int] | None = None, _depth: int = 0):
    if _memo is None:
        _memo = set()
    if _depth > 80:
        return _PREFETCH_UNSAFE
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=True) if value.dtype.kind in {"f", "i", "u"} else value.copy()
    if isinstance(value, Path):
        return Path(value)
    if isinstance(value, types.ModuleType) or callable(value):
        return _PREFETCH_UNSAFE
    if isinstance(value, dict):
        value_id = id(value)
        if value_id in _memo:
            return _PREFETCH_UNSAFE
        _memo.add(value_id)
        copied = {}
        try:
            for key, item in value.items():
                if str(key) in _PREFETCH_DROP_KEYS:
                    continue
                copied_item = _sanitize_prefetch_value(item, _memo, _depth + 1)
                if copied_item is not _PREFETCH_UNSAFE:
                    copied[str(key)] = copied_item
            return copied
        finally:
            _memo.discard(value_id)
    if isinstance(value, list):
        value_id = id(value)
        if value_id in _memo:
            return _PREFETCH_UNSAFE
        _memo.add(value_id)
        try:
            copied = []
            for item in value:
                copied_item = _sanitize_prefetch_value(item, _memo, _depth + 1)
                if copied_item is not _PREFETCH_UNSAFE:
                    copied.append(copied_item)
            return copied
        finally:
            _memo.discard(value_id)
    if isinstance(value, tuple):
        value_id = id(value)
        if value_id in _memo:
            return _PREFETCH_UNSAFE
        _memo.add(value_id)
        try:
            copied = []
            for item in value:
                copied_item = _sanitize_prefetch_value(item, _memo, _depth + 1)
                if copied_item is not _PREFETCH_UNSAFE:
                    copied.append(copied_item)
            return tuple(copied)
        finally:
            _memo.discard(value_id)
    return value


def _with_prefetch_goal_pose(value):
    if not isinstance(value, dict):
        return value
    copied = dict(value)
    if copied.get("T_world_obj_desired") is None:
        place_plan = copied.get("place_plan")
        T_world_obj_desired = getattr(place_plan, "T_world_obj_desired", None)
        if T_world_obj_desired is not None:
            try:
                copied["T_world_obj_desired"] = np.asarray(T_world_obj_desired, dtype=np.float32).reshape(4, 4).copy()
            except Exception:
                pass
    return copied


def _selected_chain_with_prefetch_goal_pose(selected_joint_chain):
    if not isinstance(selected_joint_chain, dict):
        return selected_joint_chain
    copied = dict(selected_joint_chain)
    for key in ("pre_place_choice", "place_choice"):
        if isinstance(copied.get(key), dict):
            copied[key] = _with_prefetch_goal_pose(copied[key])
    return copied


def _build_prefetch_capture_result(
    args,
    *,
    start_q,
    grasp_start_q,
    selected_joint_chain,
    grasp_choice,
    place_choice,
    q_pre_place_path,
    q_place_path,
) -> dict:
    selected_joint_chain = _selected_chain_with_prefetch_goal_pose(selected_joint_chain)
    place_choice = _with_prefetch_goal_pose(place_choice)
    return {
        "target_name": curobo_wrapper.normalize_object_name(getattr(args, "object_name", None)),
        "start_q": np.asarray(start_q, dtype=np.float32).reshape(-1)[:7].copy(),
        "grasp_start_q": np.asarray(grasp_start_q, dtype=np.float32).reshape(-1)[:7].copy(),
        "selected_joint_chain": _sanitize_prefetch_value(selected_joint_chain),
        "grasp_choice": _sanitize_prefetch_value(grasp_choice),
        "place_choice": _sanitize_prefetch_value(place_choice),
        "q_pre_place_path": [np.asarray(q, dtype=np.float32).reshape(-1)[:7].copy() for q in list(q_pre_place_path or [])],
        "q_place_path": [np.asarray(q, dtype=np.float32).reshape(-1)[:7].copy() for q in list(q_place_path or [])],
    }


class _NextCyclePlanPrefetchManager:
    def __init__(self, create_demo_func, bridge_mod, planner_mod, base_args, cycle_object_sequence):
        self.create_demo_func = create_demo_func
        self.bridge_mod = bridge_mod
        self.planner_mod = planner_mod
        self.base_args = base_args
        self.cycle_object_sequence = list(cycle_object_sequence or [])
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._reserved: dict | None = None
        self._result: dict | None = None

    def reserved_target_for_cycle(self, cycle_idx: int, scene_capture_cache, failed_targets, deferred_targets) -> str | None:
        with self._lock:
            reserved = dict(self._reserved or {})
        if int(reserved.get("cycle_idx", -1)) != int(cycle_idx):
            return None
        target_name = curobo_wrapper.normalize_object_name(reserved.get("target_name"))
        if target_name is None:
            return None
        if target_name in set(failed_targets or set()) or target_name in set(deferred_targets or set()):
            return None
        entry = _single_scene_cache_entry(scene_capture_cache, target_name)
        if isinstance(entry, dict) and bool(entry.get("placed", False)):
            return None
        return target_name

    def start_after_current_plan(
        self,
        *,
        current_args,
        scene_capture_cache,
        place_state_cache,
        rule,
        place_choice,
        predicted_next_start_q,
        next_cycle_idx: int,
        failed_targets_this_cycle,
        deferred_failed_targets,
    ) -> None:
        current_object_name = curobo_wrapper.normalize_object_name(getattr(current_args, "object_name", None))

        def record_start(status: str, *, target_name=None, success: bool = False, **fields) -> None:
            _record_prefetch_profile(
                self.base_args,
                curobo_wrapper.normalize_object_name(target_name) or current_object_name,
                "next_cycle_prefetch_start",
                success=success,
                status=status,
                cycle_idx=int(next_cycle_idx),
                current_object_name=current_object_name,
                **fields,
            )

        if not bool(getattr(current_args, "next_cycle_plan_prefetch", True)):
            record_start("DISABLED")
            return
        if bool(getattr(current_args, "skip_return_to_cycle_start", False)):
            print("[prefetch] next-cycle plan prefetch skipped: next start_q is not fixed when return_to_start is skipped")
            record_start("SKIP_RETURN_TO_START")
            return
        predicted_next_start_q = _q7_or_none(predicted_next_start_q)
        if predicted_next_start_q is None:
            record_start("NO_PREDICTED_START_Q")
            return
        scene_snapshot = _copy_scene_capture_cache_for_prefetch(scene_capture_cache)
        placed_T = _apply_predicted_place_to_scene_cache(
            scene_snapshot,
            getattr(current_args, "object_name", None),
            current_args,
            place_choice,
        )
        if placed_T is None:
            print("[prefetch] next-cycle plan prefetch skipped: current place result has no predicted object world pose")
            record_start("NO_PREDICTED_PLACED_POSE")
            return
        place_state_snapshot = _copy_place_state_cache_for_prefetch(place_state_cache)
        try:
            targeted._mark_place_rule_success(rule, place_state_snapshot, place_choice.get("slot_name"))
        except Exception:
            pass
        cached_scene_names = targeted.base.list_cached_scene_object_names(scene_snapshot)
        available_rule_names = targeted._list_cached_unplaced_rule_names(scene_snapshot)
        selected_name, target_pool, target_candidates = targeted._select_random_cycle_target(
            self.base_args,
            self.cycle_object_sequence,
            scene_snapshot,
            available_rule_names,
            set(failed_targets_this_cycle or set()),
            set(deferred_failed_targets or set()),
            int(next_cycle_idx),
        )
        selected_name = curobo_wrapper.normalize_object_name(selected_name)
        if selected_name is None:
            print("[prefetch] next-cycle plan prefetch skipped: no remaining target after predicted current place")
            record_start(
                "NO_TARGET",
                target_pool=list(target_pool or []),
                target_candidates=list(target_candidates or []),
            )
            return
        selected_obstacles = targeted._derive_cycle_obstacle_names(
            self.base_args,
            int(next_cycle_idx),
            selected_name,
            self.cycle_object_sequence,
            cached_scene_names,
        )
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                print(
                    f"[prefetch] previous next-cycle prefetch is still running; "
                    f"not starting another one for {selected_name}"
                )
                record_start(
                    "PREVIOUS_RUNNING",
                    target_name=selected_name,
                    target_pool=list(target_pool or []),
                    target_candidates=list(target_candidates or []),
                )
                return
            self._reserved = {
                "cycle_idx": int(next_cycle_idx),
                "target_name": selected_name,
                "target_pool": list(target_pool or []),
                "target_candidates": list(target_candidates or []),
            }
            self._result = None
            self._thread = threading.Thread(
                target=self._worker,
                name=f"next-cycle-prefetch-{selected_name}",
                kwargs={
                    "target_name": selected_name,
                    "next_cycle_idx": int(next_cycle_idx),
                    "selected_obstacles": list(selected_obstacles),
                    "scene_snapshot": scene_snapshot,
                    "place_state_snapshot": place_state_snapshot,
                    "predicted_start_q": predicted_next_start_q.copy(),
                    "placed_object_name": curobo_wrapper.normalize_object_name(getattr(current_args, "object_name", None)),
                    "placed_T": placed_T.copy(),
                },
                daemon=True,
            )
            self._thread.start()
        record_start(
            "STARTED",
            target_name=selected_name,
            success=True,
            target_pool=list(target_pool or []),
            target_candidates=list(target_candidates or []),
            selected_obstacles=list(selected_obstacles),
        )
        print(
            f"[prefetch] started background plan for cycle {int(next_cycle_idx)} target={selected_name}; "
            f"reserved next target from candidates={list(target_candidates or [])}"
        )

    def _worker(
        self,
        *,
        target_name: str,
        next_cycle_idx: int,
        selected_obstacles: list[str],
        scene_snapshot: dict,
        place_state_snapshot: dict,
        predicted_start_q: np.ndarray,
        placed_object_name: str | None,
        placed_T: np.ndarray,
    ) -> None:
        env = None
        status = "FAILED"
        result = None
        error_text = None
        started = time.perf_counter()
        try:
            with _profile_record_context(
                is_prefetch=True,
                prefetch_cycle_idx=int(next_cycle_idx),
                prefetch_target_name=target_name,
            ), _suspend_profile_counters_for_thread():
                prefetch_args, spec = targeted.base.make_cycle_args(self.base_args, target_name)
                prefetch_args._targeted_place_state_cache = place_state_snapshot
                prefetch_args.selected_obstacle_object_names = list(selected_obstacles)
                prefetch_args.required_scene_object_names = list(selected_obstacles)
                prefetch_args.execute_real = False
                prefetch_args.auto_execute = True
                prefetch_args.render_mode = "rgb_array"
                prefetch_args.planning_profile_enabled = bool(
                    getattr(self.base_args, "planning_profile_enabled", True)
                )
                prefetch_args.planning_profile_jsonl = getattr(self.base_args, "planning_profile_jsonl", None)
                prefetch_args.planning_profile_dir = getattr(
                    self.base_args,
                    "planning_profile_dir",
                    "planning_profile_logs",
                )
                prefetch_args.next_cycle_plan_prefetch = False
                prefetch_args.single_confirm_per_object = False
                prefetch_args._planning_prefetch_capture_only = True
                prefetch_args._curobo_planner_cache_namespace = f"prefetch_{target_name}"
                prefetch_args._targeted_place_state_cache = place_state_snapshot
                print(
                    f"[prefetch] worker planning cycle {int(next_cycle_idx)} target={target_name} "
                    f"mesh={getattr(prefetch_args, 'mesh_file', '')}"
                )
                with _profile_stage(prefetch_args, "next_cycle_prefetch_create_demo") as prof:
                    env, demo = self.create_demo_func(
                        prefetch_args,
                        self.bridge_mod,
                        self.planner_mod,
                        scene_capture_cache=scene_snapshot,
                    )
                    prof["success"] = True
                    prof["status"] = "Success"
                with _profile_stage(
                    prefetch_args,
                    "next_cycle_prefetch_activate_object",
                    selected_obstacle_count=len(list(selected_obstacles or [])),
                ) as prof:
                    _single_scene_build_actor_registry(demo, self.base_args, scene_snapshot, target_name, prefetch_args)
                    activated = _single_scene_activate_object(
                        demo,
                        self.bridge_mod,
                        self.base_args,
                        prefetch_args,
                        target_name,
                        selected_obstacles,
                        scene_snapshot,
                    )
                    prof["success"] = bool(activated)
                    prof["status"] = "Success" if activated else "ACTIVATE_FAIL"
                if not activated:
                    status = "ACTIVATE_FAIL"
                else:
                    targeted.base.sync_demo_arm_qpos(demo, predicted_start_q)
                    with _profile_stage(prefetch_args, "next_cycle_prefetch_run_episode") as prof:
                        ok = run_targeted_place_episode_curobo_direct(
                            demo,
                            self.bridge_mod,
                            None,
                            prefetch_args,
                            scene_snapshot,
                            place_state_snapshot,
                        )
                        result = getattr(prefetch_args, "_planning_prefetch_result", None)
                        status = "Success" if ok and isinstance(result, dict) else "PLAN_FAIL"
                        prof["success"] = status == "Success"
                        prof["status"] = status
                        prof["plan_available"] = isinstance(result, dict)
        except Exception:
            status = "EXCEPTION"
            error_text = traceback.format_exc()
            print("[prefetch] worker failed:\n" + error_text)
        finally:
            if env is not None:
                try:
                    targeted.base.close_env_quietly(env)
                except Exception:
                    pass
                gc.collect()
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        payload = {
            "cycle_idx": int(next_cycle_idx),
            "target_name": target_name,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "predicted_start_q": predicted_start_q.copy(),
            "placed_object_name": placed_object_name,
            "placed_T": placed_T.copy(),
            "plan": result if isinstance(result, dict) else None,
            "error_text": error_text,
        }
        with self._lock:
            self._result = payload
        _record_prefetch_profile(
            self.base_args,
            target_name,
            "next_cycle_prefetch_worker",
            success=status == "Success",
            status=status,
            cycle_idx=int(next_cycle_idx),
            elapsed_ms=elapsed_ms,
            selected_obstacle_count=len(list(selected_obstacles or [])),
            plan_available=isinstance(result, dict),
            error_text=None if error_text is None else str(error_text)[-6000:],
        )
        print(f"[prefetch] worker finished target={target_name} status={status} elapsed_ms={elapsed_ms:.1f}")

    def consume_for_current_cycle(self, args, target_name: str, start_q, scene_capture_cache) -> dict | None:
        target_name = curobo_wrapper.normalize_object_name(target_name)
        cycle_idx = int(getattr(args, "_single_scene_cycle_idx", 0) or 0)
        wait_timeout = float(max(getattr(args, "next_cycle_prefetch_wait_timeout", 30.0), 0.0))

        def record_consume(status: str, *, success: bool = False, **fields) -> None:
            _record_prefetch_profile(
                args,
                target_name,
                "next_cycle_prefetch_consume",
                success=success,
                status=status,
                cycle_idx=cycle_idx,
                wait_timeout=wait_timeout,
                **fields,
            )

        with self._lock:
            thread = self._thread
            reserved = dict(self._reserved or {})
        reserved_match = int(reserved.get("cycle_idx", -1)) == cycle_idx and reserved.get("target_name") == target_name
        if reserved_match:
            if thread is not None and thread.is_alive() and wait_timeout > 0:
                print(f"[prefetch] waiting up to {wait_timeout:.2f}s for reserved plan target={target_name}")
                thread.join(wait_timeout)
        thread_alive = bool(thread is not None and thread.is_alive())
        with self._lock:
            payload = dict(self._result or {})
        if int(payload.get("cycle_idx", -1)) != cycle_idx or payload.get("target_name") != target_name:
            record_consume(
                "NOT_READY" if reserved_match else "NO_RESERVED_RESULT",
                reserved_match=reserved_match,
                reserved_cycle_idx=reserved.get("cycle_idx"),
                reserved_target=reserved.get("target_name"),
                result_cycle_idx=payload.get("cycle_idx"),
                result_target=payload.get("target_name"),
                thread_alive=thread_alive,
            )
            return None
        if payload.get("status") != "Success" or not isinstance(payload.get("plan"), dict):
            print(f"[prefetch] reserved plan for {target_name} unavailable: status={payload.get('status')}")
            record_consume(
                "WORKER_FAIL",
                worker_status=payload.get("status"),
                worker_elapsed_ms=payload.get("elapsed_ms"),
                worker_error_text=None if payload.get("error_text") is None else str(payload.get("error_text"))[-6000:],
                thread_alive=thread_alive,
            )
            return None
        current_q = _q7_or_none(start_q)
        predicted_q = _q7_or_none(payload.get("predicted_start_q"))
        if current_q is None or predicted_q is None:
            record_consume(
                "START_Q_UNAVAILABLE",
                current_q_available=current_q is not None,
                predicted_q_available=predicted_q is not None,
                worker_elapsed_ms=payload.get("elapsed_ms"),
            )
            return None
        q_delta = float(np.max(np.abs(current_q - predicted_q)))
        q_tol = float(max(getattr(args, "next_cycle_prefetch_start_q_tolerance", 0.08), 0.0))
        if q_delta > q_tol:
            print(f"[prefetch] rejected plan for {target_name}: start_q_delta={q_delta:.4f} > tol={q_tol:.4f}")
            record_consume(
                "START_Q_MISMATCH",
                q_delta=q_delta,
                q_tol=q_tol,
                worker_elapsed_ms=payload.get("elapsed_ms"),
            )
            return None
        placed_name = curobo_wrapper.normalize_object_name(payload.get("placed_object_name"))
        predicted_T = payload.get("placed_T")
        pos_delta = None
        pos_tol = None
        if placed_name is not None and predicted_T is not None:
            entry = _single_scene_cache_entry(scene_capture_cache, placed_name)
            actual_T = None if not isinstance(entry, dict) else entry.get("T_world_obj")
            if actual_T is not None:
                try:
                    predicted_T = np.asarray(predicted_T, dtype=np.float32).reshape(4, 4)
                    actual_T = np.asarray(actual_T, dtype=np.float32).reshape(4, 4)
                    pos_delta = float(np.linalg.norm(predicted_T[:3, 3] - actual_T[:3, 3]))
                    pos_tol = float(max(getattr(args, "next_cycle_prefetch_scene_pos_tolerance_m", 0.05), 0.0))
                    if pos_delta > pos_tol:
                        print(
                            f"[prefetch] rejected plan for {target_name}: placed {placed_name} "
                            f"pos_delta={pos_delta:.4f}m > tol={pos_tol:.4f}m"
                        )
                        record_consume(
                            "SCENE_POSE_MISMATCH",
                            placed_object_name=placed_name,
                            pos_delta=pos_delta,
                            pos_tol=pos_tol,
                            q_delta=q_delta,
                            q_tol=q_tol,
                            worker_elapsed_ms=payload.get("elapsed_ms"),
                        )
                        return None
                except Exception:
                    record_consume(
                        "SCENE_POSE_COMPARE_ERROR",
                        placed_object_name=placed_name,
                        q_delta=q_delta,
                        q_tol=q_tol,
                        worker_elapsed_ms=payload.get("elapsed_ms"),
                    )
                    return None
        plan = dict(payload["plan"])
        with self._lock:
            self._result = None
            self._reserved = None
        record_consume(
            "HIT",
            success=True,
            q_delta=q_delta,
            q_tol=q_tol,
            placed_object_name=placed_name,
            pos_delta=pos_delta,
            pos_tol=pos_tol,
            worker_elapsed_ms=payload.get("elapsed_ms"),
        )
        print(
            f"[prefetch] using cached plan for {target_name}: "
            f"elapsed_ms={float(payload.get('elapsed_ms', 0.0) or 0.0):.1f}, start_q_delta={q_delta:.4f}"
        )
        return plan

    def shutdown(self, timeout: float = 0.1) -> None:
        with self._lock:
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(float(max(timeout, 0.0)))


def _estimate_real_waypoint_stream_duration_s(q_start, q_path, args) -> float:
    points = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in list(q_path or [])]
    if not points:
        return 0.0
    hz = float(max(getattr(args, "real_control_hz", 10.0), 1e-3))
    max_delta = float(max(getattr(args, "real_max_delta_per_step", 0.03), 1e-6))
    hold_steps = int(max(getattr(args, "real_hold_steps", 0), 0))
    prev = _q7_or_none(q_start)
    if prev is None:
        prev = points[0]
    total_steps = 0
    for q in points:
        delta = float(np.max(np.abs(q - prev)))
        if delta > 1e-6:
            total_steps += max(1, int(np.ceil(delta / max_delta)))
        prev = q
    if total_steps <= 0:
        total_steps = 1
    total_steps += hold_steps
    return float(total_steps) / hz


def _dry_run_motion_window_enabled(args) -> bool:
    if bool(getattr(args, "execute_real", False)):
        return False
    if bool(getattr(args, "_planning_prefetch_capture_only", False)):
        return False
    scale = float(max(getattr(args, "dry_run_motion_window_scale", 0.0), 0.0))
    return bool(scale > 0.0)


def _dry_run_motion_window_duration_s(q_start, q_path, args) -> float:
    scale = float(max(getattr(args, "dry_run_motion_window_scale", 0.0), 0.0))
    duration_s = _estimate_real_waypoint_stream_duration_s(q_start, q_path, args) * scale
    if list(q_path or []):
        duration_s = max(duration_s, float(max(getattr(args, "dry_run_motion_window_min_s", 0.0), 0.0)))
    max_s = float(getattr(args, "dry_run_motion_window_max_s", 30.0) or 0.0)
    if max_s > 0.0:
        duration_s = min(duration_s, max_s)
    return float(max(duration_s, 0.0))


def _render_dry_run_motion_frame(demo, bridge_mod, args) -> None:
    if getattr(args, "render_mode", None) != "human":
        return
    try:
        bridge_mod.render_preview(demo.env, repeats=1)
    except Exception:
        pass


def _play_dry_run_motion_window(demo, bridge_mod, label: str, q_start, q_path, args) -> None:
    q_points = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in list(q_path or [])]
    if not q_points:
        return
    duration_s = _dry_run_motion_window_duration_s(q_start, q_points, args)
    q0 = _q7_or_none(q_start)
    if q0 is None:
        q0 = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    points = [q0.copy()]
    for q in q_points:
        if not np.allclose(points[-1], q, atol=1e-6, rtol=0.0):
            points.append(q.copy())
    if len(points) == 1:
        targeted.base.sync_demo_arm_qpos(demo, points[-1])
        _render_dry_run_motion_frame(demo, bridge_mod, args)
        return
    if duration_s <= 1e-6:
        targeted.base.sync_demo_arm_qpos(demo, points[-1])
        _render_dry_run_motion_frame(demo, bridge_mod, args)
        return

    fps = float(np.clip(float(getattr(args, "real_control_hz", 10.0) or 10.0), 5.0, 30.0))
    frame_count = max(len(points), int(np.ceil(duration_s * fps)) + 1)
    frame_count = min(frame_count, max(2, int(max(duration_s * 60.0, 2.0))))
    seg_len = [float(np.max(np.abs(points[i + 1] - points[i]))) for i in range(len(points) - 1)]
    total = float(sum(seg_len))
    if total <= 1e-9:
        targeted.base.sync_demo_arm_qpos(demo, points[-1])
        _render_dry_run_motion_frame(demo, bridge_mod, args)
        return
    cumulative = np.cumsum([0.0] + seg_len)
    print(
        f"[dry-run] playing motion window for {label}: "
        f"{duration_s:.2f}s, frames={frame_count}, waypoints={len(q_points)}"
    )
    start_t = time.perf_counter()
    for frame_idx in range(frame_count):
        progress = total * float(frame_idx) / float(max(frame_count - 1, 1))
        seg_idx = int(np.searchsorted(cumulative, progress, side="right") - 1)
        seg_idx = max(0, min(seg_idx, len(points) - 2))
        denom = max(float(cumulative[seg_idx + 1] - cumulative[seg_idx]), 1e-9)
        alpha = float((progress - cumulative[seg_idx]) / denom)
        q = (1.0 - alpha) * points[seg_idx] + alpha * points[seg_idx + 1]
        targeted.base.sync_demo_arm_qpos(demo, q.astype(np.float32))
        _render_dry_run_motion_frame(demo, bridge_mod, args)
        next_t = start_t + duration_s * float(frame_idx + 1) / float(max(frame_count - 1, 1))
        sleep_s = next_t - time.perf_counter()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
    targeted.base.sync_demo_arm_qpos(demo, points[-1])


def _install_dry_run_motion_window_wrappers() -> None:
    if bool(getattr(targeted.base, "_direct_pre_place_dry_run_motion_window_wrapped", False)):
        return
    original_pose_stage = targeted.base.execute_pose_path_stage

    def _execute_pose_path_stage_with_motion_window(
        demo,
        bridge_mod,
        real_exec,
        label,
        pose,
        q_path,
        gripper_pos,
        args,
        *extra_args,
        **kwargs,
    ):
        if real_exec is None and _dry_run_motion_window_enabled(args):
            q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in list(q_path or [])]
            if not q_path:
                q_current = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
                print(f"[planner] {label} is a zero-length pose path; skipping execution")
                return True, q_current
            skip_confirmation = bool(kwargs.get("skip_confirmation", False))
            if not skip_confirmation and not targeted.base.confirm_planned_motion_or_skip(
                demo,
                bridge_mod,
                label,
                pose,
                q_path[-1],
                args,
                q_preview_path=q_path,
            ):
                print(f"[abort] user cancelled before executing {label}")
                return False, None
            if skip_confirmation:
                print(f"[planner] auto-executing {label} without separate preview/confirmation")
            q_start = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
            print(f"[dry-run] executing {label} in simulation with preserved motion window")
            _play_dry_run_motion_window(demo, bridge_mod, str(label), q_start, q_path, args)
            return True, q_path[-1]
        q_start = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
        ok, q_sent = original_pose_stage(
            demo,
            bridge_mod,
            real_exec,
            label,
            pose,
            q_path,
            gripper_pos,
            args,
            *extra_args,
            **kwargs,
        )
        if ok and real_exec is None and _dry_run_motion_window_enabled(args):
            _play_dry_run_motion_window(demo, bridge_mod, str(label), q_start, q_path, args)
        return ok, q_sent

    targeted.base.execute_pose_path_stage = _execute_pose_path_stage_with_motion_window

    original_joint_stage = getattr(targeted.base, "execute_joint_path_stage", None)
    if callable(original_joint_stage):

        def _execute_joint_path_stage_with_motion_window(
            demo,
            bridge_mod,
            real_exec,
            label,
            q_path,
            gripper_pos,
            args,
            *extra_args,
            **kwargs,
        ):
            if real_exec is None and _dry_run_motion_window_enabled(args):
                q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in list(q_path or [])]
                if not q_path:
                    q_current = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
                    print(f"[planner] {label} is a zero-length joint path; skipping execution")
                    return True, q_current
                if not targeted.base.confirm_joint_path_motion(demo, bridge_mod, label, q_path, args):
                    print(f"[abort] user cancelled before executing {label}")
                    return False, None
                q_start = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
                print(f"[dry-run] executing {label} in simulation with preserved motion window")
                _play_dry_run_motion_window(demo, bridge_mod, str(label), q_start, q_path, args)
                return True, q_path[-1]
            q_start = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
            ok, q_sent = original_joint_stage(
                demo,
                bridge_mod,
                real_exec,
                label,
                q_path,
                gripper_pos,
                args,
                *extra_args,
                **kwargs,
            )
            if ok and real_exec is None and _dry_run_motion_window_enabled(args):
                _play_dry_run_motion_window(demo, bridge_mod, str(label), q_start, q_path, args)
            return ok, q_sent

        targeted.base.execute_joint_path_stage = _execute_joint_path_stage_with_motion_window

    targeted.base._direct_pre_place_dry_run_motion_window_wrapped = True


def _run_single_scene_main(create_demo_func) -> None:
    args = parse_args()
    targeted.maybe_print_and_exit_place_rules(args)
    targeted.base.maybe_print_and_exit_object_specs(args)
    if int(args.repeat_count) < 1:
        raise ValueError("--repeat-count must be >= 1")

    base_args = targeted.argparse.Namespace(**vars(args).copy())
    base_args.object_name = targeted.base.resolve_object_spec_name(base_args.object_name) if base_args.object_name else None
    if base_args.selected_obstacle_object_names is not None:
        base_args.selected_obstacle_object_names = targeted.base.resolve_object_spec_name_list(base_args.selected_obstacle_object_names)
    if base_args.tracked_scene_object_names is not None:
        base_args.tracked_scene_object_names = targeted.base.resolve_object_spec_name_list(base_args.tracked_scene_object_names)

    cycle_object_sequence = (
        targeted.base.resolve_object_spec_name_list(base_args.cycle_object_names)
        if base_args.cycle_object_names
        else []
    )
    if cycle_object_sequence:
        targeted._validate_cycle_sources_have_place_rules(cycle_object_sequence)
    if base_args.object_name is not None:
        targeted._validate_cycle_sources_have_place_rules([base_args.object_name])
    if cycle_object_sequence and not base_args.repeat_forever:
        base_args.repeat_count = max(int(base_args.repeat_count), len(cycle_object_sequence))
    target_random_seed = getattr(base_args, "cycle_target_random_seed", None)
    if str(getattr(base_args, "target_selection_order", "risk_aware")) == "random":
        if target_random_seed is None:
            target_random_seed = int(time.time_ns() % (2**32))
            base_args.cycle_target_random_seed = int(target_random_seed)
            args.cycle_target_random_seed = int(target_random_seed)
        random.seed(int(target_random_seed))

    bridge_mod = targeted.base.load_module_from_path("jiaobang_fp_bridge_targeted", args.bridge_script_path)
    planner_mod = targeted.base.load_module_from_path("jiaobang_planner_impl_targeted", args.pick_script_path)

    print(f"Using bridge script: {Path(args.bridge_script_path).resolve()}")
    print(f"Using planner script: {Path(args.pick_script_path).resolve()}")
    print(f"Using camera extrinsic from: {args.camera_extrinsic_opencv_path}")
    print("[single_scene] scene lifecycle: create once, then switch active object actors in-place")
    if args.repeat_forever:
        print("Repeat mode: forever")
    else:
        print(f"Repeat mode: {base_args.repeat_count} cycle(s)")
    print(f"Target selection order: {getattr(base_args, 'target_selection_order', 'risk_aware')}")
    if target_random_seed is not None:
        print(f"Target selection random seed: {int(target_random_seed)}")
    if cycle_object_sequence:
        print(f"Planned cycle sequence: {cycle_object_sequence}")
    elif base_args.object_name is not None:
        print(f"Initial target object: {base_args.object_name}")
    print("Targeted place rules:")
    print(targeted.describe_place_rules() or "(none)")

    real_exec = None
    env = None
    demo = None
    final_ok = True
    cycle_idx = 0
    if not bool(getattr(args, "reuse_foundationpose_scene_across_cycles", True)):
        print("[single_scene] overriding --no-reuse-foundationpose-scene-across-cycles; single-scene mode needs the shared scene cache")
    scene_capture_cache: dict | None = {}
    place_state_cache: dict = {"used_slots_by_target": {}}
    failed_targets_this_cycle: set[str] = set()
    deferred_failed_targets: set[str] = set()
    prefetch_manager = (
        _NextCyclePlanPrefetchManager(create_demo_func, bridge_mod, planner_mod, base_args, cycle_object_sequence)
        if bool(getattr(base_args, "next_cycle_plan_prefetch", True))
        else None
    )
    try:
        if args.execute_real:
            real_exec = targeted.base.RealmanJointExecutor(args)
            if args.reset_real_before_start:
                print("\n[real robot pre-reset]")
                if not targeted.base.confirm_simple_action(
                    "reset the real robot to its hardware home pose before FoundationPose initialization",
                    args,
                ):
                    print("[abort] user cancelled before the pre-FoundationPose real robot reset")
                    return
                real_exec.reset_robot(gripper_pos=args.real_gripper_open)

        while True:
            cycle_idx += 1
            ok = False
            cached_scene_names = targeted.base.list_cached_scene_object_names(scene_capture_cache)
            available_rule_names = targeted._list_cached_unplaced_rule_names(scene_capture_cache)
            reserved_name = (
                prefetch_manager.reserved_target_for_cycle(
                    cycle_idx,
                    scene_capture_cache,
                    failed_targets_this_cycle,
                    deferred_failed_targets,
                )
                if prefetch_manager is not None
                else None
            )
            if reserved_name is not None:
                target_pool = list(available_rule_names or cycle_object_sequence or [reserved_name])
                target_candidates = [reserved_name]
                selected_name = reserved_name
                print(f"[prefetch] using reserved target for cycle {cycle_idx}: {selected_name}")
            else:
                selected_name, target_pool, target_candidates = targeted._select_random_cycle_target(
                    base_args,
                    cycle_object_sequence,
                    scene_capture_cache,
                    available_rule_names,
                    failed_targets_this_cycle,
                    deferred_failed_targets,
                    cycle_idx,
                )
            if selected_name is None:
                final_ok = False
                if target_pool and failed_targets_this_cycle:
                    print(
                        f"[abort] cycle {cycle_idx}: all selectable targets failed in this cycle: "
                        f"{sorted(failed_targets_this_cycle & set(target_pool))}"
                    )
                else:
                    print(f"[abort] cycle {cycle_idx}: no selectable target object remains")
                break

            if failed_targets_this_cycle:
                print(
                    f"\n[cycle {cycle_idx}] target pool after failures: {target_candidates}; "
                    f"failed_this_cycle={sorted(failed_targets_this_cycle)}, "
                    f"deferred_failed={sorted(deferred_failed_targets)}"
                )
            elif deferred_failed_targets:
                print(
                    f"\n[cycle {cycle_idx}] target pool: {target_candidates}; "
                    f"deferred_failed={sorted(deferred_failed_targets)}"
                )
            else:
                print(f"\n[cycle {cycle_idx}] target pool: {target_candidates}")
            print(f"[cycle {cycle_idx}] selected target object: {selected_name}")
            rule = targeted.get_place_rule(selected_name)
            if rule is None:
                final_ok = False
                print(f"[abort] no targeted-place rule is configured for {selected_name}")
                break

            cycle_args, spec = targeted.base.make_cycle_args(base_args, selected_name)
            cycle_args._targeted_place_state_cache = place_state_cache
            cycle_args._single_scene_cycle_idx = int(cycle_idx)
            cycle_args._next_cycle_prefetch_manager = prefetch_manager
            cycle_args._next_cycle_prefetch_failed_targets_this_cycle = set(failed_targets_this_cycle)
            cycle_args._next_cycle_prefetch_deferred_failed_targets = set(deferred_failed_targets)
            selected_obstacles = targeted._derive_cycle_obstacle_names(
                base_args,
                cycle_idx,
                selected_name,
                cycle_object_sequence,
                cached_scene_names,
            )
            cycle_args.selected_obstacle_object_names = list(selected_obstacles)
            cycle_args.required_scene_object_names = list(selected_obstacles)
            print(f"\n[cycle {cycle_idx}] using object spec: {spec.name}")
            print(f"[cycle {cycle_idx}] mesh file: {cycle_args.mesh_file}")
            print(f"[cycle {cycle_idx}] simulation asset file: {cycle_args.sim_asset_file}")
            print(f"[cycle {cycle_idx}] simulation asset scale: {cycle_args.sim_asset_scale}")
            print(f"[cycle {cycle_idx}] GroundingDINO target: {cycle_args.target_object_name}")
            print(f"[cycle {cycle_idx}] targeted place: {rule.primitive} -> {rule.target_object_name}")
            print(f"[cycle {cycle_idx}] selected obstacles: {cycle_args.selected_obstacle_object_names}")
            print(f"\n================ cycle {cycle_idx} ================")

            if demo is None:
                env, demo = create_demo_func(cycle_args, bridge_mod, planner_mod, scene_capture_cache=scene_capture_cache)
                _single_scene_build_actor_registry(demo, base_args, scene_capture_cache, selected_name, cycle_args)
                if not _single_scene_activate_object(
                    demo,
                    bridge_mod,
                    base_args,
                    cycle_args,
                    selected_name,
                    selected_obstacles,
                    scene_capture_cache,
                ):
                    final_ok = False
                    break
            else:
                if not _single_scene_activate_object(
                    demo,
                    bridge_mod,
                    base_args,
                    cycle_args,
                    selected_name,
                    selected_obstacles,
                    scene_capture_cache,
                ):
                    final_ok = False
                    break

            cycle_start_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7].copy()
            ok = run_targeted_place_episode_curobo_direct(
                demo,
                bridge_mod,
                real_exec,
                cycle_args,
                scene_capture_cache,
                place_state_cache,
            )
            final_sim_arm_q = None
            if ok:
                final_sim_arm_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
                print(
                    f"[cycle {cycle_idx}] final sim arm q after place cycle: "
                    f"{np.round(final_sim_arm_q, 5).tolist()}"
                )

            print(f"\ncycle {cycle_idx} success = {ok}")
            if not ok:
                if bool(getattr(base_args, "reselect_target_on_planning_failure", True)):
                    if real_exec is None:
                        _single_scene_restore_after_failed_attempt(
                            demo,
                            cycle_args,
                            scene_capture_cache,
                            selected_name,
                            cycle_start_q,
                        )
                    else:
                        print(
                            "[single_scene] real execution is active; not teleporting simulation back after "
                            f"failed target={selected_name}"
                        )
                    failed_targets_this_cycle.add(selected_name)
                    deferred_failed_targets.add(selected_name)
                    print(
                        f"[cycle {cycle_idx}] planning/execution failed; "
                        "keeping the current single scene and trying a different target."
                    )
                    cycle_idx -= 1
                    continue
                final_ok = False
                break

            failed_targets_this_cycle.clear()
            deferred_failed_targets.discard(selected_name)
            targeted.base.cache_successfully_placed_object_world_pose(demo, cycle_args.object_name, cycle_args)
            registry = _single_scene_registry(demo)
            if selected_name in registry:
                registry[selected_name]["object_args"] = cycle_args
            if not base_args.repeat_forever and cycle_idx >= int(base_args.repeat_count):
                break
            if real_exec is not None:
                real_exec.set_gripper(args.real_gripper_open)
                print(
                    f"\n[cycle {cycle_idx}] keeping the current real robot pose; "
                    "the next cycle will continue in the same scene"
                )
            print(f"[cycle {cycle_idx}] ready for the next cycle in the same scene")

        print("\nfinal success =", final_ok)
    finally:
        if prefetch_manager is not None:
            prefetch_manager.shutdown(timeout=0.1)
        if env is not None:
            targeted.base.close_env_quietly(env)
            gc.collect()
        if real_exec is not None:
            real_exec.close()


def main():
    original_create_demo = targeted.base.create_demo

    def _profiled_create_demo(args, bridge_mod, planner_mod, scene_capture_cache=None):
        with _profile_stage(args, "scene_capture") as prof:
            env, demo = original_create_demo(
                args,
                bridge_mod,
                planner_mod,
                scene_capture_cache=scene_capture_cache,
            )
            if not bool(getattr(args, "skip_goal_motion", False)):
                rule = targeted.get_place_rule(args.object_name)
                if rule is not None:
                    _render_current_target_object_goal_visual(
                        demo,
                        bridge_mod,
                        scene_capture_cache,
                        getattr(args, "_targeted_place_state_cache", None) or {"used_slots_by_target": {}},
                        rule,
                        args,
                    )
                    if getattr(args, "render_mode", None) == "human":
                        try:
                            bridge_mod.render_preview(demo.env, repeats=1)
                        except Exception:
                            pass
            cached_objects = []
            if isinstance(scene_capture_cache, dict):
                cached_objects = list((scene_capture_cache.get("objects") or {}).keys())
            prof["success"] = True
            prof["status"] = "Success"
            prof["candidate_count"] = len(cached_objects)
            return env, demo

    targeted.base.create_demo = _profiled_create_demo
    targeted.parse_args = parse_args
    targeted.run_targeted_place_episode = run_targeted_place_episode_curobo_direct
    _install_dry_run_motion_window_wrappers()
    _run_single_scene_main(_profiled_create_demo)


if __name__ == "__main__":
    main()
