#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rm75_app.runtime import direct_pre_place as direct
from rm75_app.perception import sam6d_pose_provider as sam6d_provider
from rm75_app.paths import APP_ROOT, RUNTIME_DIR


DEFAULT_SAM6D_PROVIDER_SCRIPT = str(Path(sam6d_provider.__file__).resolve())
DEFAULT_SAM6D_OUTPUT_ROOT = str(RUNTIME_DIR / "sam6d_grasp_scene_runs")
_MISSING = object()


def _normalize_name(name: str | None) -> str | None:
    return direct.curobo_wrapper.normalize_object_name(name)


def _unique_normalized_names(names) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in list(names or []):
        name = _normalize_name(raw)
        if name is None or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _scene_object_names_for_sam6d(args) -> list[str]:
    explicit = getattr(args, "sam6d_object_names", None)
    if explicit:
        normalized_explicit = [normalized for normalized in (_normalize_name(name) for name in explicit) if normalized is not None]
        if normalized_explicit:
            return normalized_explicit

    names = []
    names += list(getattr(args, "cycle_object_names", []) or [])
    if getattr(args, "object_name", None):
        names.append(getattr(args, "object_name"))
    names += list(getattr(args, "selected_obstacle_object_names", []) or [])
    names += list(getattr(args, "required_scene_object_names", []) or [])
    names += list(getattr(args, "tracked_scene_object_names", []) or [])
    return _unique_normalized_names(names)


def _select_sam6d_instance(results: list[dict], object_name: str, instance_index: int = 0) -> tuple[dict, int]:
    if not results:
        raise RuntimeError(f"SAM6D produced no candidates for {object_name}")
    max_idx = len(results) - 1
    idx = int(instance_index)
    if idx < 0:
        idx = 0
    if idx > max_idx:
        print(
            f"[sam6d grasp] requested SAM3 instance out of range for {object_name}: requested={idx}, "
            f"available={len(results)}; using last"
        )
        idx = max_idx
    chosen = results[idx]
    return chosen, idx


def _selected_obstacle_names(args, target_name: str | None) -> list[str]:
    selected = []
    selected += list(getattr(args, "selected_obstacle_object_names", []) or [])
    selected += list(getattr(args, "required_scene_object_names", []) or [])
    selected += list(getattr(args, "tracked_scene_object_names", []) or [])
    target_name = _normalize_name(target_name)
    return [name for name in _unique_normalized_names(selected) if name != target_name]


def _required_sam6d_scene_names_for_current_task(args, target_name: str | None) -> list[str]:
    required = []
    target_name = _normalize_name(target_name)
    if target_name is not None:
        required.append(target_name)
    # The direct cycle runner mirrors tracked obstacles into
    # selected_obstacle_object_names/required_scene_object_names.  For SAM6D
    # strict mode those are optional collision obstacles, not hard task
    # dependencies.  Only the current target and the current place-rule
    # destination are required here; e.g. lvmukuai requires desk, while bi
    # requires bitong.
    try:
        rule = direct.targeted.get_place_rule(target_name)
        rule_target = _normalize_name(getattr(rule, "target_object_name", None))
        if rule_target is not None:
            required.append(rule_target)
    except Exception:
        pass
    return _unique_normalized_names(required)


def _copy_args_for_cache(args):
    copier = getattr(direct.targeted.base, "_copy_scene_cache_namespace", None)
    if callable(copier):
        try:
            return copier(args)
        except Exception:
            pass
    copied = {}
    for key, value in vars(args).items():
        if str(key).startswith("_") or callable(value):
            continue
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
            continue
        try:
            copied[key] = copy.deepcopy(value)
        except Exception:
            copied[key] = value
    return argparse.Namespace(**copied)


def _object_args_for_cache(args, bridge_mod, object_name: str):
    target_name = _normalize_name(getattr(args, "object_name", None))
    if object_name == target_name:
        return _copy_args_for_cache(args)
    try:
        return _copy_args_for_cache(bridge_mod._make_object_specific_args(args, object_name))
    except Exception:
        return _copy_args_for_cache(args)


def _sam6d_cache_key(args, object_names: list[str]):
    fixed_result_file = getattr(args, "sam6d_fixed_scene_result_file", None)
    return (
        "sam6d",
        None if not fixed_result_file else str(Path(fixed_result_file).expanduser().resolve()),
        str(Path(getattr(args, "sam6d_provider_script", DEFAULT_SAM6D_PROVIDER_SCRIPT)).expanduser().resolve()),
        str(Path(getattr(args, "sam6d_root", sam6d_provider.DEFAULT_SAM6D_ROOT)).expanduser().resolve()),
        str(getattr(args, "sam6d_mask_mode", "sam3_text")),
        str(getattr(args, "sam6d_frame_dir", "") or ""),
        int(getattr(args, "camera_width", 640)),
        int(getattr(args, "camera_height", 480)),
        int(getattr(args, "camera_fps", 30)),
        str(getattr(args, "camera_serial", "") or ""),
        int(getattr(args, "warmup_frames", 30)),
        str(Path(getattr(args, "camera_extrinsic_opencv_path", "") or "").expanduser()),
        bool(getattr(args, "use_direct_camera_extrinsic", False)),
        str(getattr(args, "sam3_checkpoint_path", sam6d_provider.DEFAULT_SAM3_CHECKPOINT_PATH)),
        str(getattr(args, "sam6d_post_pem_mask_refine_objects", "")),
        float(getattr(args, "sam6d_post_pem_mask_refine_trigger_px", 6.0)),
        int(getattr(args, "sam3_max_masks_per_item", 1)),
        bool(getattr(args, "sam3_full_scene_keep_multi_instances", False)),
        int(getattr(args, "sam3_instance_index", 0)),
        bool(getattr(args, "sam6d_fix_bitong_mouth_up", True)),
    )


def _load_fixed_sam6d_summary(args) -> tuple[dict, Path] | tuple[None, None]:
    result_file = getattr(args, "sam6d_fixed_scene_result_file", None)
    if not result_file:
        return None, None
    path = Path(result_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"--sam6d-fixed-scene-result-file does not exist: {path}")
    summary = json.loads(path.read_text())
    if not isinstance(summary, dict) or not isinstance(summary.get("results"), list):
        raise ValueError(f"Invalid SAM6D fixed scene result file: {path}")
    print(f"[sam6d grasp] using fixed SAM6D scene result: {path}")
    return summary, path


def _camera_to_base_for_orientation(args, T_base_cam: np.ndarray) -> np.ndarray:
    T_base_cam = np.asarray(T_base_cam, dtype=np.float32).reshape(4, 4)
    if bool(getattr(args, "use_direct_camera_extrinsic", False)):
        return T_base_cam
    return np.linalg.inv(T_base_cam).astype(np.float32)


def _maybe_fix_bitong_mouth_orientation(args, object_name: str, result: dict, T_base_cam: np.ndarray) -> dict:
    name = _normalize_name(object_name)
    if name != "bitong" or not bool(getattr(args, "sam6d_fix_bitong_mouth_up", True)):
        return result
    if result.get("T_cam_obj") is None:
        return result

    T_cam_obj = np.asarray(result["T_cam_obj"], dtype=np.float32).reshape(4, 4).copy()
    T_base_obj = _camera_to_base_for_orientation(args, T_base_cam) @ T_cam_obj
    mouth_axis_obj = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    mouth_axis_base = T_base_obj[:3, :3] @ mouth_axis_obj
    mouth_axis_z = float(mouth_axis_base[2])
    if mouth_axis_z >= 0.0:
        result["bitong_mouth_axis_base_z"] = mouth_axis_z
        result["bitong_mouth_orientation_fixed"] = False
        return result

    # holder_85x95 is modeled with its cylinder/opening axis along local +Y.
    # A local 180deg X rotation keeps the center fixed and flips +Y upward.
    flip_local = np.eye(4, dtype=np.float32)
    flip_local[:3, :3] = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    fixed_T_cam_obj = T_cam_obj @ flip_local
    fixed_axis_base = (_camera_to_base_for_orientation(args, T_base_cam) @ fixed_T_cam_obj)[:3, :3] @ mouth_axis_obj
    fixed_axis_z = float(fixed_axis_base[2])
    fixed = dict(result)
    fixed["T_cam_obj"] = fixed_T_cam_obj.tolist()
    fixed["bitong_mouth_axis_base_z_before"] = mouth_axis_z
    fixed["bitong_mouth_axis_base_z"] = fixed_axis_z
    fixed["bitong_mouth_orientation_fixed"] = True
    print(
        "[sam6d grasp] fixed bitong mouth orientation: "
        f"mouth_axis_base_z {mouth_axis_z:.3f} -> {fixed_axis_z:.3f}"
    )
    return fixed


def _provider_bool_args(args) -> list[str]:
    cmd = []
    if bool(getattr(args, "sam6d_no_pem_warmup_during_sam3", False)):
        cmd.append("--no-pem-warmup-during-sam3")
    if bool(getattr(args, "sam6d_no_post_pem_mask_refine", False)):
        cmd.append("--no-post-pem-mask-refine")
    if bool(getattr(args, "sam6d_no_full_scene_pem_visualization", False)):
        cmd.append("--no-full-scene-pem-visualization")
    if bool(getattr(args, "sam6d_no_pem_save_visualization", False)):
        cmd.append("--no-pem-save-visualization")
    if bool(getattr(args, "sam6d_confirm_segmentation", True)):
        cmd.append("--sam3-full-scene-mask-confirm")
    else:
        cmd.append("--no-sam3-full-scene-mask-confirm")
    if bool(getattr(args, "sam6d_require_full_scene_masks", True)):
        cmd.append("--sam3-require-full-scene-masks")
    else:
        cmd.append("--no-sam3-require-full-scene-masks")
    if not bool(getattr(args, "sam6d_show_segmentation_window", True)):
        cmd.append("--no-sam3-show-full-scene-mask-window")
    return cmd


def _provider_command(args, object_names: list[str]) -> list[str]:
    provider_script = Path(getattr(args, "sam6d_provider_script", DEFAULT_SAM6D_PROVIDER_SCRIPT)).expanduser().resolve()
    output_root = Path(getattr(args, "sam6d_output_root", DEFAULT_SAM6D_OUTPUT_ROOT)).expanduser()
    cmd = [
        sys.executable,
        str(provider_script),
        "--sam6d-root",
        str(Path(getattr(args, "sam6d_root", sam6d_provider.DEFAULT_SAM6D_ROOT)).expanduser()),
        "--output-root",
        str(output_root),
        "--object-names",
        *object_names,
        "--mask-mode",
        str(getattr(args, "sam6d_mask_mode", "sam3_text")),
        "--camera-width",
        str(int(getattr(args, "camera_width", 640))),
        "--camera-height",
        str(int(getattr(args, "camera_height", 480))),
        "--camera-fps",
        str(int(getattr(args, "camera_fps", 30))),
        "--warmup-frames",
        str(int(getattr(args, "warmup_frames", 30))),
        "--camera-extrinsic-opencv-path",
        str(Path(getattr(args, "camera_extrinsic_opencv_path", sam6d_provider.DEFAULT_CAMERA_EXTRINSIC_OPENCV_PATH)).expanduser()),
        "--sam3-python",
        str(Path(getattr(args, "sam3_python", sam6d_provider.DEFAULT_SAM3_PYTHON)).expanduser()),
        "--sam3-provider-script",
        str(Path(getattr(args, "sam3_provider_script", sam6d_provider.DEFAULT_SAM3_PROVIDER_SCRIPT)).expanduser()),
        "--sam3-checkpoint-path",
        str(Path(getattr(args, "sam3_checkpoint_path", sam6d_provider.DEFAULT_SAM3_CHECKPOINT_PATH)).expanduser()),
        "--sam3-resolution",
        str(int(getattr(args, "sam3_resolution", 1008))),
        "--sam3-confidence-threshold",
        str(float(getattr(args, "sam3_confidence_threshold", 0.35))),
        "--sam3-morph-kernel",
        str(int(getattr(args, "sam3_morph_kernel", 3))),
        "--sam3-max-masks-per-item",
        str(int(getattr(args, "sam3_max_masks_per_item", 1))),
        "--pem-feature-cache-root",
        str(Path(getattr(args, "sam6d_pem_feature_cache_root", RUNTIME_DIR / "sam6d_pem_feature_cache")).expanduser()),
        "--post-pem-mask-refine-objects",
        str(getattr(args, "sam6d_post_pem_mask_refine_objects", "lvmukuai,carriot,tennis")),
        "--post-pem-mask-refine-trigger-px",
        str(float(getattr(args, "sam6d_post_pem_mask_refine_trigger_px", 6.0))),
    ]
    frame_dir = getattr(args, "sam6d_frame_dir", None)
    if frame_dir:
        cmd += ["--frame-dir", str(Path(frame_dir).expanduser())]
    camera_serial = getattr(args, "camera_serial", None)
    if camera_serial:
        cmd += ["--camera-serial", str(camera_serial)]
    if getattr(args, "sam3_device", None):
        cmd += ["--sam3-device", str(getattr(args, "sam3_device"))]
    if bool(getattr(args, "use_direct_camera_extrinsic", False)):
        cmd += ["--use-direct-camera-extrinsic"]
    if bool(getattr(args, "sam3_full_scene_keep_multi_instances", False)):
        cmd += ["--sam3-full-scene-keep-multi-instances"]
    if getattr(args, "sam3_instance_index", 0) != 0:
        cmd += ["--sam3-instance-index", str(int(getattr(args, "sam3_instance_index", 0)))]
    cmd += _provider_bool_args(args)
    return cmd


def _save_set_attr(args, saved: dict[str, object], name: str, value) -> None:
    if name not in saved:
        saved[name] = getattr(args, name, _MISSING)
    setattr(args, name, value)


def _restore_saved_attrs(args, saved: dict[str, object]) -> None:
    for name, value in saved.items():
        if value is _MISSING:
            try:
                delattr(args, name)
            except AttributeError:
                pass
        else:
            setattr(args, name, value)


def _cap_int_attr(args, saved: dict[str, object], name: str, cap: int, *, floor: int = 1) -> int:
    cap = int(max(cap, floor))
    try:
        current = int(getattr(args, name))
    except Exception:
        current = cap
    value = cap if current <= 0 else min(current, cap)
    value = int(max(value, floor))
    _save_set_attr(args, saved, name, value)
    return value


def _cap_float_attr(args, saved: dict[str, object], name: str, cap: float, *, floor: float = 0.001) -> float:
    cap = float(max(cap, floor))
    try:
        current = float(getattr(args, name))
    except Exception:
        current = cap
    value = cap if current <= 0.0 else min(current, cap)
    value = float(max(value, floor))
    _save_set_attr(args, saved, name, value)
    return value


def _apply_sam6d_prefetch_fail_fast_overrides(args) -> tuple[dict[str, object], dict[str, object]]:
    saved: dict[str, object] = {}
    summary: dict[str, object] = {}

    screen_timeout = float(getattr(args, "sam6d_prefetch_screen_timeout", 2.0))
    screen_attempts = int(getattr(args, "sam6d_prefetch_screen_max_attempts", 1))
    screen_ik_seeds = int(getattr(args, "sam6d_prefetch_screen_num_ik_seeds", 32))
    screen_trajopt_seeds = int(getattr(args, "sam6d_prefetch_screen_num_trajopt_seeds", 1))
    top_pairs = int(getattr(args, "sam6d_prefetch_fast_chain_top_pairs", 1))
    place_rank_grasp_limit = int(getattr(args, "sam6d_prefetch_fast_chain_place_rank_grasp_limit", 1))
    max_grasp_candidates = int(getattr(args, "sam6d_prefetch_max_grasp_candidates", 1))
    max_pre_place_candidates = int(getattr(args, "sam6d_prefetch_max_pre_place_candidates", 1))

    summary["joint_search_screen_timeout"] = _cap_float_attr(
        args,
        saved,
        "joint_search_screen_timeout",
        screen_timeout,
    )
    summary["joint_search_screen_max_attempts"] = _cap_int_attr(
        args,
        saved,
        "joint_search_screen_max_attempts",
        screen_attempts,
    )
    summary["joint_search_screen_num_ik_seeds"] = _cap_int_attr(
        args,
        saved,
        "joint_search_screen_num_ik_seeds",
        screen_ik_seeds,
    )
    summary["joint_search_screen_num_trajopt_seeds"] = _cap_int_attr(
        args,
        saved,
        "joint_search_screen_num_trajopt_seeds",
        screen_trajopt_seeds,
    )
    summary["joint_search_fallback_timeout"] = _cap_float_attr(
        args,
        saved,
        "joint_search_fallback_timeout",
        screen_timeout,
    )
    summary["joint_search_fallback_max_attempts"] = _cap_int_attr(
        args,
        saved,
        "joint_search_fallback_max_attempts",
        screen_attempts,
    )
    summary["joint_search_fallback_num_ik_seeds"] = _cap_int_attr(
        args,
        saved,
        "joint_search_fallback_num_ik_seeds",
        screen_ik_seeds,
    )
    summary["joint_search_fallback_num_trajopt_seeds"] = _cap_int_attr(
        args,
        saved,
        "joint_search_fallback_num_trajopt_seeds",
        screen_trajopt_seeds,
    )
    summary["joint_search_fallback_num_graph_seeds"] = _cap_int_attr(
        args,
        saved,
        "joint_search_fallback_num_graph_seeds",
        1,
    )
    summary["transport_prefilter_q_goal_timeout"] = _cap_float_attr(
        args,
        saved,
        "transport_prefilter_q_goal_timeout",
        min(screen_timeout, 2.0),
    )
    summary["transport_prefilter_q_goal_max_attempts"] = _cap_int_attr(
        args,
        saved,
        "transport_prefilter_q_goal_max_attempts",
        screen_attempts,
    )
    summary["transport_prefilter_q_goal_num_trajopt_seeds"] = _cap_int_attr(
        args,
        saved,
        "transport_prefilter_q_goal_num_trajopt_seeds",
        screen_trajopt_seeds,
    )
    summary["fast_chain_top_pairs"] = _cap_int_attr(args, saved, "fast_chain_top_pairs", top_pairs)
    summary["fast_chain_place_rank_grasp_limit"] = _cap_int_attr(
        args,
        saved,
        "fast_chain_place_rank_grasp_limit",
        place_rank_grasp_limit,
    )
    summary["fixed_tabletop_fast_chain_place_rank_grasp_limit"] = _cap_int_attr(
        args,
        saved,
        "fixed_tabletop_fast_chain_place_rank_grasp_limit",
        max(1, place_rank_grasp_limit),
    )
    summary["joint_search_max_grasp_candidates"] = _cap_int_attr(
        args,
        saved,
        "joint_search_max_grasp_candidates",
        max_grasp_candidates,
    )
    summary["joint_search_max_pre_place_candidates"] = _cap_int_attr(
        args,
        saved,
        "joint_search_max_pre_place_candidates",
        max_pre_place_candidates,
    )

    bool_overrides = {
        "joint_search_fallback_enable_graph": False,
        "joint_search_primary_fallback_after_fast_ik_fail": False,
        "place_clearance_retry": False,
        "fast_chain_preselect_first_valid": True,
    }
    int_overrides = {
        "joint_search_fallback_max_pre_place_candidates": 0,
    }
    for name, value in bool_overrides.items():
        _save_set_attr(args, saved, name, value)
        summary[name] = value
    for name, value in int_overrides.items():
        _save_set_attr(args, saved, name, value)
        summary[name] = value

    _save_set_attr(args, saved, "_sam6d_prefetch_fail_fast_active", True)
    return saved, summary


def _wrap_run_targeted_place_episode_for_sam6d_prefetch_fail_fast(original_func):
    def _wrapped(*call_args, **call_kwargs):
        episode_args = call_kwargs.get("args")
        if episode_args is None and len(call_args) >= 4:
            episode_args = call_args[3]
        if not (
            episode_args is not None
            and bool(getattr(episode_args, "_planning_prefetch_capture_only", False))
            and bool(getattr(episode_args, "sam6d_prefetch_fail_fast", True))
        ):
            return original_func(*call_args, **call_kwargs)

        saved, summary = _apply_sam6d_prefetch_fail_fast_overrides(episode_args)
        target_name = _normalize_name(getattr(episode_args, "object_name", None)) or str(
            getattr(episode_args, "object_name", "?")
        )
        print(
            "[sam6d prefetch] fail-fast planning overrides active "
            f"target={target_name}: {json.dumps(summary, sort_keys=True)}"
        )
        try:
            return original_func(*call_args, **call_kwargs)
        finally:
            _restore_saved_attrs(episode_args, saved)

    return _wrapped


def _extract_result_path(stdout: str, stderr: str, object_count: int) -> Path:
    text = "\n".join([stdout or "", stderr or ""])
    patterns = (
        r"\[sam6d-gdino\] full scene result:\s*(.+full_scene_pose_results\.json)",
        r"\[sam6d-gdino\] result:\s*(.+sam6d_pose_result\.json)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            path = Path(match.group(1).strip()).expanduser()
            if path.exists():
                return path
    raise RuntimeError(
        "SAM6D provider finished but no result JSON path was found in stdout/stderr. "
        f"object_count={object_count}"
    )


def _load_provider_summary(result_path: Path) -> dict:
    data = json.loads(Path(result_path).read_text())
    if "results" in data:
        return data
    return {
        "scene_dir": str(Path(result_path).parent),
        "object_count": 1,
        "ok_count": 1 if bool(data.get("ok", True)) else 0,
        "results": [data],
    }


def _provider_needs_live_stdio(args, object_names: list[str]) -> bool:
    return (
        len(object_names) > 1
        and str(getattr(args, "sam6d_mask_mode", "sam3_text")) == "sam3_text"
        and bool(getattr(args, "sam6d_confirm_segmentation", True))
    )


def _find_latest_provider_result(args, started_at: float, object_count: int) -> Path:
    output_root = Path(getattr(args, "sam6d_output_root", DEFAULT_SAM6D_OUTPUT_ROOT)).expanduser()
    names = ["full_scene_pose_results.json"] if object_count > 1 else ["sam6d_pose_result.json"]
    candidates = []
    for file_name in names:
        for path in output_root.rglob(file_name):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime >= started_at - 2.0:
                candidates.append((mtime, path))
    if not candidates:
        raise RuntimeError(f"SAM6D provider finished but no recent result JSON was found under {output_root}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _run_sam6d_provider(args, object_names: list[str]) -> tuple[dict, Path]:
    if not object_names:
        raise ValueError("SAM6D scene capture needs at least one object name")

    cmd = _provider_command(args, object_names)
    print("[sam6d grasp] running provider:", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(APP_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    timeout = float(getattr(args, "sam6d_provider_timeout_s", 240.0) or 240.0)
    started_at = time.time()
    t0 = time.perf_counter()
    live_stdio = _provider_needs_live_stdio(args, object_names)
    if live_stdio:
        proc = subprocess.run(
            cmd,
            cwd=str(APP_ROOT),
            text=True,
            env=env,
            timeout=timeout,
        )
        stdout = ""
        stderr = ""
    else:
        proc = subprocess.run(
            cmd,
            cwd=str(APP_ROOT),
            text=True,
            capture_output=True,
            env=env,
            timeout=timeout,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        if stdout:
            print(stdout.rstrip())
        if stderr:
            print(stderr.rstrip(), file=sys.stderr)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if proc.returncode != 0:
        tail = "\n".join((stderr or stdout).splitlines()[-40:])
        detail = f"\n{tail}" if tail else ""
        raise RuntimeError(f"SAM6D provider failed with code {proc.returncode}:{detail}")

    if live_stdio:
        result_path = _find_latest_provider_result(args, started_at, len(object_names))
    else:
        result_path = _extract_result_path(stdout, stderr, len(object_names))
    summary = _load_provider_summary(result_path)
    summary["provider_result_path"] = str(result_path)
    summary["provider_elapsed_ms"] = float(elapsed_ms)
    print(
        f"[sam6d grasp] provider ok_count={summary.get('ok_count')}/{summary.get('object_count')} "
        f"elapsed_ms={elapsed_ms:.1f} result={result_path}"
    )
    return summary, result_path


def _results_by_name(summary: dict) -> dict[str, list[dict]]:
    out = {}
    for item in list(summary.get("results") or []):
        name = _normalize_name(item.get("object_name"))
        if name is None or not bool(item.get("ok", True)) or item.get("T_cam_obj") is None:
            continue
        out.setdefault(name, []).append(item)
    for candidates in out.values():
        if isinstance(candidates, list):
            candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return out


def _cache_entry_from_sam6d_result(args, bridge_mod, object_name: str, result: dict) -> dict:
    box = result.get("detection_box_xyxy")
    if box is None:
        box = np.zeros(4, dtype=np.float32)
    return {
        "object_name": object_name,
        "label": str(result.get("prompt", object_name)),
        "score": float(result.get("score", 0.0)),
        "box": np.asarray(box, dtype=np.float32).reshape(4).copy(),
        "T_cam_obj": np.asarray(result["T_cam_obj"], dtype=np.float32).reshape(4, 4).copy(),
        "object_args": _object_args_for_cache(args, bridge_mod, object_name),
        "placed": False,
        "pose_source": "sam6d",
        "sam6d_run_dir": str(result.get("run_dir", "")),
        "sam6d_mask_source": str(result.get("mask_source", "")),
        "sam6d_score": float(result.get("score", 0.0)),
        "sam6d_sam3_instance_index": int(result.get("sam3_instance_index", 0)),
        "sam6d_bitong_mouth_orientation_fixed": bool(result.get("bitong_mouth_orientation_fixed", False)),
        "sam6d_bitong_mouth_axis_base_z": float(result.get("bitong_mouth_axis_base_z", 0.0) or 0.0),
    }


def _scene_obstacle_entries_from_cache(args, cached_objects: dict, obstacle_names: list[str]) -> list[dict]:
    scene_obstacles = []
    for name in obstacle_names:
        item = cached_objects.get(name)
        if not isinstance(item, dict):
            continue
        entry = {
            "object_name": name,
            "label": str(item.get("label", name)),
            "score": float(item.get("score", 0.0)),
            "box": np.asarray(item.get("box", np.zeros(4, dtype=np.float32)), dtype=np.float32).reshape(4).copy(),
            "placed": bool(item.get("placed", False)),
            "object_args": item.get("object_args", _copy_args_for_cache(args)),
        }
        if item.get("T_world_obj") is not None:
            entry["T_world_obj"] = np.asarray(item["T_world_obj"], dtype=np.float32).reshape(4, 4).copy()
        elif item.get("T_cam_obj") is not None:
            entry["T_cam_obj"] = np.asarray(item["T_cam_obj"], dtype=np.float32).reshape(4, 4).copy()
        scene_obstacles.append(entry)
    return scene_obstacles


def capture_or_reuse_sam6d_scene(args, bridge_mod, scene_capture_cache=None):
    target_name = _normalize_name(getattr(args, "object_name", None))
    if target_name is None:
        raise ValueError("--object-name is required for SAM6D scene capture")

    object_names = _scene_object_names_for_sam6d(args)
    if target_name not in object_names:
        object_names.insert(0, target_name)
    object_names = _unique_normalized_names(object_names)
    selected_obstacles = _selected_obstacle_names(args, target_name)
    required_names = _required_sam6d_scene_names_for_current_task(args, target_name)
    required_obstacles = [name for name in required_names if name != target_name]
    cache_key = _sam6d_cache_key(args, object_names)

    if (
        bool(getattr(args, "sam6d_reuse_scene_across_cycles", True))
        and isinstance(scene_capture_cache, dict)
        and scene_capture_cache.get("key") == cache_key
    ):
        cached_objects = dict(scene_capture_cache.get("objects", {}) or {})
        if target_name in cached_objects and all(name in cached_objects for name in required_obstacles):
            print("[sam6d grasp] reusing cached SAM6D scene")
            return (
                None,
                np.asarray(scene_capture_cache["T_base_cam"], dtype=np.float32),
                np.asarray(cached_objects[target_name]["T_cam_obj"], dtype=np.float32).reshape(4, 4),
                _scene_obstacle_entries_from_cache(args, cached_objects, selected_obstacles),
            )

    if (
        bool(getattr(args, "_planning_prefetch_capture_only", False))
        and not bool(getattr(args, "sam6d_allow_prefetch_scene_recapture", False))
    ):
        cached_objects = {}
        if isinstance(scene_capture_cache, dict):
            cached_objects = dict(scene_capture_cache.get("objects", {}) or {})
        missing_required = [name for name in required_names if name not in cached_objects]
        raise RuntimeError(
            "SAM6D prefetch scene cache miss; refusing to run SAM3/SAM6D recapture in the background. "
            f"target={target_name}, missing_required={missing_required}, cached={sorted(cached_objects.keys())}"
        )

    summary, result_path = _load_fixed_sam6d_summary(args)
    if summary is None:
        with direct._CUROBO_GPU_LOCK:
            summary, result_path = _run_sam6d_provider(args, object_names)
    results = _results_by_name(summary)
    missing = [name for name in object_names if not results.get(name)]
    missing_required = [name for name in required_names if not results.get(name)]
    if target_name not in results:
        raise RuntimeError(f"SAM6D did not produce a target pose for {target_name}; missing_required={missing_required}, missing={missing}")
    if missing_required and bool(getattr(args, "sam6d_strict_scene", True)):
        raise RuntimeError(f"SAM6D strict scene is enabled and required objects are missing: {missing_required}; optional_missing={missing}")
    if missing:
        print(f"[sam6d grasp] warning: missing non-target SAM6D object(s): {missing}")

    T_base_cam = bridge_mod.load_matrix(args.camera_extrinsic_opencv_path).astype(np.float32)
    target_instance_index = int(getattr(args, "sam3_instance_index", 0))
    cached_objects = {}
    for name in object_names:
        candidates = results.get(name, [])
        if not candidates:
            continue
        request_index = target_instance_index if name == target_name else 0
        item, selected_instance_index = _select_sam6d_instance(candidates, name, request_index)
        if len(candidates) > 1:
            item["sam6d_available_instance_count"] = int(len(candidates))
            item["sam6d_selected_instance_index"] = int(selected_instance_index)
        item = _maybe_fix_bitong_mouth_orientation(args, name, item, T_base_cam)
        cached_objects[name] = _cache_entry_from_sam6d_result(args, bridge_mod, name, item)
    if isinstance(scene_capture_cache, dict):
        scene_capture_cache.clear()
        scene_capture_cache.update(
            {
                "key": cache_key,
                "fp_rt": None,
                "T_base_cam": T_base_cam.copy(),
                "objects": cached_objects,
                "sam6d_summary_path": str(result_path),
                "sam6d_scene_dir": str(summary.get("scene_dir", Path(result_path).parent)),
            }
        )

    print(
        f"[sam6d grasp] using target {target_name}: "
        f"camera translation={np.round(cached_objects[target_name]['T_cam_obj'][:3, 3], 6).tolist()}"
    )
    return (
        None,
        T_base_cam,
        cached_objects[target_name]["T_cam_obj"],
        _scene_obstacle_entries_from_cache(args, cached_objects, selected_obstacles),
    )


def _capture_single_target_sam6d(args, target_name: str) -> dict:
    relocalize_args = SimpleNamespace(**vars(args).copy())
    relocalize_args.sam6d_object_names = [target_name]
    relocalize_args.selected_obstacle_object_names = []
    relocalize_args.required_scene_object_names = []
    relocalize_args.tracked_scene_object_names = []
    relocalize_args.cycle_object_names = []
    relocalize_args.sam6d_reuse_scene_across_cycles = False
    with direct._CUROBO_GPU_LOCK:
        summary, _ = _run_sam6d_provider(relocalize_args, [target_name])
    results = _results_by_name(summary)
    candidates = results.get(target_name, [])
    if not candidates:
        raise RuntimeError(f"SAM6D did not return target-only pose for {target_name}")
    result, _ = _select_sam6d_instance(candidates, target_name, int(getattr(args, "sam3_instance_index", 0)))
    return result


def relocalize_active_target_after_empty_grasp_sam6d(demo, bridge_mod, args, scene_capture_cache=None) -> bool:
    target_name = _normalize_name(getattr(args, "object_name", None))
    with direct._profile_stage(args, "empty_grasp_target_relocalize", target_name=target_name, backend="sam6d") as prof:
        if not bool(getattr(args, "empty_grasp_relocalize_target", True)):
            prof["success"] = False
            prof["status"] = "DISABLED"
            return False
        if target_name is None:
            prof["success"] = False
            prof["status"] = "NO_TARGET"
            return False
        try:
            result = _capture_single_target_sam6d(args, target_name)
            T_cam_obj = np.asarray(result["T_cam_obj"], dtype=np.float32).reshape(4, 4)
            T_base_cam = getattr(demo, "T_base_cam", None)
            if T_base_cam is None:
                T_base_cam = bridge_mod.load_matrix(args.camera_extrinsic_opencv_path).astype(np.float32)
            T_world_obj = bridge_mod.map_camera_pose_to_pick_world(T_cam_obj, T_base_cam, demo.env, args)
            pose = direct._pose_from_world_matrix(np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4))
            actor = getattr(getattr(demo, "base_env", None), "obj", None)
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
            else:
                bridge_mod.apply_pose_to_pick_object(demo.env, T_world_obj)
            try:
                demo.refresh_runtime_handles(rebuild_visual=False)
            except Exception as exc:
                prof["refresh_warning"] = f"{type(exc).__name__}: {exc}"
            lift_delta = 0.0
            try:
                lift_delta = float(
                    direct.targeted.base.lift_active_object_above_table_if_needed(
                        demo,
                        args,
                        min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
                    )
                    or 0.0
                )
            except Exception as exc:
                prof["post_apply_lift_warning"] = f"{type(exc).__name__}: {exc}"
            T_world_after = direct._single_scene_actor_pose_matrix(getattr(getattr(demo, "base_env", None), "obj", None))
            if T_world_after is None:
                T_world_after = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4).copy()
            cache_updated = direct._update_relocalized_target_scene_cache(
                demo,
                scene_capture_cache,
                target_name,
                T_cam_obj,
                T_world_after,
                args,
            )
            direct._single_scene_refresh_after_pose_change(demo)
            try:
                if bool(getattr(args, "freeze_active_object_before_grasp", True)):
                    direct.targeted.base.set_pregrasp_object_freeze(demo, True)
                    direct.targeted.base.refresh_frozen_active_object_pose(demo)
            except Exception as exc:
                prof["freeze_warning"] = f"{type(exc).__name__}: {exc}"
            prof["success"] = True
            prof["status"] = "Success"
            prof["cache_updated"] = bool(cache_updated)
            prof["sam6d_score"] = float(result.get("score", 0.0))
            prof["world_translation"] = np.asarray(T_world_after, dtype=np.float32).reshape(4, 4)[:3, 3].tolist()
            prof["lift_delta_m"] = float(lift_delta)
            print(f"[empty_grasp] SAM6D relocalized active target only: {target_name}")
            return True
        except SystemExit:
            prof["success"] = False
            prof["status"] = "USER_CANCELLED"
            raise
        except Exception as exc:
            prof["success"] = False
            prof["status"] = type(exc).__name__
            prof["error"] = str(exc)
            prof["traceback_tail"] = traceback.format_exc()[-4000:]
            print(f"[empty_grasp] SAM6D target-only relocalization failed: {exc}")
            return False


def build_arg_parser():
    parser = direct.build_arg_parser()
    parser.description = (
        "SAM6D/SAM3 -> direct cuRobo grasp -> close gripper -> cuRobo transport-to-hover "
        "with contact-aware final placement. This entrypoint does not load FoundationPose."
    )
    parser.set_defaults(
        empty_grasp_relocalize_target=False,
        empty_grasp_max_relocalize_retries=0,
        foundationpose_refine_after_render=False,
        reuse_foundationpose_scene_across_cycles=True,
    )
    parser.add_argument("--sam6d-provider-script", type=str, default=DEFAULT_SAM6D_PROVIDER_SCRIPT)
    parser.add_argument("--sam6d-root", type=str, default=sam6d_provider.DEFAULT_SAM6D_ROOT)
    parser.add_argument("--sam6d-output-root", type=str, default=DEFAULT_SAM6D_OUTPUT_ROOT)
    parser.add_argument(
        "--sam6d-fixed-scene-result-file",
        type=str,
        default=None,
        help="Reuse an existing SAM6D full_scene_pose_results.json instead of running live SAM3/SAM6D scene capture.",
    )
    parser.add_argument(
        "--sam6d-mask-mode",
        choices=[
            "hybrid_depth_color",
            "fastsam_bbox",
            "sam3_text",
            "sam3_bbox",
            "sam3_text_bbox",
            "depth_plane",
            "grabcut_depth",
            "depth",
            "box",
        ],
        default="sam3_text",
    )
    parser.add_argument(
        "--sam6d-object-names",
        type=str,
        nargs="*",
        default=None,
        help="Override the object list sent to SAM6D. Default uses cycle + selected/tracked scene objects.",
    )
    parser.add_argument("--sam6d-frame-dir", type=str, default=None, help="Use an offline rgb/depth/camera frame dir for SAM6D.")
    parser.add_argument("--sam6d-provider-timeout-s", type=float, default=240.0)
    parser.add_argument("--sam6d-strict-scene", dest="sam6d_strict_scene", action="store_true", default=True)
    parser.add_argument("--no-sam6d-strict-scene", dest="sam6d_strict_scene", action="store_false")
    parser.add_argument("--sam6d-reuse-scene-across-cycles", dest="sam6d_reuse_scene_across_cycles", action="store_true", default=True)
    parser.add_argument("--no-sam6d-reuse-scene-across-cycles", dest="sam6d_reuse_scene_across_cycles", action="store_false")
    parser.add_argument(
        "--sam6d-fix-bitong-mouth-up",
        dest="sam6d_fix_bitong_mouth_up",
        action="store_true",
        default=True,
        help="For bitong, flip the SAM6D pose by 180deg if the holder opening axis points downward.",
    )
    parser.add_argument("--no-sam6d-fix-bitong-mouth-up", dest="sam6d_fix_bitong_mouth_up", action="store_false")
    parser.add_argument(
        "--sam6d-allow-prefetch-scene-recapture",
        dest="sam6d_allow_prefetch_scene_recapture",
        action="store_true",
        default=False,
        help="Allow background next-cycle prefetch to run SAM3/SAM6D if the cached scene cannot be reused.",
    )
    parser.add_argument(
        "--sam6d-prefetch-fail-fast",
        dest="sam6d_prefetch_fail_fast",
        action="store_true",
        default=True,
        help="For background next-cycle prefetch only, cap cuRobo searches and disable expensive fallback passes.",
    )
    parser.add_argument(
        "--no-sam6d-prefetch-fail-fast",
        dest="sam6d_prefetch_fail_fast",
        action="store_false",
        help="Let background next-cycle prefetch use the same full planning search as foreground execution.",
    )
    parser.add_argument(
        "--sam6d-prefetch-screen-timeout",
        type=float,
        default=2.0,
        help="Maximum cuRobo timeout used by SAM6D background prefetch screening passes.",
    )
    parser.add_argument(
        "--sam6d-prefetch-screen-max-attempts",
        type=int,
        default=1,
        help="Maximum MotionGen attempts used by SAM6D background prefetch screening passes.",
    )
    parser.add_argument(
        "--sam6d-prefetch-screen-num-ik-seeds",
        type=int,
        default=32,
        help="Maximum IK seeds used by SAM6D background prefetch screening passes.",
    )
    parser.add_argument(
        "--sam6d-prefetch-screen-num-trajopt-seeds",
        type=int,
        default=1,
        help="Maximum trajopt seeds used by SAM6D background prefetch screening passes.",
    )
    parser.add_argument(
        "--sam6d-prefetch-fast-chain-top-pairs",
        type=int,
        default=1,
        help="Maximum IK-ranked grasp/place pairs allowed into SAM6D background prefetch transport planning.",
    )
    parser.add_argument(
        "--sam6d-prefetch-fast-chain-place-rank-grasp-limit",
        type=int,
        default=1,
        help="Maximum cheap-IK grasp candidates allowed to rank place candidates during SAM6D background prefetch.",
    )
    parser.add_argument(
        "--sam6d-prefetch-max-grasp-candidates",
        type=int,
        default=1,
        help="Maximum grasp candidates expanded by SAM6D background prefetch if it falls back to primary screening.",
    )
    parser.add_argument(
        "--sam6d-prefetch-max-pre-place-candidates",
        type=int,
        default=1,
        help="Maximum pre-place candidates expanded by SAM6D background prefetch if it falls back to primary screening.",
    )
    parser.add_argument(
        "--sam6d-confirm-segmentation",
        dest="sam6d_confirm_segmentation",
        action="store_true",
        default=True,
        help="After full-scene SAM3 segmentation, show the mask overlay and wait for Enter/r/q before SAM6D PEM.",
    )
    parser.add_argument("--no-sam6d-confirm-segmentation", dest="sam6d_confirm_segmentation", action="store_false")
    parser.add_argument(
        "--sam6d-require-full-scene-masks",
        dest="sam6d_require_full_scene_masks",
        action="store_true",
        default=True,
        help="Abort before SAM6D PEM if SAM3 misses any requested full-scene object.",
    )
    parser.add_argument("--no-sam6d-require-full-scene-masks", dest="sam6d_require_full_scene_masks", action="store_false")
    parser.add_argument("--sam6d-show-segmentation-window", dest="sam6d_show_segmentation_window", action="store_true", default=True)
    parser.add_argument("--no-sam6d-show-segmentation-window", dest="sam6d_show_segmentation_window", action="store_false")
    parser.add_argument("--sam3-python", type=str, default=sam6d_provider.DEFAULT_SAM3_PYTHON)
    parser.add_argument("--sam3-provider-script", type=str, default=sam6d_provider.DEFAULT_SAM3_PROVIDER_SCRIPT)
    parser.add_argument("--sam3-checkpoint-path", type=str, default=sam6d_provider.DEFAULT_SAM3_CHECKPOINT_PATH)
    parser.add_argument("--sam3-resolution", type=int, default=1008)
    parser.add_argument("--sam3-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--sam3-morph-kernel", type=int, default=3)
    parser.add_argument("--sam3-max-masks-per-item", type=int, default=1, help="Maximum SAM3 candidates returned per item.")
    parser.add_argument("--sam3-full-scene-keep-multi-instances", dest="sam3_full_scene_keep_multi_instances", action="store_true", default=False)
    parser.add_argument("--no-sam3-full-scene-keep-multi-instances", dest="sam3_full_scene_keep_multi_instances", action="store_false")
    parser.add_argument("--sam3-instance-index", type=int, default=0, help="Index into SAM3 full-scene candidates for the current object.")
    parser.add_argument("--sam3-device", type=str, default="cuda")
    parser.add_argument(
        "--sam6d-pem-feature-cache-root",
        type=str,
        default=str(RUNTIME_DIR / "sam6d_pem_feature_cache"),
    )
    parser.add_argument("--sam6d-no-pem-warmup-during-sam3", action="store_true", default=False)
    parser.add_argument(
        "--sam6d-post-pem-mask-refine",
        dest="sam6d_no_post_pem_mask_refine",
        action="store_false",
        default=True,
        help="Enable post-PEM mask/depth translation refinement. Disabled by default.",
    )
    parser.add_argument(
        "--sam6d-no-post-pem-mask-refine",
        dest="sam6d_no_post_pem_mask_refine",
        action="store_true",
        help="Disable post-PEM mask/depth translation refinement.",
    )
    parser.add_argument("--sam6d-no-full-scene-pem-visualization", action="store_true", default=False)
    parser.add_argument("--sam6d-no-pem-save-visualization", action="store_true", default=False)
    parser.add_argument("--sam6d-post-pem-mask-refine-objects", type=str, default="lvmukuai,carriot,tennis")
    parser.add_argument("--sam6d-post-pem-mask-refine-trigger-px", type=float, default=6.0)
    return parser


def parse_args():
    args = build_arg_parser().parse_args()
    if bool(getattr(args, "skip_foundationpose", False)):
        print("[sam6d grasp] ignoring --skip-foundationpose; this entrypoint uses SAM6D camera poses, not fixed world poses")
        args.skip_foundationpose = False
    direct._configure_curobo_torch_extensions(args)
    return args


def main():
    original_capture = direct.targeted.base.capture_or_reuse_foundationpose_scene
    original_relocalize = direct._relocalize_active_target_after_empty_grasp
    original_run_episode = direct.run_targeted_place_episode_curobo_direct
    original_parse_args = direct.parse_args
    try:
        direct.targeted.base.capture_or_reuse_foundationpose_scene = capture_or_reuse_sam6d_scene
        direct._relocalize_active_target_after_empty_grasp = relocalize_active_target_after_empty_grasp_sam6d
        direct.run_targeted_place_episode_curobo_direct = _wrap_run_targeted_place_episode_for_sam6d_prefetch_fail_fast(
            original_run_episode
        )
        direct.parse_args = parse_args
        direct.main()
    finally:
        direct.targeted.base.capture_or_reuse_foundationpose_scene = original_capture
        direct._relocalize_active_target_after_empty_grasp = original_relocalize
        direct.run_targeted_place_episode_curobo_direct = original_run_episode
        direct.parse_args = original_parse_args


if __name__ == "__main__":
    main()
