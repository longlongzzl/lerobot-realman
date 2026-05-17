from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat, quat2mat


PLATE_SIZE = 0.074
PLATE_THICKNESS = 0.007
DEFAULT_ROLES = "right_wall,back_wall,left_wall,front_wall"
DEFAULT_SAM6D_ROOT = "/home/zhangzhao/PycharmProjects/SAM-6D/SAM-6D"
DEFAULT_SAM6D_PYTHON = "/home/zhangzhao/anaconda3/envs/foundationpose310/bin/python"
DEFAULT_SAM6D_PROVIDER_SCRIPT = str(Path(__file__).resolve().parents[1] / "pick_jiaobang" / "sam6d_groundingdino_pose_provider.py")
DEFAULT_CAMERA_TO_WORLD_TRANSFORM = (
    "/home/zhangzhao/Desktop/lerobot-sim2real/results/realman/realman_home/base_camera/camera_extrinsic_opencv.npy"
)
DEFAULT_OBJECT_NAMES = ",".join(["red_bricks_cube"] * 4)
DEFAULT_INSTANCE_INDICES = "0,1,2,3"
DEFAULT_SAM3_MAX_MASKS_PER_ITEM = 5
DEFAULT_SAM3_CONFIDENCE_THRESHOLD = 0.20
DEFAULT_ROBOT_BASE_POSITION = "-0.615,0.0,0.0"
DEFAULT_ROBOT_BASE_YAW_DEG = float(os.environ.get("JIMU_RM75_BASE_YAW_DEG", "90.0"))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _role_list(text: Any) -> list[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def _parse_csv_ints(raw: Any) -> list[int]:
    values: list[int] = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError:
            pass
    return values


def _parse_csv_floats(raw: Any) -> list[float]:
    values: list[float] = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError:
            pass
    return values


def _normalize_pose_name(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower())


def _load_transform_matrix(path_like: str) -> np.ndarray:
    path = Path(path_like).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"transform matrix file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    elif suffix in {".json", ".js"}:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("T_world_cam", "T_cam_world", "T_base_cam", "matrix", "transform"):
                if key in payload:
                    payload = payload[key]
                    break
        data = np.asarray(payload, dtype=np.float32)
    else:
        data = np.loadtxt(path, dtype=np.float32)
    data = np.asarray(data, dtype=np.float32)
    if data.shape != (4, 4):
        raise ValueError(f"transform matrix must be 4x4, got {data.shape} from {path}")
    return data


def _matrix_to_pose_report(matrix: np.ndarray) -> dict[str, list[float]]:
    matrix = np.asarray(matrix, dtype=np.float32).reshape(4, 4)
    q = mat2quat(matrix[:3, :3]).astype(np.float32)
    return {
        "position": matrix[:3, 3].astype(float).tolist(),
        "quaternion": q.astype(float).tolist(),
    }


def _pose_report_to_matrix(pose: dict[str, Any]) -> np.ndarray:
    p = np.asarray(pose["position"], dtype=np.float32).reshape(3)
    q = np.asarray(pose["quaternion"], dtype=np.float32).reshape(4)
    q = q / max(float(np.linalg.norm(q)), 1e-8)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = quat2mat(q).astype(np.float32)
    matrix[:3, 3] = p
    return matrix


def _quat_from_axes(x_axis: list[float], y_axis: list[float], z_axis: list[float]) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float32)
    matrix[:, 0] = np.asarray(x_axis, dtype=np.float32)
    matrix[:, 1] = np.asarray(y_axis, dtype=np.float32)
    matrix[:, 2] = np.asarray(z_axis, dtype=np.float32)
    return mat2quat(matrix).astype(np.float32)


def _open_cube_target_pose_reports() -> dict[str, dict[str, list[float]]]:
    half_size = PLATE_SIZE * 0.5
    half_thickness = PLATE_THICKNESS * 0.5
    floor_top_z = PLATE_THICKNESS
    wall_center_z = floor_top_z + half_size
    assembly_offset = np.asarray([-0.20, 0.0, 0.0], dtype=np.float32)
    specs: dict[str, tuple[list[float], list[float] | np.ndarray]] = {
        "floor": ([0.0, 0.0, half_thickness], [1.0, 0.0, 0.0, 0.0]),
        "right_wall": (
            [half_size + half_thickness, 0.0, wall_center_z],
            _quat_from_axes([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ),
        "left_wall": (
            [-half_size - half_thickness, 0.0, wall_center_z],
            _quat_from_axes([0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]),
        ),
        "back_wall": (
            [0.0, half_size + half_thickness, wall_center_z],
            _quat_from_axes([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ),
        "front_wall": (
            [0.0, -half_size - half_thickness, wall_center_z],
            _quat_from_axes([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),
        ),
    }
    result: dict[str, dict[str, list[float]]] = {}
    for role, (position, quaternion) in specs.items():
        result[role] = {
            "position": (np.asarray(position, dtype=np.float32) + assembly_offset).astype(float).tolist(),
            "quaternion": np.asarray(quaternion, dtype=np.float32).astype(float).tolist(),
        }
    return result


def _parse_sam6d_results(data: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return [item for item in data["results"] if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _sam6d_result_instance_index(item: dict[str, Any], fallback: int) -> int:
    for key in ("sam3_instance_index", "_sam3_instance_index", "sam6d_instance_index"):
        try:
            return int(item[key])
        except Exception:
            pass
    return int(fallback)


def _sam6d_cad_to_sim_local_transform(item: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    custom_rpy = _parse_csv_floats(getattr(args, "sam6d_cad_to_sim_local_rpy_deg", ""))
    mode = str(getattr(args, "sam6d_cad_to_sim_correction", "auto") or "auto").strip().lower()
    object_name = _normalize_pose_name(item.get("object_name"))
    rpy_deg: list[float]
    source = mode
    if len(custom_rpy) == 3:
        rpy_deg = [float(value) for value in custom_rpy]
        source = "custom_rpy_deg"
    elif mode in {"none", "off", "identity"}:
        rpy_deg = [0.0, 0.0, 0.0]
    elif mode in {"red_jimu_cube", "red_bricks_cube"} or (mode == "auto" and object_name == "red_bricks_cube"):
        rpy_deg = [90.0, 0.0, 0.0]
        source = "red_jimu_cube_cad_y_thick_to_actor_z_thick"
    else:
        rpy_deg = [0.0, 0.0, 0.0]
        source = "identity_auto"
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = euler2mat(*np.deg2rad(np.asarray(rpy_deg, dtype=np.float32))).astype(np.float32)
    return transform, {"source": source, "rpy_deg": rpy_deg, "object_name": object_name}


def _robot_base_world_transform(args: argparse.Namespace) -> tuple[np.ndarray | None, dict[str, Any]]:
    enabled = bool(getattr(args, "sam6d_map_through_robot_base", True))
    if not enabled:
        return None, {"enabled": False}
    position = _parse_csv_floats(getattr(args, "sam6d_robot_base_position", DEFAULT_ROBOT_BASE_POSITION))
    if len(position) != 3:
        raise ValueError("--sam6d-robot-base-position must be three comma-separated floats")
    yaw_deg = float(getattr(args, "sam6d_robot_base_yaw_deg", DEFAULT_ROBOT_BASE_YAW_DEG))
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = euler2mat(0.0, 0.0, float(np.deg2rad(yaw_deg))).astype(np.float32)
    transform[:3, 3] = np.asarray(position, dtype=np.float32)
    return transform, {
        "enabled": True,
        "position": [float(value) for value in position],
        "yaw_deg": float(yaw_deg),
    }


def _map_sam6d_camera_pose_to_world(T_cam_obj: np.ndarray, T_base_cam: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    T_cam_obj = np.asarray(T_cam_obj, dtype=np.float32).reshape(4, 4)
    T_base_cam = np.asarray(T_base_cam, dtype=np.float32).reshape(4, 4)
    if bool(getattr(args, "sam6d_use_direct_camera_extrinsic", False)):
        T_robot_obj = T_base_cam @ T_cam_obj
    else:
        T_robot_obj = np.linalg.inv(T_base_cam).astype(np.float32) @ T_cam_obj
    robot_base_T, _ = _robot_base_world_transform(args)
    if robot_base_T is not None:
        return (robot_base_T @ T_robot_obj).astype(np.float32)
    return np.asarray(T_robot_obj, dtype=np.float32)


def _sam6d_result_world_matrix(item: dict[str, Any], transform: np.ndarray | None, args: argparse.Namespace) -> np.ndarray | None:
    try:
        matrix = np.asarray(item.get("T_cam_obj"), dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if matrix.size != 16:
        return None
    raw = matrix.reshape(4, 4)
    world = raw if transform is None else _map_sam6d_camera_pose_to_world(raw, transform, args)
    local_fix, _ = _sam6d_cad_to_sim_local_transform(item, args)
    return (world @ local_fix).astype(np.float32)


def _sam6d_candidate_z_values(results: list[dict[str, Any]], transform: np.ndarray, args: argparse.Namespace) -> list[float]:
    values: list[float] = []
    for item in results:
        if not bool(item.get("ok", True)):
            continue
        matrix = _sam6d_result_world_matrix(item, transform, args)
        if matrix is not None:
            values.append(float(matrix[2, 3]))
    return values


def _sam6d_z_plausibility_score(values: list[float]) -> float:
    if not values:
        return float("inf")
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    lower = -0.08
    upper = 0.25
    outside = np.maximum(lower - arr, 0.0) + np.maximum(arr - upper, 0.0)
    return float(np.count_nonzero(outside > 0.0)) * 10.0 + float(np.mean(outside * outside)) * 100.0 + abs(float(np.median(arr)) - 0.025)


def _resolve_camera_transform(raw_transform: np.ndarray, results: list[dict[str, Any]], convention: str, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    convention = str(convention or "auto").strip().lower()
    if convention in {"camera_to_world", "cam_to_world", "camera2world", "cam2world"}:
        return raw_transform, {"convention": "camera_to_world", "inverted": False}
    if convention in {"world_to_camera", "world2camera", "world_to_cam", "world2cam"}:
        return np.linalg.inv(raw_transform).astype(np.float32), {"convention": "world_to_camera", "inverted": True}
    if convention != "auto":
        raise ValueError(f"unsupported camera transform convention: {convention}")
    inverse_transform = np.linalg.inv(raw_transform).astype(np.float32)
    as_is_z = _sam6d_candidate_z_values(results, raw_transform, args)
    inverse_z = _sam6d_candidate_z_values(results, inverse_transform, args)
    as_is_score = _sam6d_z_plausibility_score(as_is_z)
    inverse_score = _sam6d_z_plausibility_score(inverse_z)
    use_inverse = inverse_score < as_is_score
    if as_is_z and inverse_z:
        use_inverse = use_inverse or (
            min(as_is_z) > 0.25
            and -0.08 <= float(np.median(np.asarray(inverse_z, dtype=np.float32))) <= 0.25
        )
    return (
        inverse_transform if use_inverse else raw_transform,
        {
            "convention": "auto",
            "selected": "world_to_camera_inverted" if use_inverse else "camera_to_world",
            "inverted": bool(use_inverse),
            "as_is_z_values": as_is_z,
            "inverse_z_values": inverse_z,
            "as_is_score": float(as_is_score),
            "inverse_score": float(inverse_score),
        },
    )


def _local_z_world_alignment(world_matrix: np.ndarray) -> float:
    axis = np.asarray(world_matrix[:3, 2], dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-8:
        return 0.0
    return abs(float(axis[2]) / norm)


def _select_floor_entry(
    candidate_entries: list[tuple[int, int, dict[str, Any], np.ndarray]],
    args: argparse.Namespace,
) -> tuple[tuple[int, int, dict[str, Any], np.ndarray], dict[str, Any]]:
    threshold = float(getattr(args, "sam6d_floor_normal_z_threshold", 0.65))
    flat_candidates = [entry for entry in candidate_entries if _local_z_world_alignment(entry[3]) >= threshold]
    candidate_debug = [
        {
            "instance_index": int(instance_index),
            "fallback_index": int(fallback_index),
            "world_z": float(matrix[2, 3]),
            "local_z_abs_dot_world_z": float(_local_z_world_alignment(matrix)),
            "score": float(item.get("score", 0.0)),
        }
        for instance_index, fallback_index, item, matrix in candidate_entries
    ]
    if flat_candidates:
        chosen = max(flat_candidates, key=lambda entry: (_local_z_world_alignment(entry[3]), -abs(float(entry[3][2, 3]))))
        method = "floor_flat_normal_z"
    else:
        chosen = min(candidate_entries, key=lambda entry: float(entry[3][2, 3]))
        method = "floor_lowest_center_z"
    return chosen, {"method": method, "normal_z_threshold": threshold, "candidates": candidate_debug}


def _floor_pose_for_target_shift(
    targets: dict[str, dict[str, list[float]]],
    floor_matrix: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    old_floor = _pose_report_to_matrix(targets["floor"])
    raw_floor = np.asarray(floor_matrix, dtype=np.float32).reshape(4, 4)
    mode = str(getattr(args, "sam6d_floor_target_shift_mode", "xy") or "xy").strip().lower()
    adjusted = old_floor.copy()
    if mode == "none":
        pass
    elif mode == "full":
        adjusted = raw_floor.copy()
    elif mode == "xyz":
        adjusted[:3, 3] = raw_floor[:3, 3]
    elif mode == "xy_yaw":
        old_yaw = float(np.arctan2(old_floor[1, 0], old_floor[0, 0]))
        raw_yaw = float(np.arctan2(raw_floor[1, 0], raw_floor[0, 0]))
        yaw_delta = raw_yaw - old_yaw
        yaw_rotation = np.asarray(
            [
                [np.cos(yaw_delta), -np.sin(yaw_delta), 0.0],
                [np.sin(yaw_delta), np.cos(yaw_delta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        adjusted[:3, :3] = yaw_rotation @ old_floor[:3, :3]
        adjusted[:2, 3] = raw_floor[:2, 3]
    else:
        mode = "xy"
        adjusted[:2, 3] = raw_floor[:2, 3]
    return adjusted.astype(np.float32), {
        "mode": mode,
        "raw_detected_floor_translation": raw_floor[:3, 3].astype(float).tolist(),
        "adjusted_floor_translation": adjusted[:3, 3].astype(float).tolist(),
    }


def _shift_targets_to_detected_floor(
    targets: dict[str, dict[str, list[float]]],
    detected_floor_matrix: np.ndarray,
) -> dict[str, dict[str, list[float]]]:
    old_floor = _pose_report_to_matrix(targets["floor"])
    new_floor = np.asarray(detected_floor_matrix, dtype=np.float32).reshape(4, 4)
    delta = new_floor @ np.linalg.inv(old_floor)
    shifted: dict[str, dict[str, list[float]]] = {}
    for role, pose in targets.items():
        shifted[role] = _matrix_to_pose_report(delta @ _pose_report_to_matrix(pose))
    return shifted


def _correct_wall_stage_z(
    world_matrix: np.ndarray,
    *,
    role: str,
    targets: dict[str, dict[str, list[float]]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode = str(getattr(args, "sam6d_wall_stage_z_mode", "table_upright") or "table_upright").strip().lower()
    if role == "floor" or mode in {"none", "raw"}:
        return world_matrix, {"applied": False, "mode": mode}
    orientation_mode = str(getattr(args, "sam6d_wall_orientation_mode", "preserve") or "preserve").strip().lower()
    corrected = np.asarray(world_matrix, dtype=np.float32).reshape(4, 4).copy()
    floor_z = float(np.asarray(targets["floor"]["position"], dtype=np.float32).reshape(3)[2])
    table_z = floor_z - PLATE_THICKNESS * 0.5
    if mode == "target_wall":
        target_z = table_z + PLATE_THICKNESS + PLATE_SIZE * 0.5
    else:
        mode = "table_upright"
        target_z = table_z + PLATE_SIZE * 0.5
    target_z += float(getattr(args, "sam6d_wall_stage_z_offset", 0.0) or 0.0)
    raw_z = float(corrected[2, 3])
    raw_local_y_z = float(corrected[2, 1])
    raw_local_z_z = float(corrected[2, 2])
    corrected[2, 3] = target_z
    orientation_corrected = False
    if orientation_mode in {"force_upright", "upright"} and mode in {"table_upright", "target_wall"}:
        up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        thin_axis = np.asarray(corrected[:3, 2], dtype=np.float32).reshape(3)
        thin_axis[2] = 0.0
        if float(np.linalg.norm(thin_axis)) < 1e-5:
            floor_xy = np.asarray(targets["floor"]["position"], dtype=np.float32).reshape(3)[:2]
            object_xy = np.asarray(corrected[:2, 3], dtype=np.float32).reshape(2)
            thin_axis = np.asarray([object_xy[0] - floor_xy[0], object_xy[1] - floor_xy[1], 0.0], dtype=np.float32)
        if float(np.linalg.norm(thin_axis)) < 1e-5:
            thin_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        thin_axis = thin_axis / max(float(np.linalg.norm(thin_axis)), 1e-8)
        x_axis = np.cross(up, thin_axis).astype(np.float32)
        x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-8)
        corrected[:3, :3] = np.stack([x_axis, up, thin_axis], axis=1)
        orientation_corrected = True
    return corrected, {
        "applied": True,
        "mode": mode,
        "raw_z": raw_z,
        "corrected_z": float(target_z),
        "table_z": float(table_z),
        "local_z_abs_dot_world_z": float(_local_z_world_alignment(world_matrix)),
        "orientation_mode": orientation_mode,
        "raw_local_y_dot_world_z": raw_local_y_z,
        "raw_local_z_dot_world_z": raw_local_z_z,
        "orientation_corrected_to_upright": orientation_corrected,
        "corrected_local_y_dot_world_z": float(corrected[2, 1]),
        "corrected_local_z_dot_world_z": float(corrected[2, 2]),
    }


def _result_score(item: dict[str, Any]) -> float:
    try:
        return float(item.get("score", 0.0))
    except Exception:
        return 0.0


def _entry_debug(entry: tuple[int, int, dict[str, Any], np.ndarray]) -> dict[str, Any]:
    instance_index, fallback_index, item, matrix = entry
    return {
        "instance_index": int(instance_index),
        "fallback_index": int(fallback_index),
        "score": _result_score(item),
        "world_position": np.asarray(matrix[:3, 3], dtype=np.float32).astype(float).tolist(),
        "local_z_abs_dot_world_z": float(_local_z_world_alignment(matrix)),
        "box": item.get("detection_box_xyxy"),
    }


def _validate_wall_center_distances(
    role_matrices: dict[str, np.ndarray],
    roles: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    min_distance = float(getattr(args, "sam6d_min_wall_center_distance", 0.0) or 0.0)
    if min_distance <= 0.0:
        return []
    bad_pairs: list[dict[str, Any]] = []
    for left_index, left_role in enumerate(roles):
        if left_role not in role_matrices:
            continue
        left_xy = np.asarray(role_matrices[left_role][:2, 3], dtype=np.float32).reshape(2)
        for right_role in roles[left_index + 1 :]:
            if right_role not in role_matrices:
                continue
            right_xy = np.asarray(role_matrices[right_role][:2, 3], dtype=np.float32).reshape(2)
            distance = float(np.linalg.norm(left_xy - right_xy))
            if distance < min_distance:
                bad_pairs.append(
                    {
                        "roles": [left_role, right_role],
                        "distance_m": distance,
                        "min_distance_m": min_distance,
                    }
                )
    if bad_pairs:
        raise RuntimeError(f"SAM6D wall centers are implausibly close: {bad_pairs}")
    return bad_pairs


def _run_sam6d_provider(args: argparse.Namespace, object_names: list[str]) -> tuple[dict[str, Any], Path]:
    provider_script = Path(args.sam6d_provider_script).expanduser()
    if not provider_script.exists():
        raise FileNotFoundError(f"SAM6D provider script not found: {provider_script}")
    output_root = Path(args.sam6d_output_root).expanduser()
    cmd: list[str] = [
        str(Path(args.sam6d_python).expanduser()),
        str(provider_script),
        "--output-root",
        str(output_root),
        "--mask-mode",
        str(args.sam6d_mask_mode),
        "--sam6d-root",
        str(args.sam6d_root),
        "--camera-width",
        str(int(args.sam6d_camera_width)),
        "--camera-height",
        str(int(args.sam6d_camera_height)),
        "--camera-fps",
        str(int(args.sam6d_camera_fps)),
        "--warmup-frames",
        str(int(args.sam6d_warmup_frames)),
    ]
    if bool(args.sam6d_sam3_full_scene_keep_multi_instances):
        cmd.append("--sam3-full-scene-keep-multi-instances")
    if int(args.sam3_max_masks_per_item) > 0:
        cmd += ["--sam3-max-masks-per-item", str(max(1, int(args.sam3_max_masks_per_item)))]
    cmd += ["--sam3-confidence-threshold", str(float(args.sam3_confidence_threshold))]
    if str(args.sam6d_frame_dir).strip():
        cmd += ["--frame-dir", str(args.sam6d_frame_dir).strip()]
    if object_names:
        cmd += ["--object-names", *object_names]
    else:
        cmd += ["--object-names", "red_bricks_cube"]
    if int(args.sam3_instance_index) != 0:
        cmd += ["--sam3-instance-index", str(int(args.sam3_instance_index))]
    if bool(args.sam6d_confirm_full_scene_masks):
        cmd.append("--sam3-full-scene-mask-confirm")
    if bool(args.sam6d_require_full_scene_masks):
        cmd.append("--sam3-require-full-scene-masks")
    if not bool(args.sam6d_show_full_scene_mask_window):
        cmd.append("--no-sam3-show-full-scene-mask-window")
    if bool(args.sam6d_skip_pem):
        cmd.append("--skip-pem")
    if str(args.sam6d_sam3_full_scene_result_json).strip():
        cmd += ["--sam3-full-scene-result-json", str(args.sam6d_sam3_full_scene_result_json).strip()]
    cmd.append("--pem-warmup-during-sam3" if bool(args.sam6d_pem_warmup_during_sam3) else "--no-pem-warmup-during-sam3")

    started = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=not bool(args.sam6d_confirm_full_scene_masks), timeout=float(args.sam6d_provider_timeout_s))
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if stdout:
        print(stdout.rstrip())
    if stderr:
        print(stderr.rstrip(), file=sys.stderr)
    if proc.returncode != 0:
        tail = "\n".join((stderr or stdout).splitlines()[-40:])
        raise RuntimeError(f"SAM6D provider failed with code {proc.returncode}:{tail}")

    combined = f"{stdout}\n{stderr}"
    result_path: Path | None = None
    for pattern in (
        r"\[sam6d-gdino\] full scene result:\s*(.+full_scene_pose_results\.json)",
        r"\[sam6d-gdino\] batch result:\s*(.+sam6d_multi_instance_pose_results\.json)",
        r"\[sam6d-gdino\] result:\s*(.+sam6d_pose_result\.json)",
    ):
        match = re.search(pattern, combined, flags=re.IGNORECASE)
        if match:
            candidate = Path(match.group(1).strip()).expanduser()
            if candidate.exists():
                result_path = candidate
                break
    if result_path is None:
        candidates = sorted(
            (
                path
                for pattern in ("full_scene_pose_results.json", "sam6d_multi_instance_pose_results.json", "sam6d_pose_result.json")
                for path in output_root.rglob(pattern)
            ),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        candidates = [candidate for candidate in candidates if candidate.stat().st_mtime >= started - 2.0]
        if not candidates:
            raise RuntimeError(f"SAM6D provider finished but no result JSON was found under {output_root}")
        result_path = candidates[0]
    summary = json.loads(result_path.read_text(encoding="utf-8"))
    if "object_count" not in summary:
        summary = {
            "scene_dir": str(Path(result_path).parent),
            "object_count": len(summary.get("results", [])),
            "ok_count": 0,
            "results": [summary],
        }
    return summary, result_path


def _run_sam3_preview_only(args: argparse.Namespace) -> dict[str, Any]:
    roles = _role_list(args.roles)
    if not roles:
        roles = _role_list(DEFAULT_ROLES)
    object_names = _role_list(args.sam6d_object_names)
    if not object_names:
        object_names = [args.sam6d_default_object_name]
    candidate_count = len(roles) + (1 if bool(args.sam6d_auto_floor_from_candidates) else 0)
    candidate_count = max(candidate_count, 1)
    if len(object_names) == 1:
        object_names = object_names * candidate_count
    if len(object_names) < candidate_count:
        object_names.extend([object_names[-1]] * (candidate_count - len(object_names)))
    object_names = object_names[:candidate_count]

    preview_args = copy.copy(args)
    preview_args.sam6d_skip_pem = True
    preview_args.sam6d_confirm_full_scene_masks = True
    preview_args.sam6d_show_full_scene_mask_window = True
    preview_args.sam6d_sam3_full_scene_keep_multi_instances = True
    preview_args.sam3_max_masks_per_item = max(int(args.sam3_max_masks_per_item), candidate_count)
    preview_args.sam6d_pem_warmup_during_sam3 = False

    summary, result_path = _run_sam6d_provider(preview_args, object_names)
    overlay_path = None
    for parent in [result_path.parent, *result_path.parents]:
        candidate = parent / "sam3_full_scene_text" / "sam3_full_scene_masks_overlay.png"
        if candidate.exists():
            overlay_path = candidate
            break
    return {
        "result": str(result_path),
        "overlay": str(overlay_path) if overlay_path is not None else "",
        "object_names": object_names,
        "ok_count": int(summary.get("ok_count", 0)) if isinstance(summary, dict) else 0,
        "object_count": int(summary.get("object_count", 0)) if isinstance(summary, dict) else len(object_names),
    }


def _load_or_run_sam6d(args: argparse.Namespace, roles: list[str]) -> tuple[dict[str, Any], Path | None, list[str]]:
    object_names = _role_list(args.sam6d_object_names)
    if not object_names:
        object_names = [args.sam6d_default_object_name]
    if len(object_names) == 1 and len(roles) > 1:
        object_names = object_names * len(roles)
    if len(object_names) < len(roles):
        object_names.extend([object_names[-1]] * (len(roles) - len(object_names)))
    object_names = object_names[: len(roles)]
    if str(args.sam6d_result_json).strip():
        result_path = Path(args.sam6d_result_json).expanduser()
        return json.loads(result_path.read_text(encoding="utf-8")), result_path, object_names
    candidate_count = len(roles) + (1 if bool(args.sam6d_auto_floor_from_candidates) else 0)
    run_args = copy.copy(args)
    run_args.sam6d_sam3_full_scene_keep_multi_instances = True
    run_args.sam3_max_masks_per_item = max(int(args.sam3_max_masks_per_item), candidate_count)
    if bool(getattr(args, "sam3_inspect_before_pem", False)):
        run_args.sam6d_confirm_full_scene_masks = True
        run_args.sam6d_show_full_scene_mask_window = True
        run_args.sam6d_skip_pem = False
    names = [object_names[min(index, len(object_names) - 1)] for index in range(candidate_count)]
    summary, result_path = _run_sam6d_provider(run_args, names)
    return summary, result_path, object_names


def build_assembly_state(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    roles = _role_list(args.roles)
    if not roles:
        raise ValueError("--roles produced no roles")
    all_roles = list(dict.fromkeys(["floor", *roles]))
    targets = _open_cube_target_pose_reports()
    summary, result_path, object_names = _load_or_run_sam6d(args, roles)
    results = _parse_sam6d_results(summary)
    raw_transform = _load_transform_matrix(args.sam6d_camera_to_world_transform) if str(args.sam6d_camera_to_world_transform).strip() else None
    transform_info: dict[str, Any] = {"enabled": False}
    transform = raw_transform
    if raw_transform is not None:
        transform, transform_info = _resolve_camera_transform(raw_transform, results, args.sam6d_camera_transform_convention, args)
        transform_info["enabled"] = True
        transform_info["path"] = str(Path(args.sam6d_camera_to_world_transform).expanduser())
        _, robot_base_info = _robot_base_world_transform(args)
        transform_info["robot_base_mapping"] = robot_base_info
    candidate_entries: list[tuple[int, int, dict[str, Any], np.ndarray]] = []
    for fallback_index, item in enumerate(results):
        if not bool(item.get("ok", True)):
            continue
        matrix = _sam6d_result_world_matrix(item, transform, args)
        if matrix is None:
            continue
        candidate_entries.append((_sam6d_result_instance_index(item, fallback_index), fallback_index, item, matrix))
    if len(candidate_entries) < len(roles):
        raise RuntimeError(f"SAM6D produced only {len(candidate_entries)} usable poses for {len(roles)} roles")

    role_matrices: dict[str, np.ndarray] = {}
    applied: list[dict[str, Any]] = []
    assignment: dict[str, Any] = {"camera_transform": transform_info}
    instance_indices = _parse_csv_ints(args.sam6d_instance_indices)
    if bool(args.sam6d_auto_floor_from_candidates) and len(candidate_entries) >= len(roles) + 1:
        floor_entry, floor_selection = _select_floor_entry(candidate_entries, args)
        floor_instance, _, floor_item, floor_matrix = floor_entry
        floor_matrix, floor_adjustment = _floor_pose_for_target_shift(targets, floor_matrix, args)
        role_matrices["floor"] = floor_matrix
        targets = _shift_targets_to_detected_floor(targets, floor_matrix)
        assignment["floor_selection"] = floor_selection
        assignment["floor_adjustment"] = floor_adjustment
        assignment["floor_instance_index"] = int(floor_instance)
        floor_min_score = float(getattr(args, "sam6d_floor_min_score", 0.0) or 0.0)
        if floor_min_score > 0.0 and _result_score(floor_item) < floor_min_score:
            raise RuntimeError(
                f"SAM6D floor candidate score is too low: score={_result_score(floor_item):.4f} "
                f"min={floor_min_score:.4f} candidate={_entry_debug(floor_entry)}"
            )
        wall_entries = [entry for entry in candidate_entries if entry is not floor_entry]
        wall_entries.sort(key=lambda entry: (entry[0], entry[1]))
        wall_min_score = float(getattr(args, "sam6d_wall_min_score", 0.0) or 0.0)
        rejected_low_score = [entry for entry in wall_entries if _result_score(entry[2]) < wall_min_score]
        if wall_min_score > 0.0:
            wall_entries = [entry for entry in wall_entries if _result_score(entry[2]) >= wall_min_score]
        assignment["wall_min_score"] = float(wall_min_score)
        assignment["rejected_wall_candidates_low_score"] = [_entry_debug(entry) for entry in rejected_low_score]
        if len(wall_entries) < len(roles):
            raise RuntimeError(
                f"SAM6D produced only {len(wall_entries)} confident wall poses for {len(roles)} roles "
                f"after score filter min={wall_min_score:.4f}; rejected={assignment['rejected_wall_candidates_low_score']}"
            )
        for role_index, role in enumerate(roles):
            if role_index >= len(wall_entries):
                continue
            instance_index, _, item, matrix = wall_entries[role_index]
            matrix, pose_adjustment = _correct_wall_stage_z(matrix, role=role, targets=targets, args=args)
            role_matrices[role] = matrix
            applied.append(
                {
                    "role": role,
                    "instance_index": int(instance_index),
                    "object_name": str(item.get("object_name") or object_names[min(role_index, len(object_names) - 1)]),
                    "score": float(item.get("score", 0.0)),
                    "assignment": "wall_non_floor_candidate_order",
                    "pose_adjustment": pose_adjustment,
                }
            )
        applied.insert(
            0,
            {
                "role": "floor",
                "instance_index": int(floor_instance),
                "object_name": str(floor_item.get("object_name") or object_names[0]),
                "score": float(floor_item.get("score", 0.0)),
                "assignment": str(floor_selection.get("method", "floor_auto")),
                "pose_adjustment": floor_adjustment,
            },
        )
    else:
        grouped: dict[str, list[dict[str, Any]]] = {}
        wall_min_score = float(getattr(args, "sam6d_wall_min_score", 0.0) or 0.0)
        for item in results:
            if bool(item.get("ok", True)):
                if wall_min_score > 0.0 and _result_score(item) < wall_min_score:
                    continue
                grouped.setdefault(_normalize_pose_name(item.get("object_name")), []).append(item)
        for value in grouped.values():
            value.sort(key=lambda item: _sam6d_result_instance_index(item, 0))
        used_idx: dict[str, int] = {}
        for role_index, role in enumerate(roles):
            object_name = object_names[role_index] if role_index < len(object_names) else object_names[-1]
            normalized = _normalize_pose_name(object_name)
            candidates = grouped.get(normalized, [])
            if not candidates:
                continue
            if role_index < len(instance_indices):
                choose_index = max(0, instance_indices[role_index])
            else:
                choose_index = used_idx.get(normalized, 0)
            choose_index = min(choose_index, len(candidates) - 1)
            chosen = candidates[choose_index]
            used_idx[normalized] = choose_index + 1
            matrix = _sam6d_result_world_matrix(chosen, transform, args)
            if matrix is None:
                continue
            matrix, pose_adjustment = _correct_wall_stage_z(matrix, role=role, targets=targets, args=args)
            role_matrices[role] = matrix
            applied.append(
                {
                    "role": role,
                    "instance_index": int(choose_index),
                    "object_name": object_name,
                    "score": float(chosen.get("score", 0.0)),
                    "assignment": "object_name_instance_order",
                    "pose_adjustment": pose_adjustment,
                }
            )
        role_matrices["floor"] = _pose_report_to_matrix(targets["floor"])

    missing = [role for role in all_roles if role not in role_matrices]
    if missing:
        raise RuntimeError(f"missing pose for roles: {missing}")
    assignment["wall_center_distance_validation"] = _validate_wall_center_distances(role_matrices, roles, args)

    role_states = {}
    for role in all_roles:
        pose = _matrix_to_pose_report(role_matrices[role])
        role_states[role] = {
            "pose": pose,
            "linear_velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
            "target_pose_error": None,
        }
    state = {
        "schema": "jimu_assembly_state_v1",
        "source": "sam6d_to_assembly_state",
        "roles": all_roles,
        "completed_roles": _role_list(args.completed_roles),
        "robot_qpos": None,
        "role_states": role_states,
        "magnetic_snap_report": {
            "connections": [],
            "active_connection_count": 0,
            "connection_error": {"connections": [], "max_point_error": 0.0, "mean_point_error": 0.0},
        },
    }
    debug = {
        "sam6d_result_json": str(result_path) if result_path is not None else "",
        "roles": roles,
        "object_names": object_names,
        "candidate_count": len(candidate_entries),
        "applied": applied,
        "assignment": assignment,
    }
    return state, debug


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-state", default=str(Path(__file__).resolve().parent / "initial_scene_state.json"))
    parser.add_argument("--out-debug", default="")
    parser.add_argument("--roles", default=DEFAULT_ROLES)
    parser.add_argument("--completed-roles", default="")
    parser.add_argument("--sam3-preview-only", action="store_true", help="Only run SAM3 segmentation preview and exit without PEM.")
    parser.add_argument(
        "--sam3-inspect-before-pem",
        action="store_true",
        help="Show/confirm SAM3 full-scene masks, then continue to SAM6D PEM pose estimation.",
    )
    parser.add_argument("--sam6d-result-json", default="")
    parser.add_argument("--sam6d-object-names", default=DEFAULT_OBJECT_NAMES)
    parser.add_argument("--sam6d-default-object-name", default="red_bricks_cube")
    parser.add_argument("--sam6d-instance-indices", default=DEFAULT_INSTANCE_INDICES)
    parser.add_argument("--sam6d-provider-script", default=DEFAULT_SAM6D_PROVIDER_SCRIPT)
    parser.add_argument("--sam6d-python", default=DEFAULT_SAM6D_PYTHON)
    parser.add_argument("--sam6d-output-root", default=str(Path(__file__).resolve().parent / "sam6d_pose_init_runs"))
    parser.add_argument("--sam6d-mask-mode", default="sam3_text")
    parser.add_argument("--sam6d-root", default=DEFAULT_SAM6D_ROOT)
    parser.add_argument("--sam6d-camera-width", type=int, default=640)
    parser.add_argument("--sam6d-camera-height", type=int, default=480)
    parser.add_argument("--sam6d-camera-fps", type=int, default=30)
    parser.add_argument("--sam6d-warmup-frames", type=int, default=30)
    parser.add_argument("--sam6d-frame-dir", default="")
    parser.add_argument("--sam6d-sam3-full-scene-keep-multi-instances", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sam6d-auto-floor-from-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sam6d-confirm-full-scene-masks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sam6d-require-full-scene-masks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sam6d-show-full-scene-mask-window", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sam6d-sam3-full-scene-result-json", default="")
    parser.add_argument("--sam6d-pem-warmup-during-sam3", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sam6d-skip-pem", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sam3-instance-index", type=int, default=0)
    parser.add_argument("--sam3-max-masks-per-item", type=int, default=DEFAULT_SAM3_MAX_MASKS_PER_ITEM)
    parser.add_argument("--sam3-confidence-threshold", type=float, default=DEFAULT_SAM3_CONFIDENCE_THRESHOLD)
    parser.add_argument("--sam6d-use-direct-camera-extrinsic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sam6d-camera-transform-convention", choices=["auto", "camera_to_world", "world_to_camera"], default="camera_to_world")
    parser.add_argument("--sam6d-map-through-robot-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sam6d-robot-base-position", default=DEFAULT_ROBOT_BASE_POSITION)
    parser.add_argument("--sam6d-robot-base-yaw-deg", type=float, default=DEFAULT_ROBOT_BASE_YAW_DEG)
    parser.add_argument("--sam6d-cad-to-sim-correction", choices=["auto", "none", "red_jimu_cube", "red_bricks_cube"], default="auto")
    parser.add_argument("--sam6d-cad-to-sim-local-rpy-deg", default="")
    parser.add_argument("--sam6d-camera-to-world-transform", default=DEFAULT_CAMERA_TO_WORLD_TRANSFORM)
    parser.add_argument("--sam6d-floor-normal-z-threshold", type=float, default=0.65)
    parser.add_argument("--sam6d-floor-min-score", type=float, default=0.05)
    parser.add_argument("--sam6d-wall-min-score", type=float, default=0.05)
    parser.add_argument("--sam6d-min-wall-center-distance", type=float, default=0.045)
    parser.add_argument("--sam6d-floor-target-shift-mode", choices=["xy", "xy_yaw", "xyz", "full", "none"], default="xy")
    parser.add_argument("--sam6d-wall-stage-z-mode", choices=["table_upright", "target_wall", "raw", "none"], default="table_upright")
    parser.add_argument("--sam6d-wall-stage-z-offset", type=float, default=0.0)
    parser.add_argument("--sam6d-wall-orientation-mode", choices=["preserve", "force_upright"], default="preserve")
    parser.add_argument("--sam6d-provider-timeout-s", type=float, default=240.0)
    args = parser.parse_args()

    if bool(args.sam3_preview_only):
        preview = _run_sam3_preview_only(args)
        print(json.dumps(_json_ready(preview), ensure_ascii=False, indent=2))
        return

    state, debug = build_assembly_state(args)
    out_state = Path(args.out_state).expanduser()
    out_state.parent.mkdir(parents=True, exist_ok=True)
    out_state.write_text(json.dumps(_json_ready(state), ensure_ascii=False, indent=2), encoding="utf-8")
    out_debug = Path(args.out_debug).expanduser() if str(args.out_debug).strip() else out_state.with_suffix(".debug.json")
    out_debug.parent.mkdir(parents=True, exist_ok=True)
    out_debug.write_text(json.dumps(_json_ready(debug), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"state": str(out_state), "debug": str(out_debug), "roles": state["roles"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
