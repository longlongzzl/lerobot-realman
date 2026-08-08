from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rm75_app.paths import DEFAULT_RM75_URDF

from .board import (
    BoardConfig,
    BoardDetection,
    board_config_from_args,
    board_object_points,
    detection_is_accepted,
    make_charuco_board,
    pose_diversity_metrics,
)
from .common import (
    as_transform,
    average_transforms,
    calibration_run_dir,
    image_write,
    invert_transform,
    load_json,
    load_matrix,
    load_realman_settings,
    pose_error,
    rvec_tvec_to_transform,
    rotation_error_deg,
    save_json,
    save_matrix_pair,
    transform_to_rvec_tvec,
)
from .reporting import normalize_calibration_report, write_html_report
from .sampling import save_qpos_plan
from .wrist_camera_board_calibration import _collect_observations, _joint_names, _planned_qpos


@dataclass
class ObservationFrame:
    index: int
    base_T_ee: np.ndarray
    detection: BoardDetection
    frame_json: dict[str, Any]
    source_image: Path | None = None
    candidate_ee_T_cam: np.ndarray | None = None
    accepted_for_optimization: bool = False
    rejection_reason: str = ""
    frame_weight: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wrist-camera calibration using a fixed robot-base board anchor and raw ChArUco corner reprojection."
        )
    )
    parser.add_argument("--config", type=Path, default=None, help="JSON with realman_settings.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--input-run",
        "--use-previous-run",
        dest="input_run",
        type=Path,
        default=None,
        help="Previous wrist_camera_board or wrist_camera_board_anchor run to reuse.",
    )
    parser.add_argument(
        "--board-anchor-run-dir",
        type=Path,
        default=None,
        help="Run directory containing Task 01 R_T_board.npy/json. base_T_board.npy is accepted as a compatibility fallback.",
    )
    parser.add_argument("--board-anchor-path", type=Path, default=None, help="Direct path to a fixed T_R_P matrix.")
    parser.add_argument("--execute-real", action="store_true", help="Move through wrist_reference_qpos_deg.")
    parser.add_argument("--robot-ip", type=str, default=None)
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--camera-serial", type=str, default=None)
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=int, default=None)
    parser.add_argument("--camera-exposure-us", type=float, default=None)
    parser.add_argument("--camera-gain", type=float, default=None)
    parser.add_argument("--camera-auto-exposure", action="store_true", default=None)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_RM75_URDF)
    parser.add_argument("--ee-link-name", type=str, default=None, help="URDF link used as the wrist-camera mount frame.")
    parser.add_argument("--joint-names", nargs="+", default=None)
    parser.add_argument("--reference-qpos-json", type=Path, default=None, help="JSON list of joint samples in degrees.")
    parser.add_argument("--settle-s", type=float, default=0.3)
    parser.add_argument("--capture-current-only", action="store_true", help="Capture one frame at the current pose.")
    parser.add_argument("--manual-capture", action="store_true", help="Press Enter/Space to capture each sample.")
    parser.add_argument("--manual-count", type=int, default=40)
    parser.add_argument("--manual-terminal", action="store_true")
    parser.add_argument("--auto-generate-samples", action="store_true")
    parser.add_argument("--sample-count", type=int, default=40)
    parser.add_argument("--sample-seed", type=int, default=7)
    parser.add_argument("--seed-qpos-deg", nargs="+", type=float, default=None)
    parser.add_argument("--joint-jitter-deg", nargs="+", type=float, default=None)
    parser.add_argument("--joint-limits-json", type=Path, default=None)
    parser.add_argument("--dry-run-plan", action="store_true", help="Write the planned qpos samples and exit.")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-delay-ms", type=int, default=1)
    parser.add_argument("--board-squares-x", type=int, default=None)
    parser.add_argument("--board-squares-y", type=int, default=None)
    parser.add_argument("--board-square-length-m", type=float, default=None)
    parser.add_argument("--board-marker-length-m", type=float, default=None)
    parser.add_argument("--board-dictionary", type=str, default=None)
    parser.add_argument("--board-legacy-pattern", action="store_true", default=None)
    parser.add_argument("--min-board-corners", type=int, default=None)
    parser.add_argument("--max-board-rmse-px", type=float, default=None)
    parser.add_argument("--min-corner-coverage", type=float, default=None)
    parser.add_argument("--min-valid-frames", type=int, default=5)
    parser.add_argument("--loss", choices=["soft_l1", "huber", "cauchy", "linear"], default="soft_l1")
    parser.add_argument("--f-scale", type=float, default=2.0)
    parser.add_argument("--max-nfev", type=int, default=200)
    parser.add_argument("--prior-weight", type=float, default=0.2)
    parser.add_argument("--prior-translation-sigma-m", type=float, default=0.08)
    parser.add_argument("--prior-rotation-sigma-deg", type=float, default=25.0)
    parser.add_argument(
        "--max-final-reprojection-rmse-px",
        type=float,
        default=6.0,
        help="Reject the run if all-corner final reprojection RMSE is above this value.",
    )
    parser.add_argument(
        "--max-final-frame-reprojection-rmse-px",
        type=float,
        default=10.0,
        help="Reject the run if any optimized frame reprojection RMSE is above this value.",
    )
    parser.add_argument(
        "--max-final-frame-pose-residual-mm",
        type=float,
        default=10.0,
        help="Reject the run if any frame disagrees with the fixed board anchor by more than this translation.",
    )
    parser.add_argument(
        "--max-final-frame-pose-residual-deg",
        type=float,
        default=2.5,
        help="Reject the run if any frame disagrees with the fixed board anchor by more than this rotation.",
    )
    parser.add_argument(
        "--allow-worse-reprojection",
        action="store_true",
        help="Do not reject when nonlinear refinement increases reprojection RMSE.",
    )
    parser.add_argument("--force-accept", action="store_true", help="Mark the run accepted even if quality gates fail.")
    parser.add_argument("--max-overlays", type=int, default=80)
    parser.add_argument("--no-html-report", action="store_true")
    parser.add_argument(
        "--write-as-refined-to-wrist-run",
        action="store_true",
        help="Also write ee_T_wrist_camera_refined.* back into --input-run.",
    )
    return parser.parse_args()


def _matrix_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.size != 16:
        return None
    return as_transform(arr.reshape(4, 4))


def _load_matrix_from_json(path: Path) -> np.ndarray | None:
    data = load_json(path)
    if not isinstance(data, dict):
        return _matrix_or_none(data)
    for key in ("matrix", "T_R_P", "R_T_board", "base_T_board", "board_anchor_used"):
        mat = _matrix_or_none(data.get(key))
        if mat is not None:
            return mat
    return _matrix_or_none(data)


def _load_named_matrix(path: Path) -> np.ndarray:
    path = Path(path).expanduser()
    if path.suffix == ".json":
        mat = _load_matrix_from_json(path)
        if mat is not None:
            return mat
    return load_matrix(path)


def _load_board_anchor(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    if args.board_anchor_path is not None:
        path = Path(args.board_anchor_path).expanduser().resolve()
        return _load_named_matrix(path), {"source": path, "source_kind": "board_anchor_path"}
    if args.board_anchor_run_dir is None:
        raise ValueError("--board-anchor-run-dir or --board-anchor-path is required unless --dry-run-plan is used.")
    run_dir = Path(args.board_anchor_run_dir).expanduser().resolve()
    if run_dir.is_file():
        return _load_named_matrix(run_dir), {"source": run_dir, "source_kind": "board_anchor_file"}
    candidates = [
        "R_T_board.npy",
        "R_T_board.json",
        "T_R_P.npy",
        "T_R_P.json",
        "board_anchor.npy",
        "board_anchor.json",
        "board_anchor_used.npy",
        "board_anchor_used.json",
        "base_T_board.npy",
        "base_T_board.json",
    ]
    for name in candidates:
        path = run_dir / name
        if path.exists():
            return _load_named_matrix(path), {"source": path, "source_kind": name}
    for name in ("global_board_anchor_report.json", "calibration_report.json", "wrist_anchor_optimization_report.json"):
        path = run_dir / name
        if not path.exists():
            continue
        mat = _load_matrix_from_json(path)
        if mat is not None:
            return mat, {"source": path, "source_kind": name}
    raise FileNotFoundError(
        f"No board anchor matrix found in {run_dir}. Expected R_T_board.npy/json from Task 01 "
        "or a compatibility base_T_board.npy/json."
    )


def _board_config_from_mapping(data: dict[str, Any]) -> BoardConfig:
    allowed = {field.name for field in fields(BoardConfig)}
    values = {key: value for key, value in data.items() if key in allowed}
    return BoardConfig(**values)


def _load_previous_board_config(run_dir: Path) -> BoardConfig | None:
    for name in ("calibration_report.json", "wrist_anchor_optimization_report.json"):
        path = run_dir / name
        if not path.exists():
            continue
        report = load_json(path)
        board_config = report.get("board_config") if isinstance(report, dict) else None
        if isinstance(board_config, dict):
            return _board_config_from_mapping(board_config)
    return None


def _board_config(args: argparse.Namespace, settings: dict[str, Any], input_run: Path | None) -> BoardConfig:
    if not settings.get("charuco_board") and input_run is not None:
        previous = _load_previous_board_config(input_run)
        if previous is not None:
            return board_config_from_args(args, {"charuco_board": asdict(previous)})
    return board_config_from_args(args, settings)


def _load_detection(item: dict[str, Any]) -> BoardDetection:
    T = _matrix_or_none(item.get("T_cam_board"))
    corners = np.asarray(item.get("corners", []), dtype=np.float64).reshape(-1, 2)
    ids = np.asarray(item.get("ids", []), dtype=np.int32).reshape(-1)
    quality = item.get("quality") if isinstance(item.get("quality"), dict) else {}
    return BoardDetection(
        ok=bool(item.get("ok", False)),
        T_cam_board=T,
        corners=corners,
        ids=ids,
        reprojection_rmse_px=item.get("reprojection_rmse_px"),
        reason=str(item.get("reason", "")),
        quality=dict(quality),
    )


def _image_path_for_frame(run_dir: Path, frame_index: int, fallback_index: int) -> Path | None:
    candidates = [
        run_dir / "images" / f"{frame_index:04d}.png",
        run_dir / "images" / f"{fallback_index:04d}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _source_image_for_frame(run_dir: Path, frame_json: dict[str, Any], frame_index: int, fallback_index: int) -> Path | None:
    image_path = _image_path_for_frame(run_dir, frame_index, fallback_index)
    if image_path is not None:
        return image_path
    source_image = frame_json.get("source_image")
    if source_image:
        source_path = Path(source_image).expanduser()
        if source_path.exists():
            return source_path.resolve()
    return None


def _load_observation_run(run_dir: Path) -> tuple[list[ObservationFrame], np.ndarray, np.ndarray, dict[str, Any]]:
    run_dir = Path(run_dir).expanduser().resolve()
    data = load_json(run_dir / "observations.json")
    frames_raw = data.get("frames", [])
    detections_raw = data.get("detections", [])
    if not isinstance(frames_raw, list) or not isinstance(detections_raw, list):
        raise TypeError(f"observations.json has unexpected schema: {run_dir / 'observations.json'}")
    if len(frames_raw) != len(detections_raw):
        raise ValueError(
            f"observations frame/detection length mismatch in {run_dir}: {len(frames_raw)} != {len(detections_raw)}"
        )
    frames: list[ObservationFrame] = []
    for idx, (frame_item, detection_item) in enumerate(zip(frames_raw, detections_raw)):
        frame_json = dict(frame_item)
        frame_index = int(frame_json.get("index", idx))
        frames.append(
            ObservationFrame(
                index=frame_index,
                base_T_ee=as_transform(frame_json["base_T_ee"]),
                detection=_load_detection(detection_item),
                frame_json=frame_json,
                source_image=_source_image_for_frame(run_dir, frame_json, frame_index, idx),
            )
        )
    intrinsic = np.load(run_dir / "camera_intrinsic.npy")
    dist_path = run_dir / "camera_dist_coeffs.npy"
    dist_coeffs = np.load(dist_path) if dist_path.exists() else np.zeros((5,), dtype=np.float64)
    return frames, intrinsic, dist_coeffs, data


def _save_observations(
    run_dir: Path,
    frames: list[ObservationFrame],
    *,
    source_run_dir: Path | None,
    source_observation_json: dict[str, Any],
) -> None:
    frames_json = []
    detections_json = []
    for frame in frames:
        item = dict(frame.frame_json)
        item.update(
            {
                "source_image": None if frame.source_image is None else str(frame.source_image),
                "accepted_for_anchor_optimization": bool(frame.accepted_for_optimization),
                "anchor_optimizer_reason": frame.rejection_reason,
                "candidate_ee_T_wrist_camera": frame.candidate_ee_T_cam,
                "frame_weight": float(frame.frame_weight),
            }
        )
        frames_json.append(item)
        detections_json.append(frame.detection.jsonable())
    payload = {
        "source_run_dir": None if source_run_dir is None else source_run_dir,
        "ee_link_name": source_observation_json.get("ee_link_name"),
        "joint_names": source_observation_json.get("joint_names"),
        "frames": frames_json,
        "detections": detections_json,
    }
    save_json(run_dir / "observations.json", payload)


def _copy_intrinsic_files(source_run_dir: Path, run_dir: Path) -> None:
    for name in ("camera_intrinsic.npy", "camera_dist_coeffs.npy", "base_T_ee.npy", "qpos_rad.npy"):
        source = source_run_dir / name
        if source.exists():
            shutil.copy2(source, run_dir / name)


def _median_abs_deviation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.median(np.abs(values - np.median(values))))


def _spread_metrics(transforms: list[np.ndarray]) -> dict[str, Any]:
    if not transforms:
        return {
            "count": 0,
            "translation_mm_median": None,
            "translation_mm_mean": None,
            "translation_mm_max": None,
            "rotation_deg_median": None,
            "rotation_deg_mean": None,
            "rotation_deg_max": None,
        }
    mats = [as_transform(item) for item in transforms]
    center = average_transforms(mats)
    translations = np.asarray([np.linalg.norm(mat[:3, 3] - center[:3, 3]) * 1000.0 for mat in mats])
    rotations = np.asarray([rotation_error_deg(mat, center) for mat in mats])
    return {
        "count": len(mats),
        "translation_mm_median": float(np.median(translations)),
        "translation_mm_mean": float(np.mean(translations)),
        "translation_mm_max": float(np.max(translations)),
        "rotation_deg_median": float(np.median(rotations)),
        "rotation_deg_mean": float(np.mean(rotations)),
        "rotation_deg_max": float(np.max(rotations)),
    }


def _robust_initial_transform(
    frames: list[ObservationFrame],
    *,
    min_valid_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidates = [(idx, frame.candidate_ee_T_cam) for idx, frame in enumerate(frames) if frame.candidate_ee_T_cam is not None]
    if len(candidates) < max(3, int(min_valid_frames)):
        raise RuntimeError(
            f"Only {len(candidates)} valid anchored frame candidates. Need at least {max(3, int(min_valid_frames))}."
        )
    transforms = [as_transform(item[1]) for item in candidates if item[1] is not None]
    preliminary = average_transforms(transforms)
    translations = np.stack([mat[:3, 3] for mat in transforms])
    translation_center = np.median(translations, axis=0)
    translation_errors = np.linalg.norm(translations - translation_center, axis=1)
    rotation_errors = np.asarray([rotation_error_deg(mat, preliminary) for mat in transforms], dtype=np.float64)
    trans_threshold = max(
        float(np.median(translation_errors) + 3.0 * 1.4826 * _median_abs_deviation(translation_errors)),
        0.03,
    )
    rot_threshold = max(
        float(np.median(rotation_errors) + 3.0 * 1.4826 * _median_abs_deviation(rotation_errors)),
        8.0,
    )
    selected_local = [
        local_idx
        for local_idx, (translation_error, rotation_error) in enumerate(zip(translation_errors, rotation_errors))
        if translation_error <= trans_threshold and rotation_error <= rot_threshold
    ]
    fallback_to_all = len(selected_local) < max(3, int(min_valid_frames))
    if fallback_to_all:
        selected_local = list(range(len(candidates)))
    selected_frame_indices = {candidates[local_idx][0] for local_idx in selected_local}
    outlier_frames = []
    for local_idx, (frame_list_idx, _) in enumerate(candidates):
        frame = frames[frame_list_idx]
        if frame_list_idx in selected_frame_indices:
            frame.accepted_for_optimization = True
            frame.rejection_reason = ""
        else:
            frame.accepted_for_optimization = False
            frame.rejection_reason = (
                f"candidate outlier: translation={translation_errors[local_idx] * 1000.0:.2f}mm "
                f"rotation={rotation_errors[local_idx]:.2f}deg"
            )
            outlier_frames.append(frame.index)
    initial = average_transforms([as_transform(frames[idx].candidate_ee_T_cam) for idx in selected_frame_indices])
    selected_transforms = [as_transform(frames[idx].candidate_ee_T_cam) for idx in selected_frame_indices]
    metrics = {
        "candidate_count": len(candidates),
        "selected_count": len(selected_transforms),
        "fallback_to_all_candidates": bool(fallback_to_all),
        "translation_outlier_threshold_mm": float(trans_threshold * 1000.0),
        "rotation_outlier_threshold_deg": float(rot_threshold),
        "outlier_frames": outlier_frames,
        "candidate_spread_all": _spread_metrics(transforms),
        "candidate_spread_selected": _spread_metrics(selected_transforms),
    }
    return initial, metrics


def _assign_frame_weights(frames: list[ObservationFrame], board_cfg: BoardConfig) -> None:
    used = [frame for frame in frames if frame.accepted_for_optimization]
    if not used:
        return
    board_total = max(1, int(board_object_points(make_charuco_board(board_cfg)).shape[0]))
    corner_counts = np.asarray([max(1, int(frame.detection.ids.size)) for frame in used], dtype=np.float64)
    coverages = np.asarray(
        [
            float(frame.detection.quality.get("corner_coverage", int(frame.detection.ids.size) / board_total))
            for frame in used
        ],
        dtype=np.float64,
    )
    rmses = np.asarray(
        [
            1.0 if frame.detection.reprojection_rmse_px is None else max(0.05, float(frame.detection.reprojection_rmse_px))
            for frame in used
        ],
        dtype=np.float64,
    )
    median_corners = max(1.0, float(np.median(corner_counts)))
    median_coverage = max(0.05, float(np.median(coverages)))
    median_rmse = max(0.05, float(np.median(rmses)))
    for frame in used:
        corners = max(1.0, float(frame.detection.ids.size))
        coverage = max(
            0.05,
            float(frame.detection.quality.get("corner_coverage", int(frame.detection.ids.size) / board_total)),
        )
        rmse = 1.0 if frame.detection.reprojection_rmse_px is None else max(0.05, float(frame.detection.reprojection_rmse_px))
        weight = np.sqrt(corners / median_corners)
        weight *= np.sqrt(coverage / median_coverage)
        weight *= np.sqrt(median_rmse / rmse)
        frame.frame_weight = float(np.clip(weight, 0.35, 2.5))


def _prepare_frames(
    frames: list[ObservationFrame],
    board_anchor: np.ndarray,
    board_cfg: BoardConfig,
    *,
    min_valid_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    for frame in frames:
        detection = frame.detection
        if not detection_is_accepted(detection, board_cfg):
            frame.rejection_reason = str(detection.reason or detection.quality.get("reason") or "detection rejected")
            continue
        if detection.T_cam_board is None or detection.ids.size == 0:
            frame.rejection_reason = "missing board pose or corners"
            continue
        frame.candidate_ee_T_cam = (
            invert_transform(frame.base_T_ee) @ as_transform(board_anchor) @ invert_transform(detection.T_cam_board)
        )
    initial, metrics = _robust_initial_transform(frames, min_valid_frames=int(min_valid_frames))
    _assign_frame_weights(frames, board_cfg)
    return initial, metrics


def _project_frame(
    frame: ObservationFrame,
    ee_T_cam: np.ndarray,
    board_anchor: np.ndarray,
    board_points: np.ndarray,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(frame.detection.ids, dtype=np.int32).reshape(-1)
    object_points = board_points[ids]
    cam_T_board = invert_transform(frame.base_T_ee @ as_transform(ee_T_cam)) @ as_transform(board_anchor)
    rvec, tvec = transform_to_rvec_tvec(cam_T_board)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, intrinsic, dist_coeffs)
    observed = np.asarray(frame.detection.corners, dtype=np.float64).reshape(-1, 2)
    return projected.reshape(-1, 2), observed, object_points


def _reprojection_metrics(
    frames: list[ObservationFrame],
    ee_T_cam: np.ndarray,
    board_anchor: np.ndarray,
    board_points: np.ndarray,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict[str, Any]:
    total_sq = 0.0
    total_points = 0
    per_frame = []
    for frame in frames:
        projected, observed, _ = _project_frame(frame, ee_T_cam, board_anchor, board_points, intrinsic, dist_coeffs)
        errors = projected - observed
        point_sq = np.sum(errors * errors, axis=1)
        rmse = float(np.sqrt(np.mean(point_sq))) if point_sq.size else 0.0
        total_sq += float(np.sum(point_sq))
        total_points += int(point_sq.size)
        per_frame.append(
            {
                "index": int(frame.index),
                "corner_count": int(point_sq.size),
                "weight": float(frame.frame_weight),
                "rmse_px": rmse,
                "mean_px": float(np.mean(np.sqrt(point_sq))) if point_sq.size else 0.0,
                "max_px": float(np.max(np.sqrt(point_sq))) if point_sq.size else 0.0,
            }
        )
    rmses = [item["rmse_px"] for item in per_frame]
    return {
        "frame_count": len(frames),
        "point_count": int(total_points),
        "rmse_px": float(np.sqrt(total_sq / max(1, total_points))),
        "per_frame_rmse_px_mean": float(np.mean(rmses)) if rmses else None,
        "per_frame_rmse_px_median": float(np.median(rmses)) if rmses else None,
        "per_frame_rmse_px_max": float(np.max(rmses)) if rmses else None,
        "per_frame": per_frame,
    }


def _frame_pose_residuals(
    frames: list[ObservationFrame],
    ee_T_cam: np.ndarray,
    board_anchor: np.ndarray,
) -> dict[str, Any]:
    per_frame = []
    for frame in frames:
        estimate = frame.base_T_ee @ as_transform(ee_T_cam) @ as_transform(frame.detection.T_cam_board)
        err = pose_error(estimate, board_anchor)
        per_frame.append(
            {
                "index": int(frame.index),
                "translation_m": float(err["translation_m"]),
                "translation_mm": float(err["translation_m"] * 1000.0),
                "rotation_deg": float(err["rotation_deg"]),
            }
        )
    translations = [item["translation_mm"] for item in per_frame]
    rotations = [item["rotation_deg"] for item in per_frame]
    return {
        "frame_count": len(frames),
        "translation_mm_median": float(np.median(translations)) if translations else None,
        "translation_mm_mean": float(np.mean(translations)) if translations else None,
        "translation_mm_max": float(np.max(translations)) if translations else None,
        "rotation_deg_median": float(np.median(rotations)) if rotations else None,
        "rotation_deg_mean": float(np.mean(rotations)) if rotations else None,
        "rotation_deg_max": float(np.max(rotations)) if rotations else None,
        "per_frame": per_frame,
    }


def _delta_to_transform(delta: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64).reshape(6)
    return rvec_tvec_to_transform(delta[:3], delta[3:6])


def _optimize_anchor_reprojection(
    frames: list[ObservationFrame],
    initial_ee_T_cam: np.ndarray,
    board_anchor: np.ndarray,
    board_cfg: BoardConfig,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    used_frames = [frame for frame in frames if frame.accepted_for_optimization]
    if len(used_frames) < max(3, int(args.min_valid_frames)):
        raise RuntimeError(
            f"Only {len(used_frames)} frames remained after anchored outlier rejection. "
            f"Need at least {max(3, int(args.min_valid_frames))}."
        )
    board_points = board_object_points(make_charuco_board(board_cfg))
    K = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    dist = np.zeros((5,), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    initial_metrics = _reprojection_metrics(used_frames, initial_ee_T_cam, board_anchor, board_points, K, dist)
    initial_pose_residuals = _frame_pose_residuals(used_frames, initial_ee_T_cam, board_anchor)

    try:
        from scipy.optimize import least_squares
    except Exception as exc:
        final_metrics = dict(initial_metrics)
        final_pose_residuals = dict(initial_pose_residuals)
        return as_transform(initial_ee_T_cam), {
            "optimized": False,
            "reason": f"scipy is unavailable: {exc}",
            "initial_reprojection": initial_metrics,
            "final_reprojection": final_metrics,
            "initial_frame_pose_residuals": initial_pose_residuals,
            "final_frame_pose_residuals": final_pose_residuals,
            "delta_from_initial": pose_error(initial_ee_T_cam, initial_ee_T_cam),
        }

    def compose(delta: np.ndarray) -> np.ndarray:
        return _delta_to_transform(delta) @ as_transform(initial_ee_T_cam)

    def residual_fn(delta: np.ndarray) -> np.ndarray:
        ee_T_cam = compose(delta)
        chunks = []
        for frame in used_frames:
            projected, observed, _ = _project_frame(frame, ee_T_cam, board_anchor, board_points, K, dist)
            chunks.append(((projected - observed) * float(frame.frame_weight)).reshape(-1))
        if float(args.prior_weight) > 0.0:
            rot_sigma = max(np.deg2rad(float(args.prior_rotation_sigma_deg)), 1e-6)
            trans_sigma = max(float(args.prior_translation_sigma_m), 1e-6)
            prior = np.concatenate([delta[:3] / rot_sigma, delta[3:6] / trans_sigma])
            chunks.append(prior * float(args.prior_weight))
        return np.concatenate(chunks)

    x0 = np.zeros((6,), dtype=np.float64)
    result = least_squares(
        residual_fn,
        x0,
        loss=str(args.loss),
        f_scale=float(args.f_scale),
        max_nfev=int(args.max_nfev),
    )
    final_ee_T_cam = compose(result.x)
    final_metrics = _reprojection_metrics(used_frames, final_ee_T_cam, board_anchor, board_points, K, dist)
    final_pose_residuals = _frame_pose_residuals(used_frames, final_ee_T_cam, board_anchor)
    return final_ee_T_cam, {
        "optimized": True,
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "loss": str(args.loss),
        "f_scale": float(args.f_scale),
        "prior_weight": float(args.prior_weight),
        "delta_vector": np.asarray(result.x, dtype=np.float64),
        "delta_from_initial": pose_error(final_ee_T_cam, initial_ee_T_cam),
        "initial_reprojection": initial_metrics,
        "final_reprojection": final_metrics,
        "initial_frame_pose_residuals": initial_pose_residuals,
        "final_frame_pose_residuals": final_pose_residuals,
    }


def _load_classical_transform(input_run: Path | None) -> tuple[np.ndarray | None, Path | None, dict[str, Any]]:
    if input_run is None:
        return None, None, {}
    input_run = Path(input_run).expanduser().resolve()
    for name in ("ee_T_wrist_camera.npy", "ee_T_wrist_camera.json"):
        path = input_run / name
        if path.exists():
            metadata = {}
            report_path = input_run / "calibration_report.json"
            if report_path.exists():
                report = load_json(report_path)
                metadata = {
                    "selected_hand_eye_method": report.get("selected_hand_eye_method"),
                    "hand_eye_metrics": report.get("hand_eye_metrics"),
                }
            return load_matrix(path), path, metadata
    return None, None, {}


def _classical_comparison(
    input_run: Path | None,
    used_frames: list[ObservationFrame],
    final_ee_T_cam: np.ndarray,
    board_anchor: np.ndarray,
    board_cfg: BoardConfig,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict[str, Any] | None:
    classical, path, metadata = _load_classical_transform(input_run)
    if classical is None or path is None:
        return None
    board_points = board_object_points(make_charuco_board(board_cfg))
    K = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    dist = np.zeros((5,), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    return {
        "source_path": path,
        "final_vs_classical_pose_error": pose_error(final_ee_T_cam, classical),
        "classical_reprojection": _reprojection_metrics(used_frames, classical, board_anchor, board_points, K, dist),
        "classical_frame_pose_residuals": _frame_pose_residuals(used_frames, classical, board_anchor),
        **metadata,
    }


def _draw_reprojection_overlay(
    image_bgr: np.ndarray,
    frame: ObservationFrame,
    initial_ee_T_cam: np.ndarray,
    final_ee_T_cam: np.ndarray,
    board_anchor: np.ndarray,
    board_points: np.ndarray,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    out = image_bgr.copy()
    final_projected, observed, _ = _project_frame(frame, final_ee_T_cam, board_anchor, board_points, intrinsic, dist_coeffs)
    initial_projected, _, _ = _project_frame(frame, initial_ee_T_cam, board_anchor, board_points, intrinsic, dist_coeffs)
    final_errors = np.linalg.norm(final_projected - observed, axis=1)
    final_rmse = float(np.sqrt(np.mean(final_errors * final_errors))) if final_errors.size else 0.0
    cam_T_board = invert_transform(frame.base_T_ee @ as_transform(final_ee_T_cam)) @ as_transform(board_anchor)
    rvec, tvec = transform_to_rvec_tvec(cam_T_board)
    try:
        cv2.drawFrameAxes(out, intrinsic, dist_coeffs, rvec, tvec, 0.05)
    except cv2.error:
        pass
    for obs, init, opt in zip(observed, initial_projected, final_projected):
        obs_i = tuple(np.round(obs).astype(int))
        init_i = tuple(np.round(init).astype(int))
        opt_i = tuple(np.round(opt).astype(int))
        cv2.circle(out, obs_i, 4, (20, 220, 20), -1, cv2.LINE_AA)
        cv2.drawMarker(out, init_i, (0, 165, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
        cv2.drawMarker(out, opt_i, (255, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2)
        cv2.line(out, obs_i, opt_i, (220, 220, 220), 1, cv2.LINE_AA)
    lines = [
        f"frame {frame.index:04d}  corners={len(observed)}  final_rmse={final_rmse:.3f}px  weight={frame.frame_weight:.2f}",
        "observed=green  initial=orange  optimized=magenta",
    ]
    y0 = 28
    for idx, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (16, y0 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (16, y0 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
    return out


def _write_overlays(
    run_dir: Path,
    frames: list[ObservationFrame],
    initial_ee_T_cam: np.ndarray,
    final_ee_T_cam: np.ndarray,
    board_anchor: np.ndarray,
    board_cfg: BoardConfig,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    max_overlays: int,
) -> list[Path]:
    overlay_paths = []
    board_points = board_object_points(make_charuco_board(board_cfg))
    K = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    dist = np.zeros((5,), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    for frame in [item for item in frames if item.accepted_for_optimization][: max(0, int(max_overlays))]:
        if frame.source_image is None:
            continue
        image = cv2.imread(str(frame.source_image), cv2.IMREAD_COLOR)
        if image is None:
            continue
        overlay = _draw_reprojection_overlay(
            image,
            frame,
            initial_ee_T_cam,
            final_ee_T_cam,
            board_anchor,
            board_points,
            K,
            dist,
        )
        path = run_dir / "overlays" / f"{frame.index:04d}.png"
        image_write(path, overlay)
        overlay_paths.append(path)
    return overlay_paths


def _write_refined_to_input_run(
    input_run: Path,
    run_dir: Path,
    ee_T_cam: np.ndarray,
) -> None:
    input_run = Path(input_run).expanduser().resolve()
    save_matrix_pair(input_run / "ee_T_wrist_camera_refined.npy", ee_T_cam)
    save_matrix_pair(input_run / "wrist_camera_T_ee_refined.npy", invert_transform(ee_T_cam))
    save_json(
        input_run / "wrist_camera_board_anchor_refined_source.json",
        {
            "source_run_dir": run_dir,
            "ee_T_wrist_camera_refined": ee_T_cam,
        },
    )
    save_json(
        input_run / "ee_T_wrist_camera_refined_metadata.json",
        {
            "schema_version": 1,
            "run_kind": "wrist_camera_board_anchor",
            "accepted": True,
            "source_run": run_dir,
            "transform_path": input_run / "ee_T_wrist_camera_refined.npy",
        },
    )


def _evaluate_quality_gates(
    args: argparse.Namespace,
    used_frames: list[ObservationFrame],
    initialization_metrics: dict[str, Any],
    optimization_metrics: dict[str, Any],
) -> tuple[bool, list[str], list[str], dict[str, Any]]:
    initial_reprojection = optimization_metrics["initial_reprojection"]
    final_reprojection = optimization_metrics["final_reprojection"]
    final_pose_residuals = optimization_metrics["final_frame_pose_residuals"]
    final_rmse = float(final_reprojection["rmse_px"])
    initial_rmse = float(initial_reprojection["rmse_px"])
    final_frame_rmse_max = final_reprojection.get("per_frame_rmse_px_max")
    final_translation_max = final_pose_residuals.get("translation_mm_max")
    final_rotation_max = final_pose_residuals.get("rotation_deg_max")

    quality_gates = {
        "min_valid_frames": int(args.min_valid_frames),
        "max_final_reprojection_rmse_px": float(args.max_final_reprojection_rmse_px),
        "max_final_frame_reprojection_rmse_px": float(args.max_final_frame_reprojection_rmse_px),
        "max_final_frame_pose_residual_mm": float(args.max_final_frame_pose_residual_mm),
        "max_final_frame_pose_residual_deg": float(args.max_final_frame_pose_residual_deg),
        "allow_worse_reprojection": bool(args.allow_worse_reprojection),
        "force_accept": bool(args.force_accept),
    }
    failure_reasons: list[str] = []
    warnings: list[str] = []

    if len(used_frames) < max(3, int(args.min_valid_frames)):
        failure_reasons.append(f"frames_used {len(used_frames)} < {max(3, int(args.min_valid_frames))}")
    if not optimization_metrics.get("optimized", False):
        failure_reasons.append(str(optimization_metrics.get("reason") or "optimizer did not run"))
    elif not bool(optimization_metrics.get("success", True)):
        failure_reasons.append(f"least_squares did not converge: {optimization_metrics.get('message', '')}")
    if not bool(args.allow_worse_reprojection) and final_rmse > initial_rmse + 1e-6:
        failure_reasons.append(f"final reprojection worsened {final_rmse:.4f}px > {initial_rmse:.4f}px")
    if final_rmse > float(args.max_final_reprojection_rmse_px):
        failure_reasons.append(
            f"final reprojection rmse {final_rmse:.4f}px > {float(args.max_final_reprojection_rmse_px):.4f}px"
        )
    if final_frame_rmse_max is not None and float(final_frame_rmse_max) > float(args.max_final_frame_reprojection_rmse_px):
        failure_reasons.append(
            "final per-frame reprojection max "
            f"{float(final_frame_rmse_max):.4f}px > {float(args.max_final_frame_reprojection_rmse_px):.4f}px"
        )
    if final_translation_max is not None and float(final_translation_max) > float(args.max_final_frame_pose_residual_mm):
        failure_reasons.append(
            "final frame pose residual translation max "
            f"{float(final_translation_max):.3f}mm > {float(args.max_final_frame_pose_residual_mm):.3f}mm"
        )
    if final_rotation_max is not None and float(final_rotation_max) > float(args.max_final_frame_pose_residual_deg):
        failure_reasons.append(
            "final frame pose residual rotation max "
            f"{float(final_rotation_max):.3f}deg > {float(args.max_final_frame_pose_residual_deg):.3f}deg"
        )

    spread = initialization_metrics.get("candidate_spread_selected", {})
    spread_max_mm = spread.get("translation_mm_max")
    if spread_max_mm is not None and float(spread_max_mm) > 15.0:
        warnings.append(f"candidate spread max is high: {float(spread_max_mm):.2f}mm")
    if bool(args.force_accept) and failure_reasons:
        warnings.append("force_accept=true; quality gate failures were overridden")
        return True, failure_reasons, warnings, quality_gates
    return not failure_reasons, failure_reasons, warnings, quality_gates


def main() -> int:
    args = parse_args()
    settings = load_realman_settings(args.config)
    run_dir = calibration_run_dir("wrist_camera_board_anchor", args.output_root)
    if args.dry_run_plan:
        qpos_samples = _planned_qpos(args, settings, controller=None)
        if not qpos_samples:
            raise ValueError("No wrist_reference_qpos_deg/reference_qpos_deg samples found for dry-run planning.")
        save_qpos_plan(run_dir / "planned_qpos_deg.json", qpos_samples, joint_names=_joint_names(args, settings))
        print(f"[wrist-anchor] dry-run plan written to {run_dir / 'planned_qpos_deg.json'}")
        return 0

    board_anchor, anchor_metadata = _load_board_anchor(args)
    input_run = Path(args.input_run).expanduser().resolve() if args.input_run else None
    source_run_for_observations = input_run if input_run is not None else run_dir
    board_cfg = _board_config(args, settings, input_run)

    if input_run is None:
        _collect_observations(args, settings, run_dir)
    else:
        _copy_intrinsic_files(input_run, run_dir)

    frames, intrinsic, dist_coeffs, source_observation_json = _load_observation_run(source_run_for_observations)
    np.save(run_dir / "camera_intrinsic.npy", np.asarray(intrinsic, dtype=np.float64).reshape(3, 3))
    np.save(run_dir / "camera_dist_coeffs.npy", np.asarray(dist_coeffs, dtype=np.float64).reshape(-1))
    if frames:
        np.save(run_dir / "base_T_ee.npy", np.stack([frame.base_T_ee for frame in frames]))

    initial_ee_T_cam, initialization_metrics = _prepare_frames(
        frames,
        board_anchor,
        board_cfg,
        min_valid_frames=int(args.min_valid_frames),
    )
    final_ee_T_cam, optimization_metrics = _optimize_anchor_reprojection(
        frames,
        initial_ee_T_cam,
        board_anchor,
        board_cfg,
        intrinsic,
        dist_coeffs,
        args,
    )
    used_frames = [frame for frame in frames if frame.accepted_for_optimization]
    initial_reprojection = optimization_metrics["initial_reprojection"]
    final_reprojection = optimization_metrics["final_reprojection"]
    initial_pose_residuals = optimization_metrics["initial_frame_pose_residuals"]
    final_pose_residuals = optimization_metrics["final_frame_pose_residuals"]
    accepted, failure_reasons, warnings, quality_gates = _evaluate_quality_gates(
        args,
        used_frames,
        initialization_metrics,
        optimization_metrics,
    )
    status = "accepted" if accepted else "rejected"
    overlays = _write_overlays(
        run_dir,
        frames,
        initial_ee_T_cam,
        final_ee_T_cam,
        board_anchor,
        board_cfg,
        intrinsic,
        dist_coeffs,
        max_overlays=int(args.max_overlays),
    )
    save_matrix_pair(run_dir / "ee_T_wrist_camera_initial.npy", initial_ee_T_cam)
    save_matrix_pair(run_dir / "ee_T_wrist_camera.npy", final_ee_T_cam)
    save_matrix_pair(run_dir / "wrist_camera_T_ee.npy", invert_transform(final_ee_T_cam))
    save_matrix_pair(run_dir / "board_anchor_used.npy", board_anchor)
    _save_observations(
        run_dir,
        frames,
        source_run_dir=source_run_for_observations,
        source_observation_json=source_observation_json,
    )
    if args.write_as_refined_to_wrist_run:
        if input_run is None:
            raise ValueError("--write-as-refined-to-wrist-run requires --input-run.")
        if accepted:
            _write_refined_to_input_run(input_run, run_dir, final_ee_T_cam)
        else:
            print("[wrist-anchor] skipped writing refined transform to input run because this run was rejected")

    classical = _classical_comparison(
        input_run,
        used_frames,
        final_ee_T_cam,
        board_anchor,
        board_cfg,
        intrinsic,
        dist_coeffs,
    )
    outlier_frames = list(initialization_metrics.get("outlier_frames", []))
    rejected_frames = [
        {"index": int(frame.index), "reason": frame.rejection_reason}
        for frame in frames
        if not frame.accepted_for_optimization
    ]
    required_summary = {
        "frames_total": len(frames),
        "frames_used": len(used_frames),
        "initial_reprojection_rmse_px": initial_reprojection["rmse_px"],
        "final_reprojection_rmse_px": final_reprojection["rmse_px"],
        "initial_candidate_spread_translation_mm_median": initialization_metrics["candidate_spread_selected"][
            "translation_mm_median"
        ],
        "final_frame_pose_residual_translation_mm_median": final_pose_residuals["translation_mm_median"],
        "outlier_frames": outlier_frames,
    }
    optimization_report = {
        **required_summary,
        "run_dir": run_dir,
        "status": status,
        "accepted": accepted,
        "failure_reasons": failure_reasons,
        "warnings": warnings,
        "quality_gates": quality_gates,
        "input_run": input_run,
        "board_anchor_source": anchor_metadata,
        "rejected_frames": rejected_frames,
        "outliers": rejected_frames + [{"index": int(idx), "reason": "robust initialization outlier"} for idx in outlier_frames],
        "initialization_metrics": initialization_metrics,
        "optimization_metrics": optimization_metrics,
        "classical_hand_eye_comparison": classical,
    }
    calibration_report = {
        **optimization_report,
        "kind": "wrist_camera_board_anchor",
        "pose_diversity": pose_diversity_metrics([frame.base_T_ee for frame in used_frames]),
        "camera_intrinsic": intrinsic,
        "camera_dist_coeffs": dist_coeffs,
        "board_config": asdict(board_cfg),
        "T_R_P": board_anchor,
        "board_anchor_used": board_anchor,
        "initial_ee_T_wrist_camera": initial_ee_T_cam,
        "ee_T_wrist_camera": final_ee_T_cam,
        "wrist_camera_T_ee": invert_transform(final_ee_T_cam),
        "initial_reprojection": initial_reprojection,
        "final_reprojection": final_reprojection,
        "initial_frame_pose_residuals": initial_pose_residuals,
        "final_frame_pose_residuals": final_pose_residuals,
        "overlay_count": len(overlays),
    }
    shared_metrics = {
        "initialization": initialization_metrics,
        "optimization": optimization_metrics,
        "classical_hand_eye_comparison": classical,
        "quality_gates": quality_gates,
        "failure_reasons": failure_reasons,
        "warnings": warnings,
        "initial_reprojection": initial_reprojection,
        "final_reprojection": final_reprojection,
        "initial_frame_pose_residuals": initial_pose_residuals,
        "final_frame_pose_residuals": final_pose_residuals,
    }
    shared_transforms = {
        "T_R_P": board_anchor,
        "T_E_Cw": final_ee_T_cam,
    }
    optimization_report = normalize_calibration_report(
        optimization_report,
        run_kind="wrist_camera_board_anchor",
        config_path=args.config,
        frames_total=len(frames),
        frames_used=len(used_frames),
        accepted=accepted,
        rejection_reason="; ".join(failure_reasons),
        transforms=shared_transforms,
        metrics=shared_metrics,
    )
    calibration_report = normalize_calibration_report(
        calibration_report,
        run_kind="wrist_camera_board_anchor",
        config_path=args.config,
        frames_total=len(frames),
        frames_used=len(used_frames),
        accepted=accepted,
        rejection_reason="; ".join(failure_reasons),
        transforms=shared_transforms,
        metrics=shared_metrics,
    )
    save_json(run_dir / "wrist_anchor_optimization_report.json", optimization_report)
    save_json(run_dir / "calibration_report.json", calibration_report)
    if not args.no_html_report:
        write_html_report(
            run_dir / "report.html",
            title="RM75 Wrist Camera Board Anchor Calibration",
            summary=required_summary
            | {"run_dir": run_dir, "overlay_count": len(overlays), "status": status, "accepted": accepted},
            sections=[
                ("Quality Gates", {"failure_reasons": failure_reasons, "warnings": warnings, **quality_gates}),
                ("Transforms", calibration_report["transforms"]),
                ("Initialization", initialization_metrics),
                ("Optimization", optimization_metrics),
                ("Classical Hand Eye Comparison", classical),
                ("Rejected Frames", rejected_frames),
            ],
            image_dirs=["overlays"],
        )
    print(f"[wrist-anchor] status={status} frames used: {len(used_frames)}/{len(frames)}")
    print(
        "[wrist-anchor] reprojection rmse px: "
        f"{initial_reprojection['rmse_px']:.4f} -> {final_reprojection['rmse_px']:.4f}"
    )
    if failure_reasons:
        print("[wrist-anchor] failure reasons: " + "; ".join(failure_reasons))
    print(f"[wrist-anchor] calibration written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
