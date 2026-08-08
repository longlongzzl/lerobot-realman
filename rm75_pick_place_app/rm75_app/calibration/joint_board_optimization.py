from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rm75_app.paths import DEFAULT_CAMERA_EXTRINSIC

from .board import (
    BoardConfig,
    BoardDetection,
    board_config_from_args,
    board_object_points,
    detection_is_accepted,
    detect_board_pose,
    make_charuco_board,
    pose_diversity_metrics,
)
from .common import (
    as_transform,
    average_transforms,
    calibration_run_dir,
    camera_opencv_to_base_camera,
    image_write,
    invert_transform,
    load_json,
    load_matrix,
    load_realman_settings,
    pose_error,
    rvec_tvec_to_transform,
    save_json,
    save_matrix_pair,
    to_jsonable,
    transform_to_rvec_tvec,
)
from .reporting import normalize_calibration_report, write_html_report


@dataclass
class BoardObservation:
    index: int
    detection: BoardDetection
    image_path: Path | None = None


@dataclass
class WristDataset:
    run_dir: Path
    observations: list[BoardObservation]
    base_T_ee: list[np.ndarray]
    intrinsic: np.ndarray
    dist_coeffs: np.ndarray
    initial_ee_T_wrist: np.ndarray


@dataclass
class GlobalDataset:
    run_dir: Path | None
    observations: list[BoardObservation]
    intrinsic: np.ndarray
    dist_coeffs: np.ndarray
    base_T_camera: np.ndarray
    board_anchor: np.ndarray
    board_anchor_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly optimize RM75 fixed-board pose and wrist-camera extrinsic from global and wrist ChArUco observations."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--global-board-run-dir", type=Path, default=None)
    parser.add_argument("--wrist-board-run-dir", type=Path, required=True)
    parser.add_argument("--base-camera-run-dir", type=Path, default=None)
    parser.add_argument("--base-camera-extrinsic-opencv-path", type=Path, default=DEFAULT_CAMERA_EXTRINSIC)
    parser.add_argument("--base-camera-use-direct", action="store_true")
    parser.add_argument("--board-anchor-path", type=Path, default=None)
    parser.add_argument("--ee-wrist-camera-path", type=Path, default=None)
    parser.add_argument("--wrist-reprojection-weight", type=float, default=1.0)
    parser.add_argument("--global-reprojection-weight", type=float, default=0.5)
    parser.add_argument("--board-prior-weight", type=float, default=1.0)
    parser.add_argument("--wrist-prior-weight", type=float, default=1.0)
    parser.add_argument("--board-prior-translation-sigma-m", type=float, default=0.003)
    parser.add_argument("--board-prior-rotation-sigma-deg", type=float, default=0.5)
    parser.add_argument("--wrist-prior-translation-sigma-m", type=float, default=0.015)
    parser.add_argument("--wrist-prior-rotation-sigma-deg", type=float, default=2.0)
    parser.add_argument("--loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="huber")
    parser.add_argument("--loss-f-scale", type=float, default=2.0)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--min-wrist-frames", type=int, default=12)
    parser.add_argument("--recommended-wrist-frames", type=int, default=30)
    parser.add_argument("--max-board-drift-m", type=float, default=0.010)
    parser.add_argument("--max-board-drift-deg", type=float, default=2.0)
    parser.add_argument("--max-wrist-drift-m", type=float, default=0.030)
    parser.add_argument("--max-wrist-drift-deg", type=float, default=5.0)
    parser.add_argument("--max-global-inconsistency-m", type=float, default=0.010)
    parser.add_argument("--max-global-inconsistency-deg", type=float, default=2.0)
    parser.add_argument("--min-pose-translation-span-m", type=float, default=0.020)
    parser.add_argument("--min-pose-rotation-span-deg", type=float, default=5.0)
    parser.add_argument("--allow-missing-global-observations", action="store_true")
    parser.add_argument("--no-html-report", action="store_true")
    parser.add_argument("--board-squares-x", type=int, default=None)
    parser.add_argument("--board-squares-y", type=int, default=None)
    parser.add_argument("--board-square-length-m", type=float, default=None)
    parser.add_argument("--board-marker-length-m", type=float, default=None)
    parser.add_argument("--board-dictionary", type=str, default=None)
    parser.add_argument("--board-legacy-pattern", action="store_true", default=None)
    parser.add_argument("--min-board-corners", type=int, default=None)
    parser.add_argument("--max-board-rmse-px", type=float, default=None)
    parser.add_argument("--min-corner-coverage", type=float, default=None)
    return parser.parse_args()


def _board_config(args: argparse.Namespace, settings: dict[str, Any], wrist_run_dir: Path | None = None) -> BoardConfig:
    if settings or any(
        getattr(args, attr, None) is not None
        for attr in (
            "board_squares_x",
            "board_squares_y",
            "board_square_length_m",
            "board_marker_length_m",
            "board_dictionary",
            "min_board_corners",
            "max_board_rmse_px",
            "min_corner_coverage",
        )
    ):
        return board_config_from_args(args, settings)
    if wrist_run_dir is not None:
        report_path = wrist_run_dir / "calibration_report.json"
        if report_path.exists():
            raw = load_json(report_path).get("board_config")
            if isinstance(raw, dict):
                return BoardConfig(**{key: raw[key] for key in raw if key in BoardConfig.__dataclass_fields__})
    return board_config_from_args(args, settings)


def _first_existing(root: Path | None, names: list[str]) -> Path | None:
    if root is None:
        return None
    root = Path(root).expanduser()
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def _load_optional_matrix(root: Path | None, names: list[str]) -> tuple[np.ndarray | None, str | None]:
    path = _first_existing(root, names)
    if path is None:
        return None, None
    return load_matrix(path), str(path)


def _load_numpy_array(root: Path | None, names: list[str], *, required: bool = True) -> np.ndarray | None:
    path = _first_existing(root, names)
    if path is None:
        if required:
            raise FileNotFoundError(f"missing any of {names} under {root}")
        return None
    return np.load(path)


def _load_detection(item: dict[str, Any]) -> BoardDetection:
    T_raw = item.get("T_cam_board")
    return BoardDetection(
        ok=bool(item.get("ok", False)),
        T_cam_board=None if T_raw is None else as_transform(T_raw),
        corners=np.asarray(item.get("corners", []), dtype=np.float64).reshape(-1, 2),
        ids=np.asarray(item.get("ids", []), dtype=np.int32).reshape(-1),
        reprojection_rmse_px=item.get("reprojection_rmse_px"),
        reason=str(item.get("reason", "")),
        quality=dict(item.get("quality", {})),
    )


def _image_path(run_dir: Path, index: int, subdirs: tuple[str, ...]) -> Path | None:
    for subdir in subdirs:
        path = run_dir / subdir / f"{index:04d}.png"
        if path.exists():
            return path
    return None


def _frame_image_path(run_dir: Path, frame: dict[str, Any], index: int, subdirs: tuple[str, ...]) -> Path | None:
    for key in ("source_image", "image_path"):
        raw = frame.get(key)
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = run_dir / path
        if path.exists():
            return path
    return _image_path(run_dir, index, subdirs)


def _observation_items_from_json(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    raw = data.get("detections")
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    frames = data.get("frames")
    if isinstance(frames, list):
        items = []
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            det = frame.get("detection") or frame.get("board_detection")
            if isinstance(det, dict):
                items.append(det)
        return items
    observations = data.get("observations")
    if isinstance(observations, list):
        return [item for item in observations if isinstance(item, dict)]
    return []


def _load_saved_observations(run_dir: Path, *, image_subdirs: tuple[str, ...]) -> list[BoardObservation]:
    json_candidates = [
        "observations.json",
        "base_board_detections.json",
        "global_board_anchor_observations.json",
    ]
    for name in json_candidates:
        path = run_dir / name
        if not path.exists():
            continue
        data = load_json(path)
        if isinstance(data, dict) and isinstance(data.get("frames"), list) and isinstance(data.get("detections"), list):
            frames = [item for item in data["frames"] if isinstance(item, dict)]
            detections = [item for item in data["detections"] if isinstance(item, dict)]
            if len(frames) == len(detections) and detections:
                observations = []
                for idx, (frame, item) in enumerate(zip(frames, detections)):
                    index = int(frame.get("index", item.get("index", idx)))
                    observations.append(
                        BoardObservation(
                            index=index,
                            detection=_load_detection(item),
                            image_path=_frame_image_path(run_dir, frame, index, image_subdirs),
                        )
                    )
                return observations
        items = _observation_items_from_json(data)
        if not items:
            continue
        observations = []
        for idx, item in enumerate(items):
            index = int(item.get("index", idx))
            observations.append(
                BoardObservation(
                    index=index,
                    detection=_load_detection(item),
                    image_path=_image_path(run_dir, index, image_subdirs),
                )
            )
        return observations
    return []


def _detect_observations_from_images(
    run_dir: Path,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
    board_cfg: BoardConfig,
    *,
    image_subdirs: tuple[str, ...],
) -> list[BoardObservation]:
    for subdir in image_subdirs:
        image_dir = run_dir / subdir
        if not image_dir.exists():
            continue
        observations = []
        for idx, path in enumerate(sorted(image_dir.glob("*.png"))):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            det = detect_board_pose(image, intrinsic, dist_coeffs, board_cfg)
            observations.append(BoardObservation(index=idx, detection=det, image_path=path))
        if observations:
            return observations
    return []


def _valid_observations(observations: list[BoardObservation], board_cfg: BoardConfig) -> list[BoardObservation]:
    return [
        obs
        for obs in observations
        if detection_is_accepted(obs.detection, board_cfg)
        and obs.detection.T_cam_board is not None
        and obs.detection.ids.size > 0
        and obs.detection.corners.size > 0
    ]


def _load_base_camera_transform(args: argparse.Namespace, global_run_dir: Path | None) -> tuple[np.ndarray, str]:
    search_roots = [args.base_camera_run_dir, global_run_dir]
    for root in search_roots:
        mat, source = _load_optional_matrix(
            root,
            [
                "base_T_camera.npy",
                "R_T_global_camera.npy",
                "R_T_camera.npy",
                "T_R_Cg.npy",
            ],
        )
        if mat is not None and source is not None:
            return mat, source
        opencv_path = _first_existing(root, ["camera_extrinsic_opencv.npy"])
        if opencv_path is not None:
            return camera_opencv_to_base_camera(
                load_matrix(opencv_path),
                use_direct=bool(args.base_camera_use_direct),
            ), str(opencv_path)
    return (
        camera_opencv_to_base_camera(
            load_matrix(args.base_camera_extrinsic_opencv_path),
            use_direct=bool(args.base_camera_use_direct),
        ),
        str(args.base_camera_extrinsic_opencv_path),
    )


def _load_board_anchor(args: argparse.Namespace, global_run_dir: Path | None, wrist_run_dir: Path) -> tuple[np.ndarray, str]:
    if args.board_anchor_path is not None:
        return load_matrix(args.board_anchor_path), str(args.board_anchor_path)
    for root in [global_run_dir, args.base_camera_run_dir, wrist_run_dir]:
        mat, source = _load_optional_matrix(
            root,
            [
                "R_T_board.npy",
                "base_T_board.npy",
                "T_R_P.npy",
                "board_anchor_used.npy",
                "R_T_board_joint.npy",
            ],
        )
        if mat is not None and source is not None:
            return mat, source
    raise FileNotFoundError("missing board anchor; pass --global-board-run-dir with R_T_board.npy or --board-anchor-path")


def _load_global_dataset(
    args: argparse.Namespace,
    board_cfg: BoardConfig,
    wrist_run_dir: Path,
) -> GlobalDataset:
    global_run_dir = None if args.global_board_run_dir is None else Path(args.global_board_run_dir).expanduser().resolve()
    base_T_camera, _base_camera_source = _load_base_camera_transform(args, global_run_dir)
    board_anchor, board_anchor_source = _load_board_anchor(args, global_run_dir, wrist_run_dir)
    intrinsic = None
    dist_coeffs = None
    for root in [global_run_dir, args.base_camera_run_dir]:
        intrinsic = _load_numpy_array(root, ["camera_intrinsic.npy"], required=False)
        dist_coeffs = _load_numpy_array(root, ["camera_dist_coeffs.npy"], required=False)
        if intrinsic is not None:
            break
    if intrinsic is None:
        raise FileNotFoundError("missing global camera_intrinsic.npy in --global-board-run-dir or --base-camera-run-dir")
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5,), dtype=np.float64)
    observations: list[BoardObservation] = []
    if global_run_dir is not None:
        observations = _load_saved_observations(global_run_dir, image_subdirs=("images", "main_images", "global_images"))
        if not observations:
            observations = _detect_observations_from_images(
                global_run_dir,
                intrinsic,
                dist_coeffs,
                board_cfg,
                image_subdirs=("images", "main_images", "global_images"),
            )
    return GlobalDataset(
        run_dir=global_run_dir,
        observations=observations,
        intrinsic=np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
        dist_coeffs=np.asarray(dist_coeffs, dtype=np.float64).reshape(-1),
        base_T_camera=base_T_camera,
        board_anchor=board_anchor,
        board_anchor_source=board_anchor_source,
    )


def _load_wrist_dataset(args: argparse.Namespace) -> WristDataset:
    run_dir = Path(args.wrist_board_run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"wrist board run dir not found: {run_dir}")
    observations = _load_saved_observations(run_dir, image_subdirs=("images", "wrist_images"))
    if not observations:
        raise FileNotFoundError(f"missing wrist observations.json with ChArUco corners in {run_dir}")
    base_T_ee = [as_transform(item["base_T_ee"]) for item in load_json(run_dir / "observations.json")["frames"]]
    intrinsic = _load_numpy_array(run_dir, ["camera_intrinsic.npy"])
    dist_coeffs = _load_numpy_array(run_dir, ["camera_dist_coeffs.npy"], required=False)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5,), dtype=np.float64)
    if args.ee_wrist_camera_path is not None:
        initial_ee_T_wrist = load_matrix(args.ee_wrist_camera_path)
    else:
        initial_ee_T_wrist, source = _load_optional_matrix(
            run_dir,
            [
                "ee_T_wrist_camera.npy",
                "ee_T_wrist_camera_initial.npy",
                "ee_T_wrist_camera_refined.npy",
            ],
        )
        if initial_ee_T_wrist is None or source is None:
            raise FileNotFoundError(f"missing ee_T_wrist_camera.npy in {run_dir}")
    return WristDataset(
        run_dir=run_dir,
        observations=observations,
        base_T_ee=base_T_ee,
        intrinsic=np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
        dist_coeffs=np.asarray(dist_coeffs, dtype=np.float64).reshape(-1),
        initial_ee_T_wrist=initial_ee_T_wrist,
    )


def _rotation_residual(current: np.ndarray, prior: np.ndarray) -> np.ndarray:
    current = as_transform(current)
    prior = as_transform(prior)
    r_delta = prior[:3, :3].T @ current[:3, :3]
    rvec, _ = cv2.Rodrigues(r_delta)
    return rvec.reshape(3)


def _prior_residuals(
    current: np.ndarray,
    prior: np.ndarray,
    *,
    translation_sigma_m: float,
    rotation_sigma_deg: float,
    weight: float,
) -> np.ndarray:
    current = as_transform(current)
    prior = as_transform(prior)
    t_sigma = max(float(translation_sigma_m), 1e-9)
    r_sigma = max(float(np.deg2rad(rotation_sigma_deg)), 1e-9)
    return float(weight) * np.concatenate(
        [
            (current[:3, 3] - prior[:3, 3]) / t_sigma,
            _rotation_residual(current, prior) / r_sigma,
        ]
    )


def _pack(ee_T_wrist: np.ndarray, base_T_board: np.ndarray) -> np.ndarray:
    ee_rvec, ee_tvec = transform_to_rvec_tvec(ee_T_wrist)
    board_rvec, board_tvec = transform_to_rvec_tvec(base_T_board)
    return np.concatenate([ee_rvec, ee_tvec, board_rvec, board_tvec])


def _unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        rvec_tvec_to_transform(params[0:3], params[3:6]),
        rvec_tvec_to_transform(params[6:9], params[9:12]),
    )


def _project_points(
    T_cam_board: np.ndarray,
    ids: np.ndarray,
    board_points: np.ndarray,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    object_points = board_points[ids]
    rvec, tvec = transform_to_rvec_tvec(T_cam_board)
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
        np.asarray(dist_coeffs, dtype=np.float64).reshape(-1),
    )
    return projected.reshape(-1, 2)


def _frame_confidence(det: BoardDetection, board_cfg: BoardConfig) -> float:
    corner_count = max(1, int(np.asarray(det.ids).reshape(-1).size))
    confidence = min(2.0, max(0.25, corner_count / max(float(board_cfg.min_corners), 1.0)))
    if det.reprojection_rmse_px is not None:
        confidence /= max(1.0, float(det.reprojection_rmse_px))
    return float(confidence)


def _reprojection_vectors(
    *,
    observations: list[BoardObservation],
    camera_poses: list[np.ndarray] | None,
    fixed_camera_pose: np.ndarray | None,
    board_points: np.ndarray,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
    base_T_board: np.ndarray,
    board_cfg: BoardConfig,
    weight: float,
) -> list[np.ndarray]:
    chunks = []
    for idx, obs in enumerate(observations):
        det = obs.detection
        if camera_poses is not None:
            base_T_camera = camera_poses[idx]
        elif fixed_camera_pose is not None:
            base_T_camera = fixed_camera_pose
        else:
            raise ValueError("camera_poses or fixed_camera_pose is required")
        cam_T_board = invert_transform(base_T_camera) @ base_T_board
        projected = _project_points(cam_T_board, det.ids, board_points, intrinsic, dist_coeffs)
        observed = det.corners.reshape(-1, 2)
        confidence = _frame_confidence(det, board_cfg)
        chunks.append((projected - observed).reshape(-1) * float(weight) * np.sqrt(confidence))
    return chunks


def _rmse_for_observations(
    *,
    observations: list[BoardObservation],
    camera_poses: list[np.ndarray] | None,
    fixed_camera_pose: np.ndarray | None,
    board_points: np.ndarray,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
    base_T_board: np.ndarray,
) -> tuple[float | None, list[dict[str, Any]]]:
    squared_distances = []
    per_frame = []
    for idx, obs in enumerate(observations):
        det = obs.detection
        base_T_camera = camera_poses[idx] if camera_poses is not None else fixed_camera_pose
        if base_T_camera is None:
            raise ValueError("camera pose missing")
        cam_T_board = invert_transform(base_T_camera) @ base_T_board
        projected = _project_points(cam_T_board, det.ids, board_points, intrinsic, dist_coeffs)
        observed = det.corners.reshape(-1, 2)
        distances2 = np.sum((projected - observed) ** 2, axis=1)
        rmse = float(np.sqrt(np.mean(distances2))) if distances2.size else None
        if rmse is not None:
            squared_distances.extend(float(item) for item in distances2)
        per_frame.append(
            {
                "index": int(obs.index),
                "corner_count": int(det.ids.size),
                "rmse_px": rmse,
                "detection_rmse_px": det.reprojection_rmse_px,
            }
        )
    if not squared_distances:
        return None, per_frame
    return float(np.sqrt(np.mean(np.asarray(squared_distances, dtype=np.float64)))), per_frame


def _global_consistency(
    observations: list[BoardObservation],
    base_T_camera: np.ndarray,
) -> dict[str, Any]:
    estimates = [
        as_transform(base_T_camera) @ as_transform(obs.detection.T_cam_board)
        for obs in observations
        if obs.detection.T_cam_board is not None
    ]
    if not estimates:
        return {"frames": 0}
    average = average_transforms(estimates)
    errors = [pose_error(item, average) for item in estimates]
    return {
        "frames": len(estimates),
        "mean_translation_m": float(np.mean([item["translation_m"] for item in errors])),
        "max_translation_m": float(np.max([item["translation_m"] for item in errors])),
        "mean_rotation_deg": float(np.mean([item["rotation_deg"] for item in errors])),
        "max_rotation_deg": float(np.max([item["rotation_deg"] for item in errors])),
        "average_R_T_board_from_global_observations": average,
        "per_frame": errors,
    }


def _pose_delta_mm_deg(error: dict[str, float]) -> dict[str, float]:
    return {
        "translation_mm": float(error["translation_m"]) * 1000.0,
        "rotation_deg": float(error["rotation_deg"]),
    }


def _optimize_joint(
    args: argparse.Namespace,
    board_cfg: BoardConfig,
    global_data: GlobalDataset,
    wrist_data: WristDataset,
    global_obs: list[BoardObservation],
    wrist_obs: list[BoardObservation],
    wrist_base_T_ee: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        from scipy.optimize import least_squares
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("scipy.optimize.least_squares is required for joint board optimization") from exc

    board_points = board_object_points(make_charuco_board(board_cfg))
    initial_params = _pack(wrist_data.initial_ee_T_wrist, global_data.board_anchor)

    def residual_fn(params: np.ndarray) -> np.ndarray:
        ee_T_wrist, base_T_board = _unpack(params)
        chunks = []
        wrist_camera_poses = [base_T_ee @ ee_T_wrist for base_T_ee in wrist_base_T_ee]
        chunks.extend(
            _reprojection_vectors(
                observations=wrist_obs,
                camera_poses=wrist_camera_poses,
                fixed_camera_pose=None,
                board_points=board_points,
                intrinsic=wrist_data.intrinsic,
                dist_coeffs=wrist_data.dist_coeffs,
                base_T_board=base_T_board,
                board_cfg=board_cfg,
                weight=float(args.wrist_reprojection_weight),
            )
        )
        chunks.extend(
            _reprojection_vectors(
                observations=global_obs,
                camera_poses=None,
                fixed_camera_pose=global_data.base_T_camera,
                board_points=board_points,
                intrinsic=global_data.intrinsic,
                dist_coeffs=global_data.dist_coeffs,
                base_T_board=base_T_board,
                board_cfg=board_cfg,
                weight=float(args.global_reprojection_weight),
            )
        )
        chunks.append(
            _prior_residuals(
                base_T_board,
                global_data.board_anchor,
                translation_sigma_m=float(args.board_prior_translation_sigma_m),
                rotation_sigma_deg=float(args.board_prior_rotation_sigma_deg),
                weight=float(args.board_prior_weight),
            )
        )
        chunks.append(
            _prior_residuals(
                ee_T_wrist,
                wrist_data.initial_ee_T_wrist,
                translation_sigma_m=float(args.wrist_prior_translation_sigma_m),
                rotation_sigma_deg=float(args.wrist_prior_rotation_sigma_deg),
                weight=float(args.wrist_prior_weight),
            )
        )
        return np.concatenate(chunks)

    before = residual_fn(initial_params)
    result = least_squares(
        residual_fn,
        initial_params,
        loss=str(args.loss),
        f_scale=float(args.loss_f_scale),
        max_nfev=int(args.max_nfev),
    )
    ee_T_wrist, base_T_board = _unpack(result.x)
    after = residual_fn(result.x)
    solver_report = {
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "initial_weighted_residual_rmse": float(np.sqrt(np.mean(before * before))),
        "final_weighted_residual_rmse": float(np.sqrt(np.mean(after * after))),
        "loss": str(args.loss),
        "loss_f_scale": float(args.loss_f_scale),
        "weights": {
            "wrist_reprojection_weight": float(args.wrist_reprojection_weight),
            "global_reprojection_weight": float(args.global_reprojection_weight),
            "board_prior_weight": float(args.board_prior_weight),
            "wrist_prior_weight": float(args.wrist_prior_weight),
        },
    }
    return ee_T_wrist, base_T_board, solver_report


def _draw_joint_overlay(
    image_bgr: np.ndarray,
    obs: BoardObservation,
    predicted_cam_T_board: np.ndarray,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
    board_points: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    out = image_bgr.copy()
    observed = obs.detection.corners.reshape(-1, 2)
    predicted = _project_points(predicted_cam_T_board, obs.detection.ids, board_points, intrinsic, dist_coeffs)
    for point in observed:
        cv2.circle(out, tuple(np.round(point).astype(int)), 3, (40, 220, 40), -1)
    for point in predicted:
        cv2.drawMarker(
            out,
            tuple(np.round(point).astype(int)),
            (30, 60, 240),
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=2,
        )
    for p_obs, p_pred in zip(observed, predicted):
        cv2.line(out, tuple(np.round(p_obs).astype(int)), tuple(np.round(p_pred).astype(int)), (255, 180, 40), 1)
    rvec, tvec = transform_to_rvec_tvec(predicted_cam_T_board)
    cv2.drawFrameAxes(
        out,
        np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
        np.asarray(dist_coeffs, dtype=np.float64).reshape(-1),
        rvec,
        tvec,
        0.07,
    )
    cv2.putText(out, label[:110], (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
    return out


def _write_overlays(
    run_dir: Path,
    board_cfg: BoardConfig,
    global_data: GlobalDataset,
    wrist_data: WristDataset,
    global_obs: list[BoardObservation],
    wrist_obs: list[BoardObservation],
    wrist_base_T_ee: list[np.ndarray],
    ee_T_wrist: np.ndarray,
    base_T_board: np.ndarray,
) -> None:
    board_points = board_object_points(make_charuco_board(board_cfg))
    for obs in global_obs:
        if obs.image_path is None:
            continue
        image = cv2.imread(str(obs.image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        cam_T_board = invert_transform(global_data.base_T_camera) @ base_T_board
        overlay = _draw_joint_overlay(
            image,
            obs,
            cam_T_board,
            global_data.intrinsic,
            global_data.dist_coeffs,
            board_points,
            label=f"global frame {obs.index:04d} joint reprojection",
        )
        image_write(run_dir / "global_overlays" / f"{obs.index:04d}.png", overlay)
    for obs, base_T_ee in zip(wrist_obs, wrist_base_T_ee):
        if obs.image_path is None:
            continue
        image = cv2.imread(str(obs.image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        cam_T_board = invert_transform(base_T_ee @ ee_T_wrist) @ base_T_board
        overlay = _draw_joint_overlay(
            image,
            obs,
            cam_T_board,
            wrist_data.intrinsic,
            wrist_data.dist_coeffs,
            board_points,
            label=f"wrist frame {obs.index:04d} joint reprojection",
        )
        image_write(run_dir / "wrist_overlays" / f"{obs.index:04d}.png", overlay)


def _copy_existing_overlays(run_dir: Path, global_data: GlobalDataset, wrist_data: WristDataset) -> None:
    if global_data.run_dir is not None:
        for source_dir_name in ("overlays", "board_overlays", "main_overlays", "global_overlays"):
            source_dir = global_data.run_dir / source_dir_name
            if source_dir.exists() and not (run_dir / "global_overlays").exists():
                shutil.copytree(source_dir, run_dir / "global_overlays", dirs_exist_ok=True)
                break
    for source_dir_name in ("board_overlays", "wrist_overlays", "overlays"):
        source_dir = wrist_data.run_dir / source_dir_name
        if source_dir.exists() and not (run_dir / "wrist_overlays").exists():
            shutil.copytree(source_dir, run_dir / "wrist_overlays", dirs_exist_ok=True)
            break


def main() -> int:
    args = parse_args()
    settings = load_realman_settings(args.config)
    wrist_data = _load_wrist_dataset(args)
    board_cfg = _board_config(args, settings, wrist_data.run_dir)
    global_data = _load_global_dataset(args, board_cfg, wrist_data.run_dir)
    run_dir = calibration_run_dir("two_camera_joint_board", args.output_root)

    valid_global_obs = _valid_observations(global_data.observations, board_cfg)
    valid_wrist_pairs = [
        (obs, base_T_ee)
        for obs, base_T_ee in zip(wrist_data.observations, wrist_data.base_T_ee)
        if detection_is_accepted(obs.detection, board_cfg)
        and obs.detection.T_cam_board is not None
        and obs.detection.ids.size > 0
        and obs.detection.corners.size > 0
    ]
    valid_wrist_obs = [item[0] for item in valid_wrist_pairs]
    valid_wrist_base_T_ee = [item[1] for item in valid_wrist_pairs]

    if not valid_wrist_obs:
        raise RuntimeError("No accepted wrist board observations are available for joint optimization.")
    if not valid_global_obs and not args.allow_missing_global_observations:
        raise RuntimeError(
            "No accepted global board observations are available. "
            "Pass a Task 01 global board anchor run or use --allow-missing-global-observations for a rejected prior-only run."
        )

    ee_T_wrist_joint, base_T_board_joint, solver_report = _optimize_joint(
        args,
        board_cfg,
        global_data,
        wrist_data,
        valid_global_obs,
        valid_wrist_obs,
        valid_wrist_base_T_ee,
    )

    board_points = board_object_points(make_charuco_board(board_cfg))
    wrist_camera_poses_initial = [base_T_ee @ wrist_data.initial_ee_T_wrist for base_T_ee in valid_wrist_base_T_ee]
    wrist_camera_poses_final = [base_T_ee @ ee_T_wrist_joint for base_T_ee in valid_wrist_base_T_ee]
    initial_wrist_rmse, initial_wrist_frames = _rmse_for_observations(
        observations=valid_wrist_obs,
        camera_poses=wrist_camera_poses_initial,
        fixed_camera_pose=None,
        board_points=board_points,
        intrinsic=wrist_data.intrinsic,
        dist_coeffs=wrist_data.dist_coeffs,
        base_T_board=global_data.board_anchor,
    )
    final_wrist_rmse, final_wrist_frames = _rmse_for_observations(
        observations=valid_wrist_obs,
        camera_poses=wrist_camera_poses_final,
        fixed_camera_pose=None,
        board_points=board_points,
        intrinsic=wrist_data.intrinsic,
        dist_coeffs=wrist_data.dist_coeffs,
        base_T_board=base_T_board_joint,
    )
    initial_global_rmse, initial_global_frames = _rmse_for_observations(
        observations=valid_global_obs,
        camera_poses=None,
        fixed_camera_pose=global_data.base_T_camera,
        board_points=board_points,
        intrinsic=global_data.intrinsic,
        dist_coeffs=global_data.dist_coeffs,
        base_T_board=global_data.board_anchor,
    )
    final_global_rmse, final_global_frames = _rmse_for_observations(
        observations=valid_global_obs,
        camera_poses=None,
        fixed_camera_pose=global_data.base_T_camera,
        board_points=board_points,
        intrinsic=global_data.intrinsic,
        dist_coeffs=global_data.dist_coeffs,
        base_T_board=base_T_board_joint,
    )

    board_delta = pose_error(base_T_board_joint, global_data.board_anchor)
    wrist_delta = pose_error(ee_T_wrist_joint, wrist_data.initial_ee_T_wrist)
    diversity = pose_diversity_metrics(valid_wrist_base_T_ee)
    global_consistency = _global_consistency(valid_global_obs, global_data.base_T_camera)

    failure_reasons: list[str] = []
    warnings: list[str] = []
    if len(valid_wrist_obs) < int(args.min_wrist_frames):
        failure_reasons.append(f"wrist_frames {len(valid_wrist_obs)} < {int(args.min_wrist_frames)}")
    elif len(valid_wrist_obs) < int(args.recommended_wrist_frames):
        warnings.append(f"wrist_frames {len(valid_wrist_obs)} < recommended {int(args.recommended_wrist_frames)}")
    if initial_wrist_rmse is not None and final_wrist_rmse is not None and final_wrist_rmse > initial_wrist_rmse:
        failure_reasons.append(f"final wrist reprojection worsened {final_wrist_rmse:.4f}px > {initial_wrist_rmse:.4f}px")
    if board_delta["translation_m"] > float(args.max_board_drift_m):
        failure_reasons.append(
            f"board drift translation {board_delta['translation_m']:.4f}m > {float(args.max_board_drift_m):.4f}m"
        )
    if board_delta["rotation_deg"] > float(args.max_board_drift_deg):
        failure_reasons.append(
            f"board drift rotation {board_delta['rotation_deg']:.3f}deg > {float(args.max_board_drift_deg):.3f}deg"
        )
    if wrist_delta["translation_m"] > float(args.max_wrist_drift_m):
        failure_reasons.append(
            f"wrist drift translation {wrist_delta['translation_m']:.4f}m > {float(args.max_wrist_drift_m):.4f}m"
        )
    if wrist_delta["rotation_deg"] > float(args.max_wrist_drift_deg):
        failure_reasons.append(
            f"wrist drift rotation {wrist_delta['rotation_deg']:.3f}deg > {float(args.max_wrist_drift_deg):.3f}deg"
        )
    if not valid_global_obs:
        message = "no global reprojection observations; optimization used board prior and wrist observations only"
        if args.allow_missing_global_observations:
            warnings.append(message)
        else:
            failure_reasons.append(message)
    if global_consistency.get("frames", 0):
        if float(global_consistency.get("max_translation_m", 0.0)) > float(args.max_global_inconsistency_m):
            failure_reasons.append(
                "global board observations translation spread "
                f"{float(global_consistency['max_translation_m']):.4f}m > {float(args.max_global_inconsistency_m):.4f}m"
            )
        if float(global_consistency.get("max_rotation_deg", 0.0)) > float(args.max_global_inconsistency_deg):
            failure_reasons.append(
                "global board observations rotation spread "
                f"{float(global_consistency['max_rotation_deg']):.3f}deg > {float(args.max_global_inconsistency_deg):.3f}deg"
            )
    if float(diversity.get("max_translation_span_m", 0.0)) < float(args.min_pose_translation_span_m):
        warnings.append(
            "wrist pose translation span is narrow: "
            f"{float(diversity.get('max_translation_span_m', 0.0)):.4f}m"
        )
    if float(diversity.get("max_pair_rotation_deg", 0.0)) < float(args.min_pose_rotation_span_deg):
        warnings.append(
            "wrist pose rotation span is narrow: "
            f"{float(diversity.get('max_pair_rotation_deg', 0.0)):.3f}deg"
        )
    if not solver_report["success"]:
        failure_reasons.append(f"least_squares did not converge: {solver_report['message']}")

    status = "rejected" if failure_reasons else "accepted"
    save_matrix_pair(run_dir / "ee_T_wrist_camera_joint.npy", ee_T_wrist_joint)
    save_matrix_pair(run_dir / "R_T_board_joint.npy", base_T_board_joint)
    save_matrix_pair(run_dir / "delta_board_anchor_to_joint.npy", invert_transform(global_data.board_anchor) @ base_T_board_joint)
    save_matrix_pair(run_dir / "delta_wrist_initial_to_joint.npy", invert_transform(wrist_data.initial_ee_T_wrist) @ ee_T_wrist_joint)

    _write_overlays(
        run_dir,
        board_cfg,
        global_data,
        wrist_data,
        valid_global_obs,
        valid_wrist_obs,
        valid_wrist_base_T_ee,
        ee_T_wrist_joint,
        base_T_board_joint,
    )
    _copy_existing_overlays(run_dir, global_data, wrist_data)

    report = {
        "run_dir": run_dir,
        "status": status,
        "accepted": status == "accepted",
        "failure_reasons": failure_reasons,
        "warnings": warnings,
        "global_board_run_dir": global_data.run_dir,
        "wrist_board_run_dir": wrist_data.run_dir,
        "board_anchor_source": global_data.board_anchor_source,
        "frames": {
            "global_total": len(global_data.observations),
            "global_used": len(valid_global_obs),
            "wrist_total": len(wrist_data.observations),
            "wrist_used": len(valid_wrist_obs),
        },
        "reprojection_rmse_px": {
            "global_initial": initial_global_rmse,
            "global_final": final_global_rmse,
            "wrist_initial": initial_wrist_rmse,
            "wrist_final": final_wrist_rmse,
        },
        "per_frame_wrist_errors": {
            "initial": initial_wrist_frames,
            "final": final_wrist_frames,
        },
        "per_frame_global_errors": {
            "initial": initial_global_frames,
            "final": final_global_frames,
        },
        "global_board_pose_delta": board_delta,
        "global_board_pose_delta_mm_deg": _pose_delta_mm_deg(board_delta),
        "wrist_extrinsic_delta": wrist_delta,
        "wrist_extrinsic_delta_mm_deg": _pose_delta_mm_deg(wrist_delta),
        "wrist_pose_diversity": diversity,
        "global_observation_consistency": global_consistency,
        "quality_gates": {
            "max_board_drift_m": float(args.max_board_drift_m),
            "max_board_drift_deg": float(args.max_board_drift_deg),
            "max_wrist_drift_m": float(args.max_wrist_drift_m),
            "max_wrist_drift_deg": float(args.max_wrist_drift_deg),
            "min_wrist_frames": int(args.min_wrist_frames),
            "recommended_wrist_frames": int(args.recommended_wrist_frames),
            "max_global_inconsistency_m": float(args.max_global_inconsistency_m),
            "max_global_inconsistency_deg": float(args.max_global_inconsistency_deg),
        },
        "solver": solver_report,
        "board_config": vars(board_cfg),
        "T_E_Cw_initial": wrist_data.initial_ee_T_wrist,
        "T_E_Cw_joint": ee_T_wrist_joint,
        "T_R_P_anchor": global_data.board_anchor,
        "T_R_P_joint": base_T_board_joint,
        "transform_aliases": {
            "T_E_Cw": "ee_T_wrist_camera_joint.npy",
            "T_R_P": "R_T_board_joint.npy",
            "delta_T_R_P_anchor_to_joint": "delta_board_anchor_to_joint.npy",
            "delta_T_E_Cw_initial_to_joint": "delta_wrist_initial_to_joint.npy",
        },
    }
    report = normalize_calibration_report(
        report,
        run_kind="two_camera_joint_board",
        config_path=args.config,
        frames_total=len(global_data.observations) + len(wrist_data.observations),
        frames_used=len(valid_global_obs) + len(valid_wrist_obs),
        accepted=status == "accepted",
        rejection_reason="; ".join(failure_reasons),
        transforms={
            "T_R_Cg": global_data.base_T_camera,
            "T_R_P": base_T_board_joint,
            "T_E_Cw": ee_T_wrist_joint,
        },
        metrics={
            "reprojection_rmse_px": {
                "global_initial": initial_global_rmse,
                "global_final": final_global_rmse,
                "wrist_initial": initial_wrist_rmse,
                "wrist_final": final_wrist_rmse,
            },
            "global_board_pose_delta_mm_deg": _pose_delta_mm_deg(board_delta),
            "wrist_extrinsic_delta_mm_deg": _pose_delta_mm_deg(wrist_delta),
            "wrist_pose_diversity": diversity,
            "global_observation_consistency": global_consistency,
            "quality_gates": report["quality_gates"],
            "solver": solver_report,
        },
    )
    save_json(run_dir / "joint_optimization_report.json", report)
    if not args.no_html_report:
        write_html_report(
            run_dir / "report.html",
            title="RM75 Two-Camera Joint Board Optimization",
            summary=report,
            sections=[
                ("Quality Gates", {"failure_reasons": failure_reasons, "warnings": warnings}),
                ("Reprojection RMSE", report["reprojection_rmse_px"]),
                ("Per-frame Wrist Errors", report["per_frame_wrist_errors"]),
                ("Global Consistency", report["global_observation_consistency"]),
                ("Solver", report["solver"]),
            ],
            image_dirs=["global_overlays", "wrist_overlays"],
        )
    print(f"[joint-board] status={status} global_used={len(valid_global_obs)} wrist_used={len(valid_wrist_obs)}")
    print(f"[joint-board] report written to {run_dir / 'joint_optimization_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
