from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import (
    as_transform,
    average_transforms,
    invert_transform,
    make_transform,
    pose_error,
    rvec_tvec_to_transform,
    save_json,
    to_jsonable,
    transform_to_rvec_tvec,
)


@dataclass
class BoardConfig:
    squares_x: int = 7
    squares_y: int = 5
    square_length_m: float = 0.035
    marker_length_m: float = 0.026
    dictionary: str = "DICT_4X4_50"
    min_corners: int = 8
    max_reprojection_rmse_px: float | None = 2.5
    min_corner_coverage: float = 0.25
    legacy_pattern: bool = False
    marker_id_layout: list[list[int]] | None = None
    enable_chessboard_inner_corners: bool = False
    max_orientation_error_px: float = 3.0


@dataclass
class BoardDetection:
    ok: bool
    T_cam_board: np.ndarray | None
    corners: np.ndarray
    ids: np.ndarray
    reprojection_rmse_px: float | None
    reason: str = ""
    quality: dict[str, Any] = field(default_factory=dict)

    def jsonable(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "T_cam_board": None if self.T_cam_board is None else self.T_cam_board,
            "corners": self.corners,
            "ids": self.ids,
            "reprojection_rmse_px": self.reprojection_rmse_px,
            "reason": self.reason,
            "quality": self.quality,
        }


def board_config_from_args(args: Any, settings: dict[str, Any] | None = None) -> BoardConfig:
    cfg = dict((settings or {}).get("charuco_board", {}))

    def pick(attr: str, key: str, default: Any) -> Any:
        value = getattr(args, attr, None)
        return cfg.get(key, default) if value is None else value

    return BoardConfig(
        squares_x=int(pick("board_squares_x", "squares_x", 7)),
        squares_y=int(pick("board_squares_y", "squares_y", 5)),
        square_length_m=float(pick("board_square_length_m", "square_length_m", 0.035)),
        marker_length_m=float(pick("board_marker_length_m", "marker_length_m", 0.026)),
        dictionary=str(pick("board_dictionary", "dictionary", "DICT_4X4_50")),
        min_corners=int(pick("min_board_corners", "min_corners", 8)),
        max_reprojection_rmse_px=(
            None
            if pick("max_board_rmse_px", "max_reprojection_rmse_px", 2.5) is None
            else float(pick("max_board_rmse_px", "max_reprojection_rmse_px", 2.5))
        ),
        min_corner_coverage=float(pick("min_corner_coverage", "min_corner_coverage", 0.25)),
        legacy_pattern=bool(pick("board_legacy_pattern", "legacy_pattern", False)),
        marker_id_layout=cfg.get("marker_id_layout"),
        enable_chessboard_inner_corners=bool(cfg.get("enable_chessboard_inner_corners", False)),
        max_orientation_error_px=float(cfg.get("max_orientation_error_px", 3.0)),
    )


def _require_aruco():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV aruco module is unavailable. Install opencv-contrib-python.")
    return cv2.aruco


def make_charuco_board(config: BoardConfig):
    aruco = _require_aruco()
    dictionary_id = getattr(aruco, config.dictionary)
    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    size = (int(config.squares_x), int(config.squares_y))
    marker_ids = marker_ids_from_layout(config)
    try:
        if marker_ids is None:
            board = aruco.CharucoBoard(size, float(config.square_length_m), float(config.marker_length_m), dictionary)
        else:
            board = aruco.CharucoBoard(
                size,
                float(config.square_length_m),
                float(config.marker_length_m),
                dictionary,
                marker_ids,
            )
    except Exception:
        board = aruco.CharucoBoard_create(
            int(config.squares_x),
            int(config.squares_y),
            float(config.square_length_m),
            float(config.marker_length_m),
            dictionary,
        )
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(bool(config.legacy_pattern))
    return board


def _detector_parameters(aruco):
    if hasattr(aruco, "DetectorParameters"):
        return aruco.DetectorParameters()
    return aruco.DetectorParameters_create()


def board_object_points(board) -> np.ndarray:
    if hasattr(board, "getChessboardCorners"):
        points = board.getChessboardCorners()
    else:
        points = board.chessboardCorners
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def board_corner_count(config: BoardConfig) -> int:
    return int(board_object_points(make_charuco_board(config)).shape[0])


def marker_ids_from_layout(config: BoardConfig) -> np.ndarray | None:
    if not config.marker_id_layout:
        return None
    ids: list[int] = []
    for row in config.marker_id_layout:
        ids.extend(int(item) for item in row)
    return np.asarray(ids, dtype=np.int32)


def _marker_centers_from_layout(config: BoardConfig) -> dict[int, np.ndarray]:
    if not config.marker_id_layout:
        return {}
    centers: dict[int, np.ndarray] = {}
    sx = int(config.squares_x)
    square = float(config.square_length_m)
    for row_idx, row in enumerate(config.marker_id_layout):
        row_ids = [int(item) for item in row]
        if len(row_ids) == (sx + 1) // 2:
            x_cells = list(range(0, sx, 2))
        elif len(row_ids) == sx // 2:
            x_cells = list(range(1, sx, 2))
        else:
            raise ValueError(
                f"marker_id_layout row {row_idx} has {len(row_ids)} ids; expected {sx // 2} or {(sx + 1) // 2}"
            )
        for x_cell, marker_id in zip(x_cells, row_ids):
            centers[marker_id] = np.asarray([(x_cell + 0.5) * square, (row_idx + 0.5) * square], dtype=np.float64)
    return centers


def _transform_grid_points(points: np.ndarray, config: BoardConfig, mode: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]
    width = float(config.squares_x) * float(config.square_length_m)
    height = float(config.squares_y) * float(config.square_length_m)
    if mode == "identity":
        out = np.column_stack([x, y])
    elif mode == "rot180":
        out = np.column_stack([width - x, height - y])
    elif mode == "mirror_x":
        out = np.column_stack([width - x, y])
    elif mode == "mirror_y":
        out = np.column_stack([x, height - y])
    elif mode == "rot90":
        out = np.column_stack([y, width - x])
    elif mode == "rot270":
        out = np.column_stack([height - y, x])
    elif mode == "diag":
        out = np.column_stack([y, x])
    elif mode == "anti_diag":
        out = np.column_stack([height - y, width - x])
    else:
        raise ValueError(f"unknown board orientation mode: {mode}")
    return out.astype(np.float64)


def _canonical_corner_ids_for_points(points_xy: np.ndarray, config: BoardConfig) -> np.ndarray:
    square = float(config.square_length_m)
    ids = []
    for x, y in np.asarray(points_xy, dtype=np.float64).reshape(-1, 2):
        col = int(round(x / square)) - 1
        row = int(round(y / square)) - 1
        if col < 0 or row < 0 or col >= int(config.squares_x) - 1 or row >= int(config.squares_y) - 1:
            ids.append(-1)
        else:
            ids.append(row * (int(config.squares_x) - 1) + col)
    return np.asarray(ids, dtype=np.int32)


def detection_quality(detection: BoardDetection, config: BoardConfig) -> dict[str, Any]:
    total_corners = max(1, board_corner_count(config))
    corner_count = int(np.asarray(detection.ids).reshape(-1).size)
    coverage = float(corner_count / total_corners)
    accepted = bool(detection.T_cam_board is not None and corner_count >= int(config.min_corners))
    reasons = []
    if detection.T_cam_board is None:
        accepted = False
        if detection.reason:
            reasons.append(detection.reason)
        else:
            reasons.append("no pose")
    if corner_count < int(config.min_corners):
        accepted = False
        reasons.append(f"corner_count {corner_count} < {int(config.min_corners)}")
    if coverage < float(config.min_corner_coverage):
        accepted = False
        reasons.append(f"corner_coverage {coverage:.3f} < {float(config.min_corner_coverage):.3f}")
    if (
        config.max_reprojection_rmse_px is not None
        and detection.reprojection_rmse_px is not None
        and float(detection.reprojection_rmse_px) > float(config.max_reprojection_rmse_px)
    ):
        accepted = False
        reasons.append(
            "reprojection_rmse_px "
            f"{float(detection.reprojection_rmse_px):.3f} > {float(config.max_reprojection_rmse_px):.3f}"
        )
    return {
        "accepted": accepted,
        "corner_count": corner_count,
        "corner_total": total_corners,
        "corner_coverage": coverage,
        "reprojection_rmse_px": detection.reprojection_rmse_px,
        "reason": "; ".join(reasons),
    }


def detection_is_accepted(detection: BoardDetection, config: BoardConfig | None = None) -> bool:
    if config is not None:
        return bool(detection_quality(detection, config)["accepted"])
    if detection.quality:
        return bool(detection.quality.get("accepted", detection.ok))
    return bool(detection.ok and detection.T_cam_board is not None)


def detect_board_pose(
    image_bgr: np.ndarray,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray | None,
    config: BoardConfig,
) -> BoardDetection:
    aruco = _require_aruco()
    board = make_charuco_board(config)
    dictionary = board.getDictionary() if hasattr(board, "getDictionary") else board.dictionary
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    params = _detector_parameters(aruco)
    if hasattr(aruco, "ArucoDetector"):
        corners, ids, _ = aruco.ArucoDetector(dictionary, params).detectMarkers(gray)
    else:
        corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
    if ids is None or len(ids) == 0:
        det = BoardDetection(False, None, np.empty((0, 2)), np.empty((0,), dtype=np.int32), None, "no markers")
        det.quality = detection_quality(det, config)
        return det
    if hasattr(aruco, "interpolateCornersCharuco"):
        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            corners,
            ids,
            gray,
            board,
            cameraMatrix=np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
            distCoeffs=None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1),
        )
        corner_count = int(retval) if retval is not None else 0
    elif hasattr(aruco, "CharucoDetector"):
        detector = aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        corner_count = 0 if charuco_ids is None else int(np.asarray(charuco_ids).reshape(-1).size)
    else:
        raise RuntimeError("OpenCV aruco module has no ChArUco detector API.")
    required_corners = max(4, int(config.min_corners))
    if charuco_ids is None or charuco_corners is None or corner_count < required_corners:
        fallback = detect_chessboard_inner_pose(
            gray,
            intrinsic,
            dist_coeffs,
            config,
            marker_corners=corners,
            marker_ids=ids,
        )
        if fallback is not None:
            return fallback
        corners_arr = (
            np.empty((0, 2))
            if charuco_corners is None
            else np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
        )
        ids_arr = (
            np.empty((0,), dtype=np.int32)
            if charuco_ids is None
            else np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
        )
        det = BoardDetection(
            False,
            None,
            corners_arr,
            ids_arr,
            None,
            f"too few charuco corners: {corner_count}",
        )
        det.quality = detection_quality(det, config)
        return det
    charuco_ids_arr = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    if hasattr(board, "checkCharucoCornersCollinear") and board.checkCharucoCornersCollinear(charuco_ids_arr):
        det = BoardDetection(
            False,
            None,
            np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2),
            charuco_ids_arr,
            None,
            "charuco corners are collinear",
        )
        det.quality = detection_quality(det, config)
        return det
    dist = np.zeros((5,), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)
    object_points = board_object_points(board)[charuco_ids_arr]
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2),
        np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
        dist.reshape(-1),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        det = BoardDetection(
            False,
            None,
            np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2),
            charuco_ids_arr,
            None,
            "solvePnP failed",
        )
        det.quality = detection_quality(det, config)
        return det
    T_cam_board = rvec_tvec_to_transform(rvec, tvec)
    rmse = charuco_reprojection_rmse(
        T_cam_board,
        np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2),
        np.asarray(charuco_ids, dtype=np.int32).reshape(-1),
        intrinsic,
        dist,
        config,
    )
    det = BoardDetection(
        True,
        T_cam_board,
        np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2),
        charuco_ids_arr,
        float(rmse),
        "",
    )
    det.quality = detection_quality(det, config)
    det.ok = bool(det.quality["accepted"])
    det.reason = str(det.quality.get("reason") or "")
    return det


def detect_chessboard_inner_pose(
    gray: np.ndarray,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray | None,
    config: BoardConfig,
    *,
    marker_corners: list[np.ndarray] | tuple[np.ndarray, ...] | None,
    marker_ids: np.ndarray | None,
) -> BoardDetection | None:
    if not config.enable_chessboard_inner_corners or not config.marker_id_layout:
        return None
    if marker_corners is None or marker_ids is None or len(marker_corners) == 0:
        return None
    marker_layout = _marker_centers_from_layout(config)
    observed_marker_centers = {}
    for corners, marker_id in zip(marker_corners, np.asarray(marker_ids, dtype=np.int32).reshape(-1)):
        marker_id_int = int(marker_id)
        if marker_id_int in marker_layout:
            observed_marker_centers[marker_id_int] = np.asarray(corners, dtype=np.float64).reshape(4, 2).mean(axis=0)
    if len(observed_marker_centers) < 4:
        return None

    patterns = [
        (int(config.squares_x) - 1, int(config.squares_y) - 1),
        (int(config.squares_y) - 1, int(config.squares_x) - 1),
    ]
    orientation_modes = ["identity", "rot180", "mirror_x", "mirror_y", "rot90", "rot270", "diag", "anti_diag"]
    square = float(config.square_length_m)
    best: tuple[float, float, str, np.ndarray, np.ndarray, np.ndarray] | None = None
    for pattern in patterns:
        if pattern[0] <= 0 or pattern[1] <= 0:
            continue
        if hasattr(cv2, "findChessboardCornersSB"):
            ok, corners = cv2.findChessboardCornersSB(
                gray,
                pattern,
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
        else:
            ok, corners = False, None
        if not ok or corners is None:
            ok, corners = cv2.findChessboardCorners(
                gray,
                pattern,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if ok and corners is not None:
                cv2.cornerSubPix(
                    gray,
                    corners,
                    (5, 5),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3),
                )
        if not ok or corners is None:
            continue
        image_points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        detected_grid = np.asarray(
            [[(col + 1) * square, (row + 1) * square] for row in range(pattern[1]) for col in range(pattern[0])],
            dtype=np.float64,
        )
        for mode in orientation_modes:
            object_xy = _transform_grid_points(detected_grid, config, mode)
            corner_ids = _canonical_corner_ids_for_points(object_xy, config)
            if np.any(corner_ids < 0) or len(set(int(item) for item in corner_ids)) != len(corner_ids):
                continue
            homography, _ = cv2.findHomography(object_xy.astype(np.float32), image_points.astype(np.float32), 0)
            if homography is None:
                continue
            errors = []
            for marker_id, marker_xy in marker_layout.items():
                if marker_id not in observed_marker_centers:
                    continue
                projected = homography @ np.asarray([marker_xy[0], marker_xy[1], 1.0], dtype=np.float64)
                projected = projected[:2] / projected[2]
                errors.append(float(np.linalg.norm(projected - observed_marker_centers[marker_id])))
            if len(errors) < 4:
                continue
            mean_error = float(np.mean(errors))
            max_error = float(np.max(errors))
            if best is None or mean_error < best[0]:
                best = (mean_error, max_error, mode, object_xy, image_points, corner_ids)
    if best is None:
        return None
    mean_error, max_error, mode, object_xy, image_points, corner_ids = best
    if max_error > float(config.max_orientation_error_px):
        return None
    object_points = np.column_stack([object_xy, np.zeros((object_xy.shape[0],), dtype=np.float64)])
    dist = np.zeros((5,), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    T_cam_board = rvec_tvec_to_transform(rvec, tvec)
    rmse = charuco_reprojection_rmse(
        T_cam_board,
        image_points,
        corner_ids,
        intrinsic,
        dist,
        config,
    )
    det = BoardDetection(
        True,
        T_cam_board,
        image_points,
        corner_ids,
        float(rmse),
        "",
    )
    det.quality = detection_quality(det, config)
    det.quality.update(
        {
            "source": "chessboard_inner_corners",
            "orientation_mode": mode,
            "orientation_marker_mean_error_px": mean_error,
            "orientation_marker_max_error_px": max_error,
        }
    )
    det.ok = bool(det.quality["accepted"])
    det.reason = str(det.quality.get("reason") or "")
    return det


def charuco_reprojection_rmse(
    T_cam_board: np.ndarray,
    corners: np.ndarray,
    ids: np.ndarray,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray | None,
    config: BoardConfig,
) -> float:
    board = make_charuco_board(config)
    object_points = board_object_points(board)[np.asarray(ids, dtype=np.int32).reshape(-1)]
    rvec, tvec = transform_to_rvec_tvec(T_cam_board)
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
        np.zeros((5,), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1),
    )
    projected = projected.reshape(-1, 2)
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=1))))


def draw_board_overlay(
    image_bgr: np.ndarray,
    detection: BoardDetection,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray | None,
    output_path: str | Path,
) -> None:
    out = draw_board_overlay_image(image_bgr, detection, intrinsic, dist_coeffs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out)


def draw_board_overlay_image(
    image_bgr: np.ndarray,
    detection: BoardDetection,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray | None,
) -> np.ndarray:
    out = image_bgr.copy()
    if detection.T_cam_board is not None:
        rvec, tvec = transform_to_rvec_tvec(detection.T_cam_board)
        cv2.drawFrameAxes(
            out,
            np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
            np.zeros((5,), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1),
            rvec,
            tvec,
            0.07,
        )
        color = (0, 255, 0) if detection.ok else (0, 165, 255)
        for point in detection.corners.reshape(-1, 2):
            cv2.circle(out, tuple(np.round(point).astype(int)), 3, color, -1)
    label = "ok" if detection.ok else (detection.reason or "rejected")
    if detection.reprojection_rmse_px is not None:
        label += f" rmse={float(detection.reprojection_rmse_px):.2f}px"
    cv2.putText(out, label[:100], (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 220, 30) if detection.ok else (0, 165, 255), 2)
    return out


def hand_eye_methods() -> dict[str, int]:
    candidates = {
        "TSAI": "CALIB_HAND_EYE_TSAI",
        "PARK": "CALIB_HAND_EYE_PARK",
        "HORAUD": "CALIB_HAND_EYE_HORAUD",
        "ANDREFF": "CALIB_HAND_EYE_ANDREFF",
        "DANIILIDIS": "CALIB_HAND_EYE_DANIILIDIS",
    }
    return {name: int(getattr(cv2, attr)) for name, attr in candidates.items() if hasattr(cv2, attr)}


def solve_eye_in_hand(
    base_T_ee_list: list[np.ndarray],
    cam_T_board_list: list[np.ndarray],
    *,
    method: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(base_T_ee_list) != len(cam_T_board_list):
        raise ValueError("base_T_ee_list and cam_T_board_list length mismatch")
    if len(base_T_ee_list) < 3:
        raise ValueError("at least 3 board observations are required")
    method = cv2.CALIB_HAND_EYE_TSAI if method is None else int(method)
    base_T_ee_list = [as_transform(t) for t in base_T_ee_list]
    cam_T_board_list = [as_transform(t) for t in cam_T_board_list]
    rotations_gripper_to_base = [t[:3, :3] for t in base_T_ee_list]
    translations_gripper_to_base = [t[:3, 3] for t in base_T_ee_list]
    rotations_target_to_cam = [t[:3, :3] for t in cam_T_board_list]
    translations_target_to_cam = [t[:3, 3] for t in cam_T_board_list]
    R_cam_to_gripper, t_cam_to_gripper = cv2.calibrateHandEye(
        rotations_gripper_to_base,
        translations_gripper_to_base,
        rotations_target_to_cam,
        translations_target_to_cam,
        method=method,
    )
    ee_T_cam = make_transform(R_cam_to_gripper, np.asarray(t_cam_to_gripper).reshape(3))
    residuals = board_pose_residuals(base_T_ee_list, cam_T_board_list, ee_T_cam)
    method_name = next((name for name, value in hand_eye_methods().items() if value == method), str(method))
    residuals["method"] = method_name
    return ee_T_cam, residuals


def solve_eye_in_hand_multi_method(
    base_T_ee_list: list[np.ndarray],
    cam_T_board_list: list[np.ndarray],
    *,
    method_names: list[str] | None = None,
    rotation_weight_m_per_deg: float = 0.002,
) -> tuple[np.ndarray, dict[str, Any]]:
    available = hand_eye_methods()
    selected_names = method_names or list(available.keys())
    candidates = []
    for name in selected_names:
        method = available.get(str(name).upper())
        if method is None:
            candidates.append({"method": str(name), "ok": False, "reason": "method unavailable"})
            continue
        try:
            ee_T_cam, metrics = solve_eye_in_hand(base_T_ee_list, cam_T_board_list, method=method)
            if not np.all(np.isfinite(ee_T_cam)):
                raise ValueError("non-finite transform")
            score = float(metrics["mean_translation_m"]) + float(rotation_weight_m_per_deg) * float(metrics["mean_rotation_deg"])
            candidates.append(
                {
                    "method": name,
                    "ok": True,
                    "score": score,
                    "ee_T_cam": ee_T_cam,
                    "metrics": metrics,
                }
            )
        except Exception as exc:
            candidates.append({"method": name, "ok": False, "reason": str(exc)})
    valid = [item for item in candidates if item.get("ok")]
    if not valid:
        raise RuntimeError(f"no hand-eye method succeeded: {candidates}")
    best = min(valid, key=lambda item: float(item["score"]))
    report = {
        "selected_method": best["method"],
        "selection_score": best["score"],
        "rotation_weight_m_per_deg": float(rotation_weight_m_per_deg),
        "candidates": candidates,
        **best["metrics"],
    }
    return as_transform(best["ee_T_cam"]), report


def pose_diversity_metrics(transforms: list[np.ndarray]) -> dict[str, Any]:
    mats = [as_transform(t) for t in transforms]
    if len(mats) < 2:
        return {
            "count": len(mats),
            "max_translation_span_m": 0.0,
            "mean_pair_translation_m": 0.0,
            "max_pair_rotation_deg": 0.0,
            "mean_pair_rotation_deg": 0.0,
        }
    translations = []
    rotations = []
    for idx in range(len(mats)):
        for jdx in range(idx + 1, len(mats)):
            err = pose_error(mats[idx], mats[jdx])
            translations.append(err["translation_m"])
            rotations.append(err["rotation_deg"])
    return {
        "count": len(mats),
        "max_translation_span_m": float(np.max(translations)),
        "mean_pair_translation_m": float(np.mean(translations)),
        "max_pair_rotation_deg": float(np.max(rotations)),
        "mean_pair_rotation_deg": float(np.mean(rotations)),
    }


def board_pose_residuals(
    base_T_ee_list: list[np.ndarray],
    cam_T_board_list: list[np.ndarray],
    ee_T_cam: np.ndarray,
) -> dict[str, Any]:
    estimates = [
        as_transform(base_T_ee) @ as_transform(ee_T_cam) @ as_transform(cam_T_board)
        for base_T_ee, cam_T_board in zip(base_T_ee_list, cam_T_board_list)
    ]
    base_T_board = average_transforms(estimates)
    errors = [pose_error(est, base_T_board) for est in estimates]
    return {
        "base_T_board": base_T_board,
        "mean_translation_m": float(np.mean([e["translation_m"] for e in errors])),
        "max_translation_m": float(np.max([e["translation_m"] for e in errors])),
        "mean_rotation_deg": float(np.mean([e["rotation_deg"] for e in errors])),
        "max_rotation_deg": float(np.max([e["rotation_deg"] for e in errors])),
        "per_frame": errors,
    }


def optimize_hand_eye_reprojection(
    base_T_ee_list: list[np.ndarray],
    detections: list[BoardDetection],
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray | None,
    config: BoardConfig,
    initial_ee_T_cam: np.ndarray,
    *,
    fixed_base_T_board: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        from scipy.optimize import least_squares
    except Exception:
        residuals = board_pose_residuals(
            base_T_ee_list,
            [as_transform(det.T_cam_board) for det in detections if det.ok and det.T_cam_board is not None],
            initial_ee_T_cam,
        )
        return as_transform(initial_ee_T_cam), as_transform(residuals["base_T_board"]), {
            "optimized": False,
            "reason": "scipy is unavailable",
            "pose_residuals": residuals,
        }

    valid_pairs = [
        (as_transform(base_T_ee), det)
        for base_T_ee, det in zip(base_T_ee_list, detections)
        if det.ok and det.T_cam_board is not None and det.ids.size > 0
    ]
    if len(valid_pairs) < 3:
        raise ValueError("at least 3 valid detections are required for reprojection optimization")
    base_T_ee_valid = [item[0] for item in valid_pairs]
    detections_valid = [item[1] for item in valid_pairs]
    initial_residual = board_pose_residuals(
        base_T_ee_valid,
        [as_transform(det.T_cam_board) for det in detections_valid],
        initial_ee_T_cam,
    )
    initial_base_T_board = as_transform(
        fixed_base_T_board if fixed_base_T_board is not None else initial_residual["base_T_board"]
    )
    ee_rvec, ee_tvec = transform_to_rvec_tvec(initial_ee_T_cam)
    board_rvec, board_tvec = transform_to_rvec_tvec(initial_base_T_board)
    if fixed_base_T_board is None:
        x0 = np.concatenate([ee_rvec, ee_tvec, board_rvec, board_tvec])
    else:
        x0 = np.concatenate([ee_rvec, ee_tvec])

    board_points = board_object_points(make_charuco_board(config))
    K = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    dist = np.zeros((5,), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ee = rvec_tvec_to_transform(params[0:3], params[3:6])
        if fixed_base_T_board is None:
            board_pose = rvec_tvec_to_transform(params[6:9], params[9:12])
        else:
            board_pose = initial_base_T_board
        return ee, board_pose

    def residual_fn(params: np.ndarray) -> np.ndarray:
        ee_T_cam, base_T_board = unpack(params)
        residual_chunks = []
        for base_T_ee, det in zip(base_T_ee_valid, detections_valid):
            base_T_cam = base_T_ee @ ee_T_cam
            cam_T_board = invert_transform(base_T_cam) @ base_T_board
            rvec, tvec = transform_to_rvec_tvec(cam_T_board)
            obj_pts = board_points[np.asarray(det.ids, dtype=np.int32).reshape(-1)]
            projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
            residual_chunks.append((projected.reshape(-1, 2) - det.corners.reshape(-1, 2)).reshape(-1))
        return np.concatenate(residual_chunks)

    before = residual_fn(x0)
    result = least_squares(residual_fn, x0, loss="huber", f_scale=2.0, max_nfev=200)
    after = residual_fn(result.x)
    ee_T_cam, base_T_board = unpack(result.x)
    metrics = {
        "optimized": True,
        "success": bool(result.success),
        "message": str(result.message),
        "frames": len(valid_pairs),
        "initial_reprojection_rmse_px": float(np.sqrt(np.mean(before * before))),
        "final_reprojection_rmse_px": float(np.sqrt(np.mean(after * after))),
        "initial_pose_residuals": initial_residual,
        "final_pose_residuals": board_pose_residuals(
            base_T_ee_valid,
            [invert_transform(base_T_ee @ ee_T_cam) @ base_T_board for base_T_ee in base_T_ee_valid],
            ee_T_cam,
        ),
    }
    return ee_T_cam, base_T_board, metrics


def save_detections(path: str | Path, detections: list[BoardDetection]) -> None:
    save_json(path, [to_jsonable(det.jsonable()) for det in detections])
