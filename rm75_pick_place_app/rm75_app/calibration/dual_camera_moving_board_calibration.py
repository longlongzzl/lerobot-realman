from __future__ import annotations

import argparse
import select
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rm75_app.paths import ASSET_DIR, DEFAULT_CAMERA_EXTRINSIC, DEFAULT_RM75_URDF

from .board import (
    BoardConfig,
    BoardDetection,
    board_config_from_args,
    board_object_points,
    detection_is_accepted,
    detect_board_pose,
    draw_board_overlay_image,
    make_charuco_board,
    pose_diversity_metrics,
)
from .common import (
    RealSenseCamera,
    RealmanSdkController,
    as_transform,
    average_transforms,
    calibration_run_dir,
    camera_opencv_to_base_camera,
    image_write,
    invert_transform,
    link_poses_from_qpos,
    load_json,
    load_matrix,
    load_realman_settings,
    load_urdf,
    pose_error,
    rvec_tvec_to_transform,
    rotation_error_deg,
    save_json,
    save_matrix_pair,
    select_link_pose,
    transform_to_rvec_tvec,
)
from .reporting import normalize_calibration_report, write_html_report


@dataclass
class DualFrame:
    index: int
    qpos_rad: np.ndarray
    base_T_ee: np.ndarray
    main_detection: BoardDetection
    wrist_detection: BoardDetection
    base_T_board: np.ndarray | None
    candidate_ee_T_wrist: np.ndarray | None
    main_image_path: Path | None = None
    wrist_image_path: Path | None = None
    accepted_for_optimization: bool = False
    rejection_reason: str = ""
    frame_weight: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the wrist camera from paired global-camera and wrist-camera observations of a moving ChArUco board."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ASSET_DIR / "calibration" / "rm75_calibration_config.example.json",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--input-run", type=Path, default=None, help="Reprocess a previous dual-board run without recapturing.")
    parser.add_argument("--base-camera-run-dir", type=Path, default=None)
    parser.add_argument("--base-camera-extrinsic-opencv-path", type=Path, default=DEFAULT_CAMERA_EXTRINSIC)
    parser.add_argument("--base-camera-use-direct", action="store_true")
    parser.add_argument("--robot-ip", type=str, default=None)
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_RM75_URDF)
    parser.add_argument("--ee-link-name", type=str, default=None)
    parser.add_argument("--joint-names", nargs="+", default=None)
    parser.add_argument("--main-camera-serial", type=str, default=None)
    parser.add_argument("--wrist-camera-serial", type=str, default=None)
    parser.add_argument("--main-camera-width", type=int, default=None)
    parser.add_argument("--main-camera-height", type=int, default=None)
    parser.add_argument("--main-camera-fps", type=int, default=None)
    parser.add_argument("--wrist-camera-width", type=int, default=None)
    parser.add_argument("--wrist-camera-height", type=int, default=None)
    parser.add_argument("--wrist-camera-fps", type=int, default=None)
    parser.add_argument("--wrist-camera-exposure-us", type=float, default=None)
    parser.add_argument("--wrist-camera-gain", type=float, default=None)
    parser.add_argument("--wrist-camera-auto-exposure", action="store_true", default=None)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--manual-count", type=int, default=40)
    parser.add_argument("--manual-terminal", action="store_true")
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--min-valid-frames", type=int, default=15)
    parser.add_argument("--loss", choices=["linear", "soft_l1", "huber", "cauchy"], default="soft_l1")
    parser.add_argument("--f-scale", type=float, default=2.0)
    parser.add_argument("--max-nfev", type=int, default=200)
    parser.add_argument("--max-final-reprojection-rmse-px", type=float, default=6.0)
    parser.add_argument("--max-final-frame-reprojection-rmse-px", type=float, default=12.0)
    parser.add_argument("--max-final-pose-residual-mm", type=float, default=10.0)
    parser.add_argument("--max-final-pose-residual-deg", type=float, default=2.5)
    parser.add_argument("--allow-worse-reprojection", action="store_true")
    parser.add_argument("--force-accept", action="store_true")
    parser.add_argument("--max-overlays", type=int, default=80)
    parser.add_argument("--no-html-report", action="store_true")
    return parser.parse_args()


def _matrix_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.size != 16:
        return None
    return as_transform(arr.reshape(4, 4))


def _load_detection(item: dict[str, Any]) -> BoardDetection:
    return BoardDetection(
        ok=bool(item.get("ok", False)),
        T_cam_board=_matrix_or_none(item.get("T_cam_board")),
        corners=np.asarray(item.get("corners", []), dtype=np.float64).reshape(-1, 2),
        ids=np.asarray(item.get("ids", []), dtype=np.int32).reshape(-1),
        reprojection_rmse_px=item.get("reprojection_rmse_px"),
        reason=str(item.get("reason", "")),
        quality=dict(item.get("quality", {})) if isinstance(item.get("quality"), dict) else {},
    )


def _load_base_camera_transform(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.base_camera_run_dir is not None:
        run_dir = Path(args.base_camera_run_dir).expanduser().resolve()
        for name in ("base_T_camera.npy", "R_T_global_camera.npy", "R_T_camera.npy", "T_R_Cg.npy"):
            path = run_dir / name
            if path.exists():
                return load_matrix(path), str(path)
        path = run_dir / "camera_extrinsic_opencv.npy"
        if path.exists():
            return camera_opencv_to_base_camera(load_matrix(path), use_direct=bool(args.base_camera_use_direct)), str(path)
    path = Path(args.base_camera_extrinsic_opencv_path).expanduser()
    return camera_opencv_to_base_camera(load_matrix(path), use_direct=bool(args.base_camera_use_direct)), str(path)


def _manual_command_from_stdin() -> str | None:
    if not sys.stdin.isatty():
        return None
    try:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
    except (OSError, ValueError):
        return None
    if not readable:
        return None
    value = sys.stdin.readline().strip().lower()
    if value in {"q", "quit", "done", "exit"}:
        return "finish"
    return "capture"


def _manual_command_from_key(key: int) -> str | None:
    if key < 0:
        return None
    key &= 0xFF
    if key in (27, ord("q")):
        return "finish"
    if key in (10, 13, 32):
        return "capture"
    return None


def _put_label(image: np.ndarray, lines: list[str]) -> np.ndarray:
    out = image.copy()
    width = out.shape[1]
    overlay = out.copy()
    cv2.rectangle(overlay, (8, 8), (min(width - 8, 940), 78), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0.0, out)
    for idx, line in enumerate(lines[:2]):
        cv2.putText(out, line[:120], (18, 34 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2)
    return out


def _stack_preview(main_bgr: np.ndarray, wrist_bgr: np.ndarray, *, sample_index: int, target_count: int) -> np.ndarray:
    h = min(main_bgr.shape[0], wrist_bgr.shape[0])
    main = cv2.resize(main_bgr, (int(main_bgr.shape[1] * h / main_bgr.shape[0]), h))
    wrist = cv2.resize(wrist_bgr, (int(wrist_bgr.shape[1] * h / wrist_bgr.shape[0]), h))
    out = np.concatenate([main, wrist], axis=1)
    return _put_label(
        out,
        [
            f"dual board sample {sample_index:02d}/{target_count:02d}   Enter/Space=capture   q/Esc=finish",
            "left=global camera, right=wrist camera",
        ],
    )


def _camera_from_settings(
    settings: dict[str, Any],
    section: str,
    *,
    serial: str | None,
    width: int | None,
    height: int | None,
    fps: int | None,
    warmup_frames: int,
    auto_exposure: bool | None = None,
    exposure_us: float | None = None,
    gain: float | None = None,
) -> RealSenseCamera:
    cfg = settings.get(section, settings.get("camera", {}))
    return RealSenseCamera(
        serial=serial or cfg.get("serial_number_or_name"),
        width=int(width or cfg.get("width", 1280)),
        height=int(height or cfg.get("height", 720)),
        fps=int(fps or cfg.get("fps", 30)),
        warmup_frames=int(warmup_frames),
        auto_exposure=cfg.get("auto_exposure") if auto_exposure is None else auto_exposure,
        exposure_us=exposure_us if exposure_us is not None else cfg.get("exposure_us"),
        gain=gain if gain is not None else cfg.get("gain"),
    )


def _detect_pair(
    main_image: np.ndarray,
    wrist_image: np.ndarray,
    main_intrinsic: np.ndarray,
    main_dist: np.ndarray,
    wrist_intrinsic: np.ndarray,
    wrist_dist: np.ndarray,
    board_cfg: BoardConfig,
) -> tuple[BoardDetection, BoardDetection]:
    return (
        detect_board_pose(main_image, main_intrinsic, main_dist, board_cfg),
        detect_board_pose(wrist_image, wrist_intrinsic, wrist_dist, board_cfg),
    )


def _collect_observations(
    args: argparse.Namespace,
    settings: dict[str, Any],
    run_dir: Path,
    base_T_main_camera: np.ndarray,
    board_cfg: BoardConfig,
) -> tuple[list[DualFrame], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    robot_ip = args.robot_ip or settings.get("robot_ip")
    if not robot_ip:
        raise ValueError("robot_ip is required in --config or --robot-ip")
    joint_names = list(args.joint_names or settings.get("joint_names") or [f"joint_{idx}" for idx in range(1, 8)])
    urdf = load_urdf(args.urdf_path or settings.get("urdf_path") or DEFAULT_RM75_URDF)
    ee_link_name = args.ee_link_name or settings.get("ee_link_name")
    main_cam = _camera_from_settings(
        settings,
        "base_camera",
        serial=args.main_camera_serial,
        width=args.main_camera_width,
        height=args.main_camera_height,
        fps=args.main_camera_fps,
        warmup_frames=int(args.warmup_frames),
    )
    wrist_cam = _camera_from_settings(
        settings,
        "wrist_camera",
        serial=args.wrist_camera_serial,
        width=args.wrist_camera_width,
        height=args.wrist_camera_height,
        fps=args.wrist_camera_fps,
        warmup_frames=int(args.warmup_frames),
        auto_exposure=args.wrist_camera_auto_exposure,
        exposure_us=args.wrist_camera_exposure_us,
        gain=args.wrist_camera_gain,
    )
    controller = RealmanSdkController(
        robot_ip=str(robot_ip),
        robot_port=int(args.robot_port or settings.get("robot_port", 8080)),
        joint_names=joint_names,
        calibration_offset=settings.get("calibration_offset", {}),
    )
    frames: list[DualFrame] = []
    main_intrinsic = main_dist = wrist_intrinsic = wrist_dist = None
    target_count = int(args.manual_count)
    window_name = "rm75 dual board calibration"
    print("[dual-board] Move the arm and/or board, then press Enter/Space to capture.")
    print("[dual-board] The board may move between samples. Keep both cameras seeing the board for each capture.")
    try:
        main_cam.start()
        wrist_cam.start()
        while len(frames) < target_count:
            main_frame = main_cam.capture()
            wrist_frame = wrist_cam.capture()
            main_intrinsic = main_frame.intrinsic
            main_dist = main_frame.dist_coeffs
            wrist_intrinsic = wrist_frame.intrinsic
            wrist_dist = wrist_frame.dist_coeffs
            main_det, wrist_det = _detect_pair(
                main_frame.color_bgr,
                wrist_frame.color_bgr,
                main_intrinsic,
                main_dist,
                wrist_intrinsic,
                wrist_dist,
                board_cfg,
            )
            main_overlay = draw_board_overlay_image(main_frame.color_bgr, main_det, main_intrinsic, main_dist)
            wrist_overlay = draw_board_overlay_image(wrist_frame.color_bgr, wrist_det, wrist_intrinsic, wrist_dist)
            if args.preview and not args.manual_terminal:
                cv2.imshow(
                    window_name,
                    _stack_preview(main_overlay, wrist_overlay, sample_index=len(frames), target_count=target_count),
                )
                key_command = _manual_command_from_key(cv2.waitKey(max(1, int(args.preview_delay_ms))))
            else:
                key_command = None
            stdin_command = _manual_command_from_stdin()
            command = key_command or stdin_command
            if args.manual_terminal and command is None:
                value = input(f"[dual-board] sample {len(frames):04d}/{target_count}: Enter=capture, q=finish > ")
                command = "finish" if value.strip().lower() in {"q", "quit", "done", "exit"} else "capture"
            if command == "finish":
                break
            if command != "capture":
                continue
            qpos = np.asarray(controller.get_qpos(), dtype=np.float64).reshape(-1)
            link_poses = link_poses_from_qpos(urdf, qpos, joint_names)
            _, base_T_ee = select_link_pose(link_poses, ee_link_name)
            idx = len(frames)
            main_path = run_dir / "main_images" / f"{idx:04d}.png"
            wrist_path = run_dir / "wrist_images" / f"{idx:04d}.png"
            image_write(main_path, main_frame.color_bgr)
            image_write(wrist_path, wrist_frame.color_bgr)
            frame = _make_dual_frame(
                idx,
                qpos,
                base_T_ee,
                main_det,
                wrist_det,
                base_T_main_camera,
                main_image_path=main_path,
                wrist_image_path=wrist_path,
            )
            frames.append(frame)
            print(
                f"[dual-board] sample {idx:04d}: "
                f"main_ok={detection_is_accepted(main_det, board_cfg)} "
                f"wrist_ok={detection_is_accepted(wrist_det, board_cfg)} "
                f"main_rmse={main_det.reprojection_rmse_px} wrist_rmse={wrist_det.reprojection_rmse_px} "
                f"reason={frame.rejection_reason}"
            )
    finally:
        if args.preview and not args.manual_terminal:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
        main_cam.stop()
        wrist_cam.stop()
        controller.disconnect()
    if not frames or main_intrinsic is None or wrist_intrinsic is None or main_dist is None or wrist_dist is None:
        raise RuntimeError("No dual-board frames were captured.")
    metadata = {"joint_names": joint_names, "ee_link_name": ee_link_name}
    return frames, main_intrinsic, main_dist, wrist_intrinsic, wrist_dist, metadata


def _make_dual_frame(
    index: int,
    qpos_rad: np.ndarray,
    base_T_ee: np.ndarray,
    main_det: BoardDetection,
    wrist_det: BoardDetection,
    base_T_main_camera: np.ndarray,
    *,
    main_image_path: Path | None,
    wrist_image_path: Path | None,
) -> DualFrame:
    base_T_board = None
    candidate = None
    reason = ""
    main_ok = main_det.T_cam_board is not None
    wrist_ok = wrist_det.T_cam_board is not None
    if main_ok:
        base_T_board = as_transform(base_T_main_camera) @ as_transform(main_det.T_cam_board)
    if main_ok and wrist_ok:
        candidate = invert_transform(base_T_ee) @ as_transform(base_T_board) @ invert_transform(wrist_det.T_cam_board)
    else:
        missing = []
        if not main_ok:
            missing.append("global board pose")
        if not wrist_ok:
            missing.append("wrist board pose")
        reason = "missing " + " and ".join(missing)
    return DualFrame(
        index=int(index),
        qpos_rad=np.asarray(qpos_rad, dtype=np.float64).reshape(-1),
        base_T_ee=as_transform(base_T_ee),
        main_detection=main_det,
        wrist_detection=wrist_det,
        base_T_board=base_T_board,
        candidate_ee_T_wrist=candidate,
        main_image_path=main_image_path,
        wrist_image_path=wrist_image_path,
        rejection_reason=reason,
    )


def _save_observations(
    run_dir: Path,
    frames: list[DualFrame],
    main_intrinsic: np.ndarray,
    main_dist: np.ndarray,
    wrist_intrinsic: np.ndarray,
    wrist_dist: np.ndarray,
    *,
    base_camera_source: str,
    base_T_main_camera: np.ndarray,
    board_cfg: BoardConfig,
    metadata: dict[str, Any],
) -> None:
    np.save(run_dir / "main_camera_intrinsic.npy", np.asarray(main_intrinsic, dtype=np.float64).reshape(3, 3))
    np.save(run_dir / "main_camera_dist_coeffs.npy", np.asarray(main_dist, dtype=np.float64).reshape(-1))
    np.save(run_dir / "wrist_camera_intrinsic.npy", np.asarray(wrist_intrinsic, dtype=np.float64).reshape(3, 3))
    np.save(run_dir / "wrist_camera_dist_coeffs.npy", np.asarray(wrist_dist, dtype=np.float64).reshape(-1))
    np.save(run_dir / "base_T_ee.npy", np.stack([frame.base_T_ee for frame in frames], axis=0))
    np.save(run_dir / "qpos_rad.npy", np.stack([frame.qpos_rad for frame in frames], axis=0))
    save_matrix_pair(run_dir / "base_T_main_camera.npy", base_T_main_camera)

    def rel_or_abs(path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return str(Path(path).resolve().relative_to(run_dir.resolve()))
        except ValueError:
            return str(Path(path).expanduser().resolve())

    payload = {
        "coordinate_convention": "T_A_B maps coordinates from frame B into frame A",
        "base_camera_source": base_camera_source,
        "base_T_main_camera": base_T_main_camera,
        "board_config": asdict(board_cfg),
        "metadata": metadata,
        "frames": [
            {
                "index": frame.index,
                "qpos_rad": frame.qpos_rad,
                "base_T_ee": frame.base_T_ee,
                "main_image": rel_or_abs(frame.main_image_path),
                "wrist_image": rel_or_abs(frame.wrist_image_path),
                "main_detection": frame.main_detection.jsonable(),
                "wrist_detection": frame.wrist_detection.jsonable(),
                "base_T_board_from_main": frame.base_T_board,
                "candidate_ee_T_wrist_camera": frame.candidate_ee_T_wrist,
                "accepted_for_optimization": frame.accepted_for_optimization,
                "rejection_reason": frame.rejection_reason,
                "frame_weight": frame.frame_weight,
            }
            for frame in frames
        ],
    }
    save_json(run_dir / "observations.json", payload)


def _load_input_run(
    args: argparse.Namespace,
    settings: dict[str, Any],
    run_dir: Path,
    base_T_main_camera: np.ndarray,
    board_cfg: BoardConfig,
) -> tuple[list[DualFrame], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    source = Path(args.input_run).expanduser().resolve()
    data = load_json(source / "observations.json")
    main_intrinsic = np.load(source / "main_camera_intrinsic.npy")
    main_dist = np.load(source / "main_camera_dist_coeffs.npy")
    wrist_intrinsic = np.load(source / "wrist_camera_intrinsic.npy")
    wrist_dist = np.load(source / "wrist_camera_dist_coeffs.npy")
    frames = []
    for order, item in enumerate(data.get("frames", [])):
        idx = int(item.get("index", order))
        main_path = source / str(item.get("main_image", f"main_images/{idx:04d}.png"))
        wrist_path = source / str(item.get("wrist_image", f"wrist_images/{idx:04d}.png"))
        main_det = _load_detection(item.get("main_detection", {}))
        wrist_det = _load_detection(item.get("wrist_detection", {}))
        frames.append(
            _make_dual_frame(
                idx,
                np.asarray(item["qpos_rad"], dtype=np.float64).reshape(-1),
                as_transform(item["base_T_ee"]),
                main_det,
                wrist_det,
                base_T_main_camera,
                main_image_path=main_path if main_path.exists() else None,
                wrist_image_path=wrist_path if wrist_path.exists() else None,
            )
        )
    metadata = dict(data.get("metadata", {})) if isinstance(data.get("metadata"), dict) else {}
    _save_observations(
        run_dir,
        frames,
        main_intrinsic,
        main_dist,
        wrist_intrinsic,
        wrist_dist,
        base_camera_source=str(data.get("base_camera_source", "")),
        base_T_main_camera=base_T_main_camera,
        board_cfg=board_cfg,
        metadata=metadata | {"source_run": source},
    )
    return frames, main_intrinsic, main_dist, wrist_intrinsic, wrist_dist, metadata


def _median_abs_deviation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.median(np.abs(values - np.median(values))))


def _spread_metrics(transforms: list[np.ndarray]) -> dict[str, Any]:
    if not transforms:
        return {"count": 0}
    center = average_transforms([as_transform(item) for item in transforms])
    translations = np.asarray([np.linalg.norm(as_transform(item)[:3, 3] - center[:3, 3]) * 1000.0 for item in transforms])
    rotations = np.asarray([rotation_error_deg(as_transform(item), center) for item in transforms])
    return {
        "count": len(transforms),
        "translation_mm_median": float(np.median(translations)),
        "translation_mm_mean": float(np.mean(translations)),
        "translation_mm_max": float(np.max(translations)),
        "rotation_deg_median": float(np.median(rotations)),
        "rotation_deg_mean": float(np.mean(rotations)),
        "rotation_deg_max": float(np.max(rotations)),
    }


def _prepare_frames(frames: list[DualFrame], board_cfg: BoardConfig, *, min_valid_frames: int) -> tuple[np.ndarray, dict[str, Any]]:
    candidates = []
    for frame in frames:
        main_ok = detection_is_accepted(frame.main_detection, board_cfg)
        wrist_ok = detection_is_accepted(frame.wrist_detection, board_cfg)
        if not main_ok:
            frame.rejection_reason = frame.main_detection.reason or "global board detection rejected"
            continue
        if not wrist_ok:
            frame.rejection_reason = frame.wrist_detection.reason or "wrist board detection rejected"
            continue
        if frame.candidate_ee_T_wrist is None or frame.base_T_board is None:
            frame.rejection_reason = frame.rejection_reason or "missing candidate transform"
            continue
        candidates.append((len(candidates), frame, as_transform(frame.candidate_ee_T_wrist)))
    if len(candidates) < max(3, int(min_valid_frames)):
        raise RuntimeError(f"Only {len(candidates)} paired valid frames. Need at least {max(3, int(min_valid_frames))}.")

    transforms = [item[2] for item in candidates]
    preliminary = average_transforms(transforms)
    translations = np.stack([mat[:3, 3] for mat in transforms])
    center_t = np.median(translations, axis=0)
    translation_errors = np.linalg.norm(translations - center_t, axis=1)
    rotation_errors = np.asarray([rotation_error_deg(mat, preliminary) for mat in transforms])
    trans_threshold = max(
        float(np.median(translation_errors) + 3.0 * 1.4826 * _median_abs_deviation(translation_errors)),
        0.03,
    )
    rot_threshold = max(
        float(np.median(rotation_errors) + 3.0 * 1.4826 * _median_abs_deviation(rotation_errors)),
        8.0,
    )
    selected = []
    outliers = []
    for local_idx, frame, transform in candidates:
        if translation_errors[local_idx] <= trans_threshold and rotation_errors[local_idx] <= rot_threshold:
            frame.accepted_for_optimization = True
            selected.append(transform)
        else:
            frame.accepted_for_optimization = False
            frame.rejection_reason = (
                f"candidate outlier: translation={translation_errors[local_idx] * 1000.0:.2f}mm "
                f"rotation={rotation_errors[local_idx]:.2f}deg"
            )
            outliers.append(frame.index)
    if len(selected) < max(3, int(min_valid_frames)):
        for _, frame, transform in candidates:
            frame.accepted_for_optimization = True
        selected = transforms
    _assign_frame_weights([frame for _, frame, _ in candidates])
    initial = average_transforms(selected)
    return initial, {
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "translation_outlier_threshold_mm": float(trans_threshold * 1000.0),
        "rotation_outlier_threshold_deg": float(rot_threshold),
        "outlier_frames": outliers,
        "candidate_spread_all": _spread_metrics(transforms),
        "candidate_spread_selected": _spread_metrics(selected),
    }


def _assign_frame_weights(frames: list[DualFrame]) -> None:
    main_rmses = []
    wrist_rmses = []
    for frame in frames:
        if frame.main_detection.reprojection_rmse_px is not None:
            main_rmses.append(max(0.05, float(frame.main_detection.reprojection_rmse_px)))
        if frame.wrist_detection.reprojection_rmse_px is not None:
            wrist_rmses.append(max(0.05, float(frame.wrist_detection.reprojection_rmse_px)))
    median_main = max(0.05, float(np.median(main_rmses))) if main_rmses else 1.0
    median_wrist = max(0.05, float(np.median(wrist_rmses))) if wrist_rmses else 1.0
    for frame in frames:
        main_rmse = median_main if frame.main_detection.reprojection_rmse_px is None else max(0.05, float(frame.main_detection.reprojection_rmse_px))
        wrist_rmse = median_wrist if frame.wrist_detection.reprojection_rmse_px is None else max(0.05, float(frame.wrist_detection.reprojection_rmse_px))
        frame.frame_weight = float(np.clip(np.sqrt(median_main / main_rmse) * np.sqrt(median_wrist / wrist_rmse), 0.35, 2.5))


def _project_wrist_frame(
    frame: DualFrame,
    ee_T_wrist: np.ndarray,
    board_points: np.ndarray,
    wrist_intrinsic: np.ndarray,
    wrist_dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if frame.base_T_board is None:
        raise ValueError("frame has no base_T_board")
    ids = np.asarray(frame.wrist_detection.ids, dtype=np.int32).reshape(-1)
    object_points = board_points[ids]
    wrist_T_board = invert_transform(frame.base_T_ee @ as_transform(ee_T_wrist)) @ as_transform(frame.base_T_board)
    rvec, tvec = transform_to_rvec_tvec(wrist_T_board)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, wrist_intrinsic, wrist_dist)
    observed = frame.wrist_detection.corners.reshape(-1, 2)
    return projected.reshape(-1, 2), observed


def _reprojection_metrics(
    frames: list[DualFrame],
    ee_T_wrist: np.ndarray,
    board_points: np.ndarray,
    wrist_intrinsic: np.ndarray,
    wrist_dist: np.ndarray,
) -> dict[str, Any]:
    total_sq = 0.0
    total_points = 0
    per_frame = []
    for frame in frames:
        projected, observed = _project_wrist_frame(frame, ee_T_wrist, board_points, wrist_intrinsic, wrist_dist)
        distances2 = np.sum((projected - observed) ** 2, axis=1)
        rmse = float(np.sqrt(np.mean(distances2))) if distances2.size else 0.0
        total_sq += float(np.sum(distances2))
        total_points += int(distances2.size)
        per_frame.append(
            {
                "index": int(frame.index),
                "corner_count": int(distances2.size),
                "weight": float(frame.frame_weight),
                "rmse_px": rmse,
                "mean_px": float(np.mean(np.sqrt(distances2))) if distances2.size else 0.0,
                "max_px": float(np.max(np.sqrt(distances2))) if distances2.size else 0.0,
            }
        )
    rmses = [item["rmse_px"] for item in per_frame]
    return {
        "frame_count": len(frames),
        "point_count": total_points,
        "rmse_px": float(np.sqrt(total_sq / max(1, total_points))),
        "per_frame_rmse_px_mean": float(np.mean(rmses)) if rmses else None,
        "per_frame_rmse_px_median": float(np.median(rmses)) if rmses else None,
        "per_frame_rmse_px_max": float(np.max(rmses)) if rmses else None,
        "per_frame": per_frame,
    }


def _pose_residual_metrics(frames: list[DualFrame], ee_T_wrist: np.ndarray) -> dict[str, Any]:
    per_frame = []
    for frame in frames:
        if frame.base_T_board is None or frame.wrist_detection.T_cam_board is None:
            continue
        board_from_wrist = frame.base_T_ee @ as_transform(ee_T_wrist) @ as_transform(frame.wrist_detection.T_cam_board)
        err = pose_error(board_from_wrist, frame.base_T_board)
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
        "frame_count": len(per_frame),
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


def _optimize(
    args: argparse.Namespace,
    used_frames: list[DualFrame],
    initial_ee_T_wrist: np.ndarray,
    board_cfg: BoardConfig,
    wrist_intrinsic: np.ndarray,
    wrist_dist: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    board_points = board_object_points(make_charuco_board(board_cfg))
    K = np.asarray(wrist_intrinsic, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(wrist_dist, dtype=np.float64).reshape(-1)
    initial_reprojection = _reprojection_metrics(used_frames, initial_ee_T_wrist, board_points, K, dist)
    initial_pose_residuals = _pose_residual_metrics(used_frames, initial_ee_T_wrist)
    try:
        from scipy.optimize import least_squares
    except Exception as exc:
        return as_transform(initial_ee_T_wrist), {
            "optimized": False,
            "success": False,
            "message": f"scipy unavailable: {exc}",
            "initial_reprojection": initial_reprojection,
            "final_reprojection": initial_reprojection,
            "initial_pose_residuals": initial_pose_residuals,
            "final_pose_residuals": initial_pose_residuals,
            "delta_from_initial": pose_error(initial_ee_T_wrist, initial_ee_T_wrist),
        }

    def compose(delta: np.ndarray) -> np.ndarray:
        return _delta_to_transform(delta) @ as_transform(initial_ee_T_wrist)

    def residual_fn(delta: np.ndarray) -> np.ndarray:
        ee_T_wrist = compose(delta)
        chunks = []
        for frame in used_frames:
            projected, observed = _project_wrist_frame(frame, ee_T_wrist, board_points, K, dist)
            chunks.append(((projected - observed) * float(frame.frame_weight)).reshape(-1))
        return np.concatenate(chunks)

    result = least_squares(
        residual_fn,
        np.zeros((6,), dtype=np.float64),
        loss=str(args.loss),
        f_scale=float(args.f_scale),
        max_nfev=int(args.max_nfev),
    )
    final = compose(result.x)
    final_reprojection = _reprojection_metrics(used_frames, final, board_points, K, dist)
    final_pose_residuals = _pose_residual_metrics(used_frames, final)
    return final, {
        "optimized": True,
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "loss": str(args.loss),
        "f_scale": float(args.f_scale),
        "delta_vector": np.asarray(result.x, dtype=np.float64),
        "delta_from_initial": pose_error(final, initial_ee_T_wrist),
        "initial_reprojection": initial_reprojection,
        "final_reprojection": final_reprojection,
        "initial_pose_residuals": initial_pose_residuals,
        "final_pose_residuals": final_pose_residuals,
    }


def _evaluate_quality(args: argparse.Namespace, used_frames: list[DualFrame], metrics: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    failures: list[str] = []
    final_reprojection = metrics["final_reprojection"]
    initial_reprojection = metrics["initial_reprojection"]
    final_pose = metrics["final_pose_residuals"]
    gates = {
        "min_valid_frames": int(args.min_valid_frames),
        "max_final_reprojection_rmse_px": float(args.max_final_reprojection_rmse_px),
        "max_final_frame_reprojection_rmse_px": float(args.max_final_frame_reprojection_rmse_px),
        "max_final_pose_residual_mm": float(args.max_final_pose_residual_mm),
        "max_final_pose_residual_deg": float(args.max_final_pose_residual_deg),
        "allow_worse_reprojection": bool(args.allow_worse_reprojection),
        "force_accept": bool(args.force_accept),
    }
    if len(used_frames) < max(3, int(args.min_valid_frames)):
        failures.append(f"frames_used {len(used_frames)} < {max(3, int(args.min_valid_frames))}")
    if not bool(metrics.get("success", False)):
        failures.append(f"optimizer did not converge: {metrics.get('message', '')}")
    final_rmse = float(final_reprojection["rmse_px"])
    initial_rmse = float(initial_reprojection["rmse_px"])
    if not bool(args.allow_worse_reprojection) and final_rmse > initial_rmse + 1e-6:
        failures.append(f"final reprojection worsened {final_rmse:.4f}px > {initial_rmse:.4f}px")
    if final_rmse > float(args.max_final_reprojection_rmse_px):
        failures.append(f"final reprojection rmse {final_rmse:.4f}px > {float(args.max_final_reprojection_rmse_px):.4f}px")
    frame_max = final_reprojection.get("per_frame_rmse_px_max")
    if frame_max is not None and float(frame_max) > float(args.max_final_frame_reprojection_rmse_px):
        failures.append(
            f"final per-frame reprojection max {float(frame_max):.4f}px > {float(args.max_final_frame_reprojection_rmse_px):.4f}px"
        )
    trans_max = final_pose.get("translation_mm_max")
    if trans_max is not None and float(trans_max) > float(args.max_final_pose_residual_mm):
        failures.append(f"final pose residual max {float(trans_max):.3f}mm > {float(args.max_final_pose_residual_mm):.3f}mm")
    rot_max = final_pose.get("rotation_deg_max")
    if rot_max is not None and float(rot_max) > float(args.max_final_pose_residual_deg):
        failures.append(f"final pose residual max {float(rot_max):.3f}deg > {float(args.max_final_pose_residual_deg):.3f}deg")
    return bool(args.force_accept) or not failures, failures, gates


def _draw_wrist_projection(
    image_bgr: np.ndarray,
    frame: DualFrame,
    ee_T_wrist: np.ndarray,
    board_cfg: BoardConfig,
    intrinsic: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    out = image_bgr.copy()
    board_points = board_object_points(make_charuco_board(board_cfg))
    projected, observed = _project_wrist_frame(frame, ee_T_wrist, board_points, intrinsic, dist)
    wrist_T_board = invert_transform(frame.base_T_ee @ as_transform(ee_T_wrist)) @ as_transform(frame.base_T_board)
    rvec, tvec = transform_to_rvec_tvec(wrist_T_board)
    try:
        cv2.drawFrameAxes(out, intrinsic, dist, rvec, tvec, 0.06)
    except cv2.error:
        pass
    for obs, pred in zip(observed, projected):
        obs_i = tuple(np.round(obs).astype(int))
        pred_i = tuple(np.round(pred).astype(int))
        cv2.circle(out, obs_i, 4, (30, 220, 30), -1, cv2.LINE_AA)
        cv2.drawMarker(out, pred_i, (255, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2)
        cv2.line(out, obs_i, pred_i, (220, 220, 220), 1, cv2.LINE_AA)
    return out


def _write_overlays(
    run_dir: Path,
    frames: list[DualFrame],
    ee_T_wrist: np.ndarray,
    board_cfg: BoardConfig,
    main_intrinsic: np.ndarray,
    main_dist: np.ndarray,
    wrist_intrinsic: np.ndarray,
    wrist_dist: np.ndarray,
    *,
    max_overlays: int,
) -> list[Path]:
    paths = []
    for frame in frames[: max(0, int(max_overlays))]:
        if frame.main_image_path is None or frame.wrist_image_path is None:
            continue
        main_image = cv2.imread(str(frame.main_image_path), cv2.IMREAD_COLOR)
        wrist_image = cv2.imread(str(frame.wrist_image_path), cv2.IMREAD_COLOR)
        if main_image is None or wrist_image is None:
            continue
        main_overlay = draw_board_overlay_image(main_image, frame.main_detection, main_intrinsic, main_dist)
        wrist_overlay = _draw_wrist_projection(wrist_image, frame, ee_T_wrist, board_cfg, wrist_intrinsic, wrist_dist)
        combined = _stack_preview(main_overlay, wrist_overlay, sample_index=frame.index, target_count=len(frames))
        path = run_dir / "overlays" / f"{frame.index:04d}.png"
        image_write(path, combined)
        paths.append(path)
    return paths


def main() -> int:
    args = parse_args()
    settings = load_realman_settings(args.config)
    board_cfg = board_config_from_args(args, settings)
    run_dir = calibration_run_dir("wrist_camera_dual_board", args.output_root)
    base_T_main_camera, base_camera_source = _load_base_camera_transform(args)
    if args.input_run is not None:
        frames, main_K, main_dist, wrist_K, wrist_dist, metadata = _load_input_run(
            args,
            settings,
            run_dir,
            base_T_main_camera,
            board_cfg,
        )
    else:
        frames, main_K, main_dist, wrist_K, wrist_dist, metadata = _collect_observations(
            args,
            settings,
            run_dir,
            base_T_main_camera,
            board_cfg,
        )
    initial, initialization = _prepare_frames(frames, board_cfg, min_valid_frames=int(args.min_valid_frames))
    used_frames = [frame for frame in frames if frame.accepted_for_optimization]
    final, optimization = _optimize(args, used_frames, initial, board_cfg, wrist_K, wrist_dist)
    accepted, failures, quality_gates = _evaluate_quality(args, used_frames, optimization)
    status = "accepted" if accepted else "rejected"

    save_matrix_pair(run_dir / "ee_T_wrist_camera_initial.npy", initial)
    save_matrix_pair(run_dir / "ee_T_wrist_camera.npy", final)
    save_matrix_pair(run_dir / "wrist_camera_T_ee.npy", invert_transform(final))
    base_T_boards = [frame.base_T_board for frame in frames if frame.base_T_board is not None]
    if base_T_boards:
        np.save(run_dir / "base_T_board_per_frame.npy", np.stack(base_T_boards, axis=0))
    _save_observations(
        run_dir,
        frames,
        main_K,
        main_dist,
        wrist_K,
        wrist_dist,
        base_camera_source=base_camera_source,
        base_T_main_camera=base_T_main_camera,
        board_cfg=board_cfg,
        metadata=metadata,
    )
    overlays = _write_overlays(
        run_dir,
        used_frames,
        final,
        board_cfg,
        main_K,
        main_dist,
        wrist_K,
        wrist_dist,
        max_overlays=int(args.max_overlays),
    )
    rejected_frames = [
        {"index": int(frame.index), "reason": frame.rejection_reason}
        for frame in frames
        if not frame.accepted_for_optimization
    ]
    final_pose = optimization["final_pose_residuals"]
    final_reprojection = optimization["final_reprojection"]
    report = {
        "run_dir": run_dir,
        "status": status,
        "accepted": accepted,
        "failure_reasons": failures,
        "quality_gates": quality_gates,
        "frames_total": len(frames),
        "frames_used": len(used_frames),
        "base_camera_source": base_camera_source,
        "board_config": asdict(board_cfg),
        "metadata": metadata,
        "rejected_frames": rejected_frames,
        "initialization": initialization,
        "optimization": optimization,
        "pose_diversity": pose_diversity_metrics([frame.base_T_ee for frame in used_frames]),
        "initial_ee_T_wrist_camera": initial,
        "ee_T_wrist_camera": final,
        "wrist_camera_T_ee": invert_transform(final),
        "T_R_Cg": base_T_main_camera,
        "overlay_count": len(overlays),
        "summary": {
            "final_reprojection_rmse_px": final_reprojection["rmse_px"],
            "final_reprojection_per_frame_max_px": final_reprojection["per_frame_rmse_px_max"],
            "cross_camera_translation_mm_median": final_pose["translation_mm_median"],
            "cross_camera_translation_mm_mean": final_pose["translation_mm_mean"],
            "cross_camera_translation_mm_max": final_pose["translation_mm_max"],
            "cross_camera_rotation_deg_median": final_pose["rotation_deg_median"],
            "cross_camera_rotation_deg_mean": final_pose["rotation_deg_mean"],
            "cross_camera_rotation_deg_max": final_pose["rotation_deg_max"],
        },
    }
    report = normalize_calibration_report(
        report,
        run_kind="wrist_camera_dual_board",
        config_path=args.config,
        frames_total=len(frames),
        frames_used=len(used_frames),
        accepted=accepted,
        rejection_reason="; ".join(failures),
        transforms={"T_R_Cg": base_T_main_camera, "T_E_Cw": final},
        metrics={
            "summary": report["summary"],
            "initialization": initialization,
            "optimization": optimization,
            "quality_gates": quality_gates,
            "failure_reasons": failures,
        },
    )
    save_json(run_dir / "calibration_report.json", report)
    save_json(run_dir / "dual_board_report.json", report)
    if not args.no_html_report:
        write_html_report(
            run_dir / "report.html",
            title="RM75 Dual-Camera Moving-Board Wrist Calibration",
            summary=report["summary"] | {"status": status, "frames_used": len(used_frames), "run_dir": run_dir},
            sections=[
                ("Quality Gates", {"failure_reasons": failures, **quality_gates}),
                ("Cross-Camera Pose Residuals", final_pose),
                ("Wrist Reprojection", final_reprojection),
                ("Initialization", initialization),
                ("Optimization", optimization),
                ("Rejected Frames", rejected_frames),
            ],
            image_dirs=["overlays"],
        )
    print(f"[dual-board] status={status} frames used: {len(used_frames)}/{len(frames)}")
    print(
        "[dual-board] wrist reprojection rmse px: "
        f"{optimization['initial_reprojection']['rmse_px']:.4f} -> {final_reprojection['rmse_px']:.4f}"
    )
    print(
        "[dual-board] cross-camera board residual: "
        f"median={final_pose['translation_mm_median']:.3f}mm/{final_pose['rotation_deg_median']:.3f}deg "
        f"mean={final_pose['translation_mm_mean']:.3f}mm/{final_pose['rotation_deg_mean']:.3f}deg "
        f"max={final_pose['translation_mm_max']:.3f}mm/{final_pose['rotation_deg_max']:.3f}deg"
    )
    if failures:
        print("[dual-board] failure reasons: " + "; ".join(failures))
    print(f"[dual-board] calibration written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
