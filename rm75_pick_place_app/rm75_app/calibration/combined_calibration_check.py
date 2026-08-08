from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from rm75_app.paths import DEFAULT_CAMERA_EXTRINSIC, DEFAULT_RM75_URDF

from .board import BoardConfig, board_config_from_args, detect_board_pose, detection_is_accepted, draw_board_overlay_image
from .common import (
    RealSenseCamera,
    RealmanSdkController,
    as_transform,
    calibration_run_dir,
    camera_opencv_to_base_camera,
    image_write,
    invert_transform,
    link_poses_from_qpos,
    load_matrix,
    load_realman_settings,
    load_urdf,
    pose_error,
    qpos_samples_from_settings,
    save_json,
    select_link_pose,
)
from .reporting import normalize_calibration_report, write_html_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly validate base-camera and wrist-camera calibration against the same table ChArUco board."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--execute-real", action="store_true")
    parser.add_argument("--robot-ip", type=str, default=None)
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--joint-names", nargs="+", default=None)
    parser.add_argument("--reference-qpos-json", type=Path, default=None)
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_RM75_URDF)
    parser.add_argument("--ee-link-name", type=str, default=None)
    parser.add_argument("--base-camera-extrinsic-opencv-path", type=Path, default=DEFAULT_CAMERA_EXTRINSIC)
    parser.add_argument("--base-camera-use-direct", action="store_true")
    parser.add_argument("--base-camera-run-dir", type=Path, default=None)
    parser.add_argument("--wrist-camera-run-dir", type=Path, default=None)
    parser.add_argument("--ee-wrist-camera-path", type=Path, default=None)
    parser.add_argument("--base-board-path", type=Path, default=None)
    parser.add_argument("--main-camera-serial", type=str, default=None)
    parser.add_argument("--wrist-camera-serial", type=str, default=None)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--board-squares-x", type=int, default=None)
    parser.add_argument("--board-squares-y", type=int, default=None)
    parser.add_argument("--board-square-length-m", type=float, default=None)
    parser.add_argument("--board-marker-length-m", type=float, default=None)
    parser.add_argument("--board-dictionary", type=str, default=None)
    parser.add_argument("--board-legacy-pattern", action="store_true", default=None)
    parser.add_argument("--min-board-corners", type=int, default=None)
    parser.add_argument("--max-board-rmse-px", type=float, default=None)
    parser.add_argument("--min-corner-coverage", type=float, default=None)
    parser.add_argument("--min-paired-frames", type=int, default=2)
    parser.add_argument("--max-main-wrist-translation-m", type=float, default=0.03)
    parser.add_argument("--max-main-wrist-rotation-deg", type=float, default=5.0)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-delay-ms", type=int, default=1)
    parser.add_argument("--no-html-report", action="store_true")
    return parser.parse_args()


def _board_config(args: argparse.Namespace, settings: dict) -> BoardConfig:
    return board_config_from_args(args, settings)


def _load_required_matrices(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    base_extrinsic_path = args.base_camera_extrinsic_opencv_path
    if args.base_camera_run_dir is not None:
        candidate = Path(args.base_camera_run_dir).expanduser() / "camera_extrinsic_opencv.npy"
        if candidate.exists():
            base_extrinsic_path = candidate
    base_T_main_cam = camera_opencv_to_base_camera(
        load_matrix(base_extrinsic_path),
        use_direct=bool(args.base_camera_use_direct),
    )
    ee_path = args.ee_wrist_camera_path
    if ee_path is None and args.wrist_camera_run_dir is not None:
        ee_path = Path(args.wrist_camera_run_dir).expanduser() / "ee_T_wrist_camera.npy"
    if ee_path is None:
        raise ValueError("Provide --ee-wrist-camera-path or --wrist-camera-run-dir")
    ee_T_wrist_cam = load_matrix(ee_path)
    base_T_board = None
    board_path = args.base_board_path
    if board_path is None and args.base_camera_run_dir is not None:
        candidate = Path(args.base_camera_run_dir).expanduser() / "base_T_board.npy"
        board_path = candidate if candidate.exists() else None
    if board_path is None and args.wrist_camera_run_dir is not None:
        candidate = Path(args.wrist_camera_run_dir).expanduser() / "base_T_board.npy"
        board_path = candidate if candidate.exists() else None
    if board_path is not None:
        base_T_board = load_matrix(board_path)
    return base_T_main_cam, ee_T_wrist_cam, base_T_board


def _reference_qpos(args: argparse.Namespace, settings: dict) -> list[np.ndarray]:
    if args.reference_qpos_json is not None:
        from .common import load_json

        raw = load_json(args.reference_qpos_json)
        return [np.deg2rad(np.asarray(sample, dtype=np.float64).reshape(-1)) for sample in raw]
    return qpos_samples_from_settings(settings, key="check_reference_qpos_deg", fallback_key="wrist_reference_qpos_deg")


def main() -> int:
    args = parse_args()
    settings = load_realman_settings(args.config)
    run_dir = calibration_run_dir("combined_calibration_check", args.output_root)
    base_T_main_cam, ee_T_wrist_cam, base_T_board_prior = _load_required_matrices(args)
    joint_names = list(args.joint_names or settings.get("joint_names") or [f"joint_{idx}" for idx in range(1, 8)])
    qpos_samples = _reference_qpos(args, settings)
    if not qpos_samples:
        raise ValueError("No check_reference_qpos_deg/wrist_reference_qpos_deg samples found.")
    if not args.execute_real:
        raise RuntimeError("Combined live validation moves/captures cameras; pass --execute-real.")
    robot_ip = args.robot_ip or settings.get("robot_ip")
    if not robot_ip:
        raise ValueError("robot_ip is required in --config or --robot-ip")
    main_cfg = settings.get("base_camera", settings.get("camera", {}))
    wrist_cfg = settings.get("wrist_camera", {})
    main_cam = RealSenseCamera(
        serial=args.main_camera_serial or main_cfg.get("serial_number_or_name"),
        width=int(args.camera_width or main_cfg.get("width", 1280)),
        height=int(args.camera_height or main_cfg.get("height", 720)),
        fps=int(args.camera_fps or main_cfg.get("fps", 30)),
        warmup_frames=int(args.warmup_frames),
    )
    wrist_cam = RealSenseCamera(
        serial=args.wrist_camera_serial or wrist_cfg.get("serial_number_or_name"),
        width=int(args.camera_width or wrist_cfg.get("width", 1280)),
        height=int(args.camera_height or wrist_cfg.get("height", 720)),
        fps=int(args.camera_fps or wrist_cfg.get("fps", 30)),
        warmup_frames=int(args.warmup_frames),
    )
    controller = RealmanSdkController(
        robot_ip=str(robot_ip),
        robot_port=int(args.robot_port or settings.get("robot_port", 8080)),
        joint_names=joint_names,
        calibration_offset=settings.get("calibration_offset", {}),
    )
    urdf = load_urdf(args.urdf_path)
    board_cfg = _board_config(args, settings)
    records = []
    base_board_from_main = []
    base_board_from_wrist = []
    maincam_wristcam_errors = []
    try:
        main_cam.start()
        wrist_cam.start()
        for idx, qpos in enumerate(qpos_samples):
            print(f"[combined-check] moving to sample {idx + 1}/{len(qpos_samples)}")
            controller.move_to_qpos(qpos)
            qpos_now = np.asarray(controller.get_qpos(flat=True), dtype=np.float64)
            main_frame = main_cam.capture()
            wrist_frame = wrist_cam.capture()
            main_det = detect_board_pose(main_frame.color_bgr, main_frame.intrinsic, main_frame.dist_coeffs, board_cfg)
            wrist_det = detect_board_pose(wrist_frame.color_bgr, wrist_frame.intrinsic, wrist_frame.dist_coeffs, board_cfg)
            link_poses = link_poses_from_qpos(urdf, qpos_now, joint_names)
            ee_link, base_T_ee = select_link_pose(link_poses, args.ee_link_name)
            base_T_wrist_cam = base_T_ee @ ee_T_wrist_cam
            rec = {
                "index": idx,
                "ee_link": ee_link,
                "qpos_rad": qpos_now,
                "main_board_ok": main_det.ok,
                "wrist_board_ok": wrist_det.ok,
                "main_board_rmse_px": main_det.reprojection_rmse_px,
                "wrist_board_rmse_px": wrist_det.reprojection_rmse_px,
                "main_board_quality": main_det.quality,
                "wrist_board_quality": wrist_det.quality,
                "main_board_reason": main_det.reason,
                "wrist_board_reason": wrist_det.reason,
            }
            image_write(run_dir / "main_images" / f"{idx:04d}.png", main_frame.color_bgr)
            image_write(run_dir / "wrist_images" / f"{idx:04d}.png", wrist_frame.color_bgr)
            main_overlay = draw_board_overlay_image(main_frame.color_bgr, main_det, main_frame.intrinsic, main_frame.dist_coeffs)
            wrist_overlay = draw_board_overlay_image(wrist_frame.color_bgr, wrist_det, wrist_frame.intrinsic, wrist_frame.dist_coeffs)
            image_write(run_dir / "main_overlays" / f"{idx:04d}.png", main_overlay)
            image_write(run_dir / "wrist_overlays" / f"{idx:04d}.png", wrist_overlay)
            if args.preview:
                preview = np.hstack([main_overlay, wrist_overlay])
                cv2.imshow("rm75 combined calibration check", preview)
                key = cv2.waitKey(max(1, int(args.preview_delay_ms))) & 0xFF
                if key in (27, ord("q")):
                    print("[combined-check] preview requested stop")
                    break
            if detection_is_accepted(main_det, board_cfg) and main_det.T_cam_board is not None:
                board_main = base_T_main_cam @ as_transform(main_det.T_cam_board)
                base_board_from_main.append(board_main)
                if base_T_board_prior is not None:
                    rec["main_vs_prior"] = pose_error(board_main, base_T_board_prior)
            if detection_is_accepted(wrist_det, board_cfg) and wrist_det.T_cam_board is not None:
                board_wrist = base_T_wrist_cam @ as_transform(wrist_det.T_cam_board)
                base_board_from_wrist.append(board_wrist)
                if base_T_board_prior is not None:
                    rec["wrist_vs_prior"] = pose_error(board_wrist, base_T_board_prior)
            if (
                detection_is_accepted(main_det, board_cfg)
                and detection_is_accepted(wrist_det, board_cfg)
                and main_det.T_cam_board is not None
                and wrist_det.T_cam_board is not None
            ):
                main_T_wrist_observed = as_transform(main_det.T_cam_board) @ invert_transform(as_transform(wrist_det.T_cam_board))
                main_T_wrist_pred = invert_transform(base_T_main_cam) @ base_T_wrist_cam
                err = pose_error(main_T_wrist_observed, main_T_wrist_pred)
                rec["maincam_T_wristcam_error"] = err
                maincam_wristcam_errors.append(err)
            records.append(rec)
    finally:
        if args.preview:
            try:
                cv2.destroyWindow("rm75 combined calibration check")
            except cv2.error:
                pass
        main_cam.stop()
        wrist_cam.stop()
        controller.disconnect()
    summary = {
        "run_dir": run_dir,
        "samples": len(records),
        "main_board_valid": len(base_board_from_main),
        "wrist_board_valid": len(base_board_from_wrist),
        "records": records,
    }
    if maincam_wristcam_errors:
        summary["maincam_wristcam_error_mean"] = {
            "translation_m": float(np.mean([item["translation_m"] for item in maincam_wristcam_errors])),
            "rotation_deg": float(np.mean([item["rotation_deg"] for item in maincam_wristcam_errors])),
        }
        summary["maincam_wristcam_error_max"] = {
            "translation_m": float(np.max([item["translation_m"] for item in maincam_wristcam_errors])),
            "rotation_deg": float(np.max([item["rotation_deg"] for item in maincam_wristcam_errors])),
        }
    for key in ("main_vs_prior", "wrist_vs_prior"):
        errors = [rec[key] for rec in records if key in rec]
        if errors:
            summary[f"{key}_mean"] = {
                "translation_m": float(np.mean([item["translation_m"] for item in errors])),
                "rotation_deg": float(np.mean([item["rotation_deg"] for item in errors])),
            }
            summary[f"{key}_max"] = {
                "translation_m": float(np.max([item["translation_m"] for item in errors])),
                "rotation_deg": float(np.max([item["rotation_deg"] for item in errors])),
            }
    failure_reasons = []
    if len(maincam_wristcam_errors) < int(args.min_paired_frames):
        failure_reasons.append(
            f"paired_frames {len(maincam_wristcam_errors)} < {int(args.min_paired_frames)}"
        )
    if maincam_wristcam_errors:
        max_err = summary["maincam_wristcam_error_max"]
        if max_err["translation_m"] > float(args.max_main_wrist_translation_m):
            failure_reasons.append(
                "max main/wrist translation "
                f"{max_err['translation_m']:.4f}m > {float(args.max_main_wrist_translation_m):.4f}m"
            )
        if max_err["rotation_deg"] > float(args.max_main_wrist_rotation_deg):
            failure_reasons.append(
                "max main/wrist rotation "
                f"{max_err['rotation_deg']:.3f}deg > {float(args.max_main_wrist_rotation_deg):.3f}deg"
            )
    outliers = []
    for rec in records:
        reasons = []
        if not bool(rec.get("main_board_ok", False)):
            reasons.append(str(rec.get("main_board_reason") or "main board rejected"))
        if not bool(rec.get("wrist_board_ok", False)):
            reasons.append(str(rec.get("wrist_board_reason") or "wrist board rejected"))
        err = rec.get("maincam_T_wristcam_error")
        if isinstance(err, dict):
            if float(err.get("translation_m", 0.0)) > float(args.max_main_wrist_translation_m):
                reasons.append("main/wrist translation above threshold")
            if float(err.get("rotation_deg", 0.0)) > float(args.max_main_wrist_rotation_deg):
                reasons.append("main/wrist rotation above threshold")
        else:
            reasons.append("no paired main/wrist board error")
        if reasons:
            outliers.append({"index": rec.get("index"), "reasons": reasons})
    summary["outliers"] = outliers
    summary["status"] = "fail" if failure_reasons else "pass"
    summary["failure_reasons"] = failure_reasons
    summary = normalize_calibration_report(
        summary,
        run_kind="combined_calibration_check",
        config_path=args.config,
        frames_total=len(records),
        frames_used=len(maincam_wristcam_errors),
        accepted=not failure_reasons,
        rejection_reason="; ".join(failure_reasons),
        transforms={
            "T_R_Cg": base_T_main_cam,
            "T_R_P": base_T_board_prior,
            "T_E_Cw": ee_T_wrist_cam,
        },
        metrics={
            key: value
            for key, value in summary.items()
            if key.endswith("_mean") or key.endswith("_max") or key in {"status", "failure_reasons"}
        },
    )
    save_json(run_dir / "combined_check_report.json", summary)
    if not args.no_html_report:
        write_html_report(
            run_dir / "report.html",
            title="RM75 Combined Calibration Check",
            summary=summary,
            sections=[("Records", records)],
            image_dirs=["main_overlays", "wrist_overlays"],
        )
    print(f"[combined-check] report written to {run_dir / 'combined_check_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
