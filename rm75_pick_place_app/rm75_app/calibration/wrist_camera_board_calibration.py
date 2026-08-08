from __future__ import annotations

import argparse
import select
import sys
from pathlib import Path

import cv2
import numpy as np

from rm75_app.paths import DEFAULT_RM75_URDF

from .board import (
    BoardConfig,
    BoardDetection,
    board_config_from_args,
    detect_board_pose,
    detection_is_accepted,
    draw_board_overlay_image,
    optimize_hand_eye_reprojection,
    pose_diversity_metrics,
    solve_eye_in_hand_multi_method,
)
from .common import (
    RealSenseCamera,
    RealmanSdkController,
    as_transform,
    calibration_run_dir,
    image_write,
    link_poses_from_qpos,
    load_json,
    load_realman_settings,
    load_urdf,
    qpos_samples_from_settings,
    save_json,
    save_matrix_pair,
    select_link_pose,
)
from .reporting import normalize_calibration_report, write_html_report
from .sampling import generate_joint_jitter_samples, save_qpos_plan, vector_deg_to_rad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic eye-in-hand calibration for an RM75 wrist camera using a fixed ChArUco board."
    )
    parser.add_argument("--config", type=Path, default=None, help="JSON with realman_settings.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--use-previous-run", type=Path, default=None)
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
    parser.add_argument("--capture-current-only", action="store_true", help="Capture one frame at the current pose without moving.")
    parser.add_argument("--manual-capture", action="store_true", help="Press Enter to capture each sample at the current arm pose; no automatic motion.")
    parser.add_argument("--manual-count", type=int, default=15, help="Target number of manual samples. Use q then Enter to finish early.")
    parser.add_argument("--manual-terminal", action="store_true", help="Use terminal-only manual capture without the live OpenCV preview.")
    parser.add_argument("--auto-generate-samples", action="store_true", help="Generate a jittered wrist pose plan from a seed pose.")
    parser.add_argument("--sample-count", type=int, default=18)
    parser.add_argument("--sample-seed", type=int, default=7)
    parser.add_argument("--seed-qpos-deg", nargs="+", type=float, default=None)
    parser.add_argument("--joint-jitter-deg", nargs="+", type=float, default=None)
    parser.add_argument("--joint-limits-json", type=Path, default=None, help="Optional JSON [[lo, hi], ...] joint limits in degrees.")
    parser.add_argument("--dry-run-plan", action="store_true", help="Write the planned qpos samples and exit without hardware.")
    parser.add_argument("--preview", action="store_true", help="Show live board overlays during capture.")
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
    parser.add_argument("--disable-reprojection-opt", action="store_true")
    parser.add_argument("--fixed-base-board", type=Path, default=None, help="Optional base_T_board.npy from the base-camera calibration.")
    parser.add_argument("--no-html-report", action="store_true")
    return parser.parse_args()


def _board_config(args: argparse.Namespace, settings: dict) -> BoardConfig:
    return board_config_from_args(args, settings)


def _reference_qpos(args: argparse.Namespace, settings: dict) -> list[np.ndarray]:
    if args.reference_qpos_json is not None:
        raw = load_json(args.reference_qpos_json)
        return [np.deg2rad(np.asarray(sample, dtype=np.float64).reshape(-1)) for sample in raw]
    return qpos_samples_from_settings(settings, key="wrist_reference_qpos_deg")


def _joint_names(args: argparse.Namespace, settings: dict) -> list[str]:
    return list(args.joint_names or settings.get("joint_names") or [f"joint_{idx}" for idx in range(1, 8)])


def _seed_qpos(args: argparse.Namespace, settings: dict, controller: RealmanSdkController | None = None) -> np.ndarray:
    if args.seed_qpos_deg is not None:
        return vector_deg_to_rad(args.seed_qpos_deg)
    if settings.get("wrist_seed_qpos_deg") is not None:
        return vector_deg_to_rad(settings["wrist_seed_qpos_deg"])
    refs = _reference_qpos(args, settings)
    if refs:
        return refs[0]
    if controller is not None:
        return np.asarray(controller.get_qpos(flat=True), dtype=np.float64).reshape(-1)
    raise ValueError("Need --seed-qpos-deg, wrist_seed_qpos_deg, or wrist_reference_qpos_deg for dry-run planning.")


def _joint_limits_deg(args: argparse.Namespace, settings: dict):
    if args.joint_limits_json is not None:
        return load_json(args.joint_limits_json)
    return settings.get("wrist_joint_limits_deg", settings.get("joint_limits_deg"))


def _planned_qpos(
    args: argparse.Namespace,
    settings: dict,
    *,
    controller: RealmanSdkController | None = None,
) -> list[np.ndarray]:
    if args.capture_current_only:
        if controller is None:
            raise ValueError("--capture-current-only requires a robot connection.")
        return [np.asarray(controller.get_qpos(flat=True), dtype=np.float64).reshape(-1)]
    if not args.auto_generate_samples:
        return _reference_qpos(args, settings)
    seed = _seed_qpos(args, settings, controller)
    jitter = args.joint_jitter_deg
    if jitter is None:
        jitter = settings.get("wrist_joint_jitter_deg", settings.get("joint_jitter_deg", 8.0))
    return generate_joint_jitter_samples(
        seed,
        count=int(args.sample_count),
        jitter_deg=jitter,
        joint_limits_deg=_joint_limits_deg(args, settings),
        random_seed=int(args.sample_seed),
        include_seed=True,
    )


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


def _draw_manual_capture_status(
    image_bgr: np.ndarray,
    detection: BoardDetection,
    *,
    sample_index: int,
    target_count: int,
) -> np.ndarray:
    out = image_bgr.copy()
    height, width = out.shape[:2]
    status = "OK" if detection.ok else "NO BOARD"
    color = (30, 220, 30) if detection.ok else (0, 165, 255)
    rmse = "--" if detection.reprojection_rmse_px is None else f"{float(detection.reprojection_rmse_px):.2f}px"
    source = str(detection.quality.get("source", "charuco"))
    lines = [
        f"sample {sample_index:02d}/{target_count:02d}   Enter/Space: capture   q/Esc: finish",
        f"{status}   corners={len(detection.ids)}   rmse={rmse}   source={source}",
    ]
    if detection.reason and not detection.ok:
        lines.append(str(detection.reason))
    elif "orientation_marker_max_error_px" in detection.quality:
        orient = float(detection.quality["orientation_marker_max_error_px"])
        mode = detection.quality.get("orientation_mode", "")
        lines.append(f"orientation={mode}   marker max error={orient:.2f}px")

    line_height = 28
    box_height = min(height - 20, 18 + line_height * len(lines))
    y0 = max(8, height - box_height - 12)
    overlay = out.copy()
    cv2.rectangle(overlay, (8, y0), (min(width - 8, 900), height - 8), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    for idx, line in enumerate(lines):
        cv2.putText(
            out,
            line[:110],
            (18, y0 + 28 + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            color if idx == 1 else (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
    return out


def _collect_observations(args: argparse.Namespace, settings: dict, run_dir: Path):
    import time

    if not args.execute_real and not args.capture_current_only and not args.manual_capture:
        raise RuntimeError(
            "Refusing to move/capture automatically without --execute-real. "
            "Use --manual-capture, --capture-current-only, or --use-previous-run."
        )
    joint_names = _joint_names(args, settings)
    robot_ip = args.robot_ip or settings.get("robot_ip")
    if not robot_ip:
        raise ValueError("robot_ip is required in --config or --robot-ip")
    camera_cfg = settings.get("wrist_camera", settings.get("camera", {}))
    serial = args.camera_serial or camera_cfg.get("serial_number_or_name")
    width = int(args.camera_width if args.camera_width is not None else camera_cfg.get("width", 1280))
    height = int(args.camera_height if args.camera_height is not None else camera_cfg.get("height", 720))
    fps = int(args.camera_fps if args.camera_fps is not None else camera_cfg.get("fps", 30))
    exposure_us = args.camera_exposure_us if args.camera_exposure_us is not None else camera_cfg.get("exposure_us")
    gain = args.camera_gain if args.camera_gain is not None else camera_cfg.get("gain")
    auto_exposure = (
        args.camera_auto_exposure
        if args.camera_auto_exposure is not None
        else camera_cfg.get("auto_exposure")
    )
    controller = RealmanSdkController(
        robot_ip=str(robot_ip),
        robot_port=int(args.robot_port or settings.get("robot_port", 8080)),
        joint_names=joint_names,
        calibration_offset=settings.get("calibration_offset", {}),
    )
    if args.manual_capture:
        qpos_samples = []
        save_json(
            run_dir / "manual_capture_instructions.json",
            {
                "mode": "manual_capture",
                "target_samples": int(args.manual_count),
                "controls": {"enter": "capture", "space": "capture", "q": "finish", "escape": "finish"},
            },
        )
    else:
        qpos_samples = _planned_qpos(args, settings, controller=controller)
        if not qpos_samples:
            raise ValueError("No wrist_reference_qpos_deg/reference_qpos_deg samples found.")
        save_qpos_plan(run_dir / "planned_qpos_deg.json", qpos_samples, joint_names=joint_names)
    camera = RealSenseCamera(
        serial=serial,
        width=width,
        height=height,
        fps=fps,
        enable_depth=False,
        warmup_frames=int(args.warmup_frames),
        auto_exposure=None if auto_exposure is None else bool(auto_exposure),
        exposure_us=None if exposure_us is None else float(exposure_us),
        gain=None if gain is None else float(gain),
    )
    urdf = load_urdf(args.urdf_path)
    board_cfg = _board_config(args, settings)
    base_T_ee_list = []
    detections: list[BoardDetection] = []
    qpos_records = []
    frames = []
    used_link_name = None

    def record_observation(idx: int, qpos_now: np.ndarray, base_T_ee: np.ndarray, frame, det: BoardDetection) -> np.ndarray:
        base_T_ee_list.append(base_T_ee)
        detections.append(det)
        qpos_records.append(qpos_now)
        frames.append(
            {
                "index": idx,
                "qpos_rad": qpos_now,
                "base_T_ee": base_T_ee,
                "board_ok": det.ok,
                "board_reprojection_rmse_px": det.reprojection_rmse_px,
                "board_quality": det.quality,
                "reason": det.reason,
            }
        )
        image_write(run_dir / "images" / f"{idx:04d}.png", frame.color_bgr)
        overlay = draw_board_overlay_image(frame.color_bgr, det, frame.intrinsic, frame.dist_coeffs)
        image_write(run_dir / "board_overlays" / f"{idx:04d}.png", overlay)
        if len(qpos_records) == 1:
            np.save(run_dir / "camera_intrinsic.npy", frame.intrinsic)
            np.save(run_dir / "camera_dist_coeffs.npy", frame.dist_coeffs)
        print(
            f"[wrist-camera] sample {idx:04d}: board_ok={det.ok} "
            f"rmse={det.reprojection_rmse_px} reason={det.reason}"
        )
        return overlay

    window_name = "rm75 wrist camera calibration"
    try:
        camera.start()
        manual_live_preview = bool(args.manual_capture and not args.manual_terminal)
        if manual_live_preview:
            target_count = max(1, int(args.manual_count))
            print("[wrist-camera] manual capture mode")
            print("[wrist-camera] Live preview is enabled. Enter/Space=capture, q/Esc=finish.")
            print("[wrist-camera] You can press Enter in this terminal, or focus the preview window and press Space/Enter.")
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            idx = 0
            while idx < target_count:
                frame = camera.capture()
                det = detect_board_pose(frame.color_bgr, frame.intrinsic, frame.dist_coeffs, board_cfg)
                overlay = draw_board_overlay_image(frame.color_bgr, det, frame.intrinsic, frame.dist_coeffs)
                overlay = _draw_manual_capture_status(overlay, det, sample_index=idx, target_count=target_count)
                cv2.imshow(window_name, overlay)
                key_command = _manual_command_from_key(cv2.waitKey(max(1, int(args.preview_delay_ms))))
                terminal_command = _manual_command_from_stdin()
                command = key_command or terminal_command
                if command == "finish":
                    print("[wrist-camera] manual capture requested finish")
                    break
                if command != "capture":
                    continue
                qpos_now = np.asarray(controller.get_qpos(flat=True), dtype=np.float64)
                link_poses = link_poses_from_qpos(urdf, qpos_now, joint_names)
                used_link_name, base_T_ee = select_link_pose(link_poses, args.ee_link_name)
                saved_overlay = record_observation(idx, qpos_now, base_T_ee, frame, det)
                cv2.imshow(window_name, _draw_manual_capture_status(saved_overlay, det, sample_index=idx + 1, target_count=target_count))
                cv2.waitKey(250)
                idx += 1
        else:
            if args.manual_capture:
                print("[wrist-camera] manual capture mode")
                print("[wrist-camera] Move the arm/board to a new view, then press Enter to capture.")
                print("[wrist-camera] Type q then Enter to finish.")
            sample_iter = range(max(1, int(args.manual_count)))
            if not args.manual_capture:
                sample_iter = enumerate(qpos_samples)
            for item in sample_iter:
                if args.manual_capture:
                    idx = int(item)
                    user_input = input(f"[wrist-camera] sample {idx:04d}/{int(args.manual_count)}: Enter=capture, q=finish > ").strip().lower()
                    if user_input in {"q", "quit", "done", "exit"}:
                        break
                else:
                    idx, qpos = item
                    if args.execute_real and not args.capture_current_only:
                        print(f"[wrist-camera] moving to sample {idx + 1}/{len(qpos_samples)}")
                        controller.move_to_qpos(qpos)
                        time.sleep(float(args.settle_s))
                qpos_now = np.asarray(controller.get_qpos(flat=True), dtype=np.float64)
                frame = camera.capture()
                det = detect_board_pose(frame.color_bgr, frame.intrinsic, frame.dist_coeffs, board_cfg)
                link_poses = link_poses_from_qpos(urdf, qpos_now, joint_names)
                used_link_name, base_T_ee = select_link_pose(link_poses, args.ee_link_name)
                overlay = record_observation(idx, qpos_now, base_T_ee, frame, det)
                if args.preview:
                    cv2.imshow(window_name, overlay)
                    key_command = _manual_command_from_key(cv2.waitKey(max(1, int(args.preview_delay_ms))))
                    if key_command == "finish":
                        print("[wrist-camera] preview requested stop")
                        break
    finally:
        if args.preview or (args.manual_capture and not args.manual_terminal):
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
        camera.stop()
        controller.disconnect()
    save_json(
        run_dir / "observations.json",
        {
            "ee_link_name": used_link_name,
            "joint_names": joint_names,
            "frames": frames,
            "detections": [det.jsonable() for det in detections],
        },
    )
    if not qpos_records:
        raise RuntimeError("No manual samples were captured.")
    np.save(run_dir / "qpos_rad.npy", np.stack(qpos_records))
    np.save(run_dir / "base_T_ee.npy", np.stack(base_T_ee_list))
    return base_T_ee_list, detections, np.load(run_dir / "camera_intrinsic.npy"), np.load(run_dir / "camera_dist_coeffs.npy")


def _load_previous(run_dir: Path) -> tuple[list[np.ndarray], list[BoardDetection], np.ndarray, np.ndarray]:
    data = load_json(run_dir / "observations.json")
    base_T_ee_list = [as_transform(item["base_T_ee"]) for item in data["frames"]]
    detections = []
    for item in data["detections"]:
        T = None if item.get("T_cam_board") is None else as_transform(item["T_cam_board"])
        detections.append(
            BoardDetection(
                ok=bool(item.get("ok", False)),
                T_cam_board=T,
                corners=np.asarray(item.get("corners", []), dtype=np.float64).reshape(-1, 2),
                ids=np.asarray(item.get("ids", []), dtype=np.int32).reshape(-1),
                reprojection_rmse_px=item.get("reprojection_rmse_px"),
                reason=str(item.get("reason", "")),
                quality=dict(item.get("quality", {})),
            )
        )
    intrinsic = np.load(run_dir / "camera_intrinsic.npy")
    dist_path = run_dir / "camera_dist_coeffs.npy"
    dist_coeffs = np.load(dist_path) if dist_path.exists() else np.zeros((5,), dtype=np.float64)
    return base_T_ee_list, detections, intrinsic, dist_coeffs


def main() -> int:
    args = parse_args()
    settings = load_realman_settings(args.config)
    run_dir = Path(args.use_previous_run).expanduser().resolve() if args.use_previous_run else calibration_run_dir("wrist_camera_board", args.output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run_plan:
        qpos_samples = _planned_qpos(args, settings, controller=None)
        if not qpos_samples:
            raise ValueError("No wrist_reference_qpos_deg/reference_qpos_deg samples found for dry-run planning.")
        save_qpos_plan(run_dir / "planned_qpos_deg.json", qpos_samples, joint_names=_joint_names(args, settings))
        print(f"[wrist-camera] dry-run plan written to {run_dir / 'planned_qpos_deg.json'}")
        return 0
    board_cfg = _board_config(args, settings)
    if args.use_previous_run:
        base_T_ee_list, detections, intrinsic, dist_coeffs = _load_previous(run_dir)
    else:
        base_T_ee_list, detections, intrinsic, dist_coeffs = _collect_observations(args, settings, run_dir)
    valid_base_T_ee = []
    valid_cam_T_board = []
    valid_detections = []
    outliers = []
    for idx, (base_T_ee, det) in enumerate(zip(base_T_ee_list, detections)):
        if detection_is_accepted(det, board_cfg) and det.T_cam_board is not None:
            valid_base_T_ee.append(base_T_ee)
            valid_cam_T_board.append(det.T_cam_board)
            valid_detections.append(det)
        else:
            outliers.append({"index": idx, "reason": det.reason, "quality": det.quality})
    min_valid = max(3, int(args.min_valid_frames))
    if len(valid_base_T_ee) < min_valid:
        raise RuntimeError(
            f"Only {len(valid_base_T_ee)} accepted board detections. "
            f"Need at least {min_valid}; 15+ is preferable for the wrist camera."
        )
    initial_ee_T_cam, hand_eye_metrics = solve_eye_in_hand_multi_method(valid_base_T_ee, valid_cam_T_board)
    ee_T_cam = initial_ee_T_cam
    base_T_board = as_transform(hand_eye_metrics["base_T_board"])
    opt_metrics = {"optimized": False, "reason": "disabled"}
    if not args.disable_reprojection_opt:
        fixed_board = None
        if args.fixed_base_board is not None:
            fixed_board = as_transform(np.load(Path(args.fixed_base_board).expanduser()))
        ee_T_cam, base_T_board, opt_metrics = optimize_hand_eye_reprojection(
            valid_base_T_ee,
            valid_detections,
            intrinsic,
            dist_coeffs,
            board_cfg,
            initial_ee_T_cam,
            fixed_base_T_board=fixed_board,
        )
    save_matrix_pair(run_dir / "ee_T_wrist_camera.npy", ee_T_cam)
    save_matrix_pair(run_dir / "wrist_camera_T_ee.npy", np.linalg.inv(ee_T_cam))
    save_matrix_pair(run_dir / "base_T_board.npy", base_T_board)
    report = {
        "run_dir": run_dir,
        "total_frames": len(detections),
        "accepted_frames": len(valid_base_T_ee),
        "rejected_frames": len(detections) - len(valid_base_T_ee),
        "selected_hand_eye_method": hand_eye_metrics.get("selected_method", hand_eye_metrics.get("method")),
        "pose_diversity": pose_diversity_metrics(valid_base_T_ee),
        "camera_intrinsic": intrinsic,
        "board_config": vars(board_cfg),
        "initial_ee_T_wrist_camera": initial_ee_T_cam,
        "ee_T_wrist_camera": ee_T_cam,
        "base_T_board": base_T_board,
        "hand_eye_metrics": hand_eye_metrics,
        "optimization_metrics": opt_metrics,
        "outliers": outliers,
    }
    report = normalize_calibration_report(
        report,
        run_kind="wrist_camera_board",
        config_path=args.config,
        frames_total=len(detections),
        frames_used=len(valid_base_T_ee),
        accepted=True,
        transforms={
            "T_R_P": base_T_board,
            "T_E_Cw": ee_T_cam,
        },
        metrics={
            "selected_hand_eye_method": hand_eye_metrics.get("selected_method", hand_eye_metrics.get("method")),
            "pose_diversity": pose_diversity_metrics(valid_base_T_ee),
            "hand_eye": hand_eye_metrics,
            "optimization": opt_metrics,
        },
    )
    save_json(run_dir / "calibration_report.json", report)
    if not args.no_html_report:
        write_html_report(
            run_dir / "report.html",
            title="RM75 Wrist Camera Calibration",
            summary=report,
            sections=[
                ("Hand Eye Metrics", hand_eye_metrics),
                ("Optimization Metrics", opt_metrics),
            ],
            image_dirs=["board_overlays"],
        )
    print(f"[wrist-camera] valid detections: {len(valid_base_T_ee)}")
    print(f"[wrist-camera] calibration written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
