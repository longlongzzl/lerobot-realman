from __future__ import annotations

import argparse
import select
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rm75_app.paths import ASSET_DIR, DEFAULT_CAMERA_EXTRINSIC

from .board import (
    board_config_from_args,
    board_object_points,
    detect_board_pose,
    detection_is_accepted,
    draw_board_overlay_image,
    make_charuco_board,
)
from .common import (
    RealSenseCamera,
    RealmanSdkController,
    as_transform,
    calibration_run_dir,
    camera_opencv_to_base_camera,
    image_write,
    load_matrix,
    load_realman_settings,
    save_json,
    transform_to_rvec_tvec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate fixed/global camera calibration by detecting a ChArUco board point "
            "and optionally moving the real robot so a configured physical tip hovers above that point."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ASSET_DIR / "calibration" / "rm75_calibration_config.example.json",
        help="JSON with realman_settings.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--base-camera-extrinsic-opencv-path", type=Path, default=DEFAULT_CAMERA_EXTRINSIC)
    parser.add_argument(
        "--base-camera-use-direct",
        action="store_true",
        help="Treat --base-camera-extrinsic-opencv-path as base_T_camera instead of camera_T_base.",
    )
    parser.add_argument("--robot-ip", type=str, default=None)
    parser.add_argument("--robot-port", type=int, default=None)
    parser.add_argument("--camera-serial", type=str, default=None)
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preview-delay-ms", type=int, default=1)
    parser.add_argument("--terminal", action="store_true", help="Use terminal prompts instead of the live OpenCV preview.")
    parser.add_argument(
        "--target-corner-id",
        type=int,
        default=None,
        help="ChArUco inner-corner id to point at. Defaults to the inner corner nearest the board center.",
    )
    parser.add_argument(
        "--target-xy-m",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Override target point in board coordinates, meters.",
    )
    parser.add_argument(
        "--execute-real",
        action="store_true",
        help="Move the real robot to the detected target hover pose. Without this, only preview/logging runs.",
    )
    parser.add_argument(
        "--touch",
        action="store_true",
        help="After hover, also move down to final-offset above the board target, then retreat.",
    )
    parser.add_argument(
        "--hover-height-m",
        type=float,
        default=0.300,
        help="Physical tip hover distance along the board normal. Defaults high for real-robot safety.",
    )
    parser.add_argument(
        "--final-offset-m",
        type=float,
        default=0.050,
        help="Physical tip final stop distance along the board normal from the board surface.",
    )
    parser.add_argument(
        "--tip-offset-tool-m",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help=(
            "Vector from the controller active TCP to the real contact tip in the current TCP/tool frame, meters. "
            "Example: 0 0 -0.12 if the tip is 12 cm along tool -Z; verify sign with a high hover first."
        ),
    )
    parser.add_argument(
        "--allow-touch-without-tip-offset",
        action="store_true",
        help="Allow --touch while assuming controller TCP is exactly the physical contact point.",
    )
    parser.add_argument(
        "--allow-low-final-offset",
        action="store_true",
        help="Allow --touch with --final-offset-m below 2 cm.",
    )
    parser.add_argument(
        "--max-commanded-descend-m",
        type=float,
        default=0.150,
        help="Refuse a single commanded TCP move that descends more than this from the current pose unless overridden.",
    )
    parser.add_argument(
        "--allow-large-descend",
        action="store_true",
        help="Allow commanded TCP descents larger than --max-commanded-descend-m.",
    )
    parser.add_argument("--hover-speed", type=int, default=10)
    parser.add_argument("--touch-speed", type=int, default=3)
    parser.add_argument(
        "--assume-current-tool-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the current TCP orientation and only replace XYZ with the board target.",
    )
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


def _manual_command_from_key(key: int) -> str | None:
    key = int(key) & 0xFF
    if key in (27, ord("q")):
        return "finish"
    if key in (10, 13, 32):
        return "capture"
    return None


def _manual_command_from_stdin() -> str | None:
    if not sys.stdin or not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return None
    value = sys.stdin.readline().strip().lower()
    if value in {"q", "quit", "done", "exit"}:
        return "finish"
    return "capture"


def _load_base_T_camera(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    path = Path(args.base_camera_extrinsic_opencv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"base camera extrinsic not found: {path}")
    mat = load_matrix(path)
    return camera_opencv_to_base_camera(mat, use_direct=bool(args.base_camera_use_direct)), str(path)


def _default_target_corner_id(board_points: np.ndarray, board_width: float, board_height: float) -> int:
    target = np.asarray([board_width * 0.5, board_height * 0.5, 0.0], dtype=np.float64)
    distances = np.linalg.norm(np.asarray(board_points, dtype=np.float64).reshape(-1, 3) - target, axis=1)
    return int(np.argmin(distances))


def _target_point_from_args(args: argparse.Namespace, board_cfg) -> tuple[np.ndarray, dict[str, Any]]:
    board = make_charuco_board(board_cfg)
    points = board_object_points(board)
    width = float(board_cfg.squares_x) * float(board_cfg.square_length_m)
    height = float(board_cfg.squares_y) * float(board_cfg.square_length_m)
    if args.target_xy_m is not None:
        xy = np.asarray(args.target_xy_m, dtype=np.float64).reshape(2)
        target = np.asarray([xy[0], xy[1], 0.0], dtype=np.float64)
        return target, {"mode": "xy", "target_xy_m": xy.tolist()}
    corner_id = int(args.target_corner_id) if args.target_corner_id is not None else _default_target_corner_id(points, width, height)
    if corner_id < 0 or corner_id >= points.shape[0]:
        raise ValueError(f"target corner id {corner_id} out of range [0, {points.shape[0] - 1}]")
    return points[corner_id].astype(np.float64), {"mode": "corner_id", "target_corner_id": corner_id}


def _transform_point(T: np.ndarray, point_xyz: np.ndarray) -> np.ndarray:
    point = np.asarray([point_xyz[0], point_xyz[1], point_xyz[2], 1.0], dtype=np.float64)
    return (as_transform(T) @ point)[:3]


def _board_approach_direction(base_T_board: np.ndarray) -> np.ndarray:
    normal = as_transform(base_T_board)[:3, :3] @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if float(normal[2]) < 0.0:
        normal = -normal
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return normal / norm


def _project_board_point(image_bgr: np.ndarray, det, intrinsic: np.ndarray, dist_coeffs: np.ndarray, target_board: np.ndarray) -> np.ndarray:
    out = draw_board_overlay_image(image_bgr, det, intrinsic, dist_coeffs)
    if det.T_cam_board is None:
        return out
    rvec, tvec = transform_to_rvec_tvec(det.T_cam_board)
    projected, _ = cv2.projectPoints(
        np.asarray(target_board, dtype=np.float64).reshape(1, 3),
        rvec,
        tvec,
        np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
        np.asarray(dist_coeffs, dtype=np.float64).reshape(-1),
    )
    u, v = np.round(projected.reshape(2)).astype(int)
    cv2.drawMarker(out, (u, v), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=28, thickness=3)
    cv2.circle(out, (u, v), 12, (0, 0, 255), 2)
    return out


def _draw_status(
    image_bgr: np.ndarray,
    *,
    sample_index: int,
    sample_count: int,
    det_ok: bool,
    rmse: float | None,
    target_base: np.ndarray | None,
    execute_real: bool,
) -> np.ndarray:
    out = image_bgr.copy()
    height, width = out.shape[:2]
    left = "OK" if det_ok else "NO BOARD"
    rmse_txt = "rmse=NA" if rmse is None else f"rmse={float(rmse):.2f}px"
    if target_base is None:
        target_txt = "target_base=NA"
    else:
        target_txt = "target_base_mm=[" + ", ".join(f"{v * 1000.0:.1f}" for v in target_base) + "]"
    mode = "move" if execute_real else "log"
    line = f"{left} {rmse_txt} {target_txt} sample {sample_index}/{sample_count} {mode} Enter/Space=accept q/Esc=finish"
    overlay = out.copy()
    cv2.rectangle(overlay, (8, height - 56), (min(width - 8, 1260), height - 8), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, out, 0.42, 0.0, out)
    cv2.putText(out, line[:160], (18, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (235, 235, 235), 2, cv2.LINE_AA)
    return out


def _connect_robot(args: argparse.Namespace, settings: dict[str, Any]) -> RealmanSdkController:
    robot_ip = args.robot_ip or settings.get("robot_ip")
    if not robot_ip:
        raise ValueError("robot_ip is required in --config or --robot-ip")
    return RealmanSdkController(
        robot_ip=str(robot_ip),
        robot_port=int(args.robot_port or settings.get("robot_port", 8080)),
        joint_names=list(settings.get("joint_names") or [f"joint_{idx}" for idx in range(1, 8)]),
        calibration_offset=settings.get("calibration_offset", {}),
    )


def _current_pose(controller: RealmanSdkController) -> list[float]:
    state = controller._read_state()
    pose = state.get("pose")
    if pose is None or len(pose) < 6:
        raise RuntimeError(f"current arm state has no 6D pose: keys={sorted(state.keys())}")
    return [float(value) for value in pose[:6]]


def _move_pose(controller: RealmanSdkController, pose: list[float], *, speed: int, linear: bool) -> None:
    speed = int(max(1, min(100, speed)))
    if linear:
        ret = controller.arm.rm_movel([float(v) for v in pose], speed, 0, 0, 1)
        name = "rm_movel"
    else:
        ret = controller.arm.rm_movej_p([float(v) for v in pose], speed, 0, 0, 1)
        name = "rm_movej_p"
    if int(ret) != 0:
        raise RuntimeError(f"{name} failed with code {ret}")


def _pose_rotation_matrix(pose: list[float]) -> np.ndarray:
    from transforms3d.euler import euler2mat

    rx, ry, rz = [float(v) for v in pose[3:6]]
    return np.asarray(euler2mat(rx, ry, rz, axes="sxyz"), dtype=np.float64).reshape(3, 3)


def _commanded_tcp_pose_for_tip(tip_base: np.ndarray, pose_template: list[float], tip_offset_tool: np.ndarray) -> list[float]:
    rotation_base_tool = _pose_rotation_matrix(pose_template)
    tcp_base = np.asarray(tip_base, dtype=np.float64).reshape(3) - rotation_base_tool @ np.asarray(tip_offset_tool, dtype=np.float64).reshape(3)
    return [float(tcp_base[0]), float(tcp_base[1]), float(tcp_base[2]), *pose_template[3:6]]


def _validate_motion_request(
    args: argparse.Namespace,
    *,
    pose_now: list[float],
    hover_pose: list[float],
    final_pose: list[float],
    tip_offset_tool: np.ndarray,
) -> None:
    if float(args.hover_height_m) <= 0.0:
        raise ValueError("--hover-height-m must be positive")
    if float(args.final_offset_m) < 0.0:
        raise ValueError("--final-offset-m must be non-negative")
    if bool(args.touch) and float(args.final_offset_m) > float(args.hover_height_m):
        raise ValueError("--final-offset-m cannot be greater than --hover-height-m when --touch is used")

    tip_offset_norm = float(np.linalg.norm(tip_offset_tool))
    if bool(args.touch) and tip_offset_norm < 1e-6 and not bool(args.allow_touch_without_tip_offset):
        raise RuntimeError(
            "--touch is refused because --tip-offset-tool-m is zero. "
            "First measure/configure the probe/gripper-tip offset, or pass --allow-touch-without-tip-offset only if the active TCP is already the contact point."
        )
    if bool(args.touch) and float(args.final_offset_m) < 0.020 and not bool(args.allow_low_final_offset):
        raise RuntimeError(
            "--touch with final offset below 2 cm is refused by default. "
            "Use --final-offset-m 0.02 or higher for verification, or pass --allow-low-final-offset after the tip offset has been verified."
        )

    max_descend = float(args.max_commanded_descend_m)
    if max_descend > 0.0 and not bool(args.allow_large_descend):
        current_z = float(pose_now[2])
        hover_descend = current_z - float(hover_pose[2])
        final_descend = current_z - float(final_pose[2])
        planned_descend = max(hover_descend, final_descend if bool(args.touch) else hover_descend)
        if planned_descend > max_descend:
            raise RuntimeError(
                f"commanded TCP would descend {planned_descend * 1000.0:.1f} mm from the current pose, "
                f"over --max-commanded-descend-m {max_descend * 1000.0:.1f} mm. "
                "Move closer manually, increase hover height, or pass --allow-large-descend after checking the pose."
            )


def _confirm(prompt: str) -> bool:
    value = input(f"{prompt} [Enter=yes, s=skip, q=quit] ").strip().lower()
    if value in {"q", "quit", "exit"}:
        raise KeyboardInterrupt
    return value not in {"s", "skip", "n", "no"}


def _execute_target_motion(
    args: argparse.Namespace,
    controller: RealmanSdkController,
    target_base: np.ndarray,
    approach_dir: np.ndarray,
) -> dict[str, Any]:
    pose_now = _current_pose(controller)
    if not bool(args.assume_current_tool_pose):
        raise ValueError("Only --assume-current-tool-pose is currently implemented.")
    tip_offset_tool = np.asarray(args.tip_offset_tool_m, dtype=np.float64).reshape(3)
    rotation_base_tool = _pose_rotation_matrix(pose_now)
    tip_offset_base = rotation_base_tool @ tip_offset_tool
    hover_tip_xyz = np.asarray(target_base, dtype=np.float64).reshape(3) + approach_dir * float(args.hover_height_m)
    final_tip_xyz = np.asarray(target_base, dtype=np.float64).reshape(3) + approach_dir * float(args.final_offset_m)
    hover_pose = _commanded_tcp_pose_for_tip(hover_tip_xyz, pose_now, tip_offset_tool)
    final_pose = _commanded_tcp_pose_for_tip(final_tip_xyz, pose_now, tip_offset_tool)
    if float(np.linalg.norm(tip_offset_tool)) < 1e-6:
        print("[base-point-check] WARNING: tip offset is zero; assuming controller TCP is the physical tip. Keep hover high.")
    print(f"[base-point-check] current TCP xyz_mm={np.round(np.asarray(pose_now[:3]) * 1000.0, 2).tolist()} rpy={np.round(pose_now[3:6], 4).tolist()}")
    print(f"[base-point-check] tip_offset_tool_mm={np.round(tip_offset_tool * 1000.0, 2).tolist()}")
    print(f"[base-point-check] tip_offset_base_mm={np.round(tip_offset_base * 1000.0, 2).tolist()}  # TCP -> physical tip in base frame")
    print(f"[base-point-check] hover physical tip xyz_mm={np.round(hover_tip_xyz * 1000.0, 2).tolist()}")
    print(f"[base-point-check] commanded TCP hover xyz_mm={np.round(np.asarray(hover_pose[:3]) * 1000.0, 2).tolist()} rpy={np.round(hover_pose[3:], 4).tolist()}")
    _validate_motion_request(
        args,
        pose_now=pose_now,
        hover_pose=hover_pose,
        final_pose=final_pose,
        tip_offset_tool=tip_offset_tool,
    )
    if _confirm("move active TCP so the configured physical tip hovers above detected board point?"):
        _move_pose(controller, hover_pose, speed=int(args.hover_speed), linear=False)
    else:
        return {
            "executed_hover": False,
            "executed_touch": False,
            "current_pose_before": pose_now,
            "tip_offset_tool_m": tip_offset_tool,
            "hover_tip_target_m": hover_tip_xyz,
            "final_tip_target_m": final_tip_xyz,
            "hover_pose": hover_pose,
            "final_pose": final_pose,
        }
    executed_touch = False
    if bool(args.touch):
        print(f"[base-point-check] final physical tip xyz_mm={np.round(final_tip_xyz * 1000.0, 2).tolist()} offset_m={float(args.final_offset_m):.4f}")
        print(f"[base-point-check] commanded TCP final xyz_mm={np.round(np.asarray(final_pose[:3]) * 1000.0, 2).tolist()}")
        if _confirm("move down so the configured physical tip reaches the final offset near the board point?"):
            _move_pose(controller, final_pose, speed=int(args.touch_speed), linear=True)
            executed_touch = True
            if _confirm("retreat back to hover?"):
                _move_pose(controller, hover_pose, speed=int(args.touch_speed), linear=True)
    return {
        "executed_hover": True,
        "executed_touch": executed_touch,
        "current_pose_before": pose_now,
        "tip_offset_tool_m": tip_offset_tool,
        "hover_tip_target_m": hover_tip_xyz,
        "final_tip_target_m": final_tip_xyz,
        "hover_pose": hover_pose,
        "final_pose": final_pose,
    }


def main() -> int:
    args = parse_args()
    settings = load_realman_settings(args.config)
    run_dir = calibration_run_dir("base_camera_point_accuracy_check", args.output_root)
    base_T_camera, extrinsic_source = _load_base_T_camera(args)
    board_cfg = board_config_from_args(args, settings)
    target_board, target_meta = _target_point_from_args(args, board_cfg)
    camera_cfg = settings.get("base_camera", settings.get("camera", {}))
    serial = args.camera_serial or camera_cfg.get("serial_number_or_name")
    width = int(args.camera_width or camera_cfg.get("width", 1280))
    height = int(args.camera_height or camera_cfg.get("height", 720))
    fps = int(args.camera_fps or camera_cfg.get("fps", 30))
    print(f"[base-point-check] run_dir={run_dir}")
    print(f"[base-point-check] base camera extrinsic={extrinsic_source}")
    print(f"[base-point-check] target_board_m={np.round(target_board, 5).tolist()} meta={target_meta}")
    if args.execute_real and not args.touch:
        print("[base-point-check] --execute-real without --touch: robot will only move to hover poses.")
    camera = RealSenseCamera(
        serial=serial,
        width=width,
        height=height,
        fps=fps,
        enable_depth=False,
        warmup_frames=int(args.warmup_frames),
    )
    controller = _connect_robot(args, settings) if bool(args.execute_real) else None
    records: list[dict[str, Any]] = []
    window_name = "rm75 base camera point accuracy check"
    sample_count = max(1, int(args.sample_count))
    try:
        camera.start()
        print("[base-point-check] Move the board, then press Enter/Space to accept the current detection.")
        print("[base-point-check] Type q then Enter, or press q/Esc in the preview, to finish.")
        while len(records) < sample_count:
            frame = camera.capture()
            det = detect_board_pose(frame.color_bgr, frame.intrinsic, frame.dist_coeffs, board_cfg)
            accepted = bool(detection_is_accepted(det, board_cfg) and det.T_cam_board is not None)
            base_T_board = None
            target_base = None
            approach_dir = None
            if accepted:
                base_T_board = base_T_camera @ as_transform(det.T_cam_board)
                target_base = _transform_point(base_T_board, target_board)
                approach_dir = _board_approach_direction(base_T_board)
            overlay = _project_board_point(frame.color_bgr, det, frame.intrinsic, frame.dist_coeffs, target_board)
            overlay = _draw_status(
                overlay,
                sample_index=len(records),
                sample_count=sample_count,
                det_ok=accepted,
                rmse=det.reprojection_rmse_px,
                target_base=target_base,
                execute_real=bool(args.execute_real),
            )
            command = None
            if args.preview and not args.terminal:
                cv2.imshow(window_name, overlay)
                command = _manual_command_from_key(cv2.waitKey(max(1, int(args.preview_delay_ms))))
            command = command or _manual_command_from_stdin()
            if args.terminal and command is None:
                value = input(f"[base-point-check] sample {len(records):04d}/{sample_count}: Enter=accept, q=finish > ")
                command = "finish" if value.strip().lower() in {"q", "quit", "done", "exit"} else "capture"
            if command == "finish":
                break
            if command != "capture":
                continue
            idx = len(records)
            image_write(run_dir / "images" / f"{idx:04d}.png", frame.color_bgr)
            image_write(run_dir / "overlays" / f"{idx:04d}.png", overlay)
            rec: dict[str, Any] = {
                "index": idx,
                "accepted": accepted,
                "target_board_m": target_board,
                "target_metadata": target_meta,
                "board_detection": det.jsonable(),
                "base_T_board": base_T_board,
                "target_base_m": target_base,
                "approach_direction_base": approach_dir,
            }
            if not accepted:
                print(f"[base-point-check] sample {idx:04d}: board rejected: {det.reason}")
                records.append(rec)
                continue
            print(
                "[base-point-check] sample "
                f"{idx:04d}: rmse={det.reprojection_rmse_px:.3f}px "
                f"target_base_mm={np.round(target_base * 1000.0, 2).tolist()}"
            )
            if controller is not None:
                try:
                    rec["motion"] = _execute_target_motion(args, controller, target_base, approach_dir)
                except RuntimeError as exc:
                    print(f"[base-point-check] motion refused/failed: {exc}")
                    rec["motion"] = {
                        "executed_hover": False,
                        "executed_touch": False,
                        "error": str(exc),
                    }
            records.append(rec)
    finally:
        if args.preview and not args.terminal:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
        camera.stop()
        if controller is not None:
            controller.disconnect()
    accepted_records = [item for item in records if bool(item.get("accepted"))]
    reproj = [
        float(item["board_detection"]["reprojection_rmse_px"])
        for item in accepted_records
        if item.get("board_detection", {}).get("reprojection_rmse_px") is not None
    ]
    summary = {
        "kind": "base_camera_point_accuracy_check",
        "run_dir": str(run_dir),
        "config": str(Path(args.config).expanduser()),
        "base_camera_extrinsic_source": extrinsic_source,
        "base_camera_use_direct": bool(args.base_camera_use_direct),
        "camera": {"serial": serial, "width": width, "height": height, "fps": fps},
        "board_config": asdict(board_cfg),
        "target": {"point_board_m": target_board, **target_meta},
        "execute_real": bool(args.execute_real),
        "touch": bool(args.touch),
        "sample_count": len(records),
        "accepted_count": len(accepted_records),
        "reprojection_rmse_px": {
            "mean": None if not reproj else float(np.mean(reproj)),
            "median": None if not reproj else float(np.median(reproj)),
            "max": None if not reproj else float(np.max(reproj)),
        },
        "records": records,
    }
    save_json(run_dir / "point_accuracy_report.json", summary)
    print(f"[base-point-check] accepted {len(accepted_records)}/{len(records)}")
    if reproj:
        print(
            "[base-point-check] reprojection rmse px "
            f"mean={np.mean(reproj):.3f} median={np.median(reproj):.3f} max={np.max(reproj):.3f}"
        )
    print(f"[base-point-check] report written to {run_dir / 'point_accuracy_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
