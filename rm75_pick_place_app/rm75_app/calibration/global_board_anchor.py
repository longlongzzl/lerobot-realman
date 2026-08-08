from __future__ import annotations

import argparse
import html
import select
import sys
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
    detect_board_pose,
    detection_is_accepted,
    draw_board_overlay_image,
)
from .common import (
    RealSenseCamera,
    as_transform,
    average_transforms,
    calibration_run_dir,
    camera_opencv_to_base_camera,
    image_write,
    invert_transform,
    load_matrix,
    load_realman_settings,
    pose_error,
    save_json,
    save_matrix_pair,
    to_jsonable,
)
from .reporting import normalize_calibration_report


@dataclass
class AnchorFrame:
    index: int
    image_bgr: np.ndarray
    intrinsic: np.ndarray
    dist_coeffs: np.ndarray
    source_path: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anchor a fixed ChArUco board in robot-base coordinates with the calibrated global camera."
    )
    parser.add_argument("--config", type=Path, default=None, help="JSON with realman_settings.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--input-run",
        type=Path,
        default=None,
        help="Reuse images from a previous global-board-anchor, base-camera, or image run.",
    )
    parser.add_argument("--base-camera-run-dir", type=Path, default=None)
    parser.add_argument("--base-camera-extrinsic-opencv-path", type=Path, default=DEFAULT_CAMERA_EXTRINSIC)
    parser.add_argument(
        "--base-camera-use-direct",
        action="store_true",
        help="Interpret camera_extrinsic_opencv as direct T_R_Cg instead of the legacy inverse convention.",
    )
    parser.add_argument("--camera-intrinsic-path", type=Path, default=None)
    parser.add_argument("--camera-dist-coeffs-path", type=Path, default=None)
    parser.add_argument("--manual-capture", action="store_true", help="Capture global-camera frames manually.")
    parser.add_argument("--manual-count", type=int, default=20)
    parser.add_argument("--manual-terminal", action="store_true", help="Use terminal-only manual capture.")
    parser.add_argument("--camera-serial", type=str, default=None)
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--preview", action="store_true", help="Show live overlays during manual capture.")
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
    parser.add_argument("--min-valid-frames", type=int, default=1)
    parser.add_argument("--disable-pose-outlier-rejection", action="store_true")
    parser.add_argument("--max-pose-translation-from-median-m", type=float, default=0.03)
    parser.add_argument("--max-pose-rotation-from-average-deg", type=float, default=5.0)
    parser.add_argument("--no-html-report", action="store_true")
    return parser.parse_args()


def _board_config(args: argparse.Namespace, settings: dict[str, Any]) -> BoardConfig:
    return board_config_from_args(args, settings)


def _load_base_camera_transform(args: argparse.Namespace) -> tuple[np.ndarray, Path]:
    if args.base_camera_run_dir is not None:
        run_dir = Path(args.base_camera_run_dir).expanduser().resolve()
        for name in ("base_T_camera.npy", "R_T_global_camera.npy", "R_T_camera.npy"):
            candidate = run_dir / name
            if candidate.exists():
                return load_matrix(candidate), candidate
        candidate = run_dir / "camera_extrinsic_opencv.npy"
        if candidate.exists():
            return (
                camera_opencv_to_base_camera(load_matrix(candidate), use_direct=bool(args.base_camera_use_direct)),
                candidate,
            )
    path = Path(args.base_camera_extrinsic_opencv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            "base camera extrinsic not found. Provide --base-camera-run-dir with base_T_camera.npy "
            "or camera_extrinsic_opencv.npy, or pass --base-camera-extrinsic-opencv-path."
        )
    return camera_opencv_to_base_camera(load_matrix(path), use_direct=bool(args.base_camera_use_direct)), path


def _load_first_existing_matrix(paths: list[Path | None], *, label: str) -> tuple[np.ndarray, Path]:
    checked: list[str] = []
    for path in paths:
        if path is None:
            continue
        candidate = Path(path).expanduser()
        checked.append(str(candidate))
        if candidate.exists():
            if candidate.suffix == ".npy":
                return np.asarray(np.load(candidate), dtype=np.float64), candidate
            return np.asarray(load_matrix(candidate), dtype=np.float64), candidate
    raise FileNotFoundError(f"{label} not found. Checked: {checked}")


def _load_intrinsics(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    input_run = Path(args.input_run).expanduser().resolve() if args.input_run is not None else None
    base_run = Path(args.base_camera_run_dir).expanduser().resolve() if args.base_camera_run_dir is not None else None
    intrinsic, intrinsic_path = _load_first_existing_matrix(
        [
            args.camera_intrinsic_path,
            None if input_run is None else input_run / "camera_intrinsic.npy",
            None if base_run is None else base_run / "camera_intrinsic.npy",
        ],
        label="camera intrinsic",
    )
    dist_candidates = [
        args.camera_dist_coeffs_path,
        None if input_run is None else input_run / "camera_dist_coeffs.npy",
        None if base_run is None else base_run / "camera_dist_coeffs.npy",
    ]
    dist_coeffs = np.zeros((5,), dtype=np.float64)
    dist_path = None
    for path in dist_candidates:
        if path is not None and Path(path).expanduser().exists():
            dist_path = Path(path).expanduser()
            dist_coeffs = np.asarray(np.load(dist_path), dtype=np.float64).reshape(-1)
            break
    return (
        intrinsic.reshape(3, 3),
        dist_coeffs,
        {
            "camera_intrinsic": str(intrinsic_path),
            "camera_dist_coeffs": "" if dist_path is None else str(dist_path),
        },
    )


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return image


def _image_paths_from_run(run_dir: Path) -> list[Path]:
    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for dirname in ("images", "main_images", "base_images"):
        root = run_dir / dirname
        if root.exists():
            paths = sorted(path for path in root.iterdir() if path.suffix.lower() in image_suffixes)
            if paths:
                return paths
    return []


def _frames_from_image_dataset(run_dir: Path, intrinsic: np.ndarray, dist_coeffs: np.ndarray) -> list[AnchorFrame]:
    dataset_path = run_dir / "image_dataset.npy"
    if not dataset_path.exists():
        return []
    raw = np.load(dataset_path, allow_pickle=True)
    if raw.dtype == object:
        item = raw.reshape(-1)[0]
        if isinstance(item, dict):
            for key in ("base_camera", "main_camera", "global_camera", "camera"):
                if key in item:
                    raw = np.asarray(item[key])
                    break
            else:
                return []
    images = np.asarray(raw)
    if images.ndim != 4:
        return []
    return [
        AnchorFrame(
            index=idx,
            image_bgr=np.asarray(image).copy(),
            intrinsic=intrinsic.copy(),
            dist_coeffs=dist_coeffs.copy(),
            source_path=str(dataset_path),
        )
        for idx, image in enumerate(images)
    ]


def _load_input_frames(args: argparse.Namespace) -> tuple[list[AnchorFrame], dict[str, str]]:
    if args.input_run is None:
        raise ValueError("--input-run is required for previous-run mode")
    run_dir = Path(args.input_run).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"input run not found: {run_dir}")
    intrinsic, dist_coeffs, sources = _load_intrinsics(args)
    image_paths = _image_paths_from_run(run_dir)
    if image_paths:
        frames = [
            AnchorFrame(
                index=idx,
                image_bgr=_read_image(path),
                intrinsic=intrinsic.copy(),
                dist_coeffs=dist_coeffs.copy(),
                source_path=str(path),
            )
            for idx, path in enumerate(image_paths)
        ]
        sources["input_images"] = str(image_paths[0].parent)
        return frames, sources
    frames = _frames_from_image_dataset(run_dir, intrinsic, dist_coeffs)
    if frames:
        sources["input_images"] = str(run_dir / "image_dataset.npy")
        return frames, sources
    raise FileNotFoundError(f"no reusable images found under {run_dir}")


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


def _draw_manual_status(
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
    lines = [
        f"sample {sample_index:02d}/{target_count:02d}   Enter/Space: capture   q/Esc: finish",
        f"{status}   corners={len(detection.ids)}   rmse={rmse}",
    ]
    if detection.reason and not detection.ok:
        lines.append(str(detection.reason))
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


def _collect_manual_frames(
    args: argparse.Namespace,
    settings: dict[str, Any],
    board_cfg: BoardConfig,
) -> tuple[list[AnchorFrame], dict[str, str]]:
    camera_cfg = settings.get("base_camera", settings.get("camera", {}))
    camera = RealSenseCamera(
        serial=args.camera_serial or camera_cfg.get("serial_number_or_name"),
        width=int(args.camera_width if args.camera_width is not None else camera_cfg.get("width", 1280)),
        height=int(args.camera_height if args.camera_height is not None else camera_cfg.get("height", 720)),
        fps=int(args.camera_fps if args.camera_fps is not None else camera_cfg.get("fps", 30)),
        enable_depth=False,
        warmup_frames=int(args.warmup_frames),
    )
    target_count = max(1, int(args.manual_count))
    frames: list[AnchorFrame] = []
    window_name = "rm75 global board anchor"
    try:
        camera.start()
        if args.preview and not args.manual_terminal:
            print("[board-anchor] manual capture with preview. Enter/Space=capture, q/Esc=finish.")
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            while len(frames) < target_count:
                frame = camera.capture()
                det = detect_board_pose(frame.color_bgr, frame.intrinsic, frame.dist_coeffs, board_cfg)
                overlay = draw_board_overlay_image(frame.color_bgr, det, frame.intrinsic, frame.dist_coeffs)
                overlay = _draw_manual_status(
                    overlay,
                    det,
                    sample_index=len(frames),
                    target_count=target_count,
                )
                cv2.imshow(window_name, overlay)
                command = _manual_command_from_key(cv2.waitKey(max(1, int(args.preview_delay_ms))))
                command = command or _manual_command_from_stdin()
                if command == "finish":
                    break
                if command != "capture":
                    continue
                frames.append(
                    AnchorFrame(
                        index=len(frames),
                        image_bgr=frame.color_bgr.copy(),
                        intrinsic=frame.intrinsic.copy(),
                        dist_coeffs=frame.dist_coeffs.copy(),
                    )
                )
                print(f"[board-anchor] captured frame {len(frames):04d}/{target_count:04d}")
        else:
            print("[board-anchor] manual capture. Press Enter to capture, q then Enter to finish.")
            while len(frames) < target_count:
                value = input(f"[board-anchor] sample {len(frames):04d}/{target_count}: Enter=capture, q=finish > ")
                if value.strip().lower() in {"q", "quit", "done", "exit"}:
                    break
                frame = camera.capture()
                frames.append(
                    AnchorFrame(
                        index=len(frames),
                        image_bgr=frame.color_bgr.copy(),
                        intrinsic=frame.intrinsic.copy(),
                        dist_coeffs=frame.dist_coeffs.copy(),
                    )
                )
    finally:
        if args.preview and not args.manual_terminal:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
        camera.stop()
    if not frames:
        raise RuntimeError("No manual global-camera frames were captured.")
    return frames, {"input_images": "manual_capture"}


def _frame_reason(det: BoardDetection, board_cfg: BoardConfig) -> str:
    if detection_is_accepted(det, board_cfg):
        return ""
    if det.reason:
        return str(det.reason)
    reason = det.quality.get("reason") if det.quality else ""
    return str(reason or "board detection rejected")


def _select_frames(
    records: list[dict[str, Any]],
    estimates: list[np.ndarray | None],
    args: argparse.Namespace,
) -> list[int]:
    candidates = [idx for idx, record in enumerate(records) if record["accepted_by_board_quality"] and estimates[idx] is not None]
    for idx in candidates:
        records[idx]["selected"] = True
        records[idx]["rejected_reason"] = ""
    if len(candidates) > 2 and not args.disable_pose_outlier_rejection:
        preliminary = average_transforms([as_transform(estimates[idx]) for idx in candidates if estimates[idx] is not None])
        selected = []
        for idx in candidates:
            err = pose_error(as_transform(estimates[idx]), preliminary)
            records[idx]["preliminary_anchor_error"] = err
            if err["translation_m"] > float(args.max_pose_translation_from_median_m):
                records[idx]["selected"] = False
                records[idx]["rejected_reason"] = (
                    "pose translation outlier "
                    f"{err['translation_m']:.4f}m > {float(args.max_pose_translation_from_median_m):.4f}m"
                )
            elif err["rotation_deg"] > float(args.max_pose_rotation_from_average_deg):
                records[idx]["selected"] = False
                records[idx]["rejected_reason"] = (
                    "pose rotation outlier "
                    f"{err['rotation_deg']:.3f}deg > {float(args.max_pose_rotation_from_average_deg):.3f}deg"
                )
            else:
                selected.append(idx)
        return selected
    return candidates


def _mean_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _max_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.max(values))


def _write_anchor_html_report(
    output_path: Path,
    *,
    summary: dict[str, Any],
    frame_records: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scalar_rows = []
    for key, value in summary.items():
        if isinstance(value, (dict, list)):
            continue
        scalar_rows.append(
            "<tr>"
            f"<th>{html.escape(str(key))}</th>"
            f"<td>{html.escape(str(to_jsonable(value)))}</td>"
            "</tr>"
        )
    frame_rows = []
    for record in frame_records:
        err = record.get("anchor_error") or {}
        trans_mm = "" if "translation_m" not in err else f"{float(err['translation_m']) * 1000.0:.2f}"
        rot_deg = "" if "rotation_deg" not in err else f"{float(err['rotation_deg']):.3f}"
        rmse = record.get("board_reprojection_rmse_px")
        rmse_text = "" if rmse is None else f"{float(rmse):.3f}"
        overlay = html.escape(str(record.get("overlay_path", "")))
        overlay_link = f"<a href=\"{overlay}\">{overlay}</a>" if overlay else ""
        frame_rows.append(
            "<tr>"
            f"<td>{int(record['index'])}</td>"
            f"<td>{'yes' if record.get('selected') else 'no'}</td>"
            f"<td>{int(record.get('corner_count', 0))}</td>"
            f"<td>{rmse_text}</td>"
            f"<td>{trans_mm}</td>"
            f"<td>{rot_deg}</td>"
            f"<td>{html.escape(str(record.get('rejected_reason') or ''))}</td>"
            f"<td>{overlay_link}</td>"
            "</tr>"
        )
    cards = []
    for record in frame_records[:24]:
        overlay = record.get("overlay_path")
        if not overlay:
            continue
        caption = f"{record['index']:04d} selected={bool(record.get('selected'))}"
        cards.append(
            "<figure>"
            f"<img src=\"{html.escape(str(overlay))}\" loading=\"lazy\">"
            f"<figcaption>{html.escape(caption)}</figcaption>"
            "</figure>"
        )
    page = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RM75 Global Board Anchor</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #172026; background: #f7f8fa; }}
    h1 {{ font-size: 28px; margin: 0 0 16px; }}
    h2 {{ font-size: 18px; margin: 28px 0 12px; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; border: 1px solid #d8dee4; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #eaeef2; vertical-align: top; }}
    th {{ color: #4d5b66; background: #f0f3f5; }}
    .summary th {{ width: 260px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }}
    figure {{ margin: 0; background: #fff; border: 1px solid #d8dee4; border-radius: 6px; overflow: hidden; }}
    img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ font-size: 12px; color: #4d5b66; padding: 6px 8px; }}
    pre {{ overflow: auto; white-space: pre-wrap; background: #fff; border: 1px solid #d8dee4; padding: 12px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>RM75 Global Board Anchor</h1>
  <table class="summary">{summary_rows}</table>
  <h2>Per-Frame Board Detection</h2>
  <table>
    <thead>
      <tr>
        <th>Frame</th><th>Selected</th><th>Corners</th><th>RMSE px</th>
        <th>Anchor trans mm</th><th>Anchor rot deg</th><th>Rejected reason</th><th>Overlay</th>
      </tr>
    </thead>
    <tbody>{frame_rows}</tbody>
  </table>
  <h2>Overlays</h2>
  <div class="grid">{cards}</div>
  <h2>Transforms</h2>
  <pre>{transforms}</pre>
  <h2>Metrics</h2>
  <pre>{metrics}</pre>
  <h2>Outliers</h2>
  <pre>{outliers}</pre>
  <h2>Metadata</h2>
  <pre>{metadata}</pre>
</body>
</html>
""".format(
        summary_rows="".join(scalar_rows),
        frame_rows="".join(frame_rows),
        cards="".join(cards),
        transforms=html.escape(str(to_jsonable(summary.get("transforms", {})))),
        metrics=html.escape(str(to_jsonable(summary.get("metrics", {})))),
        outliers=html.escape(str(to_jsonable(summary.get("outliers", [])))),
        metadata=html.escape(str(to_jsonable(summary.get("metadata", {})))),
    )
    output_path.write_text(page, encoding="utf-8")


def _process_frames(
    *,
    frames: list[AnchorFrame],
    board_cfg: BoardConfig,
    base_T_camera: np.ndarray,
    run_dir: Path,
    args: argparse.Namespace,
    sources: dict[str, str],
) -> dict[str, Any]:
    if not frames:
        raise ValueError("no frames to process")
    base_T_camera = as_transform(base_T_camera)
    np.save(run_dir / "camera_intrinsic.npy", np.asarray(frames[0].intrinsic, dtype=np.float64).reshape(3, 3))
    np.save(run_dir / "camera_dist_coeffs.npy", np.asarray(frames[0].dist_coeffs, dtype=np.float64).reshape(-1))
    save_matrix_pair(run_dir / "R_T_global_camera.npy", base_T_camera)
    save_matrix_pair(run_dir / "base_T_camera.npy", base_T_camera)

    camera_T_board_per_frame = np.full((len(frames), 4, 4), np.nan, dtype=np.float64)
    robot_T_board_per_frame = np.full((len(frames), 4, 4), np.nan, dtype=np.float64)
    detections: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    estimates: list[np.ndarray | None] = []

    for idx, frame in enumerate(frames):
        image_path = run_dir / "images" / f"{idx:04d}.png"
        overlay_path = run_dir / "overlays" / f"{idx:04d}.png"
        image_write(image_path, frame.image_bgr)
        det = detect_board_pose(frame.image_bgr, frame.intrinsic, frame.dist_coeffs, board_cfg)
        overlay = draw_board_overlay_image(frame.image_bgr, det, frame.intrinsic, frame.dist_coeffs)
        image_write(overlay_path, overlay)

        accepted_by_board = bool(detection_is_accepted(det, board_cfg) and det.T_cam_board is not None)
        camera_T_board = None if det.T_cam_board is None else as_transform(det.T_cam_board)
        robot_T_board = None if camera_T_board is None else base_T_camera @ camera_T_board
        if camera_T_board is not None:
            camera_T_board_per_frame[idx] = camera_T_board
        if robot_T_board is not None:
            robot_T_board_per_frame[idx] = robot_T_board
        detections.append(det.jsonable())
        estimates.append(robot_T_board)
        records.append(
            {
                "index": idx,
                "image_path": image_path.relative_to(run_dir).as_posix(),
                "overlay_path": overlay_path.relative_to(run_dir).as_posix(),
                "source_path": frame.source_path,
                "accepted_by_board_quality": accepted_by_board,
                "selected": False,
                "rejected_reason": "" if accepted_by_board else _frame_reason(det, board_cfg),
                "corner_count": int(np.asarray(det.ids).reshape(-1).size),
                "corner_total": int(det.quality.get("corner_total", 0) or 0),
                "corner_coverage": det.quality.get("corner_coverage"),
                "board_reprojection_rmse_px": det.reprojection_rmse_px,
                "quality": det.quality,
                "T_Cg_P": camera_T_board,
                "T_R_P_estimate": robot_T_board,
            }
        )
        print(
            f"[board-anchor] frame {idx:04d}: selected_candidate={accepted_by_board} "
            f"corners={records[-1]['corner_count']} rmse={det.reprojection_rmse_px} "
            f"reason={records[-1]['rejected_reason']}"
        )

    selected_indices = _select_frames(records, estimates, args)
    min_valid = max(1, int(args.min_valid_frames))
    if len(selected_indices) < min_valid:
        observations = {
            "frames": records,
            "detections": detections,
            "board_config": vars(board_cfg),
            "sources": sources,
        }
        save_json(run_dir / "observations.json", observations)
        np.save(run_dir / "camera_T_board_per_frame.npy", camera_T_board_per_frame)
        np.save(run_dir / "R_T_board_per_frame.npy", robot_T_board_per_frame)
        raise RuntimeError(f"Only {len(selected_indices)} selected frames. Need at least {min_valid}.")

    selected_estimates = [as_transform(estimates[idx]) for idx in selected_indices if estimates[idx] is not None]
    anchor = average_transforms(selected_estimates)
    save_matrix_pair(run_dir / "R_T_board.npy", anchor)
    save_matrix_pair(run_dir / "board_T_R.npy", invert_transform(anchor))
    save_matrix_pair(run_dir / "base_T_board.npy", anchor)
    save_matrix_pair(run_dir / "board_T_base.npy", invert_transform(anchor))
    np.save(run_dir / "camera_T_board_per_frame.npy", camera_T_board_per_frame)
    np.save(run_dir / "R_T_board_per_frame.npy", robot_T_board_per_frame)

    selected_errors = []
    selected_rmses = []
    for idx, record in enumerate(records):
        if estimates[idx] is not None:
            err = pose_error(as_transform(estimates[idx]), anchor)
            record["anchor_error"] = err
            if record.get("selected"):
                selected_errors.append(err)
        if record.get("selected") and record.get("board_reprojection_rmse_px") is not None:
            selected_rmses.append(float(record["board_reprojection_rmse_px"]))
    translation_errors = [float(item["translation_m"]) for item in selected_errors]
    rotation_errors = [float(item["rotation_deg"]) for item in selected_errors]
    rejected_frames = [record for record in records if not record.get("selected")]

    metadata = {
        "coordinate_convention": "T_A_B maps coordinates from frame B into frame A",
        "T_R_P": "R_T_board.npy",
        "T_R_Cg": "R_T_global_camera.npy",
        "T_Cg_P_per_frame": "camera_T_board_per_frame.npy",
        "selected_frame_count": len(selected_indices),
        "board_reprojection_rmse_px_mean": _mean_or_none(selected_rmses),
        "aliases": {
            "R_T_board": "T_R_P",
            "base_T_board": "T_R_P",
            "board_T_R": "inverse(T_R_P)",
            "R_T_global_camera": "T_R_Cg",
            "base_T_camera": "T_R_Cg",
            "camera_T_board_per_frame": "T_Cg_P_per_frame",
        },
    }
    report = {
        "run_dir": run_dir,
        "status": "pass",
        "frame_count": len(frames),
        "selected_frame_count": len(selected_indices),
        "rejected_frame_count": len(rejected_frames),
        "selected_frame_indices": selected_indices,
        "board_reprojection_rmse_px_mean": _mean_or_none(selected_rmses),
        "board_reprojection_rmse_px_max": _max_or_none(selected_rmses),
        "anchor_translation_spread_m_mean": _mean_or_none(translation_errors),
        "anchor_translation_spread_m_max": _max_or_none(translation_errors),
        "anchor_rotation_spread_deg_mean": _mean_or_none(rotation_errors),
        "anchor_rotation_spread_deg_max": _max_or_none(rotation_errors),
        "metadata": metadata,
        "sources": sources,
        "board_config": vars(board_cfg),
        "T_R_Cg": base_T_camera,
        "T_R_P": anchor,
        "R_T_board": anchor,
        "board_T_R": invert_transform(anchor),
        "frames": records,
        "rejected_frames": [
            {
                "index": int(record["index"]),
                "corner_count": int(record.get("corner_count", 0)),
                "board_reprojection_rmse_px": record.get("board_reprojection_rmse_px"),
                "reason": record.get("rejected_reason") or "",
            }
            for record in rejected_frames
        ],
    }
    report["outliers"] = report["rejected_frames"]
    report = normalize_calibration_report(
        report,
        run_kind="global_board_anchor",
        config_path=args.config,
        frames_total=len(frames),
        frames_used=len(selected_indices),
        accepted=True,
        transforms={
            "T_R_Cg": base_T_camera,
            "T_R_P": anchor,
        },
        metrics={
            "board_reprojection_rmse_px_mean": report["board_reprojection_rmse_px_mean"],
            "board_reprojection_rmse_px_max": report["board_reprojection_rmse_px_max"],
            "anchor_translation_spread_m_mean": report["anchor_translation_spread_m_mean"],
            "anchor_translation_spread_m_max": report["anchor_translation_spread_m_max"],
            "anchor_rotation_spread_deg_mean": report["anchor_rotation_spread_deg_mean"],
            "anchor_rotation_spread_deg_max": report["anchor_rotation_spread_deg_max"],
        },
    )
    observations = {
        "frames": records,
        "detections": detections,
        "board_config": vars(board_cfg),
        "sources": sources,
        "metadata": metadata,
    }
    save_json(run_dir / "observations.json", observations)
    save_json(run_dir / "global_board_anchor_report.json", report)
    if not args.no_html_report:
        _write_anchor_html_report(run_dir / "report.html", summary=report, frame_records=records)
    return report


def main() -> int:
    args = parse_args()
    settings = load_realman_settings(args.config)
    board_cfg = _board_config(args, settings)
    run_dir = calibration_run_dir("global_board_anchor", args.output_root)
    base_T_camera, base_camera_source = _load_base_camera_transform(args)
    sources = {"base_camera_transform": str(base_camera_source)}
    if args.input_run is not None:
        frames, frame_sources = _load_input_frames(args)
        sources.update(frame_sources)
    else:
        if not args.manual_capture:
            raise RuntimeError("Use --input-run for offline mode or --manual-capture for live global-camera capture.")
        frames, frame_sources = _collect_manual_frames(args, settings, board_cfg)
        sources.update(frame_sources)
    report = _process_frames(
        frames=frames,
        board_cfg=board_cfg,
        base_T_camera=base_T_camera,
        run_dir=run_dir,
        args=args,
        sources=sources,
    )
    print(f"[board-anchor] selected frames: {report['selected_frame_count']}/{report['frame_count']}")
    print(f"[board-anchor] calibration written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
