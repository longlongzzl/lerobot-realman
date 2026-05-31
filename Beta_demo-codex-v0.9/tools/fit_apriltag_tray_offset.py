#!/usr/bin/env python3
"""Project/check Jimu tray AprilTag pose from one saved RGB frame.

This is a diagnostic helper. It does not change the localization result. Given
an AprilTag anchor run directory or full_scene_pose_results.json, it projects
the tray CAD outline into the RGB image.  Color-contour fitting is optional and
disabled by default because the white tray, white table, reflections, and
occluding blocks make a thresholded contour unreliable.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import trimesh


DEFAULT_TRAY_MESH = (
    Path(__file__).resolve().parents[1]
    / "jimu_portable_repro"
    / "assets"
    / "jimu_liaoban_new"
    / "jimu_liaoban_new.obj"
)


def _load_result(path: Path) -> tuple[Path, dict]:
    path = path.expanduser()
    if path.is_dir():
        path = path / "full_scene_pose_results.json"
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return path, payload


def _shared_frame_dir(result_path: Path, payload: dict) -> Path:
    info = payload.get("jimu_apriltag_anchor_localization") or {}
    frame_dir = info.get("frame_dir")
    candidates = []
    if frame_dir:
        candidates.append(Path(frame_dir))
    candidates.extend([result_path.parent / "shared_frame", result_path.parent.parent / "shared_frame"])
    for candidate in candidates:
        candidate = candidate.expanduser()
        if (candidate / "rgb.png").exists() and (candidate / "camera.json").exists():
            return candidate
    raise FileNotFoundError(f"could not find shared_frame near {result_path}")


def _tray_result(payload: dict) -> dict:
    for item in payload.get("results") or []:
        if item.get("object_name") == "jimu_liaoban" and item.get("jimu_apriltag_anchor"):
            return item
    raise ValueError("result does not contain a jimu_liaoban AprilTag anchor")


def _tray_outline_points(mesh_path: Path, scale: float, samples_per_edge: int, z_mode: str = "top") -> np.ndarray:
    mesh = trimesh.load(str(mesh_path.expanduser()), force="scene")
    bounds = np.asarray(mesh.bounds, dtype=np.float32) * float(scale)
    min_v, max_v = bounds
    if z_mode == "bottom":
        z = float(min_v[2])
    elif z_mode == "tag":
        z = 0.005
    elif z_mode == "mid":
        z = float(0.5 * (min_v[2] + max_v[2]))
    else:
        z = float(max_v[2])
    pts: list[list[float]] = []
    for t in np.linspace(0.0, 1.0, samples_per_edge):
        pts.append([min_v[0] * (1 - t) + max_v[0] * t, min_v[1], z])
    for t in np.linspace(0.0, 1.0, samples_per_edge):
        pts.append([max_v[0], min_v[1] * (1 - t) + max_v[1] * t, z])
    for t in np.linspace(0.0, 1.0, samples_per_edge):
        pts.append([max_v[0] * (1 - t) + min_v[0] * t, max_v[1], z])
    for t in np.linspace(0.0, 1.0, samples_per_edge):
        pts.append([min_v[0], max_v[1] * (1 - t) + min_v[1] * t, z])
    return np.asarray(pts, dtype=np.float32)


def _mask_contours(mask: np.ndarray, roi: tuple[int, int, int, int], morph_kernel: int = 7) -> list[np.ndarray]:
    x1, y1, x2, y2 = roi
    roi_mask = np.zeros(mask.shape, np.uint8)
    roi_mask[y1:y2, x1:x2] = 255
    mask = cv2.bitwise_and(mask, roi_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((morph_kernel, morph_kernel), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if cv2.contourArea(contour) >= 500.0]


def _green_tray_contours(bgr: np.ndarray, roi: tuple[int, int, int, int]) -> list[np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.asarray([35, 25, 25], np.uint8), np.asarray([95, 255, 210], np.uint8))
    return _mask_contours(mask, roi, morph_kernel=7)


def _white_tray_contours(bgr: np.ndarray, roi: tuple[int, int, int, int]) -> list[np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # The new tray is a low-saturation white plastic body. Keep this broad;
    # we select the final component by requiring that it contains the tag.
    mask = cv2.inRange(hsv, np.asarray([0, 0, 125], np.uint8), np.asarray([180, 70, 255], np.uint8))
    return _mask_contours(mask, roi, morph_kernel=5)


def _tray_contour_auto(bgr: np.ndarray, roi: tuple[int, int, int, int], tag_center_px: np.ndarray) -> tuple[np.ndarray | None, str]:
    candidates: list[tuple[float, float, str, np.ndarray]] = []
    for label, contours in (
        ("green", _green_tray_contours(bgr, roi)),
        ("white", _white_tray_contours(bgr, roi)),
    ):
        for contour in contours:
            area = float(cv2.contourArea(contour))
            dist = float(cv2.pointPolygonTest(contour, (float(tag_center_px[0]), float(tag_center_px[1])), True))
            if dist >= -8.0:
                candidates.append((dist, area, label, contour))
    if not candidates:
        return None, "no green/white tray contour contains the detected tray tag; check visibility or ROI"
    # Prefer a contour that clearly contains the tag, then the largest connected body.
    dist, area, label, contour = max(candidates, key=lambda item: (item[0] >= 0.0, item[1]))
    return contour, label


def _tag_object_points(center: np.ndarray, size_m: float, yaw_deg: float) -> np.ndarray:
    half = 0.5 * float(size_m)
    theta = math.radians(float(yaw_deg))
    c = math.cos(theta)
    s = math.sin(theta)
    plane_u = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    plane_v = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    marker_right = c * plane_u + s * plane_v
    marker_up = -s * plane_u + c * plane_v
    return np.asarray(
        [
            center - half * marker_right + half * marker_up,
            center + half * marker_right + half * marker_up,
            center + half * marker_right - half * marker_up,
            center - half * marker_right - half * marker_up,
        ],
        dtype=np.float32,
    )


def _solve_pose(K: np.ndarray, image_points: np.ndarray, object_points: np.ndarray) -> tuple[np.ndarray, float]:
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.asarray(R, dtype=np.float32)
    T[:3, 3] = np.asarray(tvec, dtype=np.float32).reshape(3)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, None)
    reproj = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)))
    return T, reproj


def _project(K: np.ndarray, T: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pc = (T[:3, :3] @ points.T + T[:3, 3:4]).T
    uv = (K @ pc.T).T
    return uv[:, :2] / uv[:, 2:3], pc[:, 2]


def _score_outline(K: np.ndarray, T: np.ndarray, outline: np.ndarray, contour: np.ndarray, image_shape: tuple[int, int, int]) -> tuple[float, float, int]:
    uv, depth = _project(K, T, outline)
    h, w = image_shape[:2]
    visible = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[visible]
    if len(uv) < 20:
        return float("inf"), float("inf"), int(len(uv))
    distances = [
        abs(cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True))
        for point in uv[::3]
    ]
    return float(np.median(distances)), float(np.mean(distances)), int(len(uv))


def _draw_outline(canvas: np.ndarray, K: np.ndarray, T: np.ndarray, outline: np.ndarray, color: tuple[int, int, int], label: str, label_index: int) -> None:
    uv, depth = _project(K, T, outline)
    h, w = canvas.shape[:2]
    visible = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    points = np.round(uv[visible]).astype(np.int32)
    for p, q in zip(points, np.roll(points, -1, axis=0)):
        if np.linalg.norm(p - q) < 25:
            cv2.line(canvas, tuple(p), tuple(q), color, 2, cv2.LINE_AA)
    cv2.putText(canvas, label, (12, 28 + 24 * label_index), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _parse_roi(raw: str, shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    if not raw:
        h, w = shape[:2]
        return 0, int(h * 0.60), int(w * 0.32), h
    parts = [int(v) for v in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be x1,y1,x2,y2")
    return tuple(parts)  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path, help="AprilTag run directory or full_scene_pose_results.json")
    parser.add_argument("--tray-mesh", type=Path, default=DEFAULT_TRAY_MESH)
    parser.add_argument("--tray-scale", type=float, default=0.01)
    parser.add_argument(
        "--outline-z-mode",
        choices=["bottom", "tag", "mid", "top"],
        default="top",
        help="CAD bounds height used for scoring. The overlay always draws bottom/tag/top outlines.",
    )
    parser.add_argument("--roi", default="", help="Green tray ROI as x1,y1,x2,y2; default uses lower-left image")
    parser.add_argument("--yaw-min", type=int, default=-180)
    parser.add_argument("--yaw-max", type=int, default=180)
    parser.add_argument("--yaw-step", type=int, default=5)
    parser.add_argument("--offset-mm", type=int, default=10, help="Sweep +/- this many mm in tray local X/Y")
    parser.add_argument("--offset-step-mm", type=int, default=1)
    parser.add_argument(
        "--use-color-contour",
        action="store_true",
        help="Enable rough green/white color contour scoring. Off by default because it is not a precise tray edge.",
    )
    parser.add_argument(
        "--depth-tolerance-m",
        type=float,
        default=0.005,
        help="Reject CAD-outline fits whose PnP depth changes too much from the saved AprilTag pose.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    result_path, payload = _load_result(args.result)
    frame_dir = _shared_frame_dir(result_path, payload)
    bgr = cv2.imread(str(frame_dir / "rgb.png"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(frame_dir / "rgb.png")
    K = np.asarray(json.load(open(frame_dir / "camera.json", "r", encoding="utf-8"))["cam_K"], dtype=np.float32).reshape(3, 3)
    item = _tray_result(payload)
    anchor = item["jimu_apriltag_anchor"]
    image_points = np.asarray(anchor["image_corners_px"], dtype=np.float32).reshape(4, 2)
    base_center = np.asarray(
        anchor.get("object_tag_center_m") or np.mean(np.asarray(anchor["object_corners_m"], dtype=np.float32), axis=0),
        dtype=np.float32,
    ).reshape(3)
    tag_size = float(anchor["tag_size_m"])
    current_z = float(item["translation_m"][2])
    current_yaw = float(anchor.get("tag_yaw_deg", 90.0))
    current_offset = np.asarray(anchor.get("tray_center_offset_xy_m", [0.0, 0.0]), dtype=np.float32).reshape(2)

    outline = _tray_outline_points(args.tray_mesh, args.tray_scale, samples_per_edge=140, z_mode=args.outline_z_mode)
    outline_bottom = _tray_outline_points(args.tray_mesh, args.tray_scale, samples_per_edge=140, z_mode="bottom")
    outline_tag = _tray_outline_points(args.tray_mesh, args.tray_scale, samples_per_edge=140, z_mode="tag")
    outline_top = _tray_outline_points(args.tray_mesh, args.tray_scale, samples_per_edge=140, z_mode="top")
    roi = _parse_roi(args.roi, bgr.shape)
    tag_center_px = np.mean(image_points, axis=0)
    contour = None
    contour_label = None
    contour_warning = "color contour scoring disabled"
    if args.use_color_contour:
        contour, contour_label = _tray_contour_auto(bgr, roi, tag_center_px)
        contour_warning = "" if contour is not None else str(contour_label)

    rows: list[dict] = []
    all_best = None
    physical_best = None
    if contour is not None:
        for yaw in range(int(args.yaw_min), int(args.yaw_max) + 1, int(args.yaw_step)):
            for dx_mm in range(-int(args.offset_mm), int(args.offset_mm) + 1, int(args.offset_step_mm)):
                for dy_mm in range(-int(args.offset_mm), int(args.offset_mm) + 1, int(args.offset_step_mm)):
                    center = base_center.copy()
                    center[0] += float(dx_mm) / 1000.0
                    center[1] += float(dy_mm) / 1000.0
                    T, reproj = _solve_pose(K, image_points, _tag_object_points(center, tag_size, float(yaw)))
                    median_px, mean_px, visible = _score_outline(K, T, outline, contour, bgr.shape)
                    rows.append(
                        {
                            "yaw_deg": int(yaw),
                            "dx_mm": int(dx_mm),
                            "dy_mm": int(dy_mm),
                            "median_px": median_px,
                            "mean_px": mean_px,
                            "visible_samples": visible,
                            "reprojection_px": reproj,
                            "z_cam": float(T[2, 3]),
                            "z_err_m": abs(float(T[2, 3]) - current_z),
                        }
                    )

        all_best = min(rows, key=lambda row: (row["median_px"], row["mean_px"]))
        physical = [row for row in rows if row["z_err_m"] <= float(args.depth_tolerance_m)]
        if not physical:
            raise RuntimeError(f"no candidate passed depth tolerance {args.depth_tolerance_m}m")
        physical_best = min(physical, key=lambda row: (row["median_px"], row["mean_px"]))

    out_dir = args.out_dir or (result_path.parent / "tray_offset_fit")
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "tray_offset_fit_scores.json"
    overlay_path = out_dir / "tray_offset_fit_overlay.png"

    def pose_for(row: dict) -> np.ndarray:
        center = base_center.copy()
        center[0] += float(row["dx_mm"]) / 1000.0
        center[1] += float(row["dy_mm"]) / 1000.0
        return _solve_pose(K, image_points, _tag_object_points(center, tag_size, float(row["yaw_deg"])))[0]

    current_row = {
        "yaw_deg": int(round(current_yaw)),
        "dx_mm": 0,
        "dy_mm": 0,
    }
    current_T = pose_for(current_row)
    if contour is None:
        _, depth = _project(K, current_T, outline)
        current_median, current_mean, current_visible = None, None, int(np.sum(depth > 0))
    else:
        current_median, current_mean, current_visible = _score_outline(K, current_T, outline, contour, bgr.shape)
    current_summary = {
        **current_row,
        "median_px": current_median,
        "mean_px": current_mean,
        "visible_samples": current_visible,
        "z_cam": float(current_T[2, 3]),
        "z_err_m": abs(float(current_T[2, 3]) - current_z),
    }
    if physical_best is not None:
        physical_best_total_offset = {
            "x_m": float(current_offset[0]) + float(physical_best["dx_mm"]) / 1000.0,
            "y_m": float(current_offset[1]) + float(physical_best["dy_mm"]) / 1000.0,
            "x_mm": float(current_offset[0]) * 1000.0 + float(physical_best["dx_mm"]),
            "y_mm": float(current_offset[1]) * 1000.0 + float(physical_best["dy_mm"]),
        }
    else:
        physical_best_total_offset = None
    canvas = bgr.copy()
    if contour is not None:
        cv2.drawContours(canvas, [contour], -1, (255, 0, 255), 2, cv2.LINE_AA)
    current_pose = pose_for(current_row)
    _draw_outline(canvas, K, current_pose, outline_bottom, (255, 80, 80), "bottom bounds", 0)
    _draw_outline(canvas, K, current_pose, outline_tag, (0, 255, 255), "tag-plane bounds", 1)
    _draw_outline(canvas, K, current_pose, outline_top, (0, 255, 0), "top bounds", 2)
    if physical_best is not None:
        _draw_outline(canvas, K, pose_for(physical_best), outline, (0, 180, 255), "physical-best CAD outline", 3)
    elif contour_warning:
        cv2.putText(canvas, "no reliable color contour", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2, cv2.LINE_AA)
    for point in image_points:
        cv2.circle(canvas, tuple(np.round(point).astype(int)), 4, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.imwrite(str(overlay_path), canvas)

    payload_out = {
        "source_result": str(result_path),
        "frame_dir": str(frame_dir),
        "base_center_m": base_center.astype(float).tolist(),
        "current": current_summary,
        "current_total_tray_center_offset": {
            "x_m": float(current_offset[0]),
            "y_m": float(current_offset[1]),
            "x_mm": float(current_offset[0]) * 1000.0,
            "y_mm": float(current_offset[1]) * 1000.0,
        },
        "all_best": all_best,
        "depth_tolerance_m": float(args.depth_tolerance_m),
        "contour_warning": contour_warning,
        "contour_label": contour_label if contour is not None else None,
        "physical_best": physical_best,
        "physical_best_total_tray_center_offset": physical_best_total_offset,
        "physical_top20": sorted(physical, key=lambda row: (row["median_px"], row["mean_px"]))[:20] if contour is not None else [],
        "overlay": str(overlay_path),
    }
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(payload_out, f, indent=2)
    print(
        json.dumps(
            {
                "scores": str(scores_path),
                "overlay": str(overlay_path),
                "current": current_summary,
                "physical_best": physical_best,
                "physical_best_total_tray_center_offset": physical_best_total_offset,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
