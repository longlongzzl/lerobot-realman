#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from rm75_app.assets.object_specs import get_object_spec, normalize_object_name, resolve_object_spec_scales
from rm75_app.paths import RUNTIME_DIR
from rm75_app.perception import sam6d_pose_provider as sam6d_provider


DEFAULT_OUTPUT_ROOT = RUNTIME_DIR / "tabletop_pose_refine_runs"


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).expanduser().read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _matrix4(value, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {arr.shape}")
    return arr


def _load_matrix4x4(path: Path) -> np.ndarray:
    path = Path(path).expanduser()
    if path.suffix == ".npy":
        mat = np.load(path)
    else:
        mat = np.loadtxt(path)
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"expected 4x4 matrix from {path}, got {mat.shape}")
    return mat


def _effective_T_base_cam(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    raw = _load_matrix4x4(Path(args.camera_extrinsic_opencv_path))
    if bool(args.use_direct_camera_extrinsic):
        return raw, "direct: T_base_obj = T_base_cam @ T_cam_obj"
    return np.linalg.inv(raw), "inverse: T_base_obj = inv(camera_extrinsic_opencv) @ T_cam_obj"


def _find_latest_sam6d_result(root: Path) -> Path:
    root = Path(root).expanduser()
    candidates = sorted(root.glob("**/full_scene_pose_results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no full_scene_pose_results.json found under {root}")
    return candidates[0]


def _scene_dir_from_summary(path: Path, summary: dict) -> Path:
    raw = summary.get("scene_dir")
    if raw:
        return Path(raw).expanduser()
    return Path(path).expanduser().resolve().parent


def _load_frame(frame_dir: Path) -> dict:
    frame_dir = Path(frame_dir).expanduser()
    rgb_path = frame_dir / "rgb.png"
    depth_path = frame_dir / "depth.png"
    camera_path = frame_dir / "camera.json"
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(rgb_path)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(depth_path)
    cam_info = _load_json(camera_path)
    K = np.asarray(cam_info["cam_K"], dtype=np.float64).reshape(3, 3)
    depth_scale = float(cam_info.get("depth_scale", 1.0))
    depth_m = np.asarray(depth_raw, dtype=np.float32) * depth_scale / 1000.0
    return {
        "bgr": bgr,
        "rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        "depth_m": depth_m,
        "K": K,
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "camera_path": str(camera_path),
    }


def _resolve_frame_dir(args: argparse.Namespace, summary_path: Path, summary: dict) -> Path:
    if args.frame_dir is not None:
        return Path(args.frame_dir).expanduser()
    scene_dir = _scene_dir_from_summary(summary_path, summary)
    frame_dir = scene_dir / "shared_frame"
    if not frame_dir.exists():
        raise FileNotFoundError(f"cannot infer shared frame dir: {frame_dir}; pass --frame-dir")
    return frame_dir


def _object_names_filter(args: argparse.Namespace, results: list[dict]) -> set[str] | None:
    if not args.object_names:
        return None
    return {normalize_object_name(name) or str(name) for name in args.object_names}


def _skip_names(args: argparse.Namespace) -> set[str]:
    return {normalize_object_name(name) or str(name) for name in list(args.skip_object_names or [])}


def _load_mask_for_item(item: dict) -> np.ndarray:
    run_dir = Path(item.get("run_dir") or "").expanduser()
    candidates = [
        Path(item["mask_path"]).expanduser() if item.get("mask_path") else None,
        run_dir / "sam6d_binary_mask.png",
        run_dir / "sam3_mask" / "sam3_mask.png",
    ]
    for path in candidates:
        if path is None:
            continue
        mask_u8 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask_u8 is not None:
            return np.asarray(mask_u8 > 0, dtype=bool)
    raise FileNotFoundError(f"no mask image found for {item.get('object_name')} under {run_dir}")


def _mesh_for_object(object_name: str):
    spec = get_object_spec(object_name)
    if spec is None:
        raise KeyError(f"unknown object spec: {object_name}")
    mesh_scale, _ = resolve_object_spec_scales(spec)
    mesh = sam6d_provider._load_scene_mesh(spec.mesh_file, mesh_scale)
    return mesh, spec.mesh_file, float(mesh_scale)


def _rotz(theta: float) -> np.ndarray:
    c = math.cos(float(theta))
    s = math.sin(float(theta))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    return (np.asarray(T, dtype=np.float64).reshape(4, 4) @ np.concatenate([pts, ones], axis=1).T).T[:, :3]


def _project_points(K: np.ndarray, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts_cam = np.asarray(pts_cam, dtype=np.float64).reshape(-1, 3)
    z = pts_cam[:, 2]
    valid = z > 1e-6
    uv = np.zeros((pts_cam.shape[0], 2), dtype=np.float64)
    if np.any(valid):
        proj = (np.asarray(K, dtype=np.float64).reshape(3, 3) @ pts_cam[valid].T).T
        uv[valid, 0] = proj[:, 0] / proj[:, 2]
        uv[valid, 1] = proj[:, 1] / proj[:, 2]
    return uv, valid


def _mask_bbox_center(mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    ys, xs = np.where(np.asarray(mask > 0, dtype=bool))
    if xs.size == 0 or ys.size == 0:
        return None, None
    bbox = np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)
    center = np.asarray([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float64)
    return bbox, center


def _bbox_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = np.asarray(a, dtype=np.float64).reshape(-1)[:4]
    bx1, by1, bx2, by2 = np.asarray(b, dtype=np.float64).reshape(-1)[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / max(area_a + area_b - inter, 1e-6))


def _render_depth_mask(K: np.ndarray, T_cam_obj: np.ndarray, pts_obj: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = int(shape[0]), int(shape[1])
    pts_cam = _transform_points(T_cam_obj, pts_obj)
    uv, valid = _project_points(K, pts_cam)
    ix = np.rint(uv[:, 0]).astype(np.int32)
    iy = np.rint(uv[:, 1]).astype(np.int32)
    valid &= (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    if np.any(valid):
        flat_idx = iy[valid] * width + ix[valid]
        np.minimum.at(depth.reshape(-1), flat_idx, pts_cam[:, 2][valid].astype(np.float32))
    render_mask = np.isfinite(depth)
    if np.count_nonzero(render_mask) > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        render_mask = cv2.dilate(render_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return render_mask, depth


def _metrics_for_pose(
    *,
    K: np.ndarray,
    depth_m: np.ndarray,
    obs_mask: np.ndarray,
    T_cam_obj: np.ndarray,
    pts_obj: np.ndarray,
) -> dict:
    obs_mask = np.asarray(obs_mask > 0, dtype=bool)
    obs_bbox, obs_center = _mask_bbox_center(obs_mask)
    if obs_bbox is None or obs_center is None:
        return {"ok": False, "reason": "empty_observed_mask"}
    render_mask, render_depth = _render_depth_mask(K, T_cam_obj, pts_obj, obs_mask.shape[:2])
    ren_bbox, ren_center = _mask_bbox_center(render_mask)
    if ren_bbox is None or ren_center is None:
        return {"ok": False, "reason": "empty_render_mask"}
    inter = np.count_nonzero(render_mask & obs_mask)
    union = np.count_nonzero(render_mask | obs_mask)
    iou = float(inter / max(union, 1))
    center_error_px = float(np.linalg.norm(ren_center - obs_center))
    bbox_iou = _bbox_iou(ren_bbox, obs_bbox)
    real_depth = np.asarray(depth_m, dtype=np.float32)
    both = render_mask & obs_mask & np.isfinite(render_depth) & np.isfinite(real_depth) & (real_depth > 0.05) & (real_depth < 2.0)
    if np.count_nonzero(both) >= 12:
        residual = np.abs(render_depth[both].astype(np.float32) - real_depth[both].astype(np.float32))
        depth_median = float(np.median(residual))
        depth_p80 = float(np.percentile(residual, 80))
        depth_count = int(residual.size)
    else:
        depth_median = None
        depth_p80 = None
        depth_count = int(np.count_nonzero(both))
    return {
        "ok": True,
        "iou": iou,
        "bbox_iou": bbox_iou,
        "center_error_px": center_error_px,
        "depth_median_abs_m": depth_median,
        "depth_p80_abs_m": depth_p80,
        "depth_overlap_pixels": depth_count,
        "obs_pixels": int(np.count_nonzero(obs_mask)),
        "render_pixels": int(np.count_nonzero(render_mask)),
        "intersection_pixels": int(inter),
        "obs_bbox": obs_bbox.tolist(),
        "render_bbox": ren_bbox.tolist(),
        "obs_center": obs_center.tolist(),
        "render_center": ren_center.tolist(),
    }


def _score(metrics: dict, dx_m: float, dy_m: float, dyaw_rad: float) -> float:
    if not metrics.get("ok"):
        return -1e9
    iou = float(metrics.get("iou", 0.0))
    bbox_iou = float(metrics.get("bbox_iou", 0.0))
    center = math.exp(-((float(metrics.get("center_error_px", 1e6)) / 35.0) ** 2))
    depth = metrics.get("depth_median_abs_m")
    depth_score = 0.0 if depth is None else math.exp(-((float(depth) / 0.025) ** 2))
    prior = math.sqrt((float(dx_m) / 0.04) ** 2 + (float(dy_m) / 0.04) ** 2 + (float(dyaw_rad) / math.radians(35.0)) ** 2)
    return float(1.8 * iou + 0.6 * bbox_iou + 0.5 * center + 0.45 * depth_score - 0.08 * prior)


def _candidate_base_pose(T_base_init: np.ndarray, dx: float, dy: float, dyaw: float) -> np.ndarray:
    T = np.asarray(T_base_init, dtype=np.float64).reshape(4, 4).copy()
    T[:3, 3] = T[:3, 3] + np.asarray([float(dx), float(dy), 0.0], dtype=np.float64)
    T[:3, :3] = _rotz(float(dyaw)) @ T[:3, :3]
    return T


def _yaw_delta_deg(T_base_init: np.ndarray, T_base_new: np.ndarray) -> float:
    R_delta = np.asarray(T_base_new, dtype=np.float64)[:3, :3] @ np.asarray(T_base_init, dtype=np.float64)[:3, :3].T
    return float(math.degrees(math.atan2(R_delta[1, 0], R_delta[0, 0])))


def _refine_one(
    *,
    args: argparse.Namespace,
    item: dict,
    frame: dict,
    T_base_cam_eff: np.ndarray,
    run_dir: Path,
) -> tuple[dict, dict]:
    object_name = normalize_object_name(item.get("object_name")) or str(item.get("object_name"))
    mask = _load_mask_for_item(item)
    mesh, mesh_file, mesh_scale = _mesh_for_object(object_name)
    pts_obj = sam6d_provider._mesh_refine_points(mesh, int(args.sample_points))
    T_cam_init = _matrix4(item["T_cam_obj"], name=f"{object_name}.T_cam_obj")
    T_base_init = np.asarray(T_base_cam_eff, dtype=np.float64).reshape(4, 4) @ T_cam_init
    T_cam_base_eff = np.linalg.inv(T_base_cam_eff)
    K = np.asarray(frame["K"], dtype=np.float64).reshape(3, 3)
    depth_m = np.asarray(frame["depth_m"], dtype=np.float32)

    before = _metrics_for_pose(K=K, depth_m=depth_m, obs_mask=mask, T_cam_obj=T_cam_init, pts_obj=pts_obj)
    before_score = _score(before, 0.0, 0.0, 0.0)
    best = {
        "score": before_score,
        "metrics": before,
        "T_base_obj": T_base_init,
        "T_cam_obj": T_cam_init,
        "dx_m": 0.0,
        "dy_m": 0.0,
        "dyaw_rad": 0.0,
    }

    xy_radius = float(args.xy_search_radius_m)
    xy_step = float(args.xy_search_step_m)
    yaw_radius = math.radians(float(args.yaw_search_radius_deg))
    yaw_step = math.radians(float(args.yaw_search_step_deg))

    def eval_candidate(dx: float, dy: float, dyaw: float) -> None:
        nonlocal best
        T_base = _candidate_base_pose(T_base_init, dx, dy, dyaw)
        T_cam = T_cam_base_eff @ T_base
        metrics = _metrics_for_pose(K=K, depth_m=depth_m, obs_mask=mask, T_cam_obj=T_cam, pts_obj=pts_obj)
        score = _score(metrics, dx, dy, dyaw)
        if score > float(best["score"]):
            best = {
                "score": float(score),
                "metrics": metrics,
                "T_base_obj": T_base,
                "T_cam_obj": T_cam,
                "dx_m": float(dx),
                "dy_m": float(dy),
                "dyaw_rad": float(dyaw),
            }

    xs = np.arange(-xy_radius, xy_radius + 0.5 * xy_step, xy_step, dtype=np.float64)
    yaws = np.arange(-yaw_radius, yaw_radius + 0.5 * yaw_step, yaw_step, dtype=np.float64)
    evaluated = 1
    for dx in xs:
        for dy in xs:
            for dyaw in yaws:
                eval_candidate(float(dx), float(dy), float(dyaw))
                evaluated += 1

    center_dx = float(best["dx_m"])
    center_dy = float(best["dy_m"])
    center_yaw = float(best["dyaw_rad"])
    local_xy_step = xy_step * 0.5
    local_yaw_step = yaw_step * 0.5
    for _level in range(max(0, int(args.refine_levels))):
        for ox in (-local_xy_step, 0.0, local_xy_step):
            for oy in (-local_xy_step, 0.0, local_xy_step):
                for oyaw in (-local_yaw_step, 0.0, local_yaw_step):
                    eval_candidate(center_dx + ox, center_dy + oy, center_yaw + oyaw)
                    evaluated += 1
        center_dx = float(best["dx_m"])
        center_dy = float(best["dy_m"])
        center_yaw = float(best["dyaw_rad"])
        local_xy_step *= 0.5
        local_yaw_step *= 0.5

    improvement = float(best["score"] - before_score)
    accepted = bool(
        improvement >= float(args.min_score_improvement)
        and float(best["metrics"].get("iou", 0.0)) >= float(args.min_iou)
    )
    refined = dict(item)
    refine_info = {
        "enabled": True,
        "accepted": accepted,
        "object_name": object_name,
        "mode": "tabletop_xy_yaw",
        "evaluated_candidates": int(evaluated),
        "mesh_file": mesh_file,
        "mesh_scale": mesh_scale,
        "sample_points": int(len(pts_obj)),
        "before_score": float(before_score),
        "after_score": float(best["score"]),
        "score_improvement": improvement,
        "before_metrics": before,
        "after_metrics": best["metrics"],
        "delta_base_xy_m": [float(best["dx_m"]), float(best["dy_m"])],
        "delta_yaw_deg": float(math.degrees(float(best["dyaw_rad"]))),
        "base_z_locked_m": float(T_base_init[2, 3]),
        "base_translation_before_m": T_base_init[:3, 3].tolist(),
        "base_translation_after_m": np.asarray(best["T_base_obj"])[:3, 3].tolist(),
    }
    if accepted:
        refined["T_cam_obj_before_tabletop_refine"] = item["T_cam_obj"]
        refined["translation_m_before_tabletop_refine"] = item.get("translation_m")
        refined["T_cam_obj"] = np.asarray(best["T_cam_obj"], dtype=np.float64).tolist()
        refined["translation_m"] = np.asarray(best["T_cam_obj"], dtype=np.float64)[:3, 3].tolist()
    else:
        refine_info["reject_reason"] = "score_or_iou_gate"
    refined["tabletop_refine"] = refine_info

    vis_path = run_dir / f"{object_name}_tabletop_refine_overlay.png"
    _save_refine_overlay(frame["bgr"], mask, before, best["metrics"], vis_path, object_name, accepted)
    refine_info["overlay_path"] = str(vis_path)
    return refined, refine_info


def _save_refine_overlay(bgr: np.ndarray, mask: np.ndarray, before: dict, after: dict, path: Path, object_name: str, accepted: bool) -> None:
    canvas = np.asarray(bgr, dtype=np.uint8).copy()
    overlay = canvas.copy()
    overlay[np.asarray(mask > 0, dtype=bool)] = (255, 220, 0)
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0.0)
    for metrics, color, label in ((before, (0, 0, 255), "raw"), (after, (0, 255, 0), "xy_yaw")):
        if not metrics.get("ok"):
            continue
        box = np.asarray(metrics["render_bbox"], dtype=np.float64)
        center = np.asarray(metrics["render_center"], dtype=np.float64)
        cv2.rectangle(canvas, tuple(np.round(box[:2]).astype(int)), tuple(np.round(box[2:]).astype(int)), color, 2)
        cv2.circle(canvas, tuple(np.round(center).astype(int)), 4, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, label, tuple(np.round(center + np.asarray([6.0, -6.0])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    text = f"{object_name} tabletop refine accepted={accepted}"
    cv2.putText(canvas, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine SAM6D full-scene tabletop object poses by optimizing x/y/yaw against SAM3 masks and RGB-D depth."
    )
    parser.add_argument("--sam6d-result", type=Path, default=None, help="Path to full_scene_pose_results.json. Defaults to newest SAM6D result.")
    parser.add_argument("--sam6d-output-root", type=Path, default=RUNTIME_DIR / "sam6d_grasp_scene_runs")
    parser.add_argument("--frame-dir", type=Path, default=None, help="Directory with rgb.png/depth.png/camera.json. Defaults to <scene_dir>/shared_frame.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--object-names", nargs="*", default=None, help="Optional subset of objects to refine.")
    parser.add_argument("--skip-object-names", nargs="*", default=[], help="Objects to copy through without tabletop refinement.")
    parser.add_argument("--camera-extrinsic-opencv-path", type=Path, default=Path(sam6d_provider.DEFAULT_CAMERA_EXTRINSIC_OPENCV_PATH))
    parser.add_argument("--use-direct-camera-extrinsic", action="store_true", default=False)
    parser.add_argument("--xy-search-radius-m", type=float, default=0.04)
    parser.add_argument("--xy-search-step-m", type=float, default=0.01)
    parser.add_argument("--yaw-search-radius-deg", type=float, default=30.0)
    parser.add_argument("--yaw-search-step-deg", type=float, default=10.0)
    parser.add_argument("--refine-levels", type=int, default=2)
    parser.add_argument("--sample-points", type=int, default=2200)
    parser.add_argument("--min-score-improvement", type=float, default=0.025)
    parser.add_argument("--min-iou", type=float, default=0.015)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = Path(args.sam6d_result).expanduser() if args.sam6d_result else _find_latest_sam6d_result(Path(args.sam6d_output_root))
    summary = _load_json(summary_path)
    frame_dir = _resolve_frame_dir(args, summary_path, summary)
    frame = _load_frame(frame_dir)
    T_base_cam_eff, extrinsic_mode = _effective_T_base_cam(args)
    run_dir = Path(args.output_root).expanduser() / f"{_now_stamp()}_tabletop_refine"
    run_dir.mkdir(parents=True, exist_ok=True)

    wanted = _object_names_filter(args, list(summary.get("results", [])))
    skipped = _skip_names(args)
    refined_results = []
    reports = []
    for item in list(summary.get("results", [])):
        name = normalize_object_name(item.get("object_name")) or str(item.get("object_name"))
        if wanted is not None and name not in wanted:
            refined_results.append(item)
            continue
        if name in skipped or not bool(item.get("ok", True)) or item.get("T_cam_obj") is None:
            copied = dict(item)
            copied["tabletop_refine"] = {"enabled": False, "reason": "skipped_or_missing_pose", "object_name": name}
            refined_results.append(copied)
            continue
        try:
            refined, report = _refine_one(args=args, item=item, frame=frame, T_base_cam_eff=T_base_cam_eff, run_dir=run_dir)
            refined_results.append(refined)
            reports.append(report)
            print(
                f"[tabletop-refine] {name} accepted={report['accepted']} "
                f"score {report['before_score']:.3f}->{report['after_score']:.3f} "
                f"dxy={np.round(report['delta_base_xy_m'], 4).tolist()} dyaw={report['delta_yaw_deg']:.1f}deg "
                f"iou {report['before_metrics'].get('iou', 0.0):.3f}->{report['after_metrics'].get('iou', 0.0):.3f}"
            )
        except Exception as exc:
            copied = dict(item)
            copied["tabletop_refine"] = {"enabled": True, "accepted": False, "object_name": name, "error": repr(exc)}
            refined_results.append(copied)
            reports.append(copied["tabletop_refine"])
            print(f"[tabletop-refine] {name} failed: {exc!r}")

    refined_summary = dict(summary)
    refined_summary["results"] = refined_results
    refined_summary["tabletop_refinement"] = {
        "source_sam6d_result": str(summary_path),
        "frame_dir": str(frame_dir),
        "run_dir": str(run_dir),
        "extrinsic_mode": extrinsic_mode,
        "accepted_count": int(sum(1 for r in reports if r.get("accepted"))),
        "attempted_count": int(len(reports)),
        "args": vars(args) | {"sam6d_result": str(summary_path), "output_root": str(args.output_root)},
    }
    out_path = run_dir / "tabletop_refined_scene_results.json"
    _write_json(out_path, refined_summary)
    print(f"[tabletop-refine] wrote {out_path}")
    print(f"[tabletop-refine] accepted={refined_summary['tabletop_refinement']['accepted_count']}/{refined_summary['tabletop_refinement']['attempted_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
