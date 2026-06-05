#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PICK_JIAOBANG_DIR = REPO_ROOT / "pick_jiaobang"
if str(PICK_JIAOBANG_DIR) not in sys.path:
    sys.path.insert(0, str(PICK_JIAOBANG_DIR))

import object_specs  # noqa: E402
import sam6d_groundingdino_pose_provider as provider  # noqa: E402


PORTABLE_REPRO_DIR = SCRIPT_DIR / "jimu_portable_repro"
TRAY_ONLY_MESH_FILE = PORTABLE_REPRO_DIR / "assets" / "jimu_liaoban_new" / "jimu_liaoban_new.obj"
TRAY_LOADED_MESH_FILE = PORTABLE_REPRO_DIR / "assets" / "jimu_liaoban_new" / "jimu_liaoban_new_loaded_14plates.glb"
BASE_APRILTAG_TOP_Y_M = 0.0075
TRAY_APRILTAG_TOP_Z_M = 0.005
TRAY_APRILTAG_CENTER_OFFSET_X_M = 0.0
TRAY_APRILTAG_CENTER_OFFSET_Y_M = 0.0
BASE_WORLD_OFFSET_X_M = 0.0
BASE_WORLD_OFFSET_Y_M = 0.0
TRAY_WORLD_OFFSET_X_M = 0.0
TRAY_WORLD_OFFSET_Y_M = 0.0
PLATE_MESH_FILE = PORTABLE_REPRO_DIR / "assets" / "red_jimu_plate_74x6p5x74.glb"
BASE_ASSEMBLY_MESH_FILE = PORTABLE_REPRO_DIR / "assets" / "jimu_base_assembly_5plates.glb"
BUILDER_PLATE_DIMS_M = {
    "square": (0.074, 0.0065, 0.074),
    "half_square": (0.037, 0.0065, 0.074),
    "triangle": (0.074, 0.0065, 0.135),
}


def _resolve_loaded_tray_mesh() -> tuple[Path, float]:
    if TRAY_LOADED_MESH_FILE.exists():
        return TRAY_LOADED_MESH_FILE, 1.0
    if not (TRAY_ONLY_MESH_FILE.exists() and PLATE_MESH_FILE.exists()):
        return TRAY_ONLY_MESH_FILE, 0.01
    try:
        import numpy as np
        import trimesh

        tray_scale = 0.01
        tray_loaded = trimesh.load(str(TRAY_ONLY_MESH_FILE), force="scene")
        plate_loaded = trimesh.load(str(PLATE_MESH_FILE), force="scene")
        tray_meshes = list(tray_loaded.dump(concatenate=False))
        plate_meshes = list(plate_loaded.dump(concatenate=False))
        if not tray_meshes or not plate_meshes:
            return TRAY_ONLY_MESH_FILE, tray_scale

        tray_bounds = np.asarray(tray_loaded.bounds, dtype=np.float32) * tray_scale
        min_v, max_v = tray_bounds
        x_values = np.linspace(float(min_v[0] + 0.00875), float(max_v[0] - 0.00875), 7)
        y_values = [
            float(min_v[1] + 0.040),
            float(max_v[1] - 0.040),
        ]
        plate_size = 0.074
        z_center = float(max_v[2] + 0.5 * plate_size - 0.012)
        r_tray_plate = np.asarray(
            [
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        scene = trimesh.Scene()
        tray_tf = np.eye(4, dtype=np.float32)
        tray_tf[:3, :3] *= tray_scale
        for idx, mesh in enumerate(tray_meshes):
            geom = mesh.copy()
            geom.apply_transform(tray_tf)
            scene.add_geometry(geom, geom_name=f"tray_{idx:02d}")

        slot_idx = 0
        for row, y in enumerate(y_values):
            for col, x in enumerate(x_values):
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = r_tray_plate
                T[:3, 3] = [float(x), float(y), z_center]
                for mesh_idx, mesh in enumerate(plate_meshes):
                    geom = mesh.copy()
                    geom.apply_transform(T)
                    scene.add_geometry(geom, geom_name=f"slot_{slot_idx:02d}_plate_{mesh_idx:02d}")
                slot_idx += 1

        TRAY_LOADED_MESH_FILE.parent.mkdir(parents=True, exist_ok=True)
        scene.export(str(TRAY_LOADED_MESH_FILE))
        print(f"[jimu-sam6d-provider] generated loaded tray mesh: {TRAY_LOADED_MESH_FILE}")
        return TRAY_LOADED_MESH_FILE, 1.0
    except Exception as exc:
        print(f"[jimu-sam6d-provider] failed to generate loaded tray mesh, using tray-only OBJ: {exc}", file=sys.stderr)
        return TRAY_ONLY_MESH_FILE, 0.01


TRAY_MESH_FILE, TRAY_MESH_SCALE = _resolve_loaded_tray_mesh()


def _install_jimu_sam3_prompt_variants() -> None:
    original = provider.sam3_text_prompt_variants
    if getattr(provider, "_jimu_sam3_prompt_variants_installed", False):
        return

    def sam3_text_prompt_variants_jimu(object_name: str, prompt: str) -> list[str]:
        normalized = provider.normalize_object_name(object_name) or str(object_name)
        variants = list(original(object_name, prompt))
        if normalized == "jimu_liaoban":
            variants += [
                "gray tray.",
                "plastic tray.",
                "gray plastic tray.",
                "red blocks in a gray tray.",
                "red building plates in a tray.",
                "tray with red plastic plates.",
                "building blocks in a tray.",
            ]
        elif normalized == "jimu_base_assembly":
            variants += [
                "red plastic building blocks.",
                "red square plastic panels.",
                "red building block assembly.",
                "red cross shaped object.",
                "red cross shaped base.",
                "red plastic base made of blocks.",
            ]
        out = []
        seen = set()
        for item in variants:
            text = str(item).strip()
            key = text.lower()
            if text and key not in seen:
                out.append(text)
                seen.add(key)
        return out

    provider.sam3_text_prompt_variants = sam3_text_prompt_variants_jimu
    provider._jimu_sam3_prompt_variants_installed = True


def register_jimu_assembly_specs() -> None:
    _install_jimu_sam3_prompt_variants()
    if TRAY_MESH_FILE.exists():
        object_specs.OBJECT_SPECS["jimu_liaoban"] = object_specs.ObjectSpec(
            name="jimu_liaoban",
            grounding_prompt="gray plastic tray loaded with upright red square building plates.",
            mesh_file=str(TRAY_MESH_FILE),
            mesh_scale=TRAY_MESH_SCALE,
            sim_asset_file=str(TRAY_MESH_FILE),
            sim_asset_scale=TRAY_MESH_SCALE,
            fixed_goal_joints_deg=object_specs.DEFAULT_FIXED_GOAL_JOINTS_DEG,
            grasp_mode="topdown_long_axis",
        )
        object_specs.OBJECT_NAME_ALIASES["jimu_liaoban"] = "jimu_liaoban"
        object_specs.OBJECT_NAME_ALIASES["jimu_tray"] = "jimu_liaoban"
        object_specs.OBJECT_NAME_ALIASES["liaoban"] = "jimu_liaoban"

    if BASE_ASSEMBLY_MESH_FILE.exists():
        object_specs.OBJECT_SPECS["jimu_base_assembly"] = object_specs.ObjectSpec(
            name="jimu_base_assembly",
            grounding_prompt="red cross shaped base assembly made of five square plastic building plates.",
            mesh_file=str(BASE_ASSEMBLY_MESH_FILE),
            mesh_scale=1.0,
            sim_asset_file=str(BASE_ASSEMBLY_MESH_FILE),
            sim_asset_scale=1.0,
            fixed_goal_joints_deg=object_specs.DEFAULT_FIXED_GOAL_JOINTS_DEG,
            grasp_mode="topdown_long_axis",
        )
        object_specs.OBJECT_NAME_ALIASES["jimu_base_assembly"] = "jimu_base_assembly"
        object_specs.OBJECT_NAME_ALIASES["jimu_base"] = "jimu_base_assembly"
        object_specs.OBJECT_NAME_ALIASES["base_assembly"] = "jimu_base_assembly"


def _pop_manual_bbox_flag() -> bool:
    enabled = False
    filtered = [sys.argv[0]]
    for item in sys.argv[1:]:
        if item in {"--jimu-manual-bboxes", "--manual-bboxes"}:
            enabled = True
            continue
        filtered.append(item)
    if enabled:
        sys.argv[:] = filtered
    return enabled


def _pop_tabletop_anchor_flag() -> bool:
    enabled = False
    filtered = [sys.argv[0]]
    for item in sys.argv[1:]:
        if item in {"--jimu-tabletop-anchors", "--tabletop-anchors"}:
            enabled = True
            continue
        filtered.append(item)
    if enabled:
        sys.argv[:] = filtered
    return enabled


def _pop_apriltag_anchor_flag() -> bool:
    enabled = False
    filtered = [sys.argv[0]]
    for item in sys.argv[1:]:
        if item in {"--jimu-apriltag-anchors", "--apriltag-anchors"}:
            enabled = True
            continue
        filtered.append(item)
    if enabled:
        sys.argv[:] = filtered
    return enabled


def _pop_custom_arg(flag: str, default, cast):
    out = default
    filtered = [sys.argv[0]]
    skip_next = False
    prefix = f"{flag}="
    for idx, item in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if item == flag:
            rest = sys.argv[1:]
            if idx + 1 >= len(rest):
                raise ValueError(f"{flag} requires a value")
            out = cast(rest[idx + 1])
            skip_next = True
            continue
        if item.startswith(prefix):
            out = cast(item.split("=", 1)[1])
            continue
        filtered.append(item)
    sys.argv[:] = filtered
    return out


def _pop_apriltag_config() -> dict:
    return {
        "base_id": _pop_custom_arg("--jimu-apriltag-base-id", 1, int),
        "tray_id": _pop_custom_arg("--jimu-apriltag-tray-id", 0, int),
        "base_size_m": _pop_custom_arg("--jimu-apriltag-base-size-m", 0.052, float),
        "tray_size_m": _pop_custom_arg("--jimu-apriltag-tray-size-m", 0.06, float),
        "base_yaw_deg": _pop_custom_arg("--jimu-apriltag-base-yaw-deg", 0.0, float),
        "tray_yaw_deg": _pop_custom_arg("--jimu-apriltag-tray-yaw-deg", 90.0, float),
        "tray_center_offset_x_m": _pop_custom_arg(
            "--jimu-apriltag-tray-center-offset-x-m",
            TRAY_APRILTAG_CENTER_OFFSET_X_M,
            float,
        ),
        "tray_center_offset_y_m": _pop_custom_arg(
            "--jimu-apriltag-tray-center-offset-y-m",
            TRAY_APRILTAG_CENTER_OFFSET_Y_M,
            float,
        ),
        "base_world_offset_x_m": _pop_custom_arg("--jimu-apriltag-base-world-offset-x-m", BASE_WORLD_OFFSET_X_M, float),
        "base_world_offset_y_m": _pop_custom_arg("--jimu-apriltag-base-world-offset-y-m", BASE_WORLD_OFFSET_Y_M, float),
        "tray_world_offset_x_m": _pop_custom_arg("--jimu-apriltag-tray-world-offset-x-m", TRAY_WORLD_OFFSET_X_M, float),
        "tray_world_offset_y_m": _pop_custom_arg("--jimu-apriltag-tray-world-offset-y-m", TRAY_WORLD_OFFSET_Y_M, float),
        "builder_scene_json": _pop_custom_arg("--jimu-builder-scene-json", "", str),
    }


def _draw_manual_bbox_overlay(frame_bgr: np.ndarray, boxes: dict[str, list[float]], current_name: str | None = None) -> np.ndarray:
    cv2 = provider.cv2
    canvas = np.asarray(frame_bgr, dtype=np.uint8).copy()
    palette = [
        (40, 220, 255),
        (80, 255, 80),
        (255, 180, 60),
        (255, 80, 180),
        (80, 160, 255),
        (180, 120, 255),
    ]
    for idx, (name, box) in enumerate(boxes.items()):
        color = palette[idx % len(palette)]
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, str(name), (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    if current_name:
        text = f"Draw bbox for {current_name}; Enter/Space OK, c cancel"
        cv2.putText(canvas, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _select_manual_bboxes(frame: dict, object_names: list[str], scene_dir: Path) -> dict[str, list[float]]:
    cv2 = provider.cv2
    frame_bgr = np.asarray(frame["bgr"], dtype=np.uint8)
    height, width = frame_bgr.shape[:2]
    window = "Jimu manual SAM6D boxes"
    boxes: dict[str, list[float]] = {}
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, min(width, 1280), min(height, 900))
    except Exception:
        pass

    for name in object_names:
        while True:
            canvas = _draw_manual_bbox_overlay(frame_bgr, boxes, current_name=name)
            try:
                roi = cv2.selectROI(window, canvas, showCrosshair=True, fromCenter=False)
            except Exception as exc:
                raise RuntimeError(f"failed to open OpenCV ROI selector for {name}: {exc}") from exc
            x, y, w, h = [float(v) for v in roi]
            if w <= 1.0 or h <= 1.0:
                raise RuntimeError(f"manual bbox selection was cancelled or empty for {name}")
            x1 = float(np.clip(x, 0.0, max(width - 1, 0)))
            y1 = float(np.clip(y, 0.0, max(height - 1, 0)))
            x2 = float(np.clip(x + w, x1 + 1.0, width))
            y2 = float(np.clip(y + h, y1 + 1.0, height))
            boxes[str(name)] = [x1, y1, x2, y2]
            print(f"[jimu-sam6d-manual] bbox {name}: {[round(v, 1) for v in boxes[str(name)]]}")
            break

    overlay = _draw_manual_bbox_overlay(frame_bgr, boxes)
    overlay_path = scene_dir / "manual_bboxes_overlay.png"
    json_path = scene_dir / "manual_bboxes.json"
    cv2.imwrite(str(overlay_path), overlay)
    with open(json_path, "w") as f:
        json.dump({"boxes_xyxy": boxes, "overlay": str(overlay_path)}, f, indent=2)
    print(f"[jimu-sam6d-manual] saved manual bbox overlay: {overlay_path}")
    try:
        cv2.destroyWindow(window)
    except Exception:
        pass
    return boxes


def _manual_bbox_mask_mode(mask_mode: str) -> str:
    mode = str(mask_mode or "").strip()
    if mode == "" or mode.startswith("sam3_"):
        return "box"
    return mode


def _load_matrix4(path: str | Path) -> np.ndarray:
    path = Path(path).expanduser()
    if path.suffix.lower() == ".npy":
        mat = np.load(path)
    else:
        mat = np.loadtxt(path)
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape != (4, 4):
        raise ValueError(f"expected 4x4 matrix from {path}, got {mat.shape}")
    return mat


def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    return (np.asarray(T, dtype=np.float32).reshape(4, 4) @ np.concatenate([pts, ones], axis=1).T).T[:, :3]


def _deproject_mask_to_base_points(frame: dict, mask: np.ndarray, T_cam_to_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(frame["depth_m"], dtype=np.float32)
    K = np.asarray(frame["K"], dtype=np.float32).reshape(3, 3)
    valid = np.asarray(mask > 0, dtype=bool) & np.isfinite(depth) & (depth > 0.05) & (depth < 2.0)
    ys, xs = np.where(valid)
    if xs.size <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)
    z = depth[ys, xs]
    x = (xs.astype(np.float32) - float(K[0, 2])) * z / float(K[0, 0])
    y = (ys.astype(np.float32) - float(K[1, 2])) * z / float(K[1, 1])
    pts_cam = np.stack([x, y, z], axis=1).astype(np.float32)
    pts_base = _transform_points(T_cam_to_base, pts_cam)
    return pts_base, np.stack([xs, ys], axis=1).astype(np.int32)


def _box_mask(shape: tuple[int, int], box: list[float]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def _red_mask_in_box(frame: dict, box: list[float]) -> np.ndarray:
    cv2 = provider.cv2
    box_mask = _box_mask(np.asarray(frame["depth_m"]).shape[:2], box)
    hsv = cv2.cvtColor(np.asarray(frame["bgr"], dtype=np.uint8), cv2.COLOR_BGR2HSV)
    lower1 = np.asarray([0, 55, 35], dtype=np.uint8)
    upper1 = np.asarray([12, 255, 255], dtype=np.uint8)
    lower2 = np.asarray([168, 55, 35], dtype=np.uint8)
    upper2 = np.asarray([179, 255, 255], dtype=np.uint8)
    red = (cv2.inRange(hsv, lower1, upper1) > 0) | (cv2.inRange(hsv, lower2, upper2) > 0)
    return red & box_mask


def _height_object_mask(points_base: np.ndarray, pixels: np.ndarray, shape: tuple[int, int], *, table_z: float) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if points_base.shape[0] <= 0:
        return mask
    z = points_base[:, 2]
    low = float(np.percentile(z, 8.0))
    threshold = max(float(table_z) + 0.003, low + 0.003)
    keep = (z >= threshold) & (z <= float(table_z) + 0.25)
    if int(np.count_nonzero(keep)) < 80:
        keep = z >= np.percentile(z, 55.0)
    if int(np.count_nonzero(keep)) < 30:
        keep = np.ones_like(z, dtype=bool)
    px = pixels[keep]
    mask[px[:, 1], px[:, 0]] = True
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = provider.cv2.morphologyEx(mask.astype(np.uint8), provider.cv2.MORPH_CLOSE, kernel).astype(bool)
    return mask


def _principal_xy_axis(points_xy: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 6:
        axis = np.asarray(fallback, dtype=np.float32).reshape(2)
    else:
        centered = pts - np.mean(pts, axis=0, keepdims=True)
        cov = centered.T @ centered / max(pts.shape[0] - 1, 1)
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, int(np.argmax(vals))].astype(np.float32)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-6:
        axis = np.asarray([1.0, 0.0], dtype=np.float32)
    else:
        axis = axis / norm
    if float(axis[0]) < 0.0:
        axis = -axis
    return axis.astype(np.float32)


def _robust_xy_center(points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] <= 0:
        return np.zeros(2, dtype=np.float32)
    lo = np.percentile(pts, 2.0, axis=0)
    hi = np.percentile(pts, 98.0, axis=0)
    return ((lo + hi) * 0.5).astype(np.float32)


def _rotation_from_columns(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    R = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)
    if float(np.linalg.det(R)) < 0.0:
        R[:, 2] *= -1.0
    return R


def _mesh_bounds_scaled(path: Path, scale: float) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    mesh = trimesh.load(str(path), force="scene")
    bounds = np.asarray(mesh.bounds, dtype=np.float32) * float(scale)
    return bounds[0], bounds[1]


def _tray_tag_center_from_recess(path: Path, scale: float, tag_size_m: float) -> np.ndarray | None:
    import trimesh

    try:
        mesh = trimesh.load(str(path), force="scene")
        vertices = [
            np.asarray(geom.vertices, dtype=np.float32)
            for geom in mesh.geometry.values()
            if hasattr(geom, "vertices") and len(getattr(geom, "vertices", []))
        ]
        if not vertices:
            return None
        points = np.concatenate(vertices, axis=0)
        raw_tag_size = float(tag_size_m) / float(scale)
        if raw_tag_size <= 0.0:
            return None
        bounds = np.asarray([np.min(points, axis=0), np.max(points, axis=0)], dtype=np.float32)
        bounds_center = 0.5 * (bounds[0] + bounds[1])

        def best_center(axis: int) -> float | None:
            values = sorted({round(float(v), 6) for v in points[:, axis].tolist()})
            candidates: list[tuple[float, float]] = []
            for i, left in enumerate(values):
                for right in values[i + 1 :]:
                    span = right - left
                    if abs(span - raw_tag_size) <= max(0.02, raw_tag_size * 0.01):
                        center = 0.5 * (left + right)
                        candidates.append((abs(center - float(bounds_center[axis])), center))
                    if span > raw_tag_size * 1.05:
                        break
            if not candidates:
                return None
            return min(candidates, key=lambda item: item[0])[1]

        center_x = best_center(0)
        center_y = best_center(1)
        if center_x is None or center_y is None:
            return None
        return np.asarray(
            [
                float(center_x) * float(scale),
                float(center_y) * float(scale),
                TRAY_APRILTAG_TOP_Z_M,
            ],
            dtype=np.float32,
        )
    except Exception:
        return None


def _camera_pose_from_base_pose(args, T_base_cam_raw: np.ndarray, T_base_obj: np.ndarray) -> np.ndarray:
    T_base_cam_raw = np.asarray(T_base_cam_raw, dtype=np.float32).reshape(4, 4)
    T_base_obj = np.asarray(T_base_obj, dtype=np.float32).reshape(4, 4)
    if bool(getattr(args, "use_direct_camera_extrinsic", False)):
        return (np.linalg.inv(T_base_cam_raw).astype(np.float32) @ T_base_obj).astype(np.float32)
    return (T_base_cam_raw @ T_base_obj).astype(np.float32)


def _estimate_tabletop_anchor_pose(
    args,
    frame: dict,
    object_name: str,
    box: list[float],
    T_cam_to_base: np.ndarray,
    table_z: float,
) -> tuple[np.ndarray, dict, np.ndarray]:
    object_name = provider.normalize_object_name(object_name) or str(object_name)
    height, width = np.asarray(frame["depth_m"]).shape[:2]
    box_mask = _box_mask((height, width), box)
    all_points_base, all_pixels = _deproject_mask_to_base_points(frame, box_mask, T_cam_to_base)
    if all_points_base.shape[0] < 20:
        raise RuntimeError(f"not enough valid depth pixels inside bbox for {object_name}")

    if object_name == "jimu_base_assembly":
        red_mask = _red_mask_in_box(frame, box)
        red_points_base, red_pixels = _deproject_mask_to_base_points(frame, red_mask, T_cam_to_base)
        if red_points_base.shape[0] >= 80:
            work_points = red_points_base
            work_pixels = red_pixels
            used_mask = red_mask
            mask_source = "manual_bbox_red_depth"
        else:
            used_mask = _height_object_mask(all_points_base, all_pixels, (height, width), table_z=table_z) & box_mask
            work_points, work_pixels = _deproject_mask_to_base_points(frame, used_mask, T_cam_to_base)
            mask_source = "manual_bbox_height_depth"
        footprint_center_local = np.asarray([0.0, 0.0], dtype=np.float32)
        origin_z = float(table_z + 0.003)
        axis_xy = _principal_xy_axis(work_points[:, :2], np.asarray([1.0, 0.0], dtype=np.float32))
        x_axis = np.asarray([axis_xy[0], axis_xy[1], 0.0], dtype=np.float32)
        y_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        z_axis = np.cross(x_axis, y_axis).astype(np.float32)
    elif object_name == "jimu_liaoban":
        used_mask = _height_object_mask(all_points_base, all_pixels, (height, width), table_z=table_z) & box_mask
        work_points, work_pixels = _deproject_mask_to_base_points(frame, used_mask, T_cam_to_base)
        if work_points.shape[0] < 80:
            work_points = all_points_base
            work_pixels = all_pixels
            used_mask = box_mask
        mask_source = "manual_bbox_height_depth"
        min_v, max_v = _mesh_bounds_scaled(TRAY_MESH_FILE, TRAY_MESH_SCALE)
        footprint_center_local = ((min_v[:2] + max_v[:2]) * 0.5).astype(np.float32)
        origin_z = float(table_z - float(min_v[2]))
        axis_xy = _principal_xy_axis(work_points[:, :2], np.asarray([1.0, 0.0], dtype=np.float32))
        x_axis = np.asarray([axis_xy[0], axis_xy[1], 0.0], dtype=np.float32)
        z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        y_axis = np.cross(z_axis, x_axis).astype(np.float32)
    else:
        raise ValueError(f"tabletop anchor localizer only supports Jimu anchors, got {object_name}")

    if work_points.shape[0] < 20:
        raise RuntimeError(f"not enough object points after filtering for {object_name}")
    center_xy = _robust_xy_center(work_points[:, :2])
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-6)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-6)
    z_axis = z_axis / max(float(np.linalg.norm(z_axis)), 1e-6)
    R = _rotation_from_columns(x_axis, y_axis, z_axis)

    origin_xy = center_xy - R[:2, 0] * float(footprint_center_local[0]) - R[:2, 1] * float(footprint_center_local[1])
    T_base_obj = np.eye(4, dtype=np.float32)
    T_base_obj[:3, :3] = R
    T_base_obj[:3, 3] = np.asarray([origin_xy[0], origin_xy[1], origin_z], dtype=np.float32)
    debug = {
        "object_name": object_name,
        "box_xyxy": [float(v) for v in box],
        "mask_source": mask_source,
        "valid_depth_pixels": int(all_points_base.shape[0]),
        "object_pixels": int(work_points.shape[0]),
        "base_center_xy_m": center_xy.astype(float).tolist(),
        "base_origin_m": T_base_obj[:3, 3].astype(float).tolist(),
        "yaw_deg": float(math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0])))),
        "table_z_m": float(table_z),
    }
    return T_base_obj, debug, used_mask


def _save_tabletop_anchor_overlay(frame: dict, mask: np.ndarray, box: list[float], debug: dict, out_path: Path) -> None:
    cv2 = provider.cv2
    canvas = np.asarray(frame["bgr"], dtype=np.uint8).copy()
    overlay = canvas.copy()
    overlay[np.asarray(mask > 0, dtype=bool)] = (0, 180, 255)
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0.0)
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = (
        f"{debug.get('object_name')} yaw={float(debug.get('yaw_deg', 0.0)):.1f} "
        f"origin={np.round(np.asarray(debug.get('base_origin_m', [0, 0, 0]), dtype=float), 3).tolist()}"
    )
    cv2.putText(canvas, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def _matrix_from_json(value) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(4, 4)
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _tag_corners_from_pose(
    T_object_tag: np.ndarray,
    tag_size_m: float,
    *,
    mirror_u: bool = False,
    mirror_v: bool = False,
) -> np.ndarray:
    T_object_tag = np.asarray(T_object_tag, dtype=np.float32).reshape(4, 4)
    half = 0.5 * float(tag_size_m)
    center = T_object_tag[:3, 3].astype(np.float32)
    marker_right = T_object_tag[:3, 0].astype(np.float32)
    marker_up = T_object_tag[:3, 2].astype(np.float32)
    marker_right = marker_right / max(float(np.linalg.norm(marker_right)), 1e-6)
    marker_up = marker_up - float(np.dot(marker_up, marker_right)) * marker_right
    marker_up = marker_up / max(float(np.linalg.norm(marker_up)), 1e-6)
    if mirror_u:
        marker_right = -marker_right
    if mirror_v:
        marker_up = -marker_up
    return np.asarray(
        [
            center - half * marker_right + half * marker_up,
            center + half * marker_right + half * marker_up,
            center + half * marker_right - half * marker_up,
            center - half * marker_right - half * marker_up,
        ],
        dtype=np.float32,
    )


def _builder_piece_lookup(payload: dict, key: str | None) -> dict | None:
    name = provider.normalize_object_name(key)
    if not name:
        return None
    for piece in list(payload.get("pieces") or []):
        if not isinstance(piece, dict):
            continue
        if name in {str(piece.get("id") or "").strip(), str(piece.get("role") or "").strip()}:
            return piece
    return None


def _builder_piece_matrix(piece: dict) -> np.ndarray | None:
    try:
        u = np.asarray(piece.get("u"), dtype=np.float32).reshape(3)
        n = np.asarray(piece.get("n"), dtype=np.float32).reshape(3)
        center = np.asarray(piece.get("center"), dtype=np.float32).reshape(3)
    except Exception:
        return None
    x = u / max(float(np.linalg.norm(u)), 1e-8)
    y = n - float(np.dot(n, x)) * x
    y = y / max(float(np.linalg.norm(y)), 1e-8)
    z = np.cross(x, y)
    z = z / max(float(np.linalg.norm(z)), 1e-8)
    y = np.cross(z, x)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.column_stack([x, y, z]).astype(np.float32)
    T[:3, 3] = center
    return T


def _normalize_builder_locked_top_surfaces(payload: dict) -> None:
    # Keep the exported builder frame untouched.  The tag mount data and child
    # piece transforms are defined relative to the locked piece axes as saved.
    return


def _builder_attached_tag_pose(payload: dict, tag: dict) -> np.ndarray | None:
    piece = _builder_piece_lookup(payload, tag.get("attached_to_piece_id")) or _builder_piece_lookup(
        payload,
        tag.get("attached_to_role"),
    )
    if piece is None:
        return None
    T_builder_piece = _builder_piece_matrix(piece)
    if T_builder_piece is None:
        return None
    piece_type = str(piece.get("type") or "square").strip().lower()
    dims = BUILDER_PLATE_DIMS_M.get(piece_type, BUILDER_PLATE_DIMS_M["square"])
    try:
        piece_n = np.asarray(piece.get("n"), dtype=np.float32).reshape(3)
    except Exception:
        piece_n = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    normal_sign = 1.0 if float(piece_n[1]) >= 0.0 else -1.0
    center = np.asarray(tag.get("local_center_m") or [0.0, 0.0, 0.0], dtype=np.float32).reshape(3)
    if str(tag.get("attached_surface") or "") == "piece_top":
        center[1] = normal_sign * 0.5 * float(dims[1])
    yaw = math.radians(float(tag.get("local_yaw_deg") or 0.0))
    c = math.cos(yaw)
    s = math.sin(yaw)
    u = np.asarray([c, 0.0, s], dtype=np.float32)
    n = np.asarray([0.0, normal_sign, 0.0], dtype=np.float32)
    v = np.cross(u, n).astype(np.float32)
    v = v / max(float(np.linalg.norm(v)), 1e-8)
    T_piece_tag = np.eye(4, dtype=np.float32)
    T_piece_tag[:3, :3] = np.column_stack([u, n, v]).astype(np.float32)
    T_piece_tag[:3, 3] = center
    return (T_builder_piece @ T_piece_tag).astype(np.float32)


def _builder_scene_tag_corners(
    builder_scene_json: str,
    object_name: str,
    tag_id: int,
    tag_size_m: float,
) -> tuple[np.ndarray | None, dict]:
    if not builder_scene_json:
        return None, {}
    if provider.normalize_object_name(object_name) != "jimu_base_assembly":
        return None, {}
    scene_path = Path(str(builder_scene_json)).expanduser()
    if not scene_path.exists():
        raise FileNotFoundError(f"builder scene JSON not found for AprilTag localization: {scene_path}")
    payload = json.loads(scene_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "jimu_builder_scene_v1":
        raise ValueError(f"{scene_path} is not a jimu_builder_scene_v1 JSON")
    _normalize_builder_locked_top_surfaces(payload)
    apriltags = payload.get("apriltags") or {}
    for tag in list(apriltags.get("attached_tags") or []):
        if int(tag.get("tag_id", -1)) != int(tag_id):
            continue
        T_builder_tag = _builder_attached_tag_pose(payload, tag)
        if T_builder_tag is None:
            T_builder_tag = _matrix_from_json(tag.get("T_builder_tag"))
        if T_builder_tag is None:
            continue
        size = float(tag.get("tag_black_square_size_m") or tag_size_m)
        mirror_u = bool(tag.get("texture_mirror_u", False))
        mirror_v = bool(tag.get("texture_mirror_v", False))
        return _tag_corners_from_pose(T_builder_tag, size, mirror_u=mirror_u, mirror_v=mirror_v), {
            "builder_scene_json": str(scene_path),
            "builder_tag_source": "attached_tags",
            "builder_tag_name": tag.get("name"),
            "builder_tag_attached_to_role": tag.get("attached_to_role"),
            "builder_tag_size_m": size,
            "builder_tag_texture_mirror_u": mirror_u,
            "builder_tag_texture_mirror_v": mirror_v,
            "T_object_tag": T_builder_tag.astype(float).tolist(),
        }
    mount = (apriltags.get("mounts") or {}).get("base")
    if isinstance(mount, dict) and int(mount.get("tag_id", -1)) == int(tag_id):
        local_pose = mount.get("local_pose") or {}
        center = np.asarray(local_pose.get("center_m", [0.0, BASE_APRILTAG_TOP_Y_M, 0.0]), dtype=np.float32).reshape(3)
        u = np.asarray(local_pose.get("u", [1.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
        n = np.asarray(local_pose.get("n", [0.0, 1.0, 0.0]), dtype=np.float32).reshape(3)
        v = np.asarray(local_pose.get("v", [0.0, 0.0, -1.0]), dtype=np.float32).reshape(3)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = np.column_stack([u, n, v]).astype(np.float32)
        T[:3, 3] = center
        size = float(mount.get("tag_black_square_size_m") or tag_size_m)
        return _tag_corners_from_pose(T, size), {
            "builder_scene_json": str(scene_path),
            "builder_tag_source": "mounts.base",
            "builder_tag_size_m": size,
            "T_object_tag": T.astype(float).tolist(),
        }
    return None, {"builder_scene_json": str(scene_path), "builder_tag_source": "not_found"}


def _tag_corners_in_anchor_object(
    object_name: str,
    tag_size_m: float,
    tag_yaw_deg: float,
    *,
    tray_center_offset_xy_m: tuple[float, float] = (0.0, 0.0),
    tag_id: int | None = None,
    builder_scene_json: str = "",
) -> np.ndarray:
    object_name = provider.normalize_object_name(object_name) or str(object_name)
    builder_points, _builder_debug = _builder_scene_tag_corners(
        builder_scene_json,
        object_name,
        -1 if tag_id is None else int(tag_id),
        float(tag_size_m),
    )
    if builder_points is not None:
        return builder_points
    half = 0.5 * float(tag_size_m)
    theta = math.radians(float(tag_yaw_deg))
    c = math.cos(theta)
    s = math.sin(theta)
    if object_name == "jimu_base_assembly":
        # Base assembly local Y is the vertical axis in the Jimu plate frame.
        center = np.asarray([0.0, BASE_APRILTAG_TOP_Y_M, 0.0], dtype=np.float32)
        plane_u = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        # The printed tag top is aligned toward the base assembly local -Z edge.
        # Using +Z flips the thin local Y axis downward after PnP.
        plane_v = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    elif object_name == "jimu_liaoban":
        min_v, max_v = _mesh_bounds_scaled(TRAY_ONLY_MESH_FILE, 0.01)
        # The new tray has tag0 mounted at the geometric center of the tray.
        # Do not infer this from recess edges: those vertices describe the slot
        # bevels and can bias the tag center by several millimeters.
        center = np.asarray(
            [
                float((min_v[0] + max_v[0]) * 0.5),
                float((min_v[1] + max_v[1]) * 0.5),
                TRAY_APRILTAG_TOP_Z_M,
            ],
            dtype=np.float32,
        )
        center[:2] += np.asarray(tray_center_offset_xy_m, dtype=np.float32).reshape(2)
        # The tray tag is on the top face; tray local Z is the upward axis.
        plane_u = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        plane_v = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        raise ValueError(f"AprilTag anchor localization only supports Jimu anchors, got {object_name}")

    marker_right = (c * plane_u + s * plane_v).astype(np.float32)
    marker_up = (-s * plane_u + c * plane_v).astype(np.float32)
    # OpenCV ArUco/AprilTag corners are top-left, top-right, bottom-right, bottom-left
    # in the decoded marker image.  These object points encode that printed marker
    # frame directly inside the Jimu anchor frame, so solvePnP returns T_cam_anchor.
    return np.asarray(
        [
            center - half * marker_right + half * marker_up,
            center + half * marker_right + half * marker_up,
            center + half * marker_right - half * marker_up,
            center - half * marker_right - half * marker_up,
        ],
        dtype=np.float32,
    )


def _detect_apriltag_markers(frame: dict) -> tuple[list[np.ndarray], np.ndarray | None]:
    cv2 = provider.cv2
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV was built without cv2.aruco; cannot detect AprilTags")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.015
    params.maxMarkerPerimeterRate = 4.0
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 40
    params.cornerRefinementMinAccuracy = 0.01

    gray = cv2.cvtColor(np.asarray(frame["bgr"], dtype=np.uint8), cv2.COLOR_BGR2GRAY)

    def detect_once(image: np.ndarray, scale: float = 1.0):
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(dictionary, params)
            c, i, _rejected = detector.detectMarkers(image)
        else:
            c, i, _rejected = cv2.aruco.detectMarkers(image, dictionary, parameters=params)
        corners_list = list(c or [])
        if scale != 1.0:
            corners_list = [np.asarray(item, dtype=np.float32) / float(scale) for item in corners_list]
        ids_list = [] if i is None else [int(v) for v in np.asarray(i).reshape(-1).tolist()]
        return corners_list, ids_list

    variants = [
        (gray, 1.0),
        (cv2.equalizeHist(gray), 1.0),
        (cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC), 2.0),
        (cv2.resize(cv2.equalizeHist(gray), None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC), 2.0),
    ]
    found: dict[int, np.ndarray] = {}
    for image, scale in variants:
        corners_list, ids_list = detect_once(image, scale)
        for tag_id, corner in zip(ids_list, corners_list):
            found.setdefault(int(tag_id), np.asarray(corner, dtype=np.float32))
    if not found:
        return [], None
    ids_arr = np.asarray([[tag_id] for tag_id in sorted(found.keys())], dtype=np.int32)
    return [found[int(tag_id)] for tag_id in ids_arr.reshape(-1).tolist()], ids_arr


def _solve_anchor_from_tag(
    frame: dict,
    object_name: str,
    corner: np.ndarray,
    tag_size_m: float,
    tag_yaw_deg: float,
    *,
    tag_id: int | None = None,
    tray_center_offset_xy_m: tuple[float, float] = (0.0, 0.0),
    builder_scene_json: str = "",
) -> tuple[np.ndarray, dict]:
    cv2 = provider.cv2
    K = np.asarray(frame["K"], dtype=np.float32).reshape(3, 3)
    image_points = np.asarray(corner, dtype=np.float32).reshape(4, 2)
    object_points = _tag_corners_in_anchor_object(
        object_name,
        tag_size_m,
        tag_yaw_deg,
        tray_center_offset_xy_m=tray_center_offset_xy_m,
        tag_id=tag_id,
        builder_scene_json=builder_scene_json,
    )
    pnp_flags: list[tuple[str, int]] = []
    for flag_name in ("SOLVEPNP_SQPNP", "SOLVEPNP_ITERATIVE", "SOLVEPNP_IPPE"):
        flag_value = getattr(cv2, flag_name, None)
        if flag_value is not None:
            pnp_flags.append((flag_name, int(flag_value)))
    if not pnp_flags:
        pnp_flags.append(("SOLVEPNP_ITERATIVE", int(cv2.SOLVEPNP_ITERATIVE)))

    candidates = []
    for flag_name, flag_value in pnp_flags:
        try:
            ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, None, flags=flag_value)
        except Exception:
            continue
        if not ok:
            continue
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, None)
        errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
        mean_error = float(np.mean(errors))
        candidates.append((mean_error, flag_name, rvec, tvec, errors))

    if not candidates:
        raise RuntimeError(f"solvePnP failed for {object_name}")
    mean_error, flag_name, rvec, tvec, point_errors = min(candidates, key=lambda item: item[0])
    R, _ = cv2.Rodrigues(rvec)
    T_cam_obj = np.eye(4, dtype=np.float32)
    T_cam_obj[:3, :3] = np.asarray(R, dtype=np.float32)
    T_cam_obj[:3, 3] = np.asarray(tvec, dtype=np.float32).reshape(3)
    debug = {
        "object_name": provider.normalize_object_name(object_name) or str(object_name),
        "tag_size_m": float(tag_size_m),
        "tag_yaw_deg": float(tag_yaw_deg),
        "tray_center_offset_xy_m": [float(tray_center_offset_xy_m[0]), float(tray_center_offset_xy_m[1])],
        "image_corners_px": image_points.astype(float).tolist(),
        "object_corners_m": object_points.astype(float).tolist(),
        "object_tag_center_m": np.mean(object_points, axis=0).astype(float).tolist(),
        "translation_m": T_cam_obj[:3, 3].astype(float).tolist(),
        "solve_pnp_flag": str(flag_name),
        "solve_pnp_reprojection_error_px": float(mean_error),
        "solve_pnp_point_errors_px": np.asarray(point_errors, dtype=np.float32).astype(float).tolist(),
    }
    builder_points, builder_debug = _builder_scene_tag_corners(
        builder_scene_json,
        object_name,
        -1 if tag_id is None else int(tag_id),
        float(tag_size_m),
    )
    if builder_points is not None or builder_debug:
        debug["builder_scene_tag"] = builder_debug
    return T_cam_obj, debug


def _camera_to_base_transform_from_args(args) -> np.ndarray:
    T_raw = _load_matrix4(args.camera_extrinsic_opencv_path)
    if bool(getattr(args, "use_direct_camera_extrinsic", False)):
        return T_raw.astype(np.float32)
    return np.linalg.inv(T_raw).astype(np.float32)


def _apply_world_xy_offset_to_cam_pose(args, T_cam_obj: np.ndarray, offset_xy_m: tuple[float, float]) -> tuple[np.ndarray, dict]:
    offset_xy = np.asarray(offset_xy_m, dtype=np.float32).reshape(2)
    T_cam_obj = np.asarray(T_cam_obj, dtype=np.float32).reshape(4, 4).copy()
    if float(np.linalg.norm(offset_xy)) <= 1.0e-9:
        return T_cam_obj, {
            "world_xy_offset_m": [float(offset_xy[0]), float(offset_xy[1])],
            "applied": False,
        }

    T_cam_to_base = _camera_to_base_transform_from_args(args)
    T_base_obj_before = (T_cam_to_base @ T_cam_obj).astype(np.float32)
    T_base_obj_after = T_base_obj_before.copy()
    # This is a robot-base/world-table XY nudge, not a tag-local/CAD-local offset.
    T_base_obj_after[0, 3] += float(offset_xy[0])
    T_base_obj_after[1, 3] += float(offset_xy[1])
    T_base_to_cam = np.linalg.inv(T_cam_to_base).astype(np.float32)
    T_cam_obj_after = (T_base_to_cam @ T_base_obj_after).astype(np.float32)
    return T_cam_obj_after, {
        "world_xy_offset_m": [float(offset_xy[0]), float(offset_xy[1])],
        "applied": True,
        "base_translation_before_m": T_base_obj_before[:3, 3].astype(float).tolist(),
        "base_translation_after_m": T_base_obj_after[:3, 3].astype(float).tolist(),
    }


def _save_apriltag_overlay(frame: dict, detections: list[dict], out_path: Path) -> None:
    cv2 = provider.cv2
    canvas = np.asarray(frame["bgr"], dtype=np.uint8).copy()
    for item in detections:
        corners = np.asarray(item["image_corners_px"], dtype=np.float32).reshape(4, 2)
        pts = np.round(corners).astype(np.int32)
        color = (0, 255, 0) if bool(item.get("used", False)) else (0, 180, 255)
        cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
        center = np.mean(corners, axis=0)
        label = f"id={item.get('tag_id')} {item.get('object_name', '')}"
        cv2.putText(
            canvas,
            label,
            tuple(np.round(center + np.asarray([6.0, -6.0])).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def _save_tray_projection_diagnostic(summary_path: Path, summary: dict) -> dict | None:
    if not any(item.get("object_name") == "jimu_liaoban" and item.get("ok") for item in summary.get("results") or []):
        return None
    tool_path = SCRIPT_DIR / "tools" / "fit_apriltag_tray_offset.py"
    if not tool_path.exists():
        return {"ok": False, "error": f"diagnostic tool not found: {tool_path}"}
    out_dir = summary_path.parent / "tray_projection_diagnostic"
    cmd = [
        sys.executable,
        str(tool_path),
        str(summary_path),
        "--yaw-min",
        "90",
        "--yaw-max",
        "90",
        "--yaw-step",
        "1",
        "--offset-mm",
        "0",
        "--out-dir",
        str(out_dir),
    ]
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        if proc.returncode != 0:
            return {
                "ok": False,
                "command": cmd,
                "error": (proc.stderr or proc.stdout or "").strip()[-2000:],
            }
        scores_path = out_dir / "tray_offset_fit_scores.json"
        overlay_path = out_dir / "tray_offset_fit_overlay.png"
        payload = json.load(open(scores_path, "r", encoding="utf-8"))
        current = payload.get("current") or {}
        return {
            "ok": True,
            "mode": "current_pose_projection_only",
            "scores_path": str(scores_path),
            "overlay_path": str(overlay_path),
            "median_px": current.get("median_px"),
            "mean_px": current.get("mean_px"),
            "visible_samples": current.get("visible_samples"),
            "z_err_m": current.get("z_err_m"),
            "contour_warning": payload.get("contour_warning"),
        }
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def _run_apriltag_anchor_provider(config: dict) -> None:
    args = provider.parse_args()
    provider.apply_frame_dir_args(args)
    offline = args.rgb_path is not None or args.depth_path is not None or args.camera_path is not None
    if offline and not (args.rgb_path and args.depth_path and args.camera_path):
        raise ValueError("--rgb-path, --depth-path, and --camera-path must be provided together")
    object_names = list(args.object_names or ([] if args.object_name is None else [args.object_name]))
    if not object_names:
        object_names = ["jimu_base_assembly", "jimu_liaoban"]
    provider.validate_object_name_args(object_names)

    output_root = Path(args.output_root).expanduser()
    scene_dir = output_root / f"{provider._now_stamp()}_apriltag_anchor_{len(object_names)}objects_pid{os.getpid()}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    with open(scene_dir / "run_args.json", "w") as f:
        json.dump({"args": vars(args), "object_names": object_names, "apriltag_anchor_localizer": True, "apriltag_config": config}, f, indent=2)

    required_tag_ids = [int(config["base_id"]), int(config["tray_id"])]
    if offline:
        print("[jimu-apriltag] loading one RGB frame for AprilTag anchor localization")
        frame = provider.load_offline_frame(args)
        corners, ids = _detect_apriltag_markers(frame)
    else:
        max_attempts = 5
        frame = None
        corners = []
        ids = None
        best = None
        print(
            "[jimu-apriltag] capturing RGB frames for AprilTag anchor localization "
            f"(max_attempts={max_attempts}, required={required_tag_ids})"
        )
        for attempt in range(1, max_attempts + 1):
            try:
                attempt_frame = provider.capture_realsense_frame(args)
            except RuntimeError as exc:
                print(f"[jimu-apriltag] attempt {attempt}/{max_attempts}: frame capture failed: {exc}")
                continue
            attempt_corners, attempt_ids = _detect_apriltag_markers(attempt_frame)
            attempt_ids_list = [] if attempt_ids is None else [int(v) for v in np.asarray(attempt_ids).reshape(-1).tolist()]
            hit_count = sum(1 for tag_id in required_tag_ids if tag_id in set(attempt_ids_list))
            print(f"[jimu-apriltag] attempt {attempt}/{max_attempts}: detected tag ids: {attempt_ids_list}")
            if best is None or hit_count > best[0] or (hit_count == best[0] and len(attempt_ids_list) > len(best[3])):
                best = (hit_count, attempt_frame, attempt_corners, attempt_ids_list, attempt_ids)
            if hit_count >= len(set(required_tag_ids)):
                break
        if best is None:
            raise RuntimeError("failed to capture any RealSense frame for AprilTag anchor localization")
        _hit_count, frame, corners, _best_ids_list, ids = best
    shared_frame_dir = scene_dir / "shared_frame"
    shared_frame_dir.mkdir(parents=True, exist_ok=True)
    provider.save_sam6d_input_frame(frame, shared_frame_dir)
    id_values = [] if ids is None else [int(v) for v in np.asarray(ids).reshape(-1).tolist()]
    print(f"[jimu-apriltag] selected detected tag ids: {id_values}")
    corners_by_id = {tag_id: corner for tag_id, corner in zip(id_values, corners)}
    spec_by_object = {
        "jimu_base_assembly": {
            "tag_id": int(config["base_id"]),
            "tag_size_m": float(config["base_size_m"]),
            "tag_yaw_deg": float(config["base_yaw_deg"]),
            "world_offset_xy_m": (
                float(config.get("base_world_offset_x_m", 0.0)),
                float(config.get("base_world_offset_y_m", 0.0)),
            ),
        },
        "jimu_liaoban": {
            "tag_id": int(config["tray_id"]),
            "tag_size_m": float(config["tray_size_m"]),
            "tag_yaw_deg": float(config["tray_yaw_deg"]),
            "tray_center_offset_xy_m": (
                float(config.get("tray_center_offset_x_m", 0.0)),
                float(config.get("tray_center_offset_y_m", 0.0)),
            ),
            "world_offset_xy_m": (
                float(config.get("tray_world_offset_x_m", 0.0)),
                float(config.get("tray_world_offset_y_m", 0.0)),
            ),
        },
    }

    results = []
    overlay_items = [
        {"tag_id": tag_id, "image_corners_px": np.asarray(corner).reshape(4, 2).astype(float).tolist(), "used": False}
        for tag_id, corner in corners_by_id.items()
    ]
    for index, name in enumerate(object_names):
        object_name = provider.normalize_object_name(name) or str(name)
        run_dir = scene_dir / f"{index + 1:02d}_{provider._safe_cache_name(object_name)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        spec = spec_by_object.get(object_name)
        if spec is None:
            result = {"object_name": object_name, "ok": False, "error": f"unsupported AprilTag anchor object: {object_name}", "run_dir": str(run_dir)}
            results.append(result)
            continue
        tag_id = int(spec["tag_id"])
        corner = corners_by_id.get(tag_id)
        if corner is None:
            result = {
                "object_name": object_name,
                "ok": False,
                "error": f"AprilTag id {tag_id} was not detected",
                "run_dir": str(run_dir),
                "detected_tag_ids": id_values,
            }
            results.append(result)
            continue
        try:
            T_cam_obj, debug = _solve_anchor_from_tag(
                frame,
                object_name,
                corner,
                float(spec["tag_size_m"]),
                float(spec["tag_yaw_deg"]),
                tag_id=tag_id,
                tray_center_offset_xy_m=tuple(spec.get("tray_center_offset_xy_m", (0.0, 0.0))),
                builder_scene_json=str(config.get("builder_scene_json", "") or ""),
            )
            T_cam_obj, world_offset_debug = _apply_world_xy_offset_to_cam_pose(
                args,
                T_cam_obj,
                tuple(spec.get("world_offset_xy_m", (0.0, 0.0))),
            )
            debug["world_xy_offset"] = world_offset_debug
            debug["translation_m_after_world_xy_offset"] = T_cam_obj[:3, 3].astype(float).tolist()
            result = {
                "object_name": object_name,
                "prompt": "apriltag geometric anchor",
                "run_dir": str(run_dir),
                "ok": True,
                "score": 1.0,
                "sam3_instance_index": int(index),
                "mask_source": "apriltag_25h9",
                "mask_pixels": 0,
                "T_cam_obj": T_cam_obj.astype(float).tolist(),
                "translation_m": T_cam_obj[:3, 3].astype(float).tolist(),
                "jimu_apriltag_anchor": {
                    "tag_id": tag_id,
                    **debug,
                },
            }
            print(
                f"[jimu-apriltag] {object_name}: tag_id={tag_id} "
                f"size={float(spec['tag_size_m']):.3f}m t={np.round(T_cam_obj[:3, 3], 4).tolist()}"
            )
            for item in overlay_items:
                if int(item.get("tag_id", -1)) == tag_id:
                    item["used"] = True
                    item["object_name"] = object_name
        except Exception as exc:
            result = {"object_name": object_name, "ok": False, "error": repr(exc), "run_dir": str(run_dir), "tag_id": tag_id}
            print(f"[jimu-apriltag] {object_name} failed: {exc!r}")
        results.append(result)

    overlay_path = scene_dir / "apriltag_anchor_overlay.png"
    _save_apriltag_overlay(frame, overlay_items, overlay_path)
    summary = {
        "scene_dir": str(scene_dir),
        "object_count": len(object_names),
        "ok_count": sum(1 for item in results if item.get("ok")),
        "results": results,
        "jimu_apriltag_anchor_localization": {
            "enabled": True,
            "family": "tag25h9",
            "config": config,
            "detected_tag_ids": id_values,
            "overlay_path": str(overlay_path),
            "frame_dir": str(shared_frame_dir),
        },
    }
    summary_path = scene_dir / "full_scene_pose_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    tray_diagnostic = _save_tray_projection_diagnostic(summary_path, summary)
    if tray_diagnostic is not None:
        summary["jimu_apriltag_anchor_localization"]["tray_projection_diagnostic"] = tray_diagnostic
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        if tray_diagnostic.get("ok"):
            median_px = tray_diagnostic.get("median_px")
            if median_px is None:
                print(
                    "[jimu-apriltag] tray projection diagnostic: "
                    f"no reliable contour score; overlay={tray_diagnostic.get('overlay_path')}"
                )
            else:
                print(
                    "[jimu-apriltag] tray projection diagnostic: "
                    f"median_px={float(median_px):.2f} overlay={tray_diagnostic.get('overlay_path')}"
                )
        else:
            print(f"[jimu-apriltag] tray projection diagnostic failed: {tray_diagnostic.get('error')}")
    print(f"[jimu-apriltag] full scene result: {summary_path}")
    print(f"[sam6d-gdino] full scene result: {summary_path}")
    print(f"[sam6d-gdino] ok_count={summary['ok_count']}/{summary['object_count']}")


def _run_tabletop_anchor_provider() -> None:
    args = provider.parse_args()
    provider.apply_frame_dir_args(args)
    offline = args.rgb_path is not None or args.depth_path is not None or args.camera_path is not None
    if offline and not (args.rgb_path and args.depth_path and args.camera_path):
        raise ValueError("--rgb-path, --depth-path, and --camera-path must be provided together")
    object_names = list(args.object_names or ([] if args.object_name is None else [args.object_name]))
    if not object_names:
        object_names = ["jimu_base_assembly", "jimu_liaoban"]
    provider.validate_object_name_args(object_names)

    output_root = Path(args.output_root).expanduser()
    scene_dir = output_root / f"{provider._now_stamp()}_tabletop_anchor_{len(object_names)}objects_pid{os.getpid()}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    with open(scene_dir / "run_args.json", "w") as f:
        json.dump({"args": vars(args), "object_names": object_names, "tabletop_anchor_localizer": True}, f, indent=2)

    print("[jimu-tabletop] capturing one RGB-D frame for tabletop anchor localization")
    frame = provider.load_offline_frame(args) if offline else provider.capture_realsense_frame(args)
    shared_frame_dir = scene_dir / "shared_frame"
    shared_frame_dir.mkdir(parents=True, exist_ok=True)
    provider.save_sam6d_input_frame(frame, shared_frame_dir)
    boxes = _select_manual_bboxes(frame, object_names, scene_dir)

    T_base_cam_raw = _load_matrix4(args.camera_extrinsic_opencv_path)
    T_cam_to_base = T_base_cam_raw if bool(getattr(args, "use_direct_camera_extrinsic", False)) else np.linalg.inv(T_base_cam_raw).astype(np.float32)
    table_z = 0.0
    results = []
    debug_items = []
    for index, name in enumerate(object_names):
        safe_name = provider.normalize_object_name(name) or provider._safe_cache_name(name)
        run_dir = scene_dir / f"{index + 1:02d}_{safe_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            T_base_obj, debug, mask = _estimate_tabletop_anchor_pose(
                args,
                frame,
                name,
                boxes[str(name)],
                T_cam_to_base,
                table_z,
            )
            mask_path = run_dir / "tabletop_anchor_mask.png"
            provider.cv2.imwrite(str(mask_path), np.asarray(mask, dtype=np.uint8) * 255)
            overlay_path = run_dir / "tabletop_anchor_overlay.png"
            _save_tabletop_anchor_overlay(frame, mask, boxes[str(name)], debug, overlay_path)
            T_cam_obj = _camera_pose_from_base_pose(args, T_base_cam_raw, T_base_obj)
            result = {
                "object_name": provider.normalize_object_name(name) or str(name),
                "prompt": "tabletop geometric anchor",
                "run_dir": str(run_dir),
                "ok": True,
                "score": 1.0,
                "sam3_instance_index": int(index),
                "mask_source": debug["mask_source"],
                "mask_pixels": int(np.count_nonzero(mask)),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "T_cam_obj": T_cam_obj.astype(float).tolist(),
                "translation_m": T_cam_obj[:3, 3].astype(float).tolist(),
                "jimu_tabletop_anchor": debug,
            }
            print(
                f"[jimu-tabletop] {name}: origin={np.round(T_base_obj[:3, 3], 4).tolist()} "
                f"yaw={debug['yaw_deg']:.1f}deg pixels={debug['object_pixels']}"
            )
        except Exception as exc:
            result = {
                "object_name": provider.normalize_object_name(name) or str(name),
                "ok": False,
                "error": repr(exc),
                "run_dir": str(run_dir),
                "manual_bbox_xyxy": boxes.get(str(name)),
            }
            print(f"[jimu-tabletop] {name} failed: {exc!r}")
        results.append(result)
        debug_items.append(result.get("jimu_tabletop_anchor", result))

    summary = {
        "scene_dir": str(scene_dir),
        "object_count": len(object_names),
        "ok_count": sum(1 for item in results if item.get("ok")),
        "results": results,
        "manual_bboxes": boxes,
        "jimu_tabletop_anchor_localization": {
            "enabled": True,
            "frame_dir": str(shared_frame_dir),
            "debug": debug_items,
        },
    }
    summary_path = scene_dir / "full_scene_pose_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[jimu-tabletop] full scene result: {summary_path}")
    print(f"[sam6d-gdino] full scene result: {summary_path}")
    print(f"[sam6d-gdino] ok_count={summary['ok_count']}/{summary['object_count']}")


def _run_manual_bbox_provider() -> None:
    args = provider.parse_args()
    provider.apply_frame_dir_args(args)
    offline = args.rgb_path is not None or args.depth_path is not None or args.camera_path is not None
    if offline and not (args.rgb_path and args.depth_path and args.camera_path):
        raise ValueError("--rgb-path, --depth-path, and --camera-path must be provided together")

    object_names = list(args.object_names or ([] if args.object_name is None else [args.object_name]))
    if not object_names:
        raise ValueError("provide --object-name for one object or --object-names for multiple objects")
    provider.validate_object_name_args(object_names)
    if args.bbox is not None:
        raise ValueError("--jimu-manual-bboxes selects boxes interactively; do not also pass --bbox")

    sam6d_root = Path(args.sam6d_root).expanduser().resolve()
    if not (sam6d_root / "Pose_Estimation_Model" / "run_inference_custom.py").exists():
        raise FileNotFoundError(f"invalid SAM-6D root: {sam6d_root}")

    output_root = Path(args.output_root).expanduser()
    scene_dir = output_root / f"{provider._now_stamp()}_manual_bbox_{len(object_names)}objects_pid{os.getpid()}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    with open(scene_dir / "run_args.json", "w") as f:
        json.dump({"args": vars(args), "object_names": object_names, "manual_bboxes": True}, f, indent=2)

    print("[jimu-sam6d-manual] capturing one RGB-D frame for manual boxes")
    frame = provider.load_offline_frame(args) if offline else provider.capture_realsense_frame(args)
    shared_frame_dir = scene_dir / "shared_frame"
    shared_frame_dir.mkdir(parents=True, exist_ok=True)
    provider.save_sam6d_input_frame(frame, shared_frame_dir)

    boxes = _select_manual_bboxes(frame, object_names, scene_dir)
    results = []
    for index, name in enumerate(object_names):
        item_args = copy.copy(args)
        item_args.object_name = name
        item_args.bbox = boxes[str(name)]
        item_args.mask_mode = _manual_bbox_mask_mode(str(args.mask_mode))
        if str(item_args.mask_mode) != str(args.mask_mode):
            print(f"[jimu-sam6d-manual] using mask_mode={item_args.mask_mode} for {name} (manual bbox mode)")
        item_args.sam3_full_scene_keep_multi_instances = False
        item_args.sam3_full_scene_result_json = ""
        safe_name = provider.normalize_object_name(name) or provider._safe_cache_name(name)
        item_dir = scene_dir / f"{index + 1:02d}_{safe_name}"
        try:
            result = provider._run_single_object_pose(item_args, frame, sam6d_root, item_dir, detector_cache={})
            result["ok"] = True
            result["manual_bbox_xyxy"] = boxes[str(name)]
        except Exception as exc:
            result = {"object_name": name, "ok": False, "error": repr(exc), "run_dir": str(item_dir), "manual_bbox_xyxy": boxes[str(name)]}
            print(f"[jimu-sam6d-manual] object={name} failed: {exc!r}")
        results.append(result)

    summary = {
        "scene_dir": str(scene_dir),
        "object_count": len(object_names),
        "ok_count": sum(1 for item in results if item.get("ok")),
        "results": results,
        "manual_bboxes": boxes,
    }
    if not bool(args.skip_pem) and bool(getattr(args, "full_scene_pem_visualization", True)):
        vis_info = provider.save_full_scene_pem_visualization(args, frame, results, scene_dir / "full_scene_pem_overlay.png")
        summary["full_scene_pem_visualization"] = vis_info
        print(f"[sam6d-gdino] full-scene PEM overlay: {vis_info['path']} rendered={len(vis_info['rendered'])}")

    summary_path = scene_dir / "full_scene_pose_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[jimu-sam6d-manual] full scene result: {summary_path}")
    print(f"[sam6d-gdino] full scene result: {summary_path}")
    print(f"[sam6d-gdino] ok_count={summary['ok_count']}/{summary['object_count']}")


def main() -> None:
    register_jimu_assembly_specs()
    apriltag_config = _pop_apriltag_config()
    if _pop_apriltag_anchor_flag():
        _run_apriltag_anchor_provider(apriltag_config)
        return
    if _pop_tabletop_anchor_flag():
        _run_tabletop_anchor_provider()
        return
    if _pop_manual_bbox_flag():
        _run_manual_bbox_provider()
        return
    provider.main()


if __name__ == "__main__":
    main()
