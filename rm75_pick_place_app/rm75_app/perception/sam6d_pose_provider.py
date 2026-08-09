#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from transforms3d.euler import euler2mat

from rm75_app.assets.object_specs import get_object_spec, list_object_spec_names, normalize_object_name, resolve_object_spec_scales
from rm75_app.placement.place_rules import DESK_SLOT_LAYOUT_XZ, LocalPoseSpec, PlaceSlotSpec, get_place_rule, get_runtime_slot_specs
from rm75_app.paths import APP_ROOT, DEFAULT_CAMERA_EXTRINSIC, RUNTIME_DIR


DEFAULT_SAM6D_ROOT = "/home/zhangzhao/PycharmProjects/SAM-6D/SAM-6D"
DEFAULT_FOUNDATIONPOSE_ROOT = "/home/zhangzhao/PycharmProjects/FoundationPose"
DEFAULT_OUTPUT_ROOT = str(RUNTIME_DIR / "sam6d_groundingdino_runs")
DEFAULT_FASTSAM_MODEL_PATH = "/home/zhangzhao/PycharmProjects/SAM-6D/SAM-6D/Instance_Segmentation_Model/checkpoints/FastSAM/FastSAM-x.pt"
DEFAULT_SAM3_PYTHON = "/home/zhangzhao/anaconda3/envs/sam3/bin/python"
DEFAULT_SAM3_PROVIDER_SCRIPT = str(Path(__file__).resolve().with_name("sam3_mask_provider.py"))
DEFAULT_SAM3_CHECKPOINT_PATH = "/home/zhangzhao/Downloads/sam3.pt"
DEFAULT_CAMERA_EXTRINSIC_OPENCV_PATH = str(DEFAULT_CAMERA_EXTRINSIC)

_FASTSAM_MODEL = None
_FASTSAM_MODEL_PATH = None
_SAM6D_PEM_RUNNERS = {}
_REFINE_POINT_CACHE = {}


class SAM3FullSceneMaskRetryRequested(RuntimeError):
    pass


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


@contextlib.contextmanager
def _pushd(path: Path):
    old_cwd = os.getcwd()
    os.chdir(str(Path(path).expanduser()))
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _load_module_from_path(module_name: str, file_path: Path):
    file_path = file_path.expanduser().resolve()
    parent = str(file_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_hf_component_compatible(load_fn, model_id: str, **kwargs):
    try:
        return load_fn(model_id, **kwargs)
    except OSError as exc:
        if "Unknown scheme for proxy URL" not in str(exc):
            raise
        print("[groundingdino] unsupported proxy env detected, retrying local cache without proxy")
        proxy_keys = [
            "HF_HUB_HTTP_PROXY",
            "HF_HUB_HTTPS_PROXY",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ]
        saved_env = {key: os.environ.get(key) for key in proxy_keys if key in os.environ}
        try:
            for key in proxy_keys:
                os.environ.pop(key, None)
            retry_kwargs = dict(kwargs)
            retry_kwargs.setdefault("local_files_only", True)
            return load_fn(model_id, **retry_kwargs)
        finally:
            for key in proxy_keys:
                os.environ.pop(key, None)
            os.environ.update(saved_env)


def create_grounding_dino_detector(args):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = str(args.grounding_dino_model_id)
    print(f"[groundingdino] model={model_id} device={device}")
    offline_keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    saved_offline_env = {key: os.environ.get(key) for key in offline_keys}
    if bool(args.grounding_dino_local_files_only):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        processor = _load_hf_component_compatible(
            AutoProcessor.from_pretrained,
            model_id,
            use_fast=False,
            local_files_only=bool(args.grounding_dino_local_files_only),
        )
        model = _load_hf_component_compatible(
            AutoModelForZeroShotObjectDetection.from_pretrained,
            model_id,
            local_files_only=bool(args.grounding_dino_local_files_only),
        ).to(device)
    finally:
        for key, value in saved_offline_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    model.eval()
    return {"processor": processor, "model": model, "device": device}


def run_grounding_dino_detector(detector, image_rgb: np.ndarray, text_prompt: str, box_threshold: float, text_threshold: float):
    from PIL import Image

    image = Image.fromarray(image_rgb)
    processor = detector["processor"]
    model = detector["model"]
    device = detector["device"]

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=float(box_threshold),
        text_threshold=float(text_threshold),
        target_sizes=[image.size[::-1]],
    )
    result = results[0]
    detections = []
    for idx, (box, score) in enumerate(zip(result["boxes"], result["scores"])):
        labels = result.get("text_labels", [])
        label = labels[idx] if idx < len(labels) else ""
        if isinstance(label, (list, tuple)):
            label = ". ".join(str(x) for x in label if str(x).strip())
        detections.append(
            {
                "box": np.asarray(box.detach().cpu().tolist(), dtype=np.float32),
                "score": float(score.detach().cpu().item()),
                "label": str(label).strip(),
            }
        )
    detections.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return detections


def release_grounding_dino_detector_gpu(detector_cache: dict | None = None, detector: dict | None = None):
    detectors = []
    if detector is not None:
        detectors.append(detector)
    if detector_cache is not None and detector_cache.get("grounding_dino") is not None:
        detectors.append(detector_cache.pop("grounding_dino"))
    seen = set()
    for item in detectors:
        if not isinstance(item, dict):
            continue
        key = id(item)
        if key in seen:
            continue
        seen.add(key)
        model = item.get("model")
        if model is not None and hasattr(model, "to"):
            model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_scene_mesh(mesh_file: str, mesh_scale: float) -> trimesh.Trimesh:
    loaded = trimesh.load(Path(mesh_file).expanduser(), force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
    elif isinstance(loaded, trimesh.Scene):
        if hasattr(loaded, "to_geometry"):
            mesh = loaded.to_geometry()
        else:
            mesh = loaded.dump(concatenate=True)
        if mesh is None:
            raise ValueError(f"no mesh geometry found in {mesh_file}")
    else:
        raise TypeError(f"unsupported mesh type: {type(loaded)}")
    mesh.apply_scale(float(mesh_scale))
    mesh.remove_unreferenced_vertices()
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
    return mesh


def export_cad_mm(mesh_file: str, mesh_scale: float, out_path: Path) -> Path:
    mesh = _load_scene_mesh(mesh_file, mesh_scale)
    mesh.apply_scale(1000.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)
    return out_path


def resolve_object_inputs(args):
    object_name = normalize_object_name(args.object_name)
    spec = get_object_spec(object_name)
    if spec is None and not args.mesh_file:
        raise ValueError("--object-name must be a known object spec unless --mesh-file is provided")
    if spec is not None:
        spec_mesh_scale, _ = resolve_object_spec_scales(spec)
        mesh_file = str(Path(args.mesh_file or spec.mesh_file).expanduser())
        mesh_scale = float(args.mesh_scale if args.mesh_scale is not None else spec_mesh_scale)
        prompt = str(args.prompt or spec.grounding_prompt)
        object_name = normalize_object_name(spec.name) or spec.name
    else:
        mesh_file = str(Path(args.mesh_file).expanduser())
        mesh_scale = float(args.mesh_scale if args.mesh_scale is not None else 1.0)
        prompt = str(args.prompt or args.object_name or Path(mesh_file).stem)
        object_name = object_name or Path(mesh_file).stem
    return object_name, prompt, mesh_file, mesh_scale


def capture_realsense_frame(args):
    foundationpose_root = Path(args.foundationpose_root).expanduser().resolve()
    rt = _load_module_from_path("foundationpose_realsense_bridge_for_sam6d", foundationpose_root / "run_realtime_demo.py")
    reader = rt.RealSenseRGBDReader(width=args.camera_width, height=args.camera_height, fps=args.camera_fps)
    if args.camera_serial:
        reader.config.enable_device(args.camera_serial)
    try:
        reader.start()

        def _get_frame_with_timeout_retry():
            try:
                return reader.get_frame()
            except RuntimeError as exc:
                if "Frame didn't arrive" in str(exc):
                    return None
                raise

        timeout_retries = max(1, int(getattr(args, "camera_frame_timeout_retries", 3)))
        timeout_count = 0
        for _ in range(max(int(args.warmup_frames), 0)):
            warmup_frame = _get_frame_with_timeout_retry()
            if warmup_frame is None:
                timeout_count += 1
                if timeout_count >= timeout_retries:
                    break
                continue
            timeout_count = 0
        frame = None
        timeout_count = 0
        for _ in range(60):
            frame = _get_frame_with_timeout_retry()
            if frame is not None:
                break
            timeout_count += 1
            if timeout_count >= timeout_retries:
                break
        if frame is None:
            raise RuntimeError("failed to capture a RealSense RGB-D frame")
        return {
            "rgb": np.asarray(frame["color"], dtype=np.uint8),
            "bgr": np.asarray(frame["color_bgr"], dtype=np.uint8),
            "depth_m": np.asarray(frame["depth"], dtype=np.float32),
            "K": np.asarray(frame["K"], dtype=np.float32).reshape(3, 3),
        }
    finally:
        with contextlib.suppress(Exception):
            reader.stop()
        for attr in ("pipeline", "config", "align", "profile"):
            with contextlib.suppress(Exception):
                setattr(reader, attr, None)
        del reader
        gc.collect()


def load_offline_frame(args):
    rgb_bgr = cv2.imread(str(Path(args.rgb_path).expanduser()), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(args.rgb_path)
    depth_raw = cv2.imread(str(Path(args.depth_path).expanduser()), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(args.depth_path)
    with open(Path(args.camera_path).expanduser(), "r") as f:
        cam_info = json.load(f)
    K = np.asarray(cam_info["cam_K"], dtype=np.float32).reshape(3, 3)
    depth_scale = float(cam_info.get("depth_scale", 1.0))
    depth_m = np.asarray(depth_raw, dtype=np.float32) * depth_scale / 1000.0
    return {
        "rgb": cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
        "bgr": rgb_bgr,
        "depth_m": depth_m,
        "K": K,
    }


def save_sam6d_input_frame(frame, run_dir: Path):
    rgb_path = run_dir / "rgb.png"
    depth_path = run_dir / "depth.png"
    cam_path = run_dir / "camera.json"
    depth_mm = np.asarray(np.round(frame["depth_m"] * 1000.0), dtype=np.uint16)
    cv2.imwrite(str(rgb_path), frame["bgr"])
    cv2.imwrite(str(depth_path), depth_mm)
    with open(cam_path, "w") as f:
        json.dump({"cam_K": np.asarray(frame["K"], dtype=float).reshape(-1).tolist(), "depth_scale": 1.0}, f, indent=2)
    return rgb_path, depth_path, cam_path


def repair_depth_for_mask_if_needed(frame: dict, mask: np.ndarray, box_xyxy, args, depth_path: Path) -> dict:
    if not bool(getattr(args, "repair_mask_depth", True)):
        return {"applied": False, "reason": "disabled"}
    depth = np.asarray(frame["depth_m"], dtype=np.float32).copy()
    mask_bool = np.asarray(mask > 0, dtype=bool)
    mask_pixels = int(np.count_nonzero(mask_bool))
    if mask_pixels <= 0:
        return {"applied": False, "reason": "empty_mask"}
    min_depth = float(getattr(args, "min_valid_depth_m", 0.05))
    max_depth = float(getattr(args, "max_valid_depth_m", 2.0))
    valid_depth = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    mask_valid = valid_depth & mask_bool
    valid_count = int(np.count_nonzero(mask_valid))
    valid_ratio = float(valid_count / max(mask_pixels, 1))
    min_valid_pixels = int(getattr(args, "repair_mask_depth_min_valid_pixels", 128))
    min_valid_ratio = float(getattr(args, "repair_mask_depth_min_valid_ratio", 0.08))
    if valid_count >= min_valid_pixels and valid_ratio >= min_valid_ratio:
        return {
            "applied": False,
            "reason": "enough_valid_depth",
            "mask_pixels": mask_pixels,
            "valid_count": valid_count,
            "valid_ratio": valid_ratio,
        }

    height, width = depth.shape[:2]
    if box_xyxy is None:
        ys, xs = np.where(mask_bool)
        if xs.size <= 0 or ys.size <= 0:
            return {"applied": False, "reason": "empty_mask_bbox"}
        box = np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
    else:
        box = _clip_box_xyxy(box_xyxy, width, height, pad_frac=0.0)
    x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]

    values: list[np.ndarray] = []
    if valid_count >= 16:
        values.append(depth[mask_valid])
    max_search_px = int(getattr(args, "repair_mask_depth_search_px", 180))
    for pad in (20, 50, 100, max_search_px):
        pad = max(int(pad), 0)
        xx1, yy1 = max(0, x1 - pad), max(0, y1 - pad)
        xx2, yy2 = min(width, x2 + pad), min(height, y2 + pad)
        roi = depth[yy1:yy2, xx1:xx2]
        roi_valid = np.isfinite(roi) & (roi > min_depth) & (roi < max_depth)
        if np.count_nonzero(roi_valid) >= 32:
            values.append(roi[roi_valid])
            break
    if not values:
        return {
            "applied": False,
            "reason": "no_reference_depth",
            "mask_pixels": mask_pixels,
            "valid_count": valid_count,
            "valid_ratio": valid_ratio,
        }

    reference = np.concatenate([np.asarray(item, dtype=np.float32).reshape(-1) for item in values])
    fill_depth = float(np.median(reference))
    fill_depth -= float(getattr(args, "repair_mask_depth_surface_offset_m", 0.006))
    fill_depth = float(np.clip(fill_depth, min_depth, max_depth))
    depth[mask_bool] = fill_depth
    frame["depth_m"] = depth
    depth_mm = np.asarray(np.round(depth * 1000.0), dtype=np.uint16)
    cv2.imwrite(str(depth_path), depth_mm)
    return {
        "applied": True,
        "reason": "low_valid_depth",
        "mask_pixels": mask_pixels,
        "valid_count_before": valid_count,
        "valid_ratio_before": valid_ratio,
        "fill_depth_m": fill_depth,
        "reference_count": int(reference.size),
        "depth_path": str(depth_path),
    }


def _clip_box_xyxy(box, width: int, height: int, pad_frac: float = 0.0) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32).reshape(-1)[:4]
    bw = max(float(x2 - x1), 1.0)
    bh = max(float(y2 - y1), 1.0)
    pad_x = bw * float(pad_frac)
    pad_y = bh * float(pad_frac)
    x1 = np.clip(x1 - pad_x, 0, width - 1)
    y1 = np.clip(y1 - pad_y, 0, height - 1)
    x2 = np.clip(x2 + pad_x, x1 + 1, width)
    y2 = np.clip(y2 + pad_y, y1 + 1, height)
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.asarray(mask > 0, dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_label = int(np.argmax(areas)) + 1
    return labels == keep_label


def _make_depth_cluster_mask(depth_m: np.ndarray, box_xyxy, args) -> tuple[np.ndarray, str, float | None]:
    height, width = depth_m.shape[:2]
    box = _clip_box_xyxy(box_xyxy, width, height, pad_frac=float(args.bbox_pad_frac))
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    mask = np.zeros((height, width), dtype=bool)

    crop_depth = np.asarray(depth_m[y1:y2, x1:x2], dtype=np.float32)
    valid = np.isfinite(crop_depth) & (crop_depth > float(args.min_valid_depth_m)) & (crop_depth < float(args.max_valid_depth_m))
    if int(np.count_nonzero(valid)) < int(args.min_mask_area):
        mask[y1:y2, x1:x2] = True
        return mask, "box_fallback_no_depth", None

    valid_depths = crop_depth[valid]
    anchor_depth = float(np.percentile(valid_depths, float(args.depth_anchor_percentile)))
    band = float(args.depth_cluster_band_m)
    crop_mask = valid & (crop_depth >= anchor_depth - float(args.depth_cluster_back_margin_m)) & (crop_depth <= anchor_depth + band)
    kernel_size = max(int(args.mask_morph_kernel), 0)
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        crop_mask = cv2.morphologyEx(crop_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        crop_mask = cv2.morphologyEx(crop_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    crop_mask = _largest_component(crop_mask)
    if int(np.count_nonzero(crop_mask)) < int(args.min_mask_area):
        mask[y1:y2, x1:x2] = True
        return mask, "box_fallback_small_component", anchor_depth
    mask[y1:y2, x1:x2] = crop_mask
    return mask, f"depth_near_percentile_{float(args.depth_anchor_percentile):.1f}", anchor_depth


def make_depth_bbox_mask(depth_m: np.ndarray, box_xyxy, args) -> tuple[np.ndarray, str]:
    mask, source, _ = _make_depth_cluster_mask(depth_m, box_xyxy, args)
    return mask, source


def _make_box_mask(shape: tuple[int, int], box_xyxy, pad_frac: float = 0.0) -> np.ndarray:
    height, width = shape
    box = _clip_box_xyxy(box_xyxy, width, height, pad_frac=pad_frac)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def _shrink_box_xyxy(box_xyxy, width: int, height: int, frac: float) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=np.float32).reshape(-1)[:4]
    bw = max(float(x2 - x1), 1.0)
    bh = max(float(y2 - y1), 1.0)
    shrink_x = bw * float(frac)
    shrink_y = bh * float(frac)
    return _clip_box_xyxy([x1 + shrink_x, y1 + shrink_y, x2 - shrink_x, y2 - shrink_y], width, height, pad_frac=0.0)


def _ellipse_seed_mask(shape: tuple[int, int], box_xyxy, scale_x: float, scale_y: float) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=np.float32).reshape(-1)[:4]
    cx = int(round((float(x1) + float(x2)) * 0.5))
    cy = int(round((float(y1) + float(y2)) * 0.5))
    axis_x = max(int(round(max(float(x2 - x1), 1.0) * float(scale_x) * 0.5)), 1)
    axis_y = max(int(round(max(float(y2 - y1), 1.0) * float(scale_y) * 0.5)), 1)
    seed = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(seed, (cx, cy), (axis_x, axis_y), 0.0, 0.0, 360.0, 1, thickness=-1)
    return seed.astype(bool)


def _shape_smooth_mask(mask: np.ndarray, args) -> np.ndarray:
    mask_u8 = np.asarray(mask > 0, dtype=np.uint8)
    if int(np.count_nonzero(mask_u8)) < int(args.min_mask_area):
        return mask_u8.astype(bool)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_u8.astype(bool)
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < float(args.min_mask_area):
        return mask_u8.astype(bool)
    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rect
    long_axis = max(float(rw), float(rh))
    short_axis = max(min(float(rw), float(rh)), 1.0)
    aspect = long_axis / short_axis
    if aspect < float(args.hybrid_shape_smooth_min_aspect):
        smoothed = np.zeros_like(mask_u8)
        cv2.drawContours(smoothed, [contour], -1, 1, thickness=-1)
        return smoothed.astype(bool)

    if rw < rh:
        rect = ((cx, cy), (max(float(rw), 1.0) * float(args.hybrid_shape_short_axis_scale), float(rh) * float(args.hybrid_shape_long_axis_scale)), angle)
    else:
        rect = ((cx, cy), (float(rw) * float(args.hybrid_shape_long_axis_scale), max(float(rh), 1.0) * float(args.hybrid_shape_short_axis_scale)), angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    rect_mask = np.zeros_like(mask_u8)
    cv2.fillConvexPoly(rect_mask, box, 1)
    return rect_mask.astype(bool)


def make_grabcut_depth_mask(image_bgr: np.ndarray, depth_m: np.ndarray, box_xyxy, args) -> tuple[np.ndarray, str]:
    height, width = depth_m.shape[:2]
    image_bgr = np.asarray(image_bgr, dtype=np.uint8)
    padded_box = _clip_box_xyxy(box_xyxy, width, height, pad_frac=float(args.bbox_pad_frac))
    x1, y1, x2, y2 = [int(round(v)) for v in padded_box]
    if x2 <= x1 or y2 <= y1:
        return _make_box_mask((height, width), box_xyxy), "box_fallback_empty_box"

    depth_mask, depth_source, _ = _make_depth_cluster_mask(depth_m, box_xyxy, args)
    box_mask = _make_box_mask((height, width), padded_box)
    inner_box = _shrink_box_xyxy(box_xyxy, width, height, float(args.grabcut_inner_shrink_frac))
    ix1, iy1, ix2, iy2 = [int(round(v)) for v in inner_box]

    gc_mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[box_mask] = cv2.GC_PR_BGD
    if ix2 > ix1 and iy2 > iy1:
        gc_mask[iy1:iy2, ix1:ix2] = cv2.GC_PR_FGD

    core_seed = _ellipse_seed_mask(
        (height, width),
        box_xyxy,
        float(args.grabcut_core_seed_scale_x),
        float(args.grabcut_core_seed_scale_y),
    )
    core_seed &= box_mask
    if int(np.count_nonzero(core_seed)) >= 4:
        gc_mask[core_seed] = cv2.GC_FGD

    depth_fg = np.asarray(depth_mask & box_mask, dtype=np.uint8)
    if int(np.count_nonzero(depth_fg)) >= int(args.min_mask_area):
        valid_depth = np.isfinite(depth_m) & (depth_m > float(args.min_valid_depth_m)) & (depth_m < float(args.max_valid_depth_m))
        gc_mask[box_mask & valid_depth & ~depth_mask] = cv2.GC_PR_BGD
        depth_area_ratio = float(np.count_nonzero(depth_fg)) / max(float(np.count_nonzero(box_mask)), 1.0)
        if depth_area_ratio <= float(args.grabcut_depth_seed_max_box_fill_ratio):
            erode_k = max(int(args.grabcut_depth_fg_erode_kernel), 0)
            if erode_k > 1:
                kernel = np.ones((erode_k, erode_k), dtype=np.uint8)
                seed = cv2.erode(depth_fg, kernel, iterations=1).astype(bool)
                if int(np.count_nonzero(seed)) < int(args.min_mask_area):
                    seed = depth_fg.astype(bool)
            else:
                seed = depth_fg.astype(bool)
            gc_mask[seed] = cv2.GC_FGD

    try:
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(
            image_bgr,
            gc_mask,
            None,
            bgd_model,
            fgd_model,
            max(int(args.grabcut_iter), 1),
            cv2.GC_INIT_WITH_MASK,
        )
        mask = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
        mask &= box_mask
    except Exception as exc:
        print(f"[warn] grabCut failed, falling back to depth mask: {exc}")
        mask = depth_mask

    kernel_size = max(int(args.mask_morph_kernel), 0)
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    mask = _largest_component(mask)
    if int(np.count_nonzero(mask)) < int(args.min_mask_area):
        return depth_mask, f"grabcut_fallback_{depth_source}"
    return mask, f"grabcut_depth_{depth_source}"


def make_hybrid_depth_color_mask(image_bgr: np.ndarray, depth_m: np.ndarray, box_xyxy, args) -> tuple[np.ndarray, str]:
    height, width = depth_m.shape[:2]
    image_bgr = np.asarray(image_bgr, dtype=np.uint8)
    depth_plane_mask, depth_table_mask, depth_source = _make_depth_plane_foreground_table_masks(depth_m, box_xyxy, args)
    box_mask = _make_box_mask((height, width), box_xyxy, pad_frac=float(args.bbox_pad_frac))
    if int(np.count_nonzero(depth_plane_mask)) < int(args.min_mask_area):
        return depth_plane_mask, f"hybrid_fallback_{depth_source}"

    gc_mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[box_mask] = cv2.GC_PR_BGD

    valid_depth = np.isfinite(depth_m) & (depth_m > float(args.min_valid_depth_m)) & (depth_m < float(args.max_valid_depth_m))
    gc_mask[box_mask & valid_depth & ~depth_plane_mask] = cv2.GC_PR_BGD

    sure_fg = np.asarray(depth_plane_mask, dtype=np.uint8)
    erode_k = max(int(args.hybrid_depth_fg_erode_kernel), 0)
    if erode_k > 1:
        kernel = np.ones((erode_k, erode_k), dtype=np.uint8)
        eroded = cv2.erode(sure_fg, kernel, iterations=1)
        if int(np.count_nonzero(eroded)) >= int(args.min_mask_area):
            sure_fg = eroded
    sure_fg = sure_fg.astype(bool)
    gc_mask[sure_fg] = cv2.GC_FGD

    dilate_k = max(int(args.hybrid_depth_possible_dilate_kernel), 0)
    if dilate_k > 1:
        kernel = np.ones((dilate_k, dilate_k), dtype=np.uint8)
        possible = cv2.dilate(np.asarray(depth_plane_mask, dtype=np.uint8), kernel, iterations=1).astype(bool)
        gc_mask[box_mask & possible & ~sure_fg] = cv2.GC_PR_FGD

    try:
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(
            image_bgr,
            gc_mask,
            None,
            bgd_model,
            fgd_model,
            max(int(args.hybrid_grabcut_iter), 1),
            cv2.GC_INIT_WITH_MASK,
        )
        mask = ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)) & box_mask
    except Exception as exc:
        print(f"[warn] hybrid grabCut failed, falling back to depth plane: {exc}")
        mask = depth_plane_mask

    kernel_size = max(int(args.hybrid_morph_kernel), 0)
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)

    if bool(args.hybrid_table_veto):
        keep_kernel = max(int(args.hybrid_table_veto_keep_dilate_kernel), 0)
        if keep_kernel > 1:
            kernel = np.ones((keep_kernel, keep_kernel), dtype=np.uint8)
            near_depth_fg = cv2.dilate(np.asarray(depth_plane_mask, dtype=np.uint8), kernel, iterations=1).astype(bool)
        else:
            near_depth_fg = depth_plane_mask
        color_keep = _lab_color_support_mask(
            image_bgr,
            seed_mask=sure_fg,
            region_mask=box_mask,
            max_dist=float(args.hybrid_table_veto_color_dist),
        )
        # Depth is used here as a table/shadow veto, not as a strict object foreground.
        # Reflective or missing-depth object pixels can survive through RGB support.
        mask &= (~depth_table_mask) | near_depth_fg | color_keep

    mask = _largest_component(mask)
    if bool(args.hybrid_shape_smooth):
        mask = _shape_smooth_mask(mask, args) & box_mask
    if int(np.count_nonzero(mask)) < int(args.min_mask_area):
        return depth_plane_mask, f"hybrid_fallback_small_{depth_source}"
    return mask, f"hybrid_depth_color_{depth_source}"


def _fit_depth_plane(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, trim_m: float, iters: int) -> np.ndarray | None:
    xs = np.asarray(xs, dtype=np.float64).reshape(-1)
    ys = np.asarray(ys, dtype=np.float64).reshape(-1)
    zs = np.asarray(zs, dtype=np.float64).reshape(-1)
    keep = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
    if int(np.count_nonzero(keep)) < 3:
        return None
    xs, ys, zs = xs[keep], ys[keep], zs[keep]
    coef = None
    active = np.ones_like(zs, dtype=bool)
    for _ in range(max(int(iters), 1)):
        if int(np.count_nonzero(active)) < 3:
            break
        A = np.stack([xs[active], ys[active], np.ones(int(np.count_nonzero(active)), dtype=np.float64)], axis=1)
        coef = np.linalg.lstsq(A, zs[active], rcond=None)[0]
        residual = zs - (coef[0] * xs + coef[1] * ys + coef[2])
        med = float(np.median(residual[active]))
        mad = float(np.median(np.abs(residual[active] - med))) + 1e-6
        limit = max(float(trim_m), 3.0 * mad)
        active = np.abs(residual - med) <= limit
    return coef


def _make_depth_plane_foreground_table_masks(depth_m: np.ndarray, box_xyxy, args) -> tuple[np.ndarray, np.ndarray, str]:
    height, width = depth_m.shape[:2]
    empty_table = np.zeros((height, width), dtype=bool)
    original_box = _clip_box_xyxy(box_xyxy, width, height, pad_frac=0.0)
    padded_box = _clip_box_xyxy(box_xyxy, width, height, pad_frac=float(args.depth_plane_ring_pad_frac))
    ox1, oy1, ox2, oy2 = [int(round(v)) for v in original_box]
    px1, py1, px2, py2 = [int(round(v)) for v in padded_box]
    if ox2 <= ox1 or oy2 <= oy1 or px2 <= px1 or py2 <= py1:
        return _make_box_mask((height, width), box_xyxy), empty_table, "box_fallback_empty_box"

    roi = np.asarray(depth_m[py1:py2, px1:px2], dtype=np.float32)
    valid = np.isfinite(roi) & (roi > float(args.min_valid_depth_m)) & (roi < float(args.max_valid_depth_m))
    ring = valid.copy()
    rx1, ry1, rx2, ry2 = ox1 - px1, oy1 - py1, ox2 - px1, oy2 - py1
    ring[max(0, ry1):min(roi.shape[0], ry2), max(0, rx1):min(roi.shape[1], rx2)] = False
    if int(np.count_nonzero(ring)) < int(args.depth_plane_min_ring_pixels):
        fallback, source = make_depth_bbox_mask(depth_m, box_xyxy, args)
        return fallback, empty_table, f"depth_plane_fallback_{source}"

    ys, xs = np.where(ring)
    zs = roi[ys, xs]
    coef = _fit_depth_plane(
        xs,
        ys,
        zs,
        trim_m=float(args.depth_plane_fit_trim_m),
        iters=int(args.depth_plane_fit_iters),
    )
    if coef is None:
        fallback, source = make_depth_bbox_mask(depth_m, box_xyxy, args)
        return fallback, empty_table, f"depth_plane_fallback_{source}"

    yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
    plane = coef[0] * xx + coef[1] * yy + coef[2]
    inside = np.zeros_like(valid, dtype=bool)
    inside[max(0, ry1):min(roi.shape[0], ry2), max(0, rx1):min(roi.shape[1], rx2)] = True
    crop_mask = inside & valid & (roi < (plane - float(args.depth_plane_foreground_margin_m)))
    table_margin = float(getattr(args, "hybrid_table_veto_margin_m", args.depth_plane_foreground_margin_m))
    crop_table = inside & valid & (roi >= (plane - table_margin))

    kernel_size = max(int(args.mask_morph_kernel), 0)
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        crop_mask = cv2.morphologyEx(crop_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        crop_mask = cv2.morphologyEx(crop_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    crop_mask = _largest_component(crop_mask)
    if int(np.count_nonzero(crop_mask)) < int(args.min_mask_area):
        fallback, source = make_depth_bbox_mask(depth_m, box_xyxy, args)
        return fallback, empty_table, f"depth_plane_fallback_small_{source}"

    mask = np.zeros((height, width), dtype=bool)
    mask[py1:py2, px1:px2] = crop_mask
    table_mask = np.zeros((height, width), dtype=bool)
    table_mask[py1:py2, px1:px2] = crop_table
    return mask, table_mask, f"depth_plane_margin_{float(args.depth_plane_foreground_margin_m):.3f}"


def make_depth_plane_mask(depth_m: np.ndarray, box_xyxy, args) -> tuple[np.ndarray, str]:
    mask, _, source = _make_depth_plane_foreground_table_masks(depth_m, box_xyxy, args)
    return mask, source


def _lab_color_support_mask(
    image_bgr: np.ndarray,
    seed_mask: np.ndarray,
    region_mask: np.ndarray,
    max_dist: float,
) -> np.ndarray:
    seed = np.asarray(seed_mask > 0, dtype=bool) & np.asarray(region_mask > 0, dtype=bool)
    region = np.asarray(region_mask > 0, dtype=bool)
    if int(np.count_nonzero(seed)) < 8:
        return np.zeros(region.shape, dtype=bool)

    lab = cv2.cvtColor(np.asarray(image_bgr, dtype=np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    samples = lab[seed]
    if samples.shape[0] > 2000:
        step = max(samples.shape[0] // 2000, 1)
        samples = samples[::step]
    cluster_count = min(3, int(samples.shape[0]))
    if cluster_count <= 1:
        centers = samples[:1]
    else:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
        _, _, centers = cv2.kmeans(
            samples.astype(np.float32),
            cluster_count,
            None,
            criteria,
            2,
            cv2.KMEANS_PP_CENTERS,
        )
    diff = lab[:, :, None, :] - centers.reshape(1, 1, -1, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=3)).min(axis=2)
    return region & (dist <= float(max_dist))


def _mask_bbox_xyxy(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(np.asarray(mask > 0, dtype=bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _bbox_iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in np.asarray(a, dtype=np.float32).reshape(-1)[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in np.asarray(b, dtype=np.float32).reshape(-1)[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / max(area_a + area_b - inter, 1e-6))


def _get_fastsam_model(model_path: str):
    global _FASTSAM_MODEL, _FASTSAM_MODEL_PATH
    model_path = str(Path(model_path).expanduser())
    if _FASTSAM_MODEL is not None and _FASTSAM_MODEL_PATH == model_path:
        return _FASTSAM_MODEL
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from ultralytics import YOLO

    _FASTSAM_MODEL = YOLO(model_path)
    _FASTSAM_MODEL_PATH = model_path
    return _FASTSAM_MODEL


def make_fastsam_bbox_mask(image_bgr: np.ndarray, box_xyxy, args) -> tuple[np.ndarray, str]:
    height, width = image_bgr.shape[:2]
    model_path = Path(args.fastsam_model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"FastSAM checkpoint not found: {model_path}")

    target_box = _clip_box_xyxy(box_xyxy, width, height, pad_frac=0.0)
    target_region = _make_box_mask((height, width), target_box, pad_frac=float(args.fastsam_select_pad_frac))
    target_area = max(float(np.count_nonzero(target_region)), 1.0)

    model = _get_fastsam_model(str(model_path))
    device = 0 if torch.cuda.is_available() else "cpu"
    results = model.predict(
        source=cv2.cvtColor(np.asarray(image_bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB),
        device=device,
        imgsz=int(args.fastsam_imgsz),
        conf=float(args.fastsam_conf),
        iou=float(args.fastsam_iou),
        max_det=int(args.fastsam_max_det),
        retina_masks=True,
        verbose=False,
    )
    if not results or results[0].masks is None:
        return _make_box_mask((height, width), box_xyxy, pad_frac=float(args.bbox_pad_frac)), "fastsam_fallback_no_masks_box"

    masks_t = results[0].masks.data
    boxes_t = results[0].boxes.xyxy if results[0].boxes is not None else None
    if masks_t.ndim != 3 or int(masks_t.shape[0]) <= 0:
        return _make_box_mask((height, width), box_xyxy, pad_frac=float(args.bbox_pad_frac)), "fastsam_fallback_empty_box"

    masks_np = masks_t.detach().float().cpu().numpy() > 0.5
    if masks_np.shape[1:3] != (height, width):
        resized = []
        for item in masks_np:
            resized.append(cv2.resize(item.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool))
        masks_np = np.stack(resized, axis=0)
    boxes_np = boxes_t.detach().float().cpu().numpy() if boxes_t is not None else None

    best_mask = None
    best_score = -1e9
    best_idx = -1
    best_meta = None
    padded_box = _clip_box_xyxy(target_box, width, height, pad_frac=float(args.fastsam_crop_pad_frac))
    padded_region = _make_box_mask((height, width), padded_box, pad_frac=0.0)
    for idx, raw_mask in enumerate(masks_np):
        mask = np.asarray(raw_mask > 0, dtype=bool)
        mask_area = float(np.count_nonzero(mask))
        if mask_area < float(args.min_mask_area):
            continue
        overlap = float(np.count_nonzero(mask & target_region))
        if overlap <= 0.0:
            continue
        mask_box = boxes_np[idx] if boxes_np is not None and idx < len(boxes_np) else _mask_bbox_xyxy(mask)
        if mask_box is None:
            continue
        iou = _bbox_iou_xyxy(mask_box, target_box)
        box_coverage = overlap / target_area
        mask_in_box = overlap / max(mask_area, 1.0)
        outside = float(np.count_nonzero(mask & ~padded_region)) / max(mask_area, 1.0)
        score = (2.2 * box_coverage) + (1.6 * mask_in_box) + (0.8 * iou) - (0.8 * outside)
        if score > best_score:
            best_score = score
            best_mask = mask
            best_idx = idx
            best_meta = (box_coverage, mask_in_box, iou, outside, mask_area)

    if best_mask is None:
        return _make_box_mask((height, width), box_xyxy, pad_frac=float(args.bbox_pad_frac)), "fastsam_fallback_no_overlap_box"

    mask = best_mask & padded_region
    kernel_size = max(int(args.fastsam_morph_kernel), 0)
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    mask = _largest_component(mask)
    if int(np.count_nonzero(mask)) < int(args.min_mask_area):
        return best_mask & padded_region, "fastsam_small_selected"

    cov, inbox, iou, outside, area = best_meta
    return (
        mask,
        f"fastsam_bbox_idx_{best_idx}_score_{best_score:.3f}_cov_{cov:.3f}_inbox_{inbox:.3f}_iou_{iou:.3f}_outside_{outside:.3f}_area_{area:.0f}",
    )


def _sam3_subprocess_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(APP_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    # huggingface_hub/httpx rejects the local "socks://" scheme. HTTP_PROXY is enough here.
    for key in ("ALL_PROXY", "all_proxy"):
        if str(env.get(key, "")).lower().startswith("socks://"):
            env.pop(key, None)
    return env


def _sam3_result_bbox(item: dict) -> np.ndarray | None:
    box = item.get("mask_bbox")
    if box is None:
        box = item.get("selected", {}).get("box") if isinstance(item.get("selected"), dict) else None
    if box is None:
        return None
    try:
        return np.asarray(box, dtype=np.float32).reshape(4)
    except Exception:
        return None


def _bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in np.asarray(a, dtype=np.float32).reshape(4)]
    bx1, by1, bx2, by2 = [float(v) for v in np.asarray(b, dtype=np.float32).reshape(4)]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / max(area_a + area_b - inter, 1e-6))


def _dedupe_sam3_instances(items: list[dict], max_items: int) -> list[dict]:
    kept: list[dict] = []
    kept_boxes: list[np.ndarray] = []
    for item in sorted(items, key=lambda entry: float(entry.get("selected", {}).get("model_score", 0.0)), reverse=True):
        box = _sam3_result_bbox(item)
        if box is not None and any(_bbox_iou_xyxy(box, old_box) > 0.72 for old_box in kept_boxes):
            continue
        kept.append(item)
        if box is not None:
            kept_boxes.append(box)
        if len(kept) >= max_items:
            break
    return kept


def _sam3_mode_from_mask_mode(mask_mode: str) -> str:
    if mask_mode == "sam3_text":
        return "text"
    if mask_mode == "sam3_bbox":
        return "box"
    if mask_mode == "sam3_text_bbox":
        return "text_box"
    raise ValueError(f"unsupported SAM3 mask mode: {mask_mode}")


def make_sam3_mask(frame, box_xyxy, args, *, prompt: str | None, run_dir: Path | None, rgb_path: Path | None) -> tuple[np.ndarray, str]:
    if run_dir is None:
        raise ValueError("SAM3 mask mode requires run_dir")
    if rgb_path is None:
        rgb_path = Path(run_dir) / "rgb.png"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR))

    height, width = frame["depth_m"].shape[:2]
    box = None if box_xyxy is None else _clip_box_xyxy(box_xyxy, width, height, pad_frac=0.0)
    output_dir = Path(run_dir) / "sam3_mask"
    output_dir.mkdir(parents=True, exist_ok=True)
    sam3_python = Path(args.sam3_python).expanduser()
    sam3_script = Path(args.sam3_provider_script).expanduser()
    if not sam3_python.exists():
        raise FileNotFoundError(f"SAM3 python not found: {sam3_python}")
    if not sam3_script.exists():
        raise FileNotFoundError(f"SAM3 provider script not found: {sam3_script}")

    mode = _sam3_mode_from_mask_mode(str(args.mask_mode))
    cmd = [
        str(sam3_python),
        str(sam3_script),
        "--rgb-path",
        str(Path(rgb_path).expanduser()),
        "--output-dir",
        str(output_dir),
        "--mode",
        mode,
        "--confidence-threshold",
        str(float(args.sam3_confidence_threshold)),
        "--resolution",
        str(int(args.sam3_resolution)),
        "--min-mask-area",
        str(int(args.min_mask_area)),
        "--morph-kernel",
        str(int(args.sam3_morph_kernel)),
        "--sam3-max-masks-per-item",
        str(max(1, int(args.sam3_max_masks_per_item))),
    ]
    if box is not None:
        cmd += ["--select-box", *[f"{float(v):.6f}" for v in box.tolist()]]
    if mode in {"box", "text_box"}:
        if box is None:
            raise ValueError(f"{args.mask_mode} requires a bbox")
        cmd += ["--box", *[f"{float(v):.6f}" for v in box.tolist()]]
    if mode in {"text", "text_box"}:
        if not prompt:
            raise ValueError("SAM3 text mode requires a prompt")
        cmd += ["--prompt", str(prompt)]
    if args.sam3_checkpoint_path:
        cmd += ["--checkpoint-path", str(Path(args.sam3_checkpoint_path).expanduser())]
    if args.sam3_device:
        cmd += ["--device", str(args.sam3_device)]

    stdout_path = output_dir / "sam3_stdout.txt"
    stderr_path = output_dir / "sam3_stderr.txt"
    proc = subprocess.run(
        cmd,
        cwd=str(Path(sam3_script).parent),
        text=True,
        capture_output=True,
        env=_sam3_subprocess_env(),
    )
    stdout_path.write_text(proc.stdout or "")
    stderr_path.write_text(proc.stderr or "")
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-18:])
        if "GatedRepoError" in tail or "Cannot access gated repo" in tail:
            hf_cli = Path(sam3_python).parent / "hf"
            tail += (
                "\nSAM3 checkpoint is gated. Accept access for facebook/sam3, then run: "
                f"{hf_cli} auth login"
            )
        raise RuntimeError(f"SAM3 mask subprocess failed with code {proc.returncode}: {tail}")

    result_path = output_dir / "sam3_result.json"
    if not result_path.exists():
        raise RuntimeError(f"SAM3 subprocess did not write expected outputs under {output_dir}")
    with open(result_path, "r") as f:
        result = json.load(f)

    if isinstance(result, dict):
        candidates = result.get("all_candidates")
        if isinstance(candidates, list) and candidates:
            selected_result = candidates[0] if isinstance(candidates[0], dict) else result
            if not isinstance(selected_result, dict):
                selected_result = result
            mask_path = selected_result.get("mask_path")
            if isinstance(mask_path, str):
                resolved_mask_path = Path(mask_path)
                if resolved_mask_path.exists():
                    result = selected_result
                    mask_path = resolved_mask_path
                else:
                    mask_path = None
            else:
                mask_path = None
        else:
            mask_path = result.get("mask_path")
            if not isinstance(mask_path, str):
                mask_path = output_dir / "sam3_mask.png"
            else:
                mask_path = Path(mask_path)
    else:
        mask_path = output_dir / "sam3_mask.png"
    if mask_path is None or not Path(mask_path).exists():
        mask_path = output_dir / "sam3_mask.png"
        if not mask_path.exists():
            if isinstance(result, dict):
                selected_path = result.get("mask_path")
                if isinstance(selected_path, str):
                    candidate_path = Path(selected_path)
                    if candidate_path.exists():
                        mask_path = candidate_path
            if mask_path is None or not mask_path.exists():
                raise RuntimeError(f"SAM3 subprocess did not write expected mask under {output_dir}")
    mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_u8 is None:
        raise RuntimeError(f"failed to read SAM3 mask: {mask_path}")
    mask = np.asarray(mask_u8 > 0, dtype=bool)
    if int(np.count_nonzero(mask)) < int(args.min_mask_area):
        raise RuntimeError(f"SAM3 mask too small: {int(np.count_nonzero(mask))} pixels")
    selected = result.get("selected", {}) if isinstance(result, dict) else {}
    src = (
        f"{args.mask_mode}_score_{float(selected.get('model_score', 0.0)):.3f}"
        f"_cov_{float(selected.get('coverage', 0.0)):.3f}"
        f"_inbox_{float(selected.get('mask_in_box', 0.0)):.3f}"
        f"_ms_{float(result.get('infer_elapsed_ms', 0.0)):.1f}"
    )
    return mask, src


def _select_sam3_precomputed_instance(precomputed, object_name: str, instance_index: int):
    if isinstance(precomputed, list):
        if not precomputed:
            raise RuntimeError(f"SAM3 full-scene text produced no instances for {object_name}")
        index = int(instance_index)
        if index < 0:
            index = 0
        if index >= len(precomputed):
            print(
                f"[sam6d-gdino] sam3 instance index out of range for {object_name}: requested={instance_index}, "
                f"available={len(precomputed)}; using last."
            )
            index = len(precomputed) - 1
        selected = precomputed[index]
        if isinstance(selected, dict):
            selected["sam3_instance_index"] = index
        return index, selected
    return 0, precomputed


def make_detection_mask(frame, box_xyxy, args, *, prompt: str | None = None, run_dir: Path | None = None, rgb_path: Path | None = None) -> tuple[np.ndarray, str]:
    height, width = frame["depth_m"].shape[:2]
    if args.mask_mode == "box":
        return _make_box_mask((height, width), box_xyxy, pad_frac=float(args.bbox_pad_frac)), "box"
    if args.mask_mode == "fastsam_bbox":
        return make_fastsam_bbox_mask(frame["bgr"], box_xyxy, args)
    if str(args.mask_mode).startswith("sam3_"):
        return make_sam3_mask(frame, box_xyxy, args, prompt=prompt, run_dir=run_dir, rgb_path=rgb_path)
    if args.mask_mode == "hybrid_depth_color":
        return make_hybrid_depth_color_mask(frame["bgr"], frame["depth_m"], box_xyxy, args)
    if args.mask_mode == "depth_plane":
        return make_depth_plane_mask(frame["depth_m"], box_xyxy, args)
    if args.mask_mode == "depth":
        return make_depth_bbox_mask(frame["depth_m"], box_xyxy, args)
    if args.mask_mode == "grabcut_depth":
        return make_grabcut_depth_mask(frame["bgr"], frame["depth_m"], box_xyxy, args)
    raise ValueError(f"unsupported mask mode: {args.mask_mode}")


def mask_to_uncompressed_rle(binary_mask: np.ndarray) -> dict:
    mask = np.asarray(binary_mask > 0, dtype=np.uint8)
    counts = []
    last_elem = 0
    running_length = 0
    for elem in mask.ravel(order="F"):
        elem = int(elem)
        if elem != last_elem:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1
    counts.append(running_length)
    return {"counts": counts, "size": list(mask.shape)}


def make_detection_ism_entry(mask: np.ndarray, box_xyxy, score: float, label: str, **metadata):
    x1, y1, x2, y2 = [float(v) for v in np.asarray(box_xyxy, dtype=np.float32).reshape(-1)[:4]]
    det = {
        "scene_id": 0,
        "image_id": 0,
        "category_id": 1,
        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
        "score": float(score),
        "time": 0.0,
        "label": str(label),
        "segmentation": mask_to_uncompressed_rle(mask),
    }
    for key, value in metadata.items():
        if value is not None:
            det[key] = value
    return det


def write_detection_ism_entries(entries: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(entries, f)
    return entries


def write_detection_ism(mask: np.ndarray, box_xyxy, score: float, label: str, out_path: Path, **metadata):
    det = make_detection_ism_entry(mask, box_xyxy, score, label, **metadata)
    write_detection_ism_entries([det], out_path)
    return det


def save_detection_visual(frame, box_xyxy, mask: np.ndarray, out_path: Path, label: str, score: float):
    canvas = frame["bgr"].copy()
    overlay = canvas.copy()
    overlay[np.asarray(mask, dtype=bool)] = (0, 180, 255)
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0.0)
    x1, y1, x2, y2 = [int(round(v)) for v in np.asarray(box_xyxy, dtype=np.float32).reshape(-1)[:4]]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        canvas,
        f"{label} {score:.3f}",
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.imwrite(str(out_path), canvas)


def save_binary_mask(mask: np.ndarray, out_path: Path):
    cv2.imwrite(str(out_path), np.asarray(mask > 0, dtype=np.uint8) * 255)


def _template_cache_dir(args, object_name: str, mesh_file: str, mesh_scale: float) -> Path | None:
    if not args.template_cache_root:
        return None
    mesh_path = Path(mesh_file).expanduser().resolve()
    stat = mesh_path.stat()
    safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(object_name))
    key = f"{safe_name}_{mesh_path.stem}_{float(mesh_scale):.9g}_{int(stat.st_mtime_ns)}_{int(stat.st_size)}"
    return Path(args.template_cache_root).expanduser() / key


def prepare_templates(args, sam6d_root: Path, run_dir: Path, cad_path: Path, *, object_name: str, mesh_file: str, mesh_scale: float):
    dst_templates = run_dir / "templates"
    if dst_templates.exists() and list(dst_templates.glob("xyz_*.npy")):
        return dst_templates
    if args.templates_dir:
        src = Path(args.templates_dir).expanduser()
        if not src.exists():
            raise FileNotFoundError(src)
        if dst_templates.exists():
            shutil.rmtree(dst_templates)
        shutil.copytree(src, dst_templates)
        return dst_templates

    cache_dir = _template_cache_dir(args, object_name, mesh_file, mesh_scale)
    if cache_dir is not None:
        cache_templates = cache_dir / "templates"
        if cache_templates.exists() and list(cache_templates.glob("xyz_*.npy")):
            shutil.copytree(cache_templates, dst_templates)
            print(f"[sam6d] reused cached templates: {cache_templates}")
            return dst_templates

    render_script = sam6d_root / "Render" / "render_custom_templates.py"
    render_output_dir = cache_dir if cache_dir is not None else run_dir
    if render_output_dir.exists() and cache_dir is not None:
        shutil.rmtree(render_output_dir)
    render_output_dir.mkdir(parents=True, exist_ok=True)
    blenderproc_exe = Path(sys.executable).resolve().parent / "blenderproc"
    blenderproc_cmd = str(blenderproc_exe) if blenderproc_exe.exists() else (shutil.which("blenderproc") or "blenderproc")
    cmd = [
        blenderproc_cmd,
        "run",
        str(render_script),
        "--output_dir",
        str(render_output_dir),
        "--cad_path",
        str(cad_path),
    ]
    print("[sam6d] rendering templates:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(sam6d_root / "Render"), check=True)
    if cache_dir is not None:
        cache_templates = cache_dir / "templates"
        if not cache_templates.exists():
            raise RuntimeError(f"SAM-6D template rendering did not create {cache_templates}")
        shutil.copytree(cache_templates, dst_templates)
        return dst_templates
    if not dst_templates.exists():
        raise RuntimeError(f"SAM-6D template rendering did not create {dst_templates}")
    return dst_templates


def _safe_cache_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(text))


def _path_stat_record(path: Path) -> dict:
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return {"path": str(path), "mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}


def _template_dir_fingerprint(template_dir: Path) -> str:
    digest = hashlib.sha1()
    for path in sorted(Path(template_dir).glob("*")):
        if path.is_file() and path.suffix.lower() in {".png", ".npy"}:
            stat = path.stat()
            digest.update(path.name.encode("utf-8"))
            digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))
            digest.update(str(int(stat.st_size)).encode("ascii"))
    return digest.hexdigest()


def _pem_feature_cache_file(
    args,
    sam6d_root: Path,
    object_name: str,
    mesh_file: str,
    mesh_scale: float,
    template_dir: Path,
) -> Path | None:
    if bool(getattr(args, "no_pem_feature_cache", False)):
        return None
    cache_root = getattr(args, "pem_feature_cache_root", None)
    if not cache_root:
        return None
    pem_dir = Path(sam6d_root).expanduser().resolve() / "Pose_Estimation_Model"
    payload = {
        "object_name": str(object_name),
        "mesh": _path_stat_record(Path(mesh_file)),
        "mesh_scale": float(mesh_scale),
        "templates_fingerprint": _template_dir_fingerprint(template_dir),
        "config": _path_stat_record(pem_dir / "config" / "base.yaml"),
        "checkpoint": _path_stat_record(pem_dir / "checkpoints" / "sam-6d-pem-base.pth"),
        "mae_checkpoint": _path_stat_record(pem_dir / "checkpoints" / "mae_pretrain_vit_base.pth"),
        "feature_cache_version": 2,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    safe_name = _safe_cache_name(object_name)
    return Path(cache_root).expanduser() / safe_name / f"{digest}.pt"


def _torch_load_safe(path: Path, *, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_sam6d_pem_module(sam6d_root: Path):
    pem_dir = Path(sam6d_root).expanduser().resolve() / "Pose_Estimation_Model"
    extra_paths = [
        pem_dir,
        pem_dir / "provider",
        pem_dir / "utils",
        pem_dir / "model",
        pem_dir / "model" / "pointnet2",
    ]
    for item in reversed([str(p) for p in extra_paths]):
        if item in sys.path:
            sys.path.remove(item)
        sys.path.insert(0, item)
    with _pushd(pem_dir):
        return _load_module_from_path("sam6d_pem_run_inference_custom_inprocess", pem_dir / "run_inference_custom.py")


class SAM6DPEMInProcessRunner:
    def __init__(self, args, sam6d_root: Path):
        self.args = args
        self.sam6d_root = Path(sam6d_root).expanduser().resolve()
        self.pem_dir = self.sam6d_root / "Pose_Estimation_Model"
        self.pem_mod = _load_sam6d_pem_module(self.sam6d_root)
        self.cfg = self._build_cfg()
        self.model = None
        self.template_feature_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _build_cfg(self):
        cfg = self.pem_mod.gorilla.Config.fromfile(str(self.pem_dir / "config" / "base.yaml"))
        cfg.exp_name = "pose_estimation_model_base_id0"
        cfg.gpus = "0"
        cfg.model_name = "pose_estimation_model"
        cfg.log_dir = str(self.pem_dir / "log" / cfg.exp_name)
        cfg.test_iter = 600000
        cfg.det_score_thresh = float(self.args.pem_det_score_thresh)
        return cfg

    def _load_model_once(self):
        if self.model is not None:
            return self.model
        random.seed(int(self.cfg.rd_seed))
        torch.manual_seed(int(self.cfg.rd_seed))
        print("[sam6d] loading PEM model in-process")
        with _pushd(self.pem_dir):
            model_module = importlib.import_module(str(self.cfg.model_name))
            model = model_module.Net(self.cfg.model)
            model = model.cuda()
            model.eval()
            checkpoint = self.pem_dir / "checkpoints" / "sam-6d-pem-base.pth"
            self.pem_mod.gorilla.solver.load_checkpoint(model=model, filename=str(checkpoint))
        self.model = model
        return self.model

    def _get_template_features(self, template_dir: Path, feature_cache_file: Path | None):
        template_dir = Path(template_dir).expanduser().resolve()
        mem_key = str(feature_cache_file or template_dir)
        if mem_key in self.template_feature_cache:
            return self.template_feature_cache[mem_key]

        if feature_cache_file is not None and Path(feature_cache_file).exists():
            payload = _torch_load_safe(Path(feature_cache_file), map_location="cpu")
            all_tem_pts = payload["all_tem_pts"].cuda(non_blocking=True)
            all_tem_feat = payload["all_tem_feat"].cuda(non_blocking=True)
            self.template_feature_cache[mem_key] = (all_tem_pts, all_tem_feat)
            print(f"[sam6d] reused cached PEM template features: {feature_cache_file}")
            return all_tem_pts, all_tem_feat

        model = self._load_model_once()
        print("[sam6d] extracting PEM template features")
        random.seed(int(self.cfg.rd_seed))
        torch.manual_seed(int(self.cfg.rd_seed))
        with _pushd(self.pem_dir):
            all_tem, all_tem_pts_list, all_tem_choose = self.pem_mod.get_templates(str(template_dir), self.cfg.test_dataset)
            with torch.no_grad():
                all_tem_pts, all_tem_feat = model.feature_extraction.get_obj_feats(
                    all_tem,
                    all_tem_pts_list,
                    all_tem_choose,
                )
        self.template_feature_cache[mem_key] = (all_tem_pts, all_tem_feat)

        if feature_cache_file is not None:
            feature_cache_file = Path(feature_cache_file)
            feature_cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "all_tem_pts": all_tem_pts.detach().cpu(),
                    "all_tem_feat": all_tem_feat.detach().cpu(),
                    "created_at": _now_stamp(),
                },
                feature_cache_file,
            )
            print(f"[sam6d] saved PEM template features: {feature_cache_file}")
        return all_tem_pts, all_tem_feat

    def infer(
        self,
        run_dir: Path,
        cad_path: Path,
        rgb_path: Path,
        depth_path: Path,
        cam_path: Path,
        seg_path: Path,
        template_dir: Path,
        feature_cache_file: Path | None,
    ) -> Path:
        model = self._load_model_once()
        all_tem_pts, all_tem_feat = self._get_template_features(template_dir, feature_cache_file)

        random.seed(int(self.cfg.rd_seed))
        torch.manual_seed(int(self.cfg.rd_seed))
        with _pushd(self.pem_dir):
            input_data, img, _, model_points, detections = self.pem_mod.get_test_data(
                str(rgb_path),
                str(depth_path),
                str(cam_path),
                str(cad_path),
                str(seg_path),
                float(self.args.pem_det_score_thresh),
                self.cfg.test_dataset,
            )
            ninstance = input_data["pts"].size(0)
            with torch.no_grad():
                input_data["dense_po"] = all_tem_pts.repeat(ninstance, 1, 1)
                input_data["dense_fo"] = all_tem_feat.repeat(ninstance, 1, 1)
                out = model(input_data)
            torch.cuda.synchronize()

        if "pred_pose_score" in out.keys():
            pose_scores_t = out["pred_pose_score"] * out["score"]
        else:
            pose_scores_t = out["score"]
        pose_scores = pose_scores_t.detach().cpu().numpy()
        pred_rot = out["pred_R"].detach().cpu().numpy()
        pred_trans = out["pred_t"].detach().cpu().numpy() * 1000.0

        results_dir = Path(run_dir) / "sam6d_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        for idx, det in enumerate(detections):
            detections[idx]["score"] = float(pose_scores[idx])
            detections[idx]["R"] = list(pred_rot[idx].tolist())
            detections[idx]["t"] = list(pred_trans[idx].tolist())

        pem_json = results_dir / "detection_pem.json"
        with open(pem_json, "w") as f:
            json.dump(detections, f)

        if bool(getattr(self.args, "pem_save_visualization", False)):
            save_path = results_dir / "vis_pem.png"
            valid_masks = pose_scores == pose_scores.max()
            K = input_data["K"].detach().cpu().numpy()[valid_masks]
            vis_img = self.pem_mod.visualize(img, pred_rot[valid_masks], pred_trans[valid_masks], model_points * 1000, K, str(save_path))
            vis_img.save(save_path)
        return pem_json


def _get_sam6d_pem_runner(args, sam6d_root: Path) -> SAM6DPEMInProcessRunner:
    key = str(Path(sam6d_root).expanduser().resolve())
    runner = _SAM6D_PEM_RUNNERS.get(key)
    if runner is None:
        runner = SAM6DPEMInProcessRunner(args, sam6d_root)
        _SAM6D_PEM_RUNNERS[key] = runner
    return runner


def run_sam6d_pem_subprocess(args, sam6d_root: Path, run_dir: Path, cad_path: Path, rgb_path: Path, depth_path: Path, cam_path: Path, seg_path: Path):
    pem_dir = sam6d_root / "Pose_Estimation_Model"
    cmd = [
        sys.executable,
        "run_inference_custom.py",
        "--output_dir",
        str(run_dir),
        "--cad_path",
        str(cad_path),
        "--rgb_path",
        str(rgb_path),
        "--depth_path",
        str(depth_path),
        "--cam_path",
        str(cam_path),
        "--seg_path",
        str(seg_path),
    ]
    env = os.environ.copy()
    extra_paths = [
        str(pem_dir),
        str(pem_dir / "provider"),
        str(pem_dir / "utils"),
        str(pem_dir / "model"),
        str(pem_dir / "model" / "pointnet2"),
    ]
    env["PYTHONPATH"] = os.pathsep.join(extra_paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    print("[sam6d] running PEM:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(pem_dir), env=env, check=True)
    return run_dir / "sam6d_results" / "detection_pem.json"


def run_sam6d_pem(
    args,
    sam6d_root: Path,
    run_dir: Path,
    cad_path: Path,
    rgb_path: Path,
    depth_path: Path,
    cam_path: Path,
    seg_path: Path,
    *,
    template_dir: Path,
    object_name: str,
    mesh_file: str,
    mesh_scale: float,
):
    if str(getattr(args, "pem_run_mode", "inprocess")) == "subprocess":
        return run_sam6d_pem_subprocess(args, sam6d_root, run_dir, cad_path, rgb_path, depth_path, cam_path, seg_path)

    runner = _get_sam6d_pem_runner(args, sam6d_root)
    feature_cache_file = _pem_feature_cache_file(args, sam6d_root, object_name, mesh_file, mesh_scale, template_dir)
    return runner.infer(run_dir, cad_path, rgb_path, depth_path, cam_path, seg_path, template_dir, feature_cache_file)


def _pem_detection_to_transform(det: dict) -> np.ndarray:
    R = np.asarray(det["R"], dtype=np.float32).reshape(3, 3)
    t_m = np.asarray(det["t"], dtype=np.float32).reshape(3) / 1000.0
    T_cam_obj = np.eye(4, dtype=np.float32)
    T_cam_obj[:3, :3] = R
    T_cam_obj[:3, 3] = t_m
    return T_cam_obj


def parse_pem_results(pem_json: Path):
    with open(pem_json, "r") as f:
        detections = json.load(f)
    if not detections:
        raise RuntimeError(f"empty SAM-6D PEM result: {pem_json}")
    parsed = []
    for det in detections:
        if "R" not in det or "t" not in det:
            continue
        parsed.append((det, _pem_detection_to_transform(det)))
    if not parsed:
        raise RuntimeError(f"SAM-6D PEM result has no pose entries: {pem_json}")
    return parsed


def parse_pem_result(pem_json: Path):
    parsed = parse_pem_results(pem_json)
    best, T_cam_obj = max(parsed, key=lambda item: float(item[0].get("score", 0.0)))
    return best, T_cam_obj


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


def _transform_points(T: np.ndarray, pts_obj: np.ndarray) -> np.ndarray:
    pts_obj = np.asarray(pts_obj, dtype=np.float64).reshape(-1, 3)
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    pts_h = np.concatenate([pts_obj, np.ones((pts_obj.shape[0], 1), dtype=np.float64)], axis=1)
    return (T @ pts_h.T).T[:, :3]


def _mask_bbox_center(mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    ys, xs = np.where(np.asarray(mask > 0, dtype=bool))
    if xs.size == 0 or ys.size == 0:
        return None, None
    bbox = np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)
    center = np.asarray([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float64)
    return bbox, center


def _bbox_center_size(box_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    box = np.asarray(box_xyxy, dtype=np.float64).reshape(-1)[:4]
    center = np.asarray([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float64)
    size = np.asarray([max(box[2] - box[0], 1e-6), max(box[3] - box[1], 1e-6)], dtype=np.float64)
    return center, size


def _mask_bbox_touches_image_border(mask_bbox: np.ndarray | None, mask_shape: tuple[int, ...], margin_px: float) -> bool:
    if mask_bbox is None or float(margin_px) < 0.0:
        return False
    height, width = int(mask_shape[0]), int(mask_shape[1])
    x1, y1, x2, y2 = np.asarray(mask_bbox, dtype=np.float64).reshape(-1)[:4]
    margin = float(margin_px)
    return bool(x1 <= margin or y1 <= margin or x2 >= float(width) - margin or y2 >= float(height) - margin)


def _mesh_refine_points(mesh: trimesh.Trimesh, sample_count: int) -> np.ndarray:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    stat_key = (
        str(getattr(mesh, "metadata", {}).get("file_name", "")),
        tuple(np.round(bounds.reshape(-1), 8).tolist()),
        int(sample_count),
        int(len(getattr(mesh, "vertices", []))),
        int(len(getattr(mesh, "faces", []))),
    )
    cached = _REFINE_POINT_CACHE.get(stat_key)
    if cached is not None:
        return cached

    pts = []
    vertices = np.asarray(getattr(mesh, "vertices", []), dtype=np.float64).reshape(-1, 3)
    if vertices.size:
        pts.append(vertices)
    mn, mx = bounds[0], bounds[1]
    corners = np.asarray(
        [[x, y, z] for x in [mn[0], mx[0]] for y in [mn[1], mx[1]] for z in [mn[2], mx[2]]],
        dtype=np.float64,
    )
    pts.append(corners)
    if int(sample_count) > 0 and len(getattr(mesh, "faces", [])) > 0:
        old_state = np.random.get_state()
        try:
            np.random.seed(12345)
            sampled, _ = trimesh.sample.sample_surface(mesh, int(sample_count))
            pts.append(np.asarray(sampled, dtype=np.float64).reshape(-1, 3))
        finally:
            np.random.set_state(old_state)
    out = np.concatenate(pts, axis=0)
    _REFINE_POINT_CACHE[stat_key] = out
    return out


def _projected_points_bbox(K: np.ndarray, T_cam_obj: np.ndarray, pts_obj: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    pts_cam = _transform_points(T_cam_obj, pts_obj)
    uv, valid = _project_points(K, pts_cam)
    uv_valid = uv[valid]
    z_valid = pts_cam[:, 2][valid]
    if uv_valid.size == 0:
        return None, None, None, None
    bbox = np.asarray([uv_valid[:, 0].min(), uv_valid[:, 1].min(), uv_valid[:, 0].max(), uv_valid[:, 1].max()], dtype=np.float64)
    center, size = _bbox_center_size(bbox)
    return bbox, center, size, np.column_stack([uv_valid, z_valid])


def _mask_depth_reference(depth_m: np.ndarray, mask: np.ndarray, min_depth: float, max_depth: float) -> dict:
    mask_bool = np.asarray(mask > 0, dtype=bool)
    values = np.asarray(depth_m, dtype=np.float64)[mask_bool]
    values = values[np.isfinite(values) & (values >= float(min_depth)) & (values <= float(max_depth))]
    if values.size <= 0:
        return {"ok": False, "median": None, "p25": None, "valid_count": 0}
    return {
        "ok": True,
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p40": float(np.percentile(values, 40)),
        "valid_count": int(values.size),
    }


def _candidate_visible_depth_from_points(projected_points: np.ndarray | None, mask: np.ndarray) -> float | None:
    if projected_points is None or projected_points.size == 0:
        return None
    height, width = mask.shape[:2]
    uv = projected_points[:, :2]
    z = projected_points[:, 2]
    ix = np.rint(uv[:, 0]).astype(np.int32)
    iy = np.rint(uv[:, 1]).astype(np.int32)
    valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height) & np.asarray(mask[iy.clip(0, height - 1), ix.clip(0, width - 1)] > 0)
    values = z[valid]
    if values.size < 8:
        values = z[(ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)]
    if values.size <= 0:
        return None
    # Lower percentile approximates the visible surface without a full z-buffer.
    return float(np.percentile(values, 25))


def _alignment_metrics_for_pose(K: np.ndarray, T_cam_obj: np.ndarray, pts_obj: np.ndarray, mask: np.ndarray, depth_ref: dict | None = None) -> dict:
    mask_bbox, mask_center = _mask_bbox_center(mask)
    if mask_bbox is None or mask_center is None:
        return {"ok": False, "reason": "empty_mask"}
    proj_bbox, proj_center, proj_size, projected_points = _projected_points_bbox(K, T_cam_obj, pts_obj)
    if proj_bbox is None or proj_center is None or proj_size is None:
        return {"ok": False, "reason": "projection_failed"}
    _, mask_size = _bbox_center_size(mask_bbox)
    center_delta = proj_center - mask_center
    center_error_px = float(np.linalg.norm(center_delta))
    bbox_iou = _bbox_iou_xyxy(proj_bbox, mask_bbox)
    size_delta = proj_size - mask_size
    visible_depth = _candidate_visible_depth_from_points(projected_points, mask)
    depth_error_m = None
    if depth_ref and depth_ref.get("ok") and visible_depth is not None:
        depth_error_m = float(visible_depth - float(depth_ref["p40"]))
    return {
        "ok": True,
        "mask_bbox": mask_bbox.tolist(),
        "mask_center": mask_center.tolist(),
        "mask_size": mask_size.tolist(),
        "projected_bbox": proj_bbox.tolist(),
        "projected_center": proj_center.tolist(),
        "projected_size": proj_size.tolist(),
        "center_delta_px": center_delta.tolist(),
        "center_error_px": center_error_px,
        "bbox_iou": float(bbox_iou),
        "size_delta_px": size_delta.tolist(),
        "visible_depth_m": visible_depth,
        "depth_error_m": depth_error_m,
    }


def _parse_name_set(text: str | None) -> set[str]:
    if text is None:
        return set()
    out = set()
    for item in str(text).replace(",", " ").split():
        normalized = normalize_object_name(item)
        if normalized:
            out.add(normalized)
    return out


def _refine_loss(metrics: dict, initial_t: np.ndarray, candidate_t: np.ndarray, depth_weight: float) -> float:
    if not metrics.get("ok"):
        return 1e9
    center_error = float(metrics.get("center_error_px", 1e6))
    mask_size = np.asarray(metrics.get("mask_size", [1.0, 1.0]), dtype=np.float64)
    proj_size = np.asarray(metrics.get("projected_size", [1.0, 1.0]), dtype=np.float64)
    size_error = np.linalg.norm((proj_size - mask_size) / np.maximum(mask_size, 8.0))
    bbox_iou = float(metrics.get("bbox_iou", 0.0))
    depth_error = metrics.get("depth_error_m")
    depth_loss = 0.0 if depth_error is None else (float(depth_error) / 0.012) ** 2
    prior = np.linalg.norm((np.asarray(candidate_t, dtype=np.float64) - np.asarray(initial_t, dtype=np.float64)) / np.asarray([0.035, 0.035, 0.06], dtype=np.float64))
    return float((center_error / 6.0) ** 2 + 3.0 * (size_error**2) + 1.5 * (1.0 - bbox_iou) + float(depth_weight) * depth_loss + 0.04 * (prior**2))


def _sphere_translation_from_mask_depth(K: np.ndarray, mask: np.ndarray, depth_ref: dict, diameter_m: float) -> tuple[np.ndarray | None, dict]:
    mask_bbox, mask_center = _mask_bbox_center(mask)
    if mask_bbox is None or mask_center is None:
        return None, {"ok": False, "reason": "empty_mask"}
    _, mask_size = _bbox_center_size(mask_bbox)
    pixel_diameter = float(np.mean(mask_size))
    if pixel_diameter <= 1e-6:
        return None, {"ok": False, "reason": "invalid_mask_size"}
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    f_avg = 0.5 * (fx + fy)
    diameter_m = float(diameter_m)
    radius_m = diameter_m * 0.5
    z_size = diameter_m * f_avg / pixel_diameter
    z_candidates = [float(z_size)]
    if depth_ref.get("ok") and depth_ref.get("p40") is not None:
        z_candidates.append(float(depth_ref["p40"]) + radius_m)
    z = float(np.median(np.asarray(z_candidates, dtype=np.float64)))
    x = (float(mask_center[0]) - cx) / fx * z
    y = (float(mask_center[1]) - cy) / fy * z
    return np.asarray([x, y, z], dtype=np.float64), {
        "ok": True,
        "mask_bbox": mask_bbox.tolist(),
        "mask_center": mask_center.tolist(),
        "mask_size": mask_size.tolist(),
        "pixel_diameter": pixel_diameter,
        "diameter_m": diameter_m,
        "z_from_size_m": float(z_size),
        "z_candidates_m": z_candidates,
    }


def save_pem_refine_compare_visual(frame: dict, mask: np.ndarray, before: dict, after: dict, out_path: Path, object_name: str, score: float):
    canvas = np.asarray(frame["bgr"], dtype=np.uint8).copy()
    overlay = canvas.copy()
    overlay[np.asarray(mask > 0, dtype=bool)] = (255, 220, 0)
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0.0)
    for metrics, color, label in ((before, (0, 0, 255), "raw"), (after, (0, 255, 0), "refined")):
        if not metrics.get("ok"):
            continue
        box = np.asarray(metrics["projected_bbox"], dtype=np.float64)
        center = np.asarray(metrics["projected_center"], dtype=np.float64)
        cv2.rectangle(canvas, tuple(np.round(box[:2]).astype(int)), tuple(np.round(box[2:]).astype(int)), color, 2)
        cv2.circle(canvas, tuple(np.round(center).astype(int)), 5, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, label, tuple(np.round(center + np.asarray([6.0, -6.0])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    mask_center = np.asarray(before.get("mask_center", after.get("mask_center", [0, 0])), dtype=np.float64)
    cv2.circle(canvas, tuple(np.round(mask_center).astype(int)), 5, (255, 255, 255), -1, cv2.LINE_AA)
    text = (
        f"{object_name} score={score:.3f} "
        f"raw={float(before.get('center_error_px', -1.0)):.1f}px "
        f"refined={float(after.get('center_error_px', -1.0)):.1f}px"
    )
    cv2.putText(canvas, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def refine_pem_translation_against_mask_depth(args, frame: dict, object_name: str, mesh: trimesh.Trimesh, T_cam_obj: np.ndarray, mask: np.ndarray, run_dir: Path, score: float) -> dict:
    enabled = bool(getattr(args, "post_pem_mask_refine", False))
    whitelist = _parse_name_set(getattr(args, "post_pem_mask_refine_objects", "lvmukuai,tennis"))
    normalized_name = normalize_object_name(object_name) or str(object_name)
    tennis_sphere_enabled = normalized_name == "tennis" and bool(getattr(args, "post_pem_mask_refine_tennis_sphere", True))
    sample_count = int(getattr(args, "post_pem_mask_refine_sample_points", 350) or 350)
    pts_obj = _mesh_refine_points(mesh, sample_count)
    K = np.asarray(frame["K"], dtype=np.float64).reshape(3, 3)
    depth_ref = _mask_depth_reference(
        frame["depth_m"],
        mask,
        float(getattr(args, "min_valid_depth_m", 0.05)),
        float(getattr(args, "max_valid_depth_m", 2.0)),
    )
    initial_T = np.asarray(T_cam_obj, dtype=np.float64).reshape(4, 4).copy()
    initial_t = initial_T[:3, 3].copy()
    before = _alignment_metrics_for_pose(K, initial_T, pts_obj, mask, depth_ref)
    result = {
        "enabled": enabled or tennis_sphere_enabled,
        "generic_enabled": enabled,
        "tennis_sphere_enabled": tennis_sphere_enabled,
        "object_name": normalized_name,
        "applied": False,
        "reason": None,
        "elapsed_ms": 0.0,
        "raw_metrics": before,
        "refined_metrics": before,
        "raw_translation_m": initial_t.tolist(),
        "refined_translation_m": initial_t.tolist(),
        "translation_delta_m": [0.0, 0.0, 0.0],
        "depth_reference": depth_ref,
    }
    if not enabled and not tennis_sphere_enabled:
        result["reason"] = "disabled"
        return result
    if enabled and normalized_name not in whitelist and not tennis_sphere_enabled:
        result["reason"] = "not_in_whitelist"
        return result
    skip_border_px = float(getattr(args, "post_pem_mask_refine_skip_border_mask_px", 2.0) or 0.0)
    if _mask_bbox_touches_image_border(before.get("mask_bbox"), np.asarray(mask).shape, skip_border_px):
        result["reason"] = "mask_touches_image_border"
        result["border_margin_px"] = skip_border_px
        return result
    if not before.get("ok"):
        result["reason"] = "raw_metrics_failed"
        return result
    min_pose_z = float(getattr(args, "post_pem_mask_refine_min_pose_z_m", 0.2) or 0.2)
    max_pose_z = float(getattr(args, "post_pem_mask_refine_max_pose_z_m", 1.6) or 1.6)
    raw_pose_valid = bool(np.isfinite(initial_t).all() and min_pose_z <= float(initial_t[2]) <= max_pose_z)
    raw_error = float(before.get("center_error_px", 1e9))
    max_raw_center_px = float(getattr(args, "post_pem_mask_refine_max_raw_center_px", 250.0) or 250.0)
    t0 = time.perf_counter()

    if tennis_sphere_enabled:
        t_sphere, sphere_debug = _sphere_translation_from_mask_depth(K, mask, depth_ref, float(np.max(np.asarray(mesh.extents, dtype=np.float64))))
        result["sphere_refine"] = sphere_debug
        if t_sphere is not None:
            sphere_T = initial_T.copy()
            sphere_T[:3, 3] = t_sphere
            sphere_metrics = _alignment_metrics_for_pose(K, sphere_T, pts_obj, mask, depth_ref)
            sphere_error = float(sphere_metrics.get("center_error_px", raw_error)) if sphere_metrics.get("ok") else raw_error
            score_bad = float(score) <= float(getattr(args, "post_pem_mask_refine_score_trigger", 0.25))
            min_improve_px = float(getattr(args, "post_pem_mask_refine_min_improve_px", 2.0) or 2.0)
            apply_sphere = (not raw_pose_valid) or raw_error > max_raw_center_px or score_bad or (sphere_error + min_improve_px <= raw_error)
            if apply_sphere:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                result.update(
                    {
                        "applied": True,
                        "reason": "tennis_sphere",
                        "elapsed_ms": float(elapsed_ms),
                        "candidate_count": 1,
                        "refined_metrics": sphere_metrics,
                        "refined_translation_m": t_sphere.tolist(),
                        "translation_delta_m": (t_sphere - initial_t).tolist(),
                    }
                )
                try:
                    save_pem_refine_compare_visual(frame, mask, before, sphere_metrics, Path(run_dir) / "pem_refine_compare.png", normalized_name, float(score))
                    result["visualization"] = str(Path(run_dir) / "pem_refine_compare.png")
                except Exception as exc:
                    result["visualization_error"] = repr(exc)
                return result

    if not enabled:
        result["reason"] = "disabled_after_tennis_sphere"
        return result
    if normalized_name not in whitelist:
        result["reason"] = "not_in_whitelist"
        return result
    if not raw_pose_valid:
        result["reason"] = "invalid_initial_translation"
        return result
    if raw_error > max_raw_center_px:
        result["reason"] = "raw_projection_too_far"
        return result
    trigger_px = float(getattr(args, "post_pem_mask_refine_trigger_px", 8.0) or 8.0)
    if float(before.get("center_error_px", 0.0)) < trigger_px and float(score) >= float(getattr(args, "post_pem_mask_refine_score_trigger", 0.25)):
        result["reason"] = "below_trigger"
        return result

    xy_range = float(getattr(args, "post_pem_mask_refine_xy_range_m", 0.03) or 0.03)
    z_range = float(getattr(args, "post_pem_mask_refine_z_range_m", 0.06) or 0.06)
    coarse_xy = float(getattr(args, "post_pem_mask_refine_coarse_xy_step_m", 0.005) or 0.005)
    coarse_z = float(getattr(args, "post_pem_mask_refine_coarse_z_step_m", 0.01) or 0.01)
    fine_xy = float(getattr(args, "post_pem_mask_refine_fine_xy_step_m", 0.0015) or 0.0015)
    fine_z = float(getattr(args, "post_pem_mask_refine_fine_z_step_m", 0.003) or 0.003)
    depth_weight = float(getattr(args, "post_pem_mask_refine_depth_weight", 0.18) or 0.18)

    def eval_candidate(t_vec):
        T = initial_T.copy()
        T[:3, 3] = np.asarray(t_vec, dtype=np.float64)
        metrics = _alignment_metrics_for_pose(K, T, pts_obj, mask, depth_ref)
        loss = _refine_loss(metrics, initial_t, t_vec, depth_weight)
        return loss, metrics

    best_t = initial_t.copy()
    best_loss, best_metrics = eval_candidate(best_t)
    candidate_count = 1
    mask_center = np.asarray(before["mask_center"], dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])

    def centered_candidate_for_z(z_value: float, x_offset: float = 0.0, y_offset: float = 0.0):
        probe_t = initial_t.copy()
        probe_t[2] = float(z_value)
        probe_T = initial_T.copy()
        probe_T[:3, 3] = probe_t
        probe_metrics = _alignment_metrics_for_pose(K, probe_T, pts_obj, mask, depth_ref)
        if not probe_metrics.get("ok"):
            return None
        probe_center = np.asarray(probe_metrics["projected_center"], dtype=np.float64)
        z_safe = max(float(z_value), 1e-6)
        dx = (mask_center[0] - probe_center[0]) / fx * z_safe
        dy = (mask_center[1] - probe_center[1]) / fy * z_safe
        t_vec = initial_t + np.asarray([dx + float(x_offset), dy + float(y_offset), float(z_value) - initial_t[2]], dtype=np.float64)
        delta = t_vec - initial_t
        if abs(delta[0]) > xy_range or abs(delta[1]) > xy_range or abs(delta[2]) > z_range:
            return None
        return t_vec

    dzs = np.arange(-z_range, z_range + coarse_z * 0.5, coarse_z, dtype=np.float64)
    coarse_offsets = np.asarray([-coarse_xy * 0.5, 0.0, coarse_xy * 0.5], dtype=np.float64)
    for dz in dzs:
        z_value = float(initial_t[2] + dz)
        for ox in coarse_offsets:
            for oy in coarse_offsets:
                t_vec = centered_candidate_for_z(z_value, ox, oy)
                if t_vec is None:
                    continue
                loss, metrics = eval_candidate(t_vec)
                candidate_count += 1
                if loss < best_loss:
                    best_loss, best_t, best_metrics = loss, t_vec, metrics

    fine_z_radius = coarse_z * 1.2
    fdzs = np.arange(-fine_z_radius, fine_z_radius + fine_z * 0.5, fine_z, dtype=np.float64)
    coarse_best = best_t.copy()
    fine_offsets = np.asarray([-fine_xy, 0.0, fine_xy], dtype=np.float64)
    for dz in fdzs:
        z_value = float(coarse_best[2] + dz)
        for ox in fine_offsets:
            for oy in fine_offsets:
                t_vec = centered_candidate_for_z(z_value, ox, oy)
                if t_vec is None:
                    continue
                loss, metrics = eval_candidate(t_vec)
                candidate_count += 1
                if loss < best_loss:
                    best_loss, best_t, best_metrics = loss, t_vec, metrics

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    min_improve_px = float(getattr(args, "post_pem_mask_refine_min_improve_px", 2.0) or 2.0)
    raw_error = float(before.get("center_error_px", 1e9))
    refined_error = float(best_metrics.get("center_error_px", raw_error)) if best_metrics.get("ok") else raw_error
    apply = refined_error + min_improve_px <= raw_error
    if not apply:
        result["reason"] = "no_sufficient_improvement"
        best_t = initial_t.copy()
        best_metrics = before
    else:
        result["reason"] = "improved"
    result.update(
        {
            "applied": bool(apply),
            "elapsed_ms": float(elapsed_ms),
            "candidate_count": int(candidate_count),
            "raw_loss": float(_refine_loss(before, initial_t, initial_t, depth_weight)),
            "refined_loss": float(best_loss),
            "refined_metrics": best_metrics,
            "refined_translation_m": best_t.tolist(),
            "translation_delta_m": (best_t - initial_t).tolist(),
        }
    )
    try:
        save_pem_refine_compare_visual(
            frame,
            mask,
            before,
            best_metrics,
            Path(run_dir) / "pem_refine_compare.png",
            normalized_name,
            float(score),
        )
        result["visualization"] = str(Path(run_dir) / "pem_refine_compare.png")
    except Exception as exc:
        result["visualization_error"] = repr(exc)
    return result


def validate_pem_pose_result(args, object_name: str, score: float, T_cam_obj: np.ndarray, refine_info: dict | None = None) -> None:
    normalized_name = normalize_object_name(object_name) or str(object_name)
    T = np.asarray(T_cam_obj, dtype=np.float64).reshape(4, 4)
    t = T[:3, 3]
    min_z = float(getattr(args, "pem_min_valid_pose_z_m", 0.2) or 0.2)
    max_z = float(getattr(args, "pem_max_valid_pose_z_m", 1.6) or 1.6)
    min_score = float(getattr(args, "pem_min_valid_score", 1e-6) or 0.0)
    refined_by_tennis_sphere = bool((refine_info or {}).get("applied")) and str((refine_info or {}).get("reason")) == "tennis_sphere"
    if not np.isfinite(T).all():
        raise RuntimeError(f"SAM6D produced non-finite pose for {normalized_name}")
    if not (min_z <= float(t[2]) <= max_z):
        raise RuntimeError(
            f"SAM6D produced invalid pose depth for {normalized_name}: "
            f"translation={np.round(t, 6).tolist()}, expected z in [{min_z:.3f}, {max_z:.3f}]"
        )
    if float(score) <= min_score and not refined_by_tennis_sphere:
        raise RuntimeError(
            f"SAM6D PEM score is too low for {normalized_name}: "
            f"score={float(score):.6f}, translation={np.round(t, 6).tolist()}"
        )


def _draw_projected_line(canvas: np.ndarray, uv: np.ndarray, valid: np.ndarray, i: int, j: int, color, thickness: int = 2):
    if not (bool(valid[i]) and bool(valid[j])):
        return
    h, w = canvas.shape[:2]
    p1 = tuple(np.round(uv[i]).astype(int).tolist())
    p2 = tuple(np.round(uv[j]).astype(int).tolist())
    # Keep extreme out-of-frame projections from producing long, noisy lines.
    margin = 2000
    if not (-margin <= p1[0] <= w + margin and -margin <= p1[1] <= h + margin):
        return
    if not (-margin <= p2[0] <= w + margin and -margin <= p2[1] <= h + margin):
        return
    cv2.line(canvas, p1, p2, color, int(thickness), cv2.LINE_AA)


def _draw_text_outline(canvas: np.ndarray, text: str, org: tuple[int, int], color, scale: float = 0.52, thickness: int = 1):
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, float(scale), (0, 0, 0), int(thickness) + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, float(scale), color, int(thickness), cv2.LINE_AA)


def _find_named_pose_result(results: list[dict], object_name: str) -> dict | None:
    target = normalize_object_name(object_name) or str(object_name)
    for item in list(results or []):
        if not item.get("ok") or "T_cam_obj" not in item:
            continue
        name = normalize_object_name(item.get("object_name")) or str(item.get("object_name"))
        if name == target:
            return item
    return None


def _load_matrix4x4(path_like: str | None) -> np.ndarray | None:
    if not path_like:
        return None
    path = Path(path_like).expanduser()
    if not path.exists():
        return None
    if path.suffix == ".npy":
        mat = np.load(path)
    else:
        mat = np.loadtxt(path)
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix from {path}, got {mat.shape}")
    return mat


def _camera_to_base_transform_for_overlay(args) -> tuple[np.ndarray | None, str]:
    path = str(getattr(args, "camera_extrinsic_opencv_path", DEFAULT_CAMERA_EXTRINSIC_OPENCV_PATH) or "")
    mat = _load_matrix4x4(path)
    if mat is None:
        return None, "camera_extrinsic_missing"
    if bool(getattr(args, "use_direct_camera_extrinsic", False)):
        return mat, "runtime_robot_distance_direct_extrinsic"
    return np.linalg.inv(mat), "runtime_robot_distance_inverse_extrinsic"


def _desk_slot_specs_for_overlay(slot_y: float) -> list[PlaceSlotSpec]:
    return [
        PlaceSlotSpec(
            name=str(slot_name),
            object_pose_local=LocalPoseSpec(position=(float(x), float(slot_y), float(z))),
        )
        for slot_name, (x, z) in DESK_SLOT_LAYOUT_XZ
    ]


def _runtime_desk_slot_specs_for_overlay(args, T_cam_desk: np.ndarray, slot_y: float) -> tuple[list[PlaceSlotSpec], str]:
    static_slots = _desk_slot_specs_for_overlay(slot_y)
    T_base_cam, mode = _camera_to_base_transform_for_overlay(args)
    if T_base_cam is None:
        return static_slots, mode
    T_base_desk = np.asarray(T_base_cam, dtype=np.float64).reshape(4, 4) @ np.asarray(T_cam_desk, dtype=np.float64).reshape(4, 4)
    runtime_slots = get_runtime_slot_specs(
        "desk",
        static_slots,
        T_base_desk,
        np.zeros(3, dtype=np.float32),
    )
    return list(runtime_slots or static_slots), mode


def _local_pose_spec_to_matrix_for_overlay(spec: LocalPoseSpec) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    rpy_rad = np.deg2rad(np.asarray(spec.rpy_deg, dtype=np.float64).reshape(3))
    T[:3, :3] = euler2mat(float(rpy_rad[0]), float(rpy_rad[1]), float(rpy_rad[2]), axes="sxyz")
    T[:3, 3] = np.asarray(spec.position, dtype=np.float64).reshape(3)
    return T


def _draw_desk_slot_overlay(args, canvas: np.ndarray, K: np.ndarray, results: list[dict]) -> dict:
    desk_item = _find_named_pose_result(results, "desk")
    if desk_item is None:
        return {"rendered": False, "reason": "desk_pose_missing", "slot_count": 0}

    T_cam_desk = np.asarray(desk_item["T_cam_obj"], dtype=np.float64).reshape(4, 4)
    slot_y = 0.09
    half_x = 0.047
    half_z = 0.035
    color = (0, 255, 255)
    rendered = []
    h, w = canvas.shape[:2]
    slot_specs, label_mode = _runtime_desk_slot_specs_for_overlay(args, T_cam_desk, slot_y)
    for slot in slot_specs:
        slot_name = str(slot.name)
        slot_p = np.asarray(slot.object_pose_local.position, dtype=np.float64).reshape(3)
        x = float(slot_p[0])
        z = float(slot_p[2])
        center_obj = np.asarray([[x, slot_y, z]], dtype=np.float64)
        center_uv, center_valid = _project_points(K, _transform_points(T_cam_desk, center_obj))
        corners_obj = np.asarray(
            [
                [x - half_x, slot_y, z - half_z],
                [x + half_x, slot_y, z - half_z],
                [x + half_x, slot_y, z + half_z],
                [x - half_x, slot_y, z + half_z],
            ],
            dtype=np.float64,
        )
        corners_uv, corners_valid = _project_points(K, _transform_points(T_cam_desk, corners_obj))
        if np.count_nonzero(corners_valid) >= 3:
            pts = np.round(corners_uv[corners_valid]).astype(np.int32).reshape(-1, 1, 2)
            if np.all(pts[:, 0, 0] > -2000) and np.all(pts[:, 0, 0] < w + 2000) and np.all(pts[:, 0, 1] > -2000) and np.all(pts[:, 0, 1] < h + 2000):
                cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
        if bool(center_valid[0]):
            u = int(np.clip(round(float(center_uv[0, 0])), 0, w - 1))
            v = int(np.clip(round(float(center_uv[0, 1])), 0, h - 1))
            cv2.circle(canvas, (u, v), 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (u, v), 4, color, -1, cv2.LINE_AA)
            label = str(slot_name).replace("slot_", "")
            _draw_text_outline(canvas, label, (u + 7, v - 7), color, 0.62, 2)
            rendered.append(slot_name)

    if rendered:
        _draw_text_outline(canvas, "runtime desk slots", (8, h - 12), color, 0.55, 1)
    return {"rendered": bool(rendered), "slot_count": len(rendered), "slots": rendered, "label_mode": label_mode}


def _draw_mesh_wireframe_overlay(canvas: np.ndarray, K: np.ndarray, T_cam_obj: np.ndarray, mesh: trimesh.Trimesh, color, *, max_edges: int = 700) -> dict:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.size <= 0:
        return {"rendered": False, "reason": "empty_mesh"}
    uv, valid = _project_points(K, _transform_points(np.asarray(T_cam_obj, dtype=np.float64).reshape(4, 4), vertices))
    edges = np.asarray(getattr(mesh, "edges_unique", []), dtype=np.int64)
    if edges.size <= 0 and getattr(mesh, "faces", None) is not None:
        faces = np.asarray(mesh.faces, dtype=np.int64)
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0) if faces.size else np.zeros((0, 2), dtype=np.int64)
    if len(edges) > int(max_edges):
        step = max(1, int(np.ceil(len(edges) / float(max_edges))))
        edges = edges[::step]
    drawn = 0
    h, w = canvas.shape[:2]
    for i, j in edges:
        if int(i) >= len(uv) or int(j) >= len(uv) or not (bool(valid[i]) and bool(valid[j])):
            continue
        p1 = tuple(np.round(uv[i]).astype(int).tolist())
        p2 = tuple(np.round(uv[j]).astype(int).tolist())
        if not (-2000 <= p1[0] <= w + 2000 and -2000 <= p1[1] <= h + 2000):
            continue
        if not (-2000 <= p2[0] <= w + 2000 and -2000 <= p2[1] <= h + 2000):
            continue
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        drawn += 1
    center_uv, center_valid = _project_points(K, _transform_points(np.asarray(T_cam_obj, dtype=np.float64).reshape(4, 4), np.zeros((1, 3), dtype=np.float64)))
    return {
        "rendered": drawn > 0,
        "edge_count": int(drawn),
        "center_uv": None if not bool(center_valid[0]) else [float(center_uv[0, 0]), float(center_uv[0, 1])],
    }


def save_placement_preview_visualization(args, frame: dict, results: list[dict], assignments: list[dict], out_path: Path) -> dict:
    canvas = np.asarray(frame["bgr"], dtype=np.uint8).copy()
    K = np.asarray(frame["K"], dtype=np.float64).reshape(3, 3)
    base_info = save_full_scene_pem_visualization(args, frame, results, out_path)
    saved = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
    if saved is not None:
        canvas = saved

    desk_item = _find_named_pose_result(results, "desk")
    if desk_item is None:
        return {**base_info, "placement_preview": {"rendered": 0, "reason": "desk_pose_missing"}}
    T_cam_desk = np.asarray(desk_item["T_cam_obj"], dtype=np.float64).reshape(4, 4)
    slot_specs, label_mode = _runtime_desk_slot_specs_for_overlay(args, T_cam_desk, 0.09)
    slot_by_name = {str(slot.name): slot for slot in slot_specs}
    palette = [
        (255, 80, 80),
        (80, 180, 255),
        (90, 220, 90),
        (255, 180, 70),
        (200, 120, 255),
        (80, 230, 220),
        (255, 100, 180),
    ]
    rendered = []
    skipped = []
    for idx, assignment in enumerate(list(assignments or [])):
        object_name = normalize_object_name(assignment.get("object") or assignment.get("object_name"))
        destination = str(assignment.get("destination") or assignment.get("slot") or "")
        if destination and not destination.startswith("slot_"):
            destination = f"slot_{destination}"
        if not object_name:
            continue
        if object_name == "bi" or destination == "bitong":
            skipped.append({"object_name": object_name, "destination": "bitong", "reason": "fixed_bitong_target"})
            continue
        slot = slot_by_name.get(destination)
        if slot is None:
            skipped.append({"object_name": object_name, "destination": destination, "reason": "slot_missing"})
            continue
        rule = get_place_rule(object_name)
        if rule is None:
            skipped.append({"object_name": object_name, "destination": destination, "reason": "place_rule_missing"})
            continue
        try:
            spec = get_object_spec(object_name)
            mesh_scale, _ = resolve_object_spec_scales(spec)
            mesh = _load_scene_mesh(spec.mesh_file, mesh_scale)
            T_cam_obj = T_cam_desk @ _local_pose_spec_to_matrix_for_overlay(slot.object_pose_local)
            color = palette[idx % len(palette)]
            wire = _draw_mesh_wireframe_overlay(canvas, K, T_cam_obj, mesh, color)
            center_uv = wire.get("center_uv")
            if center_uv is not None:
                u = int(np.clip(round(center_uv[0]), 0, canvas.shape[1] - 1))
                v = int(np.clip(round(center_uv[1]), 16, canvas.shape[0] - 1))
                label = f"{destination.replace('slot_', '')}:{object_name}"
                _draw_text_outline(canvas, label, (u + 8, v + 14), color, 0.48, 1)
            rendered.append({"object_name": object_name, "destination": destination, **wire})
        except Exception as exc:
            skipped.append({"object_name": object_name, "destination": destination, "reason": repr(exc)})

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return {
        **base_info,
        "path": str(out_path),
        "placement_preview": {
            "rendered": len([item for item in rendered if item.get("rendered")]),
            "items": rendered,
            "skipped": skipped,
            "slot_label_mode": label_mode,
        },
    }


def save_full_scene_pem_visualization(args, frame: dict, results: list[dict], out_path: Path) -> dict:
    canvas = np.asarray(frame["bgr"], dtype=np.uint8).copy()
    K = np.asarray(frame["K"], dtype=np.float64).reshape(3, 3)
    palette = [
        (40, 220, 255),
        (80, 255, 80),
        (255, 180, 60),
        (255, 80, 180),
        (80, 160, 255),
        (180, 120, 255),
        (60, 255, 180),
        (255, 255, 80),
    ]
    edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    rendered = []
    errors = []
    for idx, item in enumerate(results):
        if not item.get("ok") or "T_cam_obj" not in item:
            continue
        object_name = normalize_object_name(item.get("object_name")) or str(item.get("object_name"))
        try:
            item_args = copy.copy(args)
            item_args.object_name = object_name
            _, _, mesh_file, mesh_scale = resolve_object_inputs(item_args)
            mesh = _load_scene_mesh(mesh_file, mesh_scale)
            bounds = np.asarray(mesh.bounds, dtype=np.float64)
            mn, mx = bounds[0], bounds[1]
            corners = np.asarray(
                [
                    [mn[0], mn[1], mn[2]],
                    [mx[0], mn[1], mn[2]],
                    [mn[0], mx[1], mn[2]],
                    [mx[0], mx[1], mn[2]],
                    [mn[0], mn[1], mx[2]],
                    [mx[0], mn[1], mx[2]],
                    [mn[0], mx[1], mx[2]],
                    [mx[0], mx[1], mx[2]],
                ],
                dtype=np.float64,
            )
            T_cam_obj = np.asarray(item["T_cam_obj"], dtype=np.float64).reshape(4, 4)
            uv, valid = _project_points(K, _transform_points(T_cam_obj, corners))
            color = palette[idx % len(palette)]
            for a, b in edges:
                _draw_projected_line(canvas, uv, valid, a, b, color, 2)

            extents = np.maximum(mx - mn, 1e-6)
            axis_len = float(np.clip(np.max(extents) * 0.55, 0.025, 0.08))
            axes_obj = np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [axis_len, 0.0, 0.0],
                    [0.0, axis_len, 0.0],
                    [0.0, 0.0, axis_len],
                ],
                dtype=np.float64,
            )
            axes_uv, axes_valid = _project_points(K, _transform_points(T_cam_obj, axes_obj))
            _draw_projected_line(canvas, axes_uv, axes_valid, 0, 1, (0, 0, 255), 3)
            _draw_projected_line(canvas, axes_uv, axes_valid, 0, 2, (0, 255, 0), 3)
            _draw_projected_line(canvas, axes_uv, axes_valid, 0, 3, (255, 0, 0), 3)

            label_points = uv[valid]
            if label_points.size:
                x = int(np.clip(np.min(label_points[:, 0]), 0, canvas.shape[1] - 1))
                y = int(np.clip(np.min(label_points[:, 1]) - 8, 18, canvas.shape[0] - 1))
            else:
                center_uv, center_valid = _project_points(K, _transform_points(T_cam_obj, np.zeros((1, 3), dtype=np.float64)))
                if bool(center_valid[0]):
                    x = int(np.clip(center_uv[0, 0], 0, canvas.shape[1] - 1))
                    y = int(np.clip(center_uv[0, 1], 18, canvas.shape[0] - 1))
                else:
                    x, y = 8, 24 + 20 * idx
            label = f"{object_name} {float(item.get('score', 0.0)):.3f}"
            cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            rendered.append(object_name)
        except Exception as exc:
            errors.append({"object_name": object_name, "error": repr(exc)})

    slot_overlay = _draw_desk_slot_overlay(args, canvas, K, results)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return {"path": str(out_path), "rendered": rendered, "errors": errors, "slot_overlay": slot_overlay}


def parse_args():
    parser = argparse.ArgumentParser(description="Bypass SAM-6D ISM with GroundingDINO + depth mask, then run SAM-6D PEM.")
    parser.add_argument("--sam6d-root", type=str, default=DEFAULT_SAM6D_ROOT)
    parser.add_argument("--foundationpose-root", type=str, default=DEFAULT_FOUNDATIONPOSE_ROOT)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--object-name", type=str, default=None)
    parser.add_argument("--object-names", type=str, nargs="+", default=None, help="Run one shared RGB-D capture through multiple object pose estimates.")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--mesh-file", type=str, default=None)
    parser.add_argument("--mesh-scale", type=float, default=None)
    parser.add_argument("--templates-dir", type=str, default=None, help="Optional pre-rendered SAM-6D templates directory.")
    parser.add_argument(
        "--template-cache-root",
        type=str,
        default=str(RUNTIME_DIR / "sam6d_template_cache"),
        help="Cache rendered SAM-6D CAD templates by object mesh and scale.",
    )

    parser.add_argument("--rgb-path", type=str, default=None)
    parser.add_argument("--depth-path", type=str, default=None)
    parser.add_argument("--camera-path", type=str, default=None)
    parser.add_argument(
        "--frame-dir",
        type=str,
        default=None,
        help="Directory containing rgb.png, depth.png, and camera.json; shorthand for the three path arguments.",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-serial", type=str, default=None)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--camera-frame-timeout-retries", type=int, default=3)
    parser.add_argument("--camera-extrinsic-opencv-path", type=str, default=DEFAULT_CAMERA_EXTRINSIC_OPENCV_PATH)
    parser.add_argument(
        "--use-direct-camera-extrinsic",
        action="store_true",
        default=False,
        help="Interpret camera_extrinsic_opencv as direct T_base_cam. Default applies inverse, matching the grasp pipeline.",
    )

    parser.add_argument("--grounding-dino-model-id", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--grounding-dino-local-files-only", dest="grounding_dino_local_files_only", action="store_true", default=True)
    parser.add_argument("--no-grounding-dino-local-files-only", dest="grounding_dino_local_files_only", action="store_false")
    parser.add_argument("--grounding-dino-box-threshold", type=float, default=0.25)
    parser.add_argument("--grounding-dino-text-threshold", type=float, default=0.2)
    parser.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"))

    parser.add_argument(
        "--mask-mode",
        choices=[
            "hybrid_depth_color",
            "fastsam_bbox",
            "sam3_text",
            "sam3_bbox",
            "sam3_text_bbox",
            "depth_plane",
            "grabcut_depth",
            "depth",
            "box",
        ],
        default="hybrid_depth_color",
    )
    parser.add_argument("--bbox-pad-frac", type=float, default=0.06)
    parser.add_argument("--depth-anchor-percentile", type=float, default=50.0)
    parser.add_argument("--depth-cluster-band-m", type=float, default=0.10)
    parser.add_argument("--depth-cluster-back-margin-m", type=float, default=0.03)
    parser.add_argument("--min-valid-depth-m", type=float, default=0.05)
    parser.add_argument("--max-valid-depth-m", type=float, default=2.0)
    parser.add_argument("--min-mask-area", type=int, default=64)
    parser.add_argument("--mask-morph-kernel", type=int, default=7)
    parser.add_argument("--depth-plane-ring-pad-frac", type=float, default=0.18)
    parser.add_argument("--depth-plane-foreground-margin-m", type=float, default=0.005)
    parser.add_argument("--depth-plane-fit-trim-m", type=float, default=0.06)
    parser.add_argument("--depth-plane-fit-iters", type=int, default=2)
    parser.add_argument("--depth-plane-min-ring-pixels", type=int, default=128)
    parser.add_argument("--hybrid-grabcut-iter", type=int, default=2)
    parser.add_argument("--hybrid-depth-fg-erode-kernel", type=int, default=3)
    parser.add_argument("--hybrid-depth-possible-dilate-kernel", type=int, default=9)
    parser.add_argument("--hybrid-morph-kernel", type=int, default=5)
    parser.add_argument("--hybrid-table-veto", dest="hybrid_table_veto", action="store_true", default=False)
    parser.add_argument("--no-hybrid-table-veto", dest="hybrid_table_veto", action="store_false")
    parser.add_argument("--hybrid-table-veto-margin-m", type=float, default=0.004)
    parser.add_argument("--hybrid-table-veto-keep-dilate-kernel", type=int, default=13)
    parser.add_argument("--hybrid-table-veto-color-dist", type=float, default=36.0)
    parser.add_argument("--hybrid-shape-smooth", dest="hybrid_shape_smooth", action="store_true", default=False)
    parser.add_argument("--no-hybrid-shape-smooth", dest="hybrid_shape_smooth", action="store_false")
    parser.add_argument("--hybrid-shape-smooth-min-aspect", type=float, default=1.8)
    parser.add_argument("--hybrid-shape-long-axis-scale", type=float, default=1.02)
    parser.add_argument("--hybrid-shape-short-axis-scale", type=float, default=1.10)
    parser.add_argument("--fastsam-model-path", type=str, default=DEFAULT_FASTSAM_MODEL_PATH)
    parser.add_argument("--fastsam-imgsz", type=int, default=640)
    parser.add_argument("--fastsam-conf", type=float, default=0.05)
    parser.add_argument("--fastsam-iou", type=float, default=0.9)
    parser.add_argument("--fastsam-max-det", type=int, default=200)
    parser.add_argument("--fastsam-select-pad-frac", type=float, default=0.02)
    parser.add_argument("--fastsam-crop-pad-frac", type=float, default=0.12)
    parser.add_argument("--fastsam-morph-kernel", type=int, default=3)
    parser.add_argument("--sam3-python", type=str, default=DEFAULT_SAM3_PYTHON)
    parser.add_argument("--sam3-provider-script", type=str, default=DEFAULT_SAM3_PROVIDER_SCRIPT)
    parser.add_argument("--sam3-checkpoint-path", type=str, default=DEFAULT_SAM3_CHECKPOINT_PATH)
    parser.add_argument("--sam3-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--sam3-resolution", type=int, default=1008)
    parser.add_argument("--sam3-morph-kernel", type=int, default=3)
    parser.add_argument("--sam3-max-masks-per-item", type=int, default=1, help="Maximum SAM3 candidates returned per prompt (used for full-scene text mode).")
    parser.add_argument("--sam3-full-scene-keep-multi-instances", dest="sam3_full_scene_keep_multi_instances", action="store_true", default=False)
    parser.add_argument("--no-sam3-full-scene-keep-multi-instances", dest="sam3_full_scene_keep_multi_instances", action="store_false")
    parser.add_argument("--sam3-instance-index", type=int, default=0, help="Instance index to use from SAM3 full-scene candidates for this object.")
    parser.add_argument("--sam3-device", type=str, default="cuda")
    parser.add_argument("--sam3-full-scene-mask-confirm", dest="sam3_full_scene_mask_confirm", action="store_true", default=False)
    parser.add_argument("--no-sam3-full-scene-mask-confirm", dest="sam3_full_scene_mask_confirm", action="store_false")
    parser.add_argument("--sam3-require-full-scene-masks", dest="sam3_require_full_scene_masks", action="store_true", default=False)
    parser.add_argument("--no-sam3-require-full-scene-masks", dest="sam3_require_full_scene_masks", action="store_false")
    parser.add_argument("--sam3-show-full-scene-mask-window", dest="sam3_show_full_scene_mask_window", action="store_true", default=True)
    parser.add_argument("--no-sam3-show-full-scene-mask-window", dest="sam3_show_full_scene_mask_window", action="store_false")
    parser.add_argument("--grabcut-iter", type=int, default=3)
    parser.add_argument("--grabcut-inner-shrink-frac", type=float, default=0.16)
    parser.add_argument("--grabcut-core-seed-scale-x", type=float, default=0.42)
    parser.add_argument("--grabcut-core-seed-scale-y", type=float, default=0.28)
    parser.add_argument("--grabcut-depth-fg-erode-kernel", type=int, default=3)
    parser.add_argument("--grabcut-depth-seed-max-box-fill-ratio", type=float, default=0.72)

    parser.add_argument(
        "--pem-det-score-thresh",
        type=float,
        default=0.2,
        help=(
            "SAM-6D PEM's script currently has a string argparse bug for this option, "
            "so the wrapper keeps generated detections at or above this score instead of passing the option."
        ),
    )
    parser.add_argument("--pem-run-mode", choices=["inprocess", "subprocess"], default="inprocess")
    parser.add_argument(
        "--pem-feature-cache-root",
        type=str,
        default=str(RUNTIME_DIR / "sam6d_pem_feature_cache"),
    )
    parser.add_argument("--no-pem-feature-cache", action="store_true")
    parser.add_argument("--pem-warmup-during-sam3", dest="pem_warmup_during_sam3", action="store_true", default=False)
    parser.add_argument("--no-pem-warmup-during-sam3", dest="pem_warmup_during_sam3", action="store_false")
    parser.add_argument("--pem-save-visualization", dest="pem_save_visualization", action="store_true", default=True)
    parser.add_argument("--no-pem-save-visualization", dest="pem_save_visualization", action="store_false")
    parser.add_argument("--full-scene-pem-visualization", dest="full_scene_pem_visualization", action="store_true", default=True)
    parser.add_argument("--no-full-scene-pem-visualization", dest="full_scene_pem_visualization", action="store_false")
    parser.add_argument("--sam3-full-scene-result-json", type=str, default="")
    parser.add_argument("--post-pem-mask-refine", dest="post_pem_mask_refine", action="store_true", default=False)
    parser.add_argument("--no-post-pem-mask-refine", dest="post_pem_mask_refine", action="store_false")
    parser.add_argument("--post-pem-mask-refine-objects", type=str, default="lvmukuai,carriot,tennis")
    parser.add_argument("--post-pem-mask-refine-trigger-px", type=float, default=6.0)
    parser.add_argument("--post-pem-mask-refine-score-trigger", type=float, default=0.25)
    parser.add_argument("--post-pem-mask-refine-xy-range-m", type=float, default=0.025)
    parser.add_argument("--post-pem-mask-refine-z-range-m", type=float, default=0.05)
    parser.add_argument("--post-pem-mask-refine-coarse-xy-step-m", type=float, default=0.0075)
    parser.add_argument("--post-pem-mask-refine-coarse-z-step-m", type=float, default=0.01)
    parser.add_argument("--post-pem-mask-refine-fine-xy-step-m", type=float, default=0.0025)
    parser.add_argument("--post-pem-mask-refine-fine-z-step-m", type=float, default=0.003)
    parser.add_argument("--post-pem-mask-refine-depth-weight", type=float, default=0.18)
    parser.add_argument("--post-pem-mask-refine-min-improve-px", type=float, default=2.0)
    parser.add_argument("--post-pem-mask-refine-sample-points", type=int, default=350)
    parser.add_argument("--post-pem-mask-refine-min-pose-z-m", type=float, default=0.2)
    parser.add_argument("--post-pem-mask-refine-max-pose-z-m", type=float, default=1.6)
    parser.add_argument("--post-pem-mask-refine-max-raw-center-px", type=float, default=250.0)
    parser.add_argument("--post-pem-mask-refine-skip-border-mask-px", type=float, default=2.0)
    parser.add_argument("--post-pem-mask-refine-tennis-sphere", dest="post_pem_mask_refine_tennis_sphere", action="store_true", default=True)
    parser.add_argument("--no-post-pem-mask-refine-tennis-sphere", dest="post_pem_mask_refine_tennis_sphere", action="store_false")
    parser.add_argument("--pem-min-valid-score", type=float, default=1e-6)
    parser.add_argument("--pem-min-valid-pose-z-m", type=float, default=0.2)
    parser.add_argument("--pem-max-valid-pose-z-m", type=float, default=1.6)
    parser.add_argument("--repair-mask-depth", dest="repair_mask_depth", action="store_true", default=True)
    parser.add_argument("--no-repair-mask-depth", dest="repair_mask_depth", action="store_false")
    parser.add_argument("--repair-mask-depth-min-valid-pixels", type=int, default=128)
    parser.add_argument("--repair-mask-depth-min-valid-ratio", type=float, default=0.08)
    parser.add_argument("--repair-mask-depth-search-px", type=int, default=180)
    parser.add_argument("--repair-mask-depth-surface-offset-m", type=float, default=0.006)
    parser.add_argument("--skip-pem", action="store_true", help="Only generate SAM-6D inputs and detection_ism.json.")
    return parser.parse_args()


def _run_single_object_pose(args, frame: dict, sam6d_root: Path, run_dir: Path, detector_cache: dict | None = None) -> dict:
    object_name, prompt, mesh_file, mesh_scale = resolve_object_inputs(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "run_args.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "object_name": object_name,
                "prompt": prompt,
                "mesh_file": mesh_file,
                "mesh_scale": mesh_scale,
            },
            f,
            indent=2,
        )

    rgb_path, depth_path, cam_path = save_sam6d_input_frame(frame, run_dir)
    cad_path = export_cad_mm(mesh_file, mesh_scale, run_dir / "cad_mm.ply")

    detector = None
    direct_sam3_text = str(args.mask_mode) == "sam3_text" and args.bbox is None
    precomputed_sam3 = None
    if direct_sam3_text and detector_cache is not None:
        precomputed_sam3 = detector_cache.get("sam3_text_results", {}).get(object_name)
        if precomputed_sam3 is None and bool(detector_cache.get("sam3_text_full_scene_attempted", False)):
            raise RuntimeError(f"SAM3 full-scene precompute produced no mask for {object_name}; see sam3_full_scene_text/sam3_batch_result.json")

    if precomputed_sam3 is not None:
        precomp_instance_index, precomputed_sam3 = _select_sam3_precomputed_instance(
            precomputed_sam3,
            object_name,
            getattr(args, "sam3_instance_index", 0),
        )
        det_score = float(precomputed_sam3.get("selected", {}).get("model_score", 1.0))
        det = {
            "box": np.asarray(precomputed_sam3.get("mask_bbox"), dtype=np.float32),
            "score": det_score,
            "label": "sam3_text",
            "sam3_instance_index": precomp_instance_index,
        }
        detections = []
    elif args.bbox is not None:
        det = {"box": np.asarray(args.bbox, dtype=np.float32), "score": 1.0, "label": "manual_bbox"}
        detections = [det]
    elif direct_sam3_text:
        det = {"box": None, "score": 1.0, "label": "sam3_text"}
        detections = []
    else:
        if detector_cache is not None:
            detector = detector_cache.get("grounding_dino")
            if detector is None:
                detector = create_grounding_dino_detector(args)
                detector_cache["grounding_dino"] = detector
        else:
            detector = create_grounding_dino_detector(args)
        detections = run_grounding_dino_detector(
            detector,
            frame["rgb"],
            prompt,
            args.grounding_dino_box_threshold,
            args.grounding_dino_text_threshold,
        )
        if not detections:
            raise RuntimeError(f"GroundingDINO found no boxes for prompt: {prompt!r}")
        det = detections[0]

    if str(args.mask_mode).startswith("sam3_"):
        release_grounding_dino_detector_gpu(detector_cache=detector_cache, detector=detector)
        detector = None

    height, width = frame["depth_m"].shape[:2]
    mask_t0 = time.perf_counter()
    if precomputed_sam3 is not None:
        mask_u8 = cv2.imread(str(precomputed_sam3["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask_u8 is None:
            raise RuntimeError(f"failed to read precomputed SAM3 mask: {precomputed_sam3['mask_path']}")
        mask = np.asarray(mask_u8 > 0, dtype=bool)
        selected = precomputed_sam3.get("selected", {})
        mask_source = (
            f"sam3_text_precomputed_score_{float(selected.get('model_score', 0.0)):.3f}"
            f"_ms_{float(precomputed_sam3.get('infer_elapsed_ms', 0.0)):.1f}"
        )
    else:
        box_for_mask = None if direct_sam3_text else _clip_box_xyxy(det["box"], width, height, pad_frac=0.0)
        mask, mask_source = make_detection_mask(frame, box_for_mask, args, prompt=prompt, run_dir=run_dir, rgb_path=rgb_path)
    mask_elapsed_ms = (time.perf_counter() - mask_t0) * 1000.0

    if det.get("box") is None:
        mask_box = _mask_bbox_xyxy(mask)
        if mask_box is None:
            raise RuntimeError(f"SAM3 produced an empty mask for {object_name}")
        det["box"] = mask_box
    box = _clip_box_xyxy(det["box"], width, height, pad_frac=0.0)
    depth_repair = repair_depth_for_mask_if_needed(frame, mask, box, args, depth_path)
    seg_path = run_dir / "sam6d_results" / "detection_ism.json"
    pem_score = max(float(det["score"]), float(args.pem_det_score_thresh) + 1e-4)
    result_instance_index = int(det.get("sam3_instance_index", getattr(args, "sam3_instance_index", 0)) or 0)
    detection_entry = write_detection_ism(
        mask,
        box,
        pem_score,
        str(det["label"] or prompt),
        seg_path,
        object_name=object_name,
        sam3_instance_index=result_instance_index,
    )
    save_detection_visual(frame, box, mask, run_dir / "groundingdino_depth_mask.png", str(det["label"] or prompt), float(det["score"]))
    save_binary_mask(mask, run_dir / "sam6d_binary_mask.png")

    with open(run_dir / "groundingdino_result.json", "w") as f:
        json.dump(
            {
                "prompt": prompt,
                "selected": det,
                "detections": detections,
                "mask_source": mask_source,
                "mask_elapsed_ms": float(mask_elapsed_ms),
                "mask_pixels": int(np.count_nonzero(mask)),
                "depth_repair": depth_repair,
            },
            f,
            indent=2,
            default=lambda x: np.asarray(x).tolist(),
        )

    print(
        f"[sam6d-gdino] object={object_name} prompt={prompt!r} "
        f"score={float(det['score']):.3f} box={[round(float(v), 1) for v in box.tolist()]} "
        f"mask_pixels={int(np.count_nonzero(mask))} mask_source={mask_source} "
        f"mask_ms={mask_elapsed_ms:.2f}"
    )
    print(f"[sam6d-gdino] run_dir={run_dir}")

    if args.skip_pem:
        print(f"[sam6d-gdino] skipped PEM; detection_ism={seg_path}")
        result = {
            "object_name": object_name,
            "prompt": prompt,
            "run_dir": str(run_dir),
            "sam3_instance_index": result_instance_index,
            "mask_source": mask_source,
            "mask_elapsed_ms": float(mask_elapsed_ms),
            "mask_pixels": int(np.count_nonzero(mask)),
            "depth_repair": depth_repair,
            "detection_ism": detection_entry,
        }
        result_path = run_dir / "sam6d_pose_result.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print("[sam6d-gdino] result:", result_path)
        return result

    template_dir = prepare_templates(args, sam6d_root, run_dir, cad_path, object_name=object_name, mesh_file=mesh_file, mesh_scale=mesh_scale)
    pem_t0 = time.perf_counter()
    pem_json = run_sam6d_pem(
        args,
        sam6d_root,
        run_dir,
        cad_path,
        rgb_path,
        depth_path,
        cam_path,
        seg_path,
        template_dir=template_dir,
        object_name=object_name,
        mesh_file=mesh_file,
        mesh_scale=mesh_scale,
    )
    pem_elapsed_ms = (time.perf_counter() - pem_t0) * 1000.0
    best, T_cam_obj = parse_pem_result(pem_json)
    raw_T_cam_obj = T_cam_obj.copy()
    refine_info = refine_pem_translation_against_mask_depth(
        args,
        frame,
        object_name,
        _load_scene_mesh(mesh_file, mesh_scale),
        T_cam_obj,
        mask,
        run_dir,
        float(best.get("score", 0.0)),
    )
    if bool(refine_info.get("applied", False)):
        T_cam_obj = np.asarray(T_cam_obj, dtype=np.float32).copy()
        T_cam_obj[:3, 3] = np.asarray(refine_info["refined_translation_m"], dtype=np.float32).reshape(3)
    score = float(best.get("score", 0.0))
    validate_pem_pose_result(args, object_name, score, T_cam_obj, refine_info)
    result = {
        "object_name": object_name,
        "prompt": prompt,
        "run_dir": str(run_dir),
        "sam3_instance_index": result_instance_index,
        "score": score,
        "mask_source": mask_source,
        "mask_elapsed_ms": float(mask_elapsed_ms),
        "mask_pixels": int(np.count_nonzero(mask)),
        "depth_repair": depth_repair,
        "detection_box_xyxy": box.tolist(),
        "pem_elapsed_ms": float(pem_elapsed_ms),
        "T_cam_obj_raw_pem": raw_T_cam_obj.tolist(),
        "translation_m_raw_pem": raw_T_cam_obj[:3, 3].tolist(),
        "T_cam_obj": T_cam_obj.tolist(),
        "translation_m": T_cam_obj[:3, 3].tolist(),
        "pem_refine": refine_info,
        "pem_detection": best,
        "detection_ism": detection_entry,
    }
    result_path = run_dir / "sam6d_pose_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print("[sam6d-gdino] pem_ms:", f"{pem_elapsed_ms:.2f}", "mode:", str(args.pem_run_mode))
    print("[sam6d-gdino] best PEM score:", f"{score:.4f}")
    if refine_info.get("enabled"):
        raw_err = float(refine_info.get("raw_metrics", {}).get("center_error_px", -1.0))
        ref_err = float(refine_info.get("refined_metrics", {}).get("center_error_px", raw_err))
        print(
            "[sam6d-gdino] pem_refine:",
            f"applied={bool(refine_info.get('applied', False))}",
            f"reason={refine_info.get('reason')}",
            f"raw_px={raw_err:.2f}",
            f"refined_px={ref_err:.2f}",
            f"dt_ms={float(refine_info.get('elapsed_ms', 0.0)):.2f}",
        )
    print("[sam6d-gdino] T_cam_obj translation(m):", np.round(T_cam_obj[:3, 3], 6).tolist())
    print("[sam6d-gdino] result:", result_path)
    return result


def _run_same_object_multi_instance_pose(
    args,
    frame: dict,
    sam6d_root: Path,
    run_dir: Path,
    detector_cache: dict | None,
    instance_count: int,
) -> list[dict]:
    object_name, prompt, mesh_file, mesh_scale = resolve_object_inputs(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    requested_count = max(1, int(instance_count))

    with open(run_dir / "run_args.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "object_name": object_name,
                "prompt": prompt,
                "mesh_file": mesh_file,
                "mesh_scale": mesh_scale,
                "requested_instances": requested_count,
                "batch_same_object_instances": True,
            },
            f,
            indent=2,
        )

    if not (str(args.mask_mode) == "sam3_text" and args.bbox is None):
        raise RuntimeError("same-object batch pose currently requires --mask-mode sam3_text without --bbox")
    if detector_cache is None:
        raise RuntimeError("same-object batch pose requires precomputed SAM3 full-scene masks")
    precomputed_all = detector_cache.get("sam3_text_results", {}).get(object_name)
    if precomputed_all is None and bool(detector_cache.get("sam3_text_full_scene_attempted", False)):
        raise RuntimeError(f"SAM3 full-scene precompute produced no mask for {object_name}; see sam3_full_scene_text/sam3_batch_result.json")
    if isinstance(precomputed_all, dict):
        precomputed_list = [precomputed_all]
    else:
        precomputed_list = list(precomputed_all or [])

    rgb_path, depth_path, cam_path = save_sam6d_input_frame(frame, run_dir)
    cad_path = export_cad_mm(mesh_file, mesh_scale, run_dir / "cad_mm.ply")
    seg_path = run_dir / "sam6d_results" / "detection_ism.json"
    height, width = frame["depth_m"].shape[:2]

    entries: list[dict] = []
    metadata_by_index: dict[int, dict] = {}
    failed_results: list[dict] = []
    mask_t0 = time.perf_counter()
    for instance_index in range(requested_count):
        item_dir = run_dir / f"instance_{instance_index:02d}"
        item_dir.mkdir(parents=True, exist_ok=True)
        if instance_index >= len(precomputed_list):
            failed_results.append(
                {
                    "object_name": object_name,
                    "ok": False,
                    "sam3_instance_index": int(instance_index),
                    "run_dir": str(item_dir),
                    "error": f"SAM3 produced only {len(precomputed_list)} instance(s), requested {requested_count}",
                }
            )
            continue
        try:
            precomp_instance_index, precomputed = _select_sam3_precomputed_instance(
                precomputed_list,
                object_name,
                instance_index,
            )
            mask_u8 = cv2.imread(str(precomputed["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if mask_u8 is None:
                raise RuntimeError(f"failed to read precomputed SAM3 mask: {precomputed['mask_path']}")
            mask = np.asarray(mask_u8 > 0, dtype=bool)
            selected = precomputed.get("selected", {})
            det_score = float(selected.get("model_score", 1.0))
            box_value = precomputed.get("mask_bbox")
            if box_value is None:
                box_value = _mask_bbox_xyxy(mask)
            if box_value is None:
                raise RuntimeError(f"SAM3 produced an empty mask for {object_name} instance {instance_index}")
            box = _clip_box_xyxy(box_value, width, height, pad_frac=0.0)
            depth_repair = repair_depth_for_mask_if_needed(frame, mask, box, args, depth_path)
            mask_source = (
                f"sam3_text_precomputed_score_{det_score:.3f}"
                f"_ms_{float(precomputed.get('infer_elapsed_ms', 0.0)):.1f}"
            )
            pem_score = max(det_score, float(args.pem_det_score_thresh) + 1e-4)
            detection_entry = make_detection_ism_entry(
                mask,
                box,
                pem_score,
                f"sam3_text_i{precomp_instance_index}",
                object_name=object_name,
                sam3_instance_index=int(precomp_instance_index),
            )
            entries.append(detection_entry)
            save_detection_visual(
                frame,
                box,
                mask,
                item_dir / "groundingdino_depth_mask.png",
                f"sam3_text_i{precomp_instance_index}",
                det_score,
            )
            save_binary_mask(mask, item_dir / "sam6d_binary_mask.png")
            metadata_by_index[int(precomp_instance_index)] = {
                "object_name": object_name,
                "prompt": prompt,
                "run_dir": str(item_dir),
                "sam3_instance_index": int(precomp_instance_index),
                "score": det_score,
                "mask_source": mask_source,
                "mask_pixels": int(np.count_nonzero(mask)),
                "depth_repair": depth_repair,
                "detection_box_xyxy": box.tolist(),
                "detection_ism": detection_entry,
                "_mask": mask,
            }
            print(
                f"[sam6d-gdino] batch object={object_name} instance={precomp_instance_index} "
                f"score={det_score:.3f} box={[round(float(v), 1) for v in box.tolist()]} "
                f"mask_pixels={int(np.count_nonzero(mask))} mask_source={mask_source}"
            )
        except Exception as exc:
            failed_results.append(
                {
                    "object_name": object_name,
                    "ok": False,
                    "sam3_instance_index": int(instance_index),
                    "run_dir": str(item_dir),
                    "error": repr(exc),
                }
            )
            print(f"[sam6d-gdino] batch object={object_name} instance={instance_index} failed: {exc!r}")

    mask_elapsed_ms = (time.perf_counter() - mask_t0) * 1000.0
    if entries:
        write_detection_ism_entries(entries, seg_path)

    with open(run_dir / "groundingdino_result.json", "w") as f:
        json.dump(
            {
                "prompt": prompt,
                "requested_instances": requested_count,
                "selected_instances": [
                    {key: value for key, value in meta.items() if key != "_mask"}
                    for _, meta in sorted(metadata_by_index.items())
                ],
                "failed_instances": failed_results,
                "mask_elapsed_ms": float(mask_elapsed_ms),
                "detection_ism_path": str(seg_path) if entries else "",
            },
            f,
            indent=2,
            default=lambda x: np.asarray(x).tolist(),
        )

    print(
        f"[sam6d-gdino] batch same-object instances: {len(entries)}/{requested_count} "
        f"object={object_name} mask_ms={mask_elapsed_ms:.2f} run_dir={run_dir}"
    )

    if not entries:
        result_path = run_dir / "sam6d_multi_instance_pose_results.json"
        with open(result_path, "w") as f:
            json.dump({"results": failed_results}, f, indent=2)
        print("[sam6d-gdino] batch result:", result_path)
        return failed_results

    if args.skip_pem:
        results = []
        for _, meta in sorted(metadata_by_index.items()):
            item = {key: value for key, value in meta.items() if key != "_mask"}
            item["ok"] = True
            item["mask_elapsed_ms"] = float(mask_elapsed_ms)
            results.append(item)
        results.extend(failed_results)
        result_path = run_dir / "sam6d_multi_instance_pose_results.json"
        with open(result_path, "w") as f:
            json.dump({"results": results}, f, indent=2)
        print(f"[sam6d-gdino] skipped PEM; detection_ism={seg_path}")
        print("[sam6d-gdino] batch result:", result_path)
        return results

    template_dir = prepare_templates(args, sam6d_root, run_dir, cad_path, object_name=object_name, mesh_file=mesh_file, mesh_scale=mesh_scale)
    pem_t0 = time.perf_counter()
    pem_json = run_sam6d_pem(
        args,
        sam6d_root,
        run_dir,
        cad_path,
        rgb_path,
        depth_path,
        cam_path,
        seg_path,
        template_dir=template_dir,
        object_name=object_name,
        mesh_file=mesh_file,
        mesh_scale=mesh_scale,
    )
    pem_elapsed_ms = (time.perf_counter() - pem_t0) * 1000.0
    parsed = parse_pem_results(pem_json)
    mesh = _load_scene_mesh(mesh_file, mesh_scale)
    returned_indices: set[int] = set()
    results: list[dict] = []
    sorted_metadata = [meta for _, meta in sorted(metadata_by_index.items())]
    for pem_order, (pem_det, T_cam_obj) in enumerate(parsed):
        try:
            instance_index = int(pem_det.get("sam3_instance_index", sorted_metadata[pem_order]["sam3_instance_index"]))
        except Exception:
            instance_index = int(sorted_metadata[min(pem_order, len(sorted_metadata) - 1)]["sam3_instance_index"])
        meta = metadata_by_index.get(instance_index)
        if meta is None:
            meta = sorted_metadata[min(pem_order, len(sorted_metadata) - 1)]
            instance_index = int(meta["sam3_instance_index"])
        returned_indices.add(instance_index)
        raw_T_cam_obj = T_cam_obj.copy()
        refine_info = refine_pem_translation_against_mask_depth(
            args,
            frame,
            object_name,
            mesh,
            T_cam_obj,
            meta["_mask"],
            Path(meta["run_dir"]),
            float(pem_det.get("score", 0.0)),
        )
        if bool(refine_info.get("applied", False)):
            T_cam_obj = np.asarray(T_cam_obj, dtype=np.float32).copy()
            T_cam_obj[:3, 3] = np.asarray(refine_info["refined_translation_m"], dtype=np.float32).reshape(3)
        score = float(pem_det.get("score", 0.0))
        item = {key: value for key, value in meta.items() if key != "_mask"}
        try:
            validate_pem_pose_result(args, object_name, score, T_cam_obj, refine_info)
            item.update(
                {
                    "ok": True,
                    "sam3_instance_index": int(instance_index),
                    "score": score,
                    "mask_elapsed_ms": float(mask_elapsed_ms),
                    "pem_elapsed_ms": float(pem_elapsed_ms),
                    "pem_batch_instance_count": int(len(entries)),
                    "T_cam_obj_raw_pem": raw_T_cam_obj.tolist(),
                    "translation_m_raw_pem": raw_T_cam_obj[:3, 3].tolist(),
                    "T_cam_obj": T_cam_obj.tolist(),
                    "translation_m": T_cam_obj[:3, 3].tolist(),
                    "pem_refine": refine_info,
                    "pem_detection": pem_det,
                }
            )
        except Exception as exc:
            item.update(
                {
                    "ok": False,
                    "sam3_instance_index": int(instance_index),
                    "score": score,
                    "mask_elapsed_ms": float(mask_elapsed_ms),
                    "pem_elapsed_ms": float(pem_elapsed_ms),
                    "pem_batch_instance_count": int(len(entries)),
                    "T_cam_obj_raw_pem": raw_T_cam_obj.tolist(),
                    "translation_m_raw_pem": raw_T_cam_obj[:3, 3].tolist(),
                    "pem_refine": refine_info,
                    "pem_detection": pem_det,
                    "error": repr(exc),
                }
            )
        result_path = Path(meta["run_dir"]) / "sam6d_pose_result.json"
        with open(result_path, "w") as f:
            json.dump(item, f, indent=2)
        results.append(item)
        print(
            f"[sam6d-gdino] instance={instance_index} PEM score={score:.4f} "
            f"T_cam_obj translation(m): {T_cam_obj[:3, 3].tolist()}"
        )

    for instance_index, meta in sorted(metadata_by_index.items()):
        if instance_index in returned_indices:
            continue
        failed_results.append(
            {
                "object_name": object_name,
                "ok": False,
                "sam3_instance_index": int(instance_index),
                "run_dir": str(meta["run_dir"]),
                "error": "SAM-6D PEM skipped this detection; mask/depth did not yield enough points",
                "mask_pixels": int(meta.get("mask_pixels", 0)),
                "depth_repair": meta.get("depth_repair", {}),
            }
        )
    results.extend(failed_results)
    result_path = run_dir / "sam6d_multi_instance_pose_results.json"
    with open(result_path, "w") as f:
        json.dump({"results": results, "pem_elapsed_ms": float(pem_elapsed_ms)}, f, indent=2)
    print("[sam6d-gdino] pem_ms:", f"{pem_elapsed_ms:.2f}", "mode:", str(args.pem_run_mode), "batch_instances:", len(entries))
    print("[sam6d-gdino] batch result:", result_path)
    return results


def _unique_object_names_for_sam3(args, object_names: list[str]) -> list[str]:
    unique_names: list[str] = []
    seen: set[str] = set()
    for name in object_names:
        item_args = copy.copy(args)
        item_args.object_name = name
        try:
            object_name, _, _, _ = resolve_object_inputs(item_args)
        except Exception:
            object_name = normalize_object_name(name) or str(name)
        key = normalize_object_name(object_name) or str(object_name)
        if key in seen:
            continue
        seen.add(key)
        unique_names.append(str(name))
    return unique_names


def _sam3_result_map_from_payload(args, payload: dict) -> dict:
    result_map = {}
    multi_instances = bool(getattr(args, "sam3_full_scene_keep_multi_instances", False))
    max_instances_per_object = max(1, int(getattr(args, "sam3_max_masks_per_item", 1)))
    for item in payload.get("results", []):
        if not item.get("ok", True):
            continue
        name = normalize_object_name(item.get("object_name") or item.get("id") or "")
        if not name:
            continue
        if multi_instances:
            bucket = result_map.setdefault(name, [])
            bucket.append(item)
        else:
            old = result_map.get(name)
            old_score = float(old.get("selected", {}).get("model_score", -1.0)) if isinstance(old, dict) else -1.0
            new_score = float(item.get("selected", {}).get("model_score", -1.0))
            if old is None or new_score > old_score:
                result_map[name] = item
    if multi_instances:
        for key, value in list(result_map.items()):
            result_map[key] = _dedupe_sam3_instances(value, max_instances_per_object)
    return result_map


def _sam3_payload_has_cuda_oom(payload: dict) -> bool:
    for item in payload.get("results", []):
        error = str(item.get("error", ""))
        if "OutOfMemoryError" in error or "CUDA out of memory" in error:
            return True
    return False


def load_sam3_full_scene_text_masks(args, result_path: Path, object_names: list[str]) -> dict:
    result_path = Path(result_path).expanduser()
    if not result_path.exists():
        raise FileNotFoundError(f"SAM3 full-scene result json not found: {result_path}")
    with open(result_path, "r") as f:
        payload = json.load(f)
    result_map = _sam3_result_map_from_payload(args, payload)
    sam3_object_names = _unique_object_names_for_sam3(args, object_names)
    print(
        f"[sam3] reused full-scene text masks: {len(result_map)}/{len(sam3_object_names)} "
        f"result={result_path}"
    )
    return result_map


def run_sam3_full_scene_text_masks(args, frame: dict, scene_dir: Path, object_names: list[str], rgb_path: Path, pem_warmup_func=None) -> dict:
    sam3_python = Path(args.sam3_python).expanduser()
    sam3_script = Path(args.sam3_provider_script).expanduser()
    if not sam3_python.exists():
        raise FileNotFoundError(f"SAM3 python not found: {sam3_python}")
    if not sam3_script.exists():
        raise FileNotFoundError(f"SAM3 provider script not found: {sam3_script}")

    output_dir = Path(scene_dir) / "sam3_full_scene_text"
    output_dir.mkdir(parents=True, exist_ok=True)
    sam3_object_names = _unique_object_names_for_sam3(args, object_names)
    items = []
    for name in sam3_object_names:
        item_args = copy.copy(args)
        item_args.object_name = name
        object_name, prompt, _, _ = resolve_object_inputs(item_args)
        for variant_index, variant_prompt in enumerate(sam3_text_prompt_variants(object_name, prompt)):
            items.append(
                {
                    "id": f"{object_name}_p{variant_index}",
                    "object_name": object_name,
                    "mode": "text",
                    "prompt": variant_prompt,
                }
            )
    items_path = output_dir / "sam3_items.json"
    with open(items_path, "w") as f:
        json.dump(items, f, indent=2)

    cmd = [
        str(sam3_python),
        str(sam3_script),
        "--rgb-path",
        str(Path(rgb_path).expanduser()),
        "--output-dir",
        str(output_dir),
        "--items-json",
        str(items_path),
        "--mode",
        "text",
        "--confidence-threshold",
        str(float(args.sam3_confidence_threshold)),
        "--resolution",
        str(int(args.sam3_resolution)),
        "--min-mask-area",
        str(int(args.min_mask_area)),
        "--morph-kernel",
        str(int(args.sam3_morph_kernel)),
        "--sam3-max-masks-per-item",
        str(max(1, int(args.sam3_max_masks_per_item))),
    ]
    if args.sam3_checkpoint_path:
        cmd += ["--checkpoint-path", str(Path(args.sam3_checkpoint_path).expanduser())]
    if args.sam3_device:
        cmd += ["--device", str(args.sam3_device)]

    stdout_path = output_dir / "sam3_stdout.txt"
    stderr_path = output_dir / "sam3_stderr.txt"
    t0 = time.perf_counter()
    warmup_info = {"attempted": False, "ok": False, "elapsed_ms": 0.0}

    def run_with_optional_warmup(enable_warmup: bool):
        nonlocal warmup_info
        warmup_info = {"attempted": False, "ok": False, "elapsed_ms": 0.0}
        if not enable_warmup or pem_warmup_func is None:
            return subprocess.run(
                cmd,
                cwd=str(Path(sam3_script).parent),
                text=True,
                capture_output=True,
                env=_sam3_subprocess_env(),
            )

        warmup_info["attempted"] = True

        def _warmup():
            wt0 = time.perf_counter()
            try:
                pem_warmup_func()
                warmup_info["ok"] = True
            except Exception as exc:
                warmup_info["error"] = repr(exc)
            finally:
                warmup_info["elapsed_ms"] = float((time.perf_counter() - wt0) * 1000.0)

        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(sam3_script).parent),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_sam3_subprocess_env(),
        )
        thread = threading.Thread(target=_warmup, name="sam6d_pem_warmup", daemon=True)
        thread.start()
        stdout, stderr = proc.communicate()
        thread.join()
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)

    enable_warmup = bool(getattr(args, "pem_warmup_during_sam3", False)) and str(getattr(args, "pem_run_mode", "inprocess")) == "inprocess"
    result_path = output_dir / "sam3_batch_result.json"

    def run_and_load(enable: bool):
        proc = run_with_optional_warmup(enable)
        elapsed = (time.perf_counter() - t0) * 1000.0
        stdout_path.write_text(proc.stdout or "")
        stderr_path.write_text(proc.stderr or "")
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-24:])
            raise RuntimeError(f"SAM3 full-scene text subprocess failed with code {proc.returncode}: {tail}")
        if not result_path.exists():
            raise RuntimeError(f"SAM3 full-scene text did not write {result_path}")
        with open(result_path, "r") as f:
            loaded_payload = json.load(f)
        return proc, elapsed, loaded_payload, _sam3_result_map_from_payload(args, loaded_payload)

    proc, elapsed_ms, payload, result_map = run_and_load(enable_warmup)
    if warmup_info.get("attempted") and not result_map and _sam3_payload_has_cuda_oom(payload):
        print("[sam3] SAM3 item inference hit CUDA OOM during concurrent PEM warmup; retrying without PEM warmup")
        t0 = time.perf_counter()
        proc, elapsed_ms, payload, result_map = run_and_load(False)
    payload["subprocess_elapsed_ms"] = float(elapsed_ms)
    payload["pem_warmup_during_sam3"] = warmup_info
    payload["result_map_keys"] = sorted(result_map.keys())
    with open(result_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"[sam3] full-scene text masks: {len(result_map)}/{len(sam3_object_names)} "
        f"elapsed_ms={elapsed_ms:.1f} result={result_path}"
    )
    _confirm_sam3_full_scene_text_masks(args, frame, output_dir, sam3_object_names, result_map)
    return result_map


def save_sam3_full_scene_mask_visualization(frame: dict, output_dir: Path, object_names: list[str], result_map: dict) -> Path:
    canvas = np.asarray(frame["bgr"], dtype=np.uint8).copy()
    overlay = canvas.copy()
    palette = [
        (0, 255, 0),
        (255, 128, 0),
        (0, 128, 255),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
        (128, 255, 128),
        (255, 128, 255),
        (128, 128, 255),
        (80, 220, 220),
    ]
    missing = []
    rendered = []
    for idx, raw_name in enumerate(object_names):
        name = normalize_object_name(raw_name) or str(raw_name)
        items = result_map.get(name)
        if not items:
            missing.append(name)
            continue
        if isinstance(items, dict):
            items = [items]
        for local_idx, item in enumerate(items):
            mask_path = item.get("mask_path")
            mask_u8 = None if mask_path is None else cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_u8 is None:
                missing.append(name)
                continue
            mask_bool = np.asarray(mask_u8 > 0, dtype=bool)
            if not np.any(mask_bool):
                missing.append(name)
                continue
            color = np.asarray(palette[(idx + local_idx) % len(palette)], dtype=np.uint8)
            overlay[mask_bool] = (0.45 * overlay[mask_bool] + 0.55 * color).astype(np.uint8)
            bbox = np.asarray(item.get("mask_bbox", _mask_bbox_xyxy(mask_bool)), dtype=np.float64).reshape(-1)[:4]
            x1, y1, x2, y2 = np.round(bbox).astype(int).tolist()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), tuple(int(v) for v in color.tolist()), 2)
            score = float(item.get("selected", {}).get("model_score", 0.0))
            index_label = ""
            if local_idx > 0:
                index_label = f"#{local_idx}"
            label = f"{name}{index_label} {score:.2f}"
            ty = max(18 + 16 * local_idx, y1 - 6)
            cv2.putText(
                overlay,
                label,
                (max(0, x1), ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                label,
                (max(0, x1), ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                tuple(int(v) for v in color.tolist()),
                1,
                cv2.LINE_AA,
            )
            rendered.append(f"{name}{index_label}")

    canvas = cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0.0)
    status = f"SAM3 masks {len(rendered)}/{max(len(object_names), len(rendered), 1)}"
    cv2.putText(canvas, status, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, status, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 1, cv2.LINE_AA)
    if missing:
        text = "missing: " + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        cv2.putText(canvas, text, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 1, cv2.LINE_AA)

    out_path = Path(output_dir) / "sam3_full_scene_masks_overlay.png"
    cv2.imwrite(str(out_path), canvas)
    return out_path


def _show_image_window_best_effort(window_name: str, image_path: Path) -> None:
    try:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("[sam3] no GUI display detected; inspect the saved overlay path instead")
            return
        for command in (
            ["eog", str(image_path)],
            ["xdg-open", str(image_path)],
            ["gio", "open", str(image_path)],
        ):
            exe = shutil.which(command[0])
            if exe is None:
                continue
            subprocess.Popen(
                [exe, *command[1:]],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            print(f"[sam3] opened mask overlay viewer: {image_path}")
            return
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image)
        for _ in range(8):
            cv2.waitKey(50)
    except Exception as exc:
        print(f"[sam3] mask preview window unavailable: {type(exc).__name__}: {exc}")


def _confirm_sam3_full_scene_text_masks(args, frame: dict, output_dir: Path, object_names: list[str], result_map: dict) -> None:
    object_names = [normalize_object_name(name) or str(name) for name in object_names]
    found = sorted(result_map.keys())
    missing = [name for name in object_names if name not in result_map]
    vis_path = save_sam3_full_scene_mask_visualization(frame, output_dir, object_names, result_map)
    print(f"[sam3] full-scene mask overlay: {vis_path}")
    print(f"[sam3] found masks: {found}")
    if missing:
        print(f"[sam3] missing masks: {missing}")

    confirm = bool(getattr(args, "sam3_full_scene_mask_confirm", False))
    require_all = bool(getattr(args, "sam3_require_full_scene_masks", False))
    if confirm and bool(getattr(args, "sam3_show_full_scene_mask_window", True)):
        _show_image_window_best_effort("SAM3 full-scene masks", vis_path)

    if confirm:
        while True:
            if missing and require_all:
                prompt = "[sam3] mask check failed. Type r then Enter to retry segmentation, or press Enter/q to abort: "
            else:
                prompt = "[sam3] inspect masks. Press Enter to continue, type r then Enter to retry segmentation, or q to abort: "
            try:
                answer = input(prompt).strip().lower()
            except EOFError:
                answer = "q"
            if answer == "r":
                raise SAM3FullSceneMaskRetryRequested("user requested SAM3 full-scene mask retry")
            if answer == "q" or (missing and require_all and answer == ""):
                raise RuntimeError(f"SAM3 full-scene masks are incomplete; missing={missing}; overlay={vis_path}")
            if answer == "":
                break
            print("[sam3] expected Enter, r, or q.")
    elif missing and require_all:
        raise RuntimeError(f"SAM3 full-scene masks are incomplete; missing={missing}; overlay={vis_path}")


def sam3_text_prompt_variants(object_name: str, prompt: str) -> list[str]:
    normalized = normalize_object_name(object_name) or str(object_name)
    variants = [str(prompt)]
    if normalized == "red_bricks_cube":
        variants += [
            "square plastic building block.",
            "transparent square building block.",
            "blue transparent square block.",
            "green transparent square block.",
            "small square toy block.",
            "square plastic panel.",
        ]
    elif normalized == "shuazi":
        variants += [
            "white plastic brush.",
            "white brush.",
            "cleaning brush.",
        ]
    elif normalized == "bitong":
        variants += [
            "small beige cup.",
            "beige cup.",
            "pen cup.",
            "yellow cup.",
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


def validate_object_name_args(object_names: list[str]):
    known = set(list_object_spec_names())
    bad_names = []
    option_like = []
    for raw in object_names:
        text = str(raw)
        normalized = normalize_object_name(text)
        if "--" in text:
            option_like.append(text)
        if normalized not in known:
            bad_names.append(text)
    if option_like:
        raise ValueError(
            "object name argument contains '--': "
            f"{option_like}. Did you miss a space before an option? "
            "Example: ... tennis --mask-mode sam3_text"
        )
    if bad_names:
        raise ValueError(
            "unknown object name(s): "
            f"{bad_names}. If this was a mask mode, use '--mask-mode sam3_text' "
            "with underscore, not 'sam3-text'."
        )


def apply_frame_dir_args(args):
    frame_dir = getattr(args, "frame_dir", None)
    if not frame_dir:
        return
    frame_dir = Path(frame_dir).expanduser()
    rgb_path = frame_dir / "rgb.png"
    depth_path = frame_dir / "depth.png"
    camera_path = frame_dir / "camera.json"
    missing = [str(path) for path in (rgb_path, depth_path, camera_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"--frame-dir is missing required file(s): {missing}")
    if args.rgb_path is None:
        args.rgb_path = str(rgb_path)
    if args.depth_path is None:
        args.depth_path = str(depth_path)
    if args.camera_path is None:
        args.camera_path = str(camera_path)


def main():
    args = parse_args()
    apply_frame_dir_args(args)
    offline = args.rgb_path is not None or args.depth_path is not None or args.camera_path is not None
    if offline and not (args.rgb_path and args.depth_path and args.camera_path):
        raise ValueError("--rgb-path, --depth-path, and --camera-path must be provided together")

    object_names = list(args.object_names or ([] if args.object_name is None else [args.object_name]))
    if not object_names:
        raise ValueError("provide --object-name for one object or --object-names for multiple objects")
    validate_object_name_args(object_names)
    if args.bbox is not None and len(object_names) > 1:
        raise ValueError("--bbox is single-object only; omit it when using --object-names")

    sam6d_root = Path(args.sam6d_root).expanduser().resolve()
    if not (sam6d_root / "Pose_Estimation_Model" / "run_inference_custom.py").exists():
        raise FileNotFoundError(f"invalid SAM-6D root: {sam6d_root}")

    # BlenderProc renders from SAM-6D/Render, so every artifact path passed to
    # it must remain valid after the subprocess changes its working directory.
    output_root = Path(args.output_root).expanduser().resolve()

    if len(object_names) == 1:
        single_name = object_names[0]
        if str(args.mask_mode) == "sam3_text" and bool(getattr(args, "sam3_full_scene_keep_multi_instances", False)):
            run_idx = 0
            while True:
                if run_idx > 0:
                    source = "offline frame" if offline else "camera"
                    print(f"[sam3] retrying full-scene masks from {source}; attempt={run_idx + 1}")
                frame = load_offline_frame(args) if offline else capture_realsense_frame(args)
                scene_dir = output_root / f"{_now_stamp()}_full_scene_1objects_pid{os.getpid()}_single{'' if run_idx == 0 else f'_retry{run_idx}'}"
                scene_dir.mkdir(parents=True, exist_ok=True)
                detector_cache: dict = {}
                shared_frame_dir = scene_dir / "shared_frame"
                shared_frame_dir.mkdir(parents=True, exist_ok=True)
                shared_rgb_path, _, _ = save_sam6d_input_frame(frame, shared_frame_dir)
                try:
                    reused_sam3_result = str(getattr(args, "sam3_full_scene_result_json", "") or "").strip()
                    if reused_sam3_result:
                        detector_cache["sam3_text_results"] = load_sam3_full_scene_text_masks(
                            args,
                            Path(reused_sam3_result),
                            [single_name],
                        )
                    else:
                        detector_cache["sam3_text_results"] = run_sam3_full_scene_text_masks(
                            args,
                            frame,
                            scene_dir,
                            [single_name],
                            shared_rgb_path,
                        )
                    detector_cache["sam3_text_full_scene_attempted"] = True
                except SAM3FullSceneMaskRetryRequested:
                    run_idx += 1
                    continue
                break
            run_dir = scene_dir / f"01_{normalize_object_name(single_name) or _safe_cache_name(single_name)}"
            single_args = copy.copy(args)
            single_args.object_name = single_name
            _run_single_object_pose(single_args, frame, sam6d_root, run_dir, detector_cache=detector_cache)
            return

        frame = load_offline_frame(args) if offline else capture_realsense_frame(args)
        detector_cache = {}
        single_args = copy.copy(args)
        single_args.object_name = single_name
        run_dir = output_root / f"{_now_stamp()}_{normalize_object_name(single_name) or single_name}_pid{os.getpid()}"
        _run_single_object_pose(single_args, frame, sam6d_root, run_dir, detector_cache=detector_cache)
        return

    retry_index = 0
    while True:
        if retry_index > 0:
            source = "offline frame" if offline else "camera"
            print(f"[sam3] retrying full-scene masks from {source}; attempt={retry_index + 1}")
        frame = load_offline_frame(args) if offline else capture_realsense_frame(args)
        detector_cache: dict = {}
        retry_suffix = "" if retry_index == 0 else f"_retry{retry_index}"
        scene_dir = output_root / f"{_now_stamp()}_full_scene_{len(object_names)}objects_pid{os.getpid()}{retry_suffix}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        with open(scene_dir / "run_args.json", "w") as f:
            json.dump({"args": vars(args), "object_names": object_names, "retry_index": retry_index}, f, indent=2)
        shared_frame_dir = scene_dir / "shared_frame"
        shared_frame_dir.mkdir(parents=True, exist_ok=True)
        shared_rgb_path, _, _ = save_sam6d_input_frame(frame, shared_frame_dir)

        try:
            if str(args.mask_mode) == "sam3_text" and (
                len(object_names) > 1 or bool(getattr(args, "sam3_full_scene_keep_multi_instances", False))
            ):
                reused_sam3_result = str(getattr(args, "sam3_full_scene_result_json", "") or "").strip()
                if reused_sam3_result:
                    detector_cache["sam3_text_results"] = load_sam3_full_scene_text_masks(
                        args,
                        Path(reused_sam3_result),
                        object_names,
                    )
                else:
                    def _warm_pem_model():
                        runner = _get_sam6d_pem_runner(args, sam6d_root)
                        runner._load_model_once()

                    detector_cache["sam3_text_results"] = run_sam3_full_scene_text_masks(
                        args,
                        frame,
                        scene_dir,
                        object_names,
                        shared_rgb_path,
                        pem_warmup_func=_warm_pem_model,
                    )
                detector_cache["sam3_text_full_scene_attempted"] = True
        except SAM3FullSceneMaskRetryRequested:
            retry_index += 1
            continue
        break

    resolved_multi_names = []
    for name in object_names:
        item_args = copy.copy(args)
        item_args.object_name = name
        resolved_name, _, _, _ = resolve_object_inputs(item_args)
        resolved_multi_names.append(resolved_name)
    same_object_batch = (
        len(object_names) > 1
        and len(set(resolved_multi_names)) == 1
        and str(args.mask_mode) == "sam3_text"
        and args.bbox is None
        and bool(getattr(args, "sam3_full_scene_keep_multi_instances", False))
    )

    if same_object_batch:
        item_args = copy.copy(args)
        item_args.object_name = object_names[0]
        safe_name = normalize_object_name(object_names[0]) or _safe_cache_name(object_names[0])
        item_dir = scene_dir / f"01_{safe_name}_batch{len(object_names)}"
        results = _run_same_object_multi_instance_pose(
            item_args,
            frame,
            sam6d_root,
            item_dir,
            detector_cache=detector_cache,
            instance_count=len(object_names),
        )
    else:
        results = []
        for index, name in enumerate(object_names):
            item_args = copy.copy(args)
            item_args.object_name = name
            safe_name = normalize_object_name(name) or _safe_cache_name(name)
            item_dir = scene_dir / f"{index + 1:02d}_{safe_name}"
            try:
                result = _run_single_object_pose(item_args, frame, sam6d_root, item_dir, detector_cache=detector_cache)
                result["ok"] = True
            except Exception as exc:
                result = {"object_name": name, "ok": False, "error": repr(exc), "run_dir": str(item_dir)}
                print(f"[sam6d-gdino] object={name} failed: {exc!r}")
            results.append(result)

    summary = {
        "scene_dir": str(scene_dir),
        "object_count": len(object_names),
        "ok_count": sum(1 for item in results if item.get("ok")),
        "results": results,
    }
    if not bool(args.skip_pem) and bool(getattr(args, "full_scene_pem_visualization", True)):
        vis_info = save_full_scene_pem_visualization(args, frame, results, scene_dir / "full_scene_pem_overlay.png")
        summary["full_scene_pem_visualization"] = vis_info
        print(f"[sam6d-gdino] full-scene PEM overlay: {vis_info['path']} rendered={len(vis_info['rendered'])}")
    summary_path = scene_dir / "full_scene_pose_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[sam6d-gdino] full scene result: {summary_path}")
    print(f"[sam6d-gdino] ok_count={summary['ok_count']}/{summary['object_count']}")


if __name__ == "__main__":
    main()
