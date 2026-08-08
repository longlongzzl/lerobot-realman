#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import select
import string
import time
import sys
from pathlib import Path

import cv2
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import trimesh
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.wrappers.record import RecordEpisode
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat, quat2mat

from rm75_app.assets.object_specs import describe_object_specs, get_object_spec, list_object_spec_names, normalize_object_name, resolve_object_spec_scales
from rm75_app.paths import APP_ROOT, DEFAULT_CAMERA_EXTRINSIC, RUNTIME_DIR


DEFAULT_FOUNDATIONPOSE_ROOT = "/home/zhangzhao/PycharmProjects/FoundationPose"
DEFAULT_CAMERA_EXTRINSIC_PATH = str(DEFAULT_CAMERA_EXTRINSIC)
DEFAULT_EXTRA_MANISKILL_PACKAGE_ROOT = "/home/zhangzhao/anaconda3/envs/realman/lib/python3.11/site-packages/mani_skill"
DEFAULT_PICK_SCRIPT = str(APP_ROOT / "rm75_app" / "execution" / "pick_move_v10.py")
DEFAULT_MESH_FILE = "/home/zhangzhao/anaconda3/envs/realman/lib/python3.11/site-packages/mani_skill/envs/tasks/digital_twins/so101_arm_with_two_cameras/jiaobang.glb"
DEFAULT_MESH_SCALE = 0.1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Initialize jiaobang pose from realtime FoundationPose, then run the RM75 pick-and-place motion-planning pipeline."
    )
    parser.add_argument("--env-id", type=str, default="Two_finger_PickJiaobang-v1")
    parser.add_argument("--extra-maniskill-package-root", type=str, default=DEFAULT_EXTRA_MANISKILL_PACKAGE_ROOT)
    parser.add_argument("--pick-script-path", type=str, default=DEFAULT_PICK_SCRIPT)
    parser.add_argument("--foundationpose-root", type=str, default=DEFAULT_FOUNDATIONPOSE_ROOT)
    parser.add_argument("--object-name", type=str, default=None, help="Object-spec key from object_specs.py.")
    parser.add_argument("--list-objects", action="store_true", help="List available object specs and exit.")

    parser.add_argument("--mesh-file", type=str, default=DEFAULT_MESH_FILE)
    parser.add_argument("--mesh-scale", type=float, default=DEFAULT_MESH_SCALE)
    parser.add_argument("--sim-asset-file", type=str, default=None)
    parser.add_argument("--sim-asset-scale", type=float, default=None)
    parser.add_argument("--camera-extrinsic-opencv-path", type=str, default=DEFAULT_CAMERA_EXTRINSIC_PATH)
    parser.add_argument(
        "--use-direct-camera-extrinsic",
        action="store_true",
        help="Interpret camera_extrinsic_opencv as direct T_base_cam. Default matches the previous FoundationPose export/replay pipeline and applies inverse first.",
    )
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument(
        "--debug-dir",
        type=str,
        default=str(RUNTIME_DIR / "debug_jiaobang_foundationpose_init"),
    )
    parser.add_argument("--est-refine-iter", dest="est_refine_iter", type=int, default=5)
    parser.add_argument("--track-refine-iter", dest="track_refine_iter", type=int, default=2)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--foundationpose-stabilize-frames", type=int, default=6, help="After FoundationPose registration, run tracking for this many additional frames before using the pose. Set to 0 to disable.")
    parser.add_argument("--foundationpose-smoothing-window", type=int, default=4, help="Smooth the final pose by averaging the last few tracked poses. Set to 1 to disable smoothing.")
    parser.add_argument("--init-mask", type=str, default=None)
    parser.add_argument("--target-object-name", type=str, default=None, help="Use GroundingDINO to detect this object and initialize FoundationPose from the best matched box.")
    parser.add_argument("--grounding-dino-model-id", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--grounding-dino-box-threshold", type=float, default=0.25)
    parser.add_argument("--grounding-dino-text-threshold", type=float, default=0.2)
    parser.add_argument("--disable-object-spec-obstacles", action="store_true", help="Do not estimate non-target object_specs as additional scene obstacles for planning.")

    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-serial", type=str, default=None)

    parser.add_argument(
        "--foundationpose-position-offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
    )
    parser.add_argument(
        "--foundationpose-local-rotation-offset-deg",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("ROLL", "PITCH", "YAW"),
    )
    parser.add_argument("--no-map-foundationpose-through-robot-base", action="store_true")
    parser.add_argument("--lock-object-z-to-table", action="store_true")
    parser.add_argument("--min-object-center-z-margin", type=float, default=0.002, help="Clamp the mapped object upward so its lowest point stays at least this far above the default tabletop clearance.")
    parser.add_argument("--foundationpose-print-mapping-diagnostics", dest="foundationpose_print_mapping_diagnostics", action="store_true", default=True, help="Print raw FoundationPose camera-frame poses together with both direct/inverse camera-extrinsic world mappings for debugging. Enabled by default.")
    parser.add_argument("--no-foundationpose-print-mapping-diagnostics", dest="foundationpose_print_mapping_diagnostics", action="store_false", help="Disable FoundationPose mapping diagnostics for the target object.")
    parser.add_argument("--foundationpose-print-obstacle-mapping-diagnostics", action="store_true", help="Also print mapping diagnostics for scene obstacles. Disabled by default to reduce log spam.")
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--urdf-path", default=None)
    parser.add_argument("--srdf-path", default=None)
    parser.add_argument(
        "--variant",
        default="array_wxyz",
        choices=["array_wxyz", "array_xyzw", "pymp_pose_wxyz", "pymp_pose_xyzw"],
    )
    parser.add_argument("--pregrasp-height", type=float, default=0.05)
    parser.add_argument("--grasp-z-offset", type=float, default=0.0)
    parser.add_argument("--goal-z-offset", type=float, default=0.0)
    parser.add_argument("--yaw-offset-deg", type=float, default=180.0)
    parser.add_argument("--max-delta-per-step", type=float, default=0.05)
    parser.add_argument("--hold-steps", type=int, default=15)
    parser.add_argument("--open-steps", type=int, default=0)
    parser.add_argument("--close-steps", type=int, default=20)
    parser.add_argument("--gripper-open", type=float, default=-1.0)
    parser.add_argument("--gripper-close", type=float, default=1.0)
    parser.add_argument("--video-dir", default="./rm75_jiaobang_pick_move_foundationpose")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--goal-monitor-steps", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--log-contact-forces", action="store_true")
    parser.add_argument("--log-force-every", type=int, default=1)
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--auto-execute", action="store_true", help="Disable the interactive Enter-to-execute confirmation after each planned motion.")
    return parser.parse_args()


def maybe_print_and_exit_object_specs(args):
    if not getattr(args, "list_objects", False):
        return
    print("Available object specs:")
    print(describe_object_specs() or "(none)")
    raise SystemExit(0)



def apply_object_spec_defaults(args):
    maybe_print_and_exit_object_specs(args)
    if not hasattr(args, "grasp_mode") or args.grasp_mode is None:
        args.grasp_mode = "object_normal"
    spec = get_object_spec(getattr(args, "object_name", None))
    if args.object_name and spec is None:
        raise ValueError(f"Unknown object spec: {args.object_name}")

    if spec is None:
        if args.sim_asset_file is None:
            args.sim_asset_file = args.mesh_file
        if args.sim_asset_scale is None:
            args.sim_asset_scale = args.mesh_scale
        if args.mesh_file is not None:
            args.mesh_file = str(Path(args.mesh_file).expanduser())
        if args.sim_asset_file is not None:
            args.sim_asset_file = str(Path(args.sim_asset_file).expanduser())
        return None

    spec_mesh_scale, spec_sim_asset_scale = resolve_object_spec_scales(spec)
    if args.mesh_file == DEFAULT_MESH_FILE:
        args.mesh_file = spec.mesh_file
    if float(args.mesh_scale) == float(DEFAULT_MESH_SCALE):
        args.mesh_scale = spec_mesh_scale
    if args.sim_asset_file is None:
        args.sim_asset_file = spec.sim_asset_file or spec.mesh_file
    if args.sim_asset_scale is None:
        args.sim_asset_scale = spec_sim_asset_scale
    if args.target_object_name is None:
        args.target_object_name = spec.grounding_prompt
    if np.allclose(np.asarray(args.foundationpose_position_offset, dtype=np.float32), 0.0) and not np.allclose(np.asarray(spec.foundationpose_position_offset, dtype=np.float32), 0.0):
        args.foundationpose_position_offset = list(spec.foundationpose_position_offset)
    if np.allclose(np.asarray(args.foundationpose_local_rotation_offset_deg, dtype=np.float32), 0.0) and not np.allclose(np.asarray(spec.foundationpose_local_rotation_offset_deg, dtype=np.float32), 0.0):
        args.foundationpose_local_rotation_offset_deg = list(spec.foundationpose_local_rotation_offset_deg)
    if spec.pregrasp_height is not None and float(args.pregrasp_height) == 0.05:
        args.pregrasp_height = spec.pregrasp_height
    if spec.grasp_z_offset is not None and float(args.grasp_z_offset) == 0.0:
        args.grasp_z_offset = spec.grasp_z_offset
    if spec.goal_z_offset is not None and float(args.goal_z_offset) == 0.0:
        args.goal_z_offset = spec.goal_z_offset
    if getattr(args, "grasp_mode", "object_normal") == "object_normal" and getattr(spec, "grasp_mode", "object_normal") != "object_normal":
        args.grasp_mode = spec.grasp_mode
    if args.mesh_file is not None:
        args.mesh_file = str(Path(args.mesh_file).expanduser())
    if args.sim_asset_file is not None:
        args.sim_asset_file = str(Path(args.sim_asset_file).expanduser())
    return spec



def load_module_from_path(module_name: str, file_path: str | Path):
    file_path = Path(file_path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def flatten_np(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)



def resolve_foundationpose_root(path_like: str) -> Path:
    candidates = []
    if path_like:
        candidates.append(Path(path_like).expanduser())
    candidates.append(Path(DEFAULT_FOUNDATIONPOSE_ROOT))

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "run_realtime_demo.py").exists() and (candidate / "estimater.py").exists():
            return candidate

    raise FileNotFoundError(f"Failed to locate FoundationPose root from: {[str(p) for p in candidates]}")



def load_foundationpose_module(root: Path):
    script_path = root / "run_realtime_demo.py"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return load_module_from_path("foundationpose_jiaobang_bridge_impl", script_path)



def ensure_pick_jiaobang_env_registered(env_id: str, extra_maniskill_package_root: str) -> str:
    if env_id in gym.envs.registry:
        return env_id

    package_root = Path(extra_maniskill_package_root).expanduser()
    if package_root.name != "mani_skill" and (package_root / "mani_skill").exists():
        package_root = package_root / "mani_skill"
    if not package_root.exists():
        raise FileNotFoundError(f"Extra ManiSkill package root not found: {package_root}")

    import mani_skill.agents.robots as robots_pkg
    import mani_skill.envs.tasks.digital_twins as dt_pkg

    robots_path = str(package_root / "agents" / "robots")
    dt_path = str(package_root / "envs" / "tasks" / "digital_twins")
    if robots_path not in robots_pkg.__path__:
        robots_pkg.__path__.append(robots_path)
    if dt_path not in dt_pkg.__path__:
        dt_pkg.__path__.append(dt_path)

    importlib.import_module("mani_skill.agents.robots.realman")
    importlib.import_module("mani_skill.envs.tasks.digital_twins.so101_arm_with_two_cameras.pick_jiaobang")

    if env_id not in gym.envs.registry:
        raise gym.error.NameNotFound(f"Environment {env_id!r} still not registered after extending ManiSkill paths")
    return env_id



def load_matrix(path_like: str) -> np.ndarray:
    path = Path(path_like).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        mat = np.load(path)
    else:
        mat = np.loadtxt(path)
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix from {path}, got {mat.shape}")
    return mat



def _load_hf_component_compatible(load_fn, model_id: str, **kwargs):
    try:
        return load_fn(model_id, **kwargs)
    except OSError as exc:
        if "Unknown scheme for proxy URL" not in str(exc):
            raise

        print("GroundingDINO: unsupported proxy env detected, retrying without proxy using local cache")
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
    cache_key = (model_id, device)
    detector_cache = getattr(create_grounding_dino_detector, "_detector_cache", None)
    if isinstance(detector_cache, dict) and cache_key in detector_cache:
        return detector_cache[cache_key]

    print(f"GroundingDINO device: {device}")
    processor = _load_hf_component_compatible(
        AutoProcessor.from_pretrained,
        model_id,
        use_fast=False,
    )
    model = _load_hf_component_compatible(
        AutoModelForZeroShotObjectDetection.from_pretrained,
        model_id,
    ).to(device)
    model.eval()
    detector = {"processor": processor, "model": model, "device": device}
    if not isinstance(detector_cache, dict):
        detector_cache = {}
        setattr(create_grounding_dino_detector, "_detector_cache", detector_cache)
    detector_cache[cache_key] = detector
    return detector



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
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )

    result = results[0]
    boxes = result["boxes"]
    scores = result["scores"]
    text_labels = result["text_labels"]

    detections = []
    for idx, (box, score) in enumerate(zip(boxes, scores)):
        label = text_labels[idx] if idx < len(text_labels) else ""
        if isinstance(label, (list, tuple)):
            label = ". ".join(str(x) for x in label if str(x).strip())
        label = str(label).strip()
        if not label:
            continue
        detections.append({
            "box": np.asarray(box.detach().cpu().tolist(), dtype=np.float32),
            "score": float(score.detach().cpu().item()),
            "label": label,
        })
    return detections



def draw_grounding_dino_overlay(image_bgr: np.ndarray, detections, prompt: str, allow_skip: bool = False, redetect_hint: str = "Press r to re-detect current frame") -> np.ndarray:
    canvas = image_bgr.copy()
    best_idx = None
    if detections:
        best_idx = int(np.argmax([det["score"] for det in detections]))

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = [int(round(v)) for v in det["box"]]
        color = (0, 255, 0) if idx == best_idx else (0, 0, 255)
        caption = f'{det["label"]}: {det["score"]:.3f}'
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, caption, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    lines = [f"Prompt: {prompt}"]
    if detections:
        best = detections[best_idx]
        lines.append(f"Best: {best['label']} {best['score']:.3f}")
        lines.append("Press s to accept best box")
    else:
        lines.append("No valid detections")
    lines.append(redetect_hint)
    lines.append("Press m to draw a manual box")
    lines.append("Press q to skip this object" if allow_skip else "Press q to quit")

    y = 28
    for line in lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 28
    return canvas


def draw_grounding_dino_batch_overlay(image_bgr: np.ndarray, items, *, title: str = "GroundingDINO Batch Review") -> np.ndarray:
    canvas = image_bgr.copy()
    palette = [
        (0, 255, 0),
        (0, 200, 255),
        (255, 180, 0),
        (255, 0, 255),
        (0, 128, 255),
        (255, 0, 128),
        (128, 255, 0),
        (180, 180, 255),
    ]
    lines = [title]
    for idx, item in enumerate(items):
        object_name = str(item.get("object_name", f"obj{idx}"))
        det = item.get("selected_detection")
        if det is None:
            lines.append(f"{object_name}: missing")
            continue
        color = palette[idx % len(palette)]
        x1, y1, x2, y2 = [int(round(v)) for v in np.asarray(det["box"], dtype=np.float32).reshape(-1)[:4]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        caption = f"{object_name}: {float(det.get('score', 0.0)):.3f}"
        cv2.putText(canvas, caption, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        lines.append(f"{object_name}: ok ({float(det.get('score', 0.0)):.3f})")
    lines.append("Enter: accept all current boxes")
    lines.append("r: recapture and re-detect all")
    lines.append("object names: re-edit only those objects")
    lines.append("window keys also work: Enter / r / q")
    lines.append("q: abort scene capture")

    y = 28
    for line in lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        y += 26
    return canvas


def select_init_frame_and_mask_with_grounding_dino(reader, detector, args, *, prompt: str | None = None, allow_skip: bool = False, window_name: str = "GroundingDINO Init", fixed_frame=None, used_boxes=None, max_overlap_iou: float = 0.5, refresh_from_reader_on_r: bool = False):
    latest_frame = fixed_frame
    detections = []
    pending_detect = True
    grab_new_frame = fixed_frame is None
    prompt = str(prompt or args.target_object_name or "").strip()
    used_boxes = [np.asarray(box, dtype=np.float32) for box in (used_boxes or []) if box is not None]
    redetect_hint = "Press r to capture latest frame and re-detect" if refresh_from_reader_on_r else "Press r to re-detect current frame"

    def build_mask_from_box(x1, y1, x2, y2):
        h, w = latest_frame["color_bgr"].shape[:2]
        x1 = max(0, min(int(round(x1)), w - 1))
        x2 = max(x1 + 1, min(int(round(x2)), w))
        y1 = max(0, min(int(round(y1)), h - 1))
        y2 = max(y1 + 1, min(int(round(y2)), h))
        box = np.asarray([x1, y1, x2, y2], dtype=np.float32)
        if used_boxes and any(_bbox_iou(box, used_box) > max_overlap_iou for used_box in used_boxes):
            print(
                f"Rejected box {[int(v) for v in box.tolist()]} because it overlaps an already-used box "
                f"above IoU threshold {max_overlap_iou:.2f}."
            )
            return None, None
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        return mask, box

    while True:
        if grab_new_frame:
            frame = reader.get_frame()
            if frame is not None:
                latest_frame = frame
                grab_new_frame = False

        if latest_frame is None:
            continue

        if pending_detect:
            detections = run_grounding_dino_detector(
                detector,
                latest_frame["color"],
                prompt,
                args.grounding_dino_box_threshold,
                args.grounding_dino_text_threshold,
            )
            if used_boxes:
                detections = [
                    det for det in detections
                    if all(_bbox_iou(det["box"], used_box) <= max_overlap_iou for used_box in used_boxes)
                ]
            print(f"GroundingDINO detections={len(detections)} for prompt={prompt!r}")
            for det in detections:
                print(
                    f'  label={det["label"]}, score={det["score"]:.3f}, '
                    f'box={[round(v, 1) for v in det["box"].tolist()]}'
                )
            pending_detect = False

        display = draw_grounding_dino_overlay(
            latest_frame["color_bgr"],
            detections,
            prompt,
            allow_skip=allow_skip,
            redetect_hint=redetect_hint,
        )
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            if allow_skip:
                print(f"Skipped GroundingDINO/FoundationPose initialization for prompt={prompt!r}")
                return latest_frame, None, "skip"
            return None, None, "quit"
        if key == ord("r"):
            pending_detect = True
            if fixed_frame is None or refresh_from_reader_on_r:
                grab_new_frame = True
            continue
        if key == ord("m"):
            manual_frame = latest_frame["color_bgr"].copy()
            cv2.putText(
                manual_frame,
                "Drag ROI and press ENTER/SPACE to accept, c to cancel",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            roi = cv2.selectROI(window_name, manual_frame, fromCenter=False, showCrosshair=True)
            x, y, w, h = [int(v) for v in roi]
            if w <= 1 or h <= 1:
                print("Manual ROI cancelled. Press m to retry or r to refresh.")
                continue
            mask, box = build_mask_from_box(x, y, x + w, y + h)
            if mask is None:
                continue
            print(f"Accepted manual ROI box={[int(v) for v in box.tolist()]}")
            return latest_frame, mask, "accepted"
        if key in (ord("s"), 13):
            if not detections:
                print("No valid GroundingDINO detection to accept. Press r to retry or m to draw manually.")
                continue
            best = max(detections, key=lambda det: det["score"])
            mask, box = build_mask_from_box(*best["box"])
            if mask is None:
                continue
            print(
                f'Accepted GroundingDINO box label={best["label"]}, score={best["score"]:.3f}, '
                f'box={[int(v) for v in box.tolist()]}'
            )
            return latest_frame, mask, "accepted"


def _run_batch_grounding_dino_detector(frame, detector, args, review_items, *, max_overlap_iou: float = 0.5):
    height, width = frame["color_bgr"].shape[:2]
    used_boxes = []
    for item in review_items:
        detections = run_grounding_dino_detector(
            detector,
            frame["color"],
            item["prompt"],
            args.grounding_dino_box_threshold,
            args.grounding_dino_text_threshold,
        )
        selected = _select_best_unused_detection(detections, used_boxes, max_iou=max_overlap_iou)
        item["detections"] = detections
        item["selected_detection"] = selected
        item["frame"] = frame
        item["mask"] = _make_box_mask(height, width, selected["box"]) if selected is not None else None
        item["box"] = None if selected is None else np.asarray(selected["box"], dtype=np.float32).reshape(4)
        item["status"] = "accepted" if selected is not None else "missing"
        if item["box"] is not None:
            used_boxes.append(item["box"])
        print(
            f"[groundingdino batch] {item['object_name']}: "
            + (
                f"score={float(selected['score']):.3f}, box={[round(v, 1) for v in item['box'].tolist()]}"
                if selected is not None
                else "no valid detection"
            )
        )


def _parse_batch_review_names(answer: str, review_items) -> list[str]:
    tokens = [normalize_object_name(token) for token in answer.replace(",", " ").split()]
    tokens = [token for token in tokens if token]
    if not tokens:
        return []
    item_names = {normalize_object_name(item["object_name"]): item["object_name"] for item in review_items}
    if "all" in tokens:
        return [item["object_name"] for item in review_items]
    resolved = []
    if "missing" in tokens:
        resolved.extend([item["object_name"] for item in review_items if item.get("selected_detection") is None])
        tokens = [token for token in tokens if token != "missing"]
    for token in tokens:
        object_name = item_names.get(token)
        if object_name is not None and object_name not in resolved:
            resolved.append(object_name)
    return resolved


def _read_batch_review_answer_with_live_window(prompt: str, window_name: str, display: np.ndarray) -> str:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    while True:
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            sys.stdout.write("\n")
            sys.stdout.flush()
            return ""
        if key == ord("r"):
            sys.stdout.write("\n")
            sys.stdout.flush()
            return "r"
        if key == ord("q"):
            sys.stdout.write("\n")
            sys.stdout.flush()
            return "q"

        ready, _, _ = select.select([sys.stdin], [], [], 0.02)
        if ready:
            line = sys.stdin.readline()
            if line == "":
                sys.stdout.write("\n")
                sys.stdout.flush()
                return "q"
            return line.strip().lower()


def _looks_like_only_punctuation(text: str) -> bool:
    text = str(text or "").strip()
    if not text:
        return False
    punctuation = set(string.punctuation) | set("，。！？；：（）【】《》、‘’“”·")
    return all(ch in punctuation for ch in text)


def review_grounding_dino_batch(reader, detector, args, review_items, *, window_name: str = "GroundingDINO Batch Review", max_overlap_iou: float = 0.5):
    latest_frame = None
    while latest_frame is None:
        latest_frame = reader.get_frame()
    need_redetect = True
    while True:
        if need_redetect:
            _run_batch_grounding_dino_detector(latest_frame, detector, args, review_items, max_overlap_iou=max_overlap_iou)
            need_redetect = False
        display = draw_grounding_dino_batch_overlay(latest_frame["color_bgr"], review_items)
        cv2.imshow(window_name, display)
        cv2.waitKey(1)
        missing_required = [
            item["object_name"]
            for item in review_items
            if not item.get("allow_skip", False) and item.get("selected_detection") is None
        ]
        if missing_required:
            print(f"[groundingdino batch] required objects missing boxes: {missing_required}")
        prompt = (
            "[groundingdino batch] Press Enter to accept all current boxes, "
            "type r to recapture and re-detect all, type object names separated by space/comma to re-edit, "
            "or type q to abort: "
        )
        answer = _read_batch_review_answer_with_live_window(prompt, window_name, display)
        if answer in ("", "s"):
            if missing_required:
                print("[groundingdino batch] Cannot accept yet because some required objects still have no box.")
                continue
            accepted = [item for item in review_items if item.get("selected_detection") is not None]
            skipped = [item["object_name"] for item in review_items if item.get("selected_detection") is None and item.get("allow_skip", False)]
            if skipped:
                print(f"[groundingdino batch] accepting current boxes and skipping: {skipped}")
            return accepted
        if answer == "q":
            return None
        if answer == "r":
            refreshed = reader.get_frame()
            if refreshed is not None:
                latest_frame = refreshed
            need_redetect = True
            continue
        if _looks_like_only_punctuation(answer):
            print("[groundingdino batch] Ignoring punctuation-only input.")
            continue
        selected_names = _parse_batch_review_names(answer, review_items)
        if not selected_names:
            print("[groundingdino batch] No valid object names were provided. Try again.")
            continue
        for object_name in selected_names:
            item = next((entry for entry in review_items if entry["object_name"] == object_name), None)
            if item is None:
                continue
            used_boxes = [
                np.asarray(other["box"], dtype=np.float32).reshape(4)
                for other in review_items
                if other["object_name"] != object_name and other.get("box") is not None
            ]
            object_args = item["object_args"]
            edited_frame, edited_mask, status = select_init_frame_and_mask_with_grounding_dino(
                reader,
                detector,
                object_args,
                prompt=item["prompt"],
                allow_skip=bool(item.get("allow_skip", False)),
                window_name=f"GroundingDINO Edit: {object_name}",
                fixed_frame=latest_frame,
                used_boxes=used_boxes,
                max_overlap_iou=max_overlap_iou,
                refresh_from_reader_on_r=True,
            )
            if status == "quit":
                return None
            if status == "skip":
                item["selected_detection"] = None
                item["frame"] = edited_frame
                item["mask"] = None
                item["box"] = None
                item["status"] = "skip"
                continue
            if status != "accepted" or edited_frame is None or edited_mask is None:
                continue
            latest_frame = edited_frame
            edited_box = _mask_to_xyxy(edited_mask)
            item["selected_detection"] = {
                "box": np.asarray(edited_box, dtype=np.float32).reshape(4),
                "score": 1.0,
                "label": item["prompt"],
            }
            item["frame"] = edited_frame
            item["mask"] = edited_mask
            item["box"] = np.asarray(edited_box, dtype=np.float32).reshape(4)
            item["status"] = "accepted"



def create_foundationpose_runtime(fp_rt, args):
    cached_models = getattr(fp_rt, "_jiaobang_foundationpose_runtime_models", None)
    if isinstance(cached_models, dict) and all(
        cached_models.get(key) is not None for key in ("scorer", "refiner", "glctx")
    ):
        runtime = {
            "scorer": cached_models["scorer"],
            "refiner": cached_models["refiner"],
            "glctx": cached_models["glctx"],
        }
    else:
        runtime = {
            "scorer": fp_rt.ScorePredictor(),
            "refiner": fp_rt.PoseRefinePredictor(),
            "glctx": fp_rt.dr.RasterizeCudaContext(),
        }
        try:
            fp_rt._jiaobang_foundationpose_runtime_models = {
                "scorer": runtime["scorer"],
                "refiner": runtime["refiner"],
                "glctx": runtime["glctx"],
            }
        except Exception:
            pass

    reader = fp_rt.RealSenseRGBDReader(
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    )
    if args.camera_serial:
        reader.config.enable_device(args.camera_serial)
    reader.start()

    fp_rt.make_debug_dirs(args.debug_dir)
    for _ in range(max(args.warmup_frames, 0)):
        if reader.get_frame() is None:
            continue

    runtime["reader"] = reader
    return runtime


def _load_scene_aware_trimesh(mesh_file: str, mesh_scale: float = 1.0) -> trimesh.Trimesh:
    mesh_or_scene = trimesh.load(mesh_file, force="scene")
    if isinstance(mesh_or_scene, trimesh.Scene):
        mesh = mesh_or_scene.dump(concatenate=True)
        if mesh is None:
            raise ValueError(f"No mesh geometry found in: {mesh_file}")
    elif isinstance(mesh_or_scene, trimesh.Trimesh):
        mesh = mesh_or_scene.copy()
    else:
        raise TypeError(f"Unsupported mesh type: {type(mesh_or_scene)}")

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Empty mesh: {mesh_file}")
    if mesh_scale <= 0:
        raise ValueError(f"--mesh_scale must be positive, got {mesh_scale}")

    mesh.remove_unreferenced_vertices()
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, "nondegenerate_faces") and hasattr(mesh, "update_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())

    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    elif hasattr(mesh, "unique_faces") and hasattr(mesh, "update_faces"):
        mesh.update_faces(mesh.unique_faces())

    try:
        mesh.process(validate=True)
    except Exception as exc:
        print(f"[_load_scene_aware_trimesh] mesh.process(validate=True) failed: {exc}")
        print("[_load_scene_aware_trimesh] retrying with validate=False for trimesh compatibility")
        mesh.remove_unreferenced_vertices()
        mesh.process(validate=False)
    mesh.apply_scale(mesh_scale)
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
    _ = mesh.vertex_normals
    return mesh


def create_foundationpose_estimator(fp_rt, args, runtime=None):
    mesh = _load_scene_aware_trimesh(args.mesh_file, args.mesh_scale)
    scorer = runtime["scorer"] if runtime is not None else fp_rt.ScorePredictor()
    refiner = runtime["refiner"] if runtime is not None else fp_rt.PoseRefinePredictor()
    glctx = runtime["glctx"] if runtime is not None else fp_rt.dr.RasterizeCudaContext()
    est = fp_rt.FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=args.debug_dir,
        debug=args.debug,
        glctx=glctx,
    )
    return est


def _mask_to_xyxy(mask) -> np.ndarray | None:
    mask = np.asarray(mask)
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _bbox_iou(box_a, box_b) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in np.asarray(box_a).reshape(-1)[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in np.asarray(box_b).reshape(-1)[:4]]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return float(inter_area / denom)


def _select_best_unused_detection(detections, used_boxes, max_iou: float = 0.5):
    if not detections:
        return None
    ordered = sorted(detections, key=lambda det: float(det.get("score", 0.0)), reverse=True)
    for det in ordered:
        box = det.get("box")
        if all(_bbox_iou(box, used_box) <= max_iou for used_box in used_boxes if used_box is not None):
            return det
    return None


def _make_box_mask(height: int, width: int, box) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in np.asarray(box, dtype=np.float32).reshape(-1)[:4]]
    x1 = max(0, min(x1, width - 1))
    x2 = max(x1 + 1, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(y1 + 1, min(y2, height))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def _make_object_specific_args(args, object_name: str):
    spec = get_object_spec(object_name)
    if spec is None:
        raise ValueError(f"Unknown object spec: {object_name}")
    spec_mesh_scale, spec_sim_asset_scale = resolve_object_spec_scales(spec)

    object_args = argparse.Namespace(**vars(args).copy())
    object_args.object_name = object_name
    object_args.mesh_file = spec.mesh_file
    object_args.mesh_scale = spec_mesh_scale
    object_args.sim_asset_file = spec.sim_asset_file or spec.mesh_file
    object_args.sim_asset_scale = spec_sim_asset_scale
    object_args.target_object_name = spec.grounding_prompt
    object_args.init_mask = None
    object_args.foundationpose_position_offset = list(spec.foundationpose_position_offset)
    object_args.foundationpose_local_rotation_offset_deg = list(spec.foundationpose_local_rotation_offset_deg)
    if spec.pregrasp_height is not None:
        object_args.pregrasp_height = spec.pregrasp_height
    if spec.grasp_z_offset is not None:
        object_args.grasp_z_offset = spec.grasp_z_offset
    if spec.goal_z_offset is not None:
        object_args.goal_z_offset = spec.goal_z_offset
    object_args.grasp_mode = getattr(spec, "grasp_mode", "object_normal") or "object_normal"
    if object_args.mesh_file is not None:
        object_args.mesh_file = str(Path(object_args.mesh_file).expanduser())
    if object_args.sim_asset_file is not None:
        object_args.sim_asset_file = str(Path(object_args.sim_asset_file).expanduser())
    return object_args


def _average_pose_matrices(poses: list[np.ndarray]) -> np.ndarray:
    if not poses:
        raise ValueError("At least one pose is required for averaging")
    translations = np.stack([np.asarray(pose, dtype=np.float64)[:3, 3] for pose in poses], axis=0)
    quats = []
    ref_quat = mat2quat(np.asarray(poses[-1], dtype=np.float64)[:3, :3])
    for pose in poses:
        quat = mat2quat(np.asarray(pose, dtype=np.float64)[:3, :3])
        if float(np.dot(quat, ref_quat)) < 0.0:
            quat = -quat
        quats.append(quat)
    quat_mean = np.mean(np.stack(quats, axis=0), axis=0)
    quat_norm = float(np.linalg.norm(quat_mean))
    if quat_norm <= 1e-8:
        quat_mean = ref_quat
    else:
        quat_mean = quat_mean / quat_norm
    averaged = np.eye(4, dtype=np.float32)
    averaged[:3, :3] = quat2mat(quat_mean).astype(np.float32)
    averaged[:3, 3] = np.mean(translations, axis=0).astype(np.float32)
    return averaged


def _stabilize_foundationpose_pose(est, reader, pose, args, label: str) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32).reshape(4, 4)
    stabilize_frames = max(int(getattr(args, "foundationpose_stabilize_frames", 0)), 0)
    smoothing_window = max(int(getattr(args, "foundationpose_smoothing_window", 1)), 1)
    if stabilize_frames <= 0 or not hasattr(est, "track_one"):
        return pose

    tracked_poses: list[np.ndarray] = [pose]
    tracked_frames = 0
    attempts = 0
    track_errors = 0
    max_attempts = max(stabilize_frames * 5, stabilize_frames)
    while tracked_frames < stabilize_frames and attempts < max_attempts:
        attempts += 1
        frame = reader.get_frame()
        if frame is None:
            continue
        try:
            tracked_pose = est.track_one(
                rgb=frame["color"],
                depth=frame["depth"],
                K=frame["K"],
                iteration=args.track_refine_iter,
            )
        except Exception as exc:
            track_errors += 1
            print(
                f"[foundationpose] {label} stabilization track_one failed on attempt "
                f"{attempts}/{max_attempts}: {type(exc).__name__}: {exc}"
            )
            continue
        if tracked_pose is None:
            continue
        tracked_poses.append(np.asarray(tracked_pose, dtype=np.float32).reshape(4, 4))
        tracked_frames += 1

    effective_window = min(smoothing_window, len(tracked_poses))
    if effective_window <= 1:
        stabilized_pose = tracked_poses[-1]
    else:
        stabilized_pose = _average_pose_matrices(tracked_poses[-effective_window:])

    print(
        f"[foundationpose] {label} stabilized with {tracked_frames} tracked frame(s); "
        f"smoothing_window={effective_window}, track_errors={track_errors}"
    )
    return stabilized_pose


def _register_pose_from_frame(est, frame, mask, args, reader=None, label: str = "target") -> np.ndarray:
    pose = est.register(
        K=frame["K"],
        rgb=frame["color"],
        depth=frame["depth"],
        ob_mask=np.asarray(mask, dtype=bool),
        iteration=args.est_refine_iter,
    )
    if pose is None:
        raise RuntimeError("FoundationPose failed to initialize an object pose from the selected ROI")
    pose = np.asarray(pose, dtype=np.float32).reshape(4, 4)
    if reader is not None:
        pose = _stabilize_foundationpose_pose(est, reader, pose, args, label=label)
    return pose


def _build_foundationpose_tracking_visuals(fp_rt, args):
    mesh = _load_scene_aware_trimesh(args.mesh_file, args.mesh_scale)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    return np.asarray(to_origin, dtype=np.float32), np.asarray(bbox, dtype=np.float32)


def interactive_refine_single_object_pose(fp_rt, args, *, object_name: str | None = None, window_name: str = "FoundationPose Refine"):
    label = str(object_name or getattr(args, "object_name", None) or "object")
    runtime = create_foundationpose_runtime(fp_rt, args)
    reader = runtime["reader"]
    detector = create_grounding_dino_detector(args) if getattr(args, "target_object_name", None) else None
    est = create_foundationpose_estimator(fp_rt, args, runtime=runtime)
    to_origin, bbox = _build_foundationpose_tracking_visuals(fp_rt, args)
    lock_window = max(int(getattr(args, "foundationpose_refine_lock_window", 4)), 1)

    try:
        pose = None
        tracked_poses: list[np.ndarray] = []
        while True:
            if pose is None:
                if detector is not None:
                    init_frame, init_mask, status = select_init_frame_and_mask_with_grounding_dino(
                        reader,
                        detector,
                        args,
                        prompt=str(getattr(args, "target_object_name", "") or ""),
                        allow_skip=True,
                        window_name=f"{window_name}: {label}",
                        refresh_from_reader_on_r=True,
                    )
                    if status in ("quit", "skip") or init_frame is None or init_mask is None:
                        return None
                else:
                    init_frame = fp_rt.wait_for_registration_frame(reader)
                    if init_frame is None:
                        return None
                    init_mask = fp_rt.select_init_mask(init_frame["color_bgr"])
                    if init_mask is None:
                        continue
                pose = _register_pose_from_frame(est, init_frame, init_mask, args, reader=None, label=f"refine {label}")
                tracked_poses = [pose]

            frame = reader.get_frame()
            if frame is None:
                continue
            tracked_pose = est.track_one(
                rgb=frame["color"],
                depth=frame["depth"],
                K=frame["K"],
                iteration=args.track_refine_iter,
            )
            if tracked_pose is None:
                continue

            pose = np.asarray(tracked_pose, dtype=np.float32).reshape(4, 4)
            tracked_poses.append(pose)
            tracked_poses = tracked_poses[-max(lock_window, 1):]

            center_pose = pose @ np.linalg.inv(to_origin)
            vis = fp_rt.draw_posed_3d_box(frame["K"], img=frame["color"], ob_in_cam=center_pose, bbox=bbox)
            vis = fp_rt.draw_xyz_axis(
                vis,
                ob_in_cam=center_pose,
                scale=0.1,
                K=frame["K"],
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            vis_bgr = vis[..., ::-1].copy()
            lines = [
                f"FoundationPose Refine: {label}",
                "Enter: lock current tracked pose",
                "r: re-register this object",
                "q: cancel refine",
            ]
            y = 28
            for line in lines:
                cv2.putText(vis_bgr, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y += 28
            cv2.imshow(window_name, vis_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (13, 10):
                locked_pose = tracked_poses[-1] if len(tracked_poses) <= 1 else _average_pose_matrices(tracked_poses[-lock_window:])
                print(
                    f"[foundationpose refine] locked {label} pose with smoothing_window="
                    f"{min(lock_window, len(tracked_poses))}: translation="
                    f"{np.round(locked_pose[:3, 3], 6).tolist()}"
                )
                return np.asarray(locked_pose, dtype=np.float32)
            if key == ord("r"):
                pose = None
                continue
            if key == ord("q"):
                return None
    finally:
        reader.stop()
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass


def capture_scene_poses_from_foundationpose(fp_rt, args):
    runtime = create_foundationpose_runtime(fp_rt, args)
    reader = runtime["reader"]
    selected_obstacle_names = [normalize_object_name(name) for name in getattr(args, "selected_obstacle_object_names", []) or []]
    selected_obstacle_names = [name for name in selected_obstacle_names if name]
    required_obstacle_names = {
        normalize_object_name(name)
        for name in (getattr(args, "required_scene_object_names", []) or [])
        if normalize_object_name(name) is not None
    }
    detector = None
    need_detector = bool((args.target_object_name and not args.init_mask) or selected_obstacle_names)
    if need_detector:
        detector = create_grounding_dino_detector(args)

    try:
        est = create_foundationpose_estimator(fp_rt, args, runtime=runtime)
        init_frame = None
        init_mask = None
        target_box = None
        accepted_review_items = []

        if detector is not None and ((args.target_object_name and not args.init_mask) or selected_obstacle_names):
            review_items = []
            target_review_name = normalize_object_name(getattr(args, "object_name", None)) or "target"
            if args.target_object_name and not args.init_mask:
                review_items.append({
                    "role": "target",
                    "object_name": target_review_name,
                    "prompt": str(args.target_object_name).strip(),
                    "object_args": args,
                    "allow_skip": False,
                })
            for object_name in selected_obstacle_names:
                object_args = _make_object_specific_args(args, object_name)
                prompt = str(getattr(object_args, "target_object_name", "") or "").strip()
                if not prompt:
                    print(f"[scene obstacle] {object_name}: skipped because no grounding prompt is configured")
                    continue
                review_items.append({
                    "role": "obstacle",
                    "object_name": object_name,
                    "prompt": prompt,
                    "object_args": object_args,
                    "allow_skip": object_name not in required_obstacle_names,
                })
            if review_items:
                accepted_review_items = review_grounding_dino_batch(
                    reader,
                    detector,
                    args,
                    review_items,
                    window_name="GroundingDINO Batch Review",
                    max_overlap_iou=0.5,
                )
                if accepted_review_items is None:
                    raise SystemExit(0)

        if args.target_object_name and not args.init_mask:
            target_item = next((item for item in accepted_review_items if item.get("role") == "target"), None)
            if target_item is None or target_item.get("frame") is None or target_item.get("mask") is None:
                raise SystemExit(0)
            init_frame = target_item["frame"]
            init_mask = target_item["mask"]
            pose = _register_pose_from_frame(est, init_frame, init_mask, args, reader=reader, label="target")
            target_box = np.asarray(target_item["box"], dtype=np.float32).reshape(4) if target_item.get("box") is not None else None
        else:
            init_frame = fp_rt.wait_for_registration_frame(reader)
            if init_frame is None:
                raise SystemExit(0)
            pose = fp_rt.initialize_pose(est, init_frame, args)
            if pose is None:
                raise RuntimeError("FoundationPose failed to initialize an object pose from the selected ROI")
            pose = np.asarray(pose, dtype=np.float32).reshape(4, 4)
            pose = _stabilize_foundationpose_pose(est, reader, pose, args, label="target")

        print("FoundationPose initialized T_cam_obj translation:", np.round(pose[:3, 3], 6).tolist())

        obstacle_results = []
        if selected_obstacle_names and detector is not None:
            accepted_obstacle_items = [item for item in accepted_review_items if item.get("role") == "obstacle"]
            for item in accepted_obstacle_items:
                object_name = str(item["object_name"])
                object_args = item["object_args"]
                prompt = str(item["prompt"])
                obstacle_frame = item.get("frame")
                obstacle_mask = item.get("mask")
                if obstacle_frame is None or obstacle_mask is None:
                    print(f"[scene obstacle] {object_name}: skipped by user")
                    continue
                try:
                    obstacle_est = create_foundationpose_estimator(fp_rt, object_args, runtime=runtime)
                    obstacle_pose = _register_pose_from_frame(
                        obstacle_est,
                        obstacle_frame,
                        obstacle_mask,
                        object_args,
                        reader=reader,
                        label=f"scene obstacle {object_name}",
                    )
                except Exception as exc:
                    print(f"[scene obstacle] {object_name}: FoundationPose init failed: {exc}")
                    continue

                obstacle_box = _mask_to_xyxy(obstacle_mask)
                obstacle_results.append({
                    "object_name": object_name,
                    "label": prompt,
                    "score": float(item.get("selected_detection", {}).get("score", 1.0)) if item.get("selected_detection") is not None else 1.0,
                    "box": obstacle_box if obstacle_box is not None else np.zeros(4, dtype=np.float32),
                    "T_cam_obj": obstacle_pose,
                    "object_args": object_args,
                })
                print(
                    f"[scene obstacle] {object_name}: accepted, "
                    f"T_cam_obj translation={np.round(obstacle_pose[:3, 3], 6).tolist()}"
                )

        return pose, obstacle_results
    finally:
        reader.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def capture_initial_pose_from_foundationpose(fp_rt, args) -> np.ndarray:
    pose, _ = capture_scene_poses_from_foundationpose(fp_rt, args)
    return pose



def get_robot_base_transform(env) -> np.ndarray | None:
    agent = getattr(env.unwrapped, "agent", None)
    robot = getattr(agent, "robot", None)
    pose = getattr(robot, "pose", None)
    if pose is None:
        return None
    mat = pose.to_transformation_matrix()
    if hasattr(mat, "detach"):
        mat = mat.detach().cpu().numpy()
    mat = np.asarray(mat)
    if mat.ndim == 3:
        mat = mat[0]
    return mat.astype(np.float32)



def get_default_object_center_z(env) -> float | None:
    object_zs = getattr(env.unwrapped, "object_zs", None)
    if object_zs is None:
        return None
    if hasattr(object_zs, "detach"):
        object_zs = object_zs.detach().cpu().numpy()
    object_zs = np.asarray(object_zs)
    if object_zs.size == 0:
        return None
    return float(object_zs.reshape(-1)[0])



def get_env_object_pose_matrix(env) -> np.ndarray | None:
    obj = getattr(env.unwrapped, "obj", None)
    pose = getattr(obj, "pose", None)
    return _pose_to_matrix(pose)


def _pose_to_matrix(pose) -> np.ndarray | None:
    if pose is None:
        return None
    if hasattr(pose, "to_transformation_matrix"):
        mat = pose.to_transformation_matrix()
    else:
        mat = pose
    if hasattr(mat, "detach"):
        mat = mat.detach().cpu().numpy()
    mat = np.asarray(mat)
    if mat.ndim == 3:
        mat = mat[0]
    if mat.shape != (4, 4):
        return None
    return mat.astype(np.float32)


def _get_local_asset_points(args) -> np.ndarray | None:
    asset_file = getattr(args, "sim_asset_file", None) or getattr(args, "mesh_file", None)
    if not asset_file:
        return None
    asset_path = Path(os.path.expanduser(str(asset_file))).expanduser()
    scale = float(getattr(args, "sim_asset_scale", None) or getattr(args, "mesh_scale", 1.0) or 1.0)
    stat = asset_path.stat()
    cache_key = (str(asset_path.resolve()), scale, int(stat.st_mtime_ns), int(stat.st_size))
    if getattr(args, "_cached_asset_points_key", None) == cache_key:
        return getattr(args, "_cached_asset_points", None)

    try:
        loaded = trimesh.load(asset_path, force="scene")
        if isinstance(loaded, trimesh.Trimesh):
            mesh = loaded.copy()
        else:
            mesh = loaded.dump(concatenate=True)
            if mesh is None:
                return None
        if abs(scale - 1.0) > 1e-8:
            mesh.apply_scale(scale)
        points = np.asarray(mesh.vertices, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            return None
    except Exception as exc:
        print(f"[warn] failed to load asset bounds for tabletop safety clamp: {exc}")
        return None

    args._cached_asset_points_key = cache_key
    args._cached_asset_points = points
    return points


def _compute_world_min_z(T_world_obj: np.ndarray, local_points: np.ndarray | None) -> float | None:
    if local_points is None:
        return None
    T_world_obj = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4)
    local_points = np.asarray(local_points, dtype=np.float32).reshape(-1, 3)
    world_pts = (T_world_obj[:3, :3] @ local_points.T).T + T_world_obj[:3, 3]
    return float(np.min(world_pts[:, 2]))


def _rotation_distance_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_rel = np.asarray(R_a, dtype=np.float64) @ np.asarray(R_b, dtype=np.float64).T
    trace = float(np.trace(R_rel))
    cos_theta = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    return float(np.rad2deg(np.arccos(cos_theta)))


def _format_pose_summary(T: np.ndarray) -> str:
    T = np.asarray(T, dtype=np.float32).reshape(4, 4)
    p = np.round(T[:3, 3], 6).tolist()
    q = np.round(mat2quat(T[:3, :3]).astype(np.float32), 6).tolist()
    return f"p={p}, q={q}"


def get_default_object_min_world_z(env, args) -> float | None:
    default_pose = get_env_object_pose_matrix(env)
    if default_pose is None:
        return None
    local_points = _get_local_asset_points(args)
    return _compute_world_min_z(default_pose, local_points)


def _adjust_foundationpose_pose_impl(T_base_obj: np.ndarray, env, args, *, emit_logs: bool) -> np.ndarray:
    adjusted = np.array(T_base_obj, dtype=np.float32, copy=True)

    if not args.no_map_foundationpose_through_robot_base:
        robot_base_T = get_robot_base_transform(env)
        if robot_base_T is not None:
            adjusted = robot_base_T @ adjusted

    local_rpy_deg = np.asarray(args.foundationpose_local_rotation_offset_deg, dtype=np.float32)
    if np.any(np.abs(local_rpy_deg) > 1e-6):
        local_fix = np.eye(4, dtype=np.float32)
        local_fix[:3, :3] = euler2mat(*np.deg2rad(local_rpy_deg)).astype(np.float32)
        adjusted = adjusted @ local_fix

    adjusted[:3, 3] += np.asarray(args.foundationpose_position_offset, dtype=np.float32)

    default_z = get_default_object_center_z(env)
    if args.lock_object_z_to_table:
        if default_z is not None:
            adjusted[2, 3] = default_z
    else:
        min_margin = float(getattr(args, "min_object_center_z_margin", 0.0))
        if min_margin > 0.0:
            default_min_z = get_default_object_min_world_z(env, args)
            adjusted_min_z = _compute_world_min_z(adjusted, _get_local_asset_points(args))
            if default_min_z is not None and adjusted_min_z is not None:
                min_allowed_z = float(default_min_z + min_margin)
                if adjusted_min_z < min_allowed_z:
                    delta_z = float(min_allowed_z - adjusted_min_z)
                    if emit_logs:
                        print(
                            f"[safety] raised mapped object by {delta_z:.4f} m so its lowest point stays above the table "
                            f"(min z {adjusted_min_z:.4f} -> {min_allowed_z:.4f})"
                        )
                    adjusted[2, 3] += delta_z
            elif default_z is not None:
                min_allowed_z = float(default_z + min_margin)
                if float(adjusted[2, 3]) < min_allowed_z:
                    if emit_logs:
                        print(
                            f"[safety] raised mapped object center z from {float(adjusted[2, 3]):.4f} to {min_allowed_z:.4f} "
                            f"to stay above the table"
                        )
                    adjusted[2, 3] = min_allowed_z

    return adjusted


def adjust_foundationpose_pose(T_base_obj: np.ndarray, env, args) -> np.ndarray:
    return _adjust_foundationpose_pose_impl(T_base_obj, env, args, emit_logs=True)


def print_foundationpose_mapping_diagnostics(
    T_cam_obj: np.ndarray,
    T_base_cam: np.ndarray,
    env,
    args,
    *,
    label: str = "target",
) -> None:
    if not bool(getattr(args, "foundationpose_print_mapping_diagnostics", False)):
        return
    T_cam_obj = np.asarray(T_cam_obj, dtype=np.float32).reshape(4, 4)
    T_base_cam = np.asarray(T_base_cam, dtype=np.float32).reshape(4, 4)

    T_base_obj_direct = T_base_cam @ T_cam_obj
    T_base_obj_inverse = np.linalg.inv(T_base_cam) @ T_cam_obj
    T_world_obj_direct = _adjust_foundationpose_pose_impl(T_base_obj_direct, env, args, emit_logs=False)
    T_world_obj_inverse = _adjust_foundationpose_pose_impl(T_base_obj_inverse, env, args, emit_logs=False)

    current_name = "direct" if bool(getattr(args, "use_direct_camera_extrinsic", False)) else "inverse"
    alt_name = "inverse" if current_name == "direct" else "direct"
    T_world_obj_current = T_world_obj_direct if current_name == "direct" else T_world_obj_inverse
    T_world_obj_alt = T_world_obj_inverse if current_name == "direct" else T_world_obj_direct
    translation_gap = float(np.linalg.norm(T_world_obj_current[:3, 3] - T_world_obj_alt[:3, 3]))
    rotation_gap_deg = _rotation_distance_deg(T_world_obj_current[:3, :3], T_world_obj_alt[:3, :3])

    print(f"[foundationpose diag] {label} raw T_cam_obj {_format_pose_summary(T_cam_obj)}")
    print(f"[foundationpose diag] {label} direct mapped {_format_pose_summary(T_world_obj_direct)}")
    print(f"[foundationpose diag] {label} inverse mapped {_format_pose_summary(T_world_obj_inverse)}")
    print(
        f"[foundationpose diag] {label} current convention={current_name}, "
        f"current-vs-{alt_name} translation_gap={translation_gap:.4f} m, rotation_gap={rotation_gap_deg:.2f} deg"
    )


def map_camera_pose_to_pick_world(T_cam_obj: np.ndarray, T_base_cam: np.ndarray, env, args) -> np.ndarray:
    T_cam_obj = np.asarray(T_cam_obj, dtype=np.float32)
    T_base_cam = np.asarray(T_base_cam, dtype=np.float32)

    if args.use_direct_camera_extrinsic:
        T_base_obj = T_base_cam @ T_cam_obj
    else:
        T_base_obj = np.linalg.inv(T_base_cam) @ T_cam_obj

    return adjust_foundationpose_pose(T_base_obj, env, args)



def apply_pose_to_pick_object(env, T_world_obj: np.ndarray):
    base_env = env.unwrapped
    pos = T_world_obj[:3, 3].astype(np.float32)
    quat = mat2quat(T_world_obj[:3, :3]).astype(np.float32)
    base_env.obj.set_pose(Pose.create_from_pq(p=pos, q=quat))
    # Teleporting a dynamic object should also clear residual motion inherited
    # from the env reset; otherwise the object can immediately drift or roll.
    try:
        zero_vel = np.zeros(3, dtype=np.float32)
        base_env.obj.set_linear_velocity(zero_vel)
        base_env.obj.set_angular_velocity(zero_vel)
    except Exception:
        pass

    if hasattr(base_env, "object_initial_height"):
        base_env.object_initial_height[:] = -1.0

    if hasattr(base_env, "get_obj_xy_shortest_edge_vector") and hasattr(base_env, "obj_xy_shortest_edge_vector"):
        with torch.no_grad():
            shortest_vec = base_env.get_obj_xy_shortest_edge_vector().detach()
            base_env.obj_xy_shortest_edge_vector[:] = shortest_vec


def apply_pick_object_physics_profile(env, args):
    import sapien

    spec = get_object_spec(getattr(args, "object_name", None))
    if spec is None:
        asset_name = Path(str(getattr(args, "sim_asset_file", "") or "")).name.lower()
        if asset_name == "pen.glb":
            spec = get_object_spec("bi")
    if spec is None:
        return

    static_friction = getattr(spec, "sim_static_friction", None)
    dynamic_friction = getattr(spec, "sim_dynamic_friction", None)
    restitution = getattr(spec, "sim_restitution", None)
    linear_damping = getattr(spec, "sim_linear_damping", None)
    angular_damping = getattr(spec, "sim_angular_damping", None)
    if all(value is None for value in (static_friction, dynamic_friction, restitution, linear_damping, angular_damping)):
        return

    base_env = env.unwrapped
    obj = getattr(base_env, "obj", None)
    if obj is None:
        return

    if linear_damping is not None:
        try:
            obj.set_linear_damping(float(linear_damping))
        except Exception:
            pass
    if angular_damping is not None:
        try:
            obj.set_angular_damping(float(angular_damping))
        except Exception:
            pass

    source_actors = list(getattr(base_env, "_objs", []) or [])
    if not source_actors:
        source_actors = list(getattr(obj, "_objs", []) or [])
    for actor in source_actors:
        try:
            component = actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        except Exception:
            component = None
        if component is None:
            continue
        for shape in getattr(component, "collision_shapes", []) or []:
            material = getattr(shape, "physical_material", None)
            if material is None:
                continue
            if static_friction is not None:
                material.static_friction = float(static_friction)
            if dynamic_friction is not None:
                material.dynamic_friction = float(dynamic_friction)
            if restitution is not None:
                material.restitution = float(restitution)

    print(
        "[sim physics] applied object profile:",
        spec.name,
        f"static_friction={static_friction}",
        f"dynamic_friction={dynamic_friction}",
        f"restitution={restitution}",
        f"linear_damping={linear_damping}",
        f"angular_damping={angular_damping}",
    )



def render_preview(env, repeats: int = 3, sleep_s: float = 0.03):
    for _ in range(max(repeats, 0)):
        try:
            env.render()
        except Exception:
            break
        if sleep_s > 0:
            time.sleep(float(sleep_s))



def prompt_with_live_render(prompt: str, env, args, poll_s: float = 0.03) -> str:
    if args.auto_execute:
        return ""
    if getattr(args, "render_mode", None) != "human" or env is None or not sys.stdin.isatty():
        try:
            return input(prompt).strip().lower()
        except EOFError:
            return "q"

    print(prompt, end="", flush=True)
    while True:
        try:
            env.render()
        except Exception:
            pass
        try:
            ready, _, _ = select.select([sys.stdin], [], [], poll_s)
        except Exception:
            try:
                return input().strip().lower()
            except EOFError:
                return "q"
        if ready:
            line = sys.stdin.readline()
            if line == "":
                return "q"
            return line.strip().lower()


def confirm_planned_motion(demo, label: str, target_pose, q_target, args) -> bool:
    demo.preview_target_pose(target_pose)
    render_preview(demo.env, repeats=3)

    print(f"\n[planned {label}]")
    print("target p:", np.round(flatten_np(target_pose.p)[:3], 6))
    print("target q:", np.round(flatten_np(target_pose.q)[:4], 6))
    print("planned arm q:", np.round(flatten_np(q_target)[:7], 6))

    if args.auto_execute:
        return True

    if args.render_mode != "human" and not getattr(args, "_warned_nonhuman_render", False):
        print("[confirm] render_mode is not human, so you may not see a live viewer. Use --render-mode human if you want to inspect visually before pressing Enter.")
        args._warned_nonhuman_render = True

    answer = prompt_with_live_render(
        f"[confirm] {label} planned. Press Enter to execute, or type q then Enter to abort: ",
        demo.env,
        args,
    )
    return answer not in {"q", "quit", "n", "no"}


def chase_goals_until_done_with_confirmation(demo, args) -> bool:
    goal_move_idx = 0
    while True:
        print(f"\n[move to goal #{goal_move_idx}]")
        goal_tcp_pose, q_goal = demo.refresh_goal_and_plan(announce=True)
        if q_goal is None:
            print("[FAIL] goal planning failed")
            return False

        if not confirm_planned_motion(demo, f"goal #{goal_move_idx}", goal_tcp_pose, q_goal, args):
            print("[abort] user cancelled before executing goal motion")
            return False

        demo.execute_linear(
            q_goal,
            gripper_value=args.gripper_close,
            max_delta_per_step=args.max_delta_per_step,
            hold_steps=args.hold_steps,
            tag=f"goal_{goal_move_idx}",
        )
        if args.log_contact_forces:
            demo.log_grasp_contact_forces(tag=f"after_goal_{goal_move_idx}")
        demo.report_final_error(goal_tcp_pose)
        flags = demo.print_step_flags(prefix=f"[goal #{goal_move_idx}] ")

        if flags["done"]:
            print("[goal loop] episode done, waiting for outer reset")
            return True

        if flags["success"]:
            print("[goal loop] success detected, environment should resample a new goal; refresh visualization and continue")
            refreshed_goal_pose = demo.build_goal_tcp_pose()
            demo.preview_target_pose(refreshed_goal_pose)
            render_preview(demo.env, repeats=3)
            print(
                "[goal loop] refreshed goal tcp p:",
                np.round(flatten_np(refreshed_goal_pose.p)[:3], 6),
                "q:",
                np.round(flatten_np(refreshed_goal_pose.q)[:4], 6),
            )
            goal_move_idx += 1
            continue

        print("[goal loop] current motion finished without success/done, hold current pose and keep monitoring")
        hold_q = demo.current_arm_qpos()
        monitor_steps = max(1, args.goal_monitor_steps)
        for monitor_idx in range(monitor_steps):
            action = demo.compose_action(hold_q, args.gripper_close)
            demo.step_and_render(action, tag="goal_monitor")
            flags = demo._extract_step_flags()
            if flags["done"]:
                demo.print_step_flags(prefix=f"[goal monitor {monitor_idx}] ")
                print("[goal loop] episode done during monitor")
                return True
            if flags["success"]:
                demo.print_step_flags(prefix=f"[goal monitor {monitor_idx}] ")
                refreshed_goal_pose = demo.build_goal_tcp_pose()
                demo.preview_target_pose(refreshed_goal_pose)
                render_preview(demo.env, repeats=3)
                print(
                    "[goal loop] refreshed goal tcp p:",
                    np.round(flatten_np(refreshed_goal_pose.p)[:3], 6),
                    "q:",
                    np.round(flatten_np(refreshed_goal_pose.q)[:4], 6),
                )
                goal_move_idx += 1
                break
        else:
            print("[goal loop] monitor window ended without success/done, replan to current goal and continue")
            goal_move_idx += 1
            continue



def run_single_episode_from_current_state(demo, args) -> bool:
    print("\n[episode] planning from FoundationPose-initialized object pose")

    print("\n[open gripper]")
    demo.hold_current_and_set_gripper(args.gripper_open, steps=args.open_steps)

    if args.settle_steps > 0:
        print("\n[settle object pose]")
        demo.hold_current_and_set_gripper(args.gripper_open, steps=args.settle_steps)

    grasp_pose = demo.build_topdown_grasp_pose()
    pregrasp_pose = demo.build_pregrasp_pose(grasp_pose)

    print("\n[poses]")
    print("grasp p:", np.round(demo.get_obj_pose()[0], 6), "object q:", np.round(demo.get_obj_pose()[1], 6))
    print("tcp grasp p:", np.round(flatten_np(grasp_pose.p)[:3], 6), "q:", np.round(flatten_np(grasp_pose.q)[:4], 6))
    print("tcp pregrasp p:", np.round(flatten_np(pregrasp_pose.p)[:3], 6), "q:", np.round(flatten_np(pregrasp_pose.q)[:4], 6))

    print("\n[move to pregrasp]")
    demo.preview_target_pose(pregrasp_pose)
    q_pre = demo.plan_terminal_q(pregrasp_pose, variant_name=args.variant)
    if q_pre is None:
        print("[FAIL] pregrasp planning failed")
        return False
    if not confirm_planned_motion(demo, "pregrasp", pregrasp_pose, q_pre, args):
        print("[abort] user cancelled before executing pregrasp")
        return False
    demo.execute_linear(
        q_pre,
        gripper_value=args.gripper_open,
        max_delta_per_step=args.max_delta_per_step,
        hold_steps=0,
        tag="pregrasp",
    )

    print("\n[move to grasp]")
    demo.preview_target_pose(grasp_pose)
    q_grasp = demo.plan_terminal_q(grasp_pose, variant_name=args.variant)
    if q_grasp is None:
        print("[FAIL] grasp planning failed")
        return False
    if not confirm_planned_motion(demo, "grasp", grasp_pose, q_grasp, args):
        print("[abort] user cancelled before executing grasp")
        return False
    demo.execute_linear(
        q_grasp,
        gripper_value=args.gripper_open,
        max_delta_per_step=args.max_delta_per_step,
        hold_steps=args.hold_steps,
        tag="grasp",
    )

    print("\n[close gripper]")
    demo.hold_current_and_set_gripper(args.gripper_close, steps=args.close_steps)
    print("[close gripper] is_grasped =", demo.is_grasped())
    demo.print_step_flags(prefix="[after close] ")
    if args.log_contact_forces:
        demo.log_grasp_contact_forces(tag="after_close")

    return chase_goals_until_done_with_confirmation(demo, args)



def main():
    args = parse_args()
    spec = apply_object_spec_defaults(args)
    args.env_id = ensure_pick_jiaobang_env_registered(args.env_id, args.extra_maniskill_package_root)

    foundationpose_root = resolve_foundationpose_root(args.foundationpose_root)
    fp_rt = load_foundationpose_module(foundationpose_root)
    planner_mod = load_module_from_path("rm75_jiaobang_pick_move_impl", args.pick_script_path)
    T_base_cam = load_matrix(args.camera_extrinsic_opencv_path)

    fp_rt.set_logging_format()
    fp_rt.set_seed(args.seed)
    os.makedirs(args.video_dir, exist_ok=True)

    print(f"Using FoundationPose root: {foundationpose_root}")
    print(f"Using picker script: {Path(args.pick_script_path).resolve()}")
    if spec is not None:
        print(f"Using object spec: {spec.name}")
    print(f"Using mesh file: {args.mesh_file}")
    print(f"Using simulation asset file: {args.sim_asset_file}")
    print(f"Using simulation asset scale: {args.sim_asset_scale}")
    print(f"Using camera extrinsic from: {args.camera_extrinsic_opencv_path}")
    if args.use_direct_camera_extrinsic:
        print("Using camera extrinsic convention: T_base_obj = T_base_cam @ T_cam_obj")
    else:
        print("Using camera extrinsic convention: T_base_obj = inv(camera_extrinsic_opencv) @ T_cam_obj")

    T_cam_obj = capture_initial_pose_from_foundationpose(fp_rt, args)

    env = gym.make(
        args.env_id,
        robot_uids="RM75",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        max_episode_steps=args.max_episode_steps,
        object_asset_path=args.sim_asset_file,
        object_scale=args.sim_asset_scale,
    )
    env = RecordEpisode(
        env,
        output_dir=args.video_dir,
        save_trajectory=False,
        save_video=True,
        source_type="motionplanning",
        source_desc="RM75 jiaobang pick move initialized by FoundationPose",
        video_fps=args.video_fps,
    )

    try:
        initial_obs, initial_info = env.reset(seed=args.seed)
        apply_pick_object_physics_profile(env, args)
        print_foundationpose_mapping_diagnostics(T_cam_obj, T_base_cam, env, args, label="target")
        T_world_obj = map_camera_pose_to_pick_world(T_cam_obj, T_base_cam, env, args)
        print("Mapped sim object translation:", np.round(T_world_obj[:3, 3], 6).tolist())
        apply_pose_to_pick_object(env, T_world_obj)

        sim_urdf_path = planner_mod.find_existing_urdf(args.urdf_path)
        planning_urdf_path, srdf_path = planner_mod.resolve_planning_artifact_paths(sim_urdf_path, args)
        os.makedirs(os.path.dirname(planning_urdf_path), exist_ok=True)
        os.makedirs(os.path.dirname(srdf_path), exist_ok=True)
        planner_mod.generate_near_collision_free_planning_urdf(sim_urdf_path, planning_urdf_path)
        if args.srdf_path is None:
            planner_mod.write_permissive_srdf(planning_urdf_path, srdf_path)

        print("Sim URDF     :", sim_urdf_path)
        print("Planning URDF:", planning_urdf_path)
        print("SRDF         :", srdf_path)
        print("Video Dir    :", os.path.abspath(args.video_dir))

        demo = planner_mod.RM75JiaobangPickMove(env, planning_urdf_path, srdf_path, args)
        demo.last_obs = initial_obs
        demo.last_info = initial_info if isinstance(initial_info, dict) else {}
        demo.last_terminated = False
        demo.last_truncated = False
        demo.refresh_runtime_handles(rebuild_visual=False)

        ok = run_single_episode_from_current_state(demo, args)
        print("\nfinal success =", ok)
    finally:
        try:
            env.flush_video()
        except Exception:
            pass
        try:
            env.flush()
        except Exception:
            pass
        try:
            env.close()
        except Exception as exc:
            print("close warning:", exc)


if __name__ == "__main__":
    main()
