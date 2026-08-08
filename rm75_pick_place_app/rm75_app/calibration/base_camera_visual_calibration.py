from __future__ import annotations

import argparse
import select
import sys
from pathlib import Path

import cv2
import numpy as np

from rm75_app.paths import ASSET_DIR, DEFAULT_RM75_URDF

from .board import board_config_from_args, detect_board_pose, detection_is_accepted, draw_board_overlay
from .common import (
    RealSenseCamera,
    RealmanSdkController,
    as_transform,
    calibration_run_dir,
    camera_opencv_to_base_camera,
    image_write,
    link_poses_from_qpos,
    load_json,
    load_matrix,
    load_realman_settings,
    make_transform,
    qpos_samples_from_settings,
    save_json,
    save_matrix_pair,
)
from .reporting import normalize_calibration_report, write_html_report


DEFAULT_BASE_VISUAL_LINKS = [
    "base_link",
    "link_1",
    "link_2",
    "link_3",
    "link_4",
    "link_5",
    "link_6",
    "link_7",
    "gripper_base_link",
    "gripper_Left_1_Link",
    "gripper_Left_Support_Link",
    "gripper_Left_2_Link",
    "gripper_Right_1_Link",
    "gripper_Right_Support_Link",
    "gripper_Right_2_Link",
]
GRIPPER_JOINT_NAMES = [
    "gripper_Left_1_Joint",
    "gripper_Left_2_Joint",
    "gripper_Left_Support_Joint",
    "gripper_Right_1_Joint",
    "gripper_Right_2_Joint",
    "gripper_Right_Support_Joint",
]
DEFAULT_BASE_INITIAL_EXTRINSIC = ASSET_DIR / "calibration" / "camera_extrinsic_opencv_alignment_seed.npy"


def _mesh_in_meter_units(mesh):
    vertices = np.asarray(getattr(mesh, "vertices", []), dtype=np.float64)
    if vertices.size == 0:
        return mesh
    extent = np.ptp(vertices, axis=0)
    if float(np.nanmax(extent)) > 5.0:
        mesh = mesh.copy()
        mesh.apply_scale(0.001)
    return mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a fixed/base camera by visually aligning rendered RM75 links to robot masks."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ASSET_DIR / "calibration" / "rm75_calibration_config.example.json",
        help="JSON with realman_settings.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--use-previous-run", type=Path, default=None, help="Reuse image/link-pose/mask arrays from a previous run.")
    parser.add_argument("--execute-real", action="store_true", help="Move the real arm through reference_qpos_deg.")
    parser.add_argument("--manual-capture", action=argparse.BooleanOptionalAction, default=True, help="Manually pose the arm, then press Enter/Space to capture.")
    parser.add_argument("--manual-count", type=int, default=8)
    parser.add_argument("--manual-terminal", action="store_true", help="Use terminal-only manual capture without the live OpenCV preview.")
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preview-delay-ms", type=int, default=1)
    parser.add_argument("--robot-ip", type=str, default=None)
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--camera-serial", type=str, default=None)
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_RM75_URDF)
    parser.add_argument("--joint-names", nargs="+", default=None)
    parser.add_argument(
        "--gripper-joint-angle-rad",
        type=float,
        default=None,
        help="Override all RM75 gripper visualization joints to this angle in radians when rendering captured datasets.",
    )
    parser.add_argument(
        "--gripper-joint-angle-deg",
        type=float,
        default=None,
        help="Override all RM75 gripper visualization joints to this angle in degrees. Takes precedence over --gripper-joint-angle-rad.",
    )
    parser.add_argument(
        "--render-link-names",
        nargs="+",
        default=None,
        help="URDF visual links used by EasyHeC. Use '__all__' to include gripper/attached links.",
    )
    parser.add_argument("--reference-qpos-json", type=Path, default=None, help="JSON list of joint samples in degrees.")
    parser.add_argument("--mask-npy", type=Path, default=None)
    parser.add_argument("--mask-dir", type=Path, default=None, help="Directory with 0000.png, 0001.png, ... binary robot masks.")
    parser.add_argument(
        "--mask-render-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clean segmentation masks by keeping only pixels near the initial rendered robot silhouette.",
    )
    parser.add_argument("--mask-gate-dilation-px", type=int, default=90)
    parser.add_argument(
        "--mask-keep-largest-components",
        type=int,
        default=1,
        help="If >0, keep only the N largest connected components in each segmentation mask before optimization.",
    )
    parser.add_argument("--interactive-sam2", action=argparse.BooleanOptionalAction, default=True, help="Use EasyHEC interactive SAM2 segmentation to create mask.npy.")
    parser.add_argument("--sam2-root", type=Path, default=None)
    parser.add_argument("--sam2-model-cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-checkpoint", type=Path, default=None)
    parser.add_argument("--sam2-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--early-stopping-steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mask-loss-mode", choices=["l2", "dice"], default="dice")
    parser.add_argument("--mask-loss-scale", type=float, default=100000.0)
    parser.add_argument(
        "--presearch-samples",
        type=int,
        default=450,
        help="Low-resolution constrained random pose samples around the initial extrinsic before gradient refinement. Use 0 to disable.",
    )
    parser.add_argument("--presearch-scale", type=float, default=0.25)
    parser.add_argument("--presearch-translation-m", type=float, default=0.05)
    parser.add_argument("--presearch-rotation-deg", type=float, default=5.0)
    parser.add_argument("--presearch-seed", type=int, default=20260602)
    parser.add_argument("--presearch-degrade-tol", type=float, default=0.012)
    parser.add_argument("--selection-degrade-tol", type=float, default=0.006)
    parser.add_argument("--selection-degrade-weight", type=float, default=2.5)
    parser.add_argument("--no-drop-worst-initial-frame", action="store_true")
    parser.add_argument(
        "--initial-extrinsic-opencv-path",
        type=Path,
        default=None,
        help=f"Use an existing camera_extrinsic_opencv.npy/json as the EasyHeC initial pose and local-search center. Defaults to {DEFAULT_BASE_INITIAL_EXTRINSIC} if present.",
    )
    parser.add_argument("--pose-prior-weight", type=float, default=1000.0)
    parser.add_argument("--pose-prior-translation-sigma-m", type=float, default=0.05)
    parser.add_argument("--pose-prior-rotation-sigma-deg", type=float, default=5.0)
    parser.add_argument("--max-delta-translation-m", type=float, default=0.05)
    parser.add_argument("--max-delta-rotation-deg", type=float, default=5.0)
    parser.add_argument(
        "--robot-base-yaw-deg",
        type=float,
        default=0.0,
        help="Yaw from the saved camera-extrinsic robot base to the URDF/render base used by EasyHeC.",
    )
    parser.add_argument("--detect-board-after", action=argparse.BooleanOptionalAction, default=True, help="Detect the table board in captured frames after calibration.")
    parser.add_argument("--board-squares-x", type=int, default=None)
    parser.add_argument("--board-squares-y", type=int, default=None)
    parser.add_argument("--board-square-length-m", type=float, default=None)
    parser.add_argument("--board-marker-length-m", type=float, default=None)
    parser.add_argument("--board-dictionary", type=str, default=None)
    parser.add_argument("--board-legacy-pattern", action="store_true", default=None)
    parser.add_argument("--min-board-corners", type=int, default=None)
    parser.add_argument("--max-board-rmse-px", type=float, default=None)
    parser.add_argument("--min-corner-coverage", type=float, default=None)
    parser.add_argument("--no-html-report", action="store_true")
    parser.add_argument(
        "--save-to-app-assets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also copy camera_extrinsic_opencv.npy into assets/calibration.",
    )
    return parser.parse_args()


def _load_reference_qpos(args: argparse.Namespace, settings: dict) -> list[np.ndarray]:
    if args.reference_qpos_json is not None:
        raw = load_json(args.reference_qpos_json)
        return [np.deg2rad(np.asarray(item, dtype=np.float64).reshape(-1)) for item in raw]
    return qpos_samples_from_settings(settings, key="base_reference_qpos_deg")


def _normalize_sam2_paths(args: argparse.Namespace) -> tuple[str, str]:
    import sys

    sam2_root = args.sam2_root
    if sam2_root is None:
        candidates = [
            Path("/home/zhangzhao/Downloads/sam2"),
            Path("/home/zhangzhao/Desktop/lerobot-sim2real/sam2"),
            Path("/home/zhangzhao/Desktop/lerobot-sim2real/sam2(code)"),
            Path("/home/zhangzhao/Desktop/lerobot/lerobot-sim2real/sam2"),
            Path("/home/zhangzhao/PycharmProjects/sam2"),
        ]
        sam2_root = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    sam2_root = Path(sam2_root).expanduser()
    if sam2_root.exists():
        for item in (sam2_root, sam2_root / "sam2"):
            if str(item) not in sys.path:
                sys.path.insert(0, str(item))
    checkpoint = args.sam2_checkpoint
    if checkpoint is None:
        checkpoint = sam2_root / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
        if not checkpoint.exists():
            checkpoint = sam2_root / "checkpoints" / "sam2.1_hiera_large.pt"
    return str(args.sam2_model_cfg), str(Path(checkpoint).expanduser())


def _link_visual_meshes(link) -> list:
    link_meshes = []
    for visual in getattr(link, "visuals", []) or []:
        geometry = getattr(visual, "geometry", None)
        mesh = None if geometry is None else getattr(geometry, "mesh", None)
        meshes = None if mesh is None else getattr(mesh, "meshes", None)
        if meshes:
            origin = getattr(visual, "origin", None)
            for item in meshes:
                item = _mesh_in_meter_units(item)
                if origin is not None:
                    item = item.copy()
                    item.apply_transform(as_transform(origin))
                link_meshes.append(item)
    return link_meshes


def _requested_render_links(args: argparse.Namespace, settings: dict) -> list[str] | None:
    raw = args.render_link_names
    if raw is None:
        raw = settings.get("base_visual_links", DEFAULT_BASE_VISUAL_LINKS)
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [item.strip() for item in raw.split(",") if item.strip()]
    names = [str(item) for item in raw]
    if any(item.lower() in {"__all__", "all", "*"} for item in names):
        return None
    return names


def _render_all_requested(args: argparse.Namespace) -> bool:
    raw = args.render_link_names
    if raw is None:
        return False
    if isinstance(raw, str):
        raw = [item.strip() for item in raw.split(",") if item.strip()]
    return any(str(item).lower() in {"__all__", "all", "*"} for item in raw)


def _renderable_links_and_meshes(urdf, requested_names: list[str] | None):
    from easyhec.utils.utils_3d import merge_meshes

    all_names = []
    all_meshes = []
    for link in urdf.links:
        link_meshes = _link_visual_meshes(link)
        merged = merge_meshes(link_meshes)
        if merged is not None:
            all_names.append(link.name)
            all_meshes.append(merged)
    if requested_names is None:
        return all_names, all_meshes
    by_name = dict(zip(all_names, all_meshes, strict=True))
    missing = [name for name in requested_names if name not in by_name]
    if missing:
        raise KeyError(f"requested render links not found in URDF: {missing}; available: {all_names}")
    return requested_names, [by_name[name] for name in requested_names]


def _make_sam2_segmenter(args: argparse.Namespace):
    from contextlib import nullcontext
    from unittest.mock import patch

    model_cfg, checkpoint = _normalize_sam2_paths(args)
    import torch
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SAM2 is not importable. Pass --sam2-root /path/to/sam2, install SAM2 in this env, "
            "or rerun with --no-interactive-sam2 and provide --mask-dir/--mask-npy."
        ) from exc

    device = str(args.sam2_device)
    patch_ctx = patch("torch.cuda.is_available", return_value=False) if device == "cpu" else nullcontext()
    with patch_ctx:
        model = build_sam2(config_file=model_cfg, ckpt_path=checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)

    def segment(image: np.ndarray, clicked_points_np: np.ndarray) -> np.ndarray:
        input_label = clicked_points_np[:, 2]
        input_point = clicked_points_np[:, :2]
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
        with torch.inference_mode(), ctx:
            predictor.set_image(image)
            mask, _, _ = predictor.predict(
                input_point,
                input_label,
                multimask_output=False,
            )
        return mask[0]

    return segment


def _load_masks(args: argparse.Namespace, run_dir: Path, images: np.ndarray) -> np.ndarray:
    image_count = int(images.shape[0])
    if args.mask_npy is not None:
        masks = np.load(Path(args.mask_npy).expanduser())
    elif args.mask_dir is not None:
        items = []
        for idx in range(image_count):
            path = Path(args.mask_dir).expanduser() / f"{idx:04d}.png"
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"mask not found: {path}")
            items.append(mask > 0)
        masks = np.stack(items, axis=0)
    elif args.interactive_sam2:
        from easyhec.segmentation.interactive import InteractiveSegmentation

        segmenter = _make_sam2_segmenter(args)
        masks = InteractiveSegmentation(segmentation_model=segmenter).get_segmentation(images)
    else:
        mask_path = run_dir / "mask.npy"
        if mask_path.exists():
            masks = np.load(mask_path)
        else:
            raise FileNotFoundError(
                "No masks were provided. Pass --mask-npy, --mask-dir, --interactive-sam2, "
                "or put mask.npy in --use-previous-run."
            )
    masks = np.asarray(masks).astype(np.float32)
    if masks.shape[0] != image_count:
        raise ValueError(f"mask count {masks.shape[0]} does not match image count {image_count}")
    return masks


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


def _draw_capture_status(image_bgr: np.ndarray, *, sample_index: int, target_count: int, captured: bool = False) -> np.ndarray:
    out = image_bgr.copy()
    height, width = out.shape[:2]
    line = f"base camera sample {sample_index:02d}/{target_count:02d}   Enter/Space=capture   q/Esc=finish"
    if captured:
        line = f"captured sample {sample_index - 1:02d}   " + line
    overlay = out.copy()
    cv2.rectangle(overlay, (8, height - 54), (min(width - 8, 980), height - 8), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0.0, out)
    cv2.putText(out, line[:130], (18, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
    return out


def _extra_joint_values_from_args(args: argparse.Namespace) -> dict[str, float] | None:
    angle = args.gripper_joint_angle_rad
    if args.gripper_joint_angle_deg is not None:
        angle = float(np.deg2rad(float(args.gripper_joint_angle_deg)))
    if angle is None:
        return None
    value = float(np.clip(float(angle), 0.0, 0.91))
    print(f"[base-camera] gripper visualization joint override: {value:.4f} rad ({np.rad2deg(value):.2f} deg)")
    return {name: value for name in GRIPPER_JOINT_NAMES}


def _collect_dataset(args: argparse.Namespace, settings: dict, run_dir: Path):
    from urchin import URDF

    joint_names = list(args.joint_names or settings.get("joint_names") or [f"joint_{idx}" for idx in range(1, 8)])
    extra_joint_values = _extra_joint_values_from_args(args)
    manual_capture = bool(args.manual_capture)
    qpos_samples = [] if manual_capture else _load_reference_qpos(args, settings)
    if not manual_capture and not qpos_samples:
        raise ValueError("No reference qpos samples found. Provide base_reference_qpos_deg/reference_qpos_deg.")
    if not manual_capture and not args.execute_real:
        raise RuntimeError("Refusing to move/capture without --execute-real. Use --use-previous-run for offline calibration.")
    robot_ip = args.robot_ip or settings.get("robot_ip")
    if not robot_ip:
        raise ValueError("robot_ip is required in --config or --robot-ip")
    camera_cfg = settings.get("base_camera", settings.get("camera", {}))
    serial = args.camera_serial or camera_cfg.get("serial_number_or_name")
    width = int(args.camera_width or camera_cfg.get("width", 1280))
    height = int(args.camera_height or camera_cfg.get("height", 720))
    fps = int(args.camera_fps or camera_cfg.get("fps", 30))
    urdf = URDF.load(str(Path(args.urdf_path).expanduser()))
    renderable_link_names, meshes = _renderable_links_and_meshes(urdf, _requested_render_links(args, settings))
    save_json(run_dir / "renderable_link_names.json", renderable_link_names)
    print(f"[base-camera] EasyHeC render links: {renderable_link_names}")
    controller = RealmanSdkController(
        robot_ip=str(robot_ip),
        robot_port=int(args.robot_port or settings.get("robot_port", 8080)),
        joint_names=joint_names,
        calibration_offset=settings.get("calibration_offset", {}),
    )
    camera = RealSenseCamera(
        serial=serial,
        width=width,
        height=height,
        fps=fps,
        enable_depth=False,
        warmup_frames=int(args.warmup_frames),
    )
    images = []
    qpos_rows = []
    link_pose_rows = []
    intrinsic = None
    dist_coeffs = None
    window_name = "rm75 base camera manual calibration"
    try:
        camera.start()
        if manual_capture:
            target_count = max(1, int(args.manual_count))
            print("[base-camera] manual capture mode")
            print("[base-camera] Move the arm to a new visible pose, then press Enter/Space to capture.")
            print("[base-camera] Type q then Enter, or press q/Esc in the preview, to finish.")
            while len(images) < target_count:
                frame = camera.capture()
                if args.preview and not args.manual_terminal:
                    cv2.imshow(
                        window_name,
                        _draw_capture_status(frame.color_bgr, sample_index=len(images), target_count=target_count),
                    )
                    key_command = _manual_command_from_key(cv2.waitKey(max(1, int(args.preview_delay_ms))))
                else:
                    key_command = None
                stdin_command = _manual_command_from_stdin()
                command = key_command or stdin_command
                if args.manual_terminal and command is None:
                    value = input(f"[base-camera] sample {len(images):04d}/{target_count}: Enter=capture, q=finish > ")
                    command = "finish" if value.strip().lower() in {"q", "quit", "done", "exit"} else "capture"
                if command == "finish":
                    break
                if command != "capture":
                    continue
                idx = len(images)
                qpos_now = np.asarray(controller.get_qpos(flat=True), dtype=np.float64)
                images.append(frame.color_bgr)
                qpos_rows.append(qpos_now)
                image_write(run_dir / "images" / f"{idx:04d}.png", frame.color_bgr)
                link_poses = link_poses_from_qpos(urdf, qpos_now, joint_names, extra_joint_values=extra_joint_values)
                pose_row = np.zeros((len(renderable_link_names), 4, 4), dtype=np.float64)
                for link_idx, link_name in enumerate(renderable_link_names):
                    pose_row[link_idx] = link_poses[link_name]
                link_pose_rows.append(pose_row)
                intrinsic = frame.intrinsic
                dist_coeffs = frame.dist_coeffs
                print(f"[base-camera] captured sample {idx:04d} qpos_deg={np.rad2deg(qpos_now).round(3).tolist()}")
                if args.preview and not args.manual_terminal:
                    cv2.imshow(
                        window_name,
                        _draw_capture_status(frame.color_bgr, sample_index=len(images), target_count=target_count, captured=True),
                    )
                    cv2.waitKey(160)
        else:
            for idx, qpos in enumerate(qpos_samples):
                print(f"[base-camera] moving to sample {idx + 1}/{len(qpos_samples)}")
                controller.move_to_qpos(qpos)
                qpos_now = np.asarray(controller.get_qpos(flat=True), dtype=np.float64)
                frame = camera.capture()
                images.append(frame.color_bgr)
                qpos_rows.append(qpos_now)
                image_write(run_dir / "images" / f"{idx:04d}.png", frame.color_bgr)
                link_poses = link_poses_from_qpos(urdf, qpos_now, joint_names, extra_joint_values=extra_joint_values)
                pose_row = np.zeros((len(renderable_link_names), 4, 4), dtype=np.float64)
                for link_idx, link_name in enumerate(renderable_link_names):
                    pose_row[link_idx] = link_poses[link_name]
                link_pose_rows.append(pose_row)
                intrinsic = frame.intrinsic
                dist_coeffs = frame.dist_coeffs
    finally:
        if args.preview and not args.manual_terminal:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
        camera.stop()
        controller.disconnect()
    if not images or intrinsic is None or dist_coeffs is None:
        raise RuntimeError("No base-camera frames were captured.")
    link_poses_dataset = np.stack(link_pose_rows, axis=0)
    image_dataset = {"base_camera": np.stack(images, axis=0)}
    np.save(run_dir / "image_dataset.npy", image_dataset)
    np.save(run_dir / "link_poses_dataset.npy", link_poses_dataset)
    np.save(run_dir / "qpos_rad.npy", np.stack(qpos_rows, axis=0))
    np.save(run_dir / "camera_intrinsic.npy", intrinsic)
    np.save(run_dir / "camera_dist_coeffs.npy", dist_coeffs)
    return image_dataset, link_poses_dataset, intrinsic, dist_coeffs, meshes, renderable_link_names


def _link_pose_dataset_from_qpos(
    *,
    urdf,
    qpos_rows: np.ndarray,
    joint_names: list[str],
    renderable_link_names: list[str],
    extra_joint_values: dict[str, float] | None = None,
) -> np.ndarray:
    rows = []
    for qpos in np.asarray(qpos_rows, dtype=np.float64):
        link_poses = link_poses_from_qpos(urdf, qpos, joint_names, extra_joint_values=extra_joint_values)
        pose_row = np.zeros((len(renderable_link_names), 4, 4), dtype=np.float64)
        for link_idx, link_name in enumerate(renderable_link_names):
            pose_row[link_idx] = link_poses[link_name]
        rows.append(pose_row)
    return np.stack(rows, axis=0)


def _load_previous_dataset(run_dir: Path, args: argparse.Namespace, settings: dict):
    from urchin import URDF

    image_dataset = np.load(run_dir / "image_dataset.npy", allow_pickle=True).reshape(-1)[0]
    link_poses_dataset = np.load(run_dir / "link_poses_dataset.npy")
    intrinsic = np.load(run_dir / "camera_intrinsic.npy")
    dist_path = run_dir / "camera_dist_coeffs.npy"
    dist_coeffs = np.load(dist_path) if dist_path.exists() else np.zeros((5,), dtype=np.float64)
    extra_joint_values = _extra_joint_values_from_args(args)
    urdf = URDF.load(str(DEFAULT_RM75_URDF))
    all_names, all_meshes = _renderable_links_and_meshes(urdf, None)
    saved_names_path = run_dir / "renderable_link_names.json"
    if saved_names_path.exists():
        saved_names = [str(item) for item in load_json(saved_names_path)]
    elif link_poses_dataset.shape[1] == len(all_names):
        saved_names = all_names
    else:
        raise RuntimeError(
            f"{run_dir} has {link_poses_dataset.shape[1]} link-pose columns but no renderable_link_names.json; "
            "recapture the dataset or provide a run captured with the current calibration code."
        )
    requested_names = _requested_render_links(args, settings)
    if requested_names is None:
        renderable_link_names = all_names if _render_all_requested(args) else saved_names
    else:
        renderable_link_names = requested_names
    if extra_joint_values is None and all(name in saved_names for name in renderable_link_names):
        indices = [saved_names.index(name) for name in renderable_link_names]
        link_poses_dataset = link_poses_dataset[:, indices]
    else:
        qpos_path = run_dir / "qpos_rad.npy"
        if not qpos_path.exists():
            missing = [name for name in renderable_link_names if name not in saved_names]
            raise KeyError(
                f"requested render links were not present in previous dataset and qpos_rad.npy is missing: {missing}; "
                f"saved: {saved_names}"
            )
        joint_names = list(args.joint_names or settings.get("joint_names") or [f"joint_{idx}" for idx in range(1, 8)])
        qpos_rows = np.load(qpos_path)
        link_poses_dataset = _link_pose_dataset_from_qpos(
            urdf=urdf,
            qpos_rows=qpos_rows,
            joint_names=joint_names,
            renderable_link_names=renderable_link_names,
            extra_joint_values=extra_joint_values,
        )
    mesh_by_name = dict(zip(all_names, all_meshes, strict=True))
    meshes = [mesh_by_name[name] for name in renderable_link_names]
    print(f"[base-camera] EasyHeC render links: {renderable_link_names}")
    return image_dataset, link_poses_dataset, intrinsic, dist_coeffs, meshes, renderable_link_names


def _initial_extrinsic_from_settings(settings: dict) -> np.ndarray:
    from easyhec.utils.camera_conversions import ros2opencv
    from transforms3d.euler import euler2mat

    cfg = settings.get("initial_extrinsic_guess", {})
    xyz = np.asarray(cfg.get("xyz", [-0.4, 0.1, 0.5]), dtype=np.float64)
    if "rpy_rad" in cfg:
        rpy = np.asarray(cfg["rpy_rad"], dtype=np.float64)
    else:
        rpy = np.deg2rad(np.asarray(cfg.get("rpy_deg", [0.0, 45.0, -36.0]), dtype=np.float64))
    ros = make_transform(euler2mat(float(rpy[0]), float(rpy[1]), float(rpy[2])), xyz)
    return as_transform(ros2opencv(ros))


def _initial_extrinsic_from_args(args: argparse.Namespace, settings: dict) -> np.ndarray:
    if args.initial_extrinsic_opencv_path is not None:
        path = Path(args.initial_extrinsic_opencv_path).expanduser()
        initial = load_matrix(path)
        print(f"[base-camera] initial/search-center extrinsic loaded from {path}")
        return initial
    cfg = settings.get("base_camera_visual_calibration", {})
    configured_path = cfg.get("initial_extrinsic_opencv_path") or settings.get("base_initial_extrinsic_opencv_path")
    if configured_path:
        path = Path(configured_path).expanduser()
        initial = load_matrix(path)
        print(f"[base-camera] initial/search-center extrinsic loaded from config: {path}")
        return initial
    if DEFAULT_BASE_INITIAL_EXTRINSIC.exists():
        initial = load_matrix(DEFAULT_BASE_INITIAL_EXTRINSIC)
        print(f"[base-camera] initial/search-center extrinsic loaded from default seed: {DEFAULT_BASE_INITIAL_EXTRINSIC}")
        return initial
    return _initial_extrinsic_from_settings(settings)


def _yaw_transform(deg: float) -> np.ndarray:
    rad = np.deg2rad(float(deg))
    c, s = float(np.cos(rad)), float(np.sin(rad))
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return out


def _optimize_easyhec_fixed_batch(
    *,
    camera_intrinsic,
    masks,
    link_poses_dataset,
    initial_extrinsic_guess,
    meshes,
    camera_width: int,
    camera_height: int,
    iterations: int,
    learning_rate: float,
    batch_size: int | None,
    mask_loss_mode: str,
    mask_loss_scale: float,
    early_stopping_steps: int,
    pose_prior_weight: float,
    pose_prior_translation_sigma_m: float,
    pose_prior_rotation_sigma_deg: float,
    max_delta_translation_m: float,
    max_delta_rotation_deg: float,
    camera_mount_poses=None,
    gt_camera_pose=None,
):
    import torch
    from tqdm import tqdm
    from easyhec.optim.rb_solver import RBSolver, RBSolverConfig
    from easyhec.utils import utils_3d

    cfg = RBSolverConfig(
        camera_width=int(camera_width),
        camera_height=int(camera_height),
        robot_masks=masks,
        link_poses_dataset=link_poses_dataset,
        meshes=meshes,
        initial_extrinsic_guess=initial_extrinsic_guess,
    )
    solver = RBSolver(cfg).to(initial_extrinsic_guess.device)
    optimizer = torch.optim.Adam(solver.parameters(), lr=float(learning_rate))
    best_predicted_extrinsic = initial_extrinsic_guess.clone()
    best_loss = float("inf")
    best_mask_loss = float("inf")
    last_loss_improvement_step = 0
    sample_count = int(masks.shape[0])
    prior_center_dof = solver.dof.detach().clone()
    pose_prior_weight = float(pose_prior_weight)
    trans_sigma = max(float(pose_prior_translation_sigma_m), 1e-6)
    rot_sigma = max(float(np.deg2rad(float(pose_prior_rotation_sigma_deg))), 1e-6)
    max_delta_translation_m = max(0.0, float(max_delta_translation_m))
    max_delta_rotation_rad = max(0.0, float(np.deg2rad(float(max_delta_rotation_deg))))
    mask_loss_mode = str(mask_loss_mode)
    mask_loss_scale = float(mask_loss_scale)

    def render_batch_masks(batch_link_poses):
        Tc_c2b = utils_3d.se3_exp_map(solver.dof[None]).permute(0, 2, 1)[0]
        rendered_frames = []
        for bid in range(int(batch_link_poses.shape[0])):
            per_link = []
            for link_idx in range(solver.nlinks):
                Tc_c2l = Tc_c2b @ batch_link_poses[bid, link_idx]
                si = solver.renderer.render_mask(
                    getattr(solver, f"vertices_{link_idx}"),
                    getattr(solver, f"faces_{link_idx}"),
                    camera_intrinsic,
                    Tc_c2l,
                )
                per_link.append(si)
            rendered_frames.append(torch.stack(per_link).sum(0).clamp(max=1))
        return torch.stack(rendered_frames)

    def compute_mask_loss(batch_masks, batch_link_poses):
        if mask_loss_mode == "l2":
            output = solver(
                {
                    "intrinsic": camera_intrinsic,
                    "link_poses": batch_link_poses,
                    "mask": batch_masks,
                }
            )
            return output["mask_loss"]
        rendered = render_batch_masks(batch_link_poses)
        target = batch_masks.float()
        eps = torch.tensor(1e-6, dtype=rendered.dtype, device=rendered.device)
        intersection = torch.sum(rendered * target, dim=(1, 2))
        denom = torch.sum(rendered, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
        dice = (2.0 * intersection + eps) / (denom + eps)
        return (1.0 - dice).mean() * mask_loss_scale

    pbar = tqdm(range(int(iterations)))
    for step in pbar:
        if batch_size is None:
            batch_masks = masks
            batch_link_poses = link_poses_dataset
        else:
            bid = torch.randperm(sample_count, device=masks.device)[: max(1, int(batch_size))]
            batch_masks = masks[bid]
            batch_link_poses = link_poses_dataset[bid]
        optimizer.zero_grad()
        mask_loss = compute_mask_loss(batch_masks, batch_link_poses)
        prior_loss = torch.zeros((), dtype=mask_loss.dtype, device=mask_loss.device)
        if pose_prior_weight > 0.0:
            dof_delta = solver.dof - prior_center_dof
            trans_prior = torch.sum((dof_delta[:3] / trans_sigma) ** 2)
            rot_prior = torch.sum((dof_delta[3:] / rot_sigma) ** 2)
            prior_loss = float(pose_prior_weight) * (trans_prior + rot_prior)
        total_loss = mask_loss + prior_loss
        total_loss.backward()
        optimizer.step()
        if max_delta_translation_m > 0.0 or max_delta_rotation_rad > 0.0:
            with torch.no_grad():
                dof_delta = solver.dof - prior_center_dof
                if max_delta_translation_m > 0.0:
                    solver.dof[:3].copy_(prior_center_dof[:3] + torch.clamp(dof_delta[:3], -max_delta_translation_m, max_delta_translation_m))
                if max_delta_rotation_rad > 0.0:
                    solver.dof[3:].copy_(prior_center_dof[3:] + torch.clamp(dof_delta[3:], -max_delta_rotation_rad, max_delta_rotation_rad))
        loss_value = float(total_loss.item())
        mask_loss_value = float(mask_loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_mask_loss = mask_loss_value
            best_predicted_extrinsic = solver.get_predicted_extrinsic()
            last_loss_improvement_step = step
        if step - last_loss_improvement_step >= int(early_stopping_steps):
            break
        if pose_prior_weight > 0.0:
            pbar.set_description(f"Loss: {loss_value:.2f}, Mask: {mask_loss_value:.2f}, Best Mask: {best_mask_loss:.2f}")
        else:
            pbar.set_description(f"Loss: {loss_value:.2f}, Best Loss: {best_loss:.2f}")
    return best_predicted_extrinsic


def _render_masks_for_extrinsic(
    *,
    camera_intrinsic,
    link_poses_dataset,
    extrinsic,
    meshes,
    camera_width: int,
    camera_height: int,
    device,
):
    import torch
    from easyhec.optim.rb_solver import RBSolver, RBSolverConfig

    dummy_masks = torch.zeros(
        (int(link_poses_dataset.shape[0]), int(camera_height), int(camera_width)),
        dtype=torch.float32,
        device=device,
    )
    cfg = RBSolverConfig(
        camera_width=int(camera_width),
        camera_height=int(camera_height),
        robot_masks=dummy_masks,
        link_poses_dataset=link_poses_dataset,
        meshes=meshes,
        initial_extrinsic_guess=extrinsic,
    )
    solver = RBSolver(cfg).to(device)
    rendered = []
    with torch.no_grad():
        Tc_c2b = solver.get_predicted_extrinsic()
        for frame_idx in range(int(link_poses_dataset.shape[0])):
            per_link = []
            for link_idx in range(solver.nlinks):
                Tc_c2l = Tc_c2b @ link_poses_dataset[frame_idx, link_idx]
                si = solver.renderer.render_mask(
                    getattr(solver, f"vertices_{link_idx}"),
                    getattr(solver, f"faces_{link_idx}"),
                    camera_intrinsic,
                    Tc_c2l,
                )
                per_link.append(si)
            rendered.append(torch.stack(per_link).sum(0).clamp(max=1).detach().cpu().numpy())
    return np.asarray(rendered, dtype=np.float32)


def _clean_masks_with_render_gate(masks: np.ndarray, rendered_masks: np.ndarray, dilation_px: int) -> np.ndarray:
    dilation_px = max(0, int(dilation_px))
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1))
    cleaned = []
    stats = []
    for idx, (mask, rendered) in enumerate(zip(masks, rendered_masks, strict=True)):
        raw = np.asarray(mask, dtype=np.float32) > 0.5
        gate = np.asarray(rendered, dtype=np.float32) > 0.5
        if dilation_px > 0:
            gate = cv2.dilate(gate.astype(np.uint8), kernel, iterations=1) > 0
        keep = raw & gate
        raw_area = int(raw.sum())
        keep_area = int(keep.sum())
        render_area = int((np.asarray(rendered) > 0.5).sum())
        if raw_area > 0 and keep_area < max(2000, int(0.15 * raw_area)):
            print(
                f"[base-camera] mask gate frame {idx:04d}: kept only {keep_area}/{raw_area} px; "
                "falling back to raw mask because initial render gate may be too far."
            )
            keep = raw
            keep_area = raw_area
        cleaned.append(keep.astype(np.float32))
        stats.append({"frame": idx, "raw_px": raw_area, "kept_px": keep_area, "render_px": render_area})
    print("[base-camera] mask render gate stats:", stats)
    return np.stack(cleaned, axis=0)


def _keep_largest_mask_components(masks: np.ndarray, keep_count: int) -> np.ndarray:
    keep_count = int(keep_count)
    if keep_count <= 0:
        return masks
    cleaned = []
    stats = []
    for idx, mask in enumerate(masks):
        raw = np.asarray(mask, dtype=np.float32) > 0.5
        num, labels, comp_stats, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8), 8)
        components = []
        for label in range(1, num):
            area = int(comp_stats[label, cv2.CC_STAT_AREA])
            components.append((area, label))
        components.sort(reverse=True)
        keep_labels = {label for _, label in components[:keep_count]}
        keep = np.isin(labels, list(keep_labels)) if keep_labels else np.zeros_like(raw, dtype=bool)
        stats.append(
            {
                "frame": idx,
                "raw_px": int(raw.sum()),
                "kept_px": int(keep.sum()),
                "kept_components": [(int(area), int(label)) for area, label in components[:keep_count]],
            }
        )
        cleaned.append(keep.astype(np.float32))
    print("[base-camera] mask component filter stats:", stats)
    return np.stack(cleaned, axis=0)


def _delta_pose_matrix(delta: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    delta = np.asarray(delta, dtype=np.float64).reshape(6)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rotation.from_rotvec(delta[3:]).as_matrix()
    out[:3, 3] = delta[:3]
    return out


def _mask_ious(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.asarray(predicted) > 0.5
    tgt = np.asarray(target) > 0.5
    inter = np.logical_and(pred, tgt).sum(axis=(1, 2))
    union = np.logical_or(pred, tgt).sum(axis=(1, 2))
    return inter / np.maximum(union, 1)


def _selected_iou_score(
    candidate_iou: np.ndarray,
    baseline_iou: np.ndarray,
    *,
    drop_worst: bool,
    degrade_tol: float,
    degrade_weight: float,
) -> tuple[float, list[int], list[float]]:
    candidate_iou = np.asarray(candidate_iou, dtype=np.float64).reshape(-1)
    baseline_iou = np.asarray(baseline_iou, dtype=np.float64).reshape(-1)
    selected = list(range(candidate_iou.size))
    if drop_worst and len(selected) > 2:
        worst = int(np.argmin(baseline_iou))
        selected = [idx for idx in selected if idx != worst]
    selected_arr = np.asarray(selected, dtype=np.int64)
    degradation = np.maximum(0.0, baseline_iou[selected_arr] - float(degrade_tol) - candidate_iou[selected_arr])
    score = float(np.mean(candidate_iou[selected_arr]) - float(degrade_weight) * np.mean(degradation))
    return score, selected, degradation.tolist()


def _render_with_existing_solver(*, solver, camera_intrinsic, link_poses_dataset, extrinsic: np.ndarray) -> np.ndarray:
    import torch

    extrinsic_t = torch.from_numpy(np.asarray(extrinsic, dtype=np.float32)).to(camera_intrinsic.device)
    rendered = []
    with torch.no_grad():
        for frame_idx in range(int(link_poses_dataset.shape[0])):
            per_link = []
            for link_idx in range(solver.nlinks):
                Tc_c2l = extrinsic_t @ link_poses_dataset[frame_idx, link_idx]
                si = solver.renderer.render_mask(
                    getattr(solver, f"vertices_{link_idx}"),
                    getattr(solver, f"faces_{link_idx}"),
                    camera_intrinsic,
                    Tc_c2l,
                )
                per_link.append(si)
            rendered.append(torch.stack(per_link).sum(0).clamp(max=1).detach().cpu().numpy())
    return np.asarray(rendered, dtype=np.float32)


def _presearch_initial_extrinsic(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    intrinsic: np.ndarray,
    masks: np.ndarray,
    link_poses_dataset: np.ndarray,
    meshes,
    initial_solver: np.ndarray,
    device,
) -> tuple[np.ndarray, dict]:
    import torch
    from easyhec.optim.rb_solver import RBSolver, RBSolverConfig

    samples = max(0, int(args.presearch_samples))
    if samples <= 0:
        return initial_solver, {"enabled": False}
    scale = float(args.presearch_scale)
    scale = min(1.0, max(0.05, scale))
    height_full, width_full = int(masks.shape[1]), int(masks.shape[2])
    width = max(32, int(round(width_full * scale)))
    height = max(18, int(round(height_full * scale)))
    search_masks = np.stack(
        [
            cv2.resize((np.asarray(mask) > 0.5).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            for mask in masks
        ],
        axis=0,
    ).astype(bool)
    search_intrinsic = np.asarray(intrinsic, dtype=np.float32).copy()
    search_intrinsic[0, :] *= width / float(width_full)
    search_intrinsic[1, :] *= height / float(height_full)
    intrinsic_t = torch.from_numpy(search_intrinsic).float().to(device)
    link_poses_t = torch.from_numpy(np.asarray(link_poses_dataset, dtype=np.float32)).to(device)
    dummy_masks = torch.zeros((int(search_masks.shape[0]), height, width), dtype=torch.float32, device=device)
    cfg = RBSolverConfig(
        camera_width=width,
        camera_height=height,
        robot_masks=dummy_masks,
        link_poses_dataset=link_poses_t,
        meshes=meshes,
        initial_extrinsic_guess=torch.from_numpy(np.asarray(initial_solver, dtype=np.float32)).to(device),
    )
    solver = RBSolver(cfg).to(device)

    baseline_rendered = _render_with_existing_solver(
        solver=solver,
        camera_intrinsic=intrinsic_t,
        link_poses_dataset=link_poses_t,
        extrinsic=initial_solver,
    )
    baseline_iou = _mask_ious(baseline_rendered, search_masks)
    drop_worst = not bool(args.no_drop_worst_initial_frame)
    bounds = np.asarray(
        [
            float(args.presearch_translation_m),
            float(args.presearch_translation_m),
            float(args.presearch_translation_m),
            np.deg2rad(float(args.presearch_rotation_deg)),
            np.deg2rad(float(args.presearch_rotation_deg)),
            np.deg2rad(float(args.presearch_rotation_deg)),
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(args.presearch_seed))
    best_delta = np.zeros(6, dtype=np.float64)
    best_iou = baseline_iou.copy()
    best_score, selected, degradation = _selected_iou_score(
        best_iou,
        baseline_iou,
        drop_worst=drop_worst,
        degrade_tol=float(args.presearch_degrade_tol),
        degrade_weight=float(args.selection_degrade_weight),
    )
    best_score -= 0.0
    accepted = 0
    history = []

    def score_delta(delta: np.ndarray) -> tuple[float, np.ndarray, list[float]]:
        candidate = _delta_pose_matrix(delta) @ initial_solver
        rendered = _render_with_existing_solver(
            solver=solver,
            camera_intrinsic=intrinsic_t,
            link_poses_dataset=link_poses_t,
            extrinsic=candidate,
        )
        iou = _mask_ious(rendered, search_masks)
        score, _, candidate_degradation = _selected_iou_score(
            iou,
            baseline_iou,
            drop_worst=drop_worst,
            degrade_tol=float(args.presearch_degrade_tol),
            degrade_weight=float(args.selection_degrade_weight),
        )
        move_penalty = 0.01 * float(np.linalg.norm(delta[:3]) / max(float(args.presearch_translation_m), 1e-6))
        rot_penalty = 0.005 * float(np.linalg.norm(delta[3:]) / max(np.deg2rad(float(args.presearch_rotation_deg)), 1e-6))
        return score - move_penalty - rot_penalty, iou, candidate_degradation

    for idx in range(samples):
        delta = rng.uniform(-bounds, bounds)
        if idx < int(0.27 * samples):
            delta *= 0.35
        elif idx < int(0.58 * samples):
            delta *= 0.65
        score, iou, candidate_degradation = score_delta(delta)
        if max(candidate_degradation, default=0.0) <= max(0.025, float(args.presearch_degrade_tol) * 2.0):
            accepted += 1
        if score > best_score:
            best_score = score
            best_delta = delta.copy()
            best_iou = iou.copy()
            history.append(
                {
                    "sample": idx,
                    "score": score,
                    "delta_translation_m": best_delta[:3],
                    "delta_rotation_deg": np.rad2deg(best_delta[3:]),
                    "ious": best_iou,
                }
            )
            print(
                "[base-camera][presearch] best",
                idx,
                f"score={score:.4f}",
                f"mean_iou={float(np.mean(best_iou)):.4f}",
                "dx_mm=",
                np.round(best_delta[:3] * 1000.0, 1).tolist(),
                "dr_deg=",
                np.round(np.rad2deg(best_delta[3:]), 2).tolist(),
            )
    best_extrinsic = _delta_pose_matrix(best_delta) @ initial_solver
    report = {
        "enabled": True,
        "samples": samples,
        "scale": scale,
        "resolution": [width, height],
        "selected_frames": selected,
        "accepted_samples": accepted,
        "baseline_iou": baseline_iou,
        "best_iou": best_iou,
        "best_score": best_score,
        "best_delta_translation_m": best_delta[:3],
        "best_delta_rotation_deg": np.rad2deg(best_delta[3:]),
        "history": history,
    }
    save_json(run_dir / "presearch_report.json", report)
    print(
        "[base-camera][presearch] selected frames",
        selected,
        f"mean IoU {float(np.mean(baseline_iou)):.4f} -> {float(np.mean(best_iou)):.4f}",
    )
    return best_extrinsic, report


def _choose_solver_extrinsic_by_iou(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    candidates: dict[str, np.ndarray],
    baseline_name: str,
    masks: np.ndarray,
    intrinsic_t,
    link_poses_dataset_t,
    meshes,
    width: int,
    height: int,
    device,
) -> tuple[str, np.ndarray, dict]:
    candidate_ious = {}
    rendered_cache = {}
    for name, extrinsic in candidates.items():
        rendered = _render_masks_for_extrinsic(
            camera_intrinsic=intrinsic_t,
            link_poses_dataset=link_poses_dataset_t,
            extrinsic=torch_from_numpy(extrinsic, device),
            meshes=meshes,
            camera_width=width,
            camera_height=height,
            device=device,
        )
        rendered_cache[name] = rendered
        candidate_ious[name] = _mask_ious(rendered, masks)
    baseline_iou = candidate_ious[baseline_name]
    scores = {}
    selected_frames = None
    for name, iou in candidate_ious.items():
        score, selected, degradation = _selected_iou_score(
            iou,
            baseline_iou,
            drop_worst=not bool(args.no_drop_worst_initial_frame),
            degrade_tol=float(args.selection_degrade_tol),
            degrade_weight=float(args.selection_degrade_weight),
        )
        selected_frames = selected
        scores[name] = {
            "score": score,
            "mean_iou": float(np.mean(iou)),
            "selected_mean_iou": float(np.mean(iou[selected])),
            "ious": iou,
            "degradation": degradation,
        }
    best_name = max(scores, key=lambda key: scores[key]["score"])
    report = {
        "selected_frames": selected_frames,
        "baseline": baseline_name,
        "best": best_name,
        "scores": scores,
    }
    save_json(run_dir / "candidate_selection_report.json", report)
    print("[base-camera] candidate selection:", {name: round(item["score"], 4) for name, item in scores.items()}, "best=", best_name)
    return best_name, candidates[best_name], report


def torch_from_numpy(matrix: np.ndarray, device):
    import torch

    return torch.from_numpy(np.asarray(matrix, dtype=np.float32)).to(device)


def main() -> int:
    args = parse_args()
    settings = load_realman_settings(args.config)
    run_dir = Path(args.use_previous_run).expanduser().resolve() if args.use_previous_run else calibration_run_dir("base_camera_visual", args.output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.use_previous_run:
        image_dataset, link_poses_dataset, intrinsic, dist_coeffs, meshes, link_names = _load_previous_dataset(run_dir, args, settings)
    else:
        image_dataset, link_poses_dataset, intrinsic, dist_coeffs, meshes, link_names = _collect_dataset(args, settings, run_dir)
    images = np.asarray(image_dataset["base_camera"])
    masks = _load_masks(args, run_dir, images)

    import torch
    from easyhec.utils.camera_conversions import opencv2ros
    from easyhec.utils import visualization

    if not torch.cuda.is_available():
        raise RuntimeError("EasyHEC/nvdiffrast calibration requires CUDA.")
    initial = _initial_extrinsic_from_args(args, settings)
    robot_to_render_base = _yaw_transform(float(args.robot_base_yaw_deg))
    initial_solver = initial @ np.linalg.inv(robot_to_render_base)
    print(f"[base-camera] robot-base -> render-base yaw: {float(args.robot_base_yaw_deg):.3f} deg")
    device = torch.device("cuda")
    camera_intrinsic_t = torch.from_numpy(intrinsic).float().to(device)
    link_poses_dataset_t = torch.from_numpy(link_poses_dataset).float().to(device)
    initial_t = torch.from_numpy(initial_solver).float().to(device)
    if args.mask_keep_largest_components > 0:
        raw_masks = masks.copy()
        np.save(run_dir / "mask_raw.npy", raw_masks)
        masks = _keep_largest_mask_components(raw_masks, args.mask_keep_largest_components)
    if args.mask_render_gate:
        raw_masks = masks.copy()
        np.save(run_dir / "mask_raw.npy", raw_masks)
        rendered_initial = _render_masks_for_extrinsic(
            camera_intrinsic=camera_intrinsic_t,
            link_poses_dataset=link_poses_dataset_t,
            extrinsic=initial_t,
            meshes=meshes,
            camera_width=int(images.shape[2]),
            camera_height=int(images.shape[1]),
            device=device,
        )
        np.save(run_dir / "mask_render_gate_initial.npy", rendered_initial)
        masks = _clean_masks_with_render_gate(raw_masks, rendered_initial, args.mask_gate_dilation_px)
    np.save(run_dir / "mask.npy", masks)
    mask_areas = np.mean(masks.reshape(masks.shape[0], -1) > 0.5, axis=1)
    presearch_solver_opencv, presearch_report = _presearch_initial_extrinsic(
        args=args,
        run_dir=run_dir,
        intrinsic=intrinsic,
        masks=masks,
        link_poses_dataset=link_poses_dataset,
        meshes=meshes,
        initial_solver=initial_solver,
        device=device,
    )
    save_matrix_pair(run_dir / "camera_extrinsic_presearch_solver_opencv.npy", presearch_solver_opencv)
    save_matrix_pair(run_dir / "camera_extrinsic_presearch_opencv.npy", presearch_solver_opencv @ robot_to_render_base)
    masks_t = torch.from_numpy(masks).float().to(device)
    optimizer_initial_t = torch.from_numpy(presearch_solver_opencv).float().to(device)
    predicted_solver_opencv = _optimize_easyhec_fixed_batch(
        camera_intrinsic=camera_intrinsic_t,
        masks=masks_t,
        link_poses_dataset=link_poses_dataset_t,
        initial_extrinsic_guess=optimizer_initial_t,
        meshes=meshes,
        camera_width=int(images.shape[2]),
        camera_height=int(images.shape[1]),
        camera_mount_poses=None,
        gt_camera_pose=None,
        iterations=int(args.train_steps),
        learning_rate=float(args.learning_rate),
        batch_size=args.batch_size,
        mask_loss_mode=str(args.mask_loss_mode),
        mask_loss_scale=float(args.mask_loss_scale),
        early_stopping_steps=int(args.early_stopping_steps),
        pose_prior_weight=float(args.pose_prior_weight),
        pose_prior_translation_sigma_m=float(args.pose_prior_translation_sigma_m),
        pose_prior_rotation_sigma_deg=float(args.pose_prior_rotation_sigma_deg),
        max_delta_translation_m=float(args.max_delta_translation_m),
        max_delta_rotation_deg=float(args.max_delta_rotation_deg),
    ).detach().cpu().numpy()
    candidate_solver_extrinsics = {
        "seed": initial_solver,
        "presearch": presearch_solver_opencv,
        "optimized": predicted_solver_opencv,
    }
    selected_candidate_name, predicted_solver_opencv, selection_report = _choose_solver_extrinsic_by_iou(
        args=args,
        run_dir=run_dir,
        candidates=candidate_solver_extrinsics,
        baseline_name="seed",
        masks=masks,
        intrinsic_t=camera_intrinsic_t,
        link_poses_dataset_t=link_poses_dataset_t,
        meshes=meshes,
        width=int(images.shape[2]),
        height=int(images.shape[1]),
        device=device,
    )
    predicted_opencv = predicted_solver_opencv @ robot_to_render_base
    predicted_ros = opencv2ros(predicted_opencv)
    base_T_camera = camera_opencv_to_base_camera(predicted_opencv, use_direct=False)
    save_matrix_pair(run_dir / "camera_extrinsic_opencv.npy", predicted_opencv)
    save_matrix_pair(run_dir / "camera_extrinsic_solver_opencv.npy", predicted_solver_opencv)
    save_matrix_pair(run_dir / "camera_extrinsic_ros.npy", predicted_ros)
    save_matrix_pair(run_dir / "base_T_camera.npy", base_T_camera)
    np.save(run_dir / "camera_intrinsic.npy", intrinsic)
    np.save(run_dir / "camera_dist_coeffs.npy", dist_coeffs)
    visualization.visualize_extrinsic_results(
        images=images,
        link_poses_dataset=link_poses_dataset,
        meshes=meshes,
        intrinsic=intrinsic,
        extrinsics=np.stack([initial_solver, presearch_solver_opencv, predicted_solver_opencv]),
        masks=masks,
        labels=["Seed Solver Guess", "Presearch Solver Extrinsic", f"Selected Solver Extrinsic ({selected_candidate_name})"],
        output_dir=str(run_dir),
    )
    board_summary = None
    board_outliers = []
    if args.detect_board_after:
        cfg = board_config_from_args(args, settings)
        estimates = []
        detections = []
        for idx, image in enumerate(images):
            det = detect_board_pose(image, intrinsic, dist_coeffs, cfg)
            detections.append(det.jsonable())
            draw_board_overlay(image, det, intrinsic, dist_coeffs, run_dir / "board_overlays" / f"{idx:04d}.png")
            accepted_board = detection_is_accepted(det, cfg) and det.T_cam_board is not None
            if accepted_board:
                estimates.append(base_T_camera @ as_transform(det.T_cam_board))
            else:
                board_outliers.append({"index": idx, "reason": det.reason, "quality": det.quality})
        if estimates:
            from .common import average_transforms

            base_T_board = average_transforms(estimates)
            save_matrix_pair(run_dir / "base_T_board.npy", base_T_board)
            board_summary = {"accepted_frames": len(estimates), "base_T_board": base_T_board, "board_config": vars(cfg)}
        save_json(run_dir / "base_board_detections.json", detections)
    report = {
        "run_dir": run_dir,
        "frames": int(images.shape[0]),
        "renderable_links": list(link_names),
        "mask_area_mean": float(np.mean(mask_areas)),
        "mask_area_min": float(np.min(mask_areas)),
        "mask_area_max": float(np.max(mask_areas)),
        "camera_intrinsic": intrinsic,
        "initial_camera_extrinsic_opencv": initial,
        "initial_camera_extrinsic_solver_opencv": initial_solver,
        "presearch": presearch_report,
        "camera_extrinsic_presearch_solver_opencv": presearch_solver_opencv,
        "selected_candidate": selected_candidate_name,
        "candidate_selection": selection_report,
        "camera_extrinsic_opencv": predicted_opencv,
        "camera_extrinsic_solver_opencv": predicted_solver_opencv,
        "base_T_camera": base_T_camera,
        "robot_base_yaw_deg": float(args.robot_base_yaw_deg),
        "board": board_summary,
        "outliers": board_outliers,
    }
    report = normalize_calibration_report(
        report,
        run_kind="base_camera_visual",
        config_path=args.config,
        frames_total=int(images.shape[0]),
        frames_used=int(images.shape[0]),
        accepted=True,
        transforms={
            "T_R_Cg": base_T_camera,
            "T_R_P": None if board_summary is None else board_summary.get("base_T_board"),
        },
        metrics={
            "mask_area_mean": float(np.mean(mask_areas)),
            "mask_area_min": float(np.min(mask_areas)),
            "mask_area_max": float(np.max(mask_areas)),
            "board": board_summary,
        },
    )
    save_json(run_dir / "calibration_report.json", report)
    if not args.no_html_report:
        image_dirs = ["board_overlays"] if args.detect_board_after else []
        write_html_report(
            run_dir / "report.html",
            title="RM75 Base Camera Visual Calibration",
            summary=report,
            sections=[("Board", board_summary or {})],
            image_dirs=image_dirs,
        )
    if args.save_to_app_assets:
        from rm75_app.paths import ASSET_DIR

        target = ASSET_DIR / "calibration" / "camera_extrinsic_opencv.npy"
        save_matrix_pair(target, predicted_opencv)
        save_matrix_pair(ASSET_DIR / "calibration" / "base_T_camera.npy", base_T_camera)
        print(f"[base-camera] saved app default extrinsic: {target}")
    print(f"[base-camera] calibration written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
