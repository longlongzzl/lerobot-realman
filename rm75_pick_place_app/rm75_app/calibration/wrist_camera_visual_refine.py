from __future__ import annotations

import argparse
import select
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rm75_app.paths import ASSET_DIR, DEFAULT_RM75_URDF, RUNTIME_DIR

from .common import (
    RealSenseCamera,
    RealmanSdkController,
    as_transform,
    calibration_run_dir,
    image_write,
    invert_transform,
    link_poses_from_qpos,
    load_json,
    load_matrix,
    load_realman_settings,
    load_urdf,
    make_transform,
    save_json,
    save_matrix_pair,
    select_link_pose,
)
from .reporting import normalize_calibration_report, write_html_report


DEFAULT_BODY_LINKS = ("base_link", "link_1", "link_2", "link_3")


@dataclass
class CapturedFrame:
    index: int
    image_bgr: np.ndarray
    qpos_rad: np.ndarray
    base_T_ee: np.ndarray
    link_poses: dict[str, np.ndarray]


@dataclass
class LinkMesh:
    link_name: str
    vertices: np.ndarray
    faces: np.ndarray
    source_files: list[str]


@dataclass
class PreparedFrame:
    index: int
    image_bgr: np.ndarray
    base_T_ee: np.ndarray
    link_poses: dict[str, np.ndarray]
    used_link_names: list[str]
    link_projected_area_px: dict[str, int]
    observed_edges: np.ndarray
    observed_dt: np.ndarray
    observed_mask: np.ndarray | None
    mask_source: str
    mask_selection: dict[str, Any]
    initial_mask: np.ndarray
    edge_pixels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine RM75 wrist-camera hand-eye calibration by jointly aligning robot body silhouettes."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ASSET_DIR / "calibration" / "rm75_calibration_config.example.json",
        help="JSON with realman_settings.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--input-run", type=Path, default=None, help="Reuse images/qpos from a previous wrist or visual-refine run.")
    parser.add_argument("--use-previous-run", type=Path, default=None, help="Reuse and write into a previous visual-refine run.")
    parser.add_argument("--wrist-camera-run-dir", type=Path, default=None, help="Run dir containing ee_T_wrist_camera.npy from ChArUco calibration.")
    parser.add_argument("--initial-ee-T-camera", type=Path, default=None, help="Initial ee_T_wrist_camera .npy/.json path.")
    parser.add_argument(
        "--manual-capture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Manually capture wrist images at the current arm pose. Enabled by default.",
    )
    parser.add_argument("--manual-count", type=int, default=12)
    parser.add_argument("--manual-terminal", action="store_true", help="Use terminal-only manual capture without the live OpenCV preview.")
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
    parser.add_argument("--preview-delay-ms", type=int, default=1)
    parser.add_argument("--urdf-path", type=Path, default=None)
    parser.add_argument("--ee-link-name", type=str, default=None, help="URDF link that the wrist camera is mounted to.")
    parser.add_argument("--joint-names", nargs="+", default=None)
    parser.add_argument("--include-links", nargs="+", default=None, help="Robot body links to use. Default: base_link link_1 link_2 link_3.")
    parser.add_argument("--exclude-link-substrings", nargs="+", default=["gripper", "pad", "camera"], help="Ignored link-name substrings when --include-links all is used.")
    parser.add_argument("--render-mode", choices=["hull", "triangles"], default="triangles", help="triangles is closer to the mesh; hull is faster but coarser.")
    parser.add_argument("--optimize-scale", type=float, default=0.5, help="Downsample factor for the silhouette loss.")
    parser.add_argument("--near-clip-m", type=float, default=0.035, help="Cull mesh vertices closer than this to the wrist camera.")
    parser.add_argument("--max-triangle-edge-px", type=float, default=420.0, help="Cull projected mesh triangles with implausibly long image edges.")
    parser.add_argument("--optimizer", choices=["nvdiffrast", "powell"], default="nvdiffrast", help="Use differentiable nvdiffrast silhouette optimization or the legacy discrete Powell optimizer.")
    parser.add_argument("--nvdiffrast-lr", type=float, default=0.045)
    parser.add_argument("--mask-dir", type=Path, default=None, help="Optional per-frame binary robot-body masks named 0000.png, 0001.png, ... . Missing frames fall back to SAM3/Canny.")
    parser.add_argument("--mask-npy", type=Path, default=None, help="Optional binary robot-body masks array [N,H,W]. Enables EasyHEC-style silhouette mask loss.")
    parser.add_argument("--sam3-mask", action=argparse.BooleanOptionalAction, default=False, help="Use SAM3 to generate robot-body masks from the projected robot ROI before refinement.")
    parser.add_argument("--sam3-python", type=Path, default=Path("/home/zhangzhao/anaconda3/envs/sam3/bin/python"))
    parser.add_argument("--sam3-provider-script", type=Path, default=Path(__file__).resolve().parents[1] / "perception" / "sam3_mask_provider.py")
    parser.add_argument("--sam3-checkpoint-path", type=Path, default=Path("/home/zhangzhao/Downloads/sam3.pt"))
    parser.add_argument("--sam3-prompt", type=str, default="white robotic arm joint")
    parser.add_argument("--sam3-device", type=str, default="cuda")
    parser.add_argument("--sam3-confidence-threshold", type=float, default=0.2)
    parser.add_argument("--sam3-resolution", type=int, default=1008)
    parser.add_argument("--sam3-min-mask-area", type=int, default=500)
    parser.add_argument("--sam3-morph-kernel", type=int, default=3)
    parser.add_argument("--sam3-max-masks-per-item", type=int, default=4)
    parser.add_argument("--sam3-min-projected-overlap", type=float, default=0.02, help="Reject SAM3 masks whose area barely overlaps the current projected robot mask.")
    parser.add_argument("--sam3-box-padding-px", type=float, default=70.0)
    parser.add_argument("--sam3-timeout-s", type=float, default=90.0)
    parser.add_argument("--sam3-reuse-masks", action=argparse.BooleanOptionalAction, default=True, help="Reuse generated masks in the run directory when present.")
    parser.add_argument("--save-sam3-candidates", action="store_true", help="Also copy accepted-frame SAM3 candidate masks/overlays into sam3_candidates/. Rejected-frame candidates are always saved.")
    parser.add_argument("--min-link-projected-area-px", type=float, default=200.0, help="Skip links whose initial projected mask area is below this many optimized-resolution pixels.")
    parser.add_argument("--min-mask-dice-improvement", type=float, default=0.0, help="When mask targets exist, require this Dice improvement before accepting the refined transform.")
    parser.add_argument("--allow-worse-refine", action="store_true", help="Keep the optimized transform even if the validation edge metric gets worse.")
    parser.add_argument("--canny-low", type=float, default=45.0)
    parser.add_argument("--canny-high", type=float, default=130.0)
    parser.add_argument("--roi-dilate-px", type=int, default=45)
    parser.add_argument("--edge-dilate-px", type=int, default=1)
    parser.add_argument("--min-observed-edge-pixels", type=int, default=80)
    parser.add_argument("--max-edge-samples", type=int, default=800)
    parser.add_argument("--loss-clip-px", type=float, default=20.0)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--max-translation-delta-m", type=float, default=0.03)
    parser.add_argument("--max-rotation-delta-deg", type=float, default=5.0)
    parser.add_argument("--translation-prior-m", type=float, default=0.012)
    parser.add_argument("--rotation-prior-deg", type=float, default=2.0)
    parser.add_argument("--regularization-weight", type=float, default=0.15)
    parser.add_argument("--no-html-report", action="store_true")
    parser.add_argument("--write-refined-to-wrist-run", action="store_true", help="Also write ee_T_wrist_camera_refined.npy/json into --wrist-camera-run-dir.")
    return parser.parse_args()


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
    line = f"sample {sample_index:02d}/{target_count:02d}   Enter/Space: capture   q/Esc: finish"
    if captured:
        line = f"captured sample {sample_index - 1:02d}   " + line
    overlay = out.copy()
    cv2.rectangle(overlay, (8, height - 54), (min(width - 8, 930), height - 8), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    cv2.putText(out, line[:120], (18, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
    return out


def _float_list(text: str | None, count: int, default: list[float]) -> list[float]:
    if not text:
        return list(default)
    values = [float(item) for item in text.split()]
    if len(values) != count:
        raise ValueError(f"expected {count} values, got {len(values)} in {text!r}")
    return values


def _rpy_matrix(rpy: list[float]) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def _origin_transform(origin: ET.Element | None) -> np.ndarray:
    if origin is None:
        return np.eye(4, dtype=np.float64)
    xyz = _float_list(origin.attrib.get("xyz"), 3, [0.0, 0.0, 0.0])
    rpy = _float_list(origin.attrib.get("rpy"), 3, [0.0, 0.0, 0.0])
    return make_transform(_rpy_matrix(rpy), np.asarray(xyz, dtype=np.float64))


def _resolve_mesh_path(filename: str, urdf_path: Path) -> Path:
    raw = str(filename)
    if raw.startswith("package://"):
        raw = raw.split("package://", 1)[1]
        parts = Path(raw).parts
        raw = str(Path(*parts[1:])) if len(parts) > 1 else str(Path(raw).name)
        candidate = urdf_path.parent.parent / raw
    else:
        candidate = (urdf_path.parent / raw).resolve()
    if candidate.suffix.lower() == ".dae":
        # The RM75 DAE files in this asset tree load as millimeters in trimesh,
        # while the matching STL files are in URDF meters. Prefer STL geometry
        # for calibration math when both are present.
        for suffix in (".STL", ".stl", ".obj"):
            alt = candidate.with_suffix(suffix)
            if alt.exists():
                return alt
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"mesh file not found: {filename} resolved to {candidate}")


def _load_trimesh_vertices_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    loaded = trimesh.load(str(path), force="mesh")
    if hasattr(loaded, "dump"):
        loaded = loaded.dump(concatenate=True)
    vertices = np.asarray(loaded.vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(loaded.faces, dtype=np.int32).reshape(-1, 3)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"empty mesh: {path}")
    return vertices, faces


def load_link_meshes(
    urdf_path: Path,
    *,
    include_links: list[str],
    exclude_substrings: list[str],
) -> list[LinkMesh]:
    urdf_path = Path(urdf_path).expanduser().resolve()
    root = ET.parse(urdf_path).getroot()
    include_all = len(include_links) == 1 and include_links[0].lower() == "all"
    include_set = set(include_links)
    excluded = [item.lower() for item in exclude_substrings]
    meshes: list[LinkMesh] = []
    for link in root.findall("link"):
        link_name = str(link.attrib.get("name", ""))
        if not link_name:
            continue
        if include_all:
            lowered = link_name.lower()
            if any(token and token in lowered for token in excluded):
                continue
        elif link_name not in include_set:
            continue
        vertices_parts = []
        faces_parts = []
        sources = []
        vertex_offset = 0
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            mesh_elem = None if geometry is None else geometry.find("mesh")
            if mesh_elem is None:
                continue
            mesh_path = _resolve_mesh_path(str(mesh_elem.attrib["filename"]), urdf_path)
            vertices, faces = _load_trimesh_vertices_faces(mesh_path)
            scale = np.asarray(_float_list(mesh_elem.attrib.get("scale"), 3, [1.0, 1.0, 1.0]), dtype=np.float64)
            vertices = vertices * scale.reshape(1, 3)
            visual_T_mesh = _origin_transform(visual.find("origin"))
            vertices_h = np.column_stack([vertices, np.ones((vertices.shape[0],), dtype=np.float64)])
            vertices = (visual_T_mesh @ vertices_h.T).T[:, :3]
            vertices_parts.append(vertices)
            faces_parts.append(faces + vertex_offset)
            vertex_offset += int(vertices.shape[0])
            sources.append(str(mesh_path))
        if vertices_parts:
            meshes.append(
                LinkMesh(
                    link_name=link_name,
                    vertices=np.concatenate(vertices_parts, axis=0),
                    faces=np.concatenate(faces_parts, axis=0),
                    source_files=sources,
                )
            )
    found = {mesh.link_name for mesh in meshes}
    missing = [name for name in include_links if name.lower() != "all" and name not in found]
    if missing:
        print(f"[wrist-visual] warning: no visual mesh found for link(s): {missing}")
    if not meshes:
        raise RuntimeError(f"No link meshes found in {urdf_path} for include_links={include_links}")
    return meshes


def _camera_config(args: argparse.Namespace, settings: dict[str, Any]) -> dict[str, Any]:
    camera_cfg = settings.get("wrist_camera", settings.get("camera", {}))
    return {
        "serial": args.camera_serial or camera_cfg.get("serial_number_or_name"),
        "width": int(args.camera_width if args.camera_width is not None else camera_cfg.get("width", 1280)),
        "height": int(args.camera_height if args.camera_height is not None else camera_cfg.get("height", 720)),
        "fps": int(args.camera_fps if args.camera_fps is not None else camera_cfg.get("fps", 30)),
        "exposure_us": args.camera_exposure_us if args.camera_exposure_us is not None else camera_cfg.get("exposure_us"),
        "gain": args.camera_gain if args.camera_gain is not None else camera_cfg.get("gain"),
        "auto_exposure": args.camera_auto_exposure if args.camera_auto_exposure is not None else camera_cfg.get("auto_exposure"),
    }


def _joint_names(args: argparse.Namespace, settings: dict[str, Any]) -> list[str]:
    return list(args.joint_names or settings.get("joint_names") or [f"joint_{idx}" for idx in range(1, 8)])


def _ee_link_name(args: argparse.Namespace, settings: dict[str, Any]) -> str | None:
    return args.ee_link_name or settings.get("ee_link_name")


def _urdf_path(args: argparse.Namespace, settings: dict[str, Any]) -> Path:
    return Path(args.urdf_path or settings.get("urdf_path") or DEFAULT_RM75_URDF).expanduser()


def _include_links(args: argparse.Namespace, settings: dict[str, Any]) -> list[str]:
    if args.include_links is not None:
        return list(args.include_links)
    return list(settings.get("wrist_visual_refine_links") or DEFAULT_BODY_LINKS)


def _latest_wrist_camera_board_run() -> Path | None:
    root = RUNTIME_DIR / "calibration_runs"
    if not root.exists():
        return None
    candidates = []
    for run_dir in root.glob("*_wrist_camera_board"):
        if not run_dir.is_dir():
            continue
        paths = [
            run_dir / "ee_T_wrist_camera_refined.npy",
            run_dir / "ee_T_wrist_camera_refined.json",
            run_dir / "ee_T_wrist_camera.npy",
            run_dir / "ee_T_wrist_camera.json",
        ]
        existing = [path for path in paths if path.exists()]
        if existing:
            candidates.append((max(path.stat().st_mtime for path in existing), run_dir))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _refined_copy_is_accepted(path: Path) -> bool:
    metadata_path = Path(path).expanduser().parent / "ee_T_wrist_camera_refined_metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = load_json(metadata_path)
    except Exception:
        return False
    return bool(metadata.get("accepted", False))


def _load_initial_ee_T_cam(args: argparse.Namespace, settings: dict[str, Any]) -> np.ndarray:
    candidates = []
    if args.initial_ee_T_camera is not None:
        candidates.append(Path(args.initial_ee_T_camera).expanduser())
    if args.wrist_camera_run_dir is not None:
        root = Path(args.wrist_camera_run_dir).expanduser()
        refined_candidates = [
            root / "ee_T_wrist_camera_refined.npy",
            root / "ee_T_wrist_camera_refined.json",
        ]
        candidates += [path for path in refined_candidates if path.exists() and _refined_copy_is_accepted(path)]
        candidates += [
            root / "ee_T_wrist_camera.npy",
            root / "ee_T_wrist_camera.json",
        ]
    for key in ("ee_T_wrist_camera_path", "wrist_camera_extrinsic_path"):
        if settings.get(key):
            candidates.append(Path(settings[key]).expanduser())
    for path in candidates:
        if path.exists():
            print(f"[wrist-visual] initial ee_T_wrist_camera: {path}")
            return load_matrix(path)
    raise FileNotFoundError(
        "Need initial wrist-camera calibration. Pass --wrist-camera-run-dir or --initial-ee-T-camera."
    )


def _scaled_intrinsic(intrinsic: np.ndarray, scale: float) -> np.ndarray:
    out = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3).copy()
    out[0, :] *= float(scale)
    out[1, :] *= float(scale)
    out[2, :] = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return out


def _project_points(points_cam: np.ndarray, intrinsic: np.ndarray, dist_coeffs: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    points_cam = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    valid = points_cam[:, 2] > 1e-4
    uv = np.zeros((points_cam.shape[0], 2), dtype=np.float64)
    if not np.any(valid):
        return uv, valid
    dist = np.zeros((5,), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    if np.any(np.abs(dist) > 1e-12):
        projected, _ = cv2.projectPoints(
            points_cam[valid],
            np.zeros((3,), dtype=np.float64),
            np.zeros((3,), dtype=np.float64),
            np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
            dist,
        )
        uv[valid] = projected.reshape(-1, 2)
    else:
        xyz = points_cam[valid]
        uv_valid = np.column_stack([xyz[:, 0] / xyz[:, 2], xyz[:, 1] / xyz[:, 2], np.ones((xyz.shape[0],))])
        uv[valid] = (np.asarray(intrinsic, dtype=np.float64).reshape(3, 3) @ uv_valid.T).T[:, :2]
    finite = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
    valid &= finite
    return uv, valid


def render_robot_mask(
    meshes: list[LinkMesh],
    link_poses: dict[str, np.ndarray],
    base_T_ee: np.ndarray,
    ee_T_cam: np.ndarray,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray | None,
    *,
    width: int,
    height: int,
    scale: float = 1.0,
    render_mode: str = "hull",
    near_clip_m: float = 0.035,
    max_triangle_edge_px: float | None = 420.0,
) -> np.ndarray:
    scale = float(scale)
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    k_scaled = _scaled_intrinsic(intrinsic, scale)
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    cam_T_base = invert_transform(as_transform(base_T_ee) @ as_transform(ee_T_cam))
    clip_min = np.asarray([-2 * out_w, -2 * out_h], dtype=np.int32)
    clip_max = np.asarray([3 * out_w, 3 * out_h], dtype=np.int32)
    for mesh in meshes:
        if mesh.link_name not in link_poses:
            continue
        cam_T_link = cam_T_base @ as_transform(link_poses[mesh.link_name])
        vertices_h = np.column_stack([mesh.vertices, np.ones((mesh.vertices.shape[0],), dtype=np.float64)])
        vertices_cam = (cam_T_link @ vertices_h.T).T[:, :3]
        vertices_cam = vertices_cam.copy()
        vertices_cam[vertices_cam[:, 2] <= float(near_clip_m), 2] = np.nan
        uv, valid = _project_points(vertices_cam, k_scaled, dist_coeffs)
        if not np.any(valid):
            continue
        if render_mode == "triangles":
            uv_i = np.clip(np.round(uv), clip_min, clip_max).astype(np.int32)
            face_valid = np.all(valid[mesh.faces], axis=1)
            triangles = uv_i[mesh.faces[face_valid]]
            if triangles.size:
                if max_triangle_edge_px is not None and float(max_triangle_edge_px) > 0.0:
                    threshold = max(16.0, float(max_triangle_edge_px) * scale)
                    edges = np.stack(
                        [
                            triangles[:, 0, :] - triangles[:, 1, :],
                            triangles[:, 1, :] - triangles[:, 2, :],
                            triangles[:, 2, :] - triangles[:, 0, :],
                        ],
                        axis=1,
                    ).astype(np.float64)
                    max_edge = np.max(np.linalg.norm(edges, axis=2), axis=1)
                    triangles = triangles[max_edge <= threshold]
                visible_bbox = (
                    (np.max(triangles[:, :, 0], axis=1) >= 0)
                    & (np.min(triangles[:, :, 0], axis=1) < out_w)
                    & (np.max(triangles[:, :, 1], axis=1) >= 0)
                    & (np.min(triangles[:, :, 1], axis=1) < out_h)
                )
                contours = [item.reshape(3, 1, 2) for item in triangles[visible_bbox]]
                if contours:
                    cv2.fillPoly(mask, contours, 255)
        else:
            points = np.clip(np.round(uv[valid]), clip_min, clip_max).astype(np.int32)
            if points.shape[0] < 3:
                continue
            hull = cv2.convexHull(points.reshape(-1, 1, 2))
            cv2.fillConvexPoly(mask, hull, 255)
    return mask


def _render_link_masks(
    meshes: list[LinkMesh],
    link_poses: dict[str, np.ndarray],
    base_T_ee: np.ndarray,
    ee_T_cam: np.ndarray,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray | None,
    *,
    width: int,
    height: int,
    scale: float,
    render_mode: str,
    near_clip_m: float,
    max_triangle_edge_px: float | None,
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for mesh in meshes:
        masks[mesh.link_name] = render_robot_mask(
            [mesh],
            link_poses,
            base_T_ee,
            ee_T_cam,
            intrinsic,
            dist_coeffs,
            width=width,
            height=height,
            scale=scale,
            render_mode=render_mode,
            near_clip_m=near_clip_m,
            max_triangle_edge_px=max_triangle_edge_px,
        )
    return masks


def _union_link_masks(link_masks: dict[str, np.ndarray], link_names: list[str]) -> np.ndarray:
    selected = [link_masks[name] for name in link_names if name in link_masks]
    if not selected:
        shape = next(iter(link_masks.values())).shape if link_masks else (1, 1)
        return np.zeros(shape, dtype=np.uint8)
    out = np.zeros_like(selected[0], dtype=np.uint8)
    for mask in selected:
        out[np.asarray(mask > 0, dtype=bool)] = 255
    return out


def _meshes_for_frame(meshes: list[LinkMesh], used_link_names: list[str]) -> list[LinkMesh]:
    used = set(used_link_names)
    return [mesh for mesh in meshes if mesh.link_name in used]


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask > 0, dtype=np.uint8)
    if not np.any(binary):
        return np.zeros_like(binary, dtype=bool)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        filled = np.zeros_like(binary)
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
        binary = filled
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    return (binary > 0) & (eroded == 0)


def _dilate_binary(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return np.asarray(mask > 0, dtype=bool)
    size = int(radius_px) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(np.asarray(mask > 0, dtype=np.uint8), kernel, iterations=1) > 0


def _prepare_edges(
    image_bgr: np.ndarray,
    initial_mask: np.ndarray,
    *,
    scale: float,
    canny_low: float,
    canny_high: float,
    roi_dilate_px: int,
    edge_dilate_px: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    width = int(round(image_bgr.shape[1] * scale))
    height = int(round(image_bgr.shape[0] * scale))
    scaled = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, float(canny_low), float(canny_high)) > 0
    initial_boundary = mask_boundary(initial_mask)
    roi = _dilate_binary(initial_boundary, max(1, int(round(roi_dilate_px * scale))))
    edges &= roi
    edge_radius = max(1, int(round(edge_dilate_px * scale))) if int(edge_dilate_px) > 0 else 0
    if edge_radius > 0:
        edges = _dilate_binary(edges, edge_radius)
    edge_pixels = int(np.count_nonzero(edges))
    dt = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)
    return edges, dt.astype(np.float32), edge_pixels


def _resize_bool(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(np.asarray(mask > 0, dtype=np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0


def _mask_pair_metrics(mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(mask_a > 0, dtype=bool)
    b = np.asarray(mask_b > 0, dtype=bool)
    intersection = int(np.count_nonzero(a & b))
    area_a = int(np.count_nonzero(a))
    area_b = int(np.count_nonzero(b))
    union = int(np.count_nonzero(a | b))
    return {
        "mask_iou": float(intersection / max(union, 1)),
        "mask_dice": float((2.0 * intersection) / max(area_a + area_b, 1)),
        "mask_intersection_px": intersection,
        "rendered_mask_area_px": area_a,
        "observed_mask_area_px": area_b,
        "mask_union_px": union,
    }


def _mask_bbox_xyxy(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(np.asarray(mask > 0, dtype=bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _expand_box_xyxy(box: np.ndarray, *, width: int, height: int, pad_px: float) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in np.asarray(box, dtype=np.float32).reshape(4)]
    pad = max(0.0, float(pad_px))
    return np.asarray(
        [
            min(max(x1 - pad, 0.0), float(width - 1)),
            min(max(y1 - pad, 0.0), float(height - 1)),
            min(max(x2 + pad, 1.0), float(width)),
            min(max(y2 + pad, 1.0), float(height)),
        ],
        dtype=np.float32,
    )


def _tail(text: str, *, lines: int = 18) -> str:
    parts = [line for line in str(text or "").splitlines() if line.strip()]
    return "\n".join(parts[-max(1, int(lines)) :])


def _run_sam3_mask_for_frame(
    args: argparse.Namespace,
    run_dir: Path,
    frame: CapturedFrame,
    initial_mask: np.ndarray,
    *,
    scale: float,
) -> np.ndarray | None:
    height, width = frame.image_bgr.shape[:2]
    target_mask_path = run_dir / "sam3_masks" / f"{frame.index:04d}.png"
    target_overlay_path = run_dir / "sam3_overlays" / f"{frame.index:04d}.png"
    if bool(args.sam3_reuse_masks) and target_mask_path.exists():
        mask = cv2.imread(str(target_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            print(f"[wrist-visual][sam3] frame {frame.index:04d}: reused {target_mask_path}")
            return _resize_bool(mask, max(1, int(round(width * scale))), max(1, int(round(height * scale))))

    bbox_scaled = _mask_bbox_xyxy(initial_mask)
    if bbox_scaled is None:
        print(f"[wrist-visual][sam3] frame {frame.index:04d}: projected robot mask is empty; falling back to Canny edges")
        return None
    bbox_full = np.asarray(bbox_scaled, dtype=np.float32) / max(float(scale), 1e-6)
    bbox_full = _expand_box_xyxy(bbox_full, width=width, height=height, pad_px=float(args.sam3_box_padding_px))

    rgb_path = run_dir / "images" / f"{frame.index:04d}.png"
    if not rgb_path.exists():
        image_write(rgb_path, frame.image_bgr)

    raw_dir = run_dir / "sam3_raw" / f"{frame.index:04d}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(Path(args.sam3_python).expanduser()),
        str(Path(args.sam3_provider_script).expanduser()),
        "--rgb-path",
        str(rgb_path),
        "--output-dir",
        str(raw_dir),
        "--prompt",
        str(args.sam3_prompt),
        "--mode",
        "text_box",
        "--box",
        f"{bbox_full[0]:.1f}",
        f"{bbox_full[1]:.1f}",
        f"{bbox_full[2]:.1f}",
        f"{bbox_full[3]:.1f}",
        "--select-box",
        f"{bbox_full[0]:.1f}",
        f"{bbox_full[1]:.1f}",
        f"{bbox_full[2]:.1f}",
        f"{bbox_full[3]:.1f}",
        "--checkpoint-path",
        str(Path(args.sam3_checkpoint_path).expanduser()),
        "--confidence-threshold",
        str(float(args.sam3_confidence_threshold)),
        "--resolution",
        str(int(args.sam3_resolution)),
        "--min-mask-area",
        str(int(args.sam3_min_mask_area)),
        "--morph-kernel",
        str(int(args.sam3_morph_kernel)),
        "--sam3-max-masks-per-item",
        str(int(args.sam3_max_masks_per_item)),
        "--device",
        str(args.sam3_device),
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(run_dir.parent),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(1.0, float(args.sam3_timeout_s)),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[wrist-visual][sam3] frame {frame.index:04d}: SAM3 failed ({exc!r}); falling back to Canny edges")
        return None
    if completed.returncode != 0:
        print(f"[wrist-visual][sam3] frame {frame.index:04d}: SAM3 returned {completed.returncode}; falling back to Canny edges")
        if completed.stderr:
            print(_tail(completed.stderr))
        if completed.stdout:
            print(_tail(completed.stdout))
        return None

    result_path = raw_dir / "sam3_result.json"
    result = load_json(result_path) if result_path.exists() else {}
    mask_path = Path(result.get("mask_path") or (raw_dir / "sam3_mask.png")).expanduser()
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[wrist-visual][sam3] frame {frame.index:04d}: missing mask {mask_path}; falling back to Canny edges")
        return None
    mask_pixels = int(np.count_nonzero(mask > 0))
    if mask_pixels < int(args.sam3_min_mask_area):
        print(f"[wrist-visual][sam3] frame {frame.index:04d}: only {mask_pixels} mask pixels; falling back to Canny edges")
        return None

    image_write(target_mask_path, mask)
    overlay_path = Path(result.get("overlay_path") or (raw_dir / "sam3_overlay.png")).expanduser()
    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    if overlay is not None:
        image_write(target_overlay_path, overlay)
    print(
        f"[wrist-visual][sam3] frame {frame.index:04d}: "
        f"mask_pixels={mask_pixels} box={bbox_full.round(1).tolist()}"
    )
    return _resize_bool(mask, max(1, int(round(width * scale))), max(1, int(round(height * scale))))


def _result_path(raw_dir: Path, value: Any, default_name: str) -> Path:
    path = Path(value or (raw_dir / default_name)).expanduser()
    if not path.is_absolute():
        path = raw_dir / path
    return path


def _rel_to_run(path: Path | None, run_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return Path(path).resolve().relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def _save_sam3_candidate_artifacts(run_dir: Path, frame_index: int, records: list[dict[str, Any]]) -> str | None:
    if not records:
        return None
    out_dir = run_dir / "sam3_candidates" / f"{frame_index:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        rank = int(record.get("candidate_rank") or len(records))
        raw_mask = record.get("raw_mask_path")
        if raw_mask:
            mask = cv2.imread(str(raw_mask), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                target = out_dir / f"candidate_{rank:02d}_mask.png"
                image_write(target, mask)
                record["saved_mask_path"] = _rel_to_run(target, run_dir)
        raw_overlay = record.get("raw_overlay_path")
        if raw_overlay:
            overlay = cv2.imread(str(raw_overlay), cv2.IMREAD_COLOR)
            if overlay is not None:
                target = out_dir / f"candidate_{rank:02d}_overlay.png"
                image_write(target, overlay)
                record["saved_overlay_path"] = _rel_to_run(target, run_dir)
    return _rel_to_run(out_dir, run_dir)


def _load_sam3_mask_from_result(
    args: argparse.Namespace,
    run_dir: Path,
    frame: CapturedFrame,
    raw_dir: Path,
    initial_mask: np.ndarray,
    *,
    scale: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    height, width = frame.image_bgr.shape[:2]
    target_mask_path = run_dir / "sam3_masks" / f"{frame.index:04d}.png"
    target_overlay_path = run_dir / "sam3_overlays" / f"{frame.index:04d}.png"
    result_path = raw_dir / "sam3_result.json"
    result = load_json(result_path) if result_path.exists() else {}
    report: dict[str, Any] = {
        "frame_index": int(frame.index),
        "source": "sam3",
        "status": "rejected",
        "used_mask": False,
        "candidate_count": 0,
        "selected_candidate_rank": None,
        "selected_candidate_score": None,
        "selected_model_score": None,
        "projected_overlap": None,
        "projected_coverage": None,
        "mask_area_px": None,
        "reason": "",
        "candidates": [],
    }
    if result and not bool(result.get("ok", True)):
        reason = str(result.get("error", "SAM3 failed"))
        report["reason"] = reason
        print(f"[wrist-visual][sam3] frame {frame.index:04d}: {reason}; falling back to Canny edges")
        return None, report
    if isinstance(result.get("results"), list):
        candidates = result.get("results")
    elif isinstance(result.get("all_candidates"), list):
        candidates = result.get("all_candidates")
    else:
        candidates = [result]
    if not candidates:
        candidates = [{"mask_path": str(raw_dir / "sam3_mask.png"), "overlay_path": str(raw_dir / "sam3_overlay.png")}]

    initial_bool = np.asarray(initial_mask > 0, dtype=bool)
    initial_area = max(1, int(np.count_nonzero(initial_bool)))
    best: tuple[float, dict[str, Any], dict[str, Any], np.ndarray] | None = None
    records: list[dict[str, Any]] = []
    for ordinal, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        if candidate and not bool(candidate.get("ok", True)):
            records.append(
                {
                    "frame_index": int(frame.index),
                    "candidate_rank": int(candidate.get("candidate_rank") or ordinal),
                    "accepted_candidate": False,
                    "rejection_reason": str(candidate.get("error", "candidate failed")),
                }
            )
            continue
        mask_path = _result_path(raw_dir, candidate.get("mask_path"), "sam3_mask.png")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            records.append(
                {
                    "frame_index": int(frame.index),
                    "candidate_rank": int(candidate.get("candidate_rank") or ordinal),
                    "accepted_candidate": False,
                    "raw_mask_path": str(mask_path),
                    "rejection_reason": "mask file missing",
                }
            )
            continue
        candidate_bool = _resize_bool(mask, initial_mask.shape[1], initial_mask.shape[0])
        candidate_area = max(1, int(np.count_nonzero(candidate_bool)))
        overlap = int(np.count_nonzero(candidate_bool & initial_bool))
        projected_overlap = float(overlap / candidate_area)
        projected_coverage = float(overlap / initial_area)
        dice = float((2.0 * overlap) / max(candidate_area + initial_area, 1))
        iou = float(overlap / max(candidate_area + initial_area - overlap, 1))
        score = dice + 0.5 * projected_overlap + 0.25 * projected_coverage
        overlay_path = _result_path(raw_dir, candidate.get("overlay_path"), "sam3_overlay.png")
        mask_pixels = int(np.count_nonzero(mask > 0))
        selected_meta = candidate.get("selected") if isinstance(candidate.get("selected"), dict) else {}
        rejection_parts = []
        if mask_pixels < int(args.sam3_min_mask_area):
            rejection_parts.append(f"mask area {mask_pixels} < {int(args.sam3_min_mask_area)}")
        if projected_overlap < float(args.sam3_min_projected_overlap):
            rejection_parts.append(
                f"projected overlap {projected_overlap:.3f} < {float(args.sam3_min_projected_overlap):.3f}"
            )
        record = {
            "frame_index": int(frame.index),
            "candidate_rank": int(candidate.get("candidate_rank") or ordinal),
            "selected_index": candidate.get("selected_index"),
            "model_score": float(selected_meta.get("model_score", 0.0)),
            "provider_rank": float(selected_meta.get("rank", 0.0)),
            "selection_score": float(score),
            "projected_overlap": float(projected_overlap),
            "projected_coverage": float(projected_coverage),
            "dice_to_initial_projection": float(dice),
            "iou_to_initial_projection": float(iou),
            "mask_area_px": int(mask_pixels),
            "mask_area_scaled_px": int(candidate_area),
            "raw_mask_path": str(mask_path),
            "raw_overlay_path": str(overlay_path),
            "accepted_candidate": not rejection_parts,
            "rejection_reason": "; ".join(rejection_parts),
        }
        records.append(record)
        if not rejection_parts and (best is None or score > best[0]):
            best = (score, candidate, record, mask)

    report["candidates"] = records
    report["candidate_count"] = len(records)
    if best is None:
        reason = "no SAM3 candidate met area/overlap gates" if records else f"missing candidate masks in {raw_dir}"
        report["reason"] = reason
        saved_dir = _save_sam3_candidate_artifacts(run_dir, frame.index, records)
        if saved_dir:
            report["saved_candidates_dir"] = saved_dir
        print(f"[wrist-visual][sam3] frame {frame.index:04d}: {reason}; falling back to Canny edges")
        return None, report
    score, result, selected_record, mask = best

    image_write(target_mask_path, mask)
    overlay_path = _result_path(raw_dir, result.get("overlay_path"), "sam3_overlay.png")
    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    if overlay is not None:
        image_write(target_overlay_path, overlay)
    if bool(args.save_sam3_candidates):
        saved_dir = _save_sam3_candidate_artifacts(run_dir, frame.index, records)
        if saved_dir:
            report["saved_candidates_dir"] = saved_dir
    report.update(
        {
            "status": "accepted",
            "used_mask": True,
            "selected_candidate_rank": selected_record.get("candidate_rank"),
            "selected_candidate_score": float(score),
            "selected_model_score": selected_record.get("model_score"),
            "projected_overlap": selected_record.get("projected_overlap"),
            "projected_coverage": selected_record.get("projected_coverage"),
            "mask_area_px": selected_record.get("mask_area_px"),
            "reason": "accepted",
            "mask_path": _rel_to_run(target_mask_path, run_dir),
            "overlay_path": _rel_to_run(target_overlay_path, run_dir),
        }
    )
    print(
        f"[wrist-visual][sam3] frame {frame.index:04d}: "
        f"mask_pixels={int(selected_record['mask_area_px'])} "
        f"projected_overlap={float(selected_record['projected_overlap']):.3f} score={score:.3f}"
    )
    return _resize_bool(mask, max(1, int(round(width * scale))), max(1, int(round(height * scale)))), report


def _run_sam3_masks_for_frames(
    args: argparse.Namespace,
    run_dir: Path,
    frames: list[CapturedFrame],
    initial_masks: list[np.ndarray],
    observed_masks: list[np.ndarray | None],
    mask_reports: dict[int, dict[str, Any]],
    *,
    scale: float,
) -> tuple[list[np.ndarray | None], dict[int, dict[str, Any]]]:
    out = list(observed_masks)
    batch_items: list[dict[str, Any]] = []
    pending: list[tuple[int, CapturedFrame, Path, np.ndarray]] = []

    for order, (frame, initial_mask, observed_mask) in enumerate(zip(frames, initial_masks, observed_masks)):
        if observed_mask is not None:
            continue
        height, width = frame.image_bgr.shape[:2]
        target_mask_path = run_dir / "sam3_masks" / f"{frame.index:04d}.png"
        if bool(args.sam3_reuse_masks) and target_mask_path.exists():
            mask = cv2.imread(str(target_mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                print(f"[wrist-visual][sam3] frame {frame.index:04d}: reused {target_mask_path}")
                out[order] = _resize_bool(mask, max(1, int(round(width * scale))), max(1, int(round(height * scale))))
                mask_reports[frame.index] = {
                    "frame_index": int(frame.index),
                    "source": "sam3_reused",
                    "status": "accepted",
                    "used_mask": True,
                    "candidate_count": None,
                    "selected_candidate_rank": None,
                    "selected_candidate_score": None,
                    "selected_model_score": None,
                    "projected_overlap": None,
                    "projected_coverage": None,
                    "mask_area_px": int(np.count_nonzero(mask > 0)),
                    "reason": "reused existing SAM3 mask",
                    "mask_path": _rel_to_run(target_mask_path, run_dir),
                }
                continue

        bbox_scaled = _mask_bbox_xyxy(initial_mask)
        if bbox_scaled is None:
            print(f"[wrist-visual][sam3] frame {frame.index:04d}: projected robot mask is empty; falling back to Canny edges")
            mask_reports[frame.index] = {
                "frame_index": int(frame.index),
                "source": "sam3",
                "status": "rejected",
                "used_mask": False,
                "candidate_count": 0,
                "selected_candidate_rank": None,
                "selected_candidate_score": None,
                "selected_model_score": None,
                "projected_overlap": None,
                "projected_coverage": None,
                "mask_area_px": None,
                "reason": "projected robot mask is empty",
            }
            continue
        bbox_full = np.asarray(bbox_scaled, dtype=np.float32) / max(float(scale), 1e-6)
        bbox_full = _expand_box_xyxy(bbox_full, width=width, height=height, pad_px=float(args.sam3_box_padding_px))

        rgb_path = run_dir / "images" / f"{frame.index:04d}.png"
        if not rgb_path.exists():
            image_write(rgb_path, frame.image_bgr)
        raw_dir = run_dir / "sam3_raw" / f"{frame.index:04d}"
        pending.append((order, frame, raw_dir, initial_mask))
        batch_items.append(
            {
                "id": f"{frame.index:04d}",
                "rgb_path": str(rgb_path.resolve()),
                "output_dir": str(raw_dir.resolve()),
                "prompt": str(args.sam3_prompt),
                "mode": "text_box",
                "box": [float(v) for v in bbox_full],
                "select_box": [float(v) for v in bbox_full],
            }
        )

    if not batch_items:
        return out, mask_reports

    batch_json = run_dir / "sam3_batch_frames.json"
    save_json(batch_json, batch_items)
    cmd = [
        str(Path(args.sam3_python).expanduser()),
        str(Path(args.sam3_provider_script).expanduser()),
        "--batch-frames-json",
        str(batch_json),
        "--output-dir",
        str((run_dir / "sam3_raw").resolve()),
        "--checkpoint-path",
        str(Path(args.sam3_checkpoint_path).expanduser()),
        "--confidence-threshold",
        str(float(args.sam3_confidence_threshold)),
        "--resolution",
        str(int(args.sam3_resolution)),
        "--min-mask-area",
        str(int(args.sam3_min_mask_area)),
        "--morph-kernel",
        str(int(args.sam3_morph_kernel)),
        "--sam3-max-masks-per-item",
        str(int(args.sam3_max_masks_per_item)),
        "--device",
        str(args.sam3_device),
    ]
    print(f"[wrist-visual][sam3] resident batch: {len(batch_items)} frames")
    timeout_s = max(float(args.sam3_timeout_s), 20.0 + 15.0 * len(batch_items))
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(run_dir.parent),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[wrist-visual][sam3] resident batch failed ({exc!r}); falling back to Canny edges")
        for _, frame, _, _ in pending:
            mask_reports[frame.index] = {
                "frame_index": int(frame.index),
                "source": "sam3",
                "status": "rejected",
                "used_mask": False,
                "candidate_count": 0,
                "reason": f"resident batch failed: {exc!r}",
            }
        return out, mask_reports
    if completed.stdout:
        print(_tail(completed.stdout, lines=24))
    if completed.returncode != 0:
        print(f"[wrist-visual][sam3] resident batch returned {completed.returncode}; falling back to Canny edges")
        if completed.stderr:
            print(_tail(completed.stderr))
        for _, frame, _, _ in pending:
            mask_reports[frame.index] = {
                "frame_index": int(frame.index),
                "source": "sam3",
                "status": "rejected",
                "used_mask": False,
                "candidate_count": 0,
                "reason": f"resident batch returned {completed.returncode}",
            }
        return out, mask_reports
    if completed.stderr:
        tail = _tail(completed.stderr, lines=6)
        if tail:
            print(tail)

    for order, frame, raw_dir, initial_mask in pending:
        out[order], report = _load_sam3_mask_from_result(args, run_dir, frame, raw_dir, initial_mask, scale=scale)
        mask_reports[frame.index] = report
    return out, mask_reports


def _load_observed_masks(
    args: argparse.Namespace, frames: list[CapturedFrame], *, scale: float
) -> tuple[list[np.ndarray | None], dict[int, dict[str, Any]]]:
    if args.mask_npy is None and args.mask_dir is None:
        return [None for _ in frames], {}
    masks_raw = None
    if args.mask_npy is not None:
        masks_raw = np.load(Path(args.mask_npy).expanduser())
        if masks_raw.shape[0] < len(frames):
            raise ValueError(f"--mask-npy has {masks_raw.shape[0]} masks but {len(frames)} frames were captured")
    out: list[np.ndarray | None] = []
    reports: dict[int, dict[str, Any]] = {}
    for order, frame in enumerate(frames):
        if masks_raw is not None:
            mask = np.asarray(masks_raw[order])
            source = "mask_npy"
            path_text = str(Path(args.mask_npy).expanduser())
        else:
            path = Path(args.mask_dir).expanduser() / f"{frame.index:04d}.png"
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                out.append(None)
                reports[frame.index] = {
                    "frame_index": int(frame.index),
                    "source": "mask_dir",
                    "status": "missing",
                    "used_mask": False,
                    "candidate_count": None,
                    "selected_candidate_rank": None,
                    "selected_candidate_score": None,
                    "selected_model_score": None,
                    "projected_overlap": None,
                    "projected_coverage": None,
                    "mask_area_px": None,
                    "reason": f"manual mask not found: {path}",
                    "mask_path": str(path),
                }
                continue
            source = "mask_dir"
            path_text = str(path)
        height = max(1, int(round(frame.image_bgr.shape[0] * scale)))
        width = max(1, int(round(frame.image_bgr.shape[1] * scale)))
        resized = _resize_bool(mask, width, height)
        out.append(resized)
        reports[frame.index] = {
            "frame_index": int(frame.index),
            "source": source,
            "status": "accepted",
            "used_mask": True,
            "candidate_count": None,
            "selected_candidate_rank": None,
            "selected_candidate_score": None,
            "selected_model_score": None,
            "projected_overlap": None,
            "projected_coverage": None,
            "mask_area_px": int(np.count_nonzero(mask > 0)),
            "reason": "manual override mask" if source == "mask_dir" else "mask loaded from npy array",
            "mask_path": path_text,
        }
    return out, reports


def draw_alignment_overlay(
    image_bgr: np.ndarray,
    observed_edges: np.ndarray,
    initial_mask: np.ndarray | None,
    refined_mask: np.ndarray | None,
    *,
    status_text: str | None = None,
) -> np.ndarray:
    out = image_bgr.copy()
    height, width = out.shape[:2]
    obs = _resize_bool(observed_edges, width, height)
    out[obs] = (0, 220, 255)
    if initial_mask is not None:
        init_edge = _resize_bool(mask_boundary(initial_mask), width, height)
        out[init_edge] = (0, 0, 255)
    if refined_mask is not None:
        final_edge = _resize_bool(mask_boundary(refined_mask), width, height)
        out[final_edge] = (0, 255, 0)
    cv2.putText(out, "observed edge=yellow  initial=red  refined=green", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA)
    if status_text:
        cv2.putText(out, str(status_text)[:120], (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)
    return out


def _collect_manual(
    args: argparse.Namespace,
    settings: dict[str, Any],
    run_dir: Path,
    urdf: Any,
) -> tuple[list[CapturedFrame], np.ndarray, np.ndarray]:
    joint_names = _joint_names(args, settings)
    robot_ip = args.robot_ip or settings.get("robot_ip")
    if not robot_ip:
        raise ValueError("robot_ip is required in --config or --robot-ip")
    camera_cfg = _camera_config(args, settings)
    controller = RealmanSdkController(
        robot_ip=str(robot_ip),
        robot_port=int(args.robot_port or settings.get("robot_port", 8080)),
        joint_names=joint_names,
        calibration_offset=settings.get("calibration_offset", {}),
    )
    camera = RealSenseCamera(
        serial=camera_cfg["serial"],
        width=int(camera_cfg["width"]),
        height=int(camera_cfg["height"]),
        fps=int(camera_cfg["fps"]),
        enable_depth=False,
        warmup_frames=int(args.warmup_frames),
        auto_exposure=None if camera_cfg["auto_exposure"] is None else bool(camera_cfg["auto_exposure"]),
        exposure_us=None if camera_cfg["exposure_us"] is None else float(camera_cfg["exposure_us"]),
        gain=None if camera_cfg["gain"] is None else float(camera_cfg["gain"]),
    )
    frames: list[CapturedFrame] = []
    target_count = max(1, int(args.manual_count))
    window_name = "rm75 wrist visual refine capture"
    try:
        camera.start()
        print("[wrist-visual] manual capture mode")
        print("[wrist-visual] Keep the arm body visible in the wrist camera. Enter/Space=capture, q/Esc=finish.")
        if not args.manual_terminal:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        idx = 0
        last_intrinsic = None
        last_dist = None
        while idx < target_count:
            frame = camera.capture()
            last_intrinsic = frame.intrinsic
            last_dist = frame.dist_coeffs
            if args.manual_terminal:
                user_input = input(f"[wrist-visual] sample {idx:04d}/{target_count}: Enter=capture, q=finish > ").strip().lower()
                command = "finish" if user_input in {"q", "quit", "done", "exit"} else "capture"
            else:
                cv2.imshow(window_name, _draw_capture_status(frame.color_bgr, sample_index=idx, target_count=target_count))
                key_command = _manual_command_from_key(cv2.waitKey(max(1, int(args.preview_delay_ms))))
                terminal_command = _manual_command_from_stdin()
                command = key_command or terminal_command
            if command == "finish":
                print("[wrist-visual] capture requested finish")
                break
            if command != "capture":
                continue
            qpos_now = np.asarray(controller.get_qpos(flat=True), dtype=np.float64)
            link_poses = link_poses_from_qpos(urdf, qpos_now, joint_names)
            _, base_T_ee = select_link_pose(link_poses, _ee_link_name(args, settings))
            item = CapturedFrame(
                index=idx,
                image_bgr=frame.color_bgr.copy(),
                qpos_rad=qpos_now,
                base_T_ee=base_T_ee,
                link_poses=link_poses,
            )
            frames.append(item)
            image_write(run_dir / "images" / f"{idx:04d}.png", frame.color_bgr)
            print(f"[wrist-visual] sample {idx:04d}: captured")
            idx += 1
            if not args.manual_terminal:
                cv2.imshow(window_name, _draw_capture_status(frame.color_bgr, sample_index=idx, target_count=target_count, captured=True))
                cv2.waitKey(180)
    finally:
        if not args.manual_terminal:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
        camera.stop()
        controller.disconnect()
    if not frames or last_intrinsic is None or last_dist is None:
        raise RuntimeError("No visual refinement frames were captured.")
    np.save(run_dir / "camera_intrinsic.npy", last_intrinsic)
    np.save(run_dir / "camera_dist_coeffs.npy", last_dist)
    _save_capture_manifest(run_dir, frames)
    return frames, last_intrinsic, last_dist


def _save_capture_manifest(run_dir: Path, frames: list[CapturedFrame]) -> None:
    save_json(
        run_dir / "visual_refine_observations.json",
        {
            "frames": [
                {
                    "index": frame.index,
                    "image": f"images/{frame.index:04d}.png",
                    "qpos_rad": frame.qpos_rad,
                    "base_T_ee": frame.base_T_ee,
                }
                for frame in frames
            ]
        },
    )
    np.save(run_dir / "qpos_rad.npy", np.stack([frame.qpos_rad for frame in frames], axis=0))
    np.save(run_dir / "base_T_ee.npy", np.stack([frame.base_T_ee for frame in frames], axis=0))


def _frames_from_run(
    source_run: Path,
    run_dir: Path,
    *,
    urdf: Any,
    joint_names: list[str],
    ee_link_name: str | None,
) -> tuple[list[CapturedFrame], np.ndarray, np.ndarray]:
    source_run = Path(source_run).expanduser().resolve()
    if not source_run.exists():
        raise FileNotFoundError(f"input run not found: {source_run}")
    obs_path = source_run / "visual_refine_observations.json"
    if not obs_path.exists():
        obs_path = source_run / "observations.json"
    if not obs_path.exists():
        raise FileNotFoundError(f"no observations json found in {source_run}")
    data = load_json(obs_path)
    frames_raw = data.get("frames", [])
    if not frames_raw:
        raise ValueError(f"no frames in {obs_path}")
    intrinsic = np.load(source_run / "camera_intrinsic.npy")
    dist_path = source_run / "camera_dist_coeffs.npy"
    dist_coeffs = np.load(dist_path) if dist_path.exists() else np.zeros((5,), dtype=np.float64)
    frames: list[CapturedFrame] = []
    for order, item in enumerate(frames_raw):
        idx = int(item.get("index", order))
        rel_image = item.get("image", f"images/{idx:04d}.png")
        image_path = source_run / str(rel_image)
        if not image_path.exists():
            image_path = source_run / "images" / f"{idx:04d}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"failed to read image: {image_path}")
        qpos = np.asarray(item.get("qpos_rad"), dtype=np.float64).reshape(-1)
        link_poses = link_poses_from_qpos(urdf, qpos, joint_names)
        _, base_T_ee = select_link_pose(link_poses, ee_link_name)
        frames.append(
            CapturedFrame(
                index=len(frames),
                image_bgr=image,
                qpos_rad=qpos,
                base_T_ee=base_T_ee,
                link_poses=link_poses,
            )
        )
        image_write(run_dir / "images" / f"{len(frames) - 1:04d}.png", image)
    np.save(run_dir / "camera_intrinsic.npy", intrinsic)
    np.save(run_dir / "camera_dist_coeffs.npy", dist_coeffs)
    _save_capture_manifest(run_dir, frames)
    return frames, intrinsic, dist_coeffs


def _prepare_frames(
    frames: list[CapturedFrame],
    meshes: list[LinkMesh],
    ee_T_cam_initial: np.ndarray,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[list[PreparedFrame], list[dict[str, Any]], list[dict[str, Any]]]:
    scale = float(args.optimize_scale)
    prepared: list[PreparedFrame] = []
    observed_masks, mask_reports_by_index = _load_observed_masks(args, frames, scale=scale)
    initial_masks: list[np.ndarray] = []
    link_reports_by_index: dict[int, dict[str, Any]] = {}
    visible_links_by_index: dict[int, list[str]] = {}
    link_areas_by_index: dict[int, dict[str, int]] = {}
    for frame in frames:
        height, width = frame.image_bgr.shape[:2]
        link_masks = _render_link_masks(
            meshes,
            frame.link_poses,
            frame.base_T_ee,
            ee_T_cam_initial,
            intrinsic,
            dist_coeffs,
            width=width,
            height=height,
            scale=scale,
            render_mode=str(args.render_mode),
            near_clip_m=float(args.near_clip_m),
            max_triangle_edge_px=float(args.max_triangle_edge_px),
        )
        link_areas = {name: int(np.count_nonzero(mask > 0)) for name, mask in link_masks.items()}
        min_area = max(0.0, float(args.min_link_projected_area_px))
        used_links = [mesh.link_name for mesh in meshes if link_areas.get(mesh.link_name, 0) >= min_area]
        visible_links_by_index[frame.index] = used_links
        link_areas_by_index[frame.index] = link_areas
        initial_masks.append(_union_link_masks(link_masks, used_links))
        link_reports_by_index[frame.index] = {
            "frame_index": int(frame.index),
            "used_in_refine": False,
            "used_links": " ".join(used_links),
            "used_link_count": len(used_links),
            "min_link_projected_area_px": float(min_area),
            "link_projected_area_px": link_areas,
            "reason": "",
        }
    if bool(args.sam3_mask):
        observed_masks, mask_reports_by_index = _run_sam3_masks_for_frames(
            args,
            run_dir,
            frames,
            initial_masks,
            observed_masks,
            mask_reports_by_index,
            scale=scale,
        )

    for frame, observed_mask, initial_mask in zip(frames, observed_masks, initial_masks):
        used_links = visible_links_by_index.get(frame.index, [])
        link_areas = link_areas_by_index.get(frame.index, {})
        if not used_links:
            reason = "no link projected area met --min-link-projected-area-px"
            print(f"[wrist-visual] skip frame {frame.index:04d}: {reason}; areas={link_areas}")
            link_reports_by_index[frame.index]["reason"] = reason
            continue
        mask_selection = mask_reports_by_index.get(frame.index)
        if mask_selection is None:
            mask_selection = {
                "frame_index": int(frame.index),
                "source": "canny_edges",
                "status": "fallback",
                "used_mask": False,
                "candidate_count": None,
                "selected_candidate_rank": None,
                "selected_candidate_score": None,
                "selected_model_score": None,
                "projected_overlap": None,
                "projected_coverage": None,
                "mask_area_px": None,
                "reason": "no mask target; using Canny edges in projected robot ROI",
            }
            mask_reports_by_index[frame.index] = mask_selection
        elif observed_mask is None and not bool(mask_selection.get("used_mask", False)):
            mask_selection = dict(mask_selection)
            old_status = str(mask_selection.get("status"))
            if old_status in {"missing", "rejected"}:
                mask_selection["status"] = "rejected_fallback" if old_status == "rejected" else "fallback"
                mask_selection["reason"] = str(mask_selection.get("reason", "")) + "; using Canny edges"
            mask_reports_by_index[frame.index] = mask_selection
        if observed_mask is not None:
            observed_edges = mask_boundary(observed_mask)
            edge_radius = max(1, int(round(int(args.edge_dilate_px) * scale))) if int(args.edge_dilate_px) > 0 else 0
            if edge_radius > 0:
                observed_edges = _dilate_binary(observed_edges, edge_radius)
            observed_dt = cv2.distanceTransform((~observed_edges).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
            edge_pixels = int(np.count_nonzero(observed_edges))
        else:
            observed_edges, observed_dt, edge_pixels = _prepare_edges(
                frame.image_bgr,
                initial_mask,
                scale=scale,
                canny_low=float(args.canny_low),
                canny_high=float(args.canny_high),
                roi_dilate_px=int(args.roi_dilate_px),
                edge_dilate_px=int(args.edge_dilate_px),
            )
        if edge_pixels < int(args.min_observed_edge_pixels):
            print(f"[wrist-visual] skip frame {frame.index:04d}: only {edge_pixels} observed edge pixels in projected ROI")
            link_reports_by_index[frame.index]["reason"] = f"only {edge_pixels} observed edge pixels in projected ROI"
            continue
        link_reports_by_index[frame.index]["used_in_refine"] = True
        link_reports_by_index[frame.index]["observed_edge_pixels"] = int(edge_pixels)
        link_reports_by_index[frame.index]["mask_source"] = str(mask_selection.get("source", ""))
        link_reports_by_index[frame.index]["mask_status"] = str(mask_selection.get("status", ""))
        prepared_frame = PreparedFrame(
            index=frame.index,
            image_bgr=frame.image_bgr,
            base_T_ee=frame.base_T_ee,
            link_poses=frame.link_poses,
            used_link_names=used_links,
            link_projected_area_px=link_areas,
            observed_edges=observed_edges,
            observed_dt=observed_dt,
            observed_mask=observed_mask,
            mask_source=str(mask_selection.get("source", "")),
            mask_selection=mask_selection,
            initial_mask=initial_mask,
            edge_pixels=edge_pixels,
        )
        prepared.append(prepared_frame)
        image_write(
            run_dir / "initial_overlays" / f"{frame.index:04d}.png",
            draw_alignment_overlay(
                frame.image_bgr,
                observed_edges,
                initial_mask,
                None,
                status_text=f"frame {frame.index:04d} initial links={','.join(used_links)} mask={mask_selection.get('source')}",
            ),
        )
    if not prepared:
        raise RuntimeError("No usable frames for visual refinement. Make sure the arm body is visible in the wrist camera.")
    frame_link_report = [link_reports_by_index[key] for key in sorted(link_reports_by_index)]
    mask_quality_report = [mask_reports_by_index[key] for key in sorted(mask_reports_by_index)]
    save_json(run_dir / "frame_link_visibility.json", frame_link_report)
    save_json(run_dir / "mask_quality_report.json", mask_quality_report)
    return prepared, frame_link_report, mask_quality_report


def _delta_to_transform(delta: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    delta = np.asarray(delta, dtype=np.float64).reshape(6)
    rotation = Rotation.from_rotvec(delta[:3]).as_matrix()
    return make_transform(rotation, delta[3:])


def _mesh_global_faces(meshes: list[LinkMesh]) -> np.ndarray:
    faces = []
    offset = 0
    for mesh in meshes:
        faces.append(np.asarray(mesh.faces, dtype=np.int32) + offset)
        offset += int(mesh.vertices.shape[0])
    if not faces:
        raise ValueError("no mesh faces")
    return np.concatenate(faces, axis=0)


def _frame_vertices_in_ee(frame: PreparedFrame, meshes: list[LinkMesh]) -> np.ndarray:
    ee_T_base = invert_transform(frame.base_T_ee)
    parts = []
    for mesh in meshes:
        if mesh.link_name not in frame.link_poses:
            continue
        base_T_link = as_transform(frame.link_poses[mesh.link_name])
        ee_T_link = ee_T_base @ base_T_link
        vertices_h = np.column_stack([mesh.vertices, np.ones((mesh.vertices.shape[0],), dtype=np.float64)])
        parts.append((ee_T_link @ vertices_h.T).T[:, :3])
    if not parts:
        raise ValueError(f"frame {frame.index} has no visible mesh vertices")
    return np.concatenate(parts, axis=0)


def _torch_skew(vector):
    import torch

    x, y, z = vector[0], vector[1], vector[2]
    zero = torch.zeros((), dtype=vector.dtype, device=vector.device)
    return torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ],
        dim=0,
    )


def _torch_rotvec_to_matrix(rotvec):
    import torch

    theta = torch.linalg.norm(rotvec)
    k = rotvec / torch.clamp(theta, min=1e-9)
    skew = _torch_skew(k)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    small = theta < 1e-7
    rodrigues = eye + sin_t * skew + (1.0 - cos_t) * (skew @ skew)
    first_order = eye + _torch_skew(rotvec)
    return torch.where(small, first_order, rodrigues)


def _torch_transform_inverse(rotation, translation):
    import torch

    rot_t = rotation.transpose(0, 1)
    trans = -(rot_t @ translation.reshape(3, 1)).reshape(3)
    return rot_t, trans


def _torch_project_clip(vertices_cam, intrinsic, *, width: int, height: int):
    import torch

    z = vertices_cam[:, 2].clamp(min=1e-6)
    u = intrinsic[0, 0] * vertices_cam[:, 0] / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * vertices_cam[:, 1] / z + intrinsic[1, 2]
    x_ndc = (2.0 * u / float(width)) - 1.0
    y_ndc = 1.0 - (2.0 * v / float(height))
    z_ndc = torch.clamp(vertices_cam[:, 2] / 2.0, -1.0, 1.0)
    return torch.stack([x_ndc, y_ndc, z_ndc, torch.ones_like(x_ndc)], dim=1)


def _torch_sobel_edges(mask):
    import torch
    import torch.nn.functional as functional

    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=mask.dtype,
        device=mask.device,
    ).reshape(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=mask.dtype,
        device=mask.device,
    ).reshape(1, 1, 3, 3)
    nchw = mask.permute(0, 3, 1, 2)
    gx = functional.conv2d(nchw, kx, padding=1)
    gy = functional.conv2d(nchw, ky, padding=1)
    edge = torch.sqrt(gx * gx + gy * gy + 1e-8)
    return edge.permute(0, 2, 3, 1)


def _render_soft_mask_nvdiffrast(
    glctx,
    vertices_ee,
    faces,
    ee_T_cam_initial,
    delta_vec,
    intrinsic,
    *,
    width: int,
    height: int,
    near_clip_m: float,
    max_triangle_edge_px: float,
):
    import nvdiffrast.torch as dr
    import torch

    delta_rot = _torch_rotvec_to_matrix(delta_vec[:3])
    delta_t = delta_vec[3:]
    initial_rot = ee_T_cam_initial[:3, :3]
    initial_t = ee_T_cam_initial[:3, 3]
    ee_R_cam = initial_rot @ delta_rot
    ee_t_cam = initial_rot @ delta_t + initial_t
    cam_R_ee, cam_t_ee = _torch_transform_inverse(ee_R_cam, ee_t_cam)
    vertices_cam = (cam_R_ee @ vertices_ee.T).T + cam_t_ee.reshape(1, 3)
    pos_clip = _torch_project_clip(vertices_cam, intrinsic, width=width, height=height)

    valid_vertices = vertices_cam[:, 2] > float(near_clip_m)
    face_valid = torch.all(valid_vertices[faces], dim=1)
    if float(max_triangle_edge_px) > 0.0:
        projected = pos_clip[:, :2]
        triangles = projected[faces]
        screen = torch.empty_like(triangles)
        screen[:, :, 0] = (triangles[:, :, 0] + 1.0) * 0.5 * float(width)
        screen[:, :, 1] = (1.0 - triangles[:, :, 1]) * 0.5 * float(height)
        edges = torch.stack(
            [
                screen[:, 0, :] - screen[:, 1, :],
                screen[:, 1, :] - screen[:, 2, :],
                screen[:, 2, :] - screen[:, 0, :],
            ],
            dim=1,
        )
        max_edge = torch.max(torch.linalg.norm(edges, dim=2), dim=1).values
        face_valid = face_valid & (max_edge <= float(max_triangle_edge_px))
    valid_faces = faces[face_valid]
    if valid_faces.numel() == 0:
        return torch.zeros((1, height, width, 1), dtype=vertices_ee.dtype, device=vertices_ee.device)
    rast, _ = dr.rasterize(glctx, pos_clip[None, :, :], valid_faces.contiguous(), resolution=[height, width])
    hard_mask = (rast[..., 3:4] > 0).to(vertices_ee.dtype)
    return dr.antialias(hard_mask, rast, pos_clip[None, :, :], valid_faces.contiguous())


def _fixed_sampled_distances(dt: np.ndarray, boundary: np.ndarray, *, sample_count: int, clip_px: float) -> np.ndarray:
    ys, xs = np.nonzero(boundary)
    if len(xs) == 0:
        return np.ones((sample_count,), dtype=np.float64)
    distances = dt[ys, xs].astype(np.float64)
    if len(distances) >= sample_count:
        order = np.linspace(0, len(distances) - 1, sample_count).astype(np.int64)
        distances = distances[order]
    else:
        pad = np.full((sample_count - len(distances),), float(clip_px), dtype=np.float64)
        distances = np.concatenate([distances, pad], axis=0)
    return np.clip(distances, 0.0, float(clip_px)) / max(float(clip_px), 1e-6)


def _evaluate_alignment(
    prepared: list[PreparedFrame],
    meshes: list[LinkMesh],
    ee_T_cam: np.ndarray,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    per_frame = []
    all_values = []
    mask_ious = []
    mask_dices = []
    scale = float(args.optimize_scale)
    clip_px = float(args.loss_clip_px)
    for frame in prepared:
        height, width = frame.image_bgr.shape[:2]
        frame_meshes = _meshes_for_frame(meshes, frame.used_link_names)
        mask = render_robot_mask(
            frame_meshes,
            frame.link_poses,
            frame.base_T_ee,
            ee_T_cam,
            intrinsic,
            dist_coeffs,
            width=width,
            height=height,
            scale=scale,
            render_mode=str(args.render_mode),
            near_clip_m=float(args.near_clip_m),
            max_triangle_edge_px=float(args.max_triangle_edge_px),
        )
        boundary = mask_boundary(mask)
        values = frame.observed_dt[boundary].astype(np.float64) if np.any(boundary) else np.asarray([], dtype=np.float64)
        frame_metrics: dict[str, Any] = {
            "index": frame.index,
            "used_links": list(frame.used_link_names),
            "observed_edge_pixels": int(frame.edge_pixels),
            "mask_source": frame.mask_source,
        }
        if values.size:
            clipped = np.clip(values, 0.0, clip_px)
            all_values.append(clipped)
            frame_metrics.update(
                {
                    "edge_distance_mean_px": float(np.mean(clipped)),
                    "edge_distance_median_px": float(np.median(clipped)),
                    "edge_distance_p90_px": float(np.percentile(clipped, 90)),
                    "rendered_edge_pixels": int(np.count_nonzero(boundary)),
                }
            )
        else:
            frame_metrics.update(
                {
                    "edge_distance_mean_px": float(clip_px),
                    "edge_distance_median_px": float(clip_px),
                    "edge_distance_p90_px": float(clip_px),
                    "rendered_edge_pixels": 0,
                }
            )
        if frame.observed_mask is not None:
            mask_metrics = _mask_pair_metrics(mask, frame.observed_mask)
            frame_metrics.update(mask_metrics)
            mask_ious.append(float(mask_metrics["mask_iou"]))
            mask_dices.append(float(mask_metrics["mask_dice"]))
        per_frame.append(frame_metrics)
    merged = np.concatenate(all_values, axis=0) if all_values else np.asarray([clip_px], dtype=np.float64)
    return {
        "frames": len(prepared),
        "edge_distance_mean_px": float(np.mean(merged)),
        "edge_distance_median_px": float(np.median(merged)),
        "edge_distance_p90_px": float(np.percentile(merged, 90)),
        "edge_distance_max_clipped_px": float(np.max(merged)),
        "mask_target_frames": int(len(mask_dices)),
        "mask_iou_mean": None if not mask_ious else float(np.mean(mask_ious)),
        "mask_dice_mean": None if not mask_dices else float(np.mean(mask_dices)),
        "mask_iou_min": None if not mask_ious else float(np.min(mask_ious)),
        "mask_dice_min": None if not mask_dices else float(np.min(mask_dices)),
        "per_frame": per_frame,
    }


def _optimize_delta(
    prepared: list[PreparedFrame],
    meshes: list[LinkMesh],
    ee_T_cam_initial: np.ndarray,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if str(args.optimizer) == "nvdiffrast":
        return _optimize_delta_nvdiffrast(prepared, meshes, ee_T_cam_initial, intrinsic, dist_coeffs, args)
    return _optimize_delta_powell(prepared, meshes, ee_T_cam_initial, intrinsic, dist_coeffs, args)


def _candidate_rejection_reason(
    initial_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    if bool(args.allow_worse_refine):
        return ""
    mask_frames = int(candidate_metrics.get("mask_target_frames") or 0)
    if mask_frames > 0:
        initial_dice = float(initial_metrics.get("mask_dice_mean") or 0.0)
        candidate_dice = float(candidate_metrics.get("mask_dice_mean") or 0.0)
        dice_delta = candidate_dice - initial_dice
        required = float(args.min_mask_dice_improvement)
        if dice_delta < required:
            return (
                f"candidate mask Dice delta {dice_delta:.6f} is below required {required:.6f}; "
                "keeping initial board calibration"
            )
        return ""
    if float(candidate_metrics["edge_distance_mean_px"]) >= float(initial_metrics["edge_distance_mean_px"]):
        return "candidate edge metric is not better than initial; keeping initial board calibration"
    return ""


def _optimize_delta_nvdiffrast(
    prepared: list[PreparedFrame],
    meshes: list[LinkMesh],
    ee_T_cam_initial: np.ndarray,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch
    import nvdiffrast.torch as dr

    if not torch.cuda.is_available():
        raise RuntimeError("nvdiffrast optimizer requires CUDA. Use --optimizer powell as a CPU fallback.")

    device = torch.device("cuda")
    dtype = torch.float32
    ee_T_cam_initial_t = torch.as_tensor(as_transform(ee_T_cam_initial), dtype=dtype, device=device)
    max_delta = torch.tensor(
        [
            np.deg2rad(float(args.max_rotation_delta_deg)),
            np.deg2rad(float(args.max_rotation_delta_deg)),
            np.deg2rad(float(args.max_rotation_delta_deg)),
            float(args.max_translation_delta_m),
            float(args.max_translation_delta_m),
            float(args.max_translation_delta_m),
        ],
        dtype=dtype,
        device=device,
    )
    rot_prior = np.deg2rad(max(1e-6, float(args.rotation_prior_deg)))
    trans_prior = max(1e-6, float(args.translation_prior_m))
    reg = float(args.regularization_weight)
    clip_px = max(1e-6, float(args.loss_clip_px))
    height, width = prepared[0].observed_dt.shape[:2]
    k_scaled = _scaled_intrinsic(intrinsic, float(args.optimize_scale))
    intrinsic_t = torch.as_tensor(k_scaled, dtype=dtype, device=device)
    frame_tensors = []
    has_mask_targets = any(frame.observed_mask is not None for frame in prepared)
    for frame in prepared:
        if frame.observed_dt.shape[:2] != (height, width):
            raise ValueError("all prepared frames must have the same optimized resolution")
        frame_meshes = _meshes_for_frame(meshes, frame.used_link_names)
        faces = torch.as_tensor(_mesh_global_faces(frame_meshes), dtype=torch.int32, device=device)
        vertices_ee = torch.as_tensor(_frame_vertices_in_ee(frame, frame_meshes), dtype=dtype, device=device)
        dt = torch.as_tensor(np.clip(frame.observed_dt, 0.0, clip_px) / clip_px, dtype=dtype, device=device).reshape(1, height, width, 1)
        mask_target = None
        if frame.observed_mask is not None:
            mask_target = torch.as_tensor(frame.observed_mask.astype(np.float32), dtype=dtype, device=device).reshape(1, height, width, 1)
        initial_area = float(np.mean(frame.initial_mask > 0))
        frame_tensors.append((frame.index, vertices_ee, faces, dt, mask_target, max(initial_area, 1e-6)))

    glctx = dr.RasterizeCudaContext()
    raw_delta = torch.zeros((6,), dtype=dtype, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([raw_delta], lr=float(args.nvdiffrast_lr))

    initial_metrics = _evaluate_alignment(prepared, meshes, ee_T_cam_initial, intrinsic, dist_coeffs, args)
    history = []
    last_loss = None
    for step in range(max(1, int(args.max_nfev))):
        optimizer.zero_grad(set_to_none=True)
        delta_vec = max_delta * torch.tanh(raw_delta)
        losses = []
        area_losses = []
        for _, vertices_ee, faces, dt, mask_target, initial_area in frame_tensors:
            soft_mask = _render_soft_mask_nvdiffrast(
                glctx,
                vertices_ee,
                faces,
                ee_T_cam_initial_t,
                delta_vec,
                intrinsic_t,
                width=width,
                height=height,
                near_clip_m=float(args.near_clip_m),
                max_triangle_edge_px=float(args.max_triangle_edge_px) * float(args.optimize_scale),
            )
            if mask_target is not None:
                mse = torch.mean((soft_mask - mask_target) ** 2)
                intersection = torch.sum(soft_mask * mask_target)
                dice = (2.0 * intersection + 1e-6) / (torch.sum(soft_mask) + torch.sum(mask_target) + 1e-6)
                losses.append(0.7 * mse + 0.3 * (1.0 - dice))
            else:
                rendered_edge = _torch_sobel_edges(soft_mask)
                edge_sum = torch.sum(rendered_edge) + 1e-6
                losses.append(torch.sum(rendered_edge * dt) / edge_sum)
            area = torch.mean(soft_mask)
            area_losses.append(((area - float(initial_area)) / float(initial_area)) ** 2)
        data_loss = torch.stack(losses).mean()
        area_loss = torch.stack(area_losses).mean()
        reg_loss = (
            torch.mean((delta_vec[:3] / float(rot_prior)) ** 2)
            + torch.mean((delta_vec[3:] / float(trans_prior)) ** 2)
        )
        reg_weight = float(reg) * (0.1 if has_mask_targets else 1.0)
        loss = data_loss + 0.02 * area_loss + reg_weight * reg_loss
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == int(args.max_nfev):
            delta_now = delta_vec.detach().cpu().numpy()
            item = {
                "step": int(step + 1),
                "loss": last_loss,
                "data_loss": float(data_loss.detach().cpu()),
                "area_loss": float(area_loss.detach().cpu()),
                "reg_loss": float(reg_loss.detach().cpu()),
                "delta_translation_m_norm": float(np.linalg.norm(delta_now[3:])),
                "delta_rotation_deg_norm": float(np.linalg.norm(delta_now[:3]) * 180.0 / np.pi),
            }
            history.append(item)
            print(
                "[wrist-visual][nvdiffrast] "
                f"step={item['step']:04d} loss={item['loss']:.5f} data={item['data_loss']:.5f} "
                f"delta={item['delta_translation_m_norm'] * 1000.0:.2f}mm/{item['delta_rotation_deg_norm']:.3f}deg"
            )

    delta_final = (max_delta * torch.tanh(raw_delta)).detach().cpu().numpy()
    delta_T = _delta_to_transform(delta_final)
    refined = as_transform(ee_T_cam_initial) @ delta_T
    final_metrics = _evaluate_alignment(prepared, meshes, refined, intrinsic, dist_coeffs, args)
    candidate_metrics = final_metrics
    candidate_delta = np.asarray(delta_final, dtype=np.float64).copy()
    candidate_delta_T = np.asarray(delta_T, dtype=np.float64).copy()
    candidate_refined = np.asarray(refined, dtype=np.float64).copy()
    accepted = True
    rejection_reason = _candidate_rejection_reason(initial_metrics, candidate_metrics, args)
    if rejection_reason:
        accepted = False
        delta_final = np.zeros((6,), dtype=np.float64)
        delta_T = np.eye(4, dtype=np.float64)
        refined = as_transform(ee_T_cam_initial)
        final_metrics = initial_metrics
    metrics = {
        "optimizer": "nvdiffrast",
        "success": True,
        "accepted": bool(accepted),
        "rejection_reason": rejection_reason,
        "used_mask_targets": bool(has_mask_targets),
        "message": "nvdiffrast differentiable silhouette optimization completed",
        "nfev": int(args.max_nfev),
        "objective": None if last_loss is None else float(last_loss),
        "history": history,
        "candidate_delta_rotvec_rad": candidate_delta[:3],
        "candidate_delta_translation_m": candidate_delta[3:],
        "candidate_delta_rotation_deg_norm": float(np.linalg.norm(candidate_delta[:3]) * 180.0 / np.pi),
        "candidate_delta_translation_m_norm": float(np.linalg.norm(candidate_delta[3:])),
        "candidate_delta_board_to_refined": candidate_delta_T,
        "candidate_ee_T_wrist_camera": candidate_refined,
        "delta_rotvec_rad": delta_final[:3],
        "delta_translation_m": delta_final[3:],
        "delta_rotation_deg_norm": float(np.linalg.norm(delta_final[:3]) * 180.0 / np.pi),
        "delta_translation_m_norm": float(np.linalg.norm(delta_final[3:])),
        "initial_metrics": initial_metrics,
        "candidate_metrics": candidate_metrics,
        "final_metrics": final_metrics,
    }
    return refined, delta_T, metrics


def _optimize_delta_powell(
    prepared: list[PreparedFrame],
    meshes: list[LinkMesh],
    ee_T_cam_initial: np.ndarray,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from scipy.optimize import Bounds, minimize

    sample_count = int(args.max_edge_samples)
    clip_px = float(args.loss_clip_px)
    scale = float(args.optimize_scale)
    max_rot = np.deg2rad(float(args.max_rotation_delta_deg))
    max_trans = float(args.max_translation_delta_m)
    rot_prior = np.deg2rad(max(1e-6, float(args.rotation_prior_deg)))
    trans_prior = max(1e-6, float(args.translation_prior_m))
    reg = float(args.regularization_weight)

    def sampled_residuals(delta_vec: np.ndarray) -> np.ndarray:
        delta_T = _delta_to_transform(delta_vec)
        ee_T_cam = as_transform(ee_T_cam_initial) @ delta_T
        chunks = []
        for frame in prepared:
            height, width = frame.image_bgr.shape[:2]
            frame_meshes = _meshes_for_frame(meshes, frame.used_link_names)
            mask = render_robot_mask(
                frame_meshes,
                frame.link_poses,
                frame.base_T_ee,
                ee_T_cam,
                intrinsic,
                dist_coeffs,
                width=width,
                height=height,
                scale=scale,
                render_mode=str(args.render_mode),
                near_clip_m=float(args.near_clip_m),
                max_triangle_edge_px=float(args.max_triangle_edge_px),
            )
            chunks.append(
                _fixed_sampled_distances(
                    frame.observed_dt,
                    mask_boundary(mask),
                    sample_count=sample_count,
                    clip_px=clip_px,
                )
            )
        chunks.append((delta_vec[:3] / rot_prior) * reg)
        chunks.append((delta_vec[3:] / trans_prior) * reg)
        return np.concatenate(chunks, axis=0)

    def objective(delta_vec: np.ndarray) -> float:
        residual = sampled_residuals(delta_vec)
        return float(np.mean(residual * residual))

    initial_metrics = _evaluate_alignment(prepared, meshes, ee_T_cam_initial, intrinsic, dist_coeffs, args)
    result = minimize(
        objective,
        np.zeros((6,), dtype=np.float64),
        method="Powell",
        bounds=Bounds(
            np.asarray([-max_rot, -max_rot, -max_rot, -max_trans, -max_trans, -max_trans], dtype=np.float64),
            np.asarray([max_rot, max_rot, max_rot, max_trans, max_trans, max_trans], dtype=np.float64),
        ),
        options={
            "maxfev": int(args.max_nfev),
            "xtol": 1e-4,
            "ftol": 1e-4,
            "disp": True,
        },
    )
    delta_T = _delta_to_transform(result.x)
    refined = as_transform(ee_T_cam_initial) @ delta_T
    final_metrics = _evaluate_alignment(prepared, meshes, refined, intrinsic, dist_coeffs, args)
    candidate_metrics = final_metrics
    candidate_delta = np.asarray(result.x, dtype=np.float64).copy()
    candidate_delta_T = np.asarray(delta_T, dtype=np.float64).copy()
    candidate_refined = np.asarray(refined, dtype=np.float64).copy()
    accepted = True
    delta_out = np.asarray(result.x, dtype=np.float64)
    rejection_reason = _candidate_rejection_reason(initial_metrics, candidate_metrics, args)
    if rejection_reason:
        accepted = False
        delta_out = np.zeros((6,), dtype=np.float64)
        delta_T = np.eye(4, dtype=np.float64)
        refined = as_transform(ee_T_cam_initial)
        final_metrics = initial_metrics
    metrics = {
        "optimizer": "powell",
        "success": bool(result.success),
        "accepted": bool(accepted),
        "rejection_reason": rejection_reason,
        "message": str(result.message),
        "nfev": int(result.nfev),
        "objective": float(result.fun),
        "used_mask_targets": bool(any(frame.observed_mask is not None for frame in prepared)),
        "candidate_delta_rotvec_rad": candidate_delta[:3],
        "candidate_delta_translation_m": candidate_delta[3:],
        "candidate_delta_rotation_deg_norm": float(np.linalg.norm(candidate_delta[:3]) * 180.0 / np.pi),
        "candidate_delta_translation_m_norm": float(np.linalg.norm(candidate_delta[3:])),
        "candidate_delta_board_to_refined": candidate_delta_T,
        "candidate_ee_T_wrist_camera": candidate_refined,
        "delta_rotvec_rad": delta_out[:3],
        "delta_translation_m": delta_out[3:],
        "delta_rotation_deg_norm": float(np.linalg.norm(delta_out[:3]) * 180.0 / np.pi),
        "delta_translation_m_norm": float(np.linalg.norm(delta_out[3:])),
        "initial_metrics": initial_metrics,
        "candidate_metrics": candidate_metrics,
        "final_metrics": final_metrics,
    }
    return refined, delta_T, metrics


def _save_alignment_overlays(
    run_dir: Path,
    prepared: list[PreparedFrame],
    meshes: list[LinkMesh],
    ee_T_cam: np.ndarray,
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
    args: argparse.Namespace,
    *,
    output_dir_name: str,
    status_prefix: str,
) -> None:
    scale = float(args.optimize_scale)
    for frame in prepared:
        height, width = frame.image_bgr.shape[:2]
        frame_meshes = _meshes_for_frame(meshes, frame.used_link_names)
        refined_mask = render_robot_mask(
            frame_meshes,
            frame.link_poses,
            frame.base_T_ee,
            ee_T_cam,
            intrinsic,
            dist_coeffs,
            width=width,
            height=height,
            scale=scale,
            render_mode=str(args.render_mode),
            near_clip_m=float(args.near_clip_m),
            max_triangle_edge_px=float(args.max_triangle_edge_px),
        )
        image_write(
            run_dir / output_dir_name / f"{frame.index:04d}.png",
            draw_alignment_overlay(
                frame.image_bgr,
                frame.observed_edges,
                frame.initial_mask,
                refined_mask,
                status_text=f"frame {frame.index:04d} {status_prefix} links={','.join(frame.used_link_names)}",
            ),
        )


def main() -> int:
    args = parse_args()
    settings = load_realman_settings(args.config)
    run_dir = Path(args.use_previous_run).expanduser().resolve() if args.use_previous_run else calibration_run_dir("wrist_camera_visual_refine", args.output_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.wrist_camera_run_dir is None and args.initial_ee_T_camera is None:
        from rm75_app.perception.wrist_relation import latest_hand_eye_path

        latest_hand_eye = latest_hand_eye_path()
        if latest_hand_eye is not None:
            args.initial_ee_T_camera = latest_hand_eye
            if latest_hand_eye.parent.name.endswith("_wrist_camera_board"):
                args.wrist_camera_run_dir = latest_hand_eye.parent
            print(f"[wrist-visual] default wrist hand-eye transform: {latest_hand_eye}")

    urdf_path = _urdf_path(args, settings)
    urdf = load_urdf(urdf_path)
    joint_names = _joint_names(args, settings)
    include_links = _include_links(args, settings)
    meshes = load_link_meshes(
        urdf_path,
        include_links=include_links,
        exclude_substrings=list(args.exclude_link_substrings or []),
    )
    ee_T_cam_initial = _load_initial_ee_T_cam(args, settings)
    save_matrix_pair(run_dir / "ee_T_wrist_camera_initial.npy", ee_T_cam_initial)

    source_run = args.input_run or args.use_previous_run
    if source_run is not None:
        frames, intrinsic, dist_coeffs = _frames_from_run(
            Path(source_run),
            run_dir,
            urdf=urdf,
            joint_names=joint_names,
            ee_link_name=_ee_link_name(args, settings),
        )
    elif args.manual_capture:
        frames, intrinsic, dist_coeffs = _collect_manual(args, settings, run_dir, urdf)
    else:
        raise RuntimeError("Use --manual-capture to collect frames, or --input-run/--use-previous-run to reuse frames.")

    prepared, frame_link_report, mask_quality_report = _prepare_frames(
        frames, meshes, ee_T_cam_initial, intrinsic, dist_coeffs, args, run_dir
    )
    print(f"[wrist-visual] usable frames: {len(prepared)}/{len(frames)}")
    ee_T_cam_refined, delta_T, metrics = _optimize_delta(
        prepared,
        meshes,
        ee_T_cam_initial,
        intrinsic,
        dist_coeffs,
        args,
    )
    accepted = bool(metrics.get("accepted", True))
    rejection_reason = str(metrics.get("rejection_reason", ""))
    candidate_ee_T_cam = np.asarray(metrics.get("candidate_ee_T_wrist_camera", ee_T_cam_refined), dtype=np.float64)
    candidate_delta_T = np.asarray(metrics.get("candidate_delta_board_to_refined", delta_T), dtype=np.float64)
    save_matrix_pair(run_dir / "ee_T_wrist_camera_candidate.npy", candidate_ee_T_cam)
    save_matrix_pair(run_dir / "delta_board_to_candidate.npy", candidate_delta_T)
    if accepted:
        save_matrix_pair(run_dir / "ee_T_wrist_camera_refined.npy", ee_T_cam_refined)
        save_matrix_pair(run_dir / "wrist_camera_T_ee_refined.npy", invert_transform(ee_T_cam_refined))
        save_matrix_pair(run_dir / "delta_board_to_refined.npy", delta_T)
    else:
        print(f"[wrist-visual] rejected candidate was saved for inspection only: {rejection_reason}")
    _save_alignment_overlays(
        run_dir,
        prepared,
        meshes,
        candidate_ee_T_cam,
        intrinsic,
        dist_coeffs,
        args,
        output_dir_name="candidate_overlays",
        status_prefix="candidate " + ("accepted" if accepted else "rejected"),
    )
    _save_alignment_overlays(
        run_dir,
        prepared,
        meshes,
        ee_T_cam_refined,
        intrinsic,
        dist_coeffs,
        args,
        output_dir_name="final_overlays",
        status_prefix="final accepted" if accepted else "final rejected using initial",
    )

    report = {
        "run_dir": run_dir,
        "input_run": None if source_run is None else Path(source_run).expanduser(),
        "wrist_camera_run_dir": None if args.wrist_camera_run_dir is None else Path(args.wrist_camera_run_dir).expanduser(),
        "frames_total": len(frames),
        "frames_used": len(prepared),
        "render_mode": str(args.render_mode),
        "optimize_scale": float(args.optimize_scale),
        "sam3_mask": bool(args.sam3_mask),
        "sam3_prompt": str(args.sam3_prompt),
        "link_names": [mesh.link_name for mesh in meshes],
        "mesh_sources": {mesh.link_name: mesh.source_files for mesh in meshes},
        "initial_ee_T_wrist_camera": ee_T_cam_initial,
        "candidate_ee_T_wrist_camera": candidate_ee_T_cam,
        "candidate_delta_board_to_refined": candidate_delta_T,
        "ee_T_wrist_camera_refined": ee_T_cam_refined if accepted else None,
        "delta_board_to_refined": delta_T if accepted else None,
        "frame_link_visibility": frame_link_report,
        "mask_quality": mask_quality_report,
        "metrics": metrics,
    }
    report = normalize_calibration_report(
        report,
        run_kind="wrist_camera_visual_refine",
        config_path=args.config,
        frames_total=len(frames),
        frames_used=len(prepared),
        accepted=accepted,
        rejection_reason=rejection_reason,
        transforms={"T_E_Cw": ee_T_cam_refined if accepted else None},
        metrics=metrics,
    )
    save_json(run_dir / "visual_refine_report.json", report)
    if not args.no_html_report:
        write_html_report(
            run_dir / "report.html",
            title="RM75 Wrist Camera Visual Refinement",
            summary={
                "frames_total": len(frames),
                "frames_used": len(prepared),
                "accepted": accepted,
                "rejection_reason": rejection_reason,
                "render_mode": str(args.render_mode),
                "sam3_mask": bool(args.sam3_mask),
                "delta_translation_m_norm": metrics["delta_translation_m_norm"],
                "delta_rotation_deg_norm": metrics["delta_rotation_deg_norm"],
                "initial_edge_distance_mean_px": metrics["initial_metrics"]["edge_distance_mean_px"],
                "candidate_edge_distance_mean_px": metrics["candidate_metrics"]["edge_distance_mean_px"],
                "final_edge_distance_mean_px": metrics["final_metrics"]["edge_distance_mean_px"],
                "initial_edge_distance_p90_px": metrics["initial_metrics"]["edge_distance_p90_px"],
                "candidate_edge_distance_p90_px": metrics["candidate_metrics"]["edge_distance_p90_px"],
                "final_edge_distance_p90_px": metrics["final_metrics"]["edge_distance_p90_px"],
                "initial_mask_dice_mean": metrics["initial_metrics"].get("mask_dice_mean"),
                "candidate_mask_dice_mean": metrics["candidate_metrics"].get("mask_dice_mean"),
                "final_mask_dice_mean": metrics["final_metrics"].get("mask_dice_mean"),
            },
            tables=[
                (
                    "Mask Selection",
                    mask_quality_report,
                    [
                        "frame_index",
                        "source",
                        "status",
                        "used_mask",
                        "selected_candidate_rank",
                        "selected_candidate_score",
                        "selected_model_score",
                        "projected_overlap",
                        "mask_area_px",
                        "candidate_count",
                        "reason",
                    ],
                ),
                (
                    "Link Visibility",
                    frame_link_report,
                    [
                        "frame_index",
                        "used_in_refine",
                        "used_link_count",
                        "used_links",
                        "observed_edge_pixels",
                        "mask_source",
                        "mask_status",
                        "reason",
                        "link_projected_area_px",
                    ],
                ),
                (
                    "Final Per-Frame Validation",
                    metrics["final_metrics"].get("per_frame", []),
                    [
                        "index",
                        "mask_source",
                        "used_links",
                        "edge_distance_mean_px",
                        "edge_distance_p90_px",
                        "mask_iou",
                        "mask_dice",
                        "observed_edge_pixels",
                        "rendered_edge_pixels",
                    ],
                ),
            ],
            sections=[
                ("Transforms", report["transforms"]),
                ("Initial Metrics", metrics["initial_metrics"]),
                ("Candidate Metrics", metrics["candidate_metrics"]),
                ("Final Metrics", metrics["final_metrics"]),
                ("Optimization", {key: value for key, value in metrics.items() if key not in {"initial_metrics", "candidate_metrics", "final_metrics"}}),
            ],
            image_dirs=["sam3_overlays", "initial_overlays", "candidate_overlays", "final_overlays"] if bool(args.sam3_mask) else ["initial_overlays", "candidate_overlays", "final_overlays"],
            max_images_per_dir=60,
        )
    if args.write_refined_to_wrist_run:
        if args.wrist_camera_run_dir is None:
            raise ValueError("--write-refined-to-wrist-run requires --wrist-camera-run-dir or an auto-detected wrist camera board run")
        target_dir = Path(args.wrist_camera_run_dir).expanduser()
        if accepted:
            save_matrix_pair(target_dir / "ee_T_wrist_camera_refined.npy", ee_T_cam_refined)
            save_matrix_pair(target_dir / "delta_board_to_visual_refined.npy", delta_T)
            save_json(
                target_dir / "ee_T_wrist_camera_refined_metadata.json",
                {
                    "schema_version": 1,
                    "run_kind": "wrist_camera_visual_refine",
                    "accepted": True,
                    "source_run": run_dir,
                    "transform_path": target_dir / "ee_T_wrist_camera_refined.npy",
                    "delta_path": target_dir / "delta_board_to_visual_refined.npy",
                    "metrics": metrics,
                },
            )
            print(f"[wrist-visual] wrote accepted refined transform into {target_dir}")
        else:
            print(f"[wrist-visual] rejected refinement was not copied into trusted wrist run: {rejection_reason}")
    print(
        "[wrist-visual] edge mean px: "
        f"{metrics['initial_metrics']['edge_distance_mean_px']:.3f} -> {metrics['final_metrics']['edge_distance_mean_px']:.3f}"
    )
    print(
        "[wrist-visual] delta: "
        f"{metrics['delta_translation_m_norm'] * 1000.0:.2f} mm, {metrics['delta_rotation_deg_norm']:.3f} deg"
    )
    print(f"[wrist-visual] refinement written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
