#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

from rm75_app.assets.object_specs import get_object_spec, normalize_object_name, resolve_object_spec_scales
from rm75_app.paths import APP_ROOT, ASSET_DIR, DEFAULT_RM75_URDF, RUNTIME_DIR
from rm75_app.perception.wrist_relation import latest_hand_eye_path


PICK_DIR = APP_ROOT.parent / "pick_jiaobang"
if str(PICK_DIR) not in sys.path:
    sys.path.insert(0, str(PICK_DIR))

from real_to_sim_pose_corrector.depth_utils import CameraIntrinsics  # noqa: E402
from real_to_sim_pose_corrector.sam3_persistent import PersistentSam3Client  # noqa: E402
from real_to_sim_pose_corrector.transforms import invert_transform, mat_to_quat_wxyz, transform_points  # noqa: E402
from run_realsim_pose_corrector import _load_frame_npz, _save_frame_npz  # noqa: E402


def _load_legacy_wrist_module():
    script = PICK_DIR / "wrist_gripper_object_relation.py"
    spec = importlib.util.spec_from_file_location("_rm75_legacy_wrist_relation", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import legacy wrist relation script: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


WRIST_LEGACY = _load_legacy_wrist_module()


OBJECT_COLORS_RGB = {
    "lvmukuai": (42, 210, 72),
    "carriot": (245, 128, 34),
    "gluestick": (245, 245, 245),
    "hongshupian": (210, 55, 45),
    "tennis": (205, 235, 45),
    "bi": (220, 40, 40),
    "shuazi": (235, 235, 235),
    "bitong": (190, 165, 118),
    "tingzi_base": (190, 190, 185),
    "tingzi_pillar_front_left": (190, 190, 185),
    "tingzi_pillar_front_right": (190, 190, 185),
    "tingzi_pillar_back_left": (190, 190, 185),
    "tingzi_pillar_back_right": (190, 190, 185),
}


@dataclass
class SceneCloud:
    name: str
    points: np.ndarray
    T_cam_cloud: np.ndarray
    color_rgb: tuple[int, int, int]
    radius_px: int = 2
    shade: bool = True


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _legacy_default_args() -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["wrist_gripper_object_relation.py"]
        return WRIST_LEGACY.parse_args()
    finally:
        sys.argv = old_argv


def _timestamp_dir(base: Path) -> Path:
    return Path(base).expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_wrist_live_overlay")


def _latest_relation_prior(object_name: str) -> Path | None:
    roots = [
        RUNTIME_DIR / "wrist_relation_priors",
        APP_ROOT.parent / "runtime_data" / "wrist_relation_priors",
        APP_ROOT.parent / "pick_jiaobang" / "wrist_relation_priors",
    ]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(root.glob(f"*_{object_name}_T_gripper_obj.json"))
            candidates.extend(root.glob(f"{object_name}_T_gripper_obj.json"))
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


class RealSenseStream:
    def __init__(self, serial: str, width: int, height: int, fps: int, warmup_frames: int):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(str(serial))
        config.enable_stream(rs.stream.depth, int(width), int(height), rs.format.z16, int(fps))
        config.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        for _ in range(max(0, int(warmup_frames))):
            self.align.process(self.pipeline.wait_for_frames())
        color_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        intr = color_profile.get_intrinsics()
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_units_per_meter = 1.0 / float(depth_sensor.get_depth_scale())
        K = np.asarray(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.intr = CameraIntrinsics(
            K=K,
            width=int(intr.width),
            height=int(intr.height),
            depth_units_per_meter=depth_units_per_meter,
        )

    def read(self) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
        frames = self.align.process(self.pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("RealSense did not return aligned color/depth frames")
        depth = np.asanyarray(depth_frame.get_data()).copy()
        bgr = np.asanyarray(color_frame.get_data()).copy()
        return bgr[:, :, ::-1].copy(), depth, self.intr

    def close(self) -> None:
        self.pipeline.stop()


def _load_app_mesh(object_name: str, *, use_sim_asset: bool = False) -> tuple[trimesh.Trimesh, Path, float]:
    spec = get_object_spec(object_name)
    if spec is None:
        raise KeyError(f"unknown object spec: {object_name}")
    mesh_scale, sim_scale = resolve_object_spec_scales(spec)
    asset = spec.sim_asset_file if use_sim_asset and spec.sim_asset_file else spec.mesh_file
    path = Path(str(asset)).expanduser()
    if not path.is_absolute():
        path = ASSET_DIR / path
    loaded = trimesh.load(str(path), force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        loaded = loaded.dump(concatenate=True)
    mesh = loaded.copy()
    scale = sim_scale if use_sim_asset else mesh_scale
    mesh.apply_scale(float(scale))
    return mesh, path, float(scale)


def _sample_render_points(mesh: trimesh.Trimesh, count: int, seed: int = 0) -> np.ndarray:
    count = int(max(100, count))
    if len(mesh.faces) > 0:
        rng_state = np.random.get_state()
        np.random.seed(int(seed))
        try:
            pts, _ = trimesh.sample.sample_surface(mesh, count)
        finally:
            np.random.set_state(rng_state)
        return np.asarray(pts, dtype=np.float64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.shape[0] > count:
        rng = np.random.default_rng(seed)
        verts = verts[rng.choice(verts.shape[0], size=count, replace=False)]
    return verts


def _rpy_matrix_deg(rpy_deg: list[float] | tuple[float, float, float]) -> np.ndarray:
    r, p, y = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64).reshape(3))
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _make_transform(xyz: list[float] | tuple[float, float, float], rpy_deg: list[float] | tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rpy_matrix_deg(rpy_deg)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64).reshape(3)
    return T


def _load_T_gripper_obj(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    payload = WRIST_LEGACY._load_transform_payload(Path(path).expanduser())
    T = WRIST_LEGACY._extract_transform(
        payload,
        ("T_gripper_obj", "matrix", "T"),
        path=Path(path).expanduser(),
    )
    return None if T is None else np.asarray(T, dtype=np.float64).reshape(4, 4)


def _mesh_path_from_urdf_mesh(urdf_path: Path, filename: str) -> Path:
    raw = str(filename)
    if raw.startswith("package://crt_ctag2f90d_gripper_visualization/"):
        rel = raw.split("package://crt_ctag2f90d_gripper_visualization/", 1)[1]
        return DEFAULT_RM75_URDF.parents[2] / "crt_ctag2f90d_gripper_visualization" / rel
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (Path(urdf_path).expanduser().parent / path).resolve()


def _load_gripper_scene_clouds(
    *,
    urdf_path: Path,
    T_gripper_cam: np.ndarray,
    gripper_joint_q: float,
    points_per_link: int,
    point_radius_px: int,
) -> list[SceneCloud]:
    try:
        from yourdfpy import URDF
    except Exception as exc:
        print(f"[warn] cannot import yourdfpy; gripper scene will not be rendered: {exc}")
        return []

    urdf_path = Path(urdf_path).expanduser()
    robot = URDF.load(str(urdf_path))
    cfg = {name: np.array(0.0, dtype=np.float64) for name in robot.joint_map.keys()}
    gripper_joint_names = [
        "gripper_Left_1_Joint",
        "gripper_Left_Support_Joint",
        "gripper_Left_2_Joint",
        "gripper_Right_1_Joint",
        "gripper_Right_Support_Joint",
        "gripper_Right_2_Joint",
    ]
    for name in gripper_joint_names:
        if name in cfg:
            cfg[name] = np.array(float(gripper_joint_q), dtype=np.float64)
    robot.update_cfg(cfg)
    if "gripper_tcp" not in robot.link_map:
        raise KeyError(f"{urdf_path} has no gripper_tcp link")

    T_base_tcp = np.asarray(robot.get_transform("gripper_tcp", "base_link"), dtype=np.float64).reshape(4, 4)
    T_tcp_base = invert_transform(T_base_tcp)
    T_cam_gripper = invert_transform(np.asarray(T_gripper_cam, dtype=np.float64).reshape(4, 4))
    clouds: list[SceneCloud] = []
    link_names = [
        "gripper_base_link",
        "gripper_Left_1_Link",
        "gripper_Left_Support_Link",
        "gripper_Left_2_Link",
        "gripper_Right_1_Link",
        "gripper_Right_Support_Link",
        "gripper_Right_2_Link",
    ]
    for link_name in link_names:
        link = robot.link_map.get(link_name)
        if link is None:
            continue
        T_base_link = np.asarray(robot.get_transform(link_name, "base_link"), dtype=np.float64).reshape(4, 4)
        T_tcp_link = T_tcp_base @ T_base_link
        for visual_idx, visual in enumerate(link.visuals or []):
            mesh_info = getattr(getattr(visual, "geometry", None), "mesh", None)
            if mesh_info is None or not getattr(mesh_info, "filename", None):
                continue
            mesh_path = _mesh_path_from_urdf_mesh(urdf_path, str(mesh_info.filename))
            loaded = trimesh.load(str(mesh_path), force="mesh")
            if not isinstance(loaded, trimesh.Trimesh):
                loaded = loaded.dump(concatenate=True)
            mesh = loaded.copy()
            scale = getattr(mesh_info, "scale", None)
            if scale is not None:
                mesh.apply_scale(np.asarray(scale, dtype=np.float64).reshape(3))
            count = int(points_per_link)
            if link_name == "gripper_base_link":
                count = int(max(points_per_link * 2, points_per_link))
            points = _sample_render_points(mesh, count, seed=abs(hash((link_name, visual_idx))) % (2**31))
            T_visual = np.eye(4, dtype=np.float64) if getattr(visual, "origin", None) is None else np.asarray(visual.origin, dtype=np.float64).reshape(4, 4)
            points_tcp = transform_points(T_tcp_link @ T_visual, points)
            color = (26, 26, 26) if link_name != "gripper_base_link" else (72, 72, 72)
            clouds.append(
                SceneCloud(
                    name=link_name,
                    points=points_tcp,
                    T_cam_cloud=T_cam_gripper,
                    color_rgb=color,
                    radius_px=max(1, int(point_radius_px)),
                    shade=True,
                )
            )

    for pad_name in ("left_pad", "right_pad"):
        if pad_name not in robot.link_map:
            continue
        try:
            T_tcp_pad = T_tcp_base @ np.asarray(robot.get_transform(pad_name, "base_link"), dtype=np.float64).reshape(4, 4)
        except Exception:
            continue
        pad_mesh = trimesh.creation.box(extents=(0.010, 0.036, 0.018))
        pad_points = _sample_render_points(pad_mesh, max(300, points_per_link // 2), seed=17 if pad_name == "left_pad" else 19)
        clouds.append(
            SceneCloud(
                name=pad_name,
                points=transform_points(T_tcp_pad, pad_points),
                T_cam_cloud=T_cam_gripper,
                color_rgb=(12, 12, 12),
                radius_px=max(1, int(point_radius_px)),
                shade=True,
            )
        )
    print(
        f"[wrist-live] gripper scene: {len(clouds)} cloud(s), "
        f"points={sum(int(c.points.shape[0]) for c in clouds)}, q={float(gripper_joint_q):.3f}"
    )
    return clouds


def _render_scene_from_wrist(
    *,
    clouds: list[SceneCloud],
    K: np.ndarray,
    width: int,
    height: int,
    background_rgb: tuple[int, int, int] = (232, 232, 232),
) -> tuple[np.ndarray, np.ndarray]:
    sim = np.full((height, width, 3), np.asarray(background_rgb, dtype=np.uint8), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    for cloud in clouds:
        pts = transform_points(np.asarray(cloud.T_cam_cloud, dtype=np.float64).reshape(4, 4), cloud.points)
        z = pts[:, 2]
        valid = np.isfinite(z) & (z > 1e-5)
        if not np.any(valid):
            continue
        pts = pts[valid]
        z = pts[:, 2]
        u = (pts[:, 0] * K[0, 0] / z) + K[0, 2]
        v = (pts[:, 1] * K[1, 1] / z) + K[1, 2]
        in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not np.any(in_img):
            continue
        z = z[in_img]
        u = np.round(u[in_img]).astype(np.int32)
        v = np.round(v[in_img]).astype(np.int32)
        base = np.asarray(cloud.color_rgb, dtype=np.float32)
        if bool(cloud.shade):
            z_near = float(np.percentile(z, 3))
            z_far = float(np.percentile(z, 97))
            denom = max(z_far - z_near, 1e-6)
            shade = 0.55 + 0.45 * (1.0 - np.clip((z - z_near) / denom, 0.0, 1.0))
        else:
            shade = np.ones_like(z)
        colors = np.clip(base[None, :] * shade[:, None], 0, 255).astype(np.uint8)
        records.append((z, u, v, colors, int(max(1, cloud.radius_px))))
    if not records:
        cv2.putText(sim, "no rendered scene", (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (90, 90, 90), 2)
        return sim, mask.astype(bool)
    z_all = np.concatenate([r[0] for r in records])
    u_all = np.concatenate([r[1] for r in records])
    v_all = np.concatenate([r[2] for r in records])
    c_all = np.concatenate([r[3] for r in records])
    radius_all = np.concatenate([np.full(r[0].shape, r[4], dtype=np.int32) for r in records])
    order = np.argsort(z_all)[::-1]
    for idx in order:
        center = (int(u_all[idx]), int(v_all[idx]))
        color = tuple(int(x) for x in c_all[idx].tolist())
        radius = int(radius_all[idx])
        cv2.circle(sim, center, radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(mask, center, radius, 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sim, contours, -1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return sim, mask.astype(bool)


def _sapien_pose_from_T(T: np.ndarray):
    import sapien

    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return sapien.Pose(
        p=T[:3, 3].astype(np.float32),
        q=mat_to_quat_wxyz(T[:3, :3]).astype(np.float32),
    )


def _opencv_cam_to_sapien_pose(T_world_cam_opencv: np.ndarray):
    T = np.asarray(T_world_cam_opencv, dtype=np.float64).reshape(4, 4)
    cv_from_ros = np.asarray(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return _sapien_pose_from_T(T @ cv_from_ros)


class SapienWristSceneRenderer:
    def __init__(
        self,
        *,
        object_name: str,
        object_mesh_path: Path,
        object_scale: float,
        object_color_rgb: tuple[int, int, int],
        urdf_path: Path,
        T_gripper_cam: np.ndarray,
        intr: CameraIntrinsics,
        gripper_joint_q: float,
        render_gripper: bool,
    ):
        import sapien
        from yourdfpy import URDF

        self.sapien = sapien
        self.object_name = str(object_name)
        self.width = int(intr.width)
        self.height = int(intr.height)
        self.background_rgb = np.asarray([196, 196, 196], dtype=np.uint8)
        self.scene = sapien.Scene()
        self.scene.set_timestep(1 / 100.0)
        self.scene.set_ambient_light([0.65, 0.65, 0.65])
        self.scene.add_directional_light([0.2, 0.4, -1.0], [1.2, 1.2, 1.2])
        self.scene.add_point_light([0.3, -0.2, 0.2], [1.0, 1.0, 1.0])

        urdf_path = Path(urdf_path).expanduser()
        if bool(render_gripper):
            robot = URDF.load(str(urdf_path))
            cfg = {name: np.array(0.0, dtype=np.float64) for name in robot.joint_map.keys()}
            for name in (
                "gripper_Left_1_Joint",
                "gripper_Left_Support_Joint",
                "gripper_Left_2_Joint",
                "gripper_Right_1_Joint",
                "gripper_Right_Support_Joint",
                "gripper_Right_2_Joint",
            ):
                if name in cfg:
                    cfg[name] = np.array(float(gripper_joint_q), dtype=np.float64)
            robot.update_cfg(cfg)
            T_base_tcp = np.asarray(robot.get_transform("gripper_tcp", "base_link"), dtype=np.float64).reshape(4, 4)
            T_tcp_base = invert_transform(T_base_tcp)
            self._add_gripper_visuals(robot=robot, urdf_path=urdf_path, T_tcp_base=T_tcp_base)

        builder = self.scene.create_actor_builder()
        rgba = [float(c) / 255.0 for c in object_color_rgb] + [1.0]
        builder.add_visual_from_file(
            str(Path(object_mesh_path).expanduser()),
            scale=(float(object_scale), float(object_scale), float(object_scale)),
            material=rgba,
            name=self.object_name,
        )
        self.object_actor = builder.build_kinematic(name=self.object_name)

        K = np.asarray(intr.K, dtype=np.float64).reshape(3, 3)
        fovy = float(2.0 * np.arctan(float(intr.height) / max(2.0 * float(K[1, 1]), 1e-9)))
        self.camera = self.scene.add_camera("wrist_overlay_camera", self.width, self.height, fovy, 0.01, 10.0)
        self.camera.entity.set_pose(_opencv_cam_to_sapien_pose(T_gripper_cam))

    def _add_gripper_visuals(self, *, robot, urdf_path: Path, T_tcp_base: np.ndarray) -> None:
        link_names = [
            "gripper_base_link",
            "gripper_Left_1_Link",
            "gripper_Left_Support_Link",
            "gripper_Left_2_Link",
            "gripper_Right_1_Link",
            "gripper_Right_Support_Link",
            "gripper_Right_2_Link",
        ]
        visual_count = 0
        for link_name in link_names:
            link = robot.link_map.get(link_name)
            if link is None:
                continue
            T_base_link = np.asarray(robot.get_transform(link_name, "base_link"), dtype=np.float64).reshape(4, 4)
            T_tcp_link = T_tcp_base @ T_base_link
            for visual_idx, visual in enumerate(link.visuals or []):
                mesh_info = getattr(getattr(visual, "geometry", None), "mesh", None)
                if mesh_info is None or not getattr(mesh_info, "filename", None):
                    continue
                mesh_path = _mesh_path_from_urdf_mesh(urdf_path, str(mesh_info.filename))
                T_visual = (
                    np.eye(4, dtype=np.float64)
                    if getattr(visual, "origin", None) is None
                    else np.asarray(visual.origin, dtype=np.float64).reshape(4, 4)
                )
                scale = getattr(mesh_info, "scale", None)
                scale_tuple = (1.0, 1.0, 1.0) if scale is None else tuple(np.asarray(scale, dtype=np.float64).reshape(3).tolist())
                material = [0.03, 0.03, 0.03, 1.0] if link_name != "gripper_base_link" else [0.25, 0.25, 0.25, 1.0]
                builder = self.scene.create_actor_builder()
                builder.add_visual_from_file(
                    str(mesh_path),
                    pose=_sapien_pose_from_T(T_tcp_link @ T_visual),
                    scale=scale_tuple,
                    material=material,
                    name=f"{link_name}_{visual_idx}",
                )
                builder.build_kinematic(name=f"{link_name}_{visual_idx}")
                visual_count += 1

        for pad_name in ("left_pad", "right_pad"):
            if pad_name not in robot.link_map:
                continue
            try:
                T_tcp_pad = T_tcp_base @ np.asarray(robot.get_transform(pad_name, "base_link"), dtype=np.float64).reshape(4, 4)
            except Exception:
                continue
            builder = self.scene.create_actor_builder()
            builder.add_box_visual(
                pose=_sapien_pose_from_T(T_tcp_pad),
                half_size=(0.005, 0.018, 0.009),
                material=[0.01, 0.01, 0.01, 1.0],
                name=pad_name,
            )
            builder.build_kinematic(name=pad_name)
            visual_count += 1
        print(f"[wrist-live] sapien gripper visuals={visual_count}")

    def render(self, *, T_gripper_obj: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        if T_gripper_obj is not None:
            self.object_actor.set_pose(_sapien_pose_from_T(T_gripper_obj))
        else:
            self.object_actor.set_pose(self.sapien.Pose(p=[0.0, 0.0, -10.0]))
        self.scene.update_render()
        self.camera.take_picture()
        color = self.camera.get_picture("Color")
        arr = np.asarray(color)[..., :3]
        if arr.dtype != np.uint8:
            arr = (arr.astype(np.float32).clip(0.0, 1.0) * 255.0).astype(np.uint8)
        rgb = np.ascontiguousarray(arr)
        bg = np.median(
            np.concatenate(
                [
                    rgb[:20, :20].reshape(-1, 3),
                    rgb[:20, -20:].reshape(-1, 3),
                    rgb[-20:, :20].reshape(-1, 3),
                    rgb[-20:, -20:].reshape(-1, 3),
                ],
                axis=0,
            ),
            axis=0,
        )
        dist = np.linalg.norm(rgb.astype(np.float32) - bg.reshape(1, 1, 3).astype(np.float32), axis=2)
        mask = dist > 9.0
        return rgb, mask

    def render_gripper_only(self) -> tuple[np.ndarray, np.ndarray]:
        return self.render(T_gripper_obj=None)


def _render_object_from_wrist(
    *,
    points_obj: np.ndarray,
    T_cam_obj: np.ndarray | None,
    K: np.ndarray,
    width: int,
    height: int,
    color_rgb: tuple[int, int, int],
    point_radius_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    sim = np.full((height, width, 3), 20, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    if T_cam_obj is None:
        cv2.putText(sim, "no pose yet", (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (190, 190, 190), 2)
        return sim, mask.astype(bool)

    pts = transform_points(np.asarray(T_cam_obj, dtype=np.float64).reshape(4, 4), points_obj)
    z = pts[:, 2]
    valid = np.isfinite(z) & (z > 1e-5)
    if not np.any(valid):
        cv2.putText(sim, "object behind camera", (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (190, 190, 190), 2)
        return sim, mask.astype(bool)

    pts = pts[valid]
    z = pts[:, 2]
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    u = (pts[:, 0] * K[0, 0] / z) + K[0, 2]
    v = (pts[:, 1] * K[1, 1] / z) + K[1, 2]
    in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(in_img):
        cv2.putText(sim, "projection outside image", (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (190, 190, 190), 2)
        return sim, mask.astype(bool)

    u = np.round(u[in_img]).astype(np.int32)
    v = np.round(v[in_img]).astype(np.int32)
    z = z[in_img]
    order = np.argsort(z)[::-1]
    base = np.asarray(color_rgb, dtype=np.float32)
    z_near = float(np.percentile(z, 5))
    z_far = float(np.percentile(z, 95))
    denom = max(z_far - z_near, 1e-6)
    radius = int(max(1, point_radius_px))
    for idx in order:
        shade = 0.55 + 0.45 * (1.0 - np.clip((float(z[idx]) - z_near) / denom, 0.0, 1.0))
        color = tuple(int(x) for x in np.clip(base * shade, 0, 255).tolist())
        cv2.circle(sim, (int(u[idx]), int(v[idx])), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(mask, (int(u[idx]), int(v[idx])), radius, 255, -1)
    if radius <= 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sim, contours, -1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return sim, mask.astype(bool)


def _bbox_from_bool_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(np.asarray(mask > 0, dtype=bool))
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _load_bool_mask(path: Path | None, shape_hw: tuple[int, int]) -> np.ndarray | None:
    if path is None:
        return None
    path = Path(path).expanduser()
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    height, width = int(shape_hw[0]), int(shape_hw[1])
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.asarray(mask > 0, dtype=bool)


def _extract_sam3_overlay(
    debug: dict[str, Any] | None,
    shape_hw: tuple[int, int],
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, str, list[tuple[int, int, int, int, str]]]:
    if not isinstance(debug, dict):
        return None, None, "SAM3", []
    sam3 = debug.get("sam3") if isinstance(debug.get("sam3"), dict) else {}
    selected = sam3.get("selected") if isinstance(sam3.get("selected"), dict) else {}
    mask_path = sam3.get("selected_mask_path") or sam3.get("mask_path") or selected.get("mask_path")
    if mask_path is None:
        for entry in sam3.get("results", []) or []:
            if isinstance(entry, dict) and entry.get("ok") and entry.get("mask_path"):
                mask_path = entry.get("mask_path")
                selected = entry
                break
    mask = _load_bool_mask(Path(mask_path).expanduser() if mask_path else None, shape_hw)
    bbox_raw = sam3.get("mask_bbox") or selected.get("mask_bbox")
    if bbox_raw is None and mask is not None:
        bbox = _bbox_from_bool_mask(mask)
    elif bbox_raw is not None:
        vals = [int(round(float(v))) for v in list(bbox_raw)[:4]]
        bbox = (vals[0], vals[1], vals[2], vals[3])
    else:
        bbox = None
    prompt = str(sam3.get("prompt") or selected.get("prompt") or "")
    pixels = int(np.count_nonzero(mask)) if mask is not None else int(sam3.get("mask_pixels") or selected.get("mask_pixels") or 0)
    candidate_boxes: list[tuple[int, int, int, int, str]] = []
    selected_entries = sam3.get("selected_entries") if isinstance(sam3.get("selected_entries"), list) else []
    if not selected_entries and isinstance(selected, dict):
        selected_entries = [selected]
    for entry_idx, entry in enumerate(selected_entries):
        if not isinstance(entry, dict):
            continue
        rank = int(entry.get("candidate_rank") or entry_idx + 1)
        entry_prompt = str(entry.get("prompt") or "")
        base_label = f"r{rank} {entry_prompt[:16]}".strip()
        component_bboxes = list(entry.get("component_bboxes") or [])
        if component_bboxes:
            for comp in component_bboxes:
                if not isinstance(comp, dict):
                    continue
                comp_bbox = comp.get("bbox_xyxy")
                if isinstance(comp_bbox, list) and len(comp_bbox) == 4:
                    vals = [int(round(float(v))) for v in comp_bbox]
                    candidate_boxes.append((vals[0], vals[1], vals[2], vals[3], f"{base_label} c{comp.get('component_index', '')}".strip()))
        else:
            entry_bbox = entry.get("mask_bbox")
            if isinstance(entry_bbox, list) and len(entry_bbox) == 4:
                vals = [int(round(float(v))) for v in entry_bbox]
                candidate_boxes.append((vals[0], vals[1], vals[2], vals[3], base_label))
    label = f"SAM3 {prompt} px={pixels}" if prompt else f"SAM3 px={pixels}"
    return mask, bbox, label, candidate_boxes


def _draw_sam3_observation(
    panel: np.ndarray,
    mask: np.ndarray | None,
    bbox: tuple[int, int, int, int] | None,
    label: str,
    candidate_bboxes: list[tuple[int, int, int, int, str]] | None = None,
) -> np.ndarray:
    out = np.asarray(panel, dtype=np.uint8).copy()
    if mask is not None and np.any(mask):
        mask_bool = np.asarray(mask > 0, dtype=bool)
        overlay = out.copy()
        overlay[mask_bool] = (25, 245, 70)
        out = cv2.addWeighted(overlay, 0.30, out, 0.70, 0.0)
        contours, _ = cv2.findContours(mask_bool.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (20, 255, 80), 2, lineType=cv2.LINE_AA)
    for idx, cand in enumerate(list(candidate_bboxes or [])[:12]):
        x0, y0, x1, y1, cand_label = cand
        h, w = out.shape[:2]
        x0 = int(np.clip(x0, 0, max(w - 1, 0)))
        x1 = int(np.clip(x1, 0, w))
        y0 = int(np.clip(y0, 0, max(h - 1, 0)))
        y1 = int(np.clip(y1, 0, h))
        if x1 <= x0 or y1 <= y0:
            continue
        color = (30, 230, 255) if idx % 2 == 0 else (255, 120, 40)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 1, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            str(cand_label)[:28],
            (x0, min(max(18, y0 + 14), h - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        h, w = out.shape[:2]
        x0 = int(np.clip(x0, 0, max(w - 1, 0)))
        x1 = int(np.clip(x1, 0, w))
        y0 = int(np.clip(y0, 0, max(h - 1, 0)))
        y1 = int(np.clip(y1, 0, h))
        if x1 > x0 and y1 > y0:
            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 230, 30), 2, lineType=cv2.LINE_AA)
            cv2.putText(out, "SAM3", (x0, max(18, y0 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 230, 30), 2, cv2.LINE_AA)
    cv2.putText(out, label[:70], (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 255, 80), 2, cv2.LINE_AA)
    return out


def _compose_three_view(
    *,
    real_rgb: np.ndarray,
    sim_rgb: np.ndarray,
    sim_mask: np.ndarray,
    alpha: float,
    status: str,
    fps_text: str,
    sam3_mask: np.ndarray | None = None,
    sam3_bbox: tuple[int, int, int, int] | None = None,
    sam3_label: str = "SAM3",
    sam3_candidate_bboxes: list[tuple[int, int, int, int, str]] | None = None,
) -> np.ndarray:
    real_base = np.asarray(real_rgb, dtype=np.uint8).copy()
    real = _draw_sam3_observation(real_base, sam3_mask, sam3_bbox, sam3_label, sam3_candidate_bboxes)
    sim = np.asarray(sim_rgb, dtype=np.uint8).copy()
    overlay = real_base.copy()
    mask = np.asarray(sim_mask > 0, dtype=bool)
    if np.any(mask):
        overlay[mask] = (
            (1.0 - float(alpha)) * overlay[mask].astype(np.float32)
            + float(alpha) * sim[mask].astype(np.float32)
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    labels = [
        ("Real wrist RGB", real),
        ("Sim wrist view", sim),
        ("Overlay", overlay),
    ]
    out = []
    for label, img in labels:
        panel = img.copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(panel, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.append(panel)
    strip = np.concatenate(out, axis=1)
    cv2.rectangle(strip, (0, strip.shape[0] - 34), (strip.shape[1], strip.shape[0]), (0, 0, 0), -1)
    cv2.putText(strip, status[:150], (12, strip.shape[0] - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    cv2.putText(strip, fps_text[:80], (strip.shape[1] - 360, strip.shape[0] - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 255, 130), 2)
    return strip


def _resize_for_display(img: np.ndarray, max_width: int) -> np.ndarray:
    max_width = int(max_width)
    if max_width <= 0 or img.shape[1] <= max_width:
        return img
    scale = float(max_width) / float(img.shape[1])
    return cv2.resize(img, (max_width, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def _save_bool_mask(path: Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.asarray(mask > 0, dtype=np.uint8) * 255)


def _save_mask_debug_overlay(rgb: np.ndarray, mask: np.ndarray, path: Path, *, label: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.asarray(rgb, dtype=np.uint8).copy()
    mask_bool = np.asarray(mask > 0, dtype=bool)
    overlay = canvas.copy()
    overlay[mask_bool] = (255, 40, 40)
    canvas = cv2.addWeighted(overlay, 0.42, canvas, 0.58, 0.0)
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(canvas, label[:90], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live wrist-camera three-view alignment: real RGB, simulated wrist view, and overlay.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--object-name", default="lvmukuai", help="Held object name. Default is the green block.")
    parser.add_argument("--prompt", default=None, help="SAM3 prompt override.")
    parser.add_argument(
        "--sam3-context-prompt",
        default="auto",
        help="SAM3 context prompt. auto asks for the object with a black occluder; none/off/object uses the object-only prompt.",
    )
    parser.add_argument(
        "--sam3-extra-prompts",
        type=str,
        nargs="*",
        default=["auto"],
        help="Extra SAM3 prompts. auto adds partially-occluded object variants; successful masks are unioned.",
    )
    parser.add_argument("--realsense-serial", default="", help="Wrist RealSense serial. Empty uses the first available device.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--period-s", type=float, default=1.0, help="Minimum time between pose refinement cycles.")
    parser.add_argument("--max-cycles", type=int, default=-1, help="Stop after this many refinement cycles; <=0 runs until q/Esc.")
    parser.add_argument("--output-dir", type=Path, default=RUNTIME_DIR / "wrist_live_overlay_runs")
    parser.add_argument("--relation-out", type=Path, default=RUNTIME_DIR / "wrist_live_overlay_latest_relation.json")
    parser.add_argument("--T-gripper-cam-path", type=Path, default=None, help="Eye-in-hand transform. Defaults to the newest ee_T_wrist_camera file.")
    parser.add_argument("--init-T-gripper-obj-path", type=Path, default=None, help="Explicit initial/previous held-object relation file.")
    parser.add_argument(
        "--initial-pose-source",
        choices=["default", "prior", "none"],
        default="default",
        help="Initial simulated object pose source before the first successful wrist refinement.",
    )
    parser.add_argument(
        "--default-gripper-obj-xyz-m",
        type=float,
        nargs=3,
        default=(0.020, 0.0, 0.020),
        metavar=("X", "Y", "Z"),
        help="Fallback T_gripper_obj translation used by --initial-pose-source default.",
    )
    parser.add_argument(
        "--default-gripper-obj-rpy-deg",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help="Fallback T_gripper_obj rotation used by --initial-pose-source default.",
    )
    parser.add_argument("--frame-npz", type=Path, default=None, help="Offline debug: reuse a saved RGB-D frame instead of opening RealSense.")
    parser.add_argument("--mask-path", type=Path, default=None, help="Optional object mask for offline/debug runs; skips SAM3 for that mask.")
    parser.add_argument("--exclude-mask-path", type=Path, default=None, help="Optional gripper/hand exclusion mask.")
    parser.add_argument("--subtract-gripper-mask", dest="subtract_gripper_mask", action="store_true", default=True)
    parser.add_argument("--no-subtract-gripper-mask", dest="subtract_gripper_mask", action="store_false")
    parser.add_argument("--pose-backend", choices=["gripper_plane", "rgbd", "rgb", "both", "benchmark"], default="gripper_plane")
    parser.add_argument(
        "--mask-refine-mode",
        choices=["none", "gripper", "split", "largest", "grabcut", "color", "depth", "hybrid", "auto", "benchmark"],
        default="gripper",
        help="Mask candidate to optimize. Wrist-held objects usually work best with gripper_subtracted.",
    )
    parser.add_argument("--mask-max-refine-candidates", type=int, default=1)
    parser.add_argument("--sam3-persistent", dest="sam3_persistent", action="store_true", default=True)
    parser.add_argument("--no-sam3-persistent", dest="sam3_persistent", action="store_false")
    parser.add_argument("--sam3-max-masks-per-item", type=int, default=5)
    parser.add_argument("--sam3-component-mode", choices=["largest", "all"], default="all")
    parser.add_argument("--sam3-split-component-min-pixels", type=int, default=160)
    parser.add_argument("--gripper-plane-refine-method", choices=["smooth", "search", "both"], default="smooth")
    parser.add_argument("--gripper-plane-opt-maxiter", type=int, default=45)
    parser.add_argument("--gripper-plane-opt-translation-bound-m", type=float, default=0.055)
    parser.add_argument("--gripper-plane-opt-rotation-bound-deg", type=float, default=45.0)
    parser.add_argument("--gripper-plane-opt-point-count", type=int, default=900)
    parser.add_argument("--gripper-plane-opt-projective-depth-weight", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--render-points", type=int, default=16000)
    parser.add_argument("--point-radius-px", type=int, default=2)
    parser.add_argument("--render-backend", choices=["sapien", "point"], default="sapien")
    parser.add_argument("--render-gripper", dest="render_gripper", action="store_true", default=True)
    parser.add_argument("--no-render-gripper", dest="render_gripper", action="store_false")
    parser.add_argument("--sim-gripper-exclude-mask", dest="sim_gripper_exclude_mask", action="store_true", default=True)
    parser.add_argument("--no-sim-gripper-exclude-mask", dest="sim_gripper_exclude_mask", action="store_false")
    parser.add_argument("--gripper-urdf", type=Path, default=DEFAULT_RM75_URDF)
    parser.add_argument("--gripper-joint-q", type=float, default=0.58, help="Visual opening value for the RM75 gripper joints.")
    parser.add_argument("--gripper-render-points-per-link", type=int, default=3500)
    parser.add_argument(
        "--scene-background-rgb",
        type=int,
        nargs=3,
        default=(232, 232, 232),
        metavar=("R", "G", "B"),
        help="Background color for the simulated wrist view.",
    )
    parser.add_argument("--display-max-width", type=int, default=1800)
    parser.add_argument("--no-window", action="store_true", help="Do not open an OpenCV window; still writes latest_strip.png.")
    parser.add_argument("--save-every-cycle", action="store_true", help="Save every three-view strip under the run directory.")
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--min-projection-iou", type=float, default=0.08)
    return parser.parse_args(argv)


def _auto_sam3_context_prompt(prompt: str) -> str:
    base = str(prompt or "").strip().rstrip(".")
    if base:
        return f"{base} with black occluder"
    return "held object and black robot gripper"


def _auto_sam3_extra_prompts(prompt: str) -> list[str]:
    base = str(prompt or "").strip().rstrip(".")
    if not base:
        return ["held object", "held object with black occluder", "partially hidden held object"]
    return [
        f"{base}",
        f"{base} partially hidden",
        f"{base} behind black object",
        f"{base} and black divider",
    ]


def _build_wrist_args(
    args: argparse.Namespace,
    *,
    frame_npz: Path,
    run_dir: Path,
    latest_T_cam_obj: np.ndarray | None,
    dynamic_exclude_mask_path: Path | None = None,
) -> argparse.Namespace:
    wrist_args = _legacy_default_args()
    wrist_args.object_name = str(args.object_name)
    wrist_args.prompt = args.prompt
    context_prompt = str(getattr(args, "sam3_context_prompt", "") or "").strip()
    if context_prompt.lower() == "auto":
        wrist_args.sam3_context_prompt = _auto_sam3_context_prompt(str(args.prompt))
    elif context_prompt.lower() in {"", "none", "off", "object", "false", "0"}:
        wrist_args.sam3_context_prompt = None
    else:
        wrist_args.sam3_context_prompt = context_prompt
    extra_prompts: list[str] = []
    for extra in list(getattr(args, "sam3_extra_prompts", None) or []):
        text = str(extra).strip()
        if not text or text.lower() in {"none", "off", "false", "0"}:
            continue
        if text.lower() == "auto":
            extra_prompts.extend(_auto_sam3_extra_prompts(str(args.prompt)))
        else:
            extra_prompts.append(text)
    wrist_args.sam3_extra_prompts = extra_prompts
    wrist_args.output_dir = run_dir
    wrist_args.relation_out = Path(args.relation_out).expanduser()
    wrist_args.realsense_serial = ""
    wrist_args.warmup_frames = 0
    wrist_args.frame_npz = Path(frame_npz)
    wrist_args.mask_path = Path(args.mask_path).expanduser() if args.mask_path else None
    if dynamic_exclude_mask_path is not None:
        wrist_args.exclude_mask_path = Path(dynamic_exclude_mask_path)
    else:
        wrist_args.exclude_mask_path = Path(args.exclude_mask_path).expanduser() if args.exclude_mask_path else None
    wrist_args.subtract_gripper_mask = bool(args.subtract_gripper_mask)
    wrist_args.T_gripper_cam_path = Path(args.T_gripper_cam_path).expanduser() if args.T_gripper_cam_path else latest_hand_eye_path()
    if args.init_T_gripper_obj_path:
        wrist_args.init_T_gripper_obj_path = Path(args.init_T_gripper_obj_path).expanduser()
    elif str(args.initial_pose_source) == "prior":
        wrist_args.init_T_gripper_obj_path = _latest_relation_prior(str(args.object_name))
    else:
        wrist_args.init_T_gripper_obj_path = None
    wrist_args.init_T_cam_obj_path = None
    wrist_args.pose_backend = str(args.pose_backend)
    wrist_args.mask_refine_mode = str(args.mask_refine_mode)
    wrist_args.mask_max_refine_candidates = int(args.mask_max_refine_candidates)
    wrist_args.sam3_max_masks_per_item = int(args.sam3_max_masks_per_item)
    wrist_args.sam3_component_mode = str(args.sam3_component_mode)
    wrist_args.sam3_split_component_min_pixels = int(args.sam3_split_component_min_pixels)
    wrist_args.gripper_plane_refine_method = str(args.gripper_plane_refine_method)
    wrist_args.gripper_plane_opt_maxiter = int(args.gripper_plane_opt_maxiter)
    wrist_args.gripper_plane_opt_translation_bound_m = float(args.gripper_plane_opt_translation_bound_m)
    wrist_args.gripper_plane_opt_rotation_bound_deg = float(args.gripper_plane_opt_rotation_bound_deg)
    wrist_args.gripper_plane_opt_point_count = int(args.gripper_plane_opt_point_count)
    wrist_args.gripper_plane_opt_projective_depth_weight = float(args.gripper_plane_opt_projective_depth_weight)
    wrist_args.min_confidence = float(args.min_confidence)
    wrist_args.min_projection_iou = float(args.min_projection_iou)
    wrist_args.continuous = False
    wrist_args.sam3_persistent = bool(args.sam3_persistent)
    if wrist_args.T_gripper_cam_path is None:
        raise FileNotFoundError("No wrist hand-eye transform found. Run calib-wrist or pass --T-gripper-cam-path.")
    return wrist_args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    object_name = normalize_object_name(args.object_name) or str(args.object_name)
    args.object_name = object_name
    prompt = args.prompt
    if prompt is None:
        spec = get_object_spec(object_name)
        prompt = getattr(spec, "grounding_prompt", None) if spec is not None else None
    if prompt is None:
        prompt = object_name
    args.prompt = str(prompt)

    run_root = _timestamp_dir(args.output_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    mesh, mesh_path, mesh_scale = _load_app_mesh(object_name, use_sim_asset=False)
    render_points = _sample_render_points(mesh, int(args.render_points))
    color = OBJECT_COLORS_RGB.get(object_name, (90, 190, 255))
    hand_eye = Path(args.T_gripper_cam_path).expanduser() if args.T_gripper_cam_path else latest_hand_eye_path()
    if hand_eye is None:
        raise FileNotFoundError("No wrist hand-eye transform found. Run calib-wrist or pass --T-gripper-cam-path.")
    T_gripper_cam, hand_eye_report = WRIST_LEGACY._load_gripper_T_cam(Path(hand_eye), "auto")
    if T_gripper_cam is None:
        raise FileNotFoundError(f"Could not load wrist hand-eye transform: {hand_eye}")

    explicit_prior = Path(args.init_T_gripper_obj_path).expanduser() if args.init_T_gripper_obj_path else None
    prior = explicit_prior
    if prior is None and str(args.initial_pose_source) == "prior":
        prior = _latest_relation_prior(object_name)

    latest_T_gripper_obj: np.ndarray | None = None
    pose_source = "none"
    if prior is not None:
        latest_T_gripper_obj = _load_T_gripper_obj(prior)
        pose_source = f"prior:{prior}"
    elif str(args.initial_pose_source) == "default":
        latest_T_gripper_obj = _make_transform(args.default_gripper_obj_xyz_m, args.default_gripper_obj_rpy_deg)
        pose_source = "default_gripper_center"
    elif str(args.initial_pose_source) == "prior":
        print(f"[wrist-live] no prior found for {object_name}; rendering gripper until first accepted pose")

    latest_T_cam_obj: np.ndarray | None = None
    if latest_T_gripper_obj is not None:
        latest_T_cam_obj = invert_transform(T_gripper_cam) @ latest_T_gripper_obj

    gripper_clouds: list[SceneCloud] = []
    if bool(args.render_gripper) and str(args.render_backend) == "point":
        gripper_clouds = _load_gripper_scene_clouds(
            urdf_path=Path(args.gripper_urdf),
            T_gripper_cam=T_gripper_cam,
            gripper_joint_q=float(args.gripper_joint_q),
            points_per_link=int(args.gripper_render_points_per_link),
            point_radius_px=int(args.point_radius_px),
        )

    print(f"[wrist-live] object={object_name} prompt={args.prompt!r}")
    print(f"[wrist-live] mesh={mesh_path} scale={mesh_scale:.6f} points={len(render_points)}")
    print(f"[wrist-live] hand_eye={hand_eye} report={hand_eye_report}")
    print(f"[wrist-live] initial_pose_source={pose_source}")
    if latest_T_gripper_obj is not None:
        print(f"[wrist-live] initial T_gripper_obj xyz={np.round(latest_T_gripper_obj[:3, 3], 4).tolist()}")
    print(f"[wrist-live] output={run_root}")
    _write_json(
        run_root / "config.json",
        {
            "args": vars(args),
            "object_name": object_name,
            "prompt": args.prompt,
            "mesh_path": mesh_path,
            "mesh_scale": mesh_scale,
            "hand_eye": hand_eye,
            "hand_eye_report": hand_eye_report,
            "T_gripper_cam": T_gripper_cam,
            "prior": prior,
            "initial_pose_source": pose_source,
            "initial_T_gripper_obj": latest_T_gripper_obj,
            "rendered_gripper_clouds": len(gripper_clouds),
        },
    )

    stream = None
    if args.frame_npz is None:
        stream = RealSenseStream(
            serial=str(args.realsense_serial or ""),
            width=int(args.width),
            height=int(args.height),
            fps=int(args.fps),
            warmup_frames=int(args.warmup_frames),
        )

    sam3_client = None
    if bool(args.sam3_persistent):
        print("[sam3] starting persistent server")
        sam3_client = PersistentSam3Client(
            max_masks_per_item=int(args.sam3_max_masks_per_item),
            component_mode=str(args.sam3_component_mode),
        )
        print(f"[sam3] ready load_ms={float(sam3_client.ready.get('load_elapsed_ms', 0.0)):.1f}")

    latest_status = f"initial pose_source={pose_source}"
    latest_sam3_mask: np.ndarray | None = None
    latest_sam3_bbox: tuple[int, int, int, int] | None = None
    latest_sam3_label = "SAM3 waiting"
    latest_sam3_candidate_bboxes: list[tuple[int, int, int, int, str]] = []
    cycle = 0
    t_last = time.perf_counter()
    sapien_renderer: SapienWristSceneRenderer | None = None
    try:
        while True:
            t0 = time.perf_counter()
            if stream is not None:
                rgb, depth, intr = stream.read()
            else:
                rgb, depth, intr = _load_frame_npz(Path(args.frame_npz))
            cycle_dir = run_root / f"cycle_{cycle:04d}"
            if str(args.render_backend) == "sapien" and sapien_renderer is None:
                try:
                    sapien_renderer = SapienWristSceneRenderer(
                        object_name=object_name,
                        object_mesh_path=mesh_path,
                        object_scale=float(mesh_scale),
                        object_color_rgb=color,
                        urdf_path=Path(args.gripper_urdf),
                        T_gripper_cam=T_gripper_cam,
                        intr=intr,
                        gripper_joint_q=float(args.gripper_joint_q),
                        render_gripper=bool(args.render_gripper),
                    )
                    print(f"[wrist-live] render_backend=sapien size={intr.width}x{intr.height}")
                except Exception as exc:
                    print(f"[warn] sapien renderer failed; falling back to point renderer: {type(exc).__name__}: {exc}")
                    args.render_backend = "point"
                    if bool(args.render_gripper) and not gripper_clouds:
                        gripper_clouds = _load_gripper_scene_clouds(
                            urdf_path=Path(args.gripper_urdf),
                            T_gripper_cam=T_gripper_cam,
                            gripper_joint_q=float(args.gripper_joint_q),
                            points_per_link=int(args.gripper_render_points_per_link),
                            point_radius_px=int(args.point_radius_px),
                        )
            dynamic_exclude_mask_path: Path | None = None
            gripper_exclude_pixels = 0
            if bool(args.subtract_gripper_mask) and bool(args.sim_gripper_exclude_mask):
                gripper_only_rgb: np.ndarray | None = None
                gripper_only_mask: np.ndarray | None = None
                if str(args.render_backend) == "sapien" and sapien_renderer is not None:
                    gripper_only_rgb, gripper_only_mask = sapien_renderer.render_gripper_only()
                elif gripper_clouds:
                    gripper_only_rgb, gripper_only_mask = _render_scene_from_wrist(
                        clouds=list(gripper_clouds),
                        K=intr.K,
                        width=int(intr.width),
                        height=int(intr.height),
                        background_rgb=tuple(int(x) for x in args.scene_background_rgb),
                    )
                if gripper_only_mask is not None:
                    gripper_only_mask = np.asarray(gripper_only_mask > 0, dtype=bool)
                    gripper_exclude_pixels = int(np.count_nonzero(gripper_only_mask))
                    if gripper_exclude_pixels > 0:
                        dynamic_exclude_mask_path = cycle_dir / "sim_gripper_exclude_mask.png"
                        _save_bool_mask(dynamic_exclude_mask_path, gripper_only_mask)
                        _save_mask_debug_overlay(
                            rgb,
                            gripper_only_mask,
                            cycle_dir / "sim_gripper_exclude_overlay.png",
                            label=f"sim gripper exclude pixels={gripper_exclude_pixels}",
                        )
            frame_npz = _save_frame_npz(cycle_dir, rgb, depth, intr)
            wrist_args = _build_wrist_args(
                args,
                frame_npz=frame_npz,
                run_dir=cycle_dir,
                latest_T_cam_obj=latest_T_cam_obj,
                dynamic_exclude_mask_path=dynamic_exclude_mask_path,
            )
            try:
                result = WRIST_LEGACY.run_once(
                    args=wrist_args,
                    run_dir=cycle_dir,
                    object_name=object_name,
                    prompt=str(args.prompt),
                    sam3_client=sam3_client,
                    T_cam_prior_override=latest_T_cam_obj,
                )
                if result.ok and result.T_cam_obj is not None:
                    latest_T_cam_obj = np.asarray(result.T_cam_obj, dtype=np.float64).reshape(4, 4)
                    if result.T_gripper_obj is not None:
                        latest_T_gripper_obj = np.asarray(result.T_gripper_obj, dtype=np.float64).reshape(4, 4)
                        latest_T_cam_obj = invert_transform(T_gripper_cam) @ latest_T_gripper_obj
                    else:
                        latest_T_gripper_obj = T_gripper_cam @ latest_T_cam_obj
                    pose_source = "estimator"
                    latest_status = (
                        f"cycle={cycle} ok source={pose_source} conf={result.confidence:.3f} iou={result.projection_iou:.3f} "
                        f"exclude=sim:{gripper_exclude_pixels} xyz_cam={np.round(latest_T_cam_obj[:3, 3], 3).tolist()}"
                    )
                else:
                    latest_status = (
                        f"cycle={cycle} rejected source={pose_source} conf={result.confidence:.3f} iou={result.projection_iou:.3f} "
                        f"exclude=sim:{gripper_exclude_pixels} reason={result.reason}; holding previous pose"
                    )
                latest_sam3_mask, latest_sam3_bbox, latest_sam3_label, latest_sam3_candidate_bboxes = _extract_sam3_overlay(
                    result.debug,
                    rgb.shape[:2],
                )
            except Exception as exc:
                latest_status = f"cycle={cycle} estimator_error source={pose_source} {type(exc).__name__}: {exc}; holding previous pose"
                print(f"[warn] {latest_status}")

            if str(args.render_backend) == "sapien" and sapien_renderer is not None:
                sim_rgb, sim_mask = sapien_renderer.render(T_gripper_obj=latest_T_gripper_obj)
            else:
                scene_clouds = list(gripper_clouds)
                if latest_T_cam_obj is not None:
                    scene_clouds.append(
                        SceneCloud(
                            name=object_name,
                            points=render_points,
                            T_cam_cloud=latest_T_cam_obj,
                            color_rgb=color,
                            radius_px=int(args.point_radius_px),
                            shade=True,
                        )
                    )
                sim_rgb, sim_mask = _render_scene_from_wrist(
                    clouds=scene_clouds,
                    K=intr.K,
                    width=int(intr.width),
                    height=int(intr.height),
                    background_rgb=tuple(int(x) for x in args.scene_background_rgb),
                )
            elapsed = time.perf_counter() - t0
            fps_text = f"dt={elapsed:.2f}s rate={1.0 / max(elapsed, 1e-6):.2f}Hz"
            strip = _compose_three_view(
                real_rgb=rgb,
                sim_rgb=sim_rgb,
                sim_mask=sim_mask,
                alpha=float(args.alpha),
                status=latest_status,
                fps_text=fps_text,
                sam3_mask=latest_sam3_mask,
                sam3_bbox=latest_sam3_bbox,
                sam3_label=latest_sam3_label,
                sam3_candidate_bboxes=latest_sam3_candidate_bboxes,
            )
            latest_strip = run_root / "latest_strip.png"
            cv2.imwrite(str(latest_strip), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
            if bool(args.save_every_cycle):
                cv2.imwrite(str(cycle_dir / "three_view_strip.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
            if latest_T_gripper_obj is not None:
                _write_json(
                    run_root / "latest_state.json",
                    {
                        "cycle": cycle,
                        "status": latest_status,
                        "pose_source": pose_source,
                        "render_backend": str(args.render_backend),
                        "sim_gripper_exclude_mask_path": dynamic_exclude_mask_path,
                        "sim_gripper_exclude_pixels": gripper_exclude_pixels,
                        "T_cam_obj": latest_T_cam_obj,
                        "T_gripper_cam": T_gripper_cam,
                        "T_gripper_obj": latest_T_gripper_obj,
                        "latest_strip": latest_strip,
                    },
                )
            if not bool(args.no_window):
                display = _resize_for_display(strip, int(args.display_max_width))
                cv2.imshow("wrist live overlay: real | sim | overlay", cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
            cycle += 1
            if int(args.max_cycles) > 0 and cycle >= int(args.max_cycles):
                break
            sleep_s = max(0.0, float(args.period_s) - (time.perf_counter() - t0))
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            t_last = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[wrist-live] stopped")
    finally:
        if sam3_client is not None:
            sam3_client.close()
        if stream is not None:
            stream.close()
        if not bool(args.no_window):
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
