from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rm75_app.paths import DEFAULT_RM75_URDF, RUNTIME_DIR


DEFAULT_JOINT_NAMES = [f"joint_{idx}" for idx in range(1, 8)]
DEFAULT_EE_LINK_CANDIDATES = ("gripper_tcp", "link_7", "Link7", "link_6")


def calibration_run_dir(kind: str, output_root: Path | None = None) -> Path:
    base = Path(output_root) if output_root is not None else RUNTIME_DIR / "calibration_runs"
    path = base.expanduser().resolve() / f"{datetime.now():%Y%m%d_%H%M%S}_{kind}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    with open(Path(path).expanduser(), "r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(to_jsonable(data), fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def as_transform(value: Any) -> np.ndarray:
    mat = np.asarray(value, dtype=np.float64).reshape(4, 4)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = mat[:3, :3]
    out[:3, 3] = mat[:3, 3]
    return out


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    out[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return out


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = as_transform(transform)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = transform[:3, :3].T
    out[:3, 3] = -(out[:3, :3] @ transform[:3, 3])
    return out


def rvec_tvec_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    return make_transform(rotation, np.asarray(tvec, dtype=np.float64).reshape(3))


def transform_to_rvec_tvec(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = as_transform(transform)
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    return rvec.reshape(3), transform[:3, 3].copy()


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    r = as_transform(a)[:3, :3] @ as_transform(b)[:3, :3].T
    cos_angle = np.clip((float(np.trace(r)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_angle)))


def pose_error(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = as_transform(a)
    b = as_transform(b)
    return {
        "translation_m": float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
        "rotation_deg": rotation_error_deg(a, b),
    }


def average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("cannot average an empty transform list")
    mats = [as_transform(t) for t in transforms]
    translations = np.stack([t[:3, 3] for t in mats])
    rot_sum = np.zeros((3, 3), dtype=np.float64)
    for mat in mats:
        rot_sum += mat[:3, :3]
    u, _, vt = np.linalg.svd(rot_sum)
    rot = u @ vt
    if np.linalg.det(rot) < 0.0:
        u[:, -1] *= -1.0
        rot = u @ vt
    return make_transform(rot, np.median(translations, axis=0))


def load_matrix(path: str | Path) -> np.ndarray:
    path = Path(path).expanduser()
    if path.suffix == ".npy":
        return as_transform(np.load(path))
    data = load_json(path)
    if "matrix" in data:
        return as_transform(data["matrix"])
    return as_transform(data)


def save_matrix_pair(path: str | Path, matrix: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, as_transform(matrix))
    save_json(path.with_suffix(".json"), {"matrix": as_transform(matrix)})


def camera_opencv_to_base_camera(matrix: np.ndarray, *, use_direct: bool = False) -> np.ndarray:
    mat = as_transform(matrix)
    return mat if use_direct else invert_transform(mat)


def load_realman_settings(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"calibration config not found: {path}")
    data = load_json(path)
    settings = data.get("realman_settings", data)
    if not isinstance(settings, dict):
        raise TypeError(f"expected object in {path}")
    return settings


def qpos_samples_from_settings(
    settings: dict[str, Any],
    *,
    key: str,
    fallback_key: str = "reference_qpos_deg",
) -> list[np.ndarray]:
    raw = settings.get(key, settings.get(fallback_key, []))
    samples = []
    for sample in raw:
        samples.append(np.deg2rad(np.asarray(sample, dtype=np.float64).reshape(-1)))
    return samples


def image_write(path: str | Path, image_bgr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"failed to write image: {path}")


@dataclass
class CameraFrame:
    color_bgr: np.ndarray
    depth_raw: np.ndarray | None
    intrinsic: np.ndarray
    dist_coeffs: np.ndarray


class RealSenseCamera:
    def __init__(
        self,
        *,
        serial: str | None = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        enable_depth: bool = False,
        warmup_frames: int = 20,
        auto_exposure: bool | None = None,
        exposure_us: float | None = None,
        gain: float | None = None,
    ) -> None:
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.enable_depth = bool(enable_depth)
        self.warmup_frames = int(max(warmup_frames, 0))
        self.auto_exposure = auto_exposure
        self.exposure_us = exposure_us
        self.gain = gain
        self.pipeline = None
        self.profile = None
        self.align = None

    def start(self) -> None:
        if self.pipeline is not None:
            return
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        cfg = rs.config()
        if self.serial:
            cfg.enable_device(str(self.serial))
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.enable_depth:
            cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        try:
            self.profile = pipeline.start(cfg)
            self.pipeline = pipeline
            self.align = rs.align(rs.stream.color) if self.enable_depth else None
            self._apply_color_options()
            for _ in range(self.warmup_frames):
                self.pipeline.wait_for_frames()
        except Exception:
            try:
                pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
            self.profile = None
            self.align = None
            raise

    def _apply_color_options(self) -> None:
        if self.profile is None:
            return
        import pyrealsense2 as rs

        for sensor in self.profile.get_device().query_sensors():
            if self.auto_exposure is not None and sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 1.0 if self.auto_exposure else 0.0)
            if self.exposure_us is not None and sensor.supports(rs.option.exposure):
                sensor.set_option(rs.option.exposure, float(self.exposure_us))
            if self.gain is not None and sensor.supports(rs.option.gain):
                sensor.set_option(rs.option.gain, float(self.gain))

    def stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError as exc:
                if "before start" not in str(exc):
                    raise
        self.pipeline = None
        self.profile = None
        self.align = None

    def capture(self) -> CameraFrame:
        if self.pipeline is None:
            self.start()
        import pyrealsense2 as rs

        frames = self.pipeline.wait_for_frames()
        if self.align is not None:
            frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("failed to fetch RealSense color frame")
        depth_frame = frames.get_depth_frame() if self.enable_depth else None
        color = np.asanyarray(color_frame.get_data()).copy()
        depth = None if depth_frame is None else np.asanyarray(depth_frame.get_data()).copy()
        intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        intrinsic = np.asarray(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist = np.asarray(intr.coeffs, dtype=np.float64).reshape(-1)
        return CameraFrame(color_bgr=color, depth_raw=depth, intrinsic=intrinsic, dist_coeffs=dist)


class RealmanSdkController:
    def __init__(
        self,
        *,
        robot_ip: str,
        robot_port: int = 8080,
        joint_names: list[str] | None = None,
        calibration_offset: dict[str, float] | None = None,
        joint_tolerance_deg: float = 0.4,
        connection_retries: int = 3,
    ) -> None:
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

        self.joint_names = list(joint_names or DEFAULT_JOINT_NAMES)
        self.calibration_offset = {
            name: float((calibration_offset or {}).get(name, 0.0)) for name in self.joint_names
        }
        self.joint_tolerance_deg = float(joint_tolerance_deg)
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = None
        last_exc = None
        for _ in range(max(1, int(connection_retries))):
            try:
                handle = self.arm.rm_create_robot_arm(str(robot_ip), int(robot_port))
                if handle is not None and getattr(handle, "id", 0) > 0:
                    break
            except Exception as exc:
                last_exc = exc
            time.sleep(1.0)
        if handle is None or getattr(handle, "id", 0) <= 0:
            raise RuntimeError(f"failed to connect Realman arm at {robot_ip}:{robot_port}") from last_exc

    def disconnect(self) -> None:
        try:
            self.arm.rm_delete_robot_arm()
        except Exception:
            pass

    def _apply_offset(self, joints_deg: np.ndarray, *, subtract: bool) -> np.ndarray:
        offsets = np.asarray([self.calibration_offset.get(name, 0.0) for name in self.joint_names])
        return joints_deg - offsets if subtract else joints_deg + offsets

    def _read_state(self, attempts: int = 3):
        last_code = None
        for _ in range(max(1, int(attempts))):
            ret, state = self.arm.rm_get_current_arm_state()
            if int(ret) == 0:
                return state
            last_code = ret
            time.sleep(0.25)
        raise RuntimeError(f"rm_get_current_arm_state failed with code {last_code}")

    def get_qpos(self, *, flat: bool = True) -> np.ndarray | dict[str, float]:
        state = self._read_state()
        joints_deg = self._apply_offset(np.asarray(state["joint"], dtype=np.float64), subtract=True)
        qpos = np.deg2rad(joints_deg)
        if flat:
            return qpos
        return {name: float(qpos[idx]) for idx, name in enumerate(self.joint_names)}

    def move_to_qpos(self, qpos_rad: np.ndarray, *, timeout_s: float = 25.0) -> None:
        target_deg = self._apply_offset(np.rad2deg(np.asarray(qpos_rad, dtype=np.float64)), subtract=False)
        ret = self.arm.rm_movej_follow(target_deg.tolist())
        if int(ret) != 0:
            raise RuntimeError(f"rm_movej_follow failed with code {ret}")
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            state = self._read_state()
            current_deg = np.asarray(state["joint"], dtype=np.float64)
            if float(np.linalg.norm(current_deg - target_deg)) <= self.joint_tolerance_deg:
                return
            time.sleep(0.2)
        raise TimeoutError("timed out waiting for Realman arm motion")


def load_urdf(urdf_path: str | Path | None = None):
    path = Path(urdf_path or DEFAULT_RM75_URDF).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"URDF not found: {path}")
    try:
        from urchin import URDF

        return URDF.load(str(path))
    except ModuleNotFoundError:
        from yourdfpy import URDF

        return URDF.load(str(path))


def link_poses_from_qpos(
    urdf,
    qpos_rad: np.ndarray,
    joint_names: list[str],
    extra_joint_values: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    qpos = np.asarray(qpos_rad, dtype=np.float64).reshape(-1)
    cfg = {}
    for idx, name in enumerate(joint_names):
        if idx < qpos.size:
            cfg[name] = float(qpos[idx])
    if extra_joint_values:
        for name, value in extra_joint_values.items():
            cfg[str(name)] = float(value)
    if hasattr(urdf, "link_fk"):
        for joint_name in urdf.joint_map.keys():
            cfg.setdefault(joint_name, 0.0)
        poses = urdf.link_fk(cfg=cfg, use_names=True)
        return {str(name): as_transform(value) for name, value in poses.items()}
    if hasattr(urdf, "update_cfg") and hasattr(urdf, "link_map"):
        urdf.update_cfg(cfg)
        poses = {}
        for link_name in urdf.link_map.keys():
            transform, _ = urdf.scene.graph.get(link_name)
            poses[str(link_name)] = as_transform(transform)
        return poses
    raise TypeError(f"unsupported URDF object type: {type(urdf)!r}")


def select_link_pose(link_poses: dict[str, np.ndarray], requested: str | None = None) -> tuple[str, np.ndarray]:
    if requested:
        if requested not in link_poses:
            raise KeyError(f"link {requested!r} not found. Available links: {sorted(link_poses)}")
        return requested, as_transform(link_poses[requested])
    for candidate in DEFAULT_EE_LINK_CANDIDATES:
        if candidate in link_poses:
            return candidate, as_transform(link_poses[candidate])
    name = sorted(link_poses)[-1]
    return name, as_transform(link_poses[name])
