import copy
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import tyro

from lerobot_sim2real.utils.safety import setup_safe_exit
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from lerobot_sim2real.config.real_robot import create_real_robot
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.utils.visualization.misc import tile_images
from mani_skill.utils import sapien_utils

try:
    import sapien.core as sapien
except Exception:
    sapien = None

try:
    import pyrealsense2 as rs  # optional
except Exception:
    rs = None


def to_numpy_image(image):
    if torch.is_tensor(image):
        return image.detach().cpu().numpy()
    return np.asarray(image)


@dataclass
class Args:
    env_id: str = "RM75GraspCube_two_cameras-v1"
    env_kwargs_json_path: Optional[str] = "../../so101_env_config.json"

    # Sim sensor resolution (match your real camera for alignment)
    width: int = 640
    height: int = 480

    # Intrinsics source (relative to this script if not absolute)
    intrinsics_json_path: str = "camera_pose_and_fov.json"
    query_realsense_if_needed: bool = True  # try to query RS to补全 cx/cy/dist

    # Real image processing
    undistort_real: bool = True
    undistort_border_value: float = 0.0
    undistort_interpolation: str = "linear"  # "linear" or "nearest"

    # Loss masking
    ignore_black_border: bool = True

    # Runtime
    render_backend: str = "cpu"  # "cpu" or "cuda" depending on install


@dataclass
class RSIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: np.ndarray  # OpenCV order: [k1,k2,p1,p2,k3,(...)]
    model: str = "brown_conrady"


def _resolve_path(path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, path_str)


def _load_intrinsics_from_json(path_str: str) -> Optional[RSIntrinsics]:
    path = _resolve_path(path_str)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return None

    intr = data.get("realsense_intrinsics", {}) or {}
    fx = float(intr.get("fx", 0.0))
    fy = float(intr.get("fy", 0.0))
    w = int(intr.get("width", 0))
    h = int(intr.get("height", 0))

    # principal point：json 没存就先默认中心
    cx = float(intr.get("cx", (w - 1) * 0.5 if w > 0 else 0.0))
    cy = float(intr.get("cy", (h - 1) * 0.5 if h > 0 else 0.0))

    dist = intr.get("dist", data.get("dist", None))
    if dist is None:
        dist_arr = np.zeros((5,), dtype=np.float64)
    else:
        dist_arr = np.asarray(dist, dtype=np.float64).reshape(-1)

    model = str(intr.get("model", data.get("model", "brown_conrady")))

    if fx <= 0 or fy <= 0 or w <= 0 or h <= 0:
        return None
    return RSIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h, dist=dist_arr, model=model)


def _update_intrinsics_json(path_str: str, intr: RSIntrinsics) -> None:
    """把 fx/fy/cx/cy/dist 回写到 json，方便以后不用再 query。best-effort。"""
    path = _resolve_path(path_str)
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
        else:
            data = {}
        block = data.get("realsense_intrinsics", {}) or {}
        block.update(
            {
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "cx": float(intr.cx),
                "cy": float(intr.cy),
                "width": int(intr.width),
                "height": int(intr.height),
                "dist": [float(x) for x in np.asarray(intr.dist, dtype=np.float64).reshape(-1).tolist()],
                "model": str(intr.model),
            }
        )
        data["realsense_intrinsics"] = block
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[intrinsics] updated json: {path}")
    except Exception as e:
        print(f"[intrinsics] failed to update json {path}: {e}")


def _try_get_intrinsics_from_camera_obj(camera_obj) -> Optional[RSIntrinsics]:
    """
    尝试从 create_real_robot() 返回的 camera 对象里取内参/畸变（不同封装写法很多，做 best-effort）。
    """
    if camera_obj is None:
        return None

    # 1) ROS CameraInfo-like
    for attr in ["camera_info", "info", "cameraInfo"]:
        if hasattr(camera_obj, attr):
            info = getattr(camera_obj, attr)
            if isinstance(info, dict) and "K" in info:
                K = np.asarray(info["K"], dtype=np.float64).reshape(3, 3)
                D = np.asarray(info.get("D", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1)
                w = int(info.get("width", 0))
                h = int(info.get("height", 0))
                if w > 0 and h > 0:
                    return RSIntrinsics(
                        fx=float(K[0, 0]),
                        fy=float(K[1, 1]),
                        cx=float(K[0, 2]),
                        cy=float(K[1, 2]),
                        width=w,
                        height=h,
                        dist=D,
                        model=str(info.get("distortion_model", "brown_conrady")),
                    )

    # 2) wrapper 直接暴露 dict intrinsics
    for meth in ["get_intrinsics", "intrinsics", "get_color_intrinsics", "get_rgb_intrinsics"]:
        if hasattr(camera_obj, meth):
            try:
                intr = getattr(camera_obj, meth)()
            except TypeError:
                intr = getattr(camera_obj, meth)
            try:
                if isinstance(intr, dict):
                    fx = float(intr["fx"])
                    fy = float(intr["fy"])
                    w = int(intr["width"])
                    h = int(intr["height"])
                    cx = float(intr.get("cx", intr.get("ppx", (w - 1) * 0.5)))
                    cy = float(intr.get("cy", intr.get("ppy", (h - 1) * 0.5)))
                    dist = np.asarray(intr.get("dist", intr.get("D", [0, 0, 0, 0, 0])), dtype=np.float64).reshape(-1)
                    model = str(intr.get("model", "brown_conrady"))
                    return RSIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h, dist=dist, model=model)
            except Exception:
                pass

    # 3) pyrealsense2 profile / pipeline
    if rs is not None:
        for attr in ["profile", "active_profile", "pipeline_profile"]:
            if hasattr(camera_obj, attr):
                prof = getattr(camera_obj, attr)
                try:
                    vsp = prof.get_stream(rs.stream.color).as_video_stream_profile()
                    intr = vsp.get_intrinsics()
                    return RSIntrinsics(
                        fx=float(intr.fx),
                        fy=float(intr.fy),
                        cx=float(intr.ppx),
                        cy=float(intr.ppy),
                        width=int(intr.width),
                        height=int(intr.height),
                        dist=np.asarray(intr.coeffs, dtype=np.float64).reshape(-1),
                        model=str(intr.model),
                    )
                except Exception:
                    pass

        for attr in ["pipeline", "_pipeline"]:
            if hasattr(camera_obj, attr):
                pipe = getattr(camera_obj, attr)
                try:
                    prof = pipe.get_active_profile()
                    vsp = prof.get_stream(rs.stream.color).as_video_stream_profile()
                    intr = vsp.get_intrinsics()
                    return RSIntrinsics(
                        fx=float(intr.fx),
                        fy=float(intr.fy),
                        cx=float(intr.ppx),
                        cy=float(intr.ppy),
                        width=int(intr.width),
                        height=int(intr.height),
                        dist=np.asarray(intr.coeffs, dtype=np.float64).reshape(-1),
                        model=str(intr.model),
                    )
                except Exception:
                    pass

    return None


def _query_realsense_intrinsics(width: int, height: int) -> Optional[RSIntrinsics]:
    """
    直接用 pyrealsense2 临时打开设备读内参/畸变。
    注意：如果相机已被其它进程/管线占用，这一步会失败；失败不致命。
    """
    if rs is None:
        return None
    pipe = None
    try:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)
        prof = pipe.start(cfg)

        vsp = prof.get_stream(rs.stream.color).as_video_stream_profile()
        intr = vsp.get_intrinsics()

        out = RSIntrinsics(
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.ppx),
            cy=float(intr.ppy),
            width=int(intr.width),
            height=int(intr.height),
            dist=np.asarray(intr.coeffs, dtype=np.float64).reshape(-1),
            model=str(intr.model),
        )
        pipe.stop()
        return out
    except Exception:
        try:
            if pipe is not None:
                pipe.stop()
        except Exception:
            pass
        return None


def _scale_intrinsics(intr: RSIntrinsics, to_w: int, to_h: int) -> RSIntrinsics:
    """分辨率不一致时，按比例缩放 fx/fy/cx/cy（畸变系数保持不变）。"""
    if intr.width == to_w and intr.height == to_h:
        return intr
    sx = float(to_w) / float(intr.width)
    sy = float(to_h) / float(intr.height)
    return RSIntrinsics(
        fx=intr.fx * sx,
        fy=intr.fy * sy,
        cx=intr.cx * sx,
        cy=intr.cy * sy,
        width=to_w,
        height=to_h,
        dist=intr.dist.copy(),
        model=intr.model,
    )


def _apply_intrinsics_to_sim_camera(sim_env, intr: RSIntrinsics, focal_scale: float = 1.0, pp_offset_px=(0.0, 0.0)):
    cam = sim_env.unwrapped._sensors["base_camera"].camera
    fx = float(intr.fx) * float(focal_scale)
    fy = float(intr.fy) * float(focal_scale)
    cx = float(intr.cx) + float(pp_offset_px[0])
    cy = float(intr.cy) + float(pp_offset_px[1])

    if hasattr(cam, "set_perspective_parameters"):
        near = float(getattr(cam, "near", 0.01))
        far = float(getattr(cam, "far", 100.0))
        try:
            cam.set_perspective_parameters(near, far, fx, fy, cx, cy, 0.0)
            return
        except Exception:
            pass

    if hasattr(cam, "set_focal_lengths"):
        cam.set_focal_lengths(fx, fy)
    if hasattr(cam, "set_principal_point"):
        cam.set_principal_point(cx, cy)


@dataclass
class UndistortMaps:
    map1: np.ndarray
    map2: np.ndarray
    mask: np.ndarray  # bool HxW


def _build_undistort_maps(intr: RSIntrinsics) -> Optional[UndistortMaps]:
    dist = np.asarray(intr.dist, dtype=np.float64).reshape(-1)
    if dist.size == 0 or np.all(np.abs(dist) < 1e-12):
        return None

    w, h = intr.width, intr.height
    K = np.array(
        [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )

    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)
    valid = (map1 >= 0) & (map1 <= (w - 1)) & (map2 >= 0) & (map2 <= (h - 1))
    return UndistortMaps(map1=map1, map2=map2, mask=valid)


def _undistort_rgb(img_rgb_float01: np.ndarray, maps: UndistortMaps, interpolation: str, border_value: float) -> np.ndarray:
    if maps is None:
        return img_rgb_float01
    interp = cv2.INTER_LINEAR if interpolation == "linear" else cv2.INTER_NEAREST
    return cv2.remap(
        img_rgb_float01,
        maps.map1,
        maps.map2,
        interpolation=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_value),
    )


# -----------------------------
# Interactive camera controls
# -----------------------------
camera_offset = torch.zeros(3, dtype=torch.float32)
rpy_offset = torch.zeros(3, dtype=torch.float32)   # roll, pitch, yaw (rad) in camera LOCAL frame
pp_offset = torch.zeros(2, dtype=torch.float32)    # cx, cy pixel offsets
focal_scale = 1.0
active_keys = set()
last_frame_time = time.time()
help_message_printed = False

# Runtime toggle
undistort_enabled = True


def on_key_press(event):
    global active_keys, undistort_enabled
    if event.key == "m":
        undistort_enabled = not undistort_enabled
        print("[toggle] undistort_enabled =", undistort_enabled)
        return
    active_keys.add(event.key)


def on_key_release(event):
    global active_keys
    active_keys.discard(event.key)


def _quat_mul(q1, q2):
    """Hamilton product (w,x,y,z). Accepts q shape (4,) or (1,4) or nested."""
    q1 = np.asarray(q1, dtype=np.float64).reshape(-1)
    q2 = np.asarray(q2, dtype=np.float64).reshape(-1)

    # Sometimes pose.q comes as (1,4) -> reshape(-1) gives 4; but if deeper nested, handle once more
    if q1.size == 1 and isinstance(q1[0], (list, tuple, np.ndarray)):
        q1 = np.asarray(q1[0], dtype=np.float64).reshape(-1)
    if q2.size == 1 and isinstance(q2[0], (list, tuple, np.ndarray)):
        q2 = np.asarray(q2[0], dtype=np.float64).reshape(-1)

    if q1.size != 4 or q2.size != 4:
        raise ValueError(f"_quat_mul expects 4D quaternions. got q1.shape={q1.shape}, q2.shape={q2.shape}")

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )



def _quat_from_axis_angle(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / n
    half = 0.5 * float(angle_rad)
    s = math.sin(half)
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def _apply_local_rpy_offset(pose, roll_rad, pitch_rad, yaw_rad):
    """
    Apply extra rotation in CAMERA LOCAL frame:
      roll  about +Z
      pitch about +X
      yaw   about +Y
    """
    if sapien is None:
        return pose
    q = np.asarray(pose.q, dtype=np.float64)  # wxyz
    q_roll = _quat_from_axis_angle([0, 0, 1], roll_rad)
    q_pitch = _quat_from_axis_angle([1, 0, 0], pitch_rad)
    q_yaw = _quat_from_axis_angle([0, 1, 0], yaw_rad)

    q_off = _quat_mul(_quat_mul(q_yaw, q_pitch), q_roll)
    q_new = _quat_mul(q, q_off)

    # --- Ensure p and q are numpy float32 with correct shapes for sapien.Pose ---
    p = pose.p
    if torch.is_tensor(p):
        p = p.detach().cpu().numpy()
    p = np.asarray(p, dtype=np.float32).reshape(-1)
    if p.size == 3:
        p = p.reshape(3, )
    else:
        # e.g. (1,3) -> flatten then take first 3
        p = p.flatten()[:3].astype(np.float32)

    q_new = np.asarray(q_new, dtype=np.float32).reshape(4, )

    return sapien.Pose(p, q_new)


def update_camera(sim_env, base_intr: RSIntrinsics, args: Args):
    """
    Keys:
      Move camera pos: w/a/s/d + up/down
      Move target:     y/h (x-+), u/j (y+-), i/k (z+-)
      Roll: q/e   Pitch: r/f   Yaw: t/g       (2 deg/s)
      Zoom: left/right  (scales fx/fy)
      Principal point (px): z/x for cx-,cx+ ; c/v for cy-,cy+
      Reset: backspace
      Toggle undistort: m
    """
    global camera_offset, rpy_offset, pp_offset, focal_scale, last_frame_time, help_message_printed

    current_time = time.time()
    dt = max(current_time - last_frame_time, 1e-6)
    last_frame_time = current_time

    MOVE_M_PER_S = 0.10
    TARGET_M_PER_S = 0.10
    ROT_RAD_PER_S = math.radians(2.0)  # 2 deg/s
    PP_PX_PER_S = 50.0
    ZOOM_PER_S = 0.50

    if "backspace" in active_keys:
        camera_offset = torch.zeros(3, dtype=torch.float32)
        rpy_offset = torch.zeros(3, dtype=torch.float32)
        pp_offset = torch.zeros(2, dtype=torch.float32)
        focal_scale = 1.0

    step = float(MOVE_M_PER_S * dt)
    if "w" in active_keys:
        camera_offset[0] -= step
    if "s" in active_keys:
        camera_offset[0] += step
    if "d" in active_keys:
        camera_offset[1] += step
    if "a" in active_keys:
        camera_offset[1] -= step
    if "up" in active_keys:
        camera_offset[2] += step
    if "down" in active_keys:
        camera_offset[2] -= step

    camera_target_offset = torch.zeros(3, dtype=torch.float32)
    tstep = float(TARGET_M_PER_S * dt)
    if "y" in active_keys:
        camera_target_offset[0] -= tstep
    if "h" in active_keys:
        camera_target_offset[0] += tstep
    if "u" in active_keys:
        camera_target_offset[1] += tstep
    if "j" in active_keys:
        camera_target_offset[1] -= tstep
    if "i" in active_keys:
        camera_target_offset[2] += tstep
    if "k" in active_keys:
        camera_target_offset[2] -= tstep

    rstep = float(ROT_RAD_PER_S * dt)
    if "q" in active_keys:
        rpy_offset[0] -= rstep
    if "e" in active_keys:
        rpy_offset[0] += rstep
    if "r" in active_keys:
        rpy_offset[1] += rstep
    if "f" in active_keys:
        rpy_offset[1] -= rstep
    if "t" in active_keys:
        rpy_offset[2] += rstep
    if "g" in active_keys:
        rpy_offset[2] -= rstep

    if "left" in active_keys:
        focal_scale = max(0.1, focal_scale * (1.0 - float(ZOOM_PER_S * dt)))
    if "right" in active_keys:
        focal_scale = min(10.0, focal_scale * (1.0 + float(ZOOM_PER_S * dt)))

    pstep = float(PP_PX_PER_S * dt)
    if "z" in active_keys:
        pp_offset[0] -= pstep
    if "x" in active_keys:
        pp_offset[0] += pstep
    if "c" in active_keys:
        pp_offset[1] -= pstep
    if "v" in active_keys:
        pp_offset[1] += pstep

    base_pos = torch.tensor(sim_env.unwrapped.base_camera_settings["pos"], dtype=torch.float32)
    base_target = torch.tensor(sim_env.unwrapped.base_camera_settings["target"], dtype=torch.float32)

    pos = base_pos + camera_offset
    target = base_target + camera_target_offset
    sim_env.unwrapped.base_camera_settings["target"] = target

    pose = sapien_utils.look_at(pos, target)
    pose = _apply_local_rpy_offset(pose, float(rpy_offset[0]), float(rpy_offset[1]), float(rpy_offset[2]))
    sim_env.unwrapped.camera_mount.set_pose(pose)

    sim_cam = sim_env.unwrapped._sensors["base_camera"].camera
    sim_w = int(getattr(sim_cam, "width", args.width))
    sim_h = int(getattr(sim_cam, "height", args.height))
    intr_scaled = _scale_intrinsics(base_intr, sim_w, sim_h)
    _apply_intrinsics_to_sim_camera(sim_env, intr_scaled, focal_scale=focal_scale, pp_offset_px=(float(pp_offset[0]), float(pp_offset[1])))

    if len(active_keys) > 0:
        help_message_printed = False
        print("current_camera_position", pose.p)
        print("camera_target", sim_env.unwrapped.base_camera_settings["target"])
        print(
            "rpy_offset_deg (roll,pitch,yaw) =",
            float(rpy_offset[0]) * 180.0 / math.pi,
            float(rpy_offset[1]) * 180.0 / math.pi,
            float(rpy_offset[2]) * 180.0 / math.pi,
        )
        print("focal_scale =", float(focal_scale), "pp_offset(px) =", float(pp_offset[0]), float(pp_offset[1]))
    elif not help_message_printed:
        print("=== Commands (scheme 5: fx/fy/cx/cy + undistort) ===")
        print("Move camera pos: w/a/s/d + up/down")
        print("Move target:     y/h (x), u/j (y), i/k (z)")
        print("Roll: q/e   Pitch: r/f   Yaw: t/g   (2 deg/s)")
        print("Zoom (intrinsics): left/right  (scales fx/fy)")
        print("Principal point (px): z/x for cx-,cx+ ; c/v for cy-,cy+")
        print("Reset: backspace")
        print("Toggle undistort: m")
        print()
        help_message_printed = True


def _get_real_image_shape(real_env) -> Tuple[int, int]:
    obs = real_env.get_obs()["sensor_data"]
    rgb = to_numpy_image(obs["base_camera"]["rgb"][0])
    h, w = rgb.shape[:2]
    return w, h


def overlay_envs(sim_env, real_env, undistort_maps: Optional[UndistortMaps], args: Args):
    """
    Overlay sim on real. 注意：undistort_maps 默认只对 base_camera 生效，
    如果你有第二个真实相机，请你为它单独提供 intrinsics/dist 并扩展成 dict[name]->maps。
    """
    real_obs = real_env.get_obs()["sensor_data"]
    sim_obs = sim_env.get_obs()["sensor_data"]
    assert sorted(real_obs.keys()) == sorted(
        sim_obs.keys()
    ), f"real camera names {real_obs.keys()} and sim camera names {sim_obs.keys()} differ"

    overlaid_dict = sim_env.get_obs()["sensor_data"]
    overlaid_imgs = []
    sim_imgs_list = []
    real_imgs_list = []

    for name in overlaid_dict:
        real_rgb = to_numpy_image(real_obs[name]["rgb"][0]).astype(np.float32) / 255.0
        sim_rgb = to_numpy_image(overlaid_dict[name]["rgb"][0]).astype(np.float32) / 255.0

        if args.undistort_real and undistort_maps is not None and name == "base_camera":
            real_rgb_u = _undistort_rgb(real_rgb, undistort_maps, args.undistort_interpolation, args.undistort_border_value)
            mask = undistort_maps.mask if args.ignore_black_border else None
        else:
            real_rgb_u = real_rgb
            mask = None

        if mask is not None:
            m = mask[..., None].astype(np.float32)
            diff2 = ((real_rgb_u - sim_rgb) ** 2) * m
            denom = np.maximum(m.sum(), 1.0)
            loss = float(diff2.sum() / denom)
        else:
            loss = float(np.mean((real_rgb_u - sim_rgb) ** 2))

        print("current loss", loss)

        overlaid_imgs.append(0.5 * real_rgb_u + 0.5 * sim_rgb)
        sim_imgs_list.append(sim_rgb)
        real_imgs_list.append(real_rgb_u)

    return tile_images(overlaid_imgs), tile_images(sim_imgs_list), tile_images(real_imgs_list)


def main(args: Args):
    global undistort_enabled

    env_kwargs = dict(
        obs_mode="rgb+segmentation",
        render_mode="human",
        reward_mode="none",
        render_backend=args.render_backend,
        sensor_configs=dict(width=args.width, height=args.height),
    )

    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            env_kwargs.update(json.load(f))
        print("Loaded env_kwargs from JSON:", env_kwargs)

    sim_env = gym.make(args.env_id, **env_kwargs)
    sim_env = FlattenRGBDObservationWrapper(sim_env)

    real_robot, camera_obj = create_real_robot()
    real_agent = LeRobotRealAgent(real_robot)
    real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent)
    setup_safe_exit(sim_env, real_env, real_agent)

    # Reset once
    real_obs, _ = real_env.reset()

    real_w, real_h = _get_real_image_shape(real_env)
    print(f"[real] image size: {real_w}x{real_h}")

    # -------- Intrinsics auto --------
    base_intr = _try_get_intrinsics_from_camera_obj(camera_obj)
    if base_intr is None:
        base_intr = _load_intrinsics_from_json(args.intrinsics_json_path)
    if base_intr is None and args.query_realsense_if_needed:
        base_intr = _query_realsense_intrinsics(real_w, real_h)

    if base_intr is None:
        base_intr = RSIntrinsics(
            fx=0.5 * real_w,
            fy=0.5 * real_h,
            cx=(real_w - 1) * 0.5,
            cy=(real_h - 1) * 0.5,
            width=real_w,
            height=real_h,
            dist=np.zeros((5,), dtype=np.float64),
            model="none",
        )

    # 如果目前 dist 全 0 或者 cx/cy 只是默认中心点，就再尝试 query 一次补全
    if args.query_realsense_if_needed and rs is not None and base_intr is not None:
        dist_is_zero = np.all(np.abs(np.asarray(base_intr.dist, dtype=np.float64)) < 1e-12)
        pp_is_center = (abs(base_intr.cx - (real_w - 1) * 0.5) < 1e-3) and (abs(base_intr.cy - (real_h - 1) * 0.5) < 1e-3)
        if dist_is_zero or pp_is_center:
            rs_intr = _query_realsense_intrinsics(real_w, real_h)
            if rs_intr is not None:
                base_intr = rs_intr
                print("[intrinsics] refreshed from RealSense device (got ppx/ppy and distortion).")
            else:
                print("[intrinsics] RealSense query failed (camera may already be in use). Proceeding with existing intrinsics.")

    # Scale intrinsics to real frame size
    base_intr = _scale_intrinsics(base_intr, real_w, real_h)

    # Persist intrinsics (including cx/cy/dist) back to JSON
    _update_intrinsics_json(args.intrinsics_json_path, base_intr)

    print("[intrinsics] fx fy cx cy =", base_intr.fx, base_intr.fy, base_intr.cx, base_intr.cy)
    print("[intrinsics] dist =", base_intr.dist, "model =", base_intr.model)

    # Apply to sim camera
    sim_cam = sim_env.unwrapped._sensors["base_camera"].camera
    sim_w = int(getattr(sim_cam, "width", args.width))
    sim_h = int(getattr(sim_cam, "height", args.height))
    intr_for_sim = _scale_intrinsics(base_intr, sim_w, sim_h)
    _apply_intrinsics_to_sim_camera(sim_env, intr_for_sim, focal_scale=1.0, pp_offset_px=(0.0, 0.0))

    # Build undistortion maps
    maps = _build_undistort_maps(base_intr) if args.undistort_real else None
    if maps is None:
        print("[undistort] disabled or no distortion coefficients (will compare raw images).")
        undistort_enabled = False
    else:
        print("[undistort] maps built (press 'm' to toggle during runtime).")
        undistort_enabled = True

    # -------- Visualization --------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    fig.canvas.manager.key_press_handler_id = None

    overlaid_imgs, sim_imgs, real_imgs = overlay_envs(sim_env, real_env, maps if undistort_enabled else None, args)
    axes[0].set_title("Real (undistorted if enabled)")
    axes[1].set_title("Sim")
    axes[2].set_title("Overlay")
    im_real = axes[0].imshow(real_imgs)
    im_sim = axes[1].imshow(sim_imgs)
    im_overlay = axes[2].imshow(overlaid_imgs)

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)

    # Optional: fixed pose for easier alignment
    fixed_qpos = torch.tensor(
        [[0, 0.0, 0.0, -1.5708, 0.0, -1.5708, 1.0472]],
        dtype=torch.float32,
        device=sim_env.agent.robot.get_qpos().device,
    )
    sim_qpos = sim_env.agent.robot.get_qpos()
    if fixed_qpos.shape[-1] <= sim_qpos.shape[-1]:
        sim_qpos[..., : fixed_qpos.shape[-1]] = fixed_qpos
    else:
        sim_qpos = fixed_qpos
    sim_env.agent.robot.set_qpos(sim_qpos)
    print("Set fixed sim qpos:", sim_qpos)

    fig.canvas.draw()
    fig.show()
    fig.canvas.flush_events()
    print("Close the figure window to exit.")

    while True:
        current_qpos = real_agent.get_rad()
        sim_qpos = sim_env.agent.robot.get_qpos()
        if current_qpos.shape[-1] < sim_qpos.shape[-1]:
            padded = sim_qpos.clone()
            padded[..., : current_qpos.shape[-1]] = current_qpos.to(sim_qpos.device)
            sim_env.agent.robot.set_qpos(padded)
        else:
            sim_env.agent.robot.set_qpos(current_qpos[..., : sim_qpos.shape[-1]])

        update_camera(sim_env, base_intr=base_intr, args=args)

        sim_env.render()
        real_maps = maps if (args.undistort_real and undistort_enabled) else None
        overlaid_imgs, sim_imgs, real_imgs = overlay_envs(sim_env, real_env, real_maps, args)

        im_real.set_data(real_imgs)
        im_sim.set_data(sim_imgs)
        im_overlay.set_data(overlaid_imgs)

        fig.canvas.draw()
        fig.canvas.flush_events()
        if not plt.fignum_exists(fig.number):
            print("The figure has been closed.")
            break

    if hasattr(real_robot, "bus"):
        real_robot.bus.disable_torque()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
