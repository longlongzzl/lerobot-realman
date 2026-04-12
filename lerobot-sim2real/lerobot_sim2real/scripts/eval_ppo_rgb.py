"""
This script is used to evaluate a random or RL trained agent on a real robot using the LeRobot system.
"""
import time
import os
from dataclasses import dataclass
import json
import random
from typing import Optional
import gymnasium as gym
import numpy as np
import torch
import tyro
import imageio
from lerobot_sim2real.config.real_robot import create_real_robot
from lerobot_sim2real.rl.ppo_rgb import Agent, get_infos

from lerobot_sim2real.utils.safety import setup_safe_exit

from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper, Flatten_Multi_RGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from tqdm import tqdm
from mani_skill.utils.visualization import tile_images
import matplotlib.pyplot as plt
@dataclass
class Args:
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to load agent weights from for evaluation. If None then a random agent will be used"""
    env_kwargs_json_path: Optional[str] = None
    """path to a json file containing additional environment kwargs to use. For real world evaluation this is not needed but if you want to turn on debug mode which visualizes the sim and real envs side by side you will need this"""
    debug: bool = False
    """if toggled, the sim and real envs will be visualized side by side"""
    save_step_debug: bool = False
    """If toggled, save per-step debug artifacts (real qpos, images, attention maps, action)"""
    debug_save_dir: Optional[str] = None
    """Root directory to save per-step debug artifacts. If None, uses debug_steps/run_YYYYmmdd_HHMMSS"""
    continuous_eval: bool = True
    """If toggled, the evaluation will run until episode ends without user input. If false, at each step the user will be prompted to press enter to let the robot continue"""
    max_episode_steps: int = 70
    """The maximum number of control steps the real robot can take before we stop the episode and reset the environment. It is recommended to set this number to be larger than the value the sim env is set to, that way you can permit the
    robot more chances to recover from failures / solve the task."""
    num_episodes: Optional[int] = None
    """The number of episodes to evaluate for. If None, the evaluation will run until the user presses ctrl+c"""
    env_id: str = "SO100GraspCube-v1"
    """The environment id to use for evaluation. This should be the same as the environment id used for training."""
    seed: int = 1
    """seed of the experiment"""
    record_dir: Optional[str] = None
    """Directory to save recordings of the camera captured images. If none no recordings are saved"""
    control_freq: Optional[int] = 15
    """The control frequency of the real robot. For safety reasons we recommend setting this to 15Hz or lower as we permit the RL agent to take larger actions to move faster. If this is none, it will use the same control frequency the sim env uses."""
    use_real_robot: bool = True
    """Whether to run with the real robot (default) or stay entirely inside simulation for quick checks."""

def overlay_envs(sim_env, real_env, camera_id=None):
    """
    Overlays sim_env observtions onto real_env observations
    Requires matching ids between the two environments' sensors
    e.g. id=phone_camera sensor in real_env / real_robot config, must have identical id in sim_env
    """

    if camera_id is None:
        real_obs = real_env.get_obs()["sensor_data"]
        sim_obs = sim_env.get_obs()["sensor_data"]
        assert sorted(real_obs.keys()) == sorted(
            sim_obs.keys()
        ), f"real camera names {real_obs.keys()} and sim camera names {sim_obs.keys()} differ"

        overlaid_dict = sim_env.get_obs()["sensor_data"]
        overlaid_imgs = []
        for name in overlaid_dict:
            real_imgs = real_obs[name]["rgb"][0] / 255
            sim_imgs = overlaid_dict[name]["rgb"][0].cpu() / 255
            overlaid_imgs.append(0.5 * real_imgs + 0.5 * sim_imgs)

        return tile_images(overlaid_imgs), real_imgs, sim_imgs

    else:
        real_obs = real_env.get_obs()["sensor_data"]
        sim_obs = sim_env.get_obs()["sensor_data"]
        assert sorted(real_obs.keys()) == sorted(
            sim_obs.keys()
        ), f"real camera names {real_obs.keys()} and sim camera names {sim_obs.keys()} differ"

        overlaid_dict = sim_env.get_obs()["sensor_data"]
        overlaid_imgs = []


        if camera_id == 0:
            name = "base_camera"
        else:
            name = "base_camera_2"
       # for name in real_obs.keys():

        # print("???",name, real_obs.keys() , sim_obs.keys())
            #print(real_obs[name].keys())
        # print(real_obs[name].keys())
        # print(sim_obs[name].keys())
        real_imgs = real_obs[name]["rgb" ][0] / 255
        sim_imgs = sim_obs[name]["rgb"][0].cpu() / 255
        overlaid_imgs.append(0.5 * real_imgs + 0.5 * sim_imgs)

        return tile_images(overlaid_imgs), real_imgs, sim_imgs


import matplotlib.animation as animation

def to_rgb(arr):
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (3, 4):        # CHW -> HWC
        arr = arr.transpose(1, 2, 0)
    if arr.ndim == 3 and arr.shape[2] == 4:             # RGBA -> RGB
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 1) * 255
        arr = arr.astype(np.uint8)
    return arr


def to_numpy_rgb(img):
    """
    将图像转换为 numpy 格式 (H, W, 3)

    Args:
        img: torch.Tensor 或 numpy.ndarray，形状可能是 (C, H, W) 或 (H, W, C)

    Returns:
        numpy.ndarray: (H, W, 3) 格式的 RGB 图像，值范围 [0, 1]
    """
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()

    # 如果是 CHW 格式，转换为 HWC
    if img.ndim == 3 and img.shape[0] in (3, 4):
        img = img.transpose(1, 2, 0)

    # 如果是 RGBA，只取 RGB
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]

    # 确保值范围在 [0, 1]
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0, 1)

    return img


def _as_numpy(arr):
    if torch.is_tensor(arr):
        return arr.detach().cpu().numpy()
    return arr


def _extract_rgb_from_obs(obs, key):
    if key not in obs:
        return None
    img = _as_numpy(obs[key])
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (3, 4):
        img = img.transpose(1, 2, 0)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    if img.ndim != 3:
        return None
    return img


def _to_uint8_rgb(img):
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img
    img_f = img.astype(np.float32)
    if img_f.max() <= 1.0:
        img_f = img_f * 255.0
    img_f = np.clip(img_f, 0, 255)
    return img_f.astype(np.uint8)


def _to_attn_2d(attn_map):
    attn_map = _as_numpy(attn_map)
    if attn_map.ndim == 4:
        attn_map = attn_map[0, 0]
    elif attn_map.ndim == 3:
        if attn_map.shape[0] == 1:
            attn_map = attn_map[0]
        else:
            attn_map = attn_map.squeeze()
    return attn_map


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_step_debug(step_dir, real_obs, action, attn_maps, qpos_rad, qpos_deg):
    _ensure_dir(step_dir)

    action_np = _as_numpy(action)
    np.save(os.path.join(step_dir, "action.npy"), action_np)
    if qpos_rad is not None:
        np.save(os.path.join(step_dir, "real_qpos_rad.npy"), qpos_rad)
    if qpos_deg is not None:
        np.save(os.path.join(step_dir, "real_qpos_deg.npy"), qpos_deg)

    for cam_key in ("rgb_0", "rgb_1"):
        img = _extract_rgb_from_obs(real_obs, cam_key)
        if img is not None:
            np.save(os.path.join(step_dir, f"{cam_key}.npy"), img)
            img_u8 = _to_uint8_rgb(img)
            if img_u8 is not None:
                imageio.imwrite(os.path.join(step_dir, f"{cam_key}.png"), img_u8)

        if attn_maps and cam_key in attn_maps:
            attn_map = attn_maps[cam_key]
            if img is not None:
                h, w = img.shape[:2]
                attn_upsampled = upsample_attention(attn_map, h, w)
            else:
                attn_upsampled = _to_attn_2d(attn_map)
            np.save(os.path.join(step_dir, f"{cam_key}_attn.npy"), attn_upsampled)

            attn_vis = np.clip(attn_upsampled, 0, 1)
            attn_u8 = (attn_vis * 255).astype(np.uint8)
            attn_color = cv2.applyColorMap(attn_u8, cv2.COLORMAP_JET)
            attn_color = cv2.cvtColor(attn_color, cv2.COLOR_BGR2RGB)
            imageio.imwrite(os.path.join(step_dir, f"{cam_key}_attn.png"), attn_color)


# ========== 初始化部分 ==========
# # 创建图形和子图
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# fig.suptitle('实时图像显示', fontsize=16)
#
# # 初始化图像对象（先用空白图像占位）
# im1 = ax1.imshow(np.zeros((224, 224, 3)))
# im2 = ax2.imshow(np.zeros((224, 224, 3)))
#
# ax1.set_title('图像 1')
# ax2.set_title('图像 2')
# ax1.axis('off')
# ax2.axis('off')
#
# plt.ion()  # 开启交互模式
# plt.show()
#
# def update_images(all_sim_imgs):
#     """
#     更新显示的两个图像
#
#     Args:
#         all_sim_imgs: 包含两个图像的列表，每个图像可以是 torch.Tensor 或 numpy.ndarray
#     """
#     assert len(all_sim_imgs) == 2, f"需要2个图像，但得到了 {len(all_sim_imgs)} 个"
#
#     # 转换第一个图像
#     img1 = to_numpy_rgb(all_sim_imgs[0])
#     img2 = to_numpy_rgb(all_sim_imgs[1])
#
#     # 更新图像数据
#     im1.set_data(img1)
#     im2.set_data(img2)
#
#     # 重绘图形
#     fig.canvas.draw()
#     fig.canvas.flush_events()

import cv2
from mani_skill.utils import common, sapien_utils
import torch.nn.functional as F

def compute_intrinsic_from_vfov(vfov: float, img_width: int, img_height: int) -> np.ndarray:
    """
    根据垂直FOV和图像尺寸计算相机内参矩阵（OpenCV约定）
    
    Args:
        vfov: 垂直视场角（弧度）
        img_width: 图像宽度
        img_height: 图像高度
        
    Returns:
        intrinsic: [3, 3] 相机内参矩阵
    """
    # 垂直FOV计算fy
    fy = (img_height / 2.0) / np.tan(vfov / 2.0)
    # 水平焦距按宽高比缩放
    fx = fy * (img_width / img_height)
    cx = img_width / 2.0
    cy = img_height / 2.0
    
    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return intrinsic

def compute_intrinsic_from_fov(fov: float, img_width: int, img_height: int) -> np.ndarray:
    """
    根据FOV和图像尺寸计算相机内参矩阵（兼容旧接口，假设fov是垂直FOV）
    
    Args:
        fov: 视场角（弧度），假设是垂直FOV
        img_width: 图像宽度
        img_height: 图像高度
        
    Returns:
        intrinsic: [3, 3] 相机内参矩阵
    """
    return compute_intrinsic_from_vfov(fov, img_width, img_height)

def compute_extrinsic_lookat_opencv(camera_pos: np.ndarray,
                                    target: np.ndarray,
                                    up: np.ndarray = np.array([0, 0, 1], dtype=np.float32)) -> np.ndarray:
    """
    返回 world->camera 的 4x4 外参，且 camera 坐标系采用 OpenCV 约定：x右、y下、z前
    
    Args:
        camera_pos: [3] 相机位置（世界坐标系）
        target: [3] 目标点（世界坐标系）
        up: [3] 上方向向量（世界坐标系），默认[0,0,1]表示Z轴向上
        
    Returns:
        extrinsic: [4, 4] 相机外参矩阵（世界到相机，OpenCV约定）
    """
    eye = camera_pos.astype(np.float32)
    tgt = target.astype(np.float32)
    up = up.astype(np.float32)

    # 相机z轴（朝前，指向target）
    z_cam_world = (tgt - eye)
    z_cam_world /= (np.linalg.norm(z_cam_world) + 1e-9)

    # 相机x轴（朝右，z×up）
    x_cam_world = np.cross(z_cam_world, up)
    x_cam_world /= (np.linalg.norm(x_cam_world) + 1e-9)

    # 相机y轴（朝下，z×x，满足右手系）
    y_cam_world = np.cross(z_cam_world, x_cam_world)
    y_cam_world /= (np.linalg.norm(y_cam_world) + 1e-9)

    # R_cw: world轴投到camera轴（world->cam）
    # 每一行是camera坐标系的一个轴在世界坐标系中的方向
    R = np.stack([x_cam_world, y_cam_world, z_cam_world], axis=0)  # (3,3)
    t = -R @ eye  # (3,)

    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    return extrinsic

def project_3d_to_2d(point_3d: np.ndarray, camera_intrinsic: np.ndarray, 
                     camera_extrinsic: np.ndarray = None,
                     debug: bool = False) -> Optional[np.ndarray]:
    """
    将3D点投影到2D图像坐标
    
    Args:
        point_3d: [3] 3D位置（世界坐标系）
        camera_intrinsic: [3, 3] 相机内参矩阵
        camera_extrinsic: [4, 4] 相机外参矩阵（世界到相机），如果为None则假设点已在相机坐标系
        debug: 是否打印调试信息
        
    Returns:
        img_coords: [2] 2D图像坐标 (u, v)，如果投影失败返回None
    """
    point_3d = np.asarray(point_3d, dtype=np.float32)
    assert point_3d.shape == (3,), f"point_3d形状错误: {point_3d.shape}, 期望 (3,)"
    assert camera_intrinsic.shape == (3, 3), f"相机内参形状错误: {camera_intrinsic.shape}, 期望 (3, 3)"
    
    # 转换到相机坐标系（外参已经是OpenCV约定：x右、y下、z前）
    if camera_extrinsic is not None:
        assert camera_extrinsic.shape == (4, 4), f"相机外参形状错误: {camera_extrinsic.shape}, 期望 (4, 4)"
        point_homo = np.concatenate([point_3d, [1.0]], axis=0)  # [4]
        point_cam = (camera_extrinsic @ point_homo)[:3]  # [3]（已经是OpenCV坐标系）
    else:
        point_cam = point_3d
    
    # 投影到图像平面（外参已经是OpenCV约定，无需转换）
    img_coords_homo = camera_intrinsic @ point_cam  # [3]
    z = img_coords_homo[2]
    
    # 检查深度是否有效
    if abs(z) <= 1e-6:
        if debug:
            # print("[ProjDebug] z≈0 => bad extrinsic or mismatched frame/unit")
            # print(f"  point_3d(world) = {point_3d}")
            # print(f"  point_cam(OpenCV) = {point_cam}, z={point_cam[2]}")
            # print(f"  K=\n{camera_intrinsic}")
            if camera_extrinsic is not None:
                pass
            pass
        return None
    
    # 如果z为负，说明点在相机后方（OpenCV约定），尝试取反
    if z < 0:
        if debug:
            pass
        img_coords_homo = -img_coords_homo
        z = -z
    
    # 归一化到像素坐标
    img_coords = img_coords_homo[:2] / z  # [2]
    
    return img_coords

def draw_tcp_on_image(img: np.ndarray, tcp_pos: np.ndarray, 
                      camera_intrinsic: np.ndarray, 
                      camera_extrinsic: np.ndarray = None,
                      color: tuple = (0, 255, 0),
                      radius: int = 10) -> np.ndarray:
    """
    在RGB图像上绘制TCP位置
    
    Args:
        img: [H, W, 3] RGB图像 (0~255 uint8 或 0~1 float)
        tcp_pos: [3] TCP的3D位置
        camera_intrinsic: [3, 3] 相机内参矩阵
        camera_extrinsic: [4, 4] 相机外参矩阵（可选）
        color: 绘制颜色 (B, G, R)
        radius: 圆圈半径
        
    Returns:
        img_with_marker: [H, W, 3] 绘制了标记的图像
    """
    assert img.ndim == 3, f"图像维度错误: {img.ndim}, 期望 3 (H, W, 3)"
    assert img.shape[2] == 3, f"图像通道数错误: {img.shape[2]}, 期望 3"
    assert tcp_pos.shape == (3,), f"TCP位置形状错误: {tcp_pos.shape}, 期望 (3,)"
    
    # 确保图像是uint8格式
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    # 投影3D点到2D（如果失败返回None，不绘制）
    img_coords = project_3d_to_2d(tcp_pos, camera_intrinsic, camera_extrinsic)
    if img_coords is None:
        # 投影失败，跳过绘制
        return img
    
    u, v = int(img_coords[0]), int(img_coords[1])
    
    # 检查坐标是否在图像范围内
    H, W = img.shape[:2]
    if 0 <= u < W and 0 <= v < H:
        # 绘制一个绿色圆圈标记TCP位置
        cv2.circle(img, (u, v), radius, color, -1)  # 实心圆
        cv2.circle(img, (u, v), radius + 3, (255, 255, 255), 2)  # 白色外圈
    
    return img

def Rx(a: float) -> np.ndarray:
    """
    绕X轴旋转矩阵（pitch）
    
    Args:
        a: 旋转角度（弧度）
        
    Returns:
        R: [3, 3] 旋转矩阵
    """
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float32)

def Ry(a: float) -> np.ndarray:
    """
    绕Y轴旋转矩阵（yaw）
    
    Args:
        a: 旋转角度（弧度）
        
    Returns:
        R: [3, 3] 旋转矩阵
    """
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float32)

def draw_tilted_axes(img_bgr: np.ndarray, Pc: np.ndarray, K: np.ndarray, 
                     tilt_yaw_deg: float = 25, tilt_pitch_deg: float = -20, 
                     axis_len: float = 0.05, thickness: int = 2) -> np.ndarray:
    """
    在图像上绘制带倾斜角的坐标轴（让蓝轴不正对镜头，三根轴都更容易看清）
    
    Args:
        img_bgr: [H, W, 3] BGR格式图像（uint8）
        Pc: [3] 点在相机坐标系的位置
        K: [3, 3] 相机内参矩阵
        tilt_yaw_deg: 倾斜yaw角度（度），默认25度
        tilt_pitch_deg: 倾斜pitch角度（度），默认-20度
        axis_len: 轴长度（世界坐标单位）
        thickness: 线条粗细
        
    Returns:
        img_bgr: 绘制了坐标轴的BGR图像
    """
    yaw = np.deg2rad(tilt_yaw_deg)
    pitch = np.deg2rad(tilt_pitch_deg)
    
    # 计算倾斜旋转矩阵
    R_tilt = Rx(pitch) @ Ry(yaw)
    
    # 将旋转矩阵转换为旋转向量（Rodrigues）
    rvec, _ = cv2.Rodrigues(R_tilt)
    tvec = np.asarray(Pc, dtype=np.float32).reshape(3, 1)
    
    # z<=0 就别画（说明点在后面）
    if tvec[2, 0] <= 1e-6:
        return img_bgr
    
    cv2.drawFrameAxes(img_bgr, K, None, rvec, tvec, axis_len, thickness)
    return img_bgr

def draw_frame_on_image(img: np.ndarray, point_w: np.ndarray, 
                       K: np.ndarray, E_w2c: np.ndarray, 
                       axis_len: float = 0.05, thickness: int = 2,
                       tilt_yaw_deg: float = 25, tilt_pitch_deg: float = -20) -> np.ndarray:
    """
    使用OpenCV的drawFrameAxes在图像上绘制坐标轴（自带透视缩放，带固定倾斜角）
    
    Args:
        img: [H, W, 3] RGB图像 (0~255 uint8 或 0~1 float)
        point_w: [3] 世界坐标系中的3D点位置
        K: [3, 3] 相机内参矩阵
        E_w2c: [4, 4] 相机外参矩阵（世界到相机，OpenCV约定）
        axis_len: 轴长度（世界坐标单位，默认5cm），投影到图上会天然远小近大
        thickness: 线条粗细
        tilt_yaw_deg: 倾斜yaw角度（度），默认25度，让蓝轴不正对镜头
        tilt_pitch_deg: 倾斜pitch角度（度），默认-20度
        
    Returns:
        img_with_frame: [H, W, 3] 绘制了坐标轴的图像
    """
    assert img.ndim == 3, f"图像维度错误: {img.ndim}, 期望 3 (H, W, 3)"
    assert img.shape[2] == 3, f"图像通道数错误: {img.shape[2]}, 期望 3"
    assert point_w.shape == (3,), f"点位置形状错误: {point_w.shape}, 期望 (3,)"
    assert K.shape == (3, 3), f"内参形状错误: {K.shape}, 期望 (3, 3)"
    assert E_w2c.shape == (4, 4), f"外参形状错误: {E_w2c.shape}, 期望 (4, 4)"
    
    # 确保图像是uint8格式（OpenCV要求BGR格式）
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    # 转换为BGR格式（OpenCV使用BGR）
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # world -> camera
    pw = np.asarray(point_w, dtype=np.float32)
    pc = (E_w2c @ np.array([pw[0], pw[1], pw[2], 1.0], dtype=np.float32))[:3]
    
    # z<=0 就别画（说明还没对齐好/点在后面）
    if pc[2] <= 1e-6:
        # 转换回RGB格式
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    
    # 使用带倾斜角的绘制函数
    img_bgr = draw_tilted_axes(img_bgr, pc, K, tilt_yaw_deg, tilt_pitch_deg, axis_len, thickness)
    
    # 转换回RGB格式
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def draw_world_axes_on_image(img_rgb: np.ndarray, p_w: np.ndarray, 
                             K: np.ndarray, E_w2c: np.ndarray, 
                             L: float = 0.05, thickness: int = 2) -> np.ndarray:
    """
    在世界坐标系中绘制固定的XYZ轴（不是TCP坐标系）
    
    Args:
        img_rgb: [H, W, 3] RGB图像 (0~255 uint8 或 0~1 float)，注意是RGB顺序
        p_w: [3] TCP的世界坐标位置
        K: [3, 3] 相机内参矩阵
        E_w2c: [4, 4] 相机外参矩阵（世界到相机）
        L: 轴长度（米），默认5cm
        thickness: 线条粗细
        
    Returns:
        img_with_axes: [H, W, 3] 绘制了坐标轴的图像
    """
    assert img_rgb.ndim == 3, f"图像维度错误: {img_rgb.ndim}, 期望 3 (H, W, 3)"
    assert img_rgb.shape[2] == 3, f"图像通道数错误: {img_rgb.shape[2]}, 期望 3"
    assert p_w.shape == (3,), f"TCP位置形状错误: {p_w.shape}, 期望 (3,)"
    assert K.shape == (3, 3), f"内参形状错误: {K.shape}, 期望 (3, 3)"
    assert E_w2c.shape == (4, 4), f"外参形状错误: {E_w2c.shape}, 期望 (4, 4)"
    
    # 确保图像是uint8格式
    if img_rgb.dtype != np.uint8:
        if img_rgb.max() <= 1.0:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        else:
            img_rgb = img_rgb.astype(np.uint8)
    
    p_w = np.asarray(p_w, dtype=np.float32).reshape(3,)
    
    # 在世界坐标系中定义四个点：TCP位置 + 三个轴终点
    # X轴: +X方向，Y轴: +Y方向，Z轴: +Z方向（世界坐标系）
    pts_w = np.float32([
        p_w,                    # TCP位置
        p_w + [L, 0, 0],        # X轴终点
        p_w + [0, L, 0],        # Y轴终点
        p_w + [0, 0, L],        # Z轴终点
    ])
    
    # 手写pinhole投影（与project_3d_to_2d一致，包含OpenGL到OpenCV转换）
    def proj(P):
        """将世界坐标点投影到图像坐标"""
        Ph = np.concatenate([P, [1.0]], axis=0)  # 齐次坐标
        Pc_gl = (E_w2c @ Ph)[:3]  # 转换到相机坐标系（OpenGL约定）
        
        # OpenGL到OpenCV坐标转换
        R_gl2cv = np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1]], dtype=np.float32)
        Pc_cv = R_gl2cv @ Pc_gl
        
        z = Pc_cv[2]
        if abs(z) <= 1e-6:  # 点在相机成像平面上或接近原点
            return None
        if z < 0:  # 点在相机后方（OpenCV约定），尝试取反
            Pc_cv = -Pc_cv
            z = -z
        
        uvw = K @ Pc_cv  # 投影到图像平面
        return (uvw[:2] / uvw[2]).astype(int)
    
    # 投影所有点
    p0 = proj(pts_w[0])  # TCP位置
    px = proj(pts_w[1])  # X轴终点
    py = proj(pts_w[2])  # Y轴终点
    pz = proj(pts_w[3])  # Z轴终点
    
    H, W = img_rgb.shape[:2]
    
    def in_img(p):
        """检查点是否在图像范围内"""
        if p is None:
            return False
        return (0 <= p[0] < W) and (0 <= p[1] < H)
    
    # 绘制XYZ轴（RGB颜色：X红，Y绿，Z蓝）
    if in_img(p0) and in_img(px):
        cv2.line(img_rgb, tuple(p0), tuple(px), (255, 0, 0), thickness)  # X轴红色 (RGB)
    if in_img(p0) and in_img(py):
        cv2.line(img_rgb, tuple(p0), tuple(py), (0, 255, 0), thickness)  # Y轴绿色 (RGB)
    if in_img(p0) and in_img(pz):
        cv2.line(img_rgb, tuple(p0), tuple(pz), (0, 0, 255), thickness)  # Z轴蓝色 (RGB)
    
    return img_rgb

def visualize_attention_on_image(img, attn_map, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    将注意力热图叠加到原始图像上
    
    Args:
        img: numpy array, shape (H, W, 3), 值范围 [0, 255] 或 [0, 1]
        attn_map: numpy array, shape (H, W) 或 (1, H, W), 值范围 [0, 1]
        alpha: 注意力热图的透明度
        colormap: OpenCV 颜色映射
    
    Returns:
        numpy array: 叠加后的图像 (H, W, 3), 值范围 [0, 255]
    """
    # 确保图像值范围在 [0, 255]
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    
    # 处理注意力图：确保是 2D (H, W)
    if attn_map.ndim == 3:
        if attn_map.shape[0] == 1:
            attn_map = attn_map[0]
        else:
            attn_map = attn_map.squeeze()
    
    # 归一化到 [0, 255] 并转换为 uint8
    attn_map_uint8 = (attn_map * 255).astype(np.uint8)
    
    # 应用颜色映射
    attn_colored = cv2.applyColorMap(attn_map_uint8, colormap)
    
    # 叠加：img * (1-alpha) + attn_colored * alpha
    overlay = cv2.addWeighted(img, 1 - alpha, attn_colored, alpha, 0)
    
    return overlay

def upsample_attention(attn_map, target_h, target_w):
    """
    将注意力图上采样到目标分辨率
    
    Args:
        attn_map: torch.Tensor, shape [B, 1, H', W'] 或 [1, H', W']
        target_h: 目标高度
        target_w: 目标宽度
    
    Returns:
        torch.Tensor: 上采样后的注意力图 [target_h, target_w]
    """
    if isinstance(attn_map, np.ndarray):
        attn_map = torch.from_numpy(attn_map)
    
    # 确保是 4D [B, 1, H, W]
    if attn_map.ndim == 2:
        attn_map = attn_map.unsqueeze(0).unsqueeze(0)
    elif attn_map.ndim == 3:
        if attn_map.shape[0] == 1:
            attn_map = attn_map.unsqueeze(0)
        else:
            attn_map = attn_map.unsqueeze(1)
    
    # 上采样到目标分辨率
    attn_upsampled = F.interpolate(
        attn_map, 
        size=(target_h, target_w), 
        mode='bilinear', 
        align_corners=False
    )
    
    # 返回第一个 batch，去掉 channel 维度，分离梯度后转换为 numpy
    return attn_upsampled[0, 0].detach().cpu().numpy()


def get_current_qpos(real_env, sim_env, use_real_robot):
    if use_real_robot:
        joint_names = real_env.agent.real_robot._joint_names
        joint_keys = [f"{jn}.pos" for jn in joint_names]
        real_obs_raw = real_env.agent.real_robot.get_observation()
        obs_qpos_deg = np.array([real_obs_raw[k] for k in joint_keys], dtype=np.float32)
        obs_qpos_rad = np.deg2rad(obs_qpos_deg)
    else:
        sim_robot = sim_env.unwrapped.agent.robot
        joint_keys = getattr(sim_robot, "_qpos_names", None)
        if joint_keys is None:
            joint_keys = [f"joint{i + 1}.pos" for i in range(sim_robot.get_qpos().shape[-1])]
        joint_names = [k.replace(".pos", "") for k in joint_keys]
        obs_qpos_rad = sim_robot.get_qpos().detach().cpu().flatten().numpy()
        obs_qpos_rad = obs_qpos_rad[: len(joint_keys)]
        obs_qpos_deg = np.rad2deg(obs_qpos_rad)
    return joint_names, joint_keys, obs_qpos_rad, obs_qpos_deg


def _fmt_vec(arr, decimals=3):
    return np.round(np.asarray(arr, dtype=np.float32), decimals=decimals).tolist()

def _extract_frame_ids_from_agent_obs(obs):#辅助 新帧到位才推理 异步读帧来提升速度
  ids = []
  i = 0
  while f"frame_id_{i}" in obs:
      v = obs[f"frame_id_{i}"]
      if isinstance(v, torch.Tensor):
          ids.append(int(v.detach().cpu().flatten()[0]))
      else:
          ids.append(int(v))
      i += 1
  return tuple(ids) if ids else None



def main(args: Args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ### Create and connect the real robot, wrap it to make it interfaceable with ManiSkill sim2real environments ###
    real_robot = None
    real_agent = None

    last_action = None#辅助 新帧到位才推理 异步读帧来提升速度
    last_frame_ids = None

    if args.use_real_robot:
        real_robot, _ = create_real_robot()
        real_agent = LeRobotRealAgent(real_robot)

# #####################fakeget_qlimits测试
#     import types
#
#     def fake_get_qlimits(self):
#         lower = torch.tensor([
#             -3.1416,  # shoulder_pan
#             -2.3562,  # shoulder_lift
#             -2.3562,  # elbow_flex
#             -2.3562,  # wrist_flex
#             -2.3562,  # wrist_roll
#             -0.6109,  # 第6关
#             0.0,  # gripper 完全闭合
#         ])
#         upper = torch.tensor([
#             3.1416,
#             0.7854,
#             2.3562,
#             2.3562,
#             2.3562,
#             0.6109,
#             0.04,  # 夹爪张开约 4 cm
#         ])
#         return torch.stack([lower, upper], dim=-1).unsqueeze(0)
#
#     real_agent.robot.get_qlimits = types.MethodType(fake_get_qlimits, real_agent.robot)
#
#
# ##############################

    ### Setup the sim environment to make various checks for sim2real alignment and debugging possible ###
    env_kwargs = dict(
        obs_mode="rgb+segmentation",
        render_mode="sensors",
        # render_mode="human",
        render_backend="cpu",
        # only sensors mode is supported right now for real envs, basically rendering the direct visual observations fed to policy
        max_episode_steps=args.max_episode_steps,  # give our robot more time to try and re-try the task
        domain_randomization=False,
        reward_mode="none",
        reconfiguration_freq=1
    )

    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            env_kwargs.update(json.load(f))

    sim_env = gym.make(
        args.env_id,
        **env_kwargs
    )

    # you can apply most wrappers freely to the sim_env and the real_env will use them as well
    sim_env = Flatten_Multi_RGBDObservationWrapper(sim_env)
    if args.record_dir is not None:
        # TODO (stao): verify this wrapper works
        sim_env = RecordEpisode(sim_env, output_dir=args.record_dir, save_trajectory=False, video_fps=sim_env.unwrapped.control_freq)

    ########## TODO:delete Test on sim env
    # sim_obs, infos = sim_env.reset()
    #
    # private_info = get_infos(args.env_id, infos)
    #
    # private_dim = private_info.shape[-1]
    # device = torch.device("cpu")
    # agent = Agent(None, 0.2, sim_env, sample_obs=sim_obs, net_type="3", choice=10, tcl_choice=5, device=device,
    #               z_detach=False, private_dim=private_dim, search_entropy=0.0, act_type=1, critic_use_pi=True)
    # if args.checkpoint:
    #     agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    #     print(f"Loaded agent from {args.checkpoint}")
    # else:
    #     print("No checkpoint provided, using random agent")
    # agent.to(device)
    #
    # print("sim_env.control_mode:",sim_env.control_mode)
    # agent_obs = sim_obs


    # while True:
    #     action = agent.get_action(agent_obs, None, deterministic=True)
    #     print(action)
    #     real_obs, _, terminated, truncated, info = sim_env.step(action.detach().cpu().numpy())
    #     sim_env.render()
    #     time.sleep(0.1)
    #     agent_obs = real_obs
    #
    #     overlaid_dict = sim_env.get_obs()["sensor_data"]
    #     all_sim_imgs = []
    #     for name in overlaid_dict:
    #         print(name)
    #         sim_imgs = overlaid_dict[name]["rgb"][0].cpu() / 255
    #         all_sim_imgs.append(sim_imgs)
    #     assert len(all_sim_imgs) == 2
    #     update_images(all_sim_imgs)


    # assert  1==2
    # The Sim2RealEnv class uses the sim_env to help make various checks for sim2real alignment (e.g. observation space is the same, cameras are the similar)
    # and will always try its best to apply all wrappers you used on the sim env to the real env as well.
    # TODO: check bug
    #
    # Sim {'state': tensor([[1.5403e+00, 7.6337e-03, -2.0552e-02, -1.5821e+00, -1.7846e-02,
    #                    -1.5720e+00, 1.0433e+00, -1.9313e-02, 1.5403e+00, 7.6337e-03,
    #                    -2.0552e-02, -1.5821e+00, -1.7846e-02, -1.5720e+00, 1.0433e+00,
    #                    -3.0455e-02, 7.6337e-03, -2.0552e-02, -1.1261e-02, -1.7846e-02,
    #                    -1.1650e-03, -3.9102e-03]])
    # Real {'state': tensor([[1.5817e+00, 1.3753e-02, 9.3375e-03, -1.5860e+00, -4.1137e-02,
    #                    -1.5576e+00, 1.0319e+00, 0.0000e+00, 1.5628e+00, -1.3125e-02,
    #                    -3.2899e-02, -1.5512e+00, -8.7266e-04, -1.5872e+00, 1.0535e+00,
    #                    -7.9588e-03, -1.3125e-02, -3.2899e-02, 1.9635e-02, -8.7266e-04,
    #                    -1.6406e-02, 6.3006e-03]])
    sim_obs, infos = sim_env.reset()
    if args.use_real_robot:
        real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent, control_freq=args.control_freq)
    else:
        real_env = sim_env
    # print(sim_obs)
    #
    # assert  1==2

    # 获取相机参数（用于TCP位置可视化）- 从config.json读取
    assert args.env_kwargs_json_path is not None, "需要提供env_kwargs_json_path参数来读取相机配置"
    
    with open(args.env_kwargs_json_path, "r") as f:
        config = json.load(f)
    
    assert "base_camera_settings" in config, f"config.json中找不到base_camera_settings，可用键: {list(config.keys())}"
    camera_settings = config["base_camera_settings"]
    
    assert "pos" in camera_settings, f"base_camera_settings中找不到pos，可用键: {list(camera_settings.keys())}"
    assert "target" in camera_settings, f"base_camera_settings中找不到target，可用键: {list(camera_settings.keys())}"
    assert "fov" in camera_settings, f"base_camera_settings中找不到fov，可用键: {list(camera_settings.keys())}"
    
    camera_pos = np.array(camera_settings["pos"], dtype=np.float32)
    camera_target = np.array(camera_settings["target"], dtype=np.float32)
    camera_fov = float(camera_settings["fov"])
    
    assert camera_pos.shape == (3,), f"相机位置形状错误: {camera_pos.shape}, 期望 (3,)"
    assert camera_target.shape == (3,), f"目标点形状错误: {camera_target.shape}, 期望 (3,)"
    
    # 获取图像尺寸
    assert "rgb_0" in sim_obs, f"找不到rgb_0观察，可用键: {list(sim_obs.keys())}"
    if isinstance(sim_obs["rgb_0"], torch.Tensor):
        img_height, img_width = sim_obs["rgb_0"].shape[1:3]
    else:
        img_height, img_width = sim_obs["rgb_0"].shape[:2]
    
    # 计算内参和外参（使用OpenCV约定的外参计算）
    camera_intrinsic = compute_intrinsic_from_fov(camera_fov, img_width, img_height)
    camera_extrinsic = compute_extrinsic_lookat_opencv(camera_pos, camera_target)
    
    assert camera_intrinsic.shape == (3, 3), f"相机内参形状错误: {camera_intrinsic.shape}, 期望 (3, 3)"
    assert camera_extrinsic.shape == (4, 4), f"相机外参形状错误: {camera_extrinsic.shape}, 期望 (4, 4)"
    
    # ========== Sanity Check：验证外参方向是否正确 ==========
    p_cam_center = (camera_extrinsic @ np.array([*camera_pos, 1.0]))[:3]
    p_cam_target = (camera_extrinsic @ np.array([*camera_target, 1.0]))[:3]
    # print("[Sanity] cam(center) should be ~[0,0,0]:", p_cam_center)
    # print("[Sanity] cam(target) x,y should be ~0 and z should be >0:", p_cam_target)
    
    if np.linalg.norm(p_cam_center) > 1e-3:
        pass
    
    if p_cam_target[2] <= 1e-6:
        pass
    # ============================================
    
    # print(f"[Info] 从config.json成功获取相机参数:")
    # print(f"  位置: {camera_pos}")
    # print(f"  目标: {camera_target}")
    # print(f"  FOV: {camera_fov} rad ({np.rad2deg(camera_fov):.2f} deg)")
    # print(f"  图像尺寸: {img_width}x{img_height}")
    # print(f"  内参:\n{camera_intrinsic}")
    # print(f"  外参:\n{camera_extrinsic}")
    
    # ========== Debug检查：检查相机参数和坐标系一致性 ==========
    target_pos_dist = np.linalg.norm(camera_target - camera_pos)
    # print(f"[Debug] ||camera_target - camera_pos|| = {target_pos_dist:.6f}")
    if target_pos_dist < 1e-3:
        pass
    
    # 检查坐标尺度
    # print(f"[Debug] camera_pos尺度: {np.linalg.norm(camera_pos):.3f}")
    # print(f"[Debug] camera_target尺度: {np.linalg.norm(camera_target):.3f}")
    
    # Debug检查：投影camera_target应该大致在图像中心
    target_2d = project_3d_to_2d(camera_target, camera_intrinsic, camera_extrinsic, debug=True)
    if target_2d is not None:
        pass
    else:
        pass
    # ============================================

    private_info = get_infos(args.env_id, infos)

    private_dim = private_info.shape[-1]
    # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    real_obs, _ = real_env.reset()

    # print("1", real_obs)
    # assert 1==2
    # print("GGGGGGGGGGGGGGGGGGGGGGGGGGGG")
    # print("sim_obs shape",sim_obs)
    # print("real_obs shape", real_obs)
    for k in sim_obs.keys():
        pass


    ### Safety setups. Close environments/turn off robot upon ctrl+c ###
    if args.use_real_robot:
        setup_safe_exit(sim_env, real_env, real_agent)

    # real_robot.bus.disable_torque()
    # while True:
    #     sim_env.agent.robot.set_qpos(real_agent.get_qpos())
    #     sim_env.render()

    ### Load our checkpoint ###
    # TODO recover

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_obs = {#去掉限速用的门控
        k: v
        for k, v in real_obs.items()
        if not k.startswith("frame_id_")
    }

    agent = Agent(None, 0.2,  sim_env, sample_obs=sample_obs, net_type="4", choice=10, tcl_choice=5, device= device, z_detach = False, private_dim= private_dim, search_entropy= 0.0, act_type=1,  critic_use_pi=True)
    if args.checkpoint:

        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
        # print(f"Loaded agent from {args.checkpoint}")
    else:
        pass
    agent.to(device)

    save_step_debug_enabled = args.save_step_debug or args.debug
    debug_root = None
    debug_joint_meta_written = False
    if save_step_debug_enabled:
        if args.debug_save_dir is None:
            run_id = time.strftime("run_%Y%m%d_%H%M%S")
            debug_root = os.path.join("debug_steps", run_id)
        else:
            debug_root = args.debug_save_dir
        _ensure_dir(debug_root)
        with open(os.path.join(debug_root, "meta.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    frames = []
    #     ### Visualization setup for debug modes ###
    show_attention = False
    if args.debug or show_attention:
        #fig = plt.figure()
        fig = plt.figure(figsize=(20, 12))  # 增加宽度以容纳注意力可视化
        # 原始布局：3行2列
        ax = fig.add_subplot(3, 3, 1)  # 叠加图像 1
        ax2 = fig.add_subplot(3, 3, 4)  # 仿真图像 1
        ax3 = fig.add_subplot(3, 3, 7)  # 真机图像 1

        ax4 = fig.add_subplot(3, 3, 2)  # 叠加图像 2
        ax5 = fig.add_subplot(3, 3, 5)  # 仿真图像 2
        ax6 = fig.add_subplot(3, 3, 8)  # 真机图像 2
        
        # 新增：注意力热图显示
        ax_attn1 = fig.add_subplot(3, 3, 3)  # 注意力热图 1
        ax_attn2 = fig.add_subplot(3, 3, 6)  # 注意力热图 2
        # 移除未使用的子图，底部右侧留空或用于其他显示
        # ax = fig.add_subplot(3, 1, 1)
        # ax2 = fig.add_subplot(3, 1, 2)
        # ax3 = fig.add_subplot(3, 1, 3)

        # Disable all default key bindings
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        fig.canvas.manager.key_press_handler_id = None


        # initialize the plot
        overlaid_imgs, real_imgs, sim_imgs = overlay_envs(sim_env, real_env, 0)
        im = ax.imshow(overlaid_imgs)
        im2 = ax2.imshow(sim_imgs)
        im3 = ax3.imshow(real_imgs)
        
        # 初始化第二相机显示（如果存在）- 注意：K1和E1在后面定义，这里先设为None
        im_1 = None
        im2_1 = None
        im3_1 = None
        
        # 初始化注意力可视化
        if show_attention:
            # 占位符：单通道灰度图（用于注意力热图）
            attn_placeholder = np.zeros((224, 224), dtype=np.float32)
            # 使用颜色映射显示注意力热图（'jet' 或 'hot' 都可以）
            im_attn1 = ax_attn1.imshow(attn_placeholder, cmap='jet', vmin=0, vmax=1)
            im_attn2 = ax_attn2.imshow(attn_placeholder, cmap='jet', vmin=0, vmax=1)
            ax_attn1.set_title("Attention Map (Camera 0)")
            ax_attn2.set_title("Attention Map (Camera 1)")
            ax_attn1.axis('off')
            ax_attn2.axis('off')


    ### Main evaluation loop ###
    episode_count = 0

    while args.num_episodes is None or episode_count < args.num_episodes:
        # print(f"Evaluation Episode {episode_count}")
        for _ in tqdm(range(args.max_episode_steps)):
            step_idx = _
            mt1=time.time()

            # real_obs =  sim_env.get_obs()["sensor_data"]

            agent_obs = real_obs
            # agent_obs = sim_env.get_obs()

            agent_obs = {k: v.to(device) for k, v in agent_obs.items() if not k.startswith("frame_id_")}


            # print("time1",time.time() - mt1)

            action = agent.get_action(agent_obs, None, deterministic=True)
            # frame_ids = _extract_frame_ids_from_agent_obs(real_obs)
            # has_new_frame = frame_ids is not None and frame_ids != last_frame_ids
            #
            # agent_obs = {
            #     k: v.to(device)
            #     for k, v in real_obs.items()
            #     if not k.startswith("frame_id_")
            # }
            #
            # if last_action is None or has_new_frame:
            #     action = agent.get_action(agent_obs, None, deterministic=True)
            #     last_action = action
            #     last_frame_ids = frame_ids
            # else:
            #     action = torch.zeros_like(last_action)

            attn_maps = {}
            if hasattr(agent.feature_net, 'last_attn_maps') and agent.feature_net.last_attn_maps:
                attn_maps = agent.feature_net.last_attn_maps
                # attn_maps 是一个字典，键是 "rgb_0", "rgb_1" 等
                # 每个值是 [B, 1, H', W'] 的注意力热图
            # ============================================

            if save_step_debug_enabled and debug_root is not None:
                try:
                    joint_names, joint_keys, obs_qpos_rad, obs_qpos_deg = get_current_qpos(real_env, sim_env, args.use_real_robot)
                    if (not debug_joint_meta_written) and joint_names is not None:
                        with open(os.path.join(debug_root, "joint_names.json"), "w") as f:
                            json.dump(joint_names, f, indent=2)
                        debug_joint_meta_written = True
                    step_dir = os.path.join(
                        debug_root,
                        f"episode_{episode_count:03d}",
                        f"step_{step_idx:04d}",
                    )
                    save_step_debug(step_dir, real_obs, action, attn_maps, obs_qpos_rad, obs_qpos_deg)
                except Exception as exc:
                    print(f"[DebugSave] failed at episode {episode_count} step {step_idx}: {exc}")
            if not args.continuous_eval:
                input("Press enter to continue to next timestep")
            # print("time2",time.time() - mt1)
            real_obs, _, terminated, truncated, info = real_env.step(action.detach().cpu().numpy())

            joint_names, joint_keys, obs_qpos_rad, obs_qpos_deg = get_current_qpos(
                real_env, sim_env, args.use_real_robot
            )
            # home_qpos_deg=[90,0,0,-90,0,-90,60]
            # if np.max(np.abs(obs_qpos_deg[:7] - home_qpos_deg[:7])) < 10.0 and step_idx > 20:
            #     print("arm returned home, break")
            #     print(real_env.agent.real_robot.arm.rm_movej([178, -5, 0, -70, 0, -102, 60], 100, 0, 0, 1))
            #     break

            # print("time3", time.time() - mt1)
        # 跟踪日志已关闭，避免高频数组格式化和打印影响运行速度。

            # print("sim drive targets & current sim qpos ===============================")
            # print(_, "sim_articulation", sim_articulation)
            # print(_, "current_sim_qpos", current_sim_qpos)

            t3 = time.time()
            # time.sleep(0.01)
            # print(action, "Inference time", t2- t1, "sim time", t3- t2)

            # overlaid_dict = sim_env.get_obs()["sensor_data"]
            # all_sim_imgs = []
            # for name in overlaid_dict:
            #     print(name)
            #     sim_imgs = overlaid_dict[name]["rgb"][0].cpu() / 255
            #     all_sim_imgs.append(sim_imgs)
            #
            # assert len(all_sim_imgs) == 2
            # update_images(all_sim_imgs)

            if args.debug or show_attention:
                overlaid_imgs, real_imgs, sim_imgs = overlay_envs(sim_env, real_env, 0)
                #             overlaid_imgs_1, real_imgs_1, sim_imgs_1 = overlay_envs(sim_env, real_env, 1)

                # 转换为 RGB 格式
                overlaid = to_rgb(overlaid_imgs)
                sim = to_rgb(sim_imgs)
                real = to_rgb(real_imgs)

                # ========== 获取用于可视化的3D位置 ==========
                # 使用TCP位置进行可视化
                assert hasattr(sim_env.unwrapped, 'agent'), "sim_env.unwrapped没有agent属性"
                assert hasattr(sim_env.unwrapped.agent, 'tcp_pose'), "agent没有tcp_pose属性"
                assert hasattr(sim_env.unwrapped.agent.tcp_pose, 'p'), "tcp_pose没有p属性"

                tcp_pos = sim_env.unwrapped.agent.tcp_pose.p  # [x, y, z]
                if isinstance(tcp_pos, torch.Tensor):
                    tcp_pos = tcp_pos.detach().cpu().numpy()
                else:
                    tcp_pos = np.array(tcp_pos, dtype=np.float32)

                # 如果tcp_pos是1D，确保是(3,)
                if tcp_pos.ndim > 1:
                    tcp_pos = tcp_pos.flatten()[:3]

                assert tcp_pos.shape == (3,), f"TCP位置形状错误: {tcp_pos.shape}, 期望 (3,)"
                vis_pos = tcp_pos
                vis_pos_type = "tcp_pos"

                # 确保vis_pos是(3,)形状的numpy数组
                vis_pos = np.array(vis_pos, dtype=np.float32).flatten()[:3]
                assert vis_pos.shape == (3,), f"可视化位置形状错误: {vis_pos.shape}, 期望 (3,)"

                # ========== Debug检查：检查位置和相机参数的坐标系一致性 ==========
                if _ == 0:  # 只在第一步打印
                    # print(f"[Info] 使用的{vis_pos_type}: {vis_pos}")
                    # print(f"[Debug] 位置尺度: {np.linalg.norm(vis_pos):.3f}")
                    # print(f"[Debug] camera_pos尺度: {np.linalg.norm(camera_pos):.3f}")
                    # print(f"[Debug] camera_target尺度: {np.linalg.norm(camera_target):.3f}")

                    # 检查坐标尺度是否一致
                    pos_scale = np.linalg.norm(vis_pos)
                    cam_scale = np.linalg.norm(camera_pos)
                    if abs(pos_scale - cam_scale) > max(pos_scale, cam_scale) * 0.5:
                        pass

                    # 测试投影位置
                    pos_2d_test = project_3d_to_2d(vis_pos, camera_intrinsic, camera_extrinsic, debug=True)
                    if pos_2d_test is not None:
                        pass
                    else:
                        pass
                # ============================================

                # ========== 在真实图像和仿真图像上绘制位置和坐标轴 ==========
                # 使用TCP位置进行可视化
                real = draw_frame_on_image(real, vis_pos, camera_intrinsic, camera_extrinsic, axis_len=0.05, thickness=2)
                sim = draw_frame_on_image(sim, vis_pos, camera_intrinsic, camera_extrinsic, axis_len=0.05, thickness=2)
                # ============================================

                # 更新基础图像显示
                im.set_data(overlaid)
                im2.set_data(sim)
                im3.set_data(real)

                # ========== 更新注意力可视化 ==========
                if show_attention and attn_maps:
                    # 获取原始图像尺寸（用于上采样）
                    img_h, img_w = sim.shape[:2] if isinstance(sim, np.ndarray) else (224, 224)

                    # 处理每个相机的注意力
                    for cam_key in ["rgb_0", "rgb_1"]:
                        if cam_key in attn_maps:
                            attn_map = attn_maps[cam_key]  # [B, 1, H', W']

                            # 提取第一个 batch
                            if attn_map.ndim == 4:
                                attn_map = attn_map[0]  # [1, H', W']

                            # 上采样到原始图像分辨率
                            attn_upsampled = upsample_attention(attn_map, img_h, img_w)

                            # 调试信息：打印注意力值的范围
                            if _ == 0:  # 只在第一步打印
                                pass

                            # 叠加到对应的图像上
                            if cam_key == "rgb_0":
                                sim_with_attn = visualize_attention_on_image(sim, attn_upsampled, alpha=0.5)
                                # 更新注意力热图显示（确保是 2D 数组，值范围 [0, 1]）
                                # 确保 attn_upsampled 是 2D 且值在 [0, 1] 范围内
                                attn_display = np.clip(attn_upsampled, 0, 1)
                                im_attn1.set_data(attn_display)
                                im_attn1.set_clim(vmin=0, vmax=1)  # 设置颜色范围
                                # 更新叠加图像
                                im.set_data(sim_with_attn)
                            elif cam_key == "rgb_1":
                                sim_1_with_attn = visualize_attention_on_image(sim_1, attn_upsampled, alpha=0.5)
                                # 更新注意力热图显示（确保是 2D 数组，值范围 [0, 1]）
                                attn_display = np.clip(attn_upsampled, 0, 1)
                                im_attn2.set_data(attn_display)
                                im_attn2.set_clim(vmin=0, vmax=1)  # 设置颜色范围
                                # 更新叠加图像
                            # im_1.set_data(sim_1_with_attn)
                # ============================================

                # Redraw the plot
                fig.canvas.draw()
                fig.show()
                fig.canvas.flush_events()

                # 收集帧用于视频保存（如果需要）
                if args.debug:
                    # 把三张图高方向拼起来：H_total = 3H，W = W
                    frame = np.vstack([overlaid, sim, real])
                    frames.append(frame)  # ★ 收集这一帧

        episode_count += 1
        real_env.reset()

        # from utils import plot_action_sequence
        #
        # plot_action_sequence(qpos_list, len(qpos_list) * (1 / 20))
        #

        if args.use_real_robot and args.debug and frames:
            # print("Done .... store ......")
            fps = 20
            h, w, _ = frames[0].shape
            out = cv2.VideoWriter("debug_video.mp4",
                                  cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps, (w, h))
            for f in frames:
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out.release()
            # print(f"[√] Saved {len(frames)} frames to debug_video.mp4")

    sim_env.close()
    if args.use_real_robot:
        real_env.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
