import copy
import json
import time
from typing import Optional
import gymnasium as gym
import torch
#from pybullet_examples.inverse_dynamics import q_pos

from lerobot_sim2real.utils.safety import setup_safe_exit
import sys

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from lerobot_sim2real.config.real_robot import create_real_robot
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
from mani_skill.envs.sim2real_env import Sim2RealEnv
import cv2
import numpy as np
import tyro
from mani_skill.utils.visualization.misc import tile_images
from mani_skill.utils import sapien_utils
from dataclasses import dataclass
import matplotlib.pyplot as plt


uid = 101


def to_numpy_image(image):
    if torch.is_tensor(image):
        return image.detach().cpu().numpy().astype(np.float32)
    return np.asarray(image, dtype=np.float32)


@dataclass
class Args:
    env_id: str = f"RM75GraspCube_two_cameras-v1"
    # env_id: str = f"SO{uid}GraspCube-v1"
    """The environment id to train on"""
    env_kwargs_json_path: Optional[str] = "../../so101_env_config.json"
    """Path to a json file containing additional environment kwargs to use."""

def overlay_envs(sim_env, real_env):
    """
    Overlays sim_env observtions onto real_env observations
    Requires matching ids between the two environments' sensors
    e.g. id=phone_camera sensor in real_env / real_robot config, must have identical id in sim_env
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
        real_imgs = to_numpy_image(real_obs[name]["rgb"][0]) / 255.0
        sim_imgs = to_numpy_image(overlaid_dict[name]["rgb"][0]) / 255.0
        overlaid_imgs.append(0.5 * real_imgs + 0.5 * sim_imgs)
        sim_imgs_list.append(sim_imgs)
        real_imgs_list.append(real_imgs)
        loss = np.mean((real_imgs - sim_imgs) ** 2)
        print("current loss", loss)

    return tile_images(overlaid_imgs), tile_images(sim_imgs_list), tile_images(real_imgs_list)


def get_real_image(real_env):
    real_obs = real_env.get_obs()["sensor_data"]
    # real_imgs = []

    real_imgs = [real_obs["base_camera"]["rgb"][0] / 255]
    return tile_images(real_imgs)

def calculate_error(sim_env, real_env):
    """
    Overlays sim_env observtions onto real_env observations
    Requires matching ids between the two environments' sensors
    e.g. id=phone_camera sensor in real_env / real_robot config, must have identical id in sim_env
    """
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

        loss = torch.mean((real_imgs - sim_imgs)**2)
  #      print("current loss", loss)

    return loss.item(), tile_images(overlaid_imgs)


def get_init_camera_pose(sim_env):
    pos = sim_env.unwrapped.base_camera_settings["pos"]
    fov = sim_env.unwrapped.base_camera_settings["fov"]
    current_list = [] #list(pos.detach().cpu().numpy())
    current_list.append(fov)
    return current_list

def set_camera_pose(sim_env, settings):

    pose = sapien_utils.look_at(torch.tensor(settings[:3], dtype=torch.float32), sim_env.unwrapped.base_camera_settings["target"])
    sim_env.unwrapped.camera_mount.set_pose(pose)

   # sim_env.unwrapped.camera_mount.set_pose(torch.tensor(settings[:3], dtype=torch.float32))
    sim_env.unwrapped._sensors["base_camera"].camera.set_fovy( settings[-1])


def update_camera(sim_env):
    global camera_offset, fov_offset, last_frame_time, help_message_printed
    current_time = time.time()
    delta_time = current_time - last_frame_time
    last_frame_time = current_time

    # Reset camera position and FOV on backspace
    if "backspace" in active_keys:
        camera_offset = torch.zeros(3, dtype=torch.float32)
        fov_offset = 0.0

    # Camera movement mapping based on active keys
    if "w" in active_keys:
        camera_offset[0] -= 0.025  # Move forward
    if "s" in active_keys:
        camera_offset[0]  += 0.025  # Move back
    if "d" in active_keys:
        camera_offset[1]  += 0.025  # Move right
    if "a" in active_keys:
        camera_offset[1] -= 0.025  # Move left
    if "up" in active_keys:
        camera_offset[2]  += 0.025  # Move up
    if "down" in active_keys:
        camera_offset[2] -= 0.025  # Move down

    camera_target_offset = torch.zeros(3, dtype=torch.float32)

    if "y" in active_keys:
        camera_target_offset[0] -= 0.025 # Move forward
    if "h" in active_keys:
        camera_target_offset[0] += 0.025 # Move back
    if "u" in active_keys:
        camera_target_offset[1] += 0.025  # Move right
    if "j" in active_keys:
        camera_target_offset[1] -= 0.025  # Move left
    if "i" in active_keys:
        camera_target_offset[2] += 0.025  # Move right
    if "k" in active_keys:
        camera_target_offset[2] -= 0.025  # Move left

    # FOV control
    if "left" in active_keys:
        fov_offset -=  0.025
    if "right" in active_keys:
        fov_offset +=  0.025

    # update camera position and fov
    base_pos = torch.tensor(sim_env.unwrapped.base_camera_settings["pos"], dtype=torch.float32)
    base_target = torch.tensor(sim_env.unwrapped.base_camera_settings["target"], dtype=torch.float32)

    print(sim_env.unwrapped.base_camera_settings["pos"], camera_offset)
    pos = base_pos + camera_offset
    target = base_target + camera_target_offset
    fov = sim_env.unwrapped.base_camera_settings["fov"] + fov_offset

    sim_env.unwrapped.base_camera_settings["target"] = target
    pose = sapien_utils.look_at(pos, target)
    sim_env.unwrapped.camera_mount.set_pose(pose)
    sim_env.unwrapped._sensors["base_camera"].camera.set_fovy(fov)

    if len(active_keys) > 0:

        print("current_camera_position", pose.p)
        print(
            "current_camera_fov",
            sim_env.unwrapped.base_camera_settings["fov"] + fov_offset,
        )
        print("camera target",  sim_env.unwrapped.base_camera_settings["target"])
        help_message_printed = False  # Reset the flag when there's movement
    elif (
        not help_message_printed
    ):  # Only print help message if it hasn't been printed yet
        print("=== Commands for controlling sim camera ===")
        print(
            "press: (w), (a) to move in x, (s), (d) to move in y, (up), (down) to move in z, (left), (right) to change fov of simulation camera"
        )
        print("press: (backspace) to reset, close figure to exit")
        print()
        help_message_printed = True

camera_offset = torch.zeros(3, dtype=torch.float32)
fov_offset = 0.0
active_keys = set()
last_frame_time = time.time()
MOVEMENT_SPEED = 0.1  # units per second
FOV_CHANGE_SPEED = 0.1  # radians per second
help_message_printed = False  # Flag to track if we've printed the help message


def on_key_press(event):
    global active_keys
    active_keys.add(event.key)


def on_key_release(event):
    global active_keys
    active_keys.discard(event.key)


def sample_noisy_poses(
        init_camera_hyper,  # list/tuple/ndarray，形如 [pose1, pose2, ...]
        n: int = 500,  # 总样本数
        sigma: float = 0.25,  # 高斯噪声 σ
        clip: bool = True,
        rng: np.random.Generator | None = None,  # 可选：传入自定义随机数生成器
):
    """
    基于多个相机基准位姿，添加 0 均值 σ 方差高斯噪声，合计生成 n 份新相机参数。

    Parameters
    ----------
    init_camera_hyper : sequence of array-like
        每个元素是一份相机位姿参数 (D,)。所有元素的维度必须一致。
    n : int
        总体样本（个体）数量。
    sigma : float
        噪声标准差 (0~0.1)。默认 0.1。
    clip : bool
        是否把噪声裁剪到 ±sigma，避免极端值。
    rng : np.random.Generator, optional
        可传入自定义随机数生成器；若为 None，则使用 np.random.default_rng().

    Returns
    -------
    poses : np.ndarray
        形状 (n, D) 的数组，按输入顺序拼接。
    """
    # --- 准备随机数发生器 -----------------------------------------------------
    if rng is None:
        rng = np.random.default_rng()

    # --- 输入检查 -------------------------------------------------------------
    bases = [np.asarray(pose, dtype=float) for pose in init_camera_hyper]
    if len(bases) == 0:
        raise ValueError("init_camera_hyper 不能为空")

    dims = {base.shape[-1] for base in bases}
    if len(dims) != 1:
        raise ValueError("所有基准位姿维度必须一致")
    D = dims.pop()

    # --- 计算每个基准分配的样本数 ---------------------------------------------
    m = len(bases)
    per_base = n // m  # 每个基准至少分到这么多
    remainder = n % m  # 余下的分到前 remainder 个基准

    poses_list = []
    for i, base in enumerate(bases):
        k = per_base + (1 if i < remainder else 0)  # 当前基准生成的数量

        if k == 0:
            continue  # 可能 n < m 的情况

        # 生成 k×D 高斯噪声
        noise = rng.normal(loc=0.0, scale=sigma, size=(k, D))
        if clip:
            noise = np.clip(noise, -sigma, sigma)

        poses_i = base[None, :] + noise
        poses_list.append(poses_i)

    # --- 拼接并返回 -----------------------------------------------------------
    if not poses_list:
        # 处理极端情况 n == 0
        return np.empty((0, D), dtype=float)

    poses = np.concatenate(poses_list, axis=0)
    poses[:, 2] = np.maximum(poses[:, 2], 0.02)
    return poses

# def sample_noisy_poses(init_camera_hyper, n=100, sigma=0.1, clip=True):
#     """
#     以 init_camera_hyper 为基准，添加 0 均值 σ 方差高斯噪声，生成 n 份新相机参数。
#
#     Parameters
#     ----------
#     init_camera_hyper : np.ndarray or sequence-like
#         原始相机位姿参数，长度可以是 3（平移）/6（平移+欧拉角）/7（平移+四元数）。
#     n : int
#         样本数（个体数）。默认 100。
#     sigma : float
#         噪声标准差（0~0.1）。默认 0.1。
#     clip : bool
#         是否把噪声裁剪到 ±sigma，避免偶发性极端值。
#
#     Returns
#     -------
#     poses : np.ndarray
#         形状 (n, D) 的数组，D 为参数维度。
#     """
#     init_np = np.asarray(init_camera_hyper, dtype=float)
#     D = init_np.shape[-1]
#
#     # 生成 n×D 的 0 均值高斯噪声
#     noise = np.random.normal(loc=0.0, scale=sigma, size=(n, D))
#     if clip:
#         noise = np.clip(noise, -sigma, sigma)
#
#     # broadcast 到 n 行，再相加
#     poses = init_np[None, :] + noise
#     return poses
import threading
import time
import tkinter as tk
from tkinter import ttk
import threading
import cv2


class ActionTuner:
    dim = 8

    def __init__(self, org_pos):
        self.root = tk.Tk()
        self.root.title("Action Tuner")
        self.v = [0.0] * self.dim

        # 设置初始Action值
        self.vals = []
        for i in range(self.dim):
            self.vals.append(tk.DoubleVar(value=org_pos[i]))

        # 创建滑块
        for i in range(self.dim):
            self.create_slider(f"{i}", self.vals[i], -3.0, 3.0, i)

        # 显示Action值标签
        self.label = tk.Label(self.root, text="")
        self.update_label()
        self.label.grid(row=self.dim, column=0, columnspan=2, pady=10)

        # 绑定更新事件
        for i in range(self.dim):
            self.vals[i].trace("w", self.update_label)

        self.updated = True

    def create_slider(self, name, variable, min_val, max_val, row):
        """创建一个滑块"""
        label = tk.Label(self.root, text=f"{name} Dim")
        label.grid(row=row, column=0, padx=10, pady=5)

        slider = ttk.Scale(
            self.root, from_=min_val, to=max_val, orient="horizontal", variable=variable, length=1000
        )
        slider.grid(row=row, column=1, padx=10, pady=5)

    def update_label(self, *args):
        """更新Action显示标签"""
        v = []
        for i in range(self.dim):
            v.append(self.vals[i].get())
        self.v = v

        # self.label.config(text=f"P: {self.p:.2f}, I: {self.i:.2f}, D: {self.d:.2f}, i_clip_thres: {self.i_clip_thres:.2f}, i_clip_coef: {self.i_clip_coef:.2f}")

        self.updated = True


class ActionTunerRunner:
    """Keeps the Tkinter GUI on the main thread while providing a polling API."""

    def __init__(self, org_pos):
        self.app = ActionTuner(org_pos)
        self._closed = False

    def pump_events(self):
        """Process Tk events; returns False if the window has been closed."""
        if self._closed:
            return False
        try:
            self.app.root.update_idletasks()
            self.app.root.update()
            return True
        except tk.TclError:
            # Window has likely been closed
            self._closed = True
            return False

import matplotlib.pyplot as plt
from collections import deque
import numpy as np

BUFFER_LEN = 1000  # 滚动窗口长度
NUM_JOINTS = 6     # 关节数
class RealtimeQposPlotter:
    """实时绘制 6 维关节 qpos。

    典型使用::

        plotter = RealtimeQposPlotter()
        plotter.show(block=False)  # 提前弹窗且不阻塞
        for step in range(N):
            plotter.feed(real_qpos, sim_qpos)  # 动态刷新
        plotter.show()  # (可选) 结束后阻塞，直到用户关闭窗口

    - **实线**：真实机械臂 qpos
    - **虚线**：仿真机械臂 qpos
    - 仅保存最近 ``BUFFER_LEN`` 步
    """

    def __init__(self):
        # 启用交互模式
        plt.ion()

        # 为真机 / 仿真各准备 6 个队列
        self.real_buf = [deque(maxlen=BUFFER_LEN) for _ in range(NUM_JOINTS)]
        self.sim_buf  = [deque(maxlen=BUFFER_LEN) for _ in range(NUM_JOINTS)]
        self.step_counter = 0

        # 画布与 6 行子图
        self.fig, self.axes = plt.subplots(NUM_JOINTS, 1, figsize=(8, 2 * NUM_JOINTS), sharex=True)
        self.real_lines, self.sim_lines = [], []
        for i, ax in enumerate(self.axes):
            real_line, = ax.plot([], [], label=f"Joint {i+1} Real", lw=1.5)
            sim_line,  = ax.plot([], [], ls="--", label=f"Joint {i+1} Sim", lw=1)
            self.real_lines.append(real_line)
            self.sim_lines.append(sim_line)
            ax.set_ylabel("qpos")
            ax.legend(loc="upper right", fontsize="small")
        self.axes[-1].set_xlabel("Step")
        plt.tight_layout()

    # ---------------------- 对外接口 ----------------------
    def feed(self, real_qpos, sim_qpos):
        """追加一帧数据并刷新窗口。"""
        if len(real_qpos) != NUM_JOINTS or len(sim_qpos) != NUM_JOINTS:
            raise ValueError(f"Each qpos list must have {NUM_JOINTS} elements.")
        self._append(self.real_buf, real_qpos)
        self._append(self.sim_buf,  sim_qpos)
        self.step_counter += 1
        self._draw()

    # -------------------- 内部辅助方法 --------------------
    @staticmethod
    def _append(buf_list, qpos):
        for i in range(NUM_JOINTS):
            buf_list[i].append(qpos[i])

    def _draw(self):
        x_range = range(max(0, self.step_counter - BUFFER_LEN), self.step_counter)
        for i in range(NUM_JOINTS):
            self.real_lines[i].set_data(x_range, list(self.real_buf[i]))
            self.sim_lines[i].set_data(x_range, list(self.sim_buf[i]))
            ax = self.axes[i]
            ax.relim()
            ax.autoscale_view()
        plt.pause(0.001)

    # ---------------------- 公共方法 ----------------------
    def show(self, block: bool = True):
        """显示窗口；``block=False`` 时非阻塞。"""
        plt.ioff()  # 切换到 show 模式
        plt.show(block=block)



def main(args: Args):


    env_kwargs = dict(
        obs_mode="rgb+segmentation",
        #render_mode="sensors",
        render_mode="human",
        reward_mode="none",
        render_backend="cpu",
        # use larger camera resolution to make it easier to align. In training we won't use this however
        sensor_configs=dict(width=1280, height=1280)
    )


    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            env_kwargs.update   (json.load(f))
        print("Loaded env_kwargs from JSON:", env_kwargs)

    sim_env = gym.make(
        args.env_id,
        **env_kwargs,
    )
    print("1")
    sim_env = FlattenRGBDObservationWrapper(sim_env)
    # sim_env.reset()
    # while True:
    #     sim_env.render()
    real_robot, camera = create_real_robot()
    #real_robot.connect(calibrate=True)
    #camera.connect()
    #     real_robot.connect(calibrate=False)
    #     real_robot.calibrate()
    # real_robot.disconnect()
    real_agent = LeRobotRealAgent(real_robot)
    real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent)
    # safety setup, now ctrl+c will first reset the robot to a resting position and then close environments and turn of torque

    print("1.5")
    setup_safe_exit(sim_env, real_env, real_agent)
    print("2")
    real_obs, _ = real_env.reset()
    print("3")
    # for plotting robot camera reads
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    print("4")
    # Disable all default key bindings
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    fig.canvas.manager.key_press_handler_id = None

    overlaid_imgs, sim_imgs, real_imgs = overlay_envs(sim_env, real_env)
    axes[0].set_title("Real Camera")
    axes[1].set_title("Sim Camera")
    axes[2].set_title("Overlay")
    im_real = axes[0].imshow(real_imgs)
    im_sim = axes[1].imshow(sim_imgs)
    im_overlay = axes[2].imshow(overlaid_imgs)

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)

    init_camera_hyper = get_init_camera_pose(sim_env)

    stop_event = threading.Event()  # 线程安全的布尔开关

    def wait_for_enter():
        input()  # 用户在终端敲一次 Enter 即返回
        stop_event.set()  # 触发退出标志

    threading.Thread(target=wait_for_enter, daemon=True).start()
    # ------------------------------------------

    print("Adjust the true camera with your hands ... press Enter in the terminal when done")

#    plt.ion()  # 打开交互模式，确保 matplotlib 界面实时刷新
#    fig.show(block=False)  # 非阻塞显示

    print("adjust the true camera with your hands .... enter 'Enter' when done")

    # --- 固定一个仿真关节姿态，直接查看叠加效果 ---
    # 固定姿态：基座绕 Z 顺时针 90 度 -> joint1 = 0
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
    # 同步真机到同一姿态（谨慎执行，会驱动机械臂）
   # real_agent.set_target_qpos(fixed_qpos)
    print("Set fixed sim qpos:", sim_qpos)

    overlaid_imgs, sim_imgs, real_imgs = overlay_envs(sim_env, real_env)
    im_real.set_data(real_imgs)
    im_sim.set_data(sim_imgs)
    im_overlay.set_data(overlaid_imgs)
    fig.canvas.draw()
    fig.show()
    fig.canvas.flush_events()
    print("Close the figure window to exit.")

    # while plt.fignum_exists(fig.number):
    #     plt.pause(0.1)

    if hasattr(real_robot, "bus"):
        real_robot.bus.disable_torque()
    # assert  1== 2

    actiontuner = ActionTunerRunner(real_env.agent.robot.qpos.detach().cpu().numpy().squeeze().tolist())

    plotter = RealtimeQposPlotter()

    while True:
        if not actiontuner.pump_events():
            print("Action tuner window closed, exiting control loop.")
            break
        # ret, frame = cap.read()
        # if not ret:
        #     print("⚠️  读帧失败，退出循环")
        #     break
        #
        # cv2.imshow(window_name, frame)

        # 按 q 或 Esc 退出
        # key = cv2.waitKey(1) & 0xFF
        # if key in (ord('q'), 27):
        #     break

        # sim_qpos = env.agent.robot.get_qpos().cpu().flatten()
        action = np.array(actiontuner.app.v)
        # real_robot.set_target_qpos(sim_qpos)
        # current_qpos = real_robot.robot.get_qpos()

        real_env.step(action)

        delta = real_env.agent.robot.get_qpos().detach().cpu().squeeze().numpy() - real_env.base_sim_env.agent.robot.qpos.cpu().flatten().numpy()

        # plotter.feed(real_env.agent.robot.get_qpos().detach().cpu().squeeze().numpy(),
        #              real_env.base_sim_env.agent.robot.qpos.cpu().flatten().numpy())
        plotter.feed(delta, np.zeros([6]))
        # env.step(current_qpos)
        sim_env.render()

        print("target qpos", action, "Real qpos", real_env.agent.robot.get_qpos(), "Sim qpos",
              real_env.base_sim_env.agent.robot.qpos.cpu().flatten())

    #Eval sim from Real
    # while True:
    #
    #     current_qpos = real_agent.get_rad()
    #     print("current_qpos", current_qpos)
    #     # 将真实 7 轴对齐到仿真 13 轴：只覆盖前 7 维，夹爪等保持不变
    #     sim_qpos = sim_env.agent.robot.get_qpos()
    #     if current_qpos.shape[-1] < sim_qpos.shape[-1]:
    #         padded = sim_qpos.clone()
    #         padded[..., : current_qpos.shape[-1]] = current_qpos.to(sim_qpos.device)
    #         sim_env.agent.robot.set_qpos(padded)
    #     else:
    #         sim_env.agent.robot.set_qpos(current_qpos[..., : sim_qpos.shape[-1]])
    #     overlaid_imgs, sim_imgs, real_imgs = overlay_envs(sim_env, real_env)
    #     im_real.set_data(real_imgs)
    #     im_sim.set_data(sim_imgs)
    #     im_overlay.set_data(overlaid_imgs)
    #     # Update camera position based on active keys
    #     update_camera(sim_env)
    #
    #     sim_env.render()
    #     # Redraw the plot
    #     fig.canvas.draw()
    #     fig.show()
    #
    #     fig.canvas.flush_events()
    #     if not plt.fignum_exists(fig.number):
    #         print("The figure has been closed.")
    #         break
    #     print("YUN XIN CHENG GONG")
    # while True:
    #
    #     all_images = []
    #     all_losses = []
    #     for hyper_ind in hyper_population:
    #         set_camera_pose(sim_env, hyper_ind)
    #         loss, overlaid_imgs = calculate_error(sim_env, real_env)
    #         all_images.append(overlaid_imgs)
    #         all_losses.append(loss)
    #     best_index= np.argmin(all_losses)
    #     print(f"{iteraction}======================================================")
    #     print("Current best index", best_index, all_losses[best_index], hyper_population[best_index])
    #     # loss越小越好
    #
    #     top_id = np.argsort(all_losses)[:10]
    #     print(top_id)
    #     im.set_data(all_images[best_index])
    #
    #     new_hyper_population = list(sample_noisy_poses(hyper_population[top_id]))
    #     for top_i in top_id:
    #         new_hyper_population.append(hyper_population[top_i])
    #     hyper_population = np.array(new_hyper_population)
    #     # Update camera position based on active keys
    #     # update_camera(sim_env)
    #     # Redraw the plot
    #     fig.canvas.draw()
    #     fig.show()
    #
    #     fig.canvas.flush_events()
    #     iteraction +=1
    #     if not plt.fignum_exists(fig.number):
    #         print("The figure has been closed.")
    #         break


    #
    #
    # print("Camera alignment: Move real camera to align with the sim camera, close figure to exit")
    # while True:
    #     overlaid_imgs = overlay_envs(sim_env, real_env)
    #     im.set_data(overlaid_imgs)
    #     # Update camera position based on active keys
    #     update_camera(sim_env)
    #     # Redraw the plot
    #     fig.canvas.draw()
    #     fig.show()
    #     fig.canvas.flush_events()
    #     if not plt.fignum_exists(fig.number):
    #         print("The figure has been closed.")
    #         break

if __name__ == "__main__":
    args = tyro.cli(Args)
    print("env_id =", args.env_id)
    main(args)
