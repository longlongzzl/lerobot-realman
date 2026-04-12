import json
import time
from typing import Optional
import gymnasium as gym
import torch
from lerobot_sim2real.utils.safety import setup_safe_exit
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
import cv2
import numpy as np
import tyro
from mani_skill.utils.visualization.misc import tile_images
from mani_skill.utils import sapien_utils
from dataclasses import dataclass

@dataclass
class Args:
    env_id: str = "SO101GraspCube-v1"
    """The environment id to train on"""
    env_kwargs_json_path: Optional[str] = None
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
    for name in overlaid_dict:
        real_imgs = real_obs[name]["rgb"][0] / 255
        sim_imgs = overlaid_dict[name]["rgb"][0].cpu() / 255
        overlaid_imgs.append(0.5 * real_imgs + 0.5 * sim_imgs)

    return tile_images(overlaid_imgs)


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
        camera_offset[0] -= MOVEMENT_SPEED * delta_time  # Move forward
    if "s" in active_keys:
        camera_offset[0] += MOVEMENT_SPEED * delta_time  # Move back
    if "d" in active_keys:
        camera_offset[1] += MOVEMENT_SPEED * delta_time  # Move right
    if "a" in active_keys:
        camera_offset[1] -= MOVEMENT_SPEED * delta_time  # Move left
    if "up" in active_keys:
        camera_offset[2] += MOVEMENT_SPEED * delta_time  # Move up
    if "down" in active_keys:
        camera_offset[2] -= MOVEMENT_SPEED * delta_time  # Move down

    # FOV control
    if "left" in active_keys:
        fov_offset -= FOV_CHANGE_SPEED * delta_time
    if "right" in active_keys:
        fov_offset += FOV_CHANGE_SPEED * delta_time

    # update camera position and fov
    pos = sim_env.unwrapped.base_camera_settings["pos"] + camera_offset
    pose = sapien_utils.look_at(pos, sim_env.unwrapped.base_camera_settings["target"])
    sim_env.unwrapped.camera_mount.set_pose(pose)
    sim_env.unwrapped._sensors["base_camera"].camera.set_fovy(
        sim_env.unwrapped.base_camera_settings["fov"] + fov_offset
    )

    if len(active_keys) > 0:
        print("current_camera_position", pose.p)
        print(
            "current_camera_fov",
            sim_env.unwrapped.base_camera_settings["fov"] + fov_offset,
        )
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



import time
import tkinter as tk
from tkinter import ttk
import threading
import cv2


class ActionTuner:
    dim = 6

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Action Tuner")
        self.v = [0.0] * self.dim

        # 设置初始Action值
        self.vals = []
        for i in range(self.dim):
            self.vals.append(tk.DoubleVar(value=0.0))

        # 创建滑块
        for i in range(self.dim):
            self.create_slider(f"{i}", self.vals[i], -1.0, 1.0, i)

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
    def __init__(self):
        self.app = None
        # 创建并启动GUI线程
        gui_thread = threading.Thread(target=self.run_gui)
        gui_thread.daemon = True  # 守护线程，主线程结束时自动退出
        gui_thread.start()

    def run_gui(self):
        """运行Tkinter GUI的线程"""
        self.app = ActionTuner()
        self.app.root.mainloop()


actiontuner = ActionTunerRunner()

while not actiontuner.app:
    time.sleep(0.1)

while not hasattr(actiontuner.app, "v"):
    time.sleep(0.1)


def main(args: Args):


    env_kwargs = dict(
        obs_mode="rgb+segmentation",
        #render_mode="sensors",
        render_mode="human",
        reward_mode="none",
        render_backend="cpu",
        # use larger camera resolution to make it easier to align. In training we won't use this however
        sensor_configs=dict(width=512, height=512)
    )
    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            env_kwargs.update(json.load(f))
    sim_env = gym.make(
        args.env_id,
        **env_kwargs,
    )
    print("1")
    sim_env = FlattenRGBDObservationWrapper(sim_env)
    sim_env.reset()
    sim_env.agent.robot.set_qpos(actiontuner.app.v)

    while True:
        # action = np.array(actiontuner.app.v)
        # print(action)
        # obs, reward, terminated, truncated, info = sim_env.step(action)
        sim_env.render()  # a display is required to render


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)