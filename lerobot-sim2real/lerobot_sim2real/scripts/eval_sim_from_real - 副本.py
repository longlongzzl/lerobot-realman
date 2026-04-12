import json
import time
from typing import Optional
import gymnasium as gym
import torch
from lerobot_sim2real.utils.safety import setup_safe_exit
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

# def main(args: Args):
#     real_robot = create_real_robot(uid="so100")
#     real_robot.connect()
# #     real_robot.connect(calibrate=False)
# #     real_robot.calibrate()
#    # real_robot.disconnect()
#
#       setup_safe_exit(sim_env, real_env, real_agent)

class LeRobotRealAgentWithQposTracking:
    def __init__(self, real_robot):
        self.real_robot = real_robot
        # 初始化min和max为非常大的正值和负值
        self.min_qpos = torch.full_like(torch.empty(1), float('inf'))
        self.max_qpos = torch.full_like(torch.empty(1), float('-inf'))

    def update_min_max_qpos(self, current_qpos):
        # 更新最小值和最大值
        self.min_qpos = torch.min(self.min_qpos, current_qpos)
        self.max_qpos = torch.max(self.max_qpos, current_qpos)

    def get_qpos_range(self):
        return self.min_qpos, self.max_qpos



import argparse

import gymnasium as gym
import mani_skill
from mani_skill.agents.controllers.base_controller import DictController
from mani_skill.envs.sapien_env import BaseEnv
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--robot-uid", type=str, default="so100", help="The id of the robot to place in the environment")
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("-c", "--control-mode", type=str, default="pd_joint_pos", help="The control mode to use. Note that for new robots being implemented if the _controller_configs is not implemented in the selected robot, we by default provide two default controllers, 'pd_joint_pos' and 'pd_joint_delta_pos' ")
    parser.add_argument("-k", "--keyframe", type=str, help="The name of the keyframe of the robot to display")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--keyframe-actions", action="store_true", help="Whether to use the selected keyframe to set joint targets to try and hold the robot in its position")
    parser.add_argument("--random-actions", action="store_true", help="Whether to sample random actions to control the agent. If False, no control signals are sent and it is just rendering.")
    parser.add_argument("--none-actions", action="store_true", help="If set, then the scene and rendering will update each timestep but no joints will be controlled via code. You can use this to control the robot freely via the GUI.")
    parser.add_argument("--zero-actions", action="store_true", help="Whether to send zero actions to the robot. If False, no control signals are sent and it is just rendering.")
    parser.add_argument("--sim-freq", type=int, default=100, help="Simulation frequency")
    parser.add_argument("--control-freq", type=int, default=20, help="Control frequency")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and environment. Default is no seed",
    )
    args = parser.parse_args()
    return args

import time
import tkinter as tk
from tkinter import ttk
import threading
import cv2


class ActionTuner:
    dim = 6

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
    def __init__(self, org_pos):
        self.app = None
        self.org_pos = org_pos
        # 创建并启动GUI线程
        gui_thread = threading.Thread(target=self.run_gui)
        gui_thread.daemon = True  # 守护线程，主线程结束时自动退出
        gui_thread.start()

    def run_gui(self):
        """运行Tkinter GUI的线程"""
        self.app = ActionTuner(self.org_pos)
        self.app.root.mainloop()


#
# def main():
#     args = parse_args()
#     env = gym.make(
#         "SO101GraspCube-v1",
#         obs_mode="none",
#         reward_mode="none",
#         enable_shadow=True,
#         control_mode=args.control_mode,
#         robot_uids=args.robot_uid,
#         sensor_configs=dict(shader_pack=args.shader),
#         human_render_camera_configs=dict(shader_pack=args.shader),
#         viewer_camera_configs=dict(shader_pack=args.shader),
#         render_mode="human",
#         sim_config=dict(sim_freq=args.sim_freq, control_freq=args.control_freq),
#         sim_backend=args.sim_backend,
#     )
#     env.reset(seed=0)
#     env: BaseEnv = env.unwrapped
#     print(f"Selected robot {args.robot_uid}. Control mode: {args.control_mode}")
#     print("Selected Robot has the following keyframes to view: ")
#     print(env.agent.keyframes.keys())
#     env.agent.robot.set_qpos(env.agent.robot.qpos * 0)
#     kf = None
#     if len(env.agent.keyframes) > 0:
#         kf_name = None
#         if args.keyframe is not None:
#             kf_name = args.keyframe
#             kf = env.agent.keyframes[kf_name]
#         else:
#             for kf_name, kf in env.agent.keyframes.items():
#                 # keep the first keyframe we find
#                 break
#         if kf.qpos is not None:
#             env.agent.robot.set_qpos(kf.qpos)
#             env.agent.controller.reset()
#         if kf.qvel is not None:
#             env.agent.robot.set_qvel(kf.qvel)
#         env.agent.robot.set_pose(kf.pose)
#         if kf_name is not None:
#             print(f"Viewing keyframe {kf_name}")
#     if env.gpu_sim_enabled:
#         env.scene._gpu_apply_all()
#         env.scene.px.gpu_update_articulation_kinematics()
#         env.scene._gpu_fetch_all()
#
#
#     env.render()
#
#     #viewer.paused = True
#
#     org_pos = env.agent.robot.qpos.cpu().flatten()
#
#     actiontuner = ActionTunerRunner(org_pos.squeeze().numpy().tolist())
#     import numpy as np
#     while not actiontuner.app:
#         time.sleep(0.1)
#
#     while not hasattr(actiontuner.app, "v"):
#         time.sleep(0.1)
#    # print(""env.agent.robot.qpos.cpu().flatten())
#     while True:
#         if args.random_actions:
#             env.step(env.action_space.sample())
#         elif args.none_actions:
#             env.step(None)
#         elif args.zero_actions:
#             env.step(env.action_space.sample() * 0)
#         elif args.keyframe_actions:
#             assert kf is not None, "this robot has no keyframes, cannot use it to set actions"
#             if isinstance(env.agent.controller, DictController):
#                 env.step(env.agent.controller.from_qpos(kf.qpos))
#             else:
#                 env.step(kf.qpos)
#
#         action = np.array(actiontuner.app.v)
#         env.step(action)
#         env.render()
#         print("org_pos", org_pos, "new pos", env.agent.robot.qpos.cpu().flatten())
#

import cv2

# ---------- 参数 ----------
camera_index = 0  # 0 = 第一台摄像头；1 = 第二台……
window_name = "LiveCam"  # 窗口标题
width, height = 1280, 720  # 期望分辨率，可按需修改

cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # 不设置分辨率

if not cap.isOpened():
    raise RuntimeError(f"无法打开摄像头 {camera_index}")

# # 读一次实际分辨率
# default_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# default_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 设置期望宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # 设置期望高度

# import threading
# import numpy as np
#
# delta = np.zeros(6, dtype=float)
# lock = threading.Lock()
#
# STEP = 0.01  # 步长，可按需修改
#
# # 建立映射表：命令 -> (index, sign)
# # 正数表示加，负数表示减
# command_map = {
#     '1': (0, +1),
#     '2': (1, +1),
#     '3': (2, +1),
#     '4': (3, +1),
#     '5': (4, +1),
#     '6': (5, +1),
#
#     'q': (0, -1),
#     'w': (1, -1),
#     'e': (2, -1),
#     'r': (3, -1),
#     't': (4, -1),
#     'y': (5, -1),
# }
#
# def apply_command(cmd: str):
#     """
#     根据指令更新 delta.
#     1-6 对应各维 +STEP
#     q-w-e-r-t-y 对应各维 -STEP
#     其它指令忽略或自行扩展
#     """
#     global delta
#     with lock:
#         if cmd in command_map:
#             idx, sgn = command_map[cmd]
#             delta[idx] += sgn * STEP
#             print(f"[cmd={cmd}] idx={idx} {'+' if sgn > 0 else '-'}{STEP:.4f} -> delta={delta}")
#         elif cmd == 'reset':
#             delta[:] = 0
#             print(f"[cmd=reset] delta 已清零 -> {delta}")
#         else:
#             # 未知命令可选择打印或静默
#             print(f"[cmd={cmd}] 未知指令，忽略。当前 delta={delta}")
#
delta = np.zeros(6)
STEP = 0.01
lock = threading.Lock()
stop_flag = {'stop': False}

command_map = {
    '1': (0, +1),'2': (1, +1),'3': (2, +1),
    '4': (3, +1),'5': (4, +1),'6': (5, +1),
    'z': (0, -1),'x': (1, -1),'c': (2, -1),
    'v': (3, -1),'b': (4, -1),'n': (5, -1),
}
from pynput import keyboard
def on_press(key):
    try:
        ch = key.char.lower()
    except:
        # 特殊键
        if key == keyboard.Key.esc:
            stop_flag['stop'] = True
        return

    if ch in command_map:
        idx, sgn = command_map[ch]
        with lock:
            delta[idx] += sgn * STEP
            print(f"[{ch}] idx={idx} {'+' if sgn>0 else '-'}{STEP} -> delta={delta}")
    elif ch == ' ':
        with lock:
            delta[:] = 0
            print("[space] reset delta")

listener = keyboard.Listener(on_press=on_press)
listener.start()

import threading, sys, queue

cmd_queue = queue.Queue()

def stdin_thread():
    print("输入命令，例如: 1 / q / reset / exit，然后回车")
    for line in sys.stdin:
        cmd = line.strip().lower()
        cmd_queue.put(cmd)
        if cmd in ('exit','quit'):
            break

threading.Thread(target=stdin_thread, daemon=True).start()


def main(args: Args):
    # args = parse_args()
    # env = gym.make(
    #     "SO101GraspCube-v1",
    #     obs_mode="none",
    #     reward_mode="none",
    #     enable_shadow=True,
    #     control_mode=args.control_mode,
    #     sensor_configs=dict(shader_pack=args.shader),
    #     human_render_camera_configs=dict(shader_pack=args.shader),
    #     viewer_camera_configs=dict(shader_pack=args.shader),
    #     render_mode="human",
    #     sim_config=dict(sim_freq=args.sim_freq, control_freq=args.control_freq),
    #     sim_backend=args.sim_backend,
    # )
    #
    # env.reset()
    # target_qpos = env.agent.robot.get_qpos()
    # qpos1 = np.array([0, -1.5708, 1.5708, 0.66, 0, -1.1])
    # qpos2 = target_qpos.detach().cpu().numpy()
    #
    # qpos1_delta_list =[]
    # qpos2_delta_list = []
    #
    # for i in range(10):
    #     real_robot = create_real_robot(uid="so101")
    #     real_robot.connect()
    #     real_robot = LeRobotRealAgent(real_robot)
    #     real_robot.reset(qpos2)
    #     time.sleep(5.0)
    #     real_qpos2 = real_robot.get_rad()
    #     qpos2_delta_list.append(qpos2 - real_qpos2.detach().cpu().numpy())
    #    # real_robot.stop()
    #
    #     real_robot.reset(qpos1)
    #
    #     time.sleep(3.0)
    #     real_robot.stop()
    # print(np.mean(qpos2_delta_list, 0))
    #
    # assert  1==2

    real_robot = create_real_robot(uid="so100")
    real_robot.connect()
    real_robot.bus.disable_torque()
    #real_robot.calibrate()
    #real_robot.disconnect()
    #assert 1 == 2
 #    real_robot.connect(calibrate=False)
 #    real_robot.calibrate()
 #    real_robot.disconnect()
    agent = LeRobotRealAgentWithQposTracking(real_robot)
    real_robot = LeRobotRealAgent(real_robot)



    args = parse_args()
    env = gym.make(
        "SO100GraspCube-v1",
        obs_mode="none",
        reward_mode="none",
        enable_shadow=True,
        control_mode=args.control_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        render_mode="human",
        sim_config=dict(sim_freq=args.sim_freq, control_freq=args.control_freq),
        sim_backend=args.sim_backend,
    )
    env.reset(seed=0)
    # while True:
    #     env.render()
    qpos = env.agent.robot.qpos.cpu().flatten()
    #real_robot.reset(qpos*0.0)
    #real_robot.set_target_qpos(qpos*0.0)
    env: BaseEnv = env.unwrapped
    print(f"Selected robot {args.robot_uid}. Control mode: {args.control_mode}")
    print("Selected Robot has the following keyframes to view: ")
    print(env.agent.keyframes.keys())
    env.reset()
    env.agent.robot.set_qpos(env.agent.robot.qpos * 0)

    pos1 = np.array([0, -1.5708, 1.5708, 0.66, 0, -1.1])
    pos2 = np.zeros([6])




    # while True:
    #   # real_robot.set_target_qpos(qpos * 0.0)
    #     current_qpos = real_robot.get_rad()
    #
    #     env.step(current_qpos)
    #     env.render()
    #     print("current_qpos", current_qpos, env.agent.robot.get_qpos())
    #     print("current delta", env.agent.robot.get_qpos() - real_robot.get_rad() )
    #
    #
    #




    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("⚠️  读帧失败，退出循环")
    #         break
    #     key = cv2.waitKey(1) & 0xFF
    #     if key != 0xFF:  # 有按键
    #         if key in (27, ord('x')):  # Esc 或 x 退出
    #             print("退出请求")
    #             break
    #         apply_command(chr(key))
    #     #cv2.imshow(window_name, frame)
    #
    #     # 按 q 或 Esc 退出
    #     # key = cv2.waitKey(1) & 0xFF
    #     # if key in (ord('q'), 27):
    #     #     break
    #
    #
    #     current_qpos = real_robot.get_rad()
    #     current_qpos +=delta
    #     print("current delta", delta)
    #
    #     env.step(current_qpos)
    #     env.render()
    #     print("Real qpos", current_qpos, "Sim qpos", env.agent.robot.qpos.cpu().flatten())
    #
    #


    while True:



        current_qpos = real_robot.get_rad()  # 获取当前qpos
        print(current_qpos)
        agent.update_min_max_qpos(current_qpos)  # 更新min和max值
        min_qpos, max_qpos = agent.get_qpos_range()  # 获取当前最小和最大qpos

        print("min + max", min_qpos, max_qpos )
        # min_qpos_str = ', '.join([f'{val:.4f}' for val in min_qpos.flatten()])
        # max_qpos_str = ', '.join([f'{val:.4f}' for val in max_qpos.flatten()])
        #
        # # 使用回车符覆盖上一行
        # print(f"Min qpos: [{min_qpos_str}] | Max qpos: [{max_qpos_str}]", end="\r", flush=True)

    real_robot.get_qpos()
    joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    target_rad = np.zeros([len(joints)])
    real_robot.reset(qpos=target_rad)
    # time.sleep(3.0)
    samples = []
    for _ in range(10):
        qpos_deg = real_robot.real_robot.bus.sync_read("Present_Position")
        samples.append([qpos_deg[j] for j in joints])
        time.sleep(0.05)
    mean_deg = np.mean(samples, axis=0)

    theory_deg = np.rad2deg(target_rad)
    offsets = mean_deg - theory_deg
    print("---- so101 joint offset（°）----")
    for j, off in zip(joints, offsets):
        print(f"{j:15s}: {off:+.3f}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)