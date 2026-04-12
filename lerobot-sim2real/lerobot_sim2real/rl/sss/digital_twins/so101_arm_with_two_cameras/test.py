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


import json
def main():
    args = parse_args()

    with open("D:/Project/Scaling/lerobot-sim2real-main/lerobot-sim2real/so101_env_config.json", "r") as f:
        env_kwargs = json.load(f)



    env = gym.make(
        "SO101GraspCube_two_cameras-v1",
        obs_mode="none",
        reward_mode="none",
        enable_shadow=True,
        control_mode=args.control_mode,
        robot_uids=args.robot_uid,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        render_mode="human",
        sim_config=dict(sim_freq=args.sim_freq, control_freq=args.control_freq),
        sim_backend=args.sim_backend,
        **env_kwargs
    )
    env.reset(seed=0)
    env: BaseEnv = env.unwrapped
    print(f"Selected robot {args.robot_uid}. Control mode: {args.control_mode}")
    print("Selected Robot has the following keyframes to view: ")
    print(env.agent.keyframes.keys())
    env.agent.robot.set_qpos(env.agent.robot.qpos * 0)
    kf = None
    if len(env.agent.keyframes) > 0:
        kf_name = None
        if args.keyframe is not None:
            kf_name = args.keyframe
            kf = env.agent.keyframes[kf_name]
        else:
            for kf_name, kf in env.agent.keyframes.items():
                # keep the first keyframe we find
                break
        if kf.qpos is not None:
            env.agent.robot.set_qpos(kf.qpos)
            env.agent.controller.reset()
        if kf.qvel is not None:
            env.agent.robot.set_qvel(kf.qvel)
        env.agent.robot.set_pose(kf.pose)
        if kf_name is not None:
            print(f"Viewing keyframe {kf_name}")
    if env.gpu_sim_enabled:
        env.scene._gpu_apply_all()
        env.scene.px.gpu_update_articulation_kinematics()
        env.scene._gpu_fetch_all()


    env.render()

    #viewer.paused = True

    org_pos = env.agent.robot.qpos.cpu().flatten()

    actiontuner = ActionTunerRunner([-0.0775,  0.0829,  0.1028,  1.5512,  1.5604,  1.1193])
    import numpy as np
    while not actiontuner.app:
        time.sleep(0.1)

    while not hasattr(actiontuner.app, "v"):
        time.sleep(0.1)
   # print(""env.agent.robot.qpos.cpu().flatten())

    env.agent.robot.set_qpos([-0.0775,  0.0829,  0.1028,  1.5512,  1.5604,  1.1193])
    #
    # -1.571
    while True:
        action = np.array(actiontuner.app.v)
        env.step(action)
        env.render()
    # while True:
    #     if args.random_actions:
    #         env.step(env.action_space.sample())
    #     elif args.none_actions:
    #         env.step(None)
    #     elif args.zero_actions:
    #         env.step(env.action_space.sample() * 0)
    #     elif args.keyframe_actions:
    #         assert kf is not None, "this robot has no keyframes, cannot use it to set actions"
    #         if isinstance(env.agent.controller, DictController):
    #             env.step(env.agent.controller.from_qpos(kf.qpos))
    #         else:
    #             env.step(kf.qpos)
    #
    #     action = np.array(actiontuner.app.v)
    #     env.step(action)
    #     env.render()
    #     print("org_pos", org_pos, "new pos", env.agent.robot.qpos.cpu().flatten())


if __name__ == "__main__":
    main()
