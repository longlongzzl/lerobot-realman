import copy
import json
import time
from typing import Optional
import gymnasium as gym
import torch
#from pybullet_examples.inverse_dynamics import q_pos

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



uid = 101
@dataclass
class Args:
    env_id: str = f"SO{uid}GraspCube_two_cameras-v1"
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
    sim_imgs_list =[]
    for name in overlaid_dict:
        real_imgs = real_obs[name]["rgb"][0] / 255
        sim_imgs = overlaid_dict[name]["rgb"][0].cpu() / 255
        overlaid_imgs.append(0.5 * real_imgs + 0.5 * sim_imgs)
        sim_imgs_list.append(sim_imgs)
        loss = torch.mean((real_imgs - sim_imgs)**2)
        print("current loss", loss)

    return tile_images(overlaid_imgs), tile_images(sim_imgs_list)


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
    current_list = list(pos.detach().cpu().numpy())
    current_list.append(fov)
    return current_list


def set_camera_pose(sim_env, settings):

    pose = sapien_utils.look_at(torch.tensor(settings[:3], dtype=torch.float32), sim_env.unwrapped.base_camera_settings["target"])
    sim_env.unwrapped.camera_mount.set_pose(pose)

   # sim_env.unwrapped.camera_mount.set_pose(torch.tensor(settings[:3], dtype=torch.float32))
    sim_env.unwrapped._sensors["base_camera"].camera.set_fovy( settings[-1])




times= 0.1


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
        camera_offset[0] -= 0.01*times  # Move forward
    if "s" in active_keys:
        camera_offset[0]  += 0.01*times  # Move back
    if "d" in active_keys:
        camera_offset[1]  += 0.01*times  # Move right
    if "a" in active_keys:
        camera_offset[1] -= 0.01*times  # Move left
    if "up" in active_keys:
        camera_offset[2]  += 0.01*times  # Move up
    if "down" in active_keys:
        camera_offset[2] -= 0.01*times  # Move down

    camera_target_offset = torch.zeros(3, dtype=torch.float32)

    if "y" in active_keys:
        camera_target_offset[0] -= 0.01*times # Move forward
    if "h" in active_keys:
        camera_target_offset[0] += 0.01*times # Move back
    if "u" in active_keys:
        camera_target_offset[1] += 0.01*times  # Move right
    if "j" in active_keys:
        camera_target_offset[1] -= 0.01*times  # Move left
    if "i" in active_keys:
        camera_target_offset[2] += 0.01*times  # Move right
    if "k" in active_keys:
        camera_target_offset[2] -= 0.01*times  # Move left

    # FOV control
    if "left" in active_keys:
        fov_offset -=  0.01*times
    if "right" in active_keys:
        fov_offset +=  0.01*times

    # update camera position and fov
    pos = sim_env.unwrapped.base_camera_2_settings["pos"] + camera_offset

    sim_env.unwrapped.base_camera_2_settings["target"] = sim_env.unwrapped.base_camera_2_settings["target"] + camera_target_offset
    pose = sapien_utils.look_at(pos, sim_env.unwrapped.base_camera_2_settings["target"])
    sim_env.unwrapped.camera_mount_2.set_pose(pose)
    sim_env.unwrapped._sensors["base_camera_2"].camera.set_fovy(
        sim_env.unwrapped.base_camera_2_settings["fov"] + fov_offset
    )

    if len(active_keys) > 0:

        print("current_camera_position", pose.p)
        print(
            "current_camera_fov",
            sim_env.unwrapped.base_camera_2_settings["fov"] + fov_offset,
        )
        print("camera target",  sim_env.unwrapped.base_camera_2_settings["target"])
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
            env_kwargs.update   (json.load(f))
    sim_env = gym.make(
        args.env_id,
        **env_kwargs,
    )
    print("1")
    sim_env = FlattenRGBDObservationWrapper(sim_env)
    # sim_env.reset()
    # while True:
    #     sim_env.render()

    real_robot = create_real_robot(uid=f"so{uid}")
    real_robot.connect(calibrate=True)
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
    fig = plt.figure()
    ax = fig.add_subplot()
    print("4")
    # Disable all default key bindings
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    fig.canvas.manager.key_press_handler_id = None

    # initialize the plot
    im = ax.imshow(overlay_envs(sim_env, real_env)[0])

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
    while not stop_event.is_set():
        real_image = get_real_image(real_env)
        im.set_data(real_image)
        fig.canvas.draw()
        fig.show()
        fig.canvas.flush_events()
        plt.pause(0.001)
    hyper_population = sample_noisy_poses([init_camera_hyper])

    print(init_camera_hyper)
    set_camera_pose(sim_env, init_camera_hyper)

    # 添加0到0.1的高斯噪声，获得100个个体
    loss, overlaid_imgs = calculate_error(sim_env, real_env)
    im.set_data(overlaid_imgs)
    fig.canvas.draw()
    fig.show()
    fig.canvas.flush_events()



    # iteraction = 0
    #
    # # TODO: very useful!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    real_robot.bus.disable_torque()
    # # assert  1 == 2
    # period = 1.0 / 15
    # t0 = time.perf_counter()
    # next_tick = t0
    # qpos_list = []
    # for i in range(50):
    #     current_qpos = real_agent.get_rad()
    #     print(current_qpos)
    #     qpos_list.append(current_qpos.detach().cpu().numpy().squeeze())
    #     next_tick += period
    #     sleep_time = next_tick - time.perf_counter()
    #     if sleep_time > 0:
    #         time.sleep(sleep_time)
    #
    #     else:
    #         # 若循环本身超时，马上开始下一周期并重置时钟偏差
    #         next_tick = time.perf_counter()
    # from utils import plot_action_sequence
    #
    # plot_action_sequence(qpos_list, len(qpos_list)*period)
    #
    # assert  1== 2

    while True:

        current_qpos = real_agent.get_rad()
        print("current_qpos", current_qpos)
        #sim_env.step(current_qpos)
        sim_env.agent.robot.set_qpos(current_qpos)
        overlaid_imgs, sim_imgs = overlay_envs(sim_env, real_env)
        im.set_data(overlaid_imgs)
        # Update camera position based on active keys
        update_camera(sim_env)

        sim_env.render()
        # Redraw the plot
        fig.canvas.draw()
        fig.show()

        fig.canvas.flush_events()
        if not plt.fignum_exists(fig.number):
            print("The figure has been closed.")
            break

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
    main(args)