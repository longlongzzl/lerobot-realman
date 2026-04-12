"""
This script is used to evaluate a random or RL trained agent on a real robot using the LeRobot system.
"""
import time
from dataclasses import dataclass
import json
import random
from typing import Optional
import gymnasium as gym
import numpy as np
import torch
import tyro
from lerobot_sim2real.config.real_robot import create_real_robot
from lerobot_sim2real.rl.single_ppo_rgb import Agent

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
    continuous_eval: bool = True
    """If toggled, the evaluation will run until episode ends without user input. If false, at each timestep the user will be prompted to press enter to let the robot continue"""
    max_episode_steps: int = 100
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

        print("???",name, real_obs.keys() , sim_obs.keys())
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


# ========== 初始化部分 ==========
# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('实时图像显示', fontsize=16)

# 初始化图像对象（先用空白图像占位）
im1 = ax1.imshow(np.zeros((224, 224, 3)))
im2 = ax2.imshow(np.zeros((224, 224, 3)))

ax1.set_title('图像 1')
ax2.set_title('图像 2')
ax1.axis('off')
ax2.axis('off')

plt.ion()  # 开启交互模式
plt.show()

def update_images(all_sim_imgs):
    """
    更新显示的两个图像

    Args:
        all_sim_imgs: 包含两个图像的列表，每个图像可以是 torch.Tensor 或 numpy.ndarray
    """
    assert len(all_sim_imgs) == 2, f"需要2个图像，但得到了 {len(all_sim_imgs)} 个"

    # 转换第一个图像
    img1 = to_numpy_rgb(all_sim_imgs[0])
    img2 = to_numpy_rgb(all_sim_imgs[1])

    # 更新图像数据
    im1.set_data(img1)
    im2.set_data(img2)

    # 重绘图形
    fig.canvas.draw()
    fig.canvas.flush_events()

import cv2
from mani_skill.utils import common
def main(args: Args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ### Create and connect the real robot, wrap it to make it interfaceable with ManiSkill sim2real environments ###
    real_robot = create_real_robot(uid="so101")
    real_robot.connect()
    real_agent = LeRobotRealAgent(real_robot)

    ### Setup the sim environment to make various checks for sim2real alignment and debugging possible ###
    env_kwargs = dict(
        obs_mode="rgb+segmentation",
        #render_mode="sensors",
        render_mode="human",
        render_backend="cpu",
        # only sensors mode is supported right now for real envs, basically rendering the direct visual observations fed to policy
        max_episode_steps=args.max_episode_steps,  # give our robot more time to try and re-try the task
        domain_randomization=False,
        reward_mode="none"
    )

    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            env_kwargs.update(json.load(f))

    sim_env = gym.make(
        args.env_id,
        **env_kwargs
    )

    # you can apply most wrappers freely to the sim_env and the real_env will use them as well
    sim_env = FlattenRGBDObservationWrapper(sim_env)
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
    # print(sim_env.control_mode)
    # agent_obs = sim_obs
    #
    #
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
    #         sim_imgs = overlaid_dict[name]["rgb"][0].cpu() / 255
    #         all_sim_imgs.append(sim_imgs)
    #
    #     assert len(all_sim_imgs) == 2
    #     update_images(all_sim_imgs)


    # assert  1==2
    # The Sim2RealEnv class uses the sim_env to help make various checks for sim2real alignment (e.g. observation space is the same, cameras are the similar)
    # and will always try its best to apply all wrappers you used on the sim env to the real env as well.
    real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent, control_freq=args.control_freq)


    # sim_env.print_sim_details()
    sim_obs, infos = sim_env.reset()


    real_obs, _ = real_env.reset()

    for k in sim_obs.keys():
        print(
            f"{k}: sim_obs shape: {sim_obs[k].shape}, real_obs shape: {real_obs[k].shape}"
        )


    ### Safety setups. Close environments/turn off robot upon ctrl+c ###
    setup_safe_exit(sim_env, real_env, real_agent)

    # real_robot.bus.disable_torque()
    # while True:
    #     sim_env.agent.robot.set_qpos(real_agent.get_qpos())
    #     sim_env.render()

    ### Load our checkpoint ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(sim_env, device, sample_obs=real_obs)
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded agent from {args.checkpoint}")
    else:
        print("No checkpoint provided, using random agent")
    agent.to(device)



    frames = []
    # ### Visualization setup for debug modes ###
    if args.debug:
        #fig = plt.figure()
        fig = plt.figure(figsize=(16, 12))  # 宽 8"，高 12"
        ax = fig.add_subplot(3, 2, 1)
        ax2 = fig.add_subplot(3, 2, 3)
        ax3 = fig.add_subplot(3, 2, 5)


        ax4 = fig.add_subplot(3, 2, 2)
        ax5 = fig.add_subplot(3, 2, 4)
        ax6 = fig.add_subplot(3, 2, 6)
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
        overlaid_imgs_1, real_imgs_1, sim_imgs_1 = overlay_envs(sim_env, real_env, 1)
        im_1 = ax4.imshow(overlaid_imgs_1)
        im2_1 = ax5.imshow(sim_imgs_1)
        im3_1 = ax6.imshow(real_imgs_1)


    ### Main evaluation loop ###
    episode_count = 0

    qpos_list = []

    print("Start!")
    action_seq = [[[0.2724, 3.6818, 2.1886, -2.8970, -2.2147, -0.3739]],
                  [[0.0130, 2.8564, 1.7193, -2.3898, -1.9116, -0.2424]],
                  [[-0.5116, 1.1299, 0.1904, -1.8268, -1.4462, -0.5183]],
                  [[-0.9209, -0.5661, -1.4775, -1.5103, -0.8792, -0.6900]],
                  [[-0.7751, -0.6836, -1.4249, -1.0669, -0.3575, -0.7398]]]


    while args.num_episodes is None or episode_count < args.num_episodes:
        print(f"Evaluation Episode {episode_count}")
        for _ in tqdm(range(args.max_episode_steps)):


            #real_obs =  sim_env.get_obs()["sensor_data"]

            agent_obs = real_obs

            agent_obs = {k: v.to(device) for k, v in agent_obs.items()}
           # action = agent.get_action(agent_obs)
            action = torch.tensor(action_seq[_])
            print("!!!!!!!!!!!! action", action)
            if not args.continuous_eval:
                input("Press enter to continue to next timestep")

            real_obs, _, terminated, truncated, info = real_env.step(action.detach().cpu().numpy())
            #print("current qpos",real_env.agent.robot.get_qpos())
            #print("!!!", sim_env.control_mode)
            #sim_env.render()
            qpos_list.append(action.detach().cpu().numpy().squeeze())
            sim_env.render()
            t3 = time.time()
            # time.sleep(0.01)
           # print(action, "Inference time", t2- t1, "sim time", t3- t2)

            overlaid_dict = sim_env.get_obs()["sensor_data"]
            all_sim_imgs = []
            for name in overlaid_dict:
                print(name)
                sim_imgs = overlaid_dict[name]["rgb"][0].cpu() / 255
                all_sim_imgs.append(sim_imgs)

            #assert len(all_sim_imgs) == 2
            #update_images(all_sim_imgs)

            if args.debug:
                overlaid_imgs, real_imgs, sim_imgs = overlay_envs(sim_env, real_env)
                im.set_data(overlaid_imgs)
                im2.set_data(sim_imgs)
                im3.set_data(real_imgs)
                # Redraw the plot
                fig.canvas.draw()
                fig.show()
                fig.canvas.flush_events()

                overlaid, real, sim = overlay_envs(sim_env, real_env, 0)
                overlaid = to_rgb(overlaid)


                im.set_data(overlaid)
                im2.set_data(sim)
                im3.set_data(real)

                overlaid_1, real_1, sim_1 = overlay_envs(sim_env, real_env, 1)
                overlaid_1 = to_rgb(overlaid_1)
                im_1.set_data(overlaid_1)
                im2_1.set_data(sim_1)
                im3_1.set_data(real_1)

                # Redraw the plot
                fig.canvas.draw()
                fig.show()
                fig.canvas.flush_events()



                real = to_rgb(real)
                sim = to_rgb(sim)

                # 把三张图高方向拼起来：H_total = 3H，W = W
                frame = np.vstack([overlaid, sim, real])

                frames.append(frame)  # ★ 收集这一帧

        episode_count += 1
        real_env.reset()

        # from utils import plot_action_sequence
        #
        # plot_action_sequence(qpos_list, len(qpos_list) * (1 / 20))
        #

        if args.debug and frames:
            print("Done .... store ......")
            fps = 20
            h, w, _ = frames[0].shape
            out = cv2.VideoWriter("debug_video.mp4",
                                  cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps, (w, h))
            for f in frames:
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out.release()
            print(f"[√] Saved {len(frames)} frames to debug_video.mp4")

    sim_env.close()
    real_env.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)