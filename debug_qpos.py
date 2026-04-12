import mani_skill.envs  # 确保 ManiSkill 环境注册
import gymnasium as gym
import torch

from lerobot_sim2real.config.real_robot import create_real_robot


def main():
    env_kwargs = dict(
        obs_mode="rgb+segmentation",
        render_mode="none",
        reward_mode="none",
        render_backend="cpu",
        sensor_configs=dict(width=128, height=128),
        domain_randomization_config=dict(
            max_camera_offset=[0.0, 0.0, 0.0],
            camera_target_noise=0.0,
            camera_view_rot_noise=0.0,
            camera_fov_noise=0.0,
        ),
    )
    env = gym.make("RM75GraspCube_two_cameras-v1", **env_kwargs)
    env.reset()
    tensor = torch.as_tensor(env.agent.controller.qpos)
    print("Sim controller qpos tensor:", tensor)
    print("Sim shape:", tensor.shape)
    env.close()

    real_robot, _ = create_real_robot(auto_connect=False)
    real_robot.connect()
    obs = real_robot.get_observation()
    joint_keys = sorted(k for k in obs.keys() if k.endswith(".pos"))
    real_qpos = [obs[key] for key in joint_keys]
    real_tensor = torch.as_tensor(real_qpos, dtype=torch.float32)
    print("Real robot qpos tensor:", real_tensor)
    print("Real shape:", real_tensor.shape)
    real_robot.disconnect()


if __name__ == "__main__":
    main()
