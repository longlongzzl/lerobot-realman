from typing import Any, Dict, List, Union

import os
import numpy as np
import sapien
import torch

try:
    from mani_skill.agents.robots.xarm6.xarm6_inspire_hand_right import XArm6InspireHandRight
except ImportError:
    XArm6InspireHandRight = Any

try:
    from mani_skill.agents.robots.fetch.fetch import Fetch
except ImportError:
    Fetch = Any

try:
    from mani_skill.agents.robots.panda.panda import Panda
except ImportError:
    Panda = Any

try:
    from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
except ImportError:
    PandaWristCam = Any

from mani_skill.agents.robots.realman import RM75Robot
try:
    from mani_skill.agents.robots import RM75RightHand
except ImportError:
    RM75RightHand = RM75Robot
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.agents.utils import get_active_joint_indices
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.agents.robots.realman import RM75Robot
from transforms3d.euler import euler2quat
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
@register_env("Two_finger_PickJiaobang-v1", max_episode_steps=500, override=True)
class PickJiaobangEnv(BaseEnv):
    """
    Pick a glue stick (胶棒) modeled by `jiaobang.glb` on the table and move it
    to a goal **pose** (position + orientation). Success requires both:
    - object position within goal_thresh of goal position
    - object orientation within goal_angle_thresh (rad) of goal orientation.
    """

    _sample_video_link = ""

    SUPPORTED_ROBOTS = [
        "panda",
        "panda_wristcam",
        "fetch",
        "xarm6_inspire_hand_right",
        "RM75_right_hand",
        "RM75"
    ]
    agent: Union[Panda, PandaWristCam, Fetch, XArm6InspireHandRight, RM75RightHand, RM75Robot]
    goal_thresh = 0.01  # position tolerance (m)
    goal_angle_thresh = 5.0  # orientation tolerance (rad), ~8.6 deg
    reward_scales = dict(arm_vel=0.03, hand_vel=0.003)  # 速度平滑惩罚系数（L1）

    # --- dex-style gripper-to-shortest-edge alignment (cached at reset) ---
    obj_xy_shortest_edge_vector: torch.Tensor
    obj_local_aabb_min: torch.Tensor
    obj_local_aabb_max: torch.Tensor

    def _sample_new_goal(self, env_idx: torch.Tensor):
        """在当前物体位置附近为指定 env 采样新的 goal。

        约束：
        - goal XY ∈ [-0.1, 0.1] × [-0.1, 0.1]
        - goal 与当前物体在 XY 平面上的距离 ≥ 0.05 m（5 cm）
        - goal 高度：在物体底面高度基础上抬升 [0.0, 0.03] m

        注意：`goal_jiaobang` 是 merge 后的 Actor，这里必须构造整批 (num_envs, 7)
        的 pose，再只替换 env_idx 对应的行，避免 shape mismatch。
        """
        with torch.device(self.device):
            # env_idx 可能来自 torch.nonzero，先确保是 long 类型的一维索引
            if not torch.is_tensor(env_idx):
                env_idx = torch.as_tensor(env_idx, device=self.device)
            env_idx = env_idx.long().flatten()
            b = len(env_idx)
            if b == 0:
                return

            # 当前所有 goal 的 pose：(N, 7)，前 3 维是 p，后 4 维是 q
            goal_pose_all = self.goal_jiaobang.pose.raw_pose.clone().to(self.device)
            # 当前所有物体位置
            obj_pos_all = self.obj.pose.p  # (N, 3)

            # 只取本次需要更新的 env 的物体位置
            obj_pos = obj_pos_all[env_idx]  # (b, 3)

            # 在 [-0.1, 0.1]^2 中采样，强制与 obj_pos 在 XY 至少 5 cm
            goal_xy = torch.zeros((b, 2), device=self.device)
            min_dist = 0.05
            max_iters = 32
            remaining = torch.ones(b, dtype=torch.bool, device=self.device)

            for _ in range(max_iters):
                if not remaining.any():
                    break
                # 为所有 env 采样候选，然后只更新还未满足的那些
                cand_xy = torch.rand((b, 2), device=self.device) * 0.2 - 0.1  # [-0.1, 0.1]
                diff_xy = cand_xy - obj_pos[:, :2]
                dist_xy = torch.linalg.norm(diff_xy, dim=-1)
                ok = (dist_xy >= min_dist) & remaining
                goal_xy[ok] = cand_xy[ok]
                remaining[ok] = False

            # 若还有少数 env 没采到满足约束的点，最后再给一次候选（可能略小于 5cm，但概率极低）
            if remaining.any():
                cand_xy = torch.rand((b, 2), device=self.device) * 0.2 - 0.1
                goal_xy[remaining] = cand_xy[remaining]

            # 高度：在物体底面高度基础上抬升 [0.0, 0.03]
            base_z = self.object_zs[env_idx]  # (b,)
            lift_z = 0.05+torch.rand((b,), device=self.device) * 0.05
            goal_z = base_z + lift_z  # (b,)

            # 组装新的 goal 位置
            goal_xyz_subset = torch.zeros((b, 3), device=self.device)
            goal_xyz_subset[:, :2] = goal_xy
            goal_xyz_subset[:, 2] = goal_z



            roll = (torch.rand((b,), device=self.device) - 0.5) * np.pi        # [-pi/2, pi/2]
            pitch = (torch.rand((b,), device=self.device) - 0.5) * np.pi       # [-pi/2, pi/2]
            yaw = torch.rand((b,), device=self.device) * (2.0 * np.pi)         # [0, 2pi]

            # 欧拉角 -> 四元数，输出顺序为 [w, x, y, z]
            cr = torch.cos(roll * 0.5)
            sr = torch.sin(roll * 0.5)
            cp = torch.cos(pitch * 0.5)
            sp = torch.sin(pitch * 0.5)
            cy = torch.cos(yaw * 0.5)
            sy = torch.sin(yaw * 0.5)

            qw = cr * cp * cy + sr * sp * sy
            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy

            goal_q_subset = torch.stack([qw, qx, qy, qz], dim=-1)
            goal_q_subset = goal_q_subset / torch.clamp(
                torch.linalg.norm(goal_q_subset, dim=-1, keepdim=True),
                min=1e-8,
            )
            # 随机朝向：这里只绕 z 轴随机
#            goal_q_subset = random_quaternions(
#                b,
#                lock_x=True,
#                lock_y=True,
#                lock_z=False,
#                bounds=(0, np.pi * 2),
#            )
#            if goal_q_subset.device != self.device:
#                goal_q_subset = goal_q_subset.to(self.device)

            # 更新整批 pose，只替换 env_idx 对应的行，避免 shape mismatch
            new_pose_subset = torch.cat([goal_xyz_subset, goal_q_subset], dim=-1)  # (b, 7)
            goal_pose_all[env_idx] = new_pose_subset

            new_p = goal_pose_all[:, :3]
            new_q = goal_pose_all[:, 3:7]
            self.goal_jiaobang.set_pose(Pose.create_from_pq(new_p, new_q))

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        reconfiguration_freq=None,
        object_asset_path=None,
        object_scale=0.1,
        object_density=1000.0,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.object_asset_path = object_asset_path
        self.object_density = float(object_density)
        scale_arr = np.asarray(object_scale, dtype=float).reshape(-1)
        if scale_arr.size == 1:
            scale_arr = np.repeat(scale_arr, 3)
        if scale_arr.size != 3:
            raise ValueError(f"object_scale must be a float or length-3 sequence, got {object_scale}")
        self.object_scale_vec = scale_arr.astype(float)

        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**20, max_rigid_patch_count=2**19
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        # 保持与 Dex_PickSingleYCB / 原 jiaobang 相同的机器人加载位姿
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # 桌面与机器人初始化
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # 资产路径与尺度，可由 env kwargs 动态覆盖，支持“一次只抓一个激活物体”
        this_dir = os.path.dirname(os.path.abspath(__file__))
        glb_path = self.object_asset_path or os.path.join(this_dir, "jiaobang.glb")
        glb_path = os.path.expanduser(glb_path)
        if not os.path.isabs(glb_path):
            glb_path = os.path.join(this_dir, glb_path)
        if not os.path.exists(glb_path):
            raise FileNotFoundError(glb_path)
        scale_vec = np.asarray(self.object_scale_vec, dtype=float)

        # 为每个子场景各建一个 jiaobang，再 merge 成一个 Actor，和 Dex_PickSingleYCB 风格一致
        self._objs: List[Actor] = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            builder.set_scene_idxs([i])
            builder.add_multiple_convex_collisions_from_file(
                glb_path,
                decomposition="coacd",
                scale=scale_vec.tolist(),
                material=None,
                density=self.object_density,
            )
            builder.add_visual_from_file(glb_path, scale=scale_vec.tolist())
            self._objs.append(builder.build(name=f"jiaobang-{i}"))
            self.remove_from_state_dict_registry(self._objs[-1])

        self.obj = Actor.merge(self._objs, name="jiaobang")
        self.add_to_state_dict_registry(self.obj)

        # 目标位姿的“示意物”：仅视觉、无碰撞；先原纹理再叠一层半透明绿，做出混合感
        self._goal_objs: List[Actor] = []
        goal_green_tint = sapien.render.RenderMaterial(
            base_color=[0.2, 0.88, 0.2, 0.5],  # 半透明绿，叠在原纹理上
            roughness=0.4,
            specular=0.2,
        )
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            builder.set_scene_idxs([i])
            builder.add_visual_from_file(glb_path, scale=scale_vec.tolist())
            builder.add_visual_from_file(
                glb_path,
                scale=scale_vec.tolist(),
                material=goal_green_tint,
            )
            # 无任何碰撞体，build_kinematic 得到纯展示刚体
            self._goal_objs.append(
                builder.build_kinematic(name=f"goal_jiaobang-{i}")
            )
            self.remove_from_state_dict_registry(self._goal_objs[-1])
        self.goal_jiaobang = Actor.merge(self._goal_objs, name="goal_jiaobang")
        self.add_to_state_dict_registry(self.goal_jiaobang)
        # 不加入 _hidden_objects：merge 后的 Actor 在 show_visual() 里会查 has_collision_shapes，会报错。
        # 目标 jiaobang 保持常显即可（agent 相机和 human 渲染都能看到目标位姿）。

        # 参考 Dex_PickSingleYCB，定义一个 xarm 灵巧手的 rest 姿态（RM75 也可复用前 7 维）
        self.rest_qpos = torch.tensor([np.pi / 2, 0, 0, -np.pi / 2, 0, -np.pi / 2, np.pi / 3, 0, 0, 0, 0, 0, 0],
                                      device=self.device)

    def _after_reconfigure(self, options: dict):
        # 和 Dex_PickSingleYCB 一样，用碰撞包围盒计算每个 env 的物体底部高度
        self.object_zs = []
        obj_local_aabb_min_list: List[np.ndarray] = []
        obj_local_aabb_max_list: List[np.ndarray] = []
        for obj in self._objs:
            collision_mesh = obj.get_first_collision_mesh()
            self.object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
            bounds = collision_mesh.bounding_box.bounds
            obj_local_aabb_min_list.append(np.asarray(bounds[0], dtype=np.float32))
            obj_local_aabb_max_list.append(np.asarray(bounds[1], dtype=np.float32))

        self.object_zs = common.to_tensor(self.object_zs, device=self.device)
        self.obj_local_aabb_min = common.to_tensor(
            np.stack(obj_local_aabb_min_list), device=self.device
        )
        self.obj_local_aabb_max = common.to_tensor(
            np.stack(obj_local_aabb_max_list), device=self.device
        )
        # 抬升奖励用的参考高度：初始全为 -1，计算奖励时若为 -1 则用当前物体 z 赋为初始 z
        self.object_initial_height = torch.full(
            (self.num_envs,), -1.0, device=self.device, dtype=torch.float32
        )
        # reset 缓存：每回合开始记录一次 XY 最短边（世界系），用于抓到前对齐奖励
        self.obj_xy_shortest_edge_vector = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )

    def get_obj_xy_shortest_edge_vector(self) -> torch.Tensor:
        """计算物体在 XY 平面 AABB 的最短边向量（世界系）。Returns: (N,3) with z=0."""
        N = self.obj.pose.p.shape[0]
        R = quaternion_to_matrix(self.obj.pose.q)  # (N, 3, 3)
        t = self.obj.pose.p  # (N, 3)
        lo = self.obj_local_aabb_min  # (N, 3)
        hi = self.obj_local_aabb_max  # (N, 3)

        x = torch.stack(
            [lo[:, 0], lo[:, 0], lo[:, 0], lo[:, 0], hi[:, 0], hi[:, 0], hi[:, 0], hi[:, 0]],
            dim=1,
        )
        y = torch.stack(
            [lo[:, 1], lo[:, 1], hi[:, 1], hi[:, 1], lo[:, 1], lo[:, 1], hi[:, 1], hi[:, 1]],
            dim=1,
        )
        z = torch.stack(
            [lo[:, 2], hi[:, 2], lo[:, 2], hi[:, 2], lo[:, 2], hi[:, 2], lo[:, 2], hi[:, 2]],
            dim=1,
        )
        corners = torch.stack([x, y, z], dim=-1)  # (N, 8, 3)
        corners_world = torch.bmm(corners, R.transpose(-2, -1)) + t.unsqueeze(1)
        wmin = corners_world.min(dim=1)[0]
        wmax = corners_world.max(dim=1)[0]
        extent_x = wmax[:, 0] - wmin[:, 0]
        extent_y = wmax[:, 1] - wmin[:, 1]
        x_shorter = extent_x <= extent_y
        length = torch.where(x_shorter, extent_x, extent_y)
        vec = torch.zeros((N, 3), device=self.device, dtype=length.dtype)
        vec[:, 0] = torch.where(x_shorter, length, torch.zeros_like(length))
        vec[:, 1] = torch.where(x_shorter, torch.zeros_like(length), length)
        return vec

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # 随机放置 jiaobang
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, :2] = torch.rand((b, 2), device=self.device) * 0.2 - 0.1
            xyz[:, 2] = self.object_zs[env_idx]
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            if qs.device != xyz.device:
                qs = qs.to(xyz.device)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            self.object_initial_height[env_idx] = -1.0  # 下一步算奖励时用当前 z 赋为初始 z
            # reset 时缓存一次最短边向量（世界系），后续 reward 里在“未抓到前”鼓励夹爪对齐该方向
            with torch.no_grad():
                shortest_vec_all = self.get_obj_xy_shortest_edge_vector()
                self.obj_xy_shortest_edge_vector[env_idx] = shortest_vec_all[env_idx].detach()

            # 目标位姿：位置 + 朝向（随机采样，只绕 z 轴随机）
            self._sample_new_goal(env_idx)
            self.agent.robot.set_qpos(
                self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1])) * 0.01
            )
            self.agent.robot.set_pose(
                Pose.create_from_pq(p=[-0.3, 0, 0], q=euler2quat(0, 0, np.pi / 2), device=self.device)
            )


    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):
        # Run the normal step first (so reward/info for the success transition is correct),
        # then if success happened, immediately re-sample a new goal and keep the episode running.
        obs, reward, terminated, truncated, info = super().step(action)
        if isinstance(info, dict) and "success" in info:
            success = info["success"]
            if torch.is_tensor(success):
                success_envs = torch.nonzero(success, as_tuple=False).flatten()
                if len(success_envs) > 0:
                    self._sample_new_goal(success_envs)
                    terminated = terminated.clone()
                    terminated[success_envs] = False
        return obs, reward, terminated, truncated, info

    def evaluate(self):
        goal_p = torch.as_tensor(self.goal_jiaobang.pose.p, device=self.device, dtype=torch.float32)
        obj_p = torch.as_tensor(self.obj.pose.p, device=self.device, dtype=torch.float32)
        tcp_p = torch.as_tensor(self.agent.tcp_pos, device=self.device, dtype=torch.float32)
        # 位置成功：物体到目标位置距离 <= goal_thresh
        pos_dist = torch.linalg.norm(
            goal_p - obj_p, axis=1
        )
        is_obj_placed_pos = pos_dist <= self.goal_thresh

        # 朝向成功：物体四元数与目标四元数夹角 <= goal_angle_thresh (rad)
        orientation_error_rad = common.quat_diff_rad(
            self.obj.pose.q, self.goal_jiaobang.pose.q
        )
        is_obj_placed_rot = orientation_error_rad <= self.goal_angle_thresh

        # 只有位置和朝向都满足才算放置成功
        is_obj_placed = is_obj_placed_pos & is_obj_placed_rot

        is_robot_static = self.agent.is_static(0.2)

        tcp_to_obj_dist = torch.linalg.norm(
            obj_p - tcp_p,
            axis=-1,
        )
        is_grasped = self.agent.is_grasping(self.obj)

        l_contact_forces = torch.as_tensor(
            self.scene.get_pairwise_contact_forces(self.agent.finger1_link, self.table_scene.table),
            device=self.device,
            dtype=torch.float32,
        )
        r_contact_forces = torch.as_tensor(
            self.scene.get_pairwise_contact_forces(self.agent.finger2_link, self.table_scene.table),
            device=self.device,
            dtype=torch.float32,
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)
        touching_table = torch.logical_or(
            lforce >= 1e-2,
            rforce >= 1e-2,
        )

        # 物体与桌面的接触合力（世界系）
        obj_table_contact_forces = torch.as_tensor(
            self.scene.get_pairwise_contact_forces(self.obj, self.table_scene.table),
            device=self.device,
            dtype=torch.float32,
        )
        obj_table_force_norm = torch.linalg.norm(obj_table_contact_forces, axis=1)
        # 桌面对物体的法向支撑力（取 z 分量，通常为正；物体对桌面的向下力大小近似相同）
        obj_table_normal_force = torch.relu(obj_table_contact_forces[:, 2])
        # 物体重力（重量）：m * g
        g = 9.81
        obj_weight = self.obj.mass.to(self.device) * g  # (N,)
        # 超过重力的额外“压桌”法向力（例如砸/按）
        obj_table_excess_normal_force = torch.relu(obj_table_normal_force - obj_weight)

        #print("aaaa", pos_dist, orientation_error_rad, is_obj_placed_pos, is_obj_placed_rot,  is_grasped)

        return {
            "touching_table": touching_table,
            "is_grasped": is_grasped,
            "success": is_obj_placed_pos & is_obj_placed_rot & is_grasped,
            "is_obj_placed": is_obj_placed,
            "is_obj_placed_pos": is_obj_placed_pos,
            "is_obj_placed_rot": is_obj_placed_rot,
            "orientation_error_rad": orientation_error_rad,
            "is_robot_static": is_robot_static,
            "obj_table_contact_forces": obj_table_contact_forces,
            "obj_table_force_norm": obj_table_force_norm,
            "obj_weight": obj_weight,
            "obj_table_normal_force": obj_table_normal_force,
            "obj_table_excess_normal_force": obj_table_excess_normal_force,
        }

    def _get_obs_extra(self, info: Dict):
        # 完全复用 Dex_PickSingleYCB 的灵巧手额外观测设计，但物体换成 jiaobang

        obs = dict(
            goal_pos=self.goal_jiaobang.pose.p,
            goal_quat=self.goal_jiaobang.pose.q
        )

        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
            )

        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # 若某 env 的初始 z 仍为 -1，则用当前物体 z 赋为初始 z
        # 与 Dex_PickSingleYCB 相同的奖励结构，只是物体换成 jiaobang
        tcp_to_obj_dist = torch.linalg.norm(
            self.obj.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward + 3 * info["is_grasped"]

        # 抓到前：鼓励夹爪闭合方向与 reset 时记录的“物体 XY 最短边方向”平行
        shortest_vec = self.obj_xy_shortest_edge_vector
        short_xy = shortest_vec[:, :2]
        short_len = torch.linalg.norm(short_xy, dim=-1, keepdim=True).clamp(min=1e-6)
        short_unit = short_xy / short_len
        R_tcp = quaternion_to_matrix(self.agent.tcp_pose.q)
        tcp_x = R_tcp[..., :, 0]  # 近似为两指闭合方向（局部 x 轴）
        tcp_x_xy = tcp_x[:, :2]
        tcp_x_len = torch.linalg.norm(tcp_x_xy, dim=-1, keepdim=True).clamp(min=1e-6)
        tcp_x_unit_xy = tcp_x_xy / tcp_x_len
        cos_sim = torch.clamp(torch.abs((short_unit * tcp_x_unit_xy).sum(dim=-1)), 0.0, 1.0)
        angle = torch.acos(cos_sim)
        parallel_to_short = torch.exp(-2.0 * angle)
        reward += 2.0 * parallel_to_short * (~info["is_grasped"]).float()

        # 大惩罚：禁止机械臂“从上向下戳/按”物体来产生姿态变化投机抓取
        # 用 finger links 对物体的接触合力在世界 z 方向的向下分量近似检测。
        # （GPU sim 下通常拿不到 contact points，因此用合力阈值作为 proxy）
#        l_f_obj = self.scene.get_pairwise_contact_forces(self.agent.finger1_link, self.obj)  # (N, 3)
#        r_f_obj = self.scene.get_pairwise_contact_forces(self.agent.finger2_link, self.obj)  # (N, 3)
#        f_obj = l_f_obj + r_f_obj
#        down_press = torch.relu(-f_obj[:, 2])  # >0 表示向下按压力（世界 z 轴向下为负）
#        press_th = 3.0  # N，可按仿真尺度调
#        poking_down = down_press > press_th
#        # 主要在未抓稳前惩罚，避免正常抓取后搬运阶段误伤
#        reward -= poking_down.float() * (~info["is_grasped"]).float() * 1.0

        # 惩罚物体对桌面作用力过大（例如“砸/按”桌面）
        obj_table_excess = info.get("obj_table_excess_normal_force", None)

        # 只惩罚超过自身重量的“额外”法向力
        excess_th = 20.0  # N，可按仿真尺度调
        reward -= (torch.relu(obj_table_excess - excess_th) > 0).float()


       # print("???-------------",obj_table_excess)

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_jiaobang.pose.p - self.obj.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * info["is_grasped"]
        reward -= 2 * info["touching_table"].float()

        orientation_error_rad = common.quat_diff_rad(
            self.obj.pose.q, self.goal_jiaobang.pose.q
        )
        place_rot_reward = 1 - torch.tanh(5 * orientation_error_rad)
        reward += 2 * place_rot_reward * info["is_obj_placed"] * info["is_grasped"]  # * lift_flag.float()




        reward[info["success"]] = 8

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8
