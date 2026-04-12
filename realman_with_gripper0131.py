from copy import deepcopy
import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosControllerConfig, PDJointPosMimicControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils import common, sapien_utils

@register_agent()
class RM75Robot(BaseAgent):
    uid = "RM75"
    urdf_path = f"{ASSET_DIR}/robots/RM75_gripper/RM75-B/urdf/RM75-B.urdf"
    disable_self_collisions = False

    # 材质配置保持不变...
    urdf_config = dict(
        _materials=dict(
            gripper_mat=dict(static_friction=4.0, dynamic_friction=4.0, restitution=0.0)
        ),
        link=dict(
            left_pad=dict(material="gripper_mat", patch_radius=0.1, min_patch_radius=0.1),
            right_pad=dict(material="gripper_mat", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([  np.pi/2,
                    0,
                    0,
                    - np.pi / 2,
                    0,
                    - np.pi / 2,
                    np.pi / 3, 0, 0, 0, 0, 0, 0]),
         #    qpos=np.array([1.6422e+00, -7.4988e-01, -4.1515e-01, -1.8833e+00, 9.8123e-01,
         # -2.1272e+00, -8.1136e-01, 1.6129e-36, 2.0998e-36, 2.9778e-37,
         # 9.4218e-36, -5.6873e-37, -5.8340e-37]),
            pose=sapien.Pose([0, 0, 0]),
        ),
    )

    arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]

    # ----------------------------------------------------------------------
    # [关键点 1] 关节名称定义
    # ----------------------------------------------------------------------
    # 原来是 Left，现在改成 Right
    gripper_drive_joint_name = "gripper_Right_1_Joint"

    # 把 Left 扔进从动列表，把 Right 拿出来
    gripper_mimic_joint_names = [
        "gripper_Right_Support_Joint",
        "gripper_Right_2_Joint",
        "gripper_Left_1_Joint",       # <--- 原废太子，现在是从动
        "gripper_Left_Support_Joint",
        "gripper_Left_2_Joint"
    ]

    # 刚度参数
    # 对齐 Panda：更软、更稳定的默认 PD 增益
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    # 对齐 Panda：夹爪 PD 增益与力限
    gripper_stiffness = 1e4
    gripper_damping = 1e3
    gripper_force_limit = 2000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.3, upper=0.3,
            #lower=-0.1, upper=0.1,
            stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True, normalize_action=True
        )

        # ------------------------------------------------------------------
        # 2. 夹爪控制器 (改为 Absolute !!)
        # ------------------------------------------------------------------
        # 这里的关键是：
        # - use_delta = False (或者不写，默认就是 False)
        # - lower/upper = 物理全范围 (0 到 0.91)
        # - normalize_action = True (把 -1~1 的动作映射到 0~0.91)

        mimic_dict = {
            "gripper_Right_Support_Joint": { "joint": self.gripper_drive_joint_name, "multiplier": 1.0, "offset": 0.0 },
            "gripper_Right_2_Joint":       { "joint": self.gripper_drive_joint_name, "multiplier": 1.0, "offset": 0.0 },
            "gripper_Left_1_Joint":        { "joint": self.gripper_drive_joint_name, "multiplier": 1.0, "offset": 0.0 },
            "gripper_Left_Support_Joint":  { "joint": self.gripper_drive_joint_name, "multiplier": 1.0, "offset": 0.0 },
            "gripper_Left_2_Joint":        { "joint": self.gripper_drive_joint_name, "multiplier": 1.0, "offset": 0.0 },
        }

        all_gripper_joints = [self.gripper_drive_joint_name] + self.gripper_mimic_joint_names

        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            joint_names=all_gripper_joints,
            # [关键修改]：这里不再是微小的 delta，而是完整的物理范围
            # 类似 Panda 的 -0.01 到 0.04
            lower=0.0,    # 对应 Action = -1 (闭合)
            upper=0.91,   # 对应 Action = +1 (张开)
            # [关键修改]：关闭 Delta，使用绝对位置
            use_delta=False,
            # [关键修改]：开启归一化，这样 -1 对应 lower, 1 对应 upper
            normalize_action=True,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic=mimic_dict
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True


        controller_configs = dict(
            pd_joint_delta_pos = dict(
                arm=arm_pd_joint_delta_pos, # 手臂用 Delta
                gripper=gripper_pd_joint_pos # 夹爪用 Absolute
            ),
            pd_joint_target_delta_pos = dict(
                arm=arm_pd_joint_target_delta_pos,
                gripper=gripper_pd_joint_pos,
            )
        )
        return deepcopy(controller_configs)




    def _after_loading_articulation(self):
        super()._after_loading_articulation()



        gripper_links = [
            "gripper_base_link",
            "gripper_Left_1_Link",
            "gripper_Left_Support_Link",
            "gripper_Left_2_Link",
            "gripper_Right_1_Link",
            "gripper_Right_Support_Link",
            "gripper_Right_2_Link",
            "link_6",  # not gripper link but is adjacent to the gripper part
            "link_7",  # not gripper link but is adjacent to the gripper part
            "left_pad",
            "right_pad"
        ]
        # print(self.robot.links_map.keys())
        for link_name in gripper_links:
            link = self.robot.links_map[link_name]
            link.set_collision_group_bit(group=2, bit_idx=31, bit=1)


        # ------------------------------------------------------------------
        # [关键点 4] 调试与物理属性检查
        # ------------------------------------------------------------------
        # 打印出来，核对你的 URDF 里的关节名字到底是不是叫这些
        active_joints = [j.get_name() for j in self.robot.get_active_joints()]
        print(f"\n[RM75 Check] Active Joints found in URDF: {active_joints}")

        # 检查 Drive Joint 是否存在
        if self.gripper_drive_joint_name not in active_joints:
            print(f"⚠️ 警告: 主动关节 '{self.gripper_drive_joint_name}' 不在 URDF 的活动关节列表中！请检查 URDF 命名！")

        # 必须给从动关节也加上刚度，否则它们在物理世界里是软的，Controller 想驱动也驱动不动
        for joint in self.robot.get_active_joints():
            if joint.get_name() in self.gripper_mimic_joint_names:
                joint.set_drive_properties(
                    stiffness=self.gripper_stiffness,
                    damping=self.gripper_damping,
                    force_limit=self.gripper_force_limit
                )

        # TCP 缓存部分
        # 使用gripper_Left_Support_Link和gripper_Right_Support_Link来判断水平状态和抓取状态
        self.finger1_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "gripper_Left_Support_Link")
        self.finger2_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "gripper_Right_Support_Link")
        # 如果Support Link不存在，使用备选方案

        self.finger1_pad = sapien_utils.get_obj_by_name(self.robot.get_links(), "left_pad")


        self.finger2_pad = sapien_utils.get_obj_by_name(self.robot.get_links(), "right_pad")


    def is_grasping(self, object: Actor, min_force=0.5, max_angle=110):
        """
        检查机器人是否抓取了物体

        @param {Actor} object - 要检查的物体
        @param {float} min_force - 最小接触力（牛顿），默认 0.5
        @param {float} max_angle - 最大接触角度（度），默认 110（参考 so101）
        @returns {torch.Tensor} 是否成功抓取
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper (参考 so101 的实现)
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)

        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)


    @property
    def tcp_pos(self):
        return (self.finger1_pad.pose.p + self.finger2_pad.pose.p) / 2

    @property
    def tcp_pose(self):
        return Pose.create_from_pq(self.tcp_pos, self.finger1_link.pose.q)

    @property
    def _sensor_configs(self):
        return []
