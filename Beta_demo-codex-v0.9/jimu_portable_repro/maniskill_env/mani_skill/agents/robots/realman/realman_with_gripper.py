from copy import deepcopy
from pathlib import Path
import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers.pd_joint_pos import (
    PDJointPosControllerConfig,
    PDJointPosMimicControllerConfig,
)
from mani_skill.agents.controllers.passive_controller import PassiveControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.actor import Actor

_PORTABLE_ASSET_DIR = Path(__file__).resolve().parents[3] / "assets"
_PORTABLE_RM75_URDF = _PORTABLE_ASSET_DIR / "robots" / "RM75_gripper" / "RM75-B" / "urdf" / "RM75-B.urdf"


@register_agent(override=True)
class RM75Robot(BaseAgent):
    uid = "RM75"
    urdf_path = str(_PORTABLE_RM75_URDF) if _PORTABLE_RM75_URDF.exists() else f"{ASSET_DIR}/robots/RM75_gripper/RM75-B/urdf/RM75-B.urdf"

    # [Recommendation] Set this to True.
    # Complex parallel grippers often have tiny internal overlaps that cause explosions.
    disable_self_collisions = False

    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            gripper_Left_Support_Link=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            gripper_Right_Support_Link=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([np.pi, 0, 0, -np.pi / 2, 0, -np.pi / 2, np.pi / 3*2, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
    )

    arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
    gripper_drive_joint_name = "gripper_Right_1_Joint"
    passive_gripper_joint_names = [
        "gripper_Right_Support_Joint", "gripper_Right_2_Joint",
        "gripper_Left_Support_Joint", "gripper_Left_2_Joint",
    ]

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_friction = 0.1
    arm_force_limit = 20 #[60, 60, 30, 30, 10, 10, 10]

    gripper_stiffness = 1e3
    gripper_damping = 100
    gripper_force_limit = 5
    gripper_friction = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        active_roots = ["gripper_Right_1_Joint", "gripper_Left_1_Joint"]
        passive_links = [
            "gripper_Right_Support_Joint", "gripper_Right_2_Joint",
            "gripper_Left_Support_Joint", "gripper_Left_2_Joint",
        ]



        mimic_dict = {
            "gripper_Left_1_Joint": {
                "joint": "gripper_Right_1_Joint",
                "multiplier": 1.0, "offset": 0.0,
            },
        }

        gripper_passive_config = PassiveControllerConfig(
            joint_names=passive_links,
            damping=0.0, friction=0.0,
        )

        gripper_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            joint_names=active_roots,
            lower=-0.05, upper=0.05,
            use_delta=True, normalize_action=True,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            mimic=mimic_dict,
        )

        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            joint_names=active_roots,
            lower=0.0, upper=0.91,
            use_delta=False, normalize_action=True,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            mimic=mimic_dict,
        )

        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            use_delta=False,
            normalize_action=False,
        )


        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.05, upper=0.05,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            use_delta=True,
            normalize_action=True,
        )

        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True
        gripper_pd_joint_target_delta_pos = deepcopy(gripper_pd_joint_delta_pos)
       # gripper_pd_joint_target_delta_pos.use_target = True


        return dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper=gripper_pd_joint_delta_pos,
                gripper_passive=gripper_passive_config,
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper=gripper_pd_joint_pos,
                gripper_passive=gripper_passive_config,
            ),
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos,
                gripper=gripper_pd_joint_target_delta_pos,
                gripper_passive=gripper_passive_config,
            ),
        )

    def _after_loading_articulation(self):
        super()._after_loading_articulation()

#        p_on_link2 = [0.040367, 0.037539, 0.007696]
#        p_on_support = [-0.014463, 0.016458, 0.00053]
        p_on_link2 = [0.0404, 0.0375, 0.0000]
        p_on_support = [-0.0141, 0.0170, 0.0000]
        def get_link(name):
            return sapien_utils.get_obj_by_name(self.robot.get_links(), name)

        r_link2 = get_link("gripper_Right_2_Link")
        r_support = get_link("gripper_Right_Support_Link")
        if r_link2 and r_support:
            r_drive = self.scene.create_drive(
                r_link2, sapien.Pose(p_on_link2),
                r_support, sapien.Pose(p_on_support)
            )
            r_drive.set_limit_x(0, 0)
            r_drive.set_limit_y(0, 0)
            r_drive.set_limit_z(0, 0)

        l_link2 = get_link("gripper_Left_2_Link")
        l_support = get_link("gripper_Left_Support_Link")
        if l_link2 and l_support:
            l_drive = self.scene.create_drive(
                l_link2, sapien.Pose(p_on_link2),
                l_support, sapien.Pose(p_on_support)
            )
            l_drive.set_limit_x(0, 0)
            l_drive.set_limit_y(0, 0)
            l_drive.set_limit_z(0, 0)

        # Collision grouping logic (Optional if disable_self_collisions=True)
        # But keeping it doesn't hurt.
        gripper_links = [
            "gripper_base_link",
            "gripper_Left_1_Link", "gripper_Left_Support_Link", "gripper_Left_2_Link",
            "gripper_Right_1_Link", "gripper_Right_Support_Link", "gripper_Right_2_Link",
            "link_6", "link_7", "left_pad", "right_pad"
        ]

        for link_name in gripper_links:
            if link_name in self.robot.links_map:
                link = self.robot.links_map[link_name]
                link.set_collision_group_bit(group=2, bit_idx=31, bit=1)

        self.finger1_link = get_link("gripper_Left_Support_Link")
        self.finger2_link = get_link("gripper_Right_Support_Link")
        self.finger1_pad = get_link("left_pad")
        self.finger2_pad = get_link("right_pad")

        self.tcp = get_link("gripper_tcp")

    # [FIX] Added is_static method to resolve NotImplementedError
    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()
        return torch.max(torch.abs(qvel), dim=1)[0] <= threshold

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=95):
        l_contact_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
        r_contact_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)

        lflag = torch.logical_and(lforce >= min_force, torch.rad2deg(langle) <= max_angle)
        rflag = torch.logical_and(rforce >= min_force, torch.rad2deg(rangle) <= max_angle)

        #print("111",lforce, rforce)

        return torch.logical_and(lflag, rflag)

    @property
    def tcp_pos(self):
        return self.tcp.pose.p #(self.finger1_pad.pose.p + self.finger2_pad.pose.p) / 2

    @property
    def tcp_pose(self):
        return Pose.create_from_pq(self.tcp_pos, self.tcp.pose.q)

    @property
    def _sensor_configs(self):
        return []
