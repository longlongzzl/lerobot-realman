from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import sapien
import torch
from transforms3d.quaternions import axangle2quat

DEFAULT_EXTRA_MANISKILL_PACKAGE_ROOT = os.environ.get(
    "JIMU_EXTRA_MANISKILL_PACKAGE_ROOT",
    "/home/zhangzhao/anaconda3/envs/realman/lib/python3.11/site-packages/mani_skill",
)


def _extend_mani_skill_realman_robot_path(extra_maniskill_package_root: str | os.PathLike[str] | None = None) -> None:
    package_root = Path(extra_maniskill_package_root or DEFAULT_EXTRA_MANISKILL_PACKAGE_ROOT).expanduser()
    if package_root.name != "mani_skill" and (package_root / "mani_skill").exists():
        package_root = package_root / "mani_skill"
    robots_path = package_root / "agents" / "robots"
    if not robots_path.exists():
        return

    import mani_skill.agents.robots as robots_pkg

    robots_path_str = str(robots_path)
    if robots_path_str not in robots_pkg.__path__:
        robots_pkg.__path__.append(robots_path_str)


_extend_mani_skill_realman_robot_path()

from mani_skill.agents.robots import Panda
from mani_skill.agents.robots.realman import RM75Robot
from mani_skill.agents.controllers.pd_joint_pos import (
    PDJointPosControllerConfig,
    PDJointPosMimicControllerConfig,
)
from mani_skill.agents.controllers.passive_controller import PassiveControllerConfig
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from magnetic_snap import MagneticSnapManager


PLATE_SIZE = 0.074
PLATE_THICKNESS = 0.007
TRIANGLE_HEIGHT = 0.134
TRIANGLE_WIDTH = 0.075
TRIANGLE_THICKNESS = 0.007


def _install_rm75_absolute_joint_pos_controller() -> None:
    marker = "_jimu_abs_joint_pos_installed"
    if getattr(RM75Robot, marker, False):
        return
    original_property = RM75Robot._controller_configs

    def patched_controller_configs(self):
        configs = deepcopy(original_property.fget(self))
        if "pd_joint_pos_abs" not in configs:
            active_roots = ["gripper_Right_1_Joint", "gripper_Left_1_Joint"]
            passive_links = [
                "gripper_Right_Support_Joint",
                "gripper_Right_2_Joint",
                "gripper_Left_Support_Joint",
                "gripper_Left_2_Joint",
            ]
            mimic_dict = {
                "gripper_Left_1_Joint": {
                    "joint": "gripper_Right_1_Joint",
                    "multiplier": 1.0,
                    "offset": 0.0,
                },
            }
            configs["pd_joint_pos_abs"] = dict(
                arm=PDJointPosControllerConfig(
                    self.arm_joint_names,
                    lower=None,
                    upper=None,
                    stiffness=self.arm_stiffness,
                    damping=self.arm_damping,
                    force_limit=self.arm_force_limit,
                    friction=self.arm_friction,
                    use_delta=False,
                    normalize_action=False,
                ),
                gripper=PDJointPosMimicControllerConfig(
                    joint_names=active_roots,
                    lower=0.0,
                    upper=0.91,
                    stiffness=self.gripper_stiffness,
                    damping=self.gripper_damping,
                    force_limit=self.gripper_force_limit,
                    friction=self.gripper_friction,
                    use_delta=False,
                    normalize_action=False,
                    mimic=mimic_dict,
                ),
                gripper_passive=PassiveControllerConfig(
                    joint_names=passive_links,
                    damping=0.0,
                    friction=0.0,
                ),
            )
        return configs

    RM75Robot._controller_configs = property(patched_controller_configs)
    setattr(RM75Robot, marker, True)


_install_rm75_absolute_joint_pos_controller()


def _write_isosceles_triangular_prism_obj(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    half_width = TRIANGLE_WIDTH / 2.0
    half_thick = TRIANGLE_THICKNESS / 2.0
    verts = [
        (-half_width, -half_thick, 0.0),
        (half_width, -half_thick, 0.0),
        (0.0, -half_thick, TRIANGLE_HEIGHT),
        (-half_width, half_thick, 0.0),
        (half_width, half_thick, 0.0),
        (0.0, half_thick, TRIANGLE_HEIGHT),
    ]
    faces = [
        (1, 2, 3),
        (4, 6, 5),
        (1, 4, 5),
        (1, 5, 2),
        (2, 5, 6),
        (2, 6, 3),
        (3, 6, 4),
        (3, 4, 1),
    ]
    lines = ["# 75mm x 134mm x 7mm isosceles triangular prism"]
    lines.extend(f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in verts)
    lines.extend(f"f {a} {b} {c}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@register_env("JimuPickCube-v1", max_episode_steps=200, override=True)
class JimuPickCubeEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "RM75"]
    agent: Panda | RM75Robot

    def __init__(
        self,
        *args: Any,
        robot_uids: str = "panda",
        robot_init_qpos_noise: float = 0.0,
        assembly_mode: str = "house",
        magnet_mode: str = "edge_pair_drive",
        drive_stiffness: float = 60.0,
        drive_damping: float = 10.0,
        drive_force_limit: float = 1.0,
        drive_angular_stiffness: float = 0.0,
        drive_angular_damping: float = 0.0,
        drive_angular_force_limit: float = 0.0,
        position_stiffness: float = 900.0,
        position_damping: float = 55.0,
        rotation_stiffness: float = 0.08,
        rotation_damping: float = 0.015,
        attract_stiffness: float = 1.6,
        attract_force_limit: float = 0.65,
        attract_torque_stiffness: float = 0.055,
        attract_torque_limit: float = 0.035,
        attract_normal_torque_stiffness: float = 0.16,
        attract_normal_torque_limit: float = 0.12,
        magnet_dynamic_update_period: int = 5,
        panel_linear_damping: float = 0.15,
        panel_angular_damping: float = 0.45,
        panel_density: float = 1000.0,
        num_plates: int = 5,
        num_triangles: int = 4,
        **kwargs: Any,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.assembly_mode = assembly_mode
        self.magnet_mode = magnet_mode
        self.drive_stiffness = drive_stiffness
        self.drive_damping = drive_damping
        self.drive_force_limit = drive_force_limit
        self.drive_angular_stiffness = drive_angular_stiffness
        self.drive_angular_damping = drive_angular_damping
        self.drive_angular_force_limit = drive_angular_force_limit
        self.position_stiffness = position_stiffness
        self.position_damping = position_damping
        self.rotation_stiffness = rotation_stiffness
        self.rotation_damping = rotation_damping
        self.attract_stiffness = attract_stiffness
        self.attract_force_limit = attract_force_limit
        self.attract_torque_stiffness = attract_torque_stiffness
        self.attract_torque_limit = attract_torque_limit
        self.attract_normal_torque_stiffness = attract_normal_torque_stiffness
        self.attract_normal_torque_limit = attract_normal_torque_limit
        self.magnet_dynamic_update_period = magnet_dynamic_update_period
        self.panel_linear_damping = panel_linear_damping
        self.panel_angular_damping = panel_angular_damping
        self.panel_density = panel_density
        self.num_plates = num_plates
        self.num_triangles = num_triangles
        self.triangle_mesh_path = Path(__file__).resolve().parent / "assets" / "isosceles_triangular_prism.obj"
        _write_isosceles_triangular_prism_obj(self.triangle_mesh_path)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=100,
            control_freq=20,
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21,
                max_rigid_patch_count=2**19,
            ),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.45, -0.55, 0.55], target=[0.0, 0.0, 0.12])
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.55, -0.65, 0.55], target=[0.0, 0.0, 0.12])
        return CameraConfig("render_camera", pose, 768, 768, 1.0, 0.01, 100)

    def _load_agent(self, options: dict):
        base_yaw_deg = float(os.environ.get("JIMU_RM75_BASE_YAW_DEG", "90.0"))
        base_quat = np.asarray(axangle2quat([0.0, 0.0, 1.0], np.deg2rad(base_yaw_deg)), dtype=np.float32)
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0.0], q=base_quat.tolist()))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()

        plate_material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        triangle_material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        plate_colors = [
            [0.55, 0.03, 0.03, 1.0],
            [0.05, 0.28, 0.60, 1.0],
            [0.08, 0.45, 0.18, 1.0],
            [0.62, 0.38, 0.02, 1.0],
            [0.35, 0.12, 0.55, 1.0],
        ]
        triangle_colors = [
            [0.78, 0.18, 0.18, 1.0],
            [0.12, 0.55, 0.78, 1.0],
            [0.20, 0.62, 0.28, 1.0],
            [0.72, 0.42, 0.10, 1.0],
        ]

        plate_half_size = [PLATE_SIZE / 2.0, PLATE_SIZE / 2.0, PLATE_THICKNESS / 2.0]
        plate_positions = [
            [-0.10, -0.08, PLATE_THICKNESS / 2.0],
            [0.00, -0.08, PLATE_THICKNESS / 2.0],
            [0.10, -0.08, PLATE_THICKNESS / 2.0],
            [-0.05, 0.02, PLATE_THICKNESS / 2.0],
            [0.05, 0.02, PLATE_THICKNESS / 2.0],
        ]
        triangle_positions = [
            [-0.15, 0.12, 0.0],
            [-0.05, 0.12, 0.0],
            [0.05, 0.12, 0.0],
            [0.15, 0.12, 0.0],
        ]

        self.plates = []
        self.triangles = []
        self.magnetic_snap = MagneticSnapManager(
            plate_size=PLATE_SIZE,
            plate_thickness=PLATE_THICKNESS,
            triangle_height=TRIANGLE_HEIGHT,
            triangle_width=TRIANGLE_WIDTH,
            triangle_thickness=TRIANGLE_THICKNESS,
            magnet_mode=self.magnet_mode,
            drive_stiffness=self.drive_stiffness,
            drive_damping=self.drive_damping,
            drive_force_limit=self.drive_force_limit,
            drive_angular_stiffness=self.drive_angular_stiffness,
            drive_angular_damping=self.drive_angular_damping,
            drive_angular_force_limit=self.drive_angular_force_limit,
            position_stiffness=self.position_stiffness,
            position_damping=self.position_damping,
            rotation_stiffness=self.rotation_stiffness,
            rotation_damping=self.rotation_damping,
            attract_stiffness=self.attract_stiffness,
            attract_force_limit=self.attract_force_limit,
            attract_torque_stiffness=self.attract_torque_stiffness,
            attract_torque_limit=self.attract_torque_limit,
            attract_normal_torque_stiffness=self.attract_normal_torque_stiffness,
            attract_normal_torque_limit=self.attract_normal_torque_limit,
            dynamic_update_period=self.magnet_dynamic_update_period,
        )
        for i in range(self.num_envs):
            for j in range(self.num_plates):
                pos = plate_positions[j % len(plate_positions)]
                builder = self.scene.create_actor_builder()
                builder.set_scene_idxs([i])
                builder.initial_pose = sapien.Pose(p=pos)
                builder.add_box_collision(
                    half_size=plate_half_size,
                    material=plate_material,
                    density=self.panel_density,
                )
                plate_visual = sapien.render.RenderMaterial(base_color=plate_colors[j % len(plate_colors)])
                builder.add_box_visual(half_size=plate_half_size, material=plate_visual)
                plate = builder.build(name=f"jimu_plate_{i}_{j}")
                plate.set_linear_damping(self.panel_linear_damping)
                plate.set_angular_damping(self.panel_angular_damping)
                self.plates.append(plate)
                self.remove_from_state_dict_registry(plate)

            for j in range(self.num_triangles):
                pos = triangle_positions[j % len(triangle_positions)]
                builder = self.scene.create_actor_builder()
                builder.set_scene_idxs([i])
                builder.initial_pose = sapien.Pose(p=pos)
                builder.add_multiple_convex_collisions_from_file(
                    str(self.triangle_mesh_path),
                    decomposition="coacd",
                    material=triangle_material,
                    density=self.panel_density,
                )
                triangle_visual = sapien.render.RenderMaterial(base_color=triangle_colors[j % len(triangle_colors)])
                builder.add_visual_from_file(str(self.triangle_mesh_path), material=triangle_visual)
                triangle = builder.build(name=f"jimu_triangle_{i}_{j}")
                triangle.set_linear_damping(self.panel_linear_damping)
                triangle.set_angular_damping(self.panel_angular_damping)
                self.triangles.append(triangle)
                self.remove_from_state_dict_registry(triangle)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)
        if self.agent is not None:
            qpos = self.agent.robot.get_qpos()
            self.agent.robot.set_qpos(torch.zeros_like(qpos))
            if hasattr(self.agent.robot, "get_qvel") and hasattr(self.agent.robot, "set_qvel"):
                self.agent.robot.set_qvel(torch.zeros_like(self.agent.robot.get_qvel()))
        if self.assembly_mode == "house":
            self.magnetic_snap.arrange_house(self.plates, self.triangles)
            self.magnetic_snap.ensure_magnetic_drives(self.scene)
        elif self.assembly_mode == "open_cube":
            self.magnetic_snap.arrange_open_cube(self.plates)
            self.magnetic_snap.register_free_triangles(self.triangles)
            self.magnetic_snap.ensure_magnetic_drives(self.scene)
        elif self.assembly_mode == "triangle_vertical":
            self.magnetic_snap.arrange_triangle_vertical(self.plates, self.triangles)
            self.magnetic_snap.ensure_magnetic_drives(self.scene)
            for plate in self.plates[1:]:
                plate.set_pose(sapien.Pose(p=[0.0, 0.0, -2.0]))
            for triangle in self.triangles[1:]:
                triangle.set_pose(sapien.Pose(p=[0.0, 0.0, -2.0]))
        else:
            self.magnetic_snap.clear()

    def _before_simulation_step(self):
        self.magnetic_snap.apply()

    def evaluate(self):
        is_static = self.agent.is_static(0.2)
        return {"success": is_static}

    def _get_obs_extra(self, info: dict):
        obs = {"tcp_pose": self.agent.tcp.pose.raw_pose}
        if "state" in self.obs_mode:
            obs.update(
                qpos=self.agent.robot.get_qpos(),
            )
        return obs

    def get_magnetic_snap_report(self):
        return self.magnetic_snap.report()

    def deactivate_magnetic_piece(self, role: str, hide: bool = False):
        self.magnetic_snap.deactivate_piece(role)
        if hide:
            for locked in self.magnetic_snap.locked_panel_poses:
                if locked.role == role:
                    locked.actor.set_pose(sapien.Pose(p=[0.0, 0.0, -2.0]))
                    locked.actor.set_linear_velocity(np.zeros(3, dtype=np.float32))
                    locked.actor.set_angular_velocity(np.zeros(3, dtype=np.float32))
                    break

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.as_tensor(info["success"], device=self.device, dtype=torch.float32)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info)


def make_env(**kwargs: Any) -> gym.Env:
    return gym.make("JimuPickCube-v1", **kwargs)
