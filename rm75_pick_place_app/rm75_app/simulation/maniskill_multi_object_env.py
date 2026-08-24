"""Configurable RM75 ManiSkill environment for a resolved multi-object scene."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sapien
import torch
from transforms3d.quaternions import mat2quat

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from rm75_app.core.frames import DEFAULT_ROBOT_BASE_WORLD_XYZ_M


@register_env("RM75-MultiObjectTask-v1", max_episode_steps=10000)
class RM75MultiObjectTaskEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["RM75"]

    def __init__(self, *args, scene_spec_file: str, robot_uids="RM75", num_envs=1, **kwargs):
        if int(num_envs) != 1:
            raise ValueError("multi-object validation currently supports num_envs=1")
        self.scene_spec_path = Path(scene_spec_file).expanduser().resolve()
        self.task_spec = json.loads(self.scene_spec_path.read_text(encoding="utf-8"))
        self.task_actors: dict[str, Actor] = {}
        self._actor_initial_poses: dict[str, np.ndarray] = {}
        super().__init__(*args, robot_uids=robot_uids, num_envs=1, reconfiguration_freq=1, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.5, 0.55, 0.65], target=[0.0, 0.0, 0.12])
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        grasp_previews = self.task_spec.get("grasp_previews") or []
        if grasp_previews:
            preview = grasp_previews[0]
            pre = np.asarray(preview["pregrasp_pose"], dtype=float)[:3, 3]
            grasp = np.asarray(preview["grasp_pose"], dtype=float)[:3, 3]
            points = [pre, grasp]
            for proxy in self.task_spec.get("collision_proxies") or []:
                points.append(np.asarray(proxy["pose"], dtype=float)[:3, 3])
            for model in self.task_spec.get("robot_collision_models") or []:
                T_world_model = np.asarray(
                    model.get("pose") or np.eye(4), dtype=float
                ).reshape(4, 4)
                for item in model.get("spheres") or []:
                    center_local = np.asarray(item["center"], dtype=float).reshape(3)
                    points.append(
                        T_world_model[:3, :3] @ center_local + T_world_model[:3, 3]
                    )
            cloud = np.asarray(points, dtype=float).reshape(-1, 3)
            lower = np.min(cloud, axis=0)
            upper = np.max(cloud, axis=0)
            focus = 0.5 * (lower + upper)
            focus[2] = max(float(focus[2]), 0.18)
            span = max(float(np.max(upper - lower)), 0.45)
            distance = max(0.62, 1.25 * span)
            return [
                CameraConfig(
                    "render_camera",
                    sapien_utils.look_at(
                        focus + [0.72 * distance, -0.72 * distance, 0.48 * distance],
                        focus,
                    ),
                    512, 512, 0.92, 0.01, 100,
                ),
                CameraConfig(
                    "top_camera",
                    sapien_utils.look_at(focus + [0.0, 0.0, 1.35 * distance], focus),
                    512, 512, 0.92, 0.01, 100,
                ),
                CameraConfig(
                    "left_camera",
                    sapien_utils.look_at(
                        focus + [0.90 * distance, 0.18 * distance, 0.40 * distance],
                        focus,
                    ),
                    512, 512, 0.92, 0.01, 100,
                ),
                CameraConfig(
                    "right_camera",
                    sapien_utils.look_at(
                        focus + [-0.55 * distance, 0.82 * distance, 0.45 * distance],
                        focus,
                    ),
                    512, 512, 0.92, 0.01, 100,
                ),
            ]
        configs = [
            CameraConfig(
                "render_camera",
                sapien_utils.look_at([0.65, 0.75, 0.65], [0.0, 0.0, 0.15]),
                512,
                512,
                1.0,
                0.01,
                100,
            )
        ]
        if bool(self.task_spec.get("preview_views", False)):
            configs.extend(
                [
                    CameraConfig("top_camera", sapien_utils.look_at([0.0, 0.0, 1.25], [0.0, 0.0, 0.05]), 512, 512, 1.05, 0.01, 100),
                    CameraConfig("left_camera", sapien_utils.look_at([0.55, -0.75, 0.55], [0.0, 0.0, 0.12]), 512, 512, 1.0, 0.01, 100),
                    CameraConfig("right_camera", sapien_utils.look_at([-0.15, 0.85, 0.60], [-0.05, 0.0, 0.12]), 512, 512, 1.0, 0.01, 100),
                ]
            )
        return configs

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=DEFAULT_ROBOT_BASE_WORLD_XYZ_M))

    def _load_scene(self, options: dict):
        del options
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=0.0)
        self.table_scene.build()
        for item in self.task_spec.get("objects") or []:
            object_id = str(item["object_id"])
            asset_file = str(Path(item["asset_file"]).expanduser().resolve())
            scale = np.asarray(item.get("scale") or [1.0, 1.0, 1.0], dtype=float).reshape(3)
            transform = np.asarray(item["pose"], dtype=float).reshape(4, 4)
            builder = self.scene.create_actor_builder()
            builder.set_scene_idxs([0])
            builder.initial_pose = sapien.Pose(p=transform[:3, 3], q=mat2quat(transform[:3, :3]))
            builder.add_multiple_convex_collisions_from_file(
                asset_file,
                decomposition="coacd",
                scale=scale.tolist(),
                density=float(item.get("density", 1000.0)),
            )
            builder.add_visual_from_file(asset_file, scale=scale.tolist())
            actor = (
                builder.build_kinematic(name=object_id)
                if bool(item.get("fixed", False))
                else builder.build(name=object_id)
            )
            self.task_actors[object_id] = actor
            self._actor_initial_poses[object_id] = transform
        self._load_target_visuals()
        self._load_grasp_preview_visuals()
        self._load_collision_proxy_visuals()
        self._load_robot_collision_sphere_visuals()

    def _load_target_visuals(self) -> None:
        """Render non-colliding task goals as translucent meshes plus RGB axes."""
        palette = (
            [0.15, 1.0, 0.25, 0.32],
            [0.15, 0.55, 1.0, 0.32],
            [1.0, 0.55, 0.1, 0.32],
            [0.85, 0.2, 1.0, 0.32],
        )
        axis_length = 0.075
        axis_width = 0.003
        axis_materials = (
            sapien.render.RenderMaterial(base_color=[1.0, 0.05, 0.05, 1.0]),
            sapien.render.RenderMaterial(base_color=[0.05, 1.0, 0.05, 1.0]),
            sapien.render.RenderMaterial(base_color=[0.05, 0.25, 1.0, 1.0]),
        )
        for index, item in enumerate(self.task_spec.get("targets") or []):
            transform = np.asarray(item["pose"], dtype=float).reshape(4, 4)
            scale = np.asarray(item.get("scale") or [1.0, 1.0, 1.0], dtype=float).reshape(3)
            target_name = f"target__{item.get('atom_id', index)}__{item.get('object_id', 'object')}"
            material = sapien.render.RenderMaterial(base_color=palette[index % len(palette)])
            builder = self.scene.create_actor_builder()
            builder.set_scene_idxs([0])
            builder.initial_pose = sapien.Pose(p=transform[:3, 3], q=mat2quat(transform[:3, :3]))
            builder.add_visual_from_file(
                str(Path(item["asset_file"]).expanduser().resolve()),
                scale=scale.tolist(),
                material=material,
            )
            builder.add_sphere_visual(radius=0.009, material=material)
            builder.add_box_visual(
                pose=sapien.Pose(p=[axis_length * 0.5, 0.0, 0.0]),
                half_size=[axis_length * 0.5, axis_width, axis_width],
                material=axis_materials[0],
            )
            builder.add_box_visual(
                pose=sapien.Pose(p=[0.0, axis_length * 0.5, 0.0]),
                half_size=[axis_width, axis_length * 0.5, axis_width],
                material=axis_materials[1],
            )
            builder.add_box_visual(
                pose=sapien.Pose(p=[0.0, 0.0, axis_length * 0.5]),
                half_size=[axis_width, axis_width, axis_length * 0.5],
                material=axis_materials[2],
            )
            builder.build_kinematic(name=target_name)

    @staticmethod
    def _axis_materials():
        return (
            sapien.render.RenderMaterial(base_color=[1.0, 0.05, 0.05, 1.0]),
            sapien.render.RenderMaterial(base_color=[0.05, 1.0, 0.05, 1.0]),
            sapien.render.RenderMaterial(base_color=[0.05, 0.25, 1.0, 1.0]),
        )

    def _load_grasp_preview_visuals(self) -> None:
        """Render schematic grippers and TCP frames at pregrasp and grasp."""
        previews = self.task_spec.get("grasp_previews") or []
        if not previews:
            return
        for index, item in enumerate(previews):
            for role, color in (
                ("pregrasp", [0.0, 0.85, 1.0, 0.38]),
                ("grasp", [1.0, 0.05, 0.55, 0.52]),
            ):
                transform = np.asarray(item[f"{role}_pose"], dtype=float).reshape(4, 4)
                material = sapien.render.RenderMaterial(base_color=color)
                axes = self._axis_materials()
                builder = self.scene.create_actor_builder()
                builder.set_scene_idxs([0])
                builder.initial_pose = sapien.Pose(
                    p=transform[:3, 3], q=mat2quat(transform[:3, :3])
                )
                # Schematic two-finger gripper in TCP coordinates.  The grasp
                # candidate's local +Z is the downward approach direction, so
                # the wrist and finger bodies extend along local -Z.  Drawing
                # them on +Z puts the rendered gripper below the object even
                # when the TCP pose itself is correct.
                builder.add_box_visual(
                    pose=sapien.Pose(p=[0.0, 0.0, -0.060]),
                    half_size=[0.050, 0.018, 0.012],
                    material=material,
                )
                for finger_y in (-0.043, 0.043):
                    builder.add_box_visual(
                        pose=sapien.Pose(p=[0.0, finger_y, -0.028]),
                        half_size=[0.012, 0.007, 0.028],
                        material=material,
                    )
                builder.add_sphere_visual(radius=0.008, material=material)
                axis_length = 0.060
                axis_width = 0.0022
                for axis_index in range(3):
                    center = np.zeros(3)
                    half_size = np.full(3, axis_width)
                    center[axis_index] = axis_length * 0.5
                    half_size[axis_index] = axis_length * 0.5
                    builder.add_box_visual(
                        pose=sapien.Pose(p=center),
                        half_size=half_size,
                        material=axes[axis_index],
                    )
                builder.build_kinematic(
                    name=f"{role}__{item.get('candidate_id', index)}"
                )

            # Dotted yellow corridor makes the commanded approach direction
            # unambiguous even when the two TCP frames overlap in projection.
            pre = np.asarray(item["pregrasp_pose"], dtype=float)[:3, 3]
            grasp = np.asarray(item["grasp_pose"], dtype=float)[:3, 3]
            corridor_material = sapien.render.RenderMaterial(
                base_color=[1.0, 0.85, 0.05, 0.9]
            )
            builder = self.scene.create_actor_builder()
            builder.set_scene_idxs([0])
            builder.initial_pose = sapien.Pose()
            for alpha in np.linspace(0.1, 0.9, 9):
                point = (1.0 - alpha) * pre + alpha * grasp
                builder.add_sphere_visual(
                    pose=sapien.Pose(p=point),
                    radius=0.0035,
                    material=corridor_material,
                )
            builder.build_kinematic(
                name=f"approach_corridor__{item.get('candidate_id', index)}"
            )

    @staticmethod
    def _add_box_wireframe(builder, half_size: np.ndarray, material, width: float) -> None:
        hx, hy, hz = np.asarray(half_size, dtype=float).reshape(3)
        for y in (-hy, hy):
            for z in (-hz, hz):
                builder.add_box_visual(
                    pose=sapien.Pose(p=[0.0, y, z]),
                    half_size=[hx, width, width],
                    material=material,
                )
        for x in (-hx, hx):
            for z in (-hz, hz):
                builder.add_box_visual(
                    pose=sapien.Pose(p=[x, 0.0, z]),
                    half_size=[width, hy, width],
                    material=material,
                )
        for x in (-hx, hx):
            for y in (-hy, hy):
                builder.add_box_visual(
                    pose=sapien.Pose(p=[x, y, 0.0]),
                    half_size=[width, width, hz],
                    material=material,
                )

    def _load_collision_proxy_visuals(self) -> None:
        """Overlay the effective cuRobo2 obstacle representation."""
        edge = sapien.render.RenderMaterial(
            base_color=[1.0, 0.35, 0.0, 1.0],
            emission=[0.55, 0.12, 0.0, 1.0],
        )
        for index, item in enumerate(self.task_spec.get("collision_proxies") or []):
            transform = np.asarray(item["pose"], dtype=float).reshape(4, 4)
            builder = self.scene.create_actor_builder()
            builder.set_scene_idxs([0])
            builder.initial_pose = sapien.Pose(
                p=transform[:3, 3], q=mat2quat(transform[:3, :3])
            )
            if str(item.get("kind")) == "sphere":
                radius = float(item["radius"])
                builder.add_sphere_visual(radius=radius, material=edge)
            else:
                dimensions = np.asarray(item["dimensions"], dtype=float).reshape(3)
                half_size = 0.5 * dimensions
                self._add_box_wireframe(
                    builder,
                    half_size,
                    edge,
                    width=max(0.0012, 0.012 * float(np.min(dimensions))),
                )
            builder.build_kinematic(
                name=f"collision_proxy__{item.get('object_id', index)}"
            )

    def _load_robot_collision_sphere_visuals(self) -> None:
        """Render cuRobo2 FK collision spheres at pregrasp and grasp IK."""
        colors = {
            "pregrasp": ([0.0, 0.95, 1.0, 0.34], [0.0, 0.35, 0.42, 1.0]),
            "grasp": ([1.0, 0.05, 0.55, 0.36], [0.48, 0.0, 0.20, 1.0]),
            "default": ([0.55, 0.65, 1.0, 0.34], [0.18, 0.22, 0.45, 1.0]),
        }
        for model_index, model in enumerate(
            self.task_spec.get("robot_collision_models") or []
        ):
            role = str(model.get("role") or "default")
            base_color, emission = colors.get(role, colors["default"])
            material = sapien.render.RenderMaterial(
                base_color=base_color,
                emission=emission,
            )
            builder = self.scene.create_actor_builder()
            builder.set_scene_idxs([0])
            T_world_model = np.asarray(
                model.get("pose") or np.eye(4), dtype=float
            ).reshape(4, 4)
            builder.initial_pose = sapien.Pose(
                p=T_world_model[:3, 3], q=mat2quat(T_world_model[:3, :3])
            )
            for sphere in model.get("spheres") or []:
                builder.add_sphere_visual(
                    pose=sapien.Pose(p=np.asarray(sphere["center"], dtype=float)),
                    radius=float(sphere["radius"]),
                    material=material,
                )
            builder.build_kinematic(
                name=f"robot_collision_spheres__{role}__{model_index}"
            )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        del options
        self.table_scene.initialize(env_idx)
        for object_id, actor in self.task_actors.items():
            transform = self._actor_initial_poses[object_id]
            actor.set_pose(
                Pose.create_from_pq(
                    p=transform[:3, 3],
                    q=mat2quat(transform[:3, :3]).astype(np.float32),
                    device=self.device,
                )
            )

    def evaluate(self):
        return {"success": torch.zeros((1,), dtype=torch.bool, device=self.device)}

    def _get_obs_extra(self, info):
        del info
        return {}

    def compute_dense_reward(self, obs, action, info):
        del obs, action, info
        return torch.zeros((1,), dtype=torch.float32, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)
