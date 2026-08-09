"""Small ManiSkill scene helpers used by the Curobo2 replay adapter."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

from rm75_app.assets.object_specs import get_object_spec


def ensure_pick_env_registered(env_id: str, extra_package_root: str) -> str:
    import gymnasium as gym

    if env_id in gym.envs.registry:
        return env_id
    package_root = Path(extra_package_root).expanduser()
    if package_root.name != "mani_skill" and (package_root / "mani_skill").exists():
        package_root = package_root / "mani_skill"
    if not package_root.exists():
        raise FileNotFoundError(f"extra ManiSkill package root not found: {package_root}")

    import mani_skill.agents.robots as robots_pkg
    import mani_skill.envs.tasks.digital_twins as digital_twins_pkg

    robots_path = str(package_root / "agents" / "robots")
    tasks_path = str(package_root / "envs" / "tasks" / "digital_twins")
    if robots_path not in robots_pkg.__path__:
        robots_pkg.__path__.append(robots_path)
    if tasks_path not in digital_twins_pkg.__path__:
        digital_twins_pkg.__path__.append(tasks_path)
    importlib.import_module("mani_skill.agents.robots.realman")
    importlib.import_module(
        "mani_skill.envs.tasks.digital_twins.so101_arm_with_two_cameras.pick_jiaobang"
    )
    if env_id not in gym.envs.registry:
        raise gym.error.NameNotFound(f"ManiSkill environment {env_id!r} is not registered")
    return env_id


def _pose_matrix(pose: Any) -> np.ndarray | None:
    if pose is None:
        return None
    matrix = pose.to_transformation_matrix() if hasattr(pose, "to_transformation_matrix") else pose
    if hasattr(matrix, "detach"):
        matrix = matrix.detach().cpu().numpy()
    matrix = np.asarray(matrix)
    if matrix.ndim == 3:
        matrix = matrix[0]
    return matrix.astype(np.float32) if matrix.shape == (4, 4) else None


def robot_base_transform(env: Any) -> np.ndarray | None:
    robot = getattr(getattr(env.unwrapped, "agent", None), "robot", None)
    return _pose_matrix(getattr(robot, "pose", None))


def object_pose_matrix(env: Any) -> np.ndarray | None:
    return _pose_matrix(getattr(getattr(env.unwrapped, "obj", None), "pose", None))


def set_object_pose(env: Any, T_world_object: np.ndarray) -> None:
    import torch
    from mani_skill.utils.structs.pose import Pose
    from transforms3d.quaternions import mat2quat

    transform = np.asarray(T_world_object, dtype=np.float32).reshape(4, 4)
    obj = env.unwrapped.obj
    obj.set_pose(
        Pose.create_from_pq(
            p=transform[:3, 3],
            q=mat2quat(transform[:3, :3]).astype(np.float32),
        )
    )
    try:
        zero = np.zeros(3, dtype=np.float32)
        obj.set_linear_velocity(zero)
        obj.set_angular_velocity(zero)
    except Exception:
        pass
    if hasattr(env.unwrapped, "object_initial_height"):
        env.unwrapped.object_initial_height[:] = -1.0
    if hasattr(env.unwrapped, "get_obj_xy_shortest_edge_vector") and hasattr(
        env.unwrapped, "obj_xy_shortest_edge_vector"
    ):
        with torch.no_grad():
            env.unwrapped.obj_xy_shortest_edge_vector[:] = (
                env.unwrapped.get_obj_xy_shortest_edge_vector().detach()
            )


def apply_object_physics_profile(env: Any, object_name: str) -> None:
    import sapien

    spec = get_object_spec(object_name)
    if spec is None:
        return
    values = (
        spec.sim_static_friction,
        spec.sim_dynamic_friction,
        spec.sim_restitution,
        spec.sim_linear_damping,
        spec.sim_angular_damping,
    )
    if all(value is None for value in values):
        return
    obj = getattr(env.unwrapped, "obj", None)
    if obj is None:
        return
    if spec.sim_linear_damping is not None:
        obj.set_linear_damping(float(spec.sim_linear_damping))
    if spec.sim_angular_damping is not None:
        obj.set_angular_damping(float(spec.sim_angular_damping))
    actors = list(getattr(env.unwrapped, "_objs", []) or getattr(obj, "_objs", []) or [])
    for actor in actors:
        component = actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        if component is None:
            continue
        for shape in component.collision_shapes:
            material = shape.physical_material
            if spec.sim_static_friction is not None:
                material.static_friction = float(spec.sim_static_friction)
            if spec.sim_dynamic_friction is not None:
                material.dynamic_friction = float(spec.sim_dynamic_friction)
            if spec.sim_restitution is not None:
                material.restitution = float(spec.sim_restitution)
