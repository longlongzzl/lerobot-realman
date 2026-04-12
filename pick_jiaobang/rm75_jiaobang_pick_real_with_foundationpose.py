#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import sapien
import trimesh
import yaml
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2mat
from transforms3d.quaternions import quat2mat

from object_specs import describe_object_specs, get_object_spec, list_object_spec_names, normalize_object_name, resolve_object_spec_scales


DEFAULT_BRIDGE_SCRIPT = "/home/zhangzhao/Desktop/lerobot/pick_jiaobang/rm75_jiaobang_pick_move_with_foundationpose.py"
DEFAULT_PICK_SCRIPT = "/home/zhangzhao/Desktop/lerobot/pick_jiaobang/rm75_jiaobang_pick_move_v10_perpendicular_to_object.py"
DEFAULT_LEROBOT_ROOT = "/home/zhangzhao/Desktop/lerobot"
DEFAULT_LEROBOT_SIM2REAL_ROOT = "/home/zhangzhao/Desktop/lerobot/lerobot-sim2real"
DEFAULT_FOUNDATIONPOSE_ROOT = "/home/zhangzhao/PycharmProjects/FoundationPose"
DEFAULT_EXTRA_MANISKILL_PACKAGE_ROOT = "/home/zhangzhao/anaconda3/envs/realman/lib/python3.11/site-packages/mani_skill"
DEFAULT_CAMERA_EXTRINSIC = "/home/zhangzhao/Desktop/lerobot-sim2real/results/realman/realman_home/base_camera/camera_extrinsic_opencv.npy"
DEFAULT_MESH_FILE = "/home/zhangzhao/anaconda3/envs/realman/lib/python3.11/site-packages/mani_skill/envs/tasks/digital_twins/so101_arm_with_two_cameras/jiaobang.glb"
DEFAULT_MESH_SCALE = 0.1


@dataclass
class PlannedGraspExecution:
    label: str
    approach_pose: Pose
    approach_q: np.ndarray
    final_grasp_pose: Pose
    final_grasp_path: list[np.ndarray] | None
    require_runtime_true_grasp_planning: bool = False


def maybe_print_and_exit_object_specs(args):
    if not getattr(args, "list_objects", False):
        return
    print("Available object specs:")
    print(describe_object_specs() or "(none)")
    raise SystemExit(0)



def apply_object_spec_defaults(args):
    maybe_print_and_exit_object_specs(args)
    if not hasattr(args, "grasp_mode") or args.grasp_mode is None:
        args.grasp_mode = "object_normal"
    spec = get_object_spec(getattr(args, "object_name", None))
    if args.object_name and spec is None:
        raise ValueError(f"Unknown object spec: {args.object_name}")

    if spec is None:
        if args.sim_asset_file is None:
            args.sim_asset_file = args.mesh_file
        if args.sim_asset_scale is None:
            args.sim_asset_scale = args.mesh_scale
        if args.mesh_file is not None:
            args.mesh_file = str(Path(args.mesh_file).expanduser())
        if args.sim_asset_file is not None:
            args.sim_asset_file = str(Path(args.sim_asset_file).expanduser())
        return None

    spec_mesh_scale, spec_sim_asset_scale = resolve_object_spec_scales(spec)
    if args.mesh_file == DEFAULT_MESH_FILE:
        args.mesh_file = spec.mesh_file
    if float(args.mesh_scale) == float(DEFAULT_MESH_SCALE):
        args.mesh_scale = spec_mesh_scale
    if args.sim_asset_file is None:
        args.sim_asset_file = spec.sim_asset_file or spec.mesh_file
    if args.sim_asset_scale is None:
        args.sim_asset_scale = spec_sim_asset_scale
    if args.target_object_name is None:
        args.target_object_name = spec.grounding_prompt
    if np.allclose(np.asarray(args.foundationpose_position_offset, dtype=np.float32), 0.0) and not np.allclose(np.asarray(spec.foundationpose_position_offset, dtype=np.float32), 0.0):
        args.foundationpose_position_offset = list(spec.foundationpose_position_offset)
    if np.allclose(np.asarray(args.foundationpose_local_rotation_offset_deg, dtype=np.float32), 0.0) and not np.allclose(np.asarray(spec.foundationpose_local_rotation_offset_deg, dtype=np.float32), 0.0):
        args.foundationpose_local_rotation_offset_deg = list(spec.foundationpose_local_rotation_offset_deg)
    if spec.pregrasp_height is not None and float(args.pregrasp_height) == 0.07:
        args.pregrasp_height = spec.pregrasp_height
    if spec.grasp_z_offset is not None and float(args.grasp_z_offset) == 0.0:
        args.grasp_z_offset = spec.grasp_z_offset
    if spec.goal_z_offset is not None and float(args.goal_z_offset) == 0.0:
        args.goal_z_offset = spec.goal_z_offset
    if spec.fixed_goal_joints_deg is not None and list(args.fixed_goal_joints_deg) == [178.0, -5.0, 0.0, -70.0, 0.0, -102.0, 60.0]:
        args.fixed_goal_joints_deg = list(spec.fixed_goal_joints_deg)
    if getattr(args, "grasp_mode", "object_normal") == "object_normal" and getattr(spec, "grasp_mode", "object_normal") != "object_normal":
        args.grasp_mode = spec.grasp_mode
    if args.mesh_file is not None:
        args.mesh_file = str(Path(args.mesh_file).expanduser())
    if args.sim_asset_file is not None:
        args.sim_asset_file = str(Path(args.sim_asset_file).expanduser())
    return spec



def load_module_from_path(module_name: str, file_path: str | Path):
    file_path = Path(file_path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flatten_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


_ASSET_BOX_SIZE_CACHE: dict[tuple[str, float, int, int], np.ndarray] = {}
_ASSET_LOCAL_POINTS_CACHE: dict[tuple[str, float, int, int], np.ndarray] = {}


def get_asset_box_size(asset_file: str, asset_scale: float) -> np.ndarray:
    asset_path = Path(asset_file).expanduser().resolve()
    scale = float(asset_scale)
    stat = asset_path.stat()
    cache_key = (str(asset_path), scale, int(stat.st_mtime_ns), int(stat.st_size))
    if cache_key in _ASSET_BOX_SIZE_CACHE:
        return _ASSET_BOX_SIZE_CACHE[cache_key].copy()

    loaded = trimesh.load(asset_path, force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
    else:
        mesh = loaded.dump(concatenate=True)
        if mesh is None:
            raise ValueError(f"No trimesh geometry found in {asset_path}")
    if abs(scale - 1.0) > 1e-8:
        mesh.apply_scale(scale)
    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    if bounds.shape != (2, 3):
        raise ValueError(f"Unexpected bounds shape for {asset_path}: {bounds.shape}")
    size = np.maximum(bounds[1] - bounds[0], 1e-4).astype(np.float32)
    _ASSET_BOX_SIZE_CACHE[cache_key] = size
    return size.copy()


def get_asset_local_points(asset_file: str, asset_scale: float) -> np.ndarray:
    asset_path = Path(asset_file).expanduser().resolve()
    scale = float(asset_scale)
    stat = asset_path.stat()
    cache_key = (str(asset_path), scale, int(stat.st_mtime_ns), int(stat.st_size))
    if cache_key in _ASSET_LOCAL_POINTS_CACHE:
        return _ASSET_LOCAL_POINTS_CACHE[cache_key].copy()

    loaded = trimesh.load(asset_path, force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
    else:
        mesh = loaded.dump(concatenate=True)
        if mesh is None:
            raise ValueError(f"No trimesh geometry found in {asset_path}")
    if abs(scale - 1.0) > 1e-8:
        mesh.apply_scale(scale)
    points = np.asarray(mesh.vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"Unexpected vertex array for {asset_path}: {points.shape}")
    _ASSET_LOCAL_POINTS_CACHE[cache_key] = points
    return points.copy()


def build_visual_obstacle_actor(env, mesh_file: str, mesh_scale: float, actor_name: str, *, box_size: np.ndarray | None = None):
    builder = env.unwrapped.scene.create_actor_builder()
    scale = [float(mesh_scale), float(mesh_scale), float(mesh_scale)]
    builder.add_visual_from_file(str(Path(mesh_file).expanduser()), scale=scale)
    if box_size is not None:
        half_size = (np.asarray(box_size, dtype=np.float32).reshape(3) * 0.5).tolist()
        try:
            builder.add_box_collision(half_size=half_size)
        except Exception as exc:
            print(f"[scene obstacle] {actor_name}: failed to add sim collision shape: {exc}")
    return builder.build_kinematic(name=actor_name)


def build_attached_box_visual_actor(env, box_size: np.ndarray, actor_name: str = "transport_attached_box_visual"):
    builder = env.unwrapped.scene.create_actor_builder()
    try:
        builder.set_initial_pose(sapien.Pose())
    except Exception:
        pass
    half_size = (np.asarray(box_size, dtype=np.float32).reshape(3) * 0.5).tolist()
    builder.add_box_visual(
        half_size=half_size,
        material=sapien.render.RenderMaterial(base_color=[1.0, 0.55, 0.0, 0.35]),
    )
    return builder.build_kinematic(name=actor_name)


def build_visual_box_actor(
    env,
    box_size: np.ndarray,
    actor_name: str,
    *,
    color=(0.25, 0.55, 1.0, 0.18),
):
    builder = env.unwrapped.scene.create_actor_builder()
    try:
        builder.set_initial_pose(sapien.Pose())
    except Exception:
        pass
    half_size = (np.asarray(box_size, dtype=np.float32).reshape(3) * 0.5).tolist()
    builder.add_box_visual(
        half_size=half_size,
        material=sapien.render.RenderMaterial(base_color=list(color)),
    )
    return builder.build_kinematic(name=actor_name)


def build_visual_sphere_actor(
    env,
    radius: float,
    actor_name: str,
    *,
    color=(1.0, 0.9, 0.2, 0.75),
):
    builder = env.unwrapped.scene.create_actor_builder()
    try:
        builder.set_initial_pose(sapien.Pose())
    except Exception:
        pass
    builder.add_sphere_visual(
        radius=float(radius),
        material=sapien.render.RenderMaterial(base_color=list(color)),
    )
    return builder.build_kinematic(name=actor_name)


def should_render_scene_obstacle_planner_boxes(args) -> bool:
    return bool(
        getattr(args, "render_scene_obstacle_planner_boxes", False)
        or getattr(args, "curobo_debug", False)
    )


def get_scene_obstacle_planner_box_color(object_name: str):
    normalized = normalize_object_name(object_name) or str(object_name)
    if normalized == "desk":
        return (1.0, 0.25, 0.15, 0.24)
    if normalized == "bitong":
        return (0.15, 0.95, 0.55, 0.22)
    return (0.15, 0.75, 1.0, 0.18)


def find_named_actor(actors, actor_name: str):
    target = str(actor_name or "")
    if not target:
        return None
    for actor in list(actors or []):
        try:
            candidate_name = str(getattr(actor, "name", "") or "")
        except Exception:
            candidate_name = ""
        if candidate_name == target:
            return actor
    return None


def sync_scene_obstacle_planner_box_actor(env, item, pos: np.ndarray, quat: np.ndarray) -> None:
    actor_name = str(item.get("planner_box_actor_name", "") or "")
    if not actor_name:
        return
    actor = find_named_actor(getattr(env.unwrapped, "_scene_obstacle_planner_box_actors", []) or [], actor_name)
    if actor is None:
        return
    actor.set_pose(Pose.create_from_pq(p=np.asarray(pos, dtype=np.float32), q=np.asarray(quat, dtype=np.float32)))


def should_render_robot_collision_spheres(args) -> bool:
    return bool(
        getattr(args, "render_robot_collision_spheres", False)
        or getattr(args, "curobo_debug", False)
    )


def should_render_attached_object_collision_spheres(args) -> bool:
    return should_render_robot_collision_spheres(args)


def _default_curobo_robot_cfg_path() -> Path:
    return Path(__file__).resolve().parent / "curobo_rm75_config" / "rm75.yml"


def _resolve_curobo_robot_cfg_path_from_args(args) -> Path | None:
    cfg_value = getattr(args, "curobo_rm75_robot_cfg", None)
    if cfg_value is not None:
        return Path(cfg_value).expanduser().resolve()
    default_path = _default_curobo_robot_cfg_path()
    if default_path.is_file():
        return default_path
    return None


def _get_robot_collision_sphere_specs_from_cfg(args):
    cfg_path = _resolve_curobo_robot_cfg_path_from_args(args)
    if cfg_path is None or not cfg_path.is_file():
        return []
    with cfg_path.open("r", encoding="utf-8") as f:
        robot_cfg_data = yaml.safe_load(f) or {}
    kinematics = (((robot_cfg_data or {}).get("robot_cfg") or {}).get("kinematics") or {})
    spheres_value = kinematics.get("collision_spheres")
    if spheres_value in (None, "", {}):
        return []
    spheres_path = Path(spheres_value)
    if not spheres_path.is_absolute():
        spheres_path = (cfg_path.parent / spheres_path).resolve()
    with spheres_path.open("r", encoding="utf-8") as f:
        sphere_data = yaml.safe_load(f) or {}
    collision_spheres = (sphere_data or {}).get("collision_spheres") or {}
    sphere_buffer_cfg = kinematics.get("collision_sphere_buffer", 0.0)
    specs = []
    for link_name, entries in collision_spheres.items():
        if not isinstance(entries, list):
            continue
        if isinstance(sphere_buffer_cfg, dict):
            link_buffer = float(sphere_buffer_cfg.get(link_name, 0.0) or 0.0)
        else:
            link_buffer = float(sphere_buffer_cfg or 0.0)
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            center = np.asarray(entry.get("center", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
            radius = float(entry.get("radius", 0.0) or 0.0) + link_buffer
            if radius <= 0.0:
                continue
            specs.append(
                {
                    "link_name": str(link_name),
                    "center": center,
                    "radius": float(radius),
                    "actor_name": f"robot_collision_sphere__{link_name}__{idx:02d}",
                }
            )
    return specs


def _get_attached_object_collision_sphere_specs_from_cfg(args, box_size: np.ndarray):
    cfg_path = _resolve_curobo_robot_cfg_path_from_args(args)
    if cfg_path is None or not cfg_path.is_file():
        return []
    with cfg_path.open("r", encoding="utf-8") as f:
        robot_cfg_data = yaml.safe_load(f) or {}
    kinematics = (((robot_cfg_data or {}).get("robot_cfg") or {}).get("kinematics") or {})
    extra_collision_spheres = kinematics.get("extra_collision_spheres") or {}
    sphere_count = int((extra_collision_spheres or {}).get("attached_object", 0) or 0)
    if sphere_count <= 0:
        return []

    box_size = np.asarray(box_size, dtype=np.float32).reshape(3)
    axis_idx = int(np.argmax(box_size))
    short_axes = [idx for idx in range(3) if idx != axis_idx]
    short_dims = box_size[short_axes]
    radius = float(max(0.5 * np.linalg.norm(short_dims), 1e-4))
    half_extent = float(max(0.5 * float(box_size[axis_idx]) - radius, 0.0))
    if sphere_count == 1:
        centers_1d = [0.0]
    else:
        centers_1d = np.linspace(-half_extent, half_extent, sphere_count, dtype=np.float32).tolist()

    specs = []
    for idx, center_axis in enumerate(centers_1d):
        center = np.zeros(3, dtype=np.float32)
        center[axis_idx] = float(center_axis)
        specs.append(
            {
                "center": center,
                "radius": radius,
                "actor_name": f"attached_object_collision_sphere__{idx:02d}",
            }
        )
    return specs


def _get_demo_robot_links_map(demo):
    links_map = getattr(demo.robot, "links_map", None)
    if isinstance(links_map, dict) and links_map:
        return {str(k): v for k, v in links_map.items()}
    result = {}
    try:
        links = list(demo.robot.get_links())
    except Exception:
        links = []
    for link in links:
        try:
            name = str(getattr(link, "name", "") or "")
        except Exception:
            name = ""
        if name:
            result[name] = link
    return result


def ensure_robot_collision_sphere_visuals(demo, args) -> None:
    if not should_render_robot_collision_spheres(args):
        return
    specs = getattr(demo, "_robot_collision_sphere_specs", None)
    if specs is None:
        specs = _get_robot_collision_sphere_specs_from_cfg(args)
        demo._robot_collision_sphere_specs = specs
    if not specs:
        return
    actors = list(getattr(demo.env.unwrapped, "_robot_collision_sphere_actors", []) or [])
    existing_names = {str(getattr(actor, "name", "") or "") for actor in actors}
    created = 0
    for spec in specs:
        actor_name = str(spec.get("actor_name", "") or "")
        if not actor_name or actor_name in existing_names:
            continue
        actor = build_visual_sphere_actor(
            demo.env,
            float(spec["radius"]),
            actor_name,
            color=(1.0, 0.9, 0.2, 0.75),
        )
        actors.append(actor)
        existing_names.add(actor_name)
        created += 1
    demo.env.unwrapped._robot_collision_sphere_actors = actors
    if created > 0:
        print(f"[curobo] rendered {created} robot collision sphere actor(s)")


def ensure_attached_object_collision_sphere_visuals(demo, args) -> None:
    if not should_render_attached_object_collision_spheres(args):
        return
    box_size = getattr(demo, "attached_box_size", None)
    if box_size is None:
        return
    specs = _get_attached_object_collision_sphere_specs_from_cfg(args, box_size)
    demo._attached_object_collision_sphere_specs = specs
    actors = list(getattr(demo.env.unwrapped, "_attached_object_collision_sphere_actors", []) or [])
    existing_names = {str(getattr(actor, "name", "") or "") for actor in actors}
    created = 0
    for spec in specs:
        actor_name = str(spec.get("actor_name", "") or "")
        if not actor_name or actor_name in existing_names:
            continue
        actor = build_visual_sphere_actor(
            demo.env,
            float(spec["radius"]),
            actor_name,
            color=(1.0, 0.45, 0.05, 0.55),
        )
        actors.append(actor)
        existing_names.add(actor_name)
        created += 1
    demo.env.unwrapped._attached_object_collision_sphere_actors = actors
    if created > 0:
        print(f"[curobo] rendered {created} attached-object collision sphere actor(s)")


def update_robot_collision_sphere_visuals(demo, args) -> None:
    if not should_render_robot_collision_spheres(args):
        return
    ensure_robot_collision_sphere_visuals(demo, args)
    specs = list(getattr(demo, "_robot_collision_sphere_specs", []) or [])
    if not specs:
        return
    actors = list(getattr(demo.env.unwrapped, "_robot_collision_sphere_actors", []) or [])
    if not actors:
        return
    actor_map = {str(getattr(actor, "name", "") or ""): actor for actor in actors}
    link_map = _get_demo_robot_links_map(demo)
    hidden_pos = np.asarray([0.0, 0.0, -10.0], dtype=np.float32)
    identity_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for spec in specs:
        actor = actor_map.get(str(spec.get("actor_name", "") or ""))
        if actor is None:
            continue
        link = link_map.get(str(spec.get("link_name", "") or ""))
        if link is None:
            actor.set_pose(Pose.create_from_pq(p=hidden_pos, q=identity_quat))
            continue
        link_pose = getattr(link, "pose", None)
        if link_pose is None:
            actor.set_pose(Pose.create_from_pq(p=hidden_pos, q=identity_quat))
            continue
        link_p = flatten_np(link_pose.p)[:3].astype(np.float32)
        link_q = flatten_np(link_pose.q)[:4].astype(np.float32)
        center_local = np.asarray(spec["center"], dtype=np.float32).reshape(3)
        center_world = link_p + quat2mat(link_q).astype(np.float32) @ center_local
        actor.set_pose(Pose.create_from_pq(p=center_world.astype(np.float32), q=identity_quat))


def update_attached_object_collision_sphere_visuals(demo, args, *, visible: bool | None = None) -> None:
    if not should_render_attached_object_collision_spheres(args):
        return
    ensure_attached_object_collision_sphere_visuals(demo, args)
    specs = list(getattr(demo, "_attached_object_collision_sphere_specs", []) or [])
    actors = list(getattr(demo.env.unwrapped, "_attached_object_collision_sphere_actors", []) or [])
    if not actors:
        return
    actor_map = {str(getattr(actor, "name", "") or ""): actor for actor in actors}
    hidden_pos = np.asarray([0.0, 0.0, -10.0], dtype=np.float32)
    identity_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if visible is None:
        visible = bool(getattr(demo, "_attached_box_visual_visible", False))
    if not visible or getattr(demo, "attached_box_pose_tcp", None) is None:
        for actor in actors:
            actor.set_pose(Pose.create_from_pq(p=hidden_pos, q=identity_quat))
        return
    T_world_box = compute_world_attached_pose(demo)
    if T_world_box is None:
        return
    R_world_box = np.asarray(T_world_box[:3, :3], dtype=np.float32)
    p_world_box = np.asarray(T_world_box[:3, 3], dtype=np.float32).reshape(3)
    for spec in specs:
        actor = actor_map.get(str(spec.get("actor_name", "") or ""))
        if actor is None:
            continue
        center_local = np.asarray(spec["center"], dtype=np.float32).reshape(3)
        center_world = p_world_box + R_world_box @ center_local
        actor.set_pose(Pose.create_from_pq(p=center_world.astype(np.float32), q=identity_quat))


def register_planner_virtual_side_wall(env, demo, args):
    if not bool(getattr(args, "planner_virtual_side_wall", True)):
        return None

    from mplib import collision_detection as mplib_cd

    wall_x = float(getattr(args, "planner_virtual_side_wall_x", -0.30))
    wall_thickness = float(max(getattr(args, "planner_virtual_side_wall_thickness", 0.03), 1e-3))
    wall_width_y = float(max(getattr(args, "planner_virtual_side_wall_width_y", 1.20), 1e-3))
    wall_height = float(max(getattr(args, "planner_virtual_side_wall_height", 0.80), 1e-3))
    wall_y = float(getattr(args, "planner_virtual_side_wall_y", 0.0))
    wall_bottom_z = float(getattr(args, "planner_virtual_side_wall_bottom_z", 0.0))

    box_size = np.asarray([wall_thickness, wall_width_y, wall_height], dtype=np.float32)
    pos = np.asarray(
        [
            wall_x - 0.5 * wall_thickness,
            wall_y,
            wall_bottom_z + 0.5 * wall_height,
        ],
        dtype=np.float32,
    )
    quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    actor_name = "scene_obstacle_virtual_side_wall"

    actor = build_visual_box_actor(env, box_size, actor_name)
    actor.set_pose(Pose.create_from_pq(p=pos, q=quat))

    planner_collision = False
    try:
        collision_object = mplib_cd.fcl.CollisionObject(
            mplib_cd.fcl.Box(box_size.tolist()),
            pos.tolist(),
            quat.tolist(),
        )
        demo.planner.set_normal_object(actor_name, collision_object)
        planner_collision = True
    except Exception as exc:
        print(f"[planner wall] failed to add planner collision object: {exc}")

    actors = list(getattr(env.unwrapped, "_scene_obstacle_actors", []) or [])
    actors.append(actor)
    env.unwrapped._scene_obstacle_actors = actors

    entry = {
        "object_name": "virtual_side_wall",
        "actor_name": actor_name,
        "label": "virtual_side_wall",
        "score": 1.0,
        "T_world_obj": pose_to_matrix(pos, quat),
        "planner_collision": planner_collision,
        "asset_file": None,
        "asset_scale": None,
        "visual_box_size": box_size.copy(),
        "planner_box_size": box_size.copy(),
    }
    scene_obstacles = list(getattr(demo, "scene_obstacles", []) or [])
    scene_obstacles.append(entry)
    demo.scene_obstacles = scene_obstacles

    print(
        f"[planner wall] applied virtual_side_wall: front_face_x={wall_x:.3f}, "
        f"center={np.round(pos, 6).tolist()}, box_size={np.round(box_size, 6).tolist()}, "
        f"planner_collision={'yes' if planner_collision else 'no'}"
    )
    return entry


def register_planner_virtual_top_wall(env, demo, args):
    if not bool(getattr(args, "planner_virtual_top_wall", True)):
        return None

    from mplib import collision_detection as mplib_cd

    wall_z = float(getattr(args, "planner_virtual_top_wall_z", 1.50))
    wall_thickness = float(max(getattr(args, "planner_virtual_top_wall_thickness", 0.03), 1e-3))
    wall_width_x = float(max(getattr(args, "planner_virtual_top_wall_width_x", 1.20), 1e-3))
    wall_width_y = float(max(getattr(args, "planner_virtual_top_wall_width_y", 1.20), 1e-3))
    wall_x = float(getattr(args, "planner_virtual_top_wall_x", 0.0))
    wall_y = float(getattr(args, "planner_virtual_top_wall_y", 0.0))

    box_size = np.asarray([wall_width_x, wall_width_y, wall_thickness], dtype=np.float32)
    pos = np.asarray(
        [
            wall_x,
            wall_y,
            wall_z + 0.5 * wall_thickness,
        ],
        dtype=np.float32,
    )
    quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    actor_name = "scene_obstacle_virtual_top_wall"

    actor = build_visual_box_actor(env, box_size, actor_name, color=(1.0, 0.45, 0.25, 0.14))
    actor.set_pose(Pose.create_from_pq(p=pos, q=quat))

    planner_collision = False
    try:
        collision_object = mplib_cd.fcl.CollisionObject(
            mplib_cd.fcl.Box(box_size.tolist()),
            pos.tolist(),
            quat.tolist(),
        )
        demo.planner.set_normal_object(actor_name, collision_object)
        planner_collision = True
    except Exception as exc:
        print(f"[planner wall] failed to add planner collision object: {exc}")

    actors = list(getattr(env.unwrapped, "_scene_obstacle_actors", []) or [])
    actors.append(actor)
    env.unwrapped._scene_obstacle_actors = actors

    entry = {
        "object_name": "virtual_top_wall",
        "actor_name": actor_name,
        "label": "virtual_top_wall",
        "score": 1.0,
        "T_world_obj": pose_to_matrix(pos, quat),
        "planner_collision": planner_collision,
        "asset_file": None,
        "asset_scale": None,
        "visual_box_size": box_size.copy(),
        "planner_box_size": box_size.copy(),
    }
    scene_obstacles = list(getattr(demo, "scene_obstacles", []) or [])
    scene_obstacles.append(entry)
    demo.scene_obstacles = scene_obstacles

    print(
        f"[planner wall] applied virtual_top_wall: bottom_z={wall_z:.3f}, "
        f"center={np.round(pos, 6).tolist()}, box_size={np.round(box_size, 6).tolist()}, "
        f"planner_collision={'yes' if planner_collision else 'no'}"
    )
    return entry


def register_scene_obstacles(env, demo, bridge_mod, T_base_cam: np.ndarray, scene_obstacles, args):
    if not scene_obstacles:
        env.unwrapped._scene_obstacle_actors = []
        env.unwrapped._scene_obstacle_planner_box_actors = []
        demo.scene_obstacles = []
        applied = []
        wall_entry = register_planner_virtual_side_wall(env, demo, args)
        if wall_entry is not None:
            applied.append(wall_entry)
        top_entry = register_planner_virtual_top_wall(env, demo, args)
        if top_entry is not None:
            applied.append(top_entry)
        return applied

    from mplib import collision_detection as mplib_cd

    actors = []
    planner_box_actors = []
    applied = []
    for item in scene_obstacles:
        object_name = str(item["object_name"])
        object_args = item["object_args"]
        object_spec = get_object_spec(object_name)
        is_placed_obstacle = bool(item.get("placed", False))
        T_world_obj_override = item.get("T_world_obj")
        if T_world_obj_override is None and bool(getattr(args, "foundationpose_print_obstacle_mapping_diagnostics", False)):
            bridge_mod.print_foundationpose_mapping_diagnostics(
                item["T_cam_obj"],
                T_base_cam,
                env,
                object_args,
                label=f"scene obstacle {object_name}",
            )
        if T_world_obj_override is not None:
            T_world_obj = np.asarray(T_world_obj_override, dtype=np.float32).reshape(4, 4)
            print(f"[scene obstacle] {object_name}: using cached world pose override for continuous multi-cycle placement")
        else:
            T_world_obj = bridge_mod.map_camera_pose_to_pick_world(item["T_cam_obj"], T_base_cam, env, object_args)

        asset_file = str(Path(object_args.sim_asset_file or object_args.mesh_file).expanduser())
        asset_scale = float(object_args.sim_asset_scale or object_args.mesh_scale or 1.0)
        actor_name = f"scene_obstacle_{object_name}"
        box_size = None
        try:
            box_size = get_asset_box_size(asset_file, asset_scale)
        except Exception as exc:
            print(f"[scene obstacle] {object_name}: failed to compute box size for sim collision: {exc}")
        global_box_scale = float(max(getattr(args, "scene_obstacle_box_scale", 1.0), 1e-3))
        object_box_scale = float(max(getattr(object_spec, "scene_obstacle_box_scale", 1.0) or 1.0, 1e-3))
        placed_box_scale = (
            float(max(getattr(args, "placed_scene_obstacle_box_scale", 1.0), 1e-3))
            if is_placed_obstacle
            else 1.0
        )
        box_scale = float(global_box_scale * object_box_scale * placed_box_scale)
        scaled_box_size = None if box_size is None else (np.asarray(box_size, dtype=np.float32) * box_scale).astype(np.float32)

        actor = build_visual_obstacle_actor(env, asset_file, asset_scale, actor_name, box_size=scaled_box_size)
        pos = T_world_obj[:3, 3].astype(np.float32)
        quat = bridge_mod.mat2quat(T_world_obj[:3, :3]).astype(np.float32)
        actor.set_pose(Pose.create_from_pq(p=pos, q=quat))
        actors.append(actor)

        planner_box_actor_name = ""
        if scaled_box_size is not None and should_render_scene_obstacle_planner_boxes(args):
            planner_box_actor_name = f"{actor_name}__planner_box"
            planner_box_actor = build_visual_box_actor(
                env,
                scaled_box_size,
                planner_box_actor_name,
                color=get_scene_obstacle_planner_box_color(object_name),
            )
            planner_box_actor.set_pose(Pose.create_from_pq(p=pos, q=quat))
            planner_box_actors.append(planner_box_actor)

        try:
            if scaled_box_size is None:
                raw_box_size = get_asset_box_size(asset_file, asset_scale)
                scaled_box_size = (np.asarray(raw_box_size, dtype=np.float32) * box_scale).astype(np.float32)
            collision_object = mplib_cd.fcl.CollisionObject(
                mplib_cd.fcl.Box(scaled_box_size.tolist()),
                pos.tolist(),
                quat.tolist(),
            )
            demo.planner.set_normal_object(actor_name, collision_object)
            planner_collision = True
        except Exception as exc:
            planner_collision = False
            print(f"[scene obstacle] {object_name}: failed to add planner collision object: {exc}")

        applied.append({
            "object_name": object_name,
            "actor_name": actor_name,
            "label": item.get("label", object_name),
            "score": float(item.get("score", 0.0)),
            "T_world_obj": T_world_obj,
            "placed": is_placed_obstacle,
            "planner_collision": planner_collision,
            "asset_file": asset_file,
            "asset_scale": float(asset_scale),
            "visual_box_size": None if box_size is None else np.asarray(box_size, dtype=np.float32).copy(),
            "planner_box_size": None if scaled_box_size is None else np.asarray(scaled_box_size, dtype=np.float32).copy(),
            "planner_box_actor_name": planner_box_actor_name,
        })
        print(
            f"[scene obstacle] applied {object_name}: world translation={np.round(T_world_obj[:3, 3], 6).tolist()}, "
            f"planner_collision={'yes' if planner_collision else 'no'}, "
            f"box_scale={box_scale:.3f} (global={global_box_scale:.3f}, object={object_box_scale:.3f}, placed={placed_box_scale:.3f}), "
            f"box_size={None if scaled_box_size is None else np.round(scaled_box_size, 6).tolist()}"
        )
        if planner_box_actor_name:
            print(
                f"[scene obstacle] rendered planner box for {object_name}: "
                f"actor={planner_box_actor_name}, box_size={np.round(np.asarray(scaled_box_size, dtype=np.float32), 6).tolist()}"
            )

    env.unwrapped._scene_obstacle_actors = actors
    env.unwrapped._scene_obstacle_planner_box_actors = planner_box_actors
    demo.scene_obstacles = applied
    wall_entry = register_planner_virtual_side_wall(env, demo, args)
    if wall_entry is not None:
        applied = list(applied) + [wall_entry]
    top_entry = register_planner_virtual_top_wall(env, demo, args)
    if top_entry is not None:
        applied = list(applied) + [top_entry]
    return applied


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Initialize jiaobang pose from realtime FoundationPose, plan in ManiSkill, and optionally execute the planned RM75 motions on the real robot."
    )
    parser.add_argument("--bridge-script-path", type=str, default=DEFAULT_BRIDGE_SCRIPT)
    parser.add_argument("--pick-script-path", type=str, default=DEFAULT_PICK_SCRIPT)
    parser.add_argument("--foundationpose-root", type=str, default=DEFAULT_FOUNDATIONPOSE_ROOT)
    parser.add_argument("--lerobot-root", type=str, default=DEFAULT_LEROBOT_ROOT)
    parser.add_argument("--lerobot-sim2real-root", type=str, default=DEFAULT_LEROBOT_SIM2REAL_ROOT)
    parser.add_argument("--extra-maniskill-package-root", type=str, default=DEFAULT_EXTRA_MANISKILL_PACKAGE_ROOT)
    parser.add_argument("--object-name", type=str, default=None, help="Object-spec key from object_specs.py.")
    parser.add_argument(
        "--cycle-object-names",
        type=str,
        nargs="*",
        default=None,
        help="Optional per-cycle target object spec keys. When provided, cycles consume these targets in order instead of prompting interactively.",
    )
    parser.add_argument(
        "--selected-obstacle-object-names",
        type=str,
        nargs="*",
        default=None,
        help="Optional obstacle object spec keys for the first cycle. If omitted, the script prompts interactively on cycle 1; later cached cycles auto-use the remaining objects as obstacles.",
    )
    parser.add_argument("--list-objects", action="store_true", help="List available object specs and exit.")

    parser.add_argument("--env-id", type=str, default="Two_finger_PickJiaobang-v1")
    parser.add_argument("--mesh-file", type=str, default=DEFAULT_MESH_FILE)
    parser.add_argument("--mesh-scale", type=float, default=DEFAULT_MESH_SCALE)
    parser.add_argument("--sim-asset-file", type=str, default=None)
    parser.add_argument("--sim-asset-scale", type=float, default=None)
    parser.add_argument("--camera-extrinsic-opencv-path", type=str, default=DEFAULT_CAMERA_EXTRINSIC)
    parser.add_argument(
        "--use-direct-camera-extrinsic",
        action="store_true",
        help="Interpret camera_extrinsic_opencv as direct T_base_cam. Default matches the previous export/replay chain and applies inverse first.",
    )
    parser.add_argument("--debug", type=int, default=0, help="Enable verbose debug diagnostics.")
    parser.add_argument(
        "--debug-dir",
        type=str,
        default="/home/zhangzhao/Desktop/lerobot/debug_jiaobang_foundationpose_real",
    )
    parser.add_argument("--est-refine-iter", dest="est_refine_iter", type=int, default=5)
    parser.add_argument("--track-refine-iter", dest="track_refine_iter", type=int, default=2)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--foundationpose-stabilize-frames", type=int, default=6, help="After FoundationPose registration, run tracking for this many additional frames before using the pose. Set to 0 to disable.")
    parser.add_argument("--foundationpose-smoothing-window", type=int, default=4, help="Smooth the final pose by averaging the last few tracked poses. Set to 1 to disable smoothing.")
    parser.add_argument("--init-mask", type=str, default=None)
    parser.add_argument("--target-object-name", type=str, default=None, help="Use GroundingDINO to detect this object and initialize FoundationPose from the best matched box.")
    parser.add_argument("--grounding-dino-model-id", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--grounding-dino-box-threshold", type=float, default=0.25)
    parser.add_argument("--grounding-dino-text-threshold", type=float, default=0.2)
    parser.add_argument("--disable-object-spec-obstacles", action="store_true", help="Do not estimate non-target object_specs as additional scene obstacles for planning.")
    parser.add_argument(
        "--placed-scene-obstacle-box-scale",
        type=float,
        default=1.0,
        help="Extra planner-box scale multiplier applied only to already placed scene obstacles cached from previous cycles.",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-serial", type=str, default=None)
    parser.add_argument(
        "--foundationpose-position-offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
    )
    parser.add_argument(
        "--foundationpose-local-rotation-offset-deg",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("ROLL", "PITCH", "YAW"),
    )
    parser.add_argument("--no-map-foundationpose-through-robot-base", action="store_true")
    parser.add_argument("--lock-object-z-to-table", action="store_true")
    parser.add_argument("--min-object-center-z-margin", type=float, default=0.001, help="Clamp the mapped object upward so its lowest point stays at least this far above the default tabletop clearance.")
    parser.add_argument("--foundationpose-print-mapping-diagnostics", dest="foundationpose_print_mapping_diagnostics", action="store_true", default=True, help="Print raw FoundationPose camera-frame poses together with both direct/inverse camera-extrinsic world mappings for debugging. Enabled by default.")
    parser.add_argument("--no-foundationpose-print-mapping-diagnostics", dest="foundationpose_print_mapping_diagnostics", action="store_false", help="Disable FoundationPose mapping diagnostics for the target object.")
    parser.add_argument("--foundationpose-print-obstacle-mapping-diagnostics", action="store_true", help="Also print mapping diagnostics for scene obstacles. Disabled by default to reduce log spam.")
    parser.add_argument("--snap-low-profile-objects-flat-on-table", dest="snap_low_profile_objects_flat_on_table", action="store_true", default=True, help="If a mapped object is low-profile and already close to the tabletop, snap it to a flat table-contact pose instead of leaving it slightly floating/tilted. Enabled by default.")
    parser.add_argument("--no-snap-low-profile-objects-flat-on-table", dest="snap_low_profile_objects_flat_on_table", action="store_false", help="Disable the low-profile tabletop flat-snap heuristic.")
    parser.add_argument("--low-profile-flat-max-thickness", type=float, default=0.06, help="Only objects whose thinnest asset dimension is at most this value are eligible for the tabletop flat-snap heuristic.")
    parser.add_argument("--low-profile-flat-bottom-z-tolerance", type=float, default=0.03, help="Only objects whose estimated lowest point is within this distance of the table plane are eligible for the tabletop flat-snap heuristic.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--urdf-path", default=None)
    parser.add_argument("--srdf-path", default=None)
    parser.add_argument(
        "--variant",
        default="array_wxyz",
        choices=["array_wxyz", "array_xyzw", "pymp_pose_wxyz", "pymp_pose_xyzw"],
    )
    parser.add_argument("--pregrasp-height", type=float, default=0.07)
    parser.add_argument(
        "--pregrasp-blocked-retry-retreat-distance",
        type=float,
        default=0.03,
        help="If pregrasp execution is blocked by a nearby obstacle before any real motion starts, first retreat the TCP away from the nearest obstacle by this many meters and then replan pregrasp once. Set <=0 to disable.",
    )
    parser.add_argument(
        "--pregrasp-blocked-retry-lift-height",
        type=float,
        default=0.0,
        help="Optional upward TCP lift to combine with the pregrasp blocked-retry retreat motion. Set <=0 to keep the escape purely horizontal.",
    )
    parser.add_argument("--grasp-z-offset", type=float, default=0.0)
    parser.add_argument("--topdown-grasp-max-insertion-depth", type=float, default=0.05, help="For top-down grasp modes, keep the TCP no deeper than this distance below the object's true current top face in world z. Set <=0 to disable this clamp.")
    parser.add_argument("--topdown-tilt-toward-robot-deg", type=float, nargs="*", default=[12.0, 20.0], help="When top-down pregrasp/grasp IK fails, also try tilting the TCP toward the robot base by these angles in degrees. Set empty or <=0 values to disable.")
    parser.add_argument("--topdown-tilt-toward-robot-shift-m", type=float, nargs="*", default=[0.0, 0.02], help="For tilted top-down fallback poses, also try shifting the TCP toward the robot in the table plane by these distances. This helps tall/far objects where pure in-place tilt is still hard to reach.")
    parser.add_argument("--topdown-grasp-yaw-variant-deg", type=float, nargs="*", default=[0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0, 180.0], help="For top-down grasp modes, also try these extra world-z yaw variants around the same grasp point. This is useful when a slightly different xy heading makes the wrist reachable.")
    parser.add_argument("--min-grasp-tcp-z", type=float, default=0.015, help="Minimum allowed world-frame TCP z for the grasp pose. If the computed grasp pose is lower, grasp and pregrasp are shifted upward together for safety.")
    parser.add_argument("-7-grasp-fallback-z-lifts", type=float, nargs="*", default=[0.012, 0.02, 0.028], help="Extra world-z lifts to try when direct grasp IK fails.")
    parser.add_argument("--grasp-fallback-approach-backoffs", type=float, nargs="*", default=[0.008, 0.015], help="Extra backoff distances along the TCP approach axis to try when direct grasp IK fails.")
    parser.add_argument("--goal-z-offset", type=float, default=0.0)
    parser.add_argument("--yaw-offset-deg", type=float, default=180.0)
    parser.add_argument("--render-mode", type=str, default="human")
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--auto-execute", action="store_true")
    parser.add_argument("--single-confirm-per-object", action="store_true", help="Ask once per object, then execute the remaining pick-place stages for that object without additional Enter confirmations.")
    parser.add_argument("--preview-trajectory-before-confirm", dest="preview_trajectory_before_confirm", action="store_true", default=True, help="Before each step-by-step confirmation, render the planned trajectory once in simulation and then restore the starting state. Enabled by default.")
    parser.add_argument("--no-preview-trajectory-before-confirm", dest="preview_trajectory_before_confirm", action="store_false", help="Disable trajectory playback before each step-by-step confirmation.")
    parser.add_argument("--trajectory-preview-max-frames", type=int, default=80, help="Maximum number of simulation frames to use when previewing a planned trajectory before confirmation.")
    parser.add_argument("--trajectory-preview-sleep", type=float, default=0.02, help="Delay between frames when previewing a planned trajectory before confirmation.")

    parser.add_argument("--execute-real", action="store_true", help="Actually connect to RM75 and execute the planned motions on hardware.")
    parser.add_argument("--robot-ip", type=str, default=None, help="Override the RM75 controller IP from lerobot_sim2real.config.real_robot.")
    parser.add_argument("--real-control-hz", type=float, default=10.0)
    parser.add_argument("--real-max-delta-per-step", type=float, default=0.03)
    parser.add_argument("--real-hold-steps", type=int, default=2)
    parser.add_argument(
        "--sim-start-pre-lift-height",
        type=float,
        default=0.0,
        help="Optional upward TCP lift used by sim_start_pre_escape after direct sim_start planning fails. Set <=0 to disable.",
    )
    parser.add_argument(
        "--sim-start-pre-retreat-distance",
        type=float,
        default=0.03,
        help="Before planning the return/alignment to sim_start, retreat the TCP in the horizontal plane away from the nearest obstacle by this many meters after direct sim_start planning fails. Set <=0 to disable.",
    )
    parser.add_argument(
        "--sim-start-direct-attempt-count-before-pre-escape",
        type=int,
        default=2,
        help="How many direct sim_start RRT attempts to try before falling back to sim_start_pre_escape. After pre-escape, the full RRT attempt budget is used.",
    )
    parser.add_argument(
        "--real-stream-waypoint-path",
        dest="real_stream_waypoint_path",
        action="store_true",
        default=True,
        help="Execute planned joint waypoint paths as one continuous streamed command sequence instead of pausing between waypoints. Enabled by default.",
    )
    parser.add_argument(
        "--no-real-stream-waypoint-path",
        dest="real_stream_waypoint_path",
        action="store_false",
        help="Disable continuous waypoint-path streaming and fall back to per-waypoint execution.",
    )
    parser.add_argument("--joint7-max-turns", type=float, default=1.0, help="Reject planned paths whose joint7 travels more than this many full turns in total or excursion from the start configuration. Set <=0 to disable.")
    parser.add_argument(
        "--rrt-attempt-count",
        type=int,
        default=6,
        help="How many increasingly aggressive RRT parameter settings to try for each pose/joint planning request before giving up. When multiple attempts succeed, the planner keeps the lowest-cost successful path.",
    )
    parser.add_argument(
        "--rrt-keep-best-success-candidate",
        dest="rrt_keep_best_success_candidate",
        action="store_true",
        default=True,
        help="When multiple RRT attempts succeed, keep planning long enough to compare them and execute the lowest-cost successful joint path instead of the first one that succeeds. Enabled by default.",
    )
    parser.add_argument(
        "--no-rrt-keep-best-success-candidate",
        dest="rrt_keep_best_success_candidate",
        action="store_false",
        help="Stop at the first successful RRT attempt instead of comparing multiple successful candidates.",
    )
    parser.add_argument("--real-gripper-open", type=float, default=0.0, help="Real gripper absolute command in [0, 0.91]; 0 means fully open.")
    parser.add_argument("--real-gripper-close", type=float, default=0.91, help="Real gripper absolute command in [0, 0.91]; 0.91 means fully closed.")
    parser.add_argument(
        "--real-gripper-blocked-margin",
        type=float,
        default=0.05,
        help="After commanding close, treat the measured real gripper as blocked by an object when gripper.pos stays at least this much below the commanded close value.",
    )
    parser.add_argument("--freeze-active-object-before-grasp", dest="freeze_active_object_before_grasp", action="store_true", default=True, help="Keep the simulated active object fixed at its initialized pose until the real gripper has closed. Enabled by default.")
    parser.add_argument("--no-freeze-active-object-before-grasp", dest="freeze_active_object_before_grasp", action="store_false", help="Allow the simulated active object to move under physics before the real gripper closes.")
    parser.add_argument("--no-move-real-to-sim-start", action="store_true")
    parser.add_argument(
        "--align-real-to-sim-start-before-cycle",
        action="store_true",
        help="Before each cycle, plan a collision-aware return to the simulation start joint state. Disabled by default so continuous runs plan directly from the current real pose.",
    )
    parser.add_argument("--skip-goal-motion", action="store_true")
    parser.add_argument(
        "--reset-real-before-start",
        dest="reset_real_before_start",
        action="store_true",
        default=True,
        help="Move the real RM75 to the hardware reset pose before aligning it to the current simulation start configuration. Enabled by default.",
    )
    parser.add_argument(
        "--no-reset-real-before-start",
        dest="reset_real_before_start",
        action="store_false",
        help="Skip the initial hardware reset before aligning the real robot to the simulation start configuration.",
    )
    parser.add_argument("--use-sim-goal-motion", action="store_true", help="Use the environment-sampled simulated goal after grasp instead of the fixed post-grasp joint target.")
    parser.add_argument("--fixed-goal-joints-deg", type=float, nargs=7, default=[178.0, -5.0, 0.0, -70.0, 0.0, -102.0, 60.0], metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"))
    parser.add_argument("--fixed-goal-planning-time", type=float, default=4.0, help="RRT planning time budget in seconds for the post-grasp transport to the fixed joint goal.")
    parser.add_argument("--fixed-goal-rrt-range", type=float, default=0.15, help="RRT range for the post-grasp transport to the fixed joint goal.")
    parser.add_argument("--transport-attached-box-scale", type=float, default=1.0, help="Scale factor applied to the attached object collision box during post-grasp transport planning.")
    parser.add_argument("--scene-obstacle-box-scale", type=float, default=0.9, help="Scale factor applied to the scene-obstacle collision boxes used in sim/planner. Values below 1.0 reduce false-positive avoidance from overly coarse bounding boxes.")
    parser.add_argument(
        "--render-scene-obstacle-planner-boxes",
        action="store_true",
        help="Render translucent box overlays for the actual planner collision boxes of scene obstacles. This is also enabled automatically by --curobo-debug.",
    )
    parser.add_argument(
        "--render-robot-collision-spheres",
        action="store_true",
        help="Render the RM75 collision spheres used by cuRobo. This is also enabled automatically by --curobo-debug.",
    )
    parser.add_argument(
        "--planner-virtual-side-wall",
        dest="planner_virtual_side_wall",
        action="store_true",
        default=True,
        help="Add a planner-only virtual wall on the negative-x side of the workspace to discourage large backward RRT loops. Enabled by default.",
    )
    parser.add_argument(
        "--no-planner-virtual-side-wall",
        dest="planner_virtual_side_wall",
        action="store_false",
        help="Disable the planner-only virtual side wall.",
    )
    parser.add_argument(
        "--planner-virtual-side-wall-x",
        type=float,
        default=-0.50,
        help="World-frame x position of the front face of the planner-only virtual side wall.",
    )
    parser.add_argument(
        "--planner-virtual-side-wall-thickness",
        type=float,
        default=0.03,
        help="Thickness of the planner-only virtual side wall in meters.",
    )
    parser.add_argument(
        "--planner-virtual-side-wall-width-y",
        type=float,
        default=1.20,
        help="Width of the planner-only virtual side wall along world y in meters.",
    )
    parser.add_argument(
        "--planner-virtual-side-wall-height",
        type=float,
        default=0.80,
        help="Height of the planner-only virtual side wall in meters.",
    )
    parser.add_argument(
        "--planner-virtual-side-wall-y",
        type=float,
        default=0.0,
        help="World-frame y center of the planner-only virtual side wall.",
    )
    parser.add_argument(
        "--planner-virtual-side-wall-bottom-z",
        type=float,
        default=0.0,
        help="World-frame bottom z of the planner-only virtual side wall.",
    )
    parser.add_argument(
        "--planner-virtual-top-wall",
        dest="planner_virtual_top_wall",
        action="store_true",
        default=True,
        help="Add a planner-only top wall to discourage large upward RRT loops. Enabled by default.",
    )
    parser.add_argument(
        "--no-planner-virtual-top-wall",
        dest="planner_virtual_top_wall",
        action="store_false",
        help="Disable the planner-only top wall.",
    )
    parser.add_argument(
        "--planner-virtual-top-wall-z",
        type=float,
        default=0.8,
        help="World-frame z position of the bottom face of the planner-only top wall.",
    )
    parser.add_argument(
        "--planner-virtual-top-wall-thickness",
        type=float,
        default=0.03,
        help="Thickness of the planner-only top wall in meters.",
    )
    parser.add_argument(
        "--planner-virtual-top-wall-width-x",
        type=float,
        default=1.20,
        help="Width of the planner-only top wall along world x in meters.",
    )
    parser.add_argument(
        "--planner-virtual-top-wall-width-y",
        type=float,
        default=1.20,
        help="Width of the planner-only top wall along world y in meters.",
    )
    parser.add_argument(
        "--planner-virtual-top-wall-x",
        type=float,
        default=0.0,
        help="World-frame x center of the planner-only top wall.",
    )
    parser.add_argument(
        "--planner-virtual-top-wall-y",
        type=float,
        default=0.0,
        help="World-frame y center of the planner-only top wall.",
    )
    parser.add_argument("--pre-transport-lift-height", type=float, default=0.03, help="Before transporting to the fixed joint goal, first lift the grasped object by this many meters in world z. Set to 0 to disable.")
    parser.add_argument("--post-grasp-retreat-distance", type=float, default=0.02, help="After grasping and before the fixed-goal transport, optionally retreat the TCP in the horizontal plane away from the nearest registered obstacle by this many meters.")
    parser.add_argument("--transport-lift-height", type=float, default=0.08, help="If direct post-grasp transport to the fixed goal fails, first plan a lift of this many meters in world z before retrying the transport.")
    parser.add_argument(
        "--teleport-attached-object-during-transport",
        dest="teleport_attached_object_during_transport",
        action="store_true",
        default=True,
        help="During real-execution transport shadowing, visually keep the active simulated target attached to the TCP so the scene matches the carried object. Enabled by default.",
    )
    parser.add_argument(
        "--no-teleport-attached-object-during-transport",
        dest="teleport_attached_object_during_transport",
        action="store_false",
        help="Do not visually attach the simulated target to the TCP during real-execution transport shadowing.",
    )
    parser.add_argument("--fixed-goal-no-attach", action="store_true", help="Disable attached-object collision checking during post-grasp transport planning to the fixed goal.")
    parser.add_argument("--fixed-goal-rrt-probe-no-attach", action="store_true", help="When --debug is enabled, run diagnostic fixed-goal RRT probes on the same start/goal joint pair with use_attach=False and without linear fallback.")
    parser.add_argument("--disable-linear-joint-fallback", action="store_true", help="Disable the collision-checked linear joint-path fallback after RRT transport planning fails.")
    parser.add_argument("--repeat-count", type=int, default=1, help="Number of pick cycles to run. Each completed cycle recreates the vision+simulation state and, when executing on hardware, resets the robot before the next cycle.")
    parser.add_argument("--repeat-forever", action="store_true", help="Repeat pick cycles until a planning failure or user cancellation occurs.")
    parser.add_argument(
        "--reuse-foundationpose-scene-across-cycles",
        dest="reuse_foundationpose_scene_across_cycles",
        action="store_true",
        default=True,
        help="Cache the first FoundationPose scene capture and reuse it across later cycles when the target object and selected obstacles stay the same. Enabled by default.",
    )
    parser.add_argument(
        "--no-reuse-foundationpose-scene-across-cycles",
        dest="reuse_foundationpose_scene_across_cycles",
        action="store_false",
        help="Force a fresh FoundationPose capture at the start of every cycle.",
    )
    parser.add_argument(
        "--foundationpose-refine-after-render",
        dest="foundationpose_refine_after_render",
        action="store_true",
        default=False,
        help="After the initialized scene is rendered, allow interactively re-registering and live-tracking individual objects before planning.",
    )
    parser.add_argument(
        "--no-foundationpose-refine-after-render",
        dest="foundationpose_refine_after_render",
        action="store_false",
        help="Skip the post-render interactive FoundationPose refine prompt.",
    )
    parser.add_argument(
        "--foundationpose-refine-lock-window",
        type=int,
        default=4,
        help="When locking a manually refined FoundationPose track, average this many recent tracked poses.",
    )
    return parser


def parse_args():
    return build_arg_parser().parse_args()


class RealmanJointExecutor:
    def __init__(self, args):
        lerobot_root = Path(args.lerobot_root).expanduser().resolve()
        sim2real_root = Path(args.lerobot_sim2real_root).expanduser().resolve()
        for path in [str(sim2real_root), str(lerobot_root)]:
            if path not in sys.path:
                sys.path.insert(0, path)

        from lerobot_sim2real.config.real_robot import create_real_robot

        self.real_robot, self.camera = create_real_robot(auto_connect=False)
        if args.robot_ip:
            self.real_robot.config.ip = args.robot_ip
        self.real_robot.connect()
        self._ensure_command_socket()
        self.use_degrees = bool(getattr(self.real_robot.config, "use_degrees", False))
        self.arm_keys, self.has_gripper = self._infer_action_keys()
        print(f"Connected real robot at {self.real_robot.config.ip}. Arm keys: {self.arm_keys}")

    def _ensure_command_socket(self):
        if getattr(self.real_robot, "client", None) is not None:
            return
        host = str(self.real_robot.config.ip)
        port = int(getattr(self.real_robot.config, "port", 8080))
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(3.0)
        client.connect((host, port))
        client.settimeout(None)
        self.real_robot.client = client
        print(f"Established RealMan command socket at {host}:{port}")

    def _infer_action_keys(self):
        obs = self.real_robot.get_observation()
        ordered = [k for k in obs.keys() if k.endswith(".pos")]
        arm_keys = [k for k in ordered if k != "gripper.pos"]
        return arm_keys, ("gripper.pos" in ordered)

    def get_arm_qpos(self) -> np.ndarray:
        obs = self.real_robot.get_observation()
        q = np.asarray([float(obs[k]) for k in self.arm_keys], dtype=np.float32)
        if self.use_degrees:
            q = np.deg2rad(q)
        return q

    def get_gripper_pos(self) -> float | None:
        if not self.has_gripper:
            return None
        try:
            obs = self.real_robot.get_observation()
            return float(obs["gripper.pos"])
        except Exception:
            return None

    def send_action(self, arm_q: np.ndarray | None = None, gripper_pos: float | None = None):
        action = {}
        if arm_q is not None:
            arm_q = np.asarray(arm_q, dtype=np.float32).reshape(-1)
            arm_send = np.rad2deg(arm_q) if self.use_degrees else arm_q
            for key, value in zip(self.arm_keys, arm_send):
                action[key] = float(value)
        if gripper_pos is not None and self.has_gripper:
            action["gripper.pos"] = float(np.clip(gripper_pos, 0.0, 0.91))
        if not action:
            return {}
        return self.real_robot.send_action(action)

    def reset_robot(self, gripper_pos: float | None = None):
        arm = getattr(self.real_robot, "arm", None)
        if arm is None:
            raise RuntimeError("real robot arm handle is not available")
        arm.rm_set_hand_follow_pos([0], False)
        arm.rm_movej([90, 0, 0, -90, 0, -90, 60], 100, 0, 0, 1)
        try:
            arm.rm_set_arm_max_line_speed(0.0001)
        except Exception:
            pass
        try:
            arm.rm_set_hand_speed(1000)
        except Exception:
            pass
        self._ensure_command_socket()
        if gripper_pos is not None and self.has_gripper:
            self.set_gripper(gripper_pos, repeats=2, hz=5.0)
        print("[real reset] moved RM75 to the hardware reset pose")

    def set_gripper(self, gripper_pos: float, repeats: int = 3, hz: float = 5.0):
        period = 1.0 / max(hz, 1e-3)
        for _ in range(max(repeats, 1)):
            self.send_action(gripper_pos=gripper_pos)
            time.sleep(period)

    def move_linear(
        self,
        q_target: np.ndarray,
        gripper_pos: float | None,
        max_delta_per_step: float,
        hz: float,
        hold_steps: int,
        label: str,
        shadow_callback=None,
    ):
        q_start = self.get_arm_qpos()
        q_target = np.asarray(q_target, dtype=np.float32).reshape(-1)[: len(self.arm_keys)]
        q_delta = (q_target - q_start + np.pi) % (2 * np.pi) - np.pi
        q_target_short = q_start + q_delta
        max_delta = float(np.max(np.abs(q_delta))) if q_delta.size > 0 else 0.0
        num_steps = max(1, int(np.ceil(max_delta / max(max_delta_per_step, 1e-6))))
        period = 1.0 / max(hz, 1e-3)
        print(f"[real {label}] execute_linear num_steps={num_steps}")
        # Skip alpha=0 so each streamed segment actually advances toward the
        # target instead of repeatedly resending the current joint state.
        for alpha in np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float32)[1:]:
            q_cmd = (1.0 - alpha) * q_start + alpha * q_target_short
            self.send_action(arm_q=q_cmd, gripper_pos=gripper_pos)
            if callable(shadow_callback):
                try:
                    shadow_callback(np.asarray(q_cmd, dtype=np.float32))
                except Exception:
                    pass
            time.sleep(period)
        for _ in range(max(hold_steps, 0)):
            self.send_action(arm_q=q_target_short, gripper_pos=gripper_pos)
            if callable(shadow_callback):
                try:
                    shadow_callback(np.asarray(q_target_short, dtype=np.float32))
                except Exception:
                    pass
            time.sleep(period)
        return q_target_short.astype(np.float32)

    def _build_streamed_waypoint_commands(self, q_start: np.ndarray, q_path, max_delta_per_step: float):
        q_prev = np.asarray(q_start, dtype=np.float32).reshape(-1)[: len(self.arm_keys)]
        streamed_cmds = []
        for q_target in list(q_path or []):
            q_target = np.asarray(q_target, dtype=np.float32).reshape(-1)[: len(self.arm_keys)]
            q_delta = (q_target - q_prev + np.pi) % (2 * np.pi) - np.pi
            q_target_short = q_prev + q_delta
            max_delta = float(np.max(np.abs(q_delta))) if q_delta.size > 0 else 0.0
            num_steps = max(1, int(np.ceil(max_delta / max(max_delta_per_step, 1e-6))))
            for alpha in np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float32)[1:]:
                q_cmd = (1.0 - alpha) * q_prev + alpha * q_target_short
                streamed_cmds.append(np.asarray(q_cmd, dtype=np.float32))
            q_prev = q_target_short.astype(np.float32)
        return streamed_cmds, q_prev.astype(np.float32)

    def move_waypoint_path(
        self,
        q_path,
        gripper_pos: float | None,
        max_delta_per_step: float,
        hz: float,
        hold_steps: int,
        label: str,
        shadow_callback=None,
    ):
        q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[: len(self.arm_keys)] for q in q_path]
        if not q_path:
            return self.get_arm_qpos().astype(np.float32)
        q_start = self.get_arm_qpos().astype(np.float32)
        q_cmds, q_sent = self._build_streamed_waypoint_commands(q_start, q_path, max_delta_per_step)
        period = 1.0 / max(hz, 1e-3)
        print(
            f"[real {label}] execute_stream "
            f"path_waypoints={len(q_path)} num_steps={len(q_cmds)}"
        )
        for q_cmd in q_cmds:
            self.send_action(arm_q=q_cmd, gripper_pos=gripper_pos)
            if callable(shadow_callback):
                try:
                    shadow_callback(np.asarray(q_cmd, dtype=np.float32))
                except Exception:
                    pass
            time.sleep(period)
        for _ in range(max(hold_steps, 0)):
            self.send_action(arm_q=q_sent, gripper_pos=gripper_pos)
            if callable(shadow_callback):
                try:
                    shadow_callback(np.asarray(q_sent, dtype=np.float32))
                except Exception:
                    pass
            time.sleep(period)
        return np.asarray(q_sent, dtype=np.float32)

    def close(self):
        try:
            if self.camera is not None and getattr(self.camera, "is_connected", False):
                self.camera.disconnect()
        except Exception:
            pass
        try:
            self.real_robot.disconnect()
        except Exception:
            pass


def move_actor_out_of_view(actor):
    if actor is None:
        return
    actor.set_pose(Pose.create_from_pq(p=np.asarray([0.0, 0.0, -5.0], dtype=np.float32), q=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)))


def hide_env_goal_visual(env):
    try:
        move_actor_out_of_view(getattr(env.unwrapped, "goal_jiaobang", None))
    except Exception:
        pass


def compute_world_attached_pose(demo):
    attach_pose_tcp = getattr(demo, "attached_box_pose_tcp", None)
    if attach_pose_tcp is None:
        return None
    tcp_pose = demo.tcp.pose
    T_world_tcp = pose_to_matrix(flatten_np(tcp_pose.p)[:3], flatten_np(tcp_pose.q)[:4])
    attach_pose_tcp = np.asarray(attach_pose_tcp, dtype=np.float32).reshape(7)
    T_tcp_box = pose_to_matrix(attach_pose_tcp[:3], attach_pose_tcp[3:7])
    return T_world_tcp @ T_tcp_box



def update_attached_payload_pose(demo, active: bool | None = None):
    obj = getattr(getattr(demo, "base_env", None), "obj", None)
    if obj is None or getattr(demo, "attached_box_pose_tcp", None) is None:
        return
    if not bool(getattr(getattr(demo, "args", None), "teleport_attached_object_during_transport", False)):
        return
    if active is None:
        active = bool(getattr(demo, "_attached_object_visual_active", False))
    demo._attached_object_visual_active = bool(active)
    if not active:
        return
    T_world_obj = compute_world_attached_pose(demo)
    if T_world_obj is None:
        return
    pose = Pose.create_from_pq(
        p=T_world_obj[:3, 3].astype(np.float32),
        q=bridge_mod_mat2quat(T_world_obj[:3, :3]).astype(np.float32),
    )
    try:
        obj.set_pose(pose)
        set_vel = getattr(obj, "set_velocity", None)
        if callable(set_vel):
            obj.set_velocity(np.zeros(3, dtype=np.float32))
        set_ang_vel = getattr(obj, "set_angular_velocity", None)
        if callable(set_ang_vel):
            obj.set_angular_velocity(np.zeros(3, dtype=np.float32))
    except Exception:
        pass



def update_attached_box_visual(demo, visible: bool | None = None):
    actor = getattr(demo, "attached_box_visual", None)
    attach_pose_tcp = getattr(demo, "attached_box_pose_tcp", None)
    if actor is None or attach_pose_tcp is None:
        update_attached_payload_pose(demo)
        update_attached_object_collision_sphere_visuals(demo, getattr(demo, "args", None), visible=False)
        return
    if visible is None:
        visible = bool(getattr(demo, "_attached_box_visual_visible", False))
    demo._attached_box_visual_visible = bool(visible)
    if not visible:
        move_actor_out_of_view(actor)
        update_attached_payload_pose(demo)
        update_attached_object_collision_sphere_visuals(demo, getattr(demo, "args", None), visible=False)
        return
    T_world_box = compute_world_attached_pose(demo)
    if T_world_box is None:
        return
    actor.set_pose(Pose.create_from_pq(p=T_world_box[:3, 3].astype(np.float32), q=bridge_mod_mat2quat(T_world_box[:3, :3]).astype(np.float32)))
    update_attached_payload_pose(demo)
    update_attached_object_collision_sphere_visuals(demo, getattr(demo, "args", None), visible=True)



def setup_attached_box_visual(demo, env, box_size: np.ndarray, attach_pose_tcp: np.ndarray):
    actor = getattr(demo, "attached_box_visual", None)
    if actor is None:
        demo.attached_box_visual = build_attached_box_visual_actor(env, box_size)
    demo.attached_box_size = np.asarray(box_size, dtype=np.float32).reshape(3)
    demo.attached_box_pose_tcp = np.asarray(attach_pose_tcp, dtype=np.float32).reshape(7)
    demo._attached_box_visual_visible = True
    demo._attached_object_visual_active = bool(getattr(getattr(demo, "args", None), "teleport_attached_object_during_transport", False))
    ensure_attached_object_collision_sphere_visuals(demo, getattr(demo, "args", None))
    update_attached_box_visual(demo, visible=True)



def collision_to_text(collision) -> str:
    link1 = getattr(collision, "link_name1", "?")
    obj1 = getattr(collision, "object_name1", "?")
    link2 = getattr(collision, "link_name2", "?")
    obj2 = getattr(collision, "object_name2", "?")
    return f"{link1} of entity {obj1} <-> {link2} of entity {obj2}"



def print_collision_list(label: str, collisions):
    collisions = list(collisions or [])
    if not collisions:
        print(f"[collision] {label}: none")
        return
    print(f"[collision] {label}: {len(collisions)}")
    for collision in collisions[:10]:
        print(f"[collision]   - {collision_to_text(collision)}")
    if len(collisions) > 10:
        print(f"[collision]   ... {len(collisions) - 10} more")



def compute_oriented_box_world_corners(center_p, quat_wxyz, box_size: np.ndarray) -> np.ndarray:
    from transforms3d.quaternions import quat2mat

    center_p = np.asarray(center_p, dtype=np.float32).reshape(3)
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    half = np.asarray(box_size, dtype=np.float32).reshape(3) * 0.5
    corners = np.asarray([
        [sx, sy, sz]
        for sx in (-half[0], half[0])
        for sy in (-half[1], half[1])
        for sz in (-half[2], half[2])
    ], dtype=np.float32)
    R = quat2mat(quat_wxyz).astype(np.float32)
    return corners @ R.T + center_p.reshape(1, 3)


def transform_local_points_to_world(center_p, quat_wxyz, local_points: np.ndarray) -> np.ndarray:
    from transforms3d.quaternions import quat2mat

    center_p = np.asarray(center_p, dtype=np.float32).reshape(3)
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    local_points = np.asarray(local_points, dtype=np.float32).reshape(-1, 3)
    R = quat2mat(quat_wxyz).astype(np.float32)
    return local_points @ R.T + center_p.reshape(1, 3)


def compute_oriented_box_bottom_z(center_p, quat_wxyz, box_size: np.ndarray) -> float:
    world = compute_oriented_box_world_corners(center_p, quat_wxyz, box_size)
    return float(world[:, 2].min())


def compute_oriented_box_top_z(center_p, quat_wxyz, box_size: np.ndarray) -> float:
    world = compute_oriented_box_world_corners(center_p, quat_wxyz, box_size)
    return float(world[:, 2].max())



def get_object_bottom_z(demo, args):
    try:
        corners = get_demo_object_world_bbox_corners(demo, args)
        if corners is None:
            return None
        return float(np.min(corners[:, 2]))
    except Exception:
        return None


def get_demo_object_world_bbox_corners(demo, args):
    try:
        obj_p, obj_q = demo.get_obj_pose()
        local_points = get_asset_local_points(args.sim_asset_file, args.sim_asset_scale)
        world_points = transform_local_points_to_world(obj_p, obj_q, local_points)
        lo = world_points.min(axis=0)
        hi = world_points.max(axis=0)
        return np.asarray(
            [[sx, sy, sz] for sx in (lo[0], hi[0]) for sy in (lo[1], hi[1]) for sz in (lo[2], hi[2])],
            dtype=np.float32,
        )
    except Exception:
        try:
            size = get_asset_box_size(args.sim_asset_file, args.sim_asset_scale)
            obj_p, obj_q = demo.get_obj_pose()
            return compute_oriented_box_world_corners(obj_p, obj_q, size)
        except Exception:
            return None


def get_demo_object_world_bbox_center(demo, args):
    corners = get_demo_object_world_bbox_corners(demo, args)
    if corners is None:
        return None
    return np.asarray(0.5 * (corners.min(axis=0) + corners.max(axis=0)), dtype=np.float32)


def get_demo_object_world_top_z(demo, args):
    corners = get_demo_object_world_bbox_corners(demo, args)
    if corners is None:
        return None
    return float(np.max(corners[:, 2]))


def clamp_box_pose_above_table(position, quat, box_size, *, min_clearance: float = 0.001):
    pos = np.asarray(position, dtype=np.float32).reshape(3).copy()
    quat = np.asarray(quat, dtype=np.float32).reshape(4).copy()
    size = np.asarray(box_size, dtype=np.float32).reshape(3)
    min_clearance = float(max(min_clearance, 0.0))
    bottom_z = compute_oriented_box_bottom_z(pos, quat, size)
    if bottom_z >= min_clearance:
        return pos, 0.0, float(bottom_z)
    delta_z = float(min_clearance - bottom_z)
    pos[2] += delta_z
    return pos, delta_z, float(bottom_z)


def clamp_asset_pose_above_table(position, quat, asset_file: str, asset_scale: float, *, min_clearance: float = 0.001):
    pos = np.asarray(position, dtype=np.float32).reshape(3).copy()
    quat = np.asarray(quat, dtype=np.float32).reshape(4).copy()
    min_clearance = float(max(min_clearance, 0.0))
    try:
        local_points = get_asset_local_points(asset_file, asset_scale)
    except Exception:
        size = get_asset_box_size(asset_file, asset_scale)
        return clamp_box_pose_above_table(pos, quat, size, min_clearance=min_clearance)
    world_points = transform_local_points_to_world(pos, quat, local_points)
    bottom_z = float(np.min(world_points[:, 2]))
    if bottom_z >= min_clearance:
        return pos, 0.0, bottom_z
    delta_z = float(min_clearance - bottom_z)
    pos[2] += delta_z
    return pos, delta_z, bottom_z


def snap_asset_pose_flat_on_table_if_needed(
    position,
    quat,
    asset_file: str,
    asset_scale: float,
    *,
    max_thickness: float = 0.06,
    bottom_z_tolerance: float = 0.03,
    min_clearance: float = 0.001,
):
    pos = np.asarray(position, dtype=np.float32).reshape(3).copy()
    quat = np.asarray(quat, dtype=np.float32).reshape(4).copy()
    max_thickness = float(max(max_thickness, 0.0))
    bottom_z_tolerance = float(max(bottom_z_tolerance, 0.0))
    min_clearance = float(max(min_clearance, 0.0))
    if max_thickness <= 0.0 or bottom_z_tolerance <= 0.0:
        return pos, quat, 0.0, 0.0, False, None

    try:
        size = np.asarray(get_asset_box_size(asset_file, asset_scale), dtype=np.float32).reshape(3)
        local_points = get_asset_local_points(asset_file, asset_scale)
    except Exception:
        return pos, quat, 0.0, 0.0, False, None
    min_extent = float(np.min(size))
    if min_extent > max_thickness:
        return pos, quat, 0.0, 0.0, False, min_extent

    world_points = transform_local_points_to_world(pos, quat, local_points)
    bottom_z_before = float(np.min(world_points[:, 2]))
    if abs(bottom_z_before) > bottom_z_tolerance:
        return pos, quat, 0.0, 0.0, False, min_extent

    R = quat2mat(quat).astype(np.float32)
    candidate_axes = np.where(size <= float(np.min(size)) * 1.15 + 1e-6)[0]
    if candidate_axes.size == 0:
        candidate_axes = np.asarray([int(np.argmin(size))], dtype=np.int32)
    vertical_scores = [abs(float(np.dot(R[:, int(idx)], np.array([0.0, 0.0, 1.0], dtype=np.float32)))) for idx in candidate_axes]
    support_axis = int(candidate_axes[int(np.argmax(vertical_scores))])
    support_vec = _normalize_vec(R[:, support_axis])
    if support_vec is None:
        return pos, quat, 0.0, 0.0, False, min_extent

    target_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if float(np.dot(support_vec, target_vec)) < 0.0:
        target_vec = -target_vec

    dot_val = float(np.clip(np.dot(support_vec, target_vec), -1.0, 1.0))
    angle_rad = 0.0
    R_new = R
    if dot_val < 0.999999:
        rot_axis = np.cross(support_vec, target_vec)
        axis_norm = float(np.linalg.norm(rot_axis))
        if axis_norm <= 1e-8:
            fallback_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(float(np.dot(fallback_axis, support_vec))) > 0.9:
                fallback_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            rot_axis = np.cross(support_vec, fallback_axis)
            axis_norm = float(np.linalg.norm(rot_axis))
            if axis_norm <= 1e-8:
                return pos, quat, 0.0, 0.0, False, min_extent
            rot_axis = rot_axis / axis_norm
            angle_rad = float(np.pi)
        else:
            rot_axis = rot_axis / axis_norm
            angle_rad = float(np.arctan2(axis_norm, dot_val))
        R_new = (_rotation_matrix_from_axis_angle(rot_axis, angle_rad) @ R).astype(np.float32)

    quat_new = bridge_mod_mat2quat(R_new).astype(np.float32)
    world_points_new = transform_local_points_to_world(pos, quat_new, local_points)
    bottom_z_after = float(np.min(world_points_new[:, 2]))
    delta_z = float(min_clearance - bottom_z_after)
    pos[2] += delta_z
    return pos, quat_new, delta_z, float(np.rad2deg(angle_rad)), True, min_extent


def lift_active_object_above_table_if_needed(demo, args, *, min_clearance: float = 0.001) -> float:
    obj_pose = demo.base_env.obj.pose
    obj_p = flatten_np(obj_pose.p)[:3].copy()
    obj_q = flatten_np(obj_pose.q)[:4].copy()
    obj_p, delta_z, object_bottom_z = clamp_asset_pose_above_table(
        obj_p,
        obj_q,
        args.sim_asset_file,
        args.sim_asset_scale,
        min_clearance=min_clearance,
    )
    if delta_z <= 0.0:
        return 0.0
    demo.base_env.obj.set_pose(Pose.create_from_pq(p=obj_p.astype(np.float32), q=obj_q.astype(np.float32)))
    try:
        zero_vel = np.zeros(3, dtype=np.float32)
        demo.base_env.obj.set_linear_velocity(zero_vel)
        demo.base_env.obj.set_angular_velocity(zero_vel)
    except Exception:
        pass
    try:
        demo.refresh_runtime_handles(rebuild_visual=False)
    except Exception:
        pass
    sync_planner_qpos_from_demo(demo)
    update_attached_box_visual(demo)
    print(
        f"[safety] raised active object by {delta_z:.4f} m so its lowest point stays above the table "
        f"(bottom z {object_bottom_z:.4f} -> {min_clearance:.4f})"
    )
    return delta_z


def snap_active_object_flat_on_table_if_needed(demo, args, *, min_clearance: float = 0.001) -> bool:
    if not bool(getattr(args, "snap_low_profile_objects_flat_on_table", True)):
        return False
    obj_pose = demo.base_env.obj.pose
    obj_p = flatten_np(obj_pose.p)[:3].copy()
    obj_q = flatten_np(obj_pose.q)[:4].copy()
    obj_p, obj_q, delta_z, angle_deg, snapped, min_extent = snap_asset_pose_flat_on_table_if_needed(
        obj_p,
        obj_q,
        args.sim_asset_file,
        args.sim_asset_scale,
        max_thickness=float(getattr(args, "low_profile_flat_max_thickness", 0.06)),
        bottom_z_tolerance=float(getattr(args, "low_profile_flat_bottom_z_tolerance", 0.03)),
        min_clearance=min_clearance,
    )
    if not snapped:
        return False
    demo.base_env.obj.set_pose(Pose.create_from_pq(p=obj_p.astype(np.float32), q=obj_q.astype(np.float32)))
    try:
        zero_vel = np.zeros(3, dtype=np.float32)
        demo.base_env.obj.set_linear_velocity(zero_vel)
        demo.base_env.obj.set_angular_velocity(zero_vel)
    except Exception:
        pass
    try:
        demo.refresh_runtime_handles(rebuild_visual=False)
    except Exception:
        pass
    sync_planner_qpos_from_demo(demo)
    update_attached_box_visual(demo)
    print(
        f"[safety] snapped active object flat to the table "
        f"(min_extent={float(min_extent):.4f} m, rotation={angle_deg:.1f} deg, dz={delta_z:+.4f} m)"
    )
    return True


def lift_scene_obstacles_above_table_if_needed(env, demo, *, min_clearance: float = 0.001) -> int:
    scene_obstacles = list(getattr(demo, "scene_obstacles", []) or [])
    if not scene_obstacles:
        return 0

    actors_by_name = {}
    for actor in list(getattr(env.unwrapped, "_scene_obstacle_actors", []) or []):
        try:
            actor_name = str(getattr(actor, "name", ""))
        except Exception:
            actor_name = ""
        if actor_name:
            actors_by_name[actor_name] = actor

    lifted_count = 0
    for item in scene_obstacles:
        actor_name = str(item.get("actor_name", "") or "")
        actor = actors_by_name.get(actor_name)
        asset_file = item.get("asset_file")
        asset_scale = item.get("asset_scale")
        box_size = item.get("visual_box_size")
        if actor is None:
            continue
        pose = actor.pose
        pos = flatten_np(pose.p)[:3].copy()
        quat = flatten_np(pose.q)[:4].copy()
        if asset_file is not None and asset_scale is not None:
            pos, delta_z, bottom_z = clamp_asset_pose_above_table(
                pos,
                quat,
                str(asset_file),
                float(asset_scale),
                min_clearance=min_clearance,
            )
        elif box_size is not None:
            pos, delta_z, bottom_z = clamp_box_pose_above_table(
                pos,
                quat,
                box_size,
                min_clearance=min_clearance,
            )
        else:
            continue
        if delta_z <= 0.0:
            continue
        actor.set_pose(Pose.create_from_pq(p=pos.astype(np.float32), q=quat.astype(np.float32)))
        sync_scene_obstacle_planner_box_actor(env, item, pos.astype(np.float32), quat.astype(np.float32))
        T_world_obj = np.asarray(item.get("T_world_obj"), dtype=np.float32).reshape(4, 4).copy()
        T_world_obj[:3, 3] = pos.astype(np.float32)
        item["T_world_obj"] = T_world_obj
        item["bottom_z_before_lift"] = float(bottom_z)
        item["table_lift_delta_z"] = float(delta_z)
        try:
            if bool(item.get("planner_collision", False)):
                from mplib import collision_detection as mplib_cd

                planner_box_size = np.asarray(
                    item.get("planner_box_size", item.get("visual_box_size")),
                    dtype=np.float32,
                ).reshape(3)
                collision_object = mplib_cd.fcl.CollisionObject(
                    mplib_cd.fcl.Box(planner_box_size.tolist()),
                    pos.tolist(),
                    quat.tolist(),
                )
                demo.planner.set_normal_object(actor_name, collision_object)
        except Exception as exc:
            print(f"[scene obstacle] {actor_name}: failed to refresh planner collision after table lift: {exc}")
        lifted_count += 1
        print(
            f"[safety] raised scene obstacle {item.get('object_name', actor_name)} by {delta_z:.4f} m "
            f"so its lowest point stays above the table (bottom z {bottom_z:.4f} -> {float(max(min_clearance, 0.0)):.4f})"
        )
    return lifted_count


def snap_scene_obstacles_flat_on_table_if_needed(env, demo, args, *, min_clearance: float = 0.001) -> int:
    if not bool(getattr(args, "snap_low_profile_objects_flat_on_table", True)):
        return 0
    scene_obstacles = list(getattr(demo, "scene_obstacles", []) or [])
    if not scene_obstacles:
        return 0

    actors_by_name = {}
    for actor in list(getattr(env.unwrapped, "_scene_obstacle_actors", []) or []):
        actor_name = str(getattr(actor, "name", "") or "")
        if actor_name:
            actors_by_name[actor_name] = actor

    snapped_count = 0
    for item in scene_obstacles:
        actor_name = str(item.get("actor_name", "") or "")
        actor = actors_by_name.get(actor_name)
        asset_file = item.get("asset_file")
        asset_scale = item.get("asset_scale")
        if actor is None or asset_file is None or asset_scale is None:
            continue
        pose = actor.pose
        pos = flatten_np(pose.p)[:3].copy()
        quat = flatten_np(pose.q)[:4].copy()
        pos, quat, delta_z, angle_deg, snapped, min_extent = snap_asset_pose_flat_on_table_if_needed(
            pos,
            quat,
            str(asset_file),
            float(asset_scale),
            max_thickness=float(getattr(args, "low_profile_flat_max_thickness", 0.06)),
            bottom_z_tolerance=float(getattr(args, "low_profile_flat_bottom_z_tolerance", 0.03)),
            min_clearance=min_clearance,
        )
        if not snapped:
            continue
        actor.set_pose(Pose.create_from_pq(p=pos.astype(np.float32), q=quat.astype(np.float32)))
        sync_scene_obstacle_planner_box_actor(env, item, pos.astype(np.float32), quat.astype(np.float32))
        T_world_obj = np.asarray(item.get("T_world_obj"), dtype=np.float32).reshape(4, 4).copy()
        T_world_obj[:3, :3] = quat2mat(quat).astype(np.float32)
        T_world_obj[:3, 3] = pos.astype(np.float32)
        item["T_world_obj"] = T_world_obj
        try:
            if bool(item.get("planner_collision", False)):
                from mplib import collision_detection as mplib_cd

                planner_box_size = np.asarray(
                    item.get("planner_box_size", item.get("visual_box_size")),
                    dtype=np.float32,
                ).reshape(3)
                collision_object = mplib_cd.fcl.CollisionObject(
                    mplib_cd.fcl.Box(planner_box_size.tolist()),
                    pos.tolist(),
                    quat.tolist(),
                )
                demo.planner.set_normal_object(actor_name, collision_object)
        except Exception as exc:
            print(f"[scene obstacle] {actor_name}: failed to refresh planner collision after flat-snap: {exc}")
        snapped_count += 1
        print(
            f"[safety] snapped scene obstacle {item.get('object_name', actor_name)} flat to the table "
            f"(min_extent={float(min_extent):.4f} m, rotation={angle_deg:.1f} deg, dz={delta_z:+.4f} m)"
        )
    return snapped_count



def get_attached_box_bottom_z(demo):
    size = getattr(demo, "attached_box_size", None)
    attach_pose_tcp = getattr(demo, "attached_box_pose_tcp", None)
    if size is None or attach_pose_tcp is None:
        return None
    tcp_pose = demo.tcp.pose
    T_world_tcp = pose_to_matrix(flatten_np(tcp_pose.p)[:3], flatten_np(tcp_pose.q)[:4])
    attach_pose_tcp = np.asarray(attach_pose_tcp, dtype=np.float32).reshape(7)
    T_tcp_box = pose_to_matrix(attach_pose_tcp[:3], attach_pose_tcp[3:7])
    T_world_box = T_world_tcp @ T_tcp_box
    return compute_oriented_box_bottom_z(T_world_box[:3, 3], bridge_mod_mat2quat(T_world_box[:3, :3]), size)



def print_failure_diagnostics(demo, args, label: str, *, q_target=None, use_attach: bool = False):
    q_diag = None if q_target is None else np.asarray(q_target, dtype=np.float32).reshape(-1)[:7]
    if q_diag is not None:
        try:
            self_collisions = demo.planner.check_for_self_collision(qpos=q_diag)
        except Exception as exc:
            self_collisions = []
            print(f"[collision] failed to query self-collision diagnostics for {label}: {exc}")
        try:
            env_collisions = demo.planner.check_for_env_collision(qpos=q_diag, with_point_cloud=False, use_attach=use_attach)
        except Exception as exc:
            env_collisions = []
            print(f"[collision] failed to query env-collision diagnostics for {label}: {exc}")
        print_collision_list(f"{label} self", self_collisions)
        print_collision_list(f"{label} env", env_collisions)
        if not self_collisions and not env_collisions:
            print(f"[collision] {label}: target joint state itself is collision-free; this failure is more likely path-search timeout / narrow passage than direct target collision.")
    else:
        print(f"[collision] {label}: no target joint state available, so only box/table clearance diagnostics are shown.")

    object_bottom_z = get_object_bottom_z(demo, args)
    if object_bottom_z is not None:
        print(f"[collision] active object bottom z: {object_bottom_z:.4f} m (table plane approx z=0.0000)")
        if object_bottom_z < 0:
            print(f"[collision] active object appears to penetrate the table by {-object_bottom_z:.4f} m")

    if use_attach:
        attached_bottom_z = get_attached_box_bottom_z(demo)
        if attached_bottom_z is not None:
            print(f"[collision] attached box bottom z: {attached_bottom_z:.4f} m (table plane approx z=0.0000)")
            if attached_bottom_z < 0:
                print(f"[collision] attached box appears to penetrate the table by {-attached_bottom_z:.4f} m")



def sync_planner_qpos_from_demo(demo):
    try:
        sim_q = np.asarray(flatten_np(demo.robot.get_qpos()), dtype=np.float32).reshape(-1)
        planner_q = np.asarray(flatten_np(demo.planner.robot.get_qpos()), dtype=np.float32).reshape(-1)
        if sim_q.shape == planner_q.shape:
            demo.planner.robot.set_qpos(sim_q, True)
    except Exception:
        pass


def set_pregrasp_object_freeze(demo, active: bool) -> None:
    demo._freeze_active_object_before_grasp = bool(active)
    if active and getattr(demo, "_frozen_active_object_pose", None) is None:
        try:
            obj_pose = demo.base_env.obj.pose
            demo._frozen_active_object_pose = (
                flatten_np(obj_pose.p)[:3].astype(np.float32).copy(),
                flatten_np(obj_pose.q)[:4].astype(np.float32).copy(),
            )
        except Exception:
            demo._frozen_active_object_pose = None


def refresh_frozen_active_object_pose(demo) -> None:
    if not bool(getattr(demo, "_freeze_active_object_before_grasp", False)):
        return
    frozen_pose = getattr(demo, "_frozen_active_object_pose", None)
    if frozen_pose is None:
        return
    obj = getattr(getattr(demo, "base_env", None), "obj", None)
    if obj is None:
        return
    obj_p, obj_q = frozen_pose
    try:
        obj.set_pose(Pose.create_from_pq(p=np.asarray(obj_p, dtype=np.float32), q=np.asarray(obj_q, dtype=np.float32)))
        zero_vel = np.zeros(3, dtype=np.float32)
        for method_name in ("set_linear_velocity", "set_velocity"):
            method = getattr(obj, method_name, None)
            if callable(method):
                method(zero_vel)
        ang_method = getattr(obj, "set_angular_velocity", None)
        if callable(ang_method):
            ang_method(zero_vel)
    except Exception:
        return


def sync_demo_arm_qpos(demo, arm_q: np.ndarray):
    full_q = flatten_np(demo.robot.get_qpos())
    full_q = np.asarray(full_q, dtype=np.float32).copy()
    arm_q = np.asarray(arm_q, dtype=np.float32).reshape(-1)
    full_q[demo.arm_indices] = arm_q[: len(demo.arm_indices)]
    demo.robot.set_qpos(full_q)
    set_qvel = getattr(demo.robot, "set_qvel", None)
    if callable(set_qvel):
        try:
            demo.robot.set_qvel(np.zeros_like(full_q, dtype=np.float32))
        except Exception:
            pass
    refresh_frozen_active_object_pose(demo)
    sync_planner_qpos_from_demo(demo)
    update_attached_box_visual(demo)
    refresh_runtime_handles = getattr(demo, "refresh_runtime_handles", None)
    if callable(refresh_runtime_handles):
        try:
            demo.refresh_runtime_handles(rebuild_visual=False)
        except Exception:
            pass
    update_robot_collision_sphere_visuals(demo, getattr(demo, "args", None))


def get_sim_gripper_pad_gap(demo):
    try:
        p1 = flatten_np(demo.base_env.agent.finger1_pad.pose.p)[:3]
        p2 = flatten_np(demo.base_env.agent.finger2_pad.pose.p)[:3]
        return float(np.linalg.norm(np.asarray(p1, dtype=np.float32) - np.asarray(p2, dtype=np.float32)))
    except Exception:
        return None


def real_gripper_blocked_after_close(real_exec: RealmanJointExecutor | None, close_cmd: float, blocked_margin: float) -> tuple[float | None, bool | None]:
    if real_exec is None:
        return None, None
    pos = real_exec.get_gripper_pos()
    if pos is None:
        return None, None
    blocked = bool(pos <= float(close_cmd) - float(blocked_margin))
    return pos, blocked



def sync_demo_gripper_state(demo, closed: bool, steps: int = 3):
    sim_gripper_value = 1.0 if closed else -1.0
    min_steps = 20
    total_steps = max(int(steps), min_steps)
    try:
        demo.hold_current_and_set_gripper(sim_gripper_value, steps=total_steps)
        refresh_frozen_active_object_pose(demo)
        demo.refresh_runtime_handles(rebuild_visual=False)
        sync_planner_qpos_from_demo(demo)
        update_attached_box_visual(demo)
        gap = get_sim_gripper_pad_gap(demo)
        if gap is not None:
            print(f"[sim gripper] {'closed' if closed else 'opened'} preview after {total_steps} steps, pad gap={gap:.4f} m")
    except Exception as exc:
        print(f"[warn] failed to sync simulated gripper state: {exc}")


def get_current_active_object_world_pose_matrix(demo) -> np.ndarray | None:
    try:
        obj_pose = demo.base_env.obj.pose
        return pose_to_matrix(flatten_np(obj_pose.p)[:3], flatten_np(obj_pose.q)[:4]).astype(np.float32)
    except Exception:
        return None


def remember_current_active_object_world_pose_for_scene_cache(demo) -> np.ndarray | None:
    T_world_obj = get_current_active_object_world_pose_matrix(demo)
    if T_world_obj is None:
        return None
    demo._scene_cache_placed_object_world_pose = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4).copy()
    return demo._scene_cache_placed_object_world_pose


def settle_released_active_object_for_scene_cache(demo, args) -> np.ndarray | None:
    settle_steps = 60
    demo._freeze_active_object_before_grasp = False
    demo._frozen_active_object_pose = None
    if settle_steps > 0:
        gripper_open_value = float(getattr(args, "gripper_open", -1.0))
        try:
            demo.hold_current_and_set_gripper(gripper_open_value, steps=settle_steps)
            print(f"[place settle] simulated {settle_steps} step(s) after release to let the object settle")
        except Exception as exc:
            print(f"[warn] failed to settle released object in simulation: {exc}")
    try:
        demo.refresh_runtime_handles(rebuild_visual=False)
    except Exception:
        pass
    sync_planner_qpos_from_demo(demo)
    update_attached_box_visual(demo, visible=False)
    T_world_obj = remember_current_active_object_world_pose_for_scene_cache(demo)
    if T_world_obj is not None:
        print(
            "[place settle] remembered released object pose at world translation="
            f"{np.round(T_world_obj[:3, 3], 6).tolist()}"
        )
    return T_world_obj


def confirm_simple_action(label: str, args, bridge_mod=None, env=None, repeats: int = 8) -> bool:
    if bool(getattr(args, "_skip_remaining_step_confirms_in_object", False)):
        print(f"[confirm] auto-approved {label} because single-confirm-per-object is active")
        return True
    if bridge_mod is not None and env is not None and getattr(args, "render_mode", None) == "human":
        bridge_mod.render_preview(env, repeats=repeats)
    if args.auto_execute:
        return True
    if bridge_mod is not None:
        answer = bridge_mod.prompt_with_live_render(
            f"[confirm] {label}. Press Enter to execute, or type q then Enter to abort: ",
            env,
            args,
        )
    else:
        try:
            answer = input(f"[confirm] {label}. Press Enter to execute, or type q then Enter to abort: ").strip().lower()
        except EOFError:
            answer = "q"
    return answer not in {"q", "quit", "n", "no"}


def begin_single_confirm_window_for_object(demo, bridge_mod, args) -> bool:
    if args.auto_execute or not bool(getattr(args, "single_confirm_per_object", False)):
        return True
    if bool(getattr(args, "_skip_remaining_step_confirms_in_object", False)):
        return True
    if args.render_mode == "human":
        bridge_mod.render_preview(demo.env, repeats=6)
    answer = bridge_mod.prompt_with_live_render(
        "[confirm] execute the full pick-place sequence for this object with no further step-by-step confirmations? Press Enter to continue, or type q then Enter to abort: ",
        demo.env,
        args,
    )
    if answer in {"q", "quit", "n", "no"}:
        return False
    args._skip_remaining_step_confirms_in_object = True
    return True


def _subsample_preview_path(q_path, max_frames: int) -> list[np.ndarray]:
    q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in list(q_path or [])]
    if not q_path:
        return []
    max_frames = max(int(max_frames), 1)
    if len(q_path) <= max_frames:
        return q_path
    indices = np.linspace(0, len(q_path) - 1, max_frames)
    indices = np.unique(np.round(indices).astype(int))
    return [q_path[int(idx)] for idx in indices]


def _build_linear_preview_path(q_start: np.ndarray, q_goal: np.ndarray, *, max_frames: int) -> list[np.ndarray]:
    q_start = np.asarray(q_start, dtype=np.float32).reshape(-1)[:7]
    q_goal = np.asarray(q_goal, dtype=np.float32).reshape(-1)[:7]
    q_delta = q_goal - q_start
    nominal_frames = max(2, int(np.ceil(np.max(np.abs(q_delta)) / 0.06)) + 1)
    num_frames = min(max(nominal_frames, 2), max(int(max_frames), 2))
    return [
        ((1.0 - alpha) * q_start + alpha * q_goal).astype(np.float32)
        for alpha in np.linspace(0.0, 1.0, num_frames)
    ]


def _collect_robot_render_material_states(demo):
    try:
        links = list(demo.robot.get_links())
    except Exception:
        return []
    states = []
    seen = set()
    for link in links:
        try:
            render_body = link.find_component_by_type(sapien.render.RenderBodyComponent)
        except Exception:
            render_body = None
        if render_body is None:
            continue
        for render_shape in getattr(render_body, "render_shapes", []) or []:
            for part in getattr(render_shape, "parts", []) or []:
                material = getattr(part, "material", None)
                if material is None:
                    continue
                key = id(material)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    base_color = np.asarray(material.get_base_color(), dtype=np.float32).reshape(-1)[:4].copy()
                except Exception:
                    try:
                        base_color = np.asarray(material.base_color, dtype=np.float32).reshape(-1)[:4].copy()
                    except Exception:
                        base_color = None
                states.append((material, base_color))
    return states


def _set_robot_materials_base_color(material_states, rgba) -> None:
    rgba = np.asarray(rgba, dtype=np.float32).reshape(-1)[:4]
    for material, _ in material_states:
        try:
            material.set_base_color(rgba.tolist())
        except Exception:
            pass


def _restore_robot_render_material_states(material_states) -> None:
    for material, base_color in material_states:
        if base_color is None:
            continue
        try:
            material.set_base_color(np.asarray(base_color, dtype=np.float32).reshape(-1)[:4].tolist())
        except Exception:
            pass


def render_planned_trajectory_preview(demo, bridge_mod, label: str, q_preview_path, args) -> None:
    if bool(getattr(args, "auto_execute", False)):
        return
    if not bool(getattr(args, "preview_trajectory_before_confirm", True)):
        return
    if getattr(args, "render_mode", None) != "human":
        return
    q_preview_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in list(q_preview_path or [])]
    if not q_preview_path:
        return
    q_start = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    preview_path = _subsample_preview_path(q_preview_path, int(getattr(args, "trajectory_preview_max_frames", 80)))
    if not preview_path:
        return
    print(
        f"[preview] rendering planned {label} trajectory in sim "
        f"({len(preview_path)}/{len(q_preview_path)} frames)"
    )
    sleep_s = float(max(getattr(args, "trajectory_preview_sleep", 0.02), 0.0))
    material_states = _collect_robot_render_material_states(demo)
    _set_robot_materials_base_color(material_states, [0.25, 0.95, 0.25, 1.0])
    try:
        for q in preview_path:
            sync_demo_arm_qpos(demo, q)
            bridge_mod.render_preview(demo.env, repeats=1, sleep_s=sleep_s)
        bridge_mod.render_preview(demo.env, repeats=2, sleep_s=sleep_s)
    finally:
        _restore_robot_render_material_states(material_states)
        sync_demo_arm_qpos(demo, q_start)
        bridge_mod.render_preview(demo.env, repeats=2, sleep_s=sleep_s)


def confirm_planned_motion_or_skip(demo, bridge_mod, label: str, target_pose, q_target, args, *, q_preview_path=None) -> bool:
    if bool(getattr(args, "_skip_remaining_step_confirms_in_object", False)):
        demo.preview_target_pose(target_pose)
        if args.render_mode == "human":
            bridge_mod.render_preview(demo.env, repeats=3)
        print(f"\n[planned {label}]")
        print("target p:", np.round(flatten_np(target_pose.p)[:3], 6))
        print("target q:", np.round(flatten_np(target_pose.q)[:4], 6))
        print("planned arm q:", np.round(flatten_np(q_target)[:7], 6))
        print(f"[confirm] auto-approved {label} because single-confirm-per-object is active")
        return True
    demo.preview_target_pose(target_pose)
    if q_preview_path is None:
        q_current = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
        q_preview_path = _build_linear_preview_path(
            q_current,
            np.asarray(q_target, dtype=np.float32).reshape(-1)[:7],
            max_frames=int(getattr(args, "trajectory_preview_max_frames", 80)),
        )
    render_planned_trajectory_preview(demo, bridge_mod, label, q_preview_path, args)
    bridge_mod.render_preview(demo.env, repeats=3)

    print(f"\n[planned {label}]")
    print("target p:", np.round(flatten_np(target_pose.p)[:3], 6))
    print("target q:", np.round(flatten_np(target_pose.q)[:4], 6))
    print("planned arm q:", np.round(flatten_np(q_target)[:7], 6))

    if args.auto_execute:
        return True

    if args.render_mode != "human" and not getattr(args, "_warned_nonhuman_render", False):
        print("[confirm] render_mode is not human, so you may not see a live viewer. Use --render-mode human if you want to inspect visually before pressing Enter.")
        args._warned_nonhuman_render = True

    while True:
        answer = bridge_mod.prompt_with_live_render(
            f"[confirm] {label} planned. Press Enter to execute, type r then Enter to replay the preview, or type q then Enter to abort: ",
            demo.env,
            args,
        )
        if answer == "r":
            render_planned_trajectory_preview(demo, bridge_mod, label, q_preview_path, args)
            bridge_mod.render_preview(demo.env, repeats=3)
            continue
        return answer not in {"q", "quit", "n", "no"}


def make_pose_with_position(pose, p_xyz):
    q_wxyz = flatten_np(pose.q)[:4].copy()
    p_xyz = np.asarray(p_xyz, dtype=np.float32).reshape(3)
    # Always build a plain SAPIEN pose here so confirmation/logging never
    # inherits a backend-specific pose object that keeps tensors on CUDA.
    return Pose.create_from_pq(p=p_xyz, q=q_wxyz)



def pose_to_matrix(p_xyz, q_wxyz) -> np.ndarray:
    from transforms3d.quaternions import quat2mat

    p_xyz = np.asarray(p_xyz, dtype=np.float32).reshape(3)
    q_wxyz = np.asarray(q_wxyz, dtype=np.float32).reshape(4)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quat2mat(q_wxyz).astype(np.float32)
    T[:3, 3] = p_xyz
    return T


def make_attached_box_pose(demo, asset_box_size: np.ndarray) -> np.ndarray:
    obj_p, obj_q = demo.get_obj_pose()
    tcp_pose = demo.tcp.pose
    tcp_p = flatten_np(tcp_pose.p)[:3]
    tcp_q = flatten_np(tcp_pose.q)[:4]
    T_world_obj = pose_to_matrix(obj_p, obj_q)
    T_world_tcp = pose_to_matrix(tcp_p, tcp_q)
    T_tcp_obj = np.linalg.inv(T_world_tcp) @ T_world_obj
    attach_pose = np.concatenate([T_tcp_obj[:3, 3], bridge_mod_mat2quat(T_tcp_obj[:3, :3])]).astype(np.float32)
    return attach_pose


def bridge_mod_mat2quat(R: np.ndarray) -> np.ndarray:
    from transforms3d.quaternions import mat2quat

    return np.asarray(mat2quat(np.asarray(R, dtype=np.float32)), dtype=np.float32)


def retarget_goal_q_to_nearest_equivalent(demo, q_goal: np.ndarray, q_start: np.ndarray) -> np.ndarray:
    q_goal = np.asarray(q_goal, dtype=np.float32).reshape(-1).copy()
    q_start = np.asarray(q_start, dtype=np.float32).reshape(-1)
    joint_types = list(getattr(demo.planner, "joint_types", []))
    joint_limits = np.asarray(getattr(demo.planner, "joint_limits", []), dtype=np.float32)
    n = min(len(q_goal), len(q_start), len(joint_types), len(joint_limits))
    for i in range(n):
        joint_type = str(joint_types[i])
        if not joint_type.startswith("JointModelR"):
            continue
        q_ref = float(q_start[i])
        q_raw = float(q_goal[i])
        q_min = float(joint_limits[i][0])
        q_max = float(joint_limits[i][1])
        candidates = []
        for k in range(-3, 4):
            q_candidate = q_raw + 2.0 * np.pi * float(k)
            if q_min - 1e-3 <= q_candidate <= q_max + 1e-3:
                candidates.append(q_candidate)
        if candidates:
            q_goal[i] = min(candidates, key=lambda q: abs(q - q_ref))
    return q_goal.astype(np.float32)


def _get_joint7_max_turns(demo) -> float:
    demo_args = getattr(demo, "args", None)
    if demo_args is None:
        return 1.0
    return float(max(getattr(demo_args, "joint7_max_turns", 1.0), 0.0))


def _retarget_joint7_with_turn_budget(demo, q_raw: np.ndarray, q_prev: np.ndarray, q_start: np.ndarray) -> np.ndarray | None:
    q = np.asarray(q_raw, dtype=np.float32).reshape(-1)[:7].copy()
    q_prev = np.asarray(q_prev, dtype=np.float32).reshape(-1)[:7]
    q_start = np.asarray(q_start, dtype=np.float32).reshape(-1)[:7]
    max_turns = _get_joint7_max_turns(demo)
    if max_turns <= 0.0 or q.shape[0] < 7 or q_prev.shape[0] < 7 or q_start.shape[0] < 7:
        return q

    joint_types = list(getattr(demo.planner, "joint_types", []))
    joint_limits = np.asarray(getattr(demo.planner, "joint_limits", []), dtype=np.float32)
    joint_idx = 6
    if joint_idx >= len(joint_types) or joint_idx >= len(joint_limits):
        return q
    if not str(joint_types[joint_idx]).startswith("JointModelR"):
        return q

    q_prev_j = float(q_prev[joint_idx])
    q_start_j = float(q_start[joint_idx])
    q_raw_j = float(q[joint_idx])
    q_min = float(joint_limits[joint_idx][0])
    q_max = float(joint_limits[joint_idx][1])
    max_abs_delta = float(max_turns * 2.0 * np.pi)
    tol = 1e-3

    candidates = []
    for k in range(-4, 5):
        q_candidate = q_raw_j + 2.0 * np.pi * float(k)
        if q_min - tol <= q_candidate <= q_max + tol and abs(q_candidate - q_start_j) <= max_abs_delta + tol:
            candidates.append(q_candidate)
    if not candidates:
        return None
    q[joint_idx] = min(candidates, key=lambda v: abs(v - q_prev_j))
    return q


def validate_joint7_turn_budget(demo, q_start: np.ndarray, q_path, *, label: str) -> bool:
    max_turns = _get_joint7_max_turns(demo)
    if max_turns <= 0.0:
        return True
    q_start = np.asarray(q_start, dtype=np.float32).reshape(-1)[:7]
    if q_start.shape[0] < 7:
        return True

    limit = float(max_turns * 2.0 * np.pi)
    q_prev = float(q_start[6])
    q_ref = float(q_start[6])
    total_abs_motion = 0.0
    max_excursion = 0.0
    for q in list(q_path or []):
        q = np.asarray(q, dtype=np.float32).reshape(-1)[:7]
        if q.shape[0] < 7:
            continue
        q_j = float(q[6])
        total_abs_motion += abs(q_j - q_prev)
        max_excursion = max(max_excursion, abs(q_j - q_ref))
        q_prev = q_j
    tol = 1e-3
    if total_abs_motion <= limit + tol and max_excursion <= limit + tol:
        return True
    print(
        f"[planner] {label} rejected by joint7 turn budget: "
        f"total_motion={total_abs_motion:.3f} rad, max_excursion={max_excursion:.3f} rad, "
        f"limit={limit:.3f} rad ({max_turns:.2f} turns)"
    )
    return False


def retarget_joint_path_to_nearest_equivalents(demo, q_start: np.ndarray, q_path) -> list[np.ndarray] | None:
    q_prev = np.asarray(q_start, dtype=np.float32).reshape(-1)[:7]
    q_ref = q_prev.copy()
    normalized_path = []
    for q in list(q_path or []):
        q = np.asarray(q, dtype=np.float32).reshape(-1)[:7]
        q = retarget_goal_q_to_nearest_equivalent(demo, q, q_prev)
        q = _retarget_joint7_with_turn_budget(demo, q, q_prev, q_ref)
        if q is None:
            return None
        normalized_path.append(q)
        q_prev = q
    return normalized_path


def _joint_path_quality_metrics(q_start: np.ndarray, q_path) -> dict:
    q_start = np.asarray(q_start, dtype=np.float32).reshape(-1)[:7]
    points = [q_start] + [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in list(q_path or [])]
    total_motion = 0.0
    joint7_total_motion = 0.0
    joint7_max_excursion = 0.0
    max_step_norm = 0.0
    for q_prev, q_curr in zip(points[:-1], points[1:]):
        dq = q_curr - q_prev
        step_norm = float(np.linalg.norm(dq))
        total_motion += step_norm
        max_step_norm = max(max_step_norm, step_norm)
        if dq.shape[0] >= 7:
            joint7_total_motion += abs(float(dq[6]))
            joint7_max_excursion = max(joint7_max_excursion, abs(float(q_curr[6] - q_start[6])))
    waypoint_count = max(0, len(points) - 1)
    return {
        "total_motion": float(total_motion),
        "joint7_total_motion": float(joint7_total_motion),
        "joint7_max_excursion": float(joint7_max_excursion),
        "max_step_norm": float(max_step_norm),
        "waypoint_count": int(waypoint_count),
    }


def _joint_path_quality_score(metrics: dict) -> float:
    return float(
        metrics["total_motion"]
        + 0.35 * metrics["joint7_total_motion"]
        + 0.15 * metrics["joint7_max_excursion"]
        + 0.03 * metrics["max_step_norm"]
        + 0.01 * metrics["waypoint_count"]
    )


def _select_best_rrt_success_candidate(label: str, q_start: np.ndarray, candidates):
    if not candidates:
        return None
    if len(candidates) == 1:
        candidate = dict(candidates[0])
        candidate["metrics"] = _joint_path_quality_metrics(q_start, candidate["path"])
        candidate["score"] = _joint_path_quality_score(candidate["metrics"])
        return candidate

    scored_candidates = []
    for candidate in candidates:
        candidate = dict(candidate)
        metrics = _joint_path_quality_metrics(q_start, candidate["path"])
        score = _joint_path_quality_score(metrics)
        candidate["metrics"] = metrics
        candidate["score"] = score
        scored_candidates.append(candidate)
        print(
            f"[planner] {label} success candidate {candidate['attempt_label']}: "
            f"score={score:.3f}, total_motion={metrics['total_motion']:.3f} rad, "
            f"joint7_total={metrics['joint7_total_motion']:.3f} rad, "
            f"joint7_excursion={metrics['joint7_max_excursion']:.3f} rad, "
            f"waypoints={metrics['waypoint_count']}"
        )

    best = min(
        scored_candidates,
        key=lambda item: (
            float(item["score"]),
            float(item["metrics"]["total_motion"]),
            float(item["metrics"]["joint7_total_motion"]),
            int(item["metrics"]["waypoint_count"]),
        ),
    )
    print(
        f"[planner] {label} selected best RRT candidate {best['attempt_label']}: "
        f"score={best['score']:.3f}, total_motion={best['metrics']['total_motion']:.3f} rad, "
        f"joint7_total={best['metrics']['joint7_total_motion']:.3f} rad, "
        f"joint7_excursion={best['metrics']['joint7_max_excursion']:.3f} rad, "
        f"waypoints={best['metrics']['waypoint_count']}"
    )
    return best


def _build_rrt_attempt_configs(planning_time: float, rrt_range: float, attempt_count: int):
    planning_time = float(max(planning_time, 0.5))
    rrt_range = float(max(rrt_range, 0.05))
    attempt_count = max(int(attempt_count), 1)
    names = [
        "primary",
        "retry_longer",
        "retry_wider",
        "retry_longer_wider",
        "retry_xlong_xwide",
        "retry_max",
    ]
    configs = []
    for idx in range(attempt_count):
        if idx == 0:
            time_scale, time_add = 1.0, 0.0
            range_scale, range_floor = 1.0, 0.05
        else:
            severity = float(idx)
            time_scale = 1.0 + 0.75 * severity
            time_add = 2.0 * severity
            range_scale = 1.0 + 0.35 * severity
            range_floor = 0.2 + 0.05 * max(0.0, severity - 1.0)
        attempt_name = names[idx] if idx < len(names) else f"retry_{idx+1}"
        attempt_time = float(max(planning_time * time_scale, planning_time + time_add))
        attempt_range = float(max(rrt_range * range_scale, range_floor))
        configs.append((attempt_name, attempt_time, attempt_range))
    return configs


def build_collision_checked_linear_joint_path(demo, q_start: np.ndarray, q_goal: np.ndarray, *, use_attach: bool = False, label: str = "joint_goal", max_delta: float = 0.08, allow_start_in_collision: bool = False):
    q_start = np.asarray(q_start, dtype=np.float32).reshape(-1)[:7]
    q_goal = np.asarray(q_goal, dtype=np.float32).reshape(-1)[:7]
    q_delta = q_goal - q_start
    max_delta = float(max(max_delta, 1e-3))
    num_steps = max(2, int(np.ceil(np.max(np.abs(q_delta)) / max_delta)))
    print(f"[planner] {label} trying collision-checked linear joint fallback with {num_steps} waypoints")

    start_self = demo.planner.check_for_self_collision(qpos=q_start)
    start_env = demo.planner.check_for_env_collision(qpos=q_start, with_point_cloud=False, use_attach=use_attach)
    if start_self or start_env:
        if not allow_start_in_collision:
            print(f"[planner] {label} linear fallback aborted: current state is already in collision")
            print_collision_list(f"{label} linear current self", start_self)
            print_collision_list(f"{label} linear current env", start_env)
            return None
        print(f"[planner] {label} linear fallback tolerating a colliding current state to escape locally")
        print_collision_list(f"{label} linear current self", start_self)
        print_collision_list(f"{label} linear current env", start_env)

    q_path = []
    for idx, alpha in enumerate(np.linspace(0.0, 1.0, num_steps + 1)[1:], start=1):
        q = ((1.0 - alpha) * q_start + alpha * q_goal).astype(np.float32)
        self_collisions = demo.planner.check_for_self_collision(qpos=q)
        env_collisions = demo.planner.check_for_env_collision(qpos=q, with_point_cloud=False, use_attach=use_attach)
        if self_collisions or env_collisions:
            print(f"[planner] {label} linear fallback collides at waypoint {idx}/{num_steps}")
            print_collision_list(f"{label} linear self", self_collisions)
            print_collision_list(f"{label} linear env", env_collisions)
            return None
        q_path.append(q)

    print(f"[planner] {label} linear fallback succeeded with {len(q_path)} waypoints")
    return q_path


def _build_collision_checked_linear_joint_path_quiet(demo, q_start: np.ndarray, q_goal: np.ndarray, *, use_attach: bool = False, max_delta: float = 0.08, allow_start_in_collision: bool = False):
    q_start = np.asarray(q_start, dtype=np.float32).reshape(-1)[:7]
    q_goal = np.asarray(q_goal, dtype=np.float32).reshape(-1)[:7]
    q_delta = q_goal - q_start
    max_delta = float(max(max_delta, 1e-3))
    num_steps = max(2, int(np.ceil(np.max(np.abs(q_delta)) / max_delta)))

    start_self = demo.planner.check_for_self_collision(qpos=q_start)
    start_env = demo.planner.check_for_env_collision(qpos=q_start, with_point_cloud=False, use_attach=use_attach)
    if start_self or start_env:
        if not allow_start_in_collision:
            return None

    q_path = []
    for alpha in np.linspace(0.0, 1.0, num_steps + 1)[1:]:
        q = ((1.0 - alpha) * q_start + alpha * q_goal).astype(np.float32)
        self_collisions = demo.planner.check_for_self_collision(qpos=q)
        env_collisions = demo.planner.check_for_env_collision(qpos=q, with_point_cloud=False, use_attach=use_attach)
        if self_collisions or env_collisions:
            return None
        q_path.append(q)
    return q_path


def _dense_collision_validate_max_delta(use_attach: bool) -> float:
    return 0.01 if use_attach else 0.03


def _validate_joint_path_segments_quiet(
    demo,
    q_start: np.ndarray,
    q_path,
    *,
    use_attach: bool = False,
    max_delta: float = 0.03,
    allow_start_in_collision: bool = False,
):
    q_prev = np.asarray(q_start, dtype=np.float32).reshape(-1)[:7]
    for idx, q_target in enumerate(q_path):
        q_target = np.asarray(q_target, dtype=np.float32).reshape(-1)[:7]
        q_target = retarget_goal_q_to_nearest_equivalent(demo, q_target, q_prev)
        seg = _build_collision_checked_linear_joint_path_quiet(
            demo,
            q_prev,
            q_target,
            use_attach=use_attach,
            max_delta=max_delta,
            allow_start_in_collision=bool(allow_start_in_collision and idx == 0),
        )
        if seg is None:
            return False, idx, q_prev, q_target
        q_prev = q_target
    return True, None, None, None


def validate_joint_path_segments(
    demo,
    q_start: np.ndarray,
    q_path,
    *,
    use_attach: bool = False,
    label: str = "joint_path",
    max_delta: float = 0.03,
    allow_start_in_collision: bool = False,
):
    ok, bad_idx, bad_start, bad_goal = _validate_joint_path_segments_quiet(
        demo,
        q_start,
        q_path,
        use_attach=use_attach,
        max_delta=max_delta,
        allow_start_in_collision=allow_start_in_collision,
    )
    if ok:
        return True
    print(f"[planner] {label} rejected by dense segment collision check at waypoint {bad_idx}")
    build_collision_checked_linear_joint_path(
        demo,
        bad_start,
        bad_goal,
        use_attach=use_attach,
        label=f"{label}_dense_precheck",
        max_delta=max_delta,
        allow_start_in_collision=allow_start_in_collision,
    )
    return False


def shortcut_joint_path(demo, q_start: np.ndarray, q_path, *, use_attach: bool = False, label: str = "joint_goal", max_delta: float = 0.08):
    q_start = np.asarray(q_start, dtype=np.float32).reshape(-1)[:7]
    q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in q_path]
    if len(q_path) <= 2:
        return q_path

    points = [q_start] + q_path
    simplified = [points[0]]
    i = 0
    while i < len(points) - 1:
        chosen_j = i + 1
        chosen_segment = None
        for j in range(len(points) - 1, i, -1):
            seg = _build_collision_checked_linear_joint_path_quiet(
                demo,
                points[i],
                points[j],
                use_attach=use_attach,
                max_delta=max_delta,
            )
            if seg is not None:
                chosen_j = j
                chosen_segment = seg
                break
        if chosen_segment is None:
            chosen_segment = [points[chosen_j]]
        simplified.extend(chosen_segment)
        i = chosen_j

    simplified_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in simplified[1:]]
    if len(simplified_path) < len(q_path):
        print(f"[planner] {label} shortcut reduced waypoints: {len(q_path)} -> {len(simplified_path)}")
        return simplified_path
    print(f"[planner] {label} shortcut kept original waypoint count: {len(q_path)}")
    return q_path



def plan_joint_path(
    demo,
    q_goal: np.ndarray,
    *,
    use_attach: bool = False,
    label: str = "joint_goal",
    planning_time: float = 4.0,
    rrt_range: float = 0.15,
    start_q=None,
    allow_linear_fallback: bool = True,
    prefer_linear_first: bool = False,
    allow_reverse_rrt_fallback: bool = False,
    prefer_reverse_rrt: bool = False,
    allow_shortcut: bool = True,
    allow_start_in_collision: bool = False,
    attempt_count_override: int | None = None,
):
    q_goal = np.asarray(q_goal, dtype=np.float32).reshape(-1)[:7]
    q_start = np.asarray(demo.current_arm_qpos() if start_q is None else start_q, dtype=np.float32).reshape(-1)[:7]
    q_goal_adjusted = retarget_goal_q_to_nearest_equivalent(demo, q_goal, q_start)
    print(f"[planner] current arm q for {label}:", np.round(q_start, 6))
    print(f"[planner] target arm q for {label}:", np.round(q_goal_adjusted, 6))
    if not np.allclose(q_goal_adjusted, q_goal, atol=1e-6):
        print(f"[planner] target arm q raw for {label}:", np.round(q_goal, 6))
    q_goal = q_goal_adjusted
    linear_max_delta = 0.08 if use_attach else 0.12
    dense_validate_delta = _dense_collision_validate_max_delta(use_attach)
    keep_best_success_candidate = bool(getattr(getattr(demo, "args", None), "rrt_keep_best_success_candidate", True))
    if prefer_linear_first:
        if not allow_linear_fallback:
            print(f"[planner] {label} prefer_linear_first requested but linear joint fallback is disabled")
        else:
            linear_path = build_collision_checked_linear_joint_path(
                demo,
                q_start,
                q_goal,
                use_attach=use_attach,
                label=label,
                max_delta=linear_max_delta,
                allow_start_in_collision=allow_start_in_collision,
            )
            if linear_path is not None:
                return linear_path
            print(f"[planner] {label} linear-first attempt failed; falling back to RRT")
    attempt_configs = _build_rrt_attempt_configs(
        planning_time,
        rrt_range,
        int(
            attempt_count_override
            if attempt_count_override is not None
            else getattr(getattr(demo, "args", None), "rrt_attempt_count", 6)
        ),
    )

    def _run_rrt_attempts(reverse: bool = False):
        nonlocal q_start, q_goal
        last_status_local = None
        path_label = f"{label} reverse" if reverse else label
        reverse_last_status = None
        success_candidates = []
        for attempt_name, attempt_time, attempt_range in attempt_configs:
            if reverse:
                print(
                    f"[planner] {label} reverse attempt={attempt_name}, "
                    f"planning_time={attempt_time:.2f}, rrt_range={attempt_range:.3f}, "
                    f"use_attach={use_attach}"
                )
            else:
                print(
                    f"[planner] {label} attempt={attempt_name}, "
                    f"planning_time={attempt_time:.2f}, rrt_range={attempt_range:.3f}, "
                    f"use_attach={use_attach}"
                )
            result = demo.planner.plan_qpos_to_qpos(
                [q_start] if reverse else [q_goal],
                q_goal if reverse else q_start,
                time_step=demo.control_timestep,
                rrt_range=attempt_range,
                planning_time=attempt_time,
                use_point_cloud=False,
                use_attach=use_attach,
            )
            last_status_local = result.get("status", result)
            if reverse:
                reverse_last_status = last_status_local
                print(f"[planner] {label} reverse status:", reverse_last_status)
            else:
                print(f"[planner] {label} status:", last_status_local)
            if result.get("status", "") != "Success":
                continue

            q_path_raw = [flatten_np(q)[:7].astype(np.float32) for q in result.get("position", [])]
            if reverse:
                if not q_path_raw:
                    print(f"[planner] {label} reverse returned no waypoints on attempt={attempt_name}")
                    continue
                q_path = list(reversed(q_path_raw))
                q_path = retarget_joint_path_to_nearest_equivalents(demo, q_start, q_path)
                if q_path is None:
                    print(f"[planner] {label} reverse rejected: joint7 could not be retargeted within the configured turn budget")
                    continue
                if q_path and np.allclose(q_path[0], q_start, atol=1e-5):
                    q_path = q_path[1:]
                if not q_path or not np.allclose(q_path[-1], q_goal, atol=1e-5):
                    q_path.append(q_goal.astype(np.float32))
                print(f"[planner] {label} reverse fallback waypoints:", len(q_path))
                print(f"[planner] {label} reverse fallback last waypoint:", np.round(q_path[-1], 6))
            else:
                q_path = retarget_joint_path_to_nearest_equivalents(demo, q_start, q_path_raw)
                if q_path is None:
                    print(f"[planner] {label} rejected: joint7 could not be retargeted within the configured turn budget")
                    continue
                if not q_path:
                    if np.allclose(q_start, q_goal, atol=1e-5):
                        print(
                            f"[planner] {label} returned no waypoints on attempt={attempt_name}; "
                            "treating this as a trivial success because start==goal"
                        )
                        return [], last_status_local
                    print(f"[planner] {label} returned no waypoints on attempt={attempt_name}")
                    continue
                print(f"[planner] {label} waypoints:", len(q_path))
                print(f"[planner] {label} last waypoint:", np.round(q_path[-1], 6))

            if not validate_joint7_turn_budget(demo, q_start, q_path, label=path_label):
                continue
            if not validate_joint_path_segments(
                demo,
                q_start,
                q_path,
                use_attach=use_attach,
                label=path_label,
                max_delta=dense_validate_delta,
                allow_start_in_collision=allow_start_in_collision,
            ):
                continue
            if not allow_shortcut:
                final_path = q_path
                if reverse:
                    print(f"[planner] {label} reverse shortcut disabled; keeping original waypoint count: {len(q_path)}")
                else:
                    print(f"[planner] {label} shortcut disabled; keeping original waypoint count: {len(q_path)}")
            else:
                shortcut_path = shortcut_joint_path(
                    demo,
                    q_start,
                    q_path,
                    use_attach=use_attach,
                    label=path_label,
                    max_delta=linear_max_delta,
                )
                if shortcut_path is not q_path and not validate_joint_path_segments(
                    demo,
                    q_start,
                    shortcut_path,
                    use_attach=use_attach,
                    label=f"{path_label}_shortcut",
                    max_delta=dense_validate_delta,
                    allow_start_in_collision=allow_start_in_collision,
                ):
                    print(f"[planner] {path_label} shortcut path rejected by dense validation; keeping the original RRT path")
                    final_path = q_path
                else:
                    final_path = shortcut_path
            if not keep_best_success_candidate:
                return final_path, last_status_local
            success_candidates.append(
                {
                    "attempt_label": f"{'reverse_' if reverse else ''}{attempt_name}",
                    "path": final_path,
                }
            )
        if success_candidates:
            best_candidate = _select_best_rrt_success_candidate(path_label, q_start, success_candidates)
            return best_candidate["path"], last_status_local
        return None, reverse_last_status if reverse else last_status_local

    if prefer_reverse_rrt and allow_reverse_rrt_fallback and not np.allclose(q_start, q_goal, atol=1e-5):
        print(f"[planner] {label} trying reverse-RRT first")
        q_path, reverse_status = _run_rrt_attempts(reverse=True)
        if q_path is not None:
            return q_path
        print(f"[planner] {label} reverse-first attempt failed; falling back to forward RRT")
        if reverse_status is not None:
            print(f"[planner] {label} reverse-first last status={reverse_status}")

    q_path, last_status = _run_rrt_attempts(reverse=False)
    if q_path is not None:
        return q_path

    if allow_reverse_rrt_fallback and not prefer_reverse_rrt and not np.allclose(q_start, q_goal, atol=1e-5):
        print(f"[planner] {label} forward RRT failed; trying reverse-RRT fallback")
        q_path, reverse_last_status = _run_rrt_attempts(reverse=True)
        if q_path is not None:
            return q_path
        print(f"[planner] {label} reverse-RRT fallback failed; last status={reverse_last_status}")

    print(f"[planner] {label} failed after {len(attempt_configs)} attempts; last status={last_status}")
    if not allow_linear_fallback:
        print(f"[planner] {label} linear joint fallback disabled")
        return None
    linear_path = build_collision_checked_linear_joint_path(
        demo,
        q_start,
        q_goal,
        use_attach=use_attach,
        label=label,
        max_delta=linear_max_delta,
        allow_start_in_collision=allow_start_in_collision,
    )
    if linear_path is not None:
        return linear_path
    return None


def run_fixed_goal_rrt_probe_suite(
    demo,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    planning_time: float,
    rrt_range: float,
):
    q_start = np.asarray(q_start, dtype=np.float32).reshape(-1)[:7]
    q_goal = np.asarray(q_goal, dtype=np.float32).reshape(-1)[:7]

    def _print_probe_result(label: str, q_path) -> None:
        if q_path is None:
            print(f"[planner] {label} result: failed")
        else:
            print(f"[planner] {label} result: success ({len(q_path)} waypoints)")

    def _run_probe(label: str, q_probe_start: np.ndarray, q_probe_goal: np.ndarray):
        return plan_joint_path(
            demo,
            q_probe_goal,
            use_attach=False,
            label=label,
            planning_time=planning_time,
            rrt_range=rrt_range,
            start_q=q_probe_start,
            allow_linear_fallback=False,
        )

    print("[planner] running same-start/same-goal RRT probe with use_attach=False")
    same_path = _run_probe("same_start_same_goal_probe", q_start, q_start)
    _print_probe_result("same_start_same_goal_probe", same_path)

    print("[planner] running fixed-goal RRT probe with use_attach=False on the same start/goal pair")
    probe_path = _run_probe("fixed_goal_rrt_probe_no_attach", q_start, q_goal)
    _print_probe_result("fixed_goal_rrt_probe_no_attach", probe_path)

    if probe_path is not None:
        return

    linear_probe_path = build_collision_checked_linear_joint_path(
        demo,
        q_start,
        q_goal,
        use_attach=False,
        label="fixed_goal_linear_probe_no_attach",
        max_delta=0.12,
    )
    if linear_probe_path is None:
        print("[planner] segmented fixed-goal RRT probe skipped: no linear no-attach path available")
        return
    if len(linear_probe_path) < 2:
        print("[planner] segmented fixed-goal RRT probe skipped: linear no-attach path has no interior waypoint")
        return

    mid_idx = max(0, (len(linear_probe_path) - 1) // 2)
    q_mid = np.asarray(linear_probe_path[mid_idx], dtype=np.float32).reshape(-1)[:7]
    print(
        "[planner] running segmented fixed-goal RRT probes with use_attach=False "
        f"using linear-path waypoint {mid_idx + 1}/{len(linear_probe_path)} as q_mid"
    )
    print(f"[planner] q_mid for segmented probe: {np.round(q_mid, 6)}")

    seg_a_path = _run_probe("fixed_goal_rrt_probe_no_attach_seg_a", q_start, q_mid)
    _print_probe_result("fixed_goal_rrt_probe_no_attach_seg_a", seg_a_path)

    seg_b_path = _run_probe("fixed_goal_rrt_probe_no_attach_seg_b", q_mid, q_goal)
    _print_probe_result("fixed_goal_rrt_probe_no_attach_seg_b", seg_b_path)
    if seg_b_path is not None:
        return

    print("[planner] running reverse-direction RRT probe on the failing seg_b interval")
    seg_b_rev_path = _run_probe("fixed_goal_rrt_probe_no_attach_seg_b_reverse", q_goal, q_mid)
    _print_probe_result("fixed_goal_rrt_probe_no_attach_seg_b_reverse", seg_b_rev_path)

    linear_probe_path_seg_b = build_collision_checked_linear_joint_path(
        demo,
        q_mid,
        q_goal,
        use_attach=False,
        label="fixed_goal_linear_probe_no_attach_seg_b",
        max_delta=0.12,
    )
    if linear_probe_path_seg_b is None:
        print("[planner] second-level segmented probe on seg_b skipped: no linear no-attach path available")
        return
    if len(linear_probe_path_seg_b) < 2:
        print("[planner] second-level segmented probe on seg_b skipped: linear seg_b path has no interior waypoint")
        return

    mid_idx_b = max(0, (len(linear_probe_path_seg_b) - 1) // 2)
    q_mid_b = np.asarray(linear_probe_path_seg_b[mid_idx_b], dtype=np.float32).reshape(-1)[:7]
    print(
        "[planner] running second-level segmented RRT probes on failing seg_b with use_attach=False "
        f"using linear-path waypoint {mid_idx_b + 1}/{len(linear_probe_path_seg_b)} as q_mid_b"
    )
    print(f"[planner] q_mid_b for seg_b split: {np.round(q_mid_b, 6)}")

    seg_ba_path = _run_probe("fixed_goal_rrt_probe_no_attach_seg_ba", q_mid, q_mid_b)
    _print_probe_result("fixed_goal_rrt_probe_no_attach_seg_ba", seg_ba_path)

    seg_bb_path = _run_probe("fixed_goal_rrt_probe_no_attach_seg_bb", q_mid_b, q_goal)
    _print_probe_result("fixed_goal_rrt_probe_no_attach_seg_bb", seg_bb_path)


def plan_pose_path(demo, pose, *, variant_name: str = "array_wxyz", use_attach: bool = False, label: str = "pose_goal", planning_time: float = 4.0, rrt_range: float = 0.15, start_q=None, allow_start_in_collision: bool = False):
    q_start = np.asarray(demo.current_arm_qpos() if start_q is None else start_q, dtype=np.float32).reshape(-1)[:7]
    target = demo.make_target(pose, variant_name)
    print(f"[planner] current arm q for {label}:", np.round(q_start, 6))
    print(f"[planner] target pose p for {label}:", np.round(flatten_np(pose.p)[:3], 6))
    print(f"[planner] target pose q for {label}:", np.round(flatten_np(pose.q)[:4], 6))
    attempt_configs = _build_rrt_attempt_configs(
        planning_time,
        rrt_range,
        getattr(getattr(demo, "args", None), "rrt_attempt_count", 6),
    )

    last_status = None
    dense_validate_delta = _dense_collision_validate_max_delta(use_attach)
    keep_best_success_candidate = bool(getattr(getattr(demo, "args", None), "rrt_keep_best_success_candidate", True))
    success_candidates = []
    for attempt_name, attempt_time, attempt_range in attempt_configs:
        print(
            f"[planner] {label} attempt={attempt_name}, "
            f"planning_time={attempt_time:.2f}, rrt_range={attempt_range:.3f}, "
            f"use_attach={use_attach}"
        )
        result = demo.planner.plan_qpos_to_pose(
            target,
            q_start,
            time_step=demo.control_timestep,
            rrt_range=attempt_range,
            planning_time=attempt_time,
            use_point_cloud=False,
            use_attach=use_attach,
        )
        last_status = result.get("status", result)
        print(f"[planner] {label} status:", last_status)
        if result.get("status", "") != "Success":
            continue

        q_path = [flatten_np(q)[:7].astype(np.float32) for q in result.get("position", [])]
        if not q_path:
            print(f"[planner] {label} returned no waypoints on attempt={attempt_name}")
            continue
        q_path = retarget_joint_path_to_nearest_equivalents(demo, q_start, q_path)
        if q_path is None:
            print(f"[planner] {label} rejected: joint7 could not be retargeted within the configured turn budget")
            continue
        print(f"[planner] {label} waypoints:", len(q_path))
        print(f"[planner] {label} last waypoint:", np.round(q_path[-1], 6))
        if not validate_joint7_turn_budget(demo, q_start, q_path, label=label):
            continue
        if not validate_joint_path_segments(
            demo,
            q_start,
            q_path,
            use_attach=use_attach,
            label=label,
            max_delta=dense_validate_delta,
            allow_start_in_collision=allow_start_in_collision,
        ):
            continue
        if not keep_best_success_candidate:
            return q_path
        success_candidates.append(
            {
                "attempt_label": attempt_name,
                "path": q_path,
            }
        )

    if success_candidates:
        best_candidate = _select_best_rrt_success_candidate(label, q_start, success_candidates)
        return best_candidate["path"]

    print(f"[planner] {label} failed after {len(attempt_configs)} attempts; last status={last_status}")
    return None


def _pose_to_wxyz_array(pose) -> np.ndarray:
    p = flatten_np(pose.p)[:3]
    q = flatten_np(pose.q)[:4]
    return np.concatenate([p, q]).astype(np.float64)


def _normalize_quat_wxyz(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)[:4]
    norm = float(np.linalg.norm(q))
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def _slerp_quat_wxyz(q0, q1, alpha: float) -> np.ndarray:
    q0 = _normalize_quat_wxyz(q0)
    q1 = _normalize_quat_wxyz(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + float(alpha) * (q1 - q0)
        return _normalize_quat_wxyz(q).astype(np.float32)
    theta_0 = float(np.arccos(dot))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * float(alpha)
    sin_theta = float(np.sin(theta))
    s0 = float(np.sin(theta_0 - theta) / sin_theta_0)
    s1 = float(sin_theta / sin_theta_0)
    q = s0 * q0 + s1 * q1
    return _normalize_quat_wxyz(q).astype(np.float32)


def _pick_nearest_ik_solution(
    demo,
    ik_results,
    q_ref: np.ndarray,
    *,
    max_joint_delta: float | None = None,
    max_joint7_delta: float | None = None,
    max_norm_delta: float | None = None,
) -> np.ndarray | None:
    q_ref = np.asarray(q_ref, dtype=np.float32).reshape(-1)[:7]
    candidates = []
    for q in ik_results:
        q = np.asarray(q, dtype=np.float32).reshape(-1)[:7]
        q = retarget_goal_q_to_nearest_equivalent(demo, q, q_ref)
        q_delta = q - q_ref
        if max_joint_delta is not None and float(np.max(np.abs(q_delta))) > float(max_joint_delta):
            continue
        if max_joint7_delta is not None and abs(float(q_delta[6])) > float(max_joint7_delta):
            continue
        if max_norm_delta is not None and float(np.linalg.norm(q_delta)) > float(max_norm_delta):
            continue
        candidates.append(q)
    if not candidates:
        return None
    return min(candidates, key=lambda q: float(np.linalg.norm(q - q_ref)))


def plan_local_cartesian_lift_path(
    demo,
    pose,
    *,
    use_attach: bool = False,
    label: str = "lift",
    start_q=None,
    max_cartesian_step: float = 0.003,
    max_segment_joint_delta: float = 0.20,
    max_segment_joint7_delta: float = 0.25,
    max_segment_norm_delta: float = 0.35,
    allow_start_in_collision: bool = False,
):
    q_start = np.asarray(demo.current_arm_qpos() if start_q is None else start_q, dtype=np.float32).reshape(-1)[:7]
    start_pose = demo.tcp.pose
    p_start = flatten_np(start_pose.p)[:3].astype(np.float32)
    q_start_pose = flatten_np(start_pose.q)[:4].astype(np.float32)
    p_goal = flatten_np(pose.p)[:3].astype(np.float32)
    q_goal_pose = flatten_np(pose.q)[:4].astype(np.float32)

    delta = p_goal - p_start
    lift_distance = float(np.linalg.norm(delta))
    if lift_distance <= 1e-6:
        return []

    num_segments = max(1, int(np.ceil(lift_distance / max(max_cartesian_step, 1e-3))))
    q_cursor = q_start.copy()
    q_path: list[np.ndarray] = []

    print(
        f"[planner] {label} trying local cartesian escape with {num_segments} segments "
        f"(distance={lift_distance:.4f} m, use_attach={use_attach})"
    )

    for seg_idx, alpha in enumerate(np.linspace(0.0, 1.0, num_segments + 1)[1:], start=1):
        p_target = (1.0 - alpha) * p_start + alpha * p_goal
        q_target = _slerp_quat_wxyz(q_start_pose, q_goal_pose, float(alpha))
        ik_target = np.concatenate([p_target, q_target]).astype(np.float64)
        status, ik_results = demo.planner.IK(ik_target, q_cursor, threshold=0.002)
        print(f"[planner] {label} local_ik segment={seg_idx}/{num_segments} status: {status}")
        if status != "Success":
            return None

        q_goal = _pick_nearest_ik_solution(
            demo,
            ik_results,
            q_cursor,
            max_joint_delta=max_segment_joint_delta,
            max_joint7_delta=max_segment_joint7_delta,
            max_norm_delta=max_segment_norm_delta,
        )
        if q_goal is None:
            print(
                f"[planner] {label} local_ik segment={seg_idx}/{num_segments} rejected non-local IK solutions "
                f"(max_joint_delta={max_segment_joint_delta:.3f}, "
                f"max_joint7_delta={max_segment_joint7_delta:.3f}, "
                f"max_norm_delta={max_segment_norm_delta:.3f})"
            )
            return None

        seg_path = build_collision_checked_linear_joint_path(
            demo,
            q_cursor,
            q_goal,
            use_attach=use_attach,
            label=f"{label}_local_{seg_idx:02d}",
            max_delta=0.03 if use_attach else 0.04,
            allow_start_in_collision=bool(allow_start_in_collision and seg_idx == 1),
        )
        if seg_path is None:
            return None
        q_path.extend(seg_path)
        q_cursor = q_goal

    print(f"[planner] {label} local cartesian escape succeeded with {len(q_path)} waypoints")
    return q_path


def plan_lift_path(
    demo,
    pose,
    *,
    variant_name: str = "array_wxyz",
    use_attach: bool = False,
    label: str = "lift",
    planning_time: float = 4.0,
    rrt_range: float = 0.15,
    start_q=None,
    allow_pose_rrt_fallback: bool = False,
    max_segment_joint_delta: float = 0.20,
    max_segment_joint7_delta: float = 0.25,
    max_segment_norm_delta: float = 0.35,
    allow_start_in_collision: bool = False,
):
    local_path = plan_local_cartesian_lift_path(
        demo,
        pose,
        use_attach=use_attach,
        label=label,
        start_q=start_q,
        max_segment_joint_delta=max_segment_joint_delta,
        max_segment_joint7_delta=max_segment_joint7_delta,
        max_segment_norm_delta=max_segment_norm_delta,
        allow_start_in_collision=allow_start_in_collision,
    )
    if local_path:
        return local_path
    if not allow_pose_rrt_fallback:
        print(f"[planner] {label} local cartesian escape failed; pose RRT fallback disabled for sim2real-friendly lift")
        return None
    print(f"[planner] {label} local cartesian escape failed; falling back to pose RRT")
    return plan_pose_path(
        demo,
        pose,
        variant_name=variant_name,
        use_attach=use_attach,
        label=label,
        planning_time=planning_time,
        rrt_range=rrt_range,
        start_q=start_q,
        allow_start_in_collision=allow_start_in_collision,
    )


def _nearest_obstacle_escape_xy(demo, reference_xy: np.ndarray) -> np.ndarray | None:
    scene_obstacles = list(getattr(demo, "scene_obstacles", []) or [])
    if not scene_obstacles:
        return None
    ref_xy = np.asarray(reference_xy, dtype=np.float32).reshape(2)
    best_dir = None
    best_dist = None
    for item in scene_obstacles:
        T_world_obj = item.get("T_world_obj")
        if T_world_obj is None:
            continue
        obs_xy = np.asarray(T_world_obj[:2, 3], dtype=np.float32).reshape(2)
        delta_xy = ref_xy - obs_xy
        dist_xy = float(np.linalg.norm(delta_xy))
        if dist_xy <= 1e-6:
            continue
        if best_dist is None or dist_xy < best_dist:
            best_dist = dist_xy
            best_dir = (delta_xy / dist_xy).astype(np.float32)
    return best_dir


def make_lifted_tcp_pose(demo, lift_height: float, retreat_distance: float = 0.0):
    tcp_pose = demo.tcp.pose
    tcp_p = flatten_np(tcp_pose.p)[:3].copy()
    tcp_p[2] += float(lift_height)
    retreat_distance = float(retreat_distance)
    if retreat_distance > 1e-6:
        escape_xy = _nearest_obstacle_escape_xy(demo, tcp_p[:2])
        if escape_xy is None:
            escape_xy = np.array([1.0, 0.0], dtype=np.float32)
        tcp_p[:2] += escape_xy * retreat_distance
        print(
            "[planner] post-grasp escape retreat:",
            f"distance={retreat_distance:.4f} m, direction_xy={np.round(escape_xy, 4)}",
        )
    return make_pose_with_position(tcp_pose, tcp_p)


def make_tcp_axis_retreat_pose(demo, retreat_distance: float):
    from transforms3d.quaternions import quat2mat

    tcp_pose = demo.tcp.pose
    tcp_p = flatten_np(tcp_pose.p)[:3].copy().astype(np.float32)
    tcp_q = flatten_np(tcp_pose.q)[:4].copy().astype(np.float32)
    R_tcp = quat2mat(tcp_q).astype(np.float32)
    approaching = _normalize_vec(R_tcp[:, 2])
    if approaching is None:
        return make_pose_with_position(tcp_pose, tcp_p)
    tcp_p = (tcp_p - approaching * float(retreat_distance)).astype(np.float32)
    return make_pose_with_position(tcp_pose, tcp_p)


def plan_post_place_clearance_path(
    demo,
    *,
    retreat_distance: float = 0.05,
    start_q=None,
    label: str = "post_place_clearance",
):
    clearance_pose = make_tcp_axis_retreat_pose(demo, retreat_distance)
    q_start = np.asarray(demo.current_arm_qpos() if start_q is None else start_q, dtype=np.float32).reshape(-1)[:7]
    q_path = plan_lift_path(
        demo,
        clearance_pose,
        use_attach=False,
        label=label,
        start_q=q_start,
        planning_time=2.0,
        rrt_range=0.10,
        allow_pose_rrt_fallback=True,
        max_segment_joint_delta=0.35,
        max_segment_joint7_delta=0.45,
        max_segment_norm_delta=0.60,
    )
    return clearance_pose, q_path


def confirm_joint_path_motion(demo, bridge_mod, label: str, q_path, args) -> bool:
    if not q_path:
        print(f"[planner] {label} is a zero-length joint path; nothing to confirm")
        return True
    q_target = np.asarray(q_path[-1], dtype=np.float32).reshape(-1)[:7]
    render_planned_trajectory_preview(demo, bridge_mod, label, q_path, args)
    return confirm_joint_motion(demo, bridge_mod, f"{label} (path, {len(q_path)} waypoints)", q_target, args)


def _validate_real_waypoint_segment(demo, q_start: np.ndarray, q_goal: np.ndarray, *, use_attach: bool, label: str, allow_start_in_collision: bool = False) -> bool:
    q_start = np.asarray(q_start, dtype=np.float32).reshape(-1)[:7]
    q_goal = np.asarray(q_goal, dtype=np.float32).reshape(-1)[:7]
    q_goal = retarget_goal_q_to_nearest_equivalent(demo, q_goal, q_start)
    seg = _build_collision_checked_linear_joint_path_quiet(
        demo,
        q_start,
        q_goal,
        use_attach=use_attach,
        max_delta=_dense_collision_validate_max_delta(use_attach),
        allow_start_in_collision=allow_start_in_collision,
    )
    if seg is not None:
        return True
    print(f"[planner] {label} blocked before execution: collision detected on the next real segment")
    build_collision_checked_linear_joint_path(
        demo,
        q_start,
        q_goal,
        use_attach=use_attach,
        label=f"{label}_precheck",
        max_delta=_dense_collision_validate_max_delta(use_attach),
        allow_start_in_collision=allow_start_in_collision,
    )
    return False


def _render_real_shadow_step(demo, bridge_mod, args, q_shadow: np.ndarray):
    sync_demo_arm_qpos(demo, q_shadow)
    scene = None
    try:
        scene = getattr(getattr(demo, "base_env", None), "scene", None)
        if scene is None:
            scene = getattr(getattr(getattr(demo, "env", None), "unwrapped", None), "scene", None)
        update_render = getattr(scene, "update_render", None)
        if callable(update_render):
            update_render()
    except Exception:
        pass
    if getattr(args, "render_mode", None) == "human":
        try:
            demo.env.render()
        except Exception:
            try:
                bridge_mod.render_preview(demo.env, repeats=1)
            except Exception:
                pass


def execute_real_waypoint_path_with_shadow(
    demo,
    bridge_mod,
    real_exec: RealmanJointExecutor,
    label: str,
    q_path,
    gripper_pos: float,
    args,
    *,
    use_attach: bool = False,
    allow_start_in_collision: bool = False,
):
    q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in q_path]
    q_prev = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    q_sent = q_prev
    if bool(getattr(args, "real_stream_waypoint_path", True)):
        dense_ok = validate_joint_path_segments(
            demo,
            q_prev,
            q_path,
            use_attach=use_attach,
            label=label,
            max_delta=_dense_collision_validate_max_delta(use_attach),
            allow_start_in_collision=allow_start_in_collision,
        )
        if not dense_ok:
            return False, None
        hold_steps = int(args.real_hold_steps) if q_path else 0
        q_sent = real_exec.move_waypoint_path(
            q_path,
            gripper_pos=gripper_pos,
            max_delta_per_step=args.real_max_delta_per_step,
            hz=args.real_control_hz,
            hold_steps=hold_steps,
            label=label,
            shadow_callback=lambda q_cmd: _render_real_shadow_step(demo, bridge_mod, args, q_cmd),
        )
        _render_real_shadow_step(demo, bridge_mod, args, q_sent)
        return True, q_sent
    for idx, q_target in enumerate(q_path):
        stage_label = f"{label}_wp{idx}"
        if not _validate_real_waypoint_segment(
            demo,
            q_prev,
            q_target,
            use_attach=use_attach,
            label=stage_label,
            allow_start_in_collision=bool(allow_start_in_collision and idx == 0),
        ):
            return False, None
        hold_steps = int(args.real_hold_steps) if idx == len(q_path) - 1 else 0
        q_sent = real_exec.move_linear(
            q_target,
            gripper_pos=gripper_pos,
            max_delta_per_step=args.real_max_delta_per_step,
            hz=args.real_control_hz,
            hold_steps=hold_steps,
            label=stage_label,
            shadow_callback=lambda q_cmd: _render_real_shadow_step(demo, bridge_mod, args, q_cmd),
        )
        _render_real_shadow_step(demo, bridge_mod, args, q_sent)
        q_prev = np.asarray(q_sent, dtype=np.float32).reshape(-1)[:7]
    return True, q_sent


def execute_joint_path_stage(demo, bridge_mod, real_exec: RealmanJointExecutor | None, label: str, q_path, gripper_pos: float, args, *, use_attach: bool = False, allow_start_in_collision: bool = False):
    q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in q_path]
    if not q_path:
        q_current = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
        print(f"[planner] {label} is a zero-length joint path; skipping execution")
        return True, q_current
    if not confirm_joint_path_motion(demo, bridge_mod, label, q_path, args):
        print(f"[abort] user cancelled before executing {label}")
        return False, None
    if real_exec is not None:
        ok, q_sent = execute_real_waypoint_path_with_shadow(
            demo,
            bridge_mod,
            real_exec,
            label,
            q_path,
            gripper_pos,
            args,
            use_attach=use_attach,
            allow_start_in_collision=allow_start_in_collision,
        )
        if not ok:
            return False, None
    else:
        print(f"[dry-run] skipped real execution for {label}")
        sync_demo_arm_qpos(demo, q_path[-1])
    return True, q_path[-1]


def align_real_robot_to_sim_start(demo, bridge_mod, real_exec: RealmanJointExecutor, q_target: np.ndarray, args) -> tuple[bool, np.ndarray | None]:
    q_target = np.asarray(q_target, dtype=np.float32).reshape(-1)[:7]
    real_exec.set_gripper(args.real_gripper_open)
    sync_demo_gripper_state(demo, closed=False, steps=4)
    q_real_start = np.asarray(real_exec.get_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    sync_demo_arm_qpos(demo, q_real_start)
    if bool(getattr(args, "no_move_real_to_sim_start", False)):
        print("[real robot setup] skipped moving the real robot to the simulation start pose")
        return True, q_real_start
    if not bool(getattr(args, "align_real_to_sim_start_before_cycle", False)):
        print("[real robot setup] keeping the current real robot pose; planning will start from the current arm configuration instead of sim_start")
        return True, q_real_start
    q_delta = (q_target - q_real_start + np.pi) % (2.0 * np.pi) - np.pi
    if float(np.max(np.abs(q_delta))) <= 1e-3:
        print("[real robot setup] real robot is already aligned with the simulation start pose")
        sync_demo_arm_qpos(demo, q_real_start)
        return True, q_real_start

    pre_lift_height = float(getattr(args, "sim_start_pre_lift_height", 0.0))
    pre_retreat_distance = float(getattr(args, "sim_start_pre_retreat_distance", 0.0))
    direct_attempt_count_before_pre_escape = int(
        max(1, getattr(args, "sim_start_direct_attempt_count_before_pre_escape", 2))
    )

    def _plan_sim_start_path(q_start_for_plan: np.ndarray, *, limited_before_pre_escape: bool = False):
        print("[planner] planning a collision-aware real alignment path to sim_start")
        attempt_count_override = None
        if limited_before_pre_escape and max(pre_lift_height, pre_retreat_distance) > 1e-6:
            attempt_count_override = direct_attempt_count_before_pre_escape
            print(
                f"[planner] sim_start direct pre-escape probe uses only {attempt_count_override} RRT attempt(s) before falling back to sim_start_pre_escape"
            )
        return plan_joint_path(
            demo,
            q_target,
            use_attach=False,
            label="sim_start",
            planning_time=4.0,
            rrt_range=0.15,
            start_q=q_start_for_plan,
            allow_linear_fallback=True,
            allow_reverse_rrt_fallback=True,
            prefer_reverse_rrt=True,
            attempt_count_override=attempt_count_override,
        )

    q_align_start = q_real_start.copy()
    q_path = _plan_sim_start_path(q_align_start, limited_before_pre_escape=True)
    if q_path is None and max(pre_lift_height, pre_retreat_distance) > 1e-6:
        pre_escape_candidates: list[tuple[float, float]] = []
        for retreat_m, lift_m in [
            (pre_retreat_distance, pre_lift_height),
            (max(pre_retreat_distance, 0.05), pre_lift_height),
            (pre_retreat_distance, max(pre_lift_height, 0.03)),
            (max(pre_retreat_distance, 0.05), max(pre_lift_height, 0.03)),
            (max(pre_retreat_distance, 0.07), max(pre_lift_height, 0.05)),
        ]:
            candidate = (float(max(retreat_m, 0.0)), float(max(lift_m, 0.0)))
            if candidate not in pre_escape_candidates and max(candidate) > 1e-6:
                pre_escape_candidates.append(candidate)

        pre_escape_executed = False
        for candidate_idx, (candidate_retreat_m, candidate_lift_m) in enumerate(pre_escape_candidates, start=1):
            print(
                f"[planner] direct sim_start alignment failed; trying sim_start_pre_escape "
                f"candidate {candidate_idx}/{len(pre_escape_candidates)} "
                f"(retreat={candidate_retreat_m:.3f} m, lift={candidate_lift_m:.3f} m)"
            )
            pre_escape_label = (
                "sim_start_pre_escape"
                if len(pre_escape_candidates) == 1
                else f"sim_start_pre_escape_r{int(round(candidate_retreat_m * 1000.0))}_z{int(round(candidate_lift_m * 1000.0))}"
            )
            pre_escape_pose = make_lifted_tcp_pose(
                demo,
                candidate_lift_m,
                retreat_distance=candidate_retreat_m,
            )
            pre_escape_path = plan_lift_path(
                demo,
                pre_escape_pose,
                use_attach=False,
                label=pre_escape_label,
                start_q=q_align_start,
                planning_time=3.0,
                rrt_range=0.15,
                allow_pose_rrt_fallback=True,
                max_segment_joint_delta=0.35,
                max_segment_joint7_delta=0.60,
                max_segment_norm_delta=0.60,
                allow_start_in_collision=True,
            )
            if pre_escape_path is None or len(pre_escape_path) == 0:
                print(
                    f"[warn] {pre_escape_label} planning failed for retreat={candidate_retreat_m:.3f} m, "
                    f"lift={candidate_lift_m:.3f} m"
                )
                continue

            ok, q_sent = execute_joint_path_stage(
                demo,
                bridge_mod,
                real_exec,
                pre_escape_label,
                pre_escape_path,
                args.real_gripper_open,
                args,
                use_attach=False,
                allow_start_in_collision=True,
            )
            if not ok:
                print(f"[abort] failed while executing {pre_escape_label}")
                return False, None

            q_align_start = np.asarray(q_sent, dtype=np.float32).reshape(-1)[:7]
            sync_demo_arm_qpos(demo, q_align_start)
            q_path = _plan_sim_start_path(q_align_start, limited_before_pre_escape=False)
            pre_escape_executed = True
            if q_path is not None:
                break

        if not pre_escape_executed:
            print("[warn] sim_start_pre_escape exhausted all candidates without finding a valid escape path")

    if q_path is None:
        print("[FAIL] sim_start alignment planning failed")
        print_failure_diagnostics(demo, args, "sim_start", q_target=q_target, use_attach=False)
        if getattr(args, "render_mode", None) == "human":
            bridge_mod.render_preview(demo.env, repeats=12)
        return False, None

    ok, q_sent = execute_joint_path_stage(
        demo,
        bridge_mod,
        real_exec,
        "sim_start",
        q_path,
        args.real_gripper_open,
        args,
        use_attach=False,
    )
    if ok and getattr(args, "render_mode", None) == "human":
        bridge_mod.render_preview(demo.env, repeats=5)
    return ok, q_sent


def execute_pose_path_stage(
    demo,
    bridge_mod,
    real_exec: RealmanJointExecutor | None,
    label: str,
    pose,
    q_path,
    gripper_pos: float,
    args,
    *,
    use_attach: bool = False,
    allow_start_in_collision: bool = False,
    skip_confirmation: bool = False,
):
    q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in q_path]
    if not q_path:
        q_current = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
        print(f"[planner] {label} is a zero-length pose path; skipping execution")
        return True, q_current
    if not skip_confirmation and not confirm_planned_motion_or_skip(demo, bridge_mod, label, pose, q_path[-1], args, q_preview_path=q_path):
        print(f"[abort] user cancelled before executing {label}")
        return False, None
    if skip_confirmation:
        print(f"[planner] auto-executing {label} without separate preview/confirmation")
    if real_exec is not None:
        ok, q_sent = execute_real_waypoint_path_with_shadow(
            demo,
            bridge_mod,
            real_exec,
            label,
            q_path,
            gripper_pos,
            args,
            use_attach=use_attach,
            allow_start_in_collision=allow_start_in_collision,
        )
        if not ok:
            return False, None
    else:
        print(f"[dry-run] skipped real execution for {label}")
        sync_demo_arm_qpos(demo, q_path[-1])
    return True, q_path[-1]


def retry_pregrasp_after_escape(
    demo,
    bridge_mod,
    real_exec: RealmanJointExecutor | None,
    label: str,
    pregrasp_pose,
    gripper_pos: float,
    args,
):
    if real_exec is None:
        return False, None

    base_retreat = float(max(getattr(args, "pregrasp_blocked_retry_retreat_distance", 0.0), 0.0))
    base_lift = float(max(getattr(args, "pregrasp_blocked_retry_lift_height", 0.0), 0.0))
    if max(base_retreat, base_lift) <= 1e-6:
        return False, None

    candidates: list[tuple[float, float]] = []
    for retreat_m, lift_m in [
        (base_retreat, base_lift),
        (max(base_retreat, 0.05), base_lift),
        (base_retreat, max(base_lift, 0.02)),
        (max(base_retreat, 0.05), max(base_lift, 0.02)),
    ]:
        candidate = (float(max(retreat_m, 0.0)), float(max(lift_m, 0.0)))
        if candidate not in candidates and max(candidate) > 1e-6:
            candidates.append(candidate)

    if not candidates:
        return False, None

    for candidate_idx, (candidate_retreat_m, candidate_lift_m) in enumerate(candidates, start=1):
        print(
            f"[planner] {label} execution was blocked before any real motion; "
            f"trying {label}_pre_escape candidate {candidate_idx}/{len(candidates)} "
            f"(retreat={candidate_retreat_m:.3f} m, lift={candidate_lift_m:.3f} m)"
        )
        escape_label = (
            f"{label}_pre_escape"
            if len(candidates) == 1
            else f"{label}_pre_escape_r{int(round(candidate_retreat_m * 1000.0))}_z{int(round(candidate_lift_m * 1000.0))}"
        )
        escape_pose = make_lifted_tcp_pose(
            demo,
            candidate_lift_m,
            retreat_distance=candidate_retreat_m,
        )
        escape_path = plan_lift_path(
            demo,
            escape_pose,
            variant_name=args.variant,
            use_attach=False,
            label=escape_label,
            start_q=np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7],
            planning_time=3.0,
            rrt_range=0.15,
            allow_pose_rrt_fallback=True,
            max_segment_joint_delta=0.35,
            max_segment_joint7_delta=0.80,
            max_segment_norm_delta=0.60,
            allow_start_in_collision=True,
        )
        if escape_path is None or len(escape_path) == 0:
            print(
                f"[warn] {escape_label} planning failed for retreat={candidate_retreat_m:.3f} m, "
                f"lift={candidate_lift_m:.3f} m"
            )
            continue

        ok_escape, _ = execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            escape_label,
            escape_pose,
            escape_path,
            gripper_pos,
            args,
            use_attach=False,
            allow_start_in_collision=True,
        )
        if not ok_escape:
            print(f"[warn] {escape_label} execution failed")
            continue

        retry_label = f"{label}_after_pre_escape"
        retry_path = plan_pose_path(
            demo,
            pregrasp_pose,
            variant_name=args.variant,
            label=retry_label,
            use_attach=False,
            start_q=np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7],
        )
        if retry_path is not None:
            ok_retry, q_sent = execute_pose_path_stage(
                demo,
                bridge_mod,
                real_exec,
                retry_label,
                pregrasp_pose,
                retry_path,
                gripper_pos,
                args,
                use_attach=False,
            )
            if ok_retry:
                return True, q_sent

        print(f"[planner] {retry_label} pose-path planning failed after pre-escape; falling back to terminal joint target")
        q_retry = demo.plan_terminal_q(pregrasp_pose, variant_name=args.variant)
        if q_retry is None:
            print(f"[warn] {retry_label} terminal IK failed after pre-escape")
            continue
        ok_retry, q_sent = execute_stage(
            demo,
            bridge_mod,
            real_exec,
            retry_label,
            pregrasp_pose,
            q_retry,
            gripper_pos,
            args,
        )
        if ok_retry:
            return True, q_sent

    print(f"[warn] {label} pre-execution escape retry exhausted all candidates")
    return False, None



def enforce_min_grasp_tcp_z(grasp_pose, pregrasp_pose, min_tcp_z: float):
    min_tcp_z = float(min_tcp_z)
    if min_tcp_z <= 0:
        return grasp_pose, pregrasp_pose, 0.0

    grasp_p = flatten_np(grasp_pose.p)[:3].copy()
    if grasp_p[2] >= min_tcp_z:
        return grasp_pose, pregrasp_pose, 0.0

    delta_z = float(min_tcp_z - grasp_p[2])
    grasp_p[2] += delta_z
    pregrasp_p = flatten_np(pregrasp_pose.p)[:3].copy()
    pregrasp_p[2] += delta_z
    grasp_pose = make_pose_with_position(grasp_pose, grasp_p)
    pregrasp_pose = make_pose_with_position(pregrasp_pose, pregrasp_p)
    return grasp_pose, pregrasp_pose, delta_z


def enforce_topdown_grasp_insertion_limit(demo, args, grasp_pose, pregrasp_pose):
    grasp_mode = str(getattr(args, "grasp_mode", "object_normal") or "object_normal").strip().lower()
    if grasp_mode not in {"topdown_long_axis", "pen_topdown_insert_ready", "long_axis_adaptive"}:
        return grasp_pose, pregrasp_pose, 0.0

    max_insertion_depth = float(getattr(args, "topdown_grasp_max_insertion_depth", 0.0))
    if max_insertion_depth <= 0:
        return grasp_pose, pregrasp_pose, 0.0

    object_top_z = get_demo_object_world_top_z(demo, args)
    if object_top_z is None:
        return grasp_pose, pregrasp_pose, 0.0
    desired_grasp_z = float(object_top_z - max_insertion_depth)
    grasp_p = flatten_np(grasp_pose.p)[:3].copy()
    if grasp_p[2] >= desired_grasp_z:
        return grasp_pose, pregrasp_pose, 0.0

    delta_z = float(desired_grasp_z - grasp_p[2])
    grasp_p[2] += delta_z
    pregrasp_p = flatten_np(pregrasp_pose.p)[:3].copy()
    pregrasp_p[2] += delta_z
    grasp_pose = make_pose_with_position(grasp_pose, grasp_p)
    pregrasp_pose = make_pose_with_position(pregrasp_pose, pregrasp_p)
    return grasp_pose, pregrasp_pose, delta_z


def shift_pose_world_z(pose, delta_z: float):
    p = flatten_np(pose.p)[:3].copy()
    p[2] += float(delta_z)
    return make_pose_with_position(pose, p)


def shift_pose_along_approach(pose, backoff: float):
    from transforms3d.quaternions import quat2mat

    p = flatten_np(pose.p)[:3].copy()
    q = flatten_np(pose.q)[:4].copy()
    approaching = np.asarray(quat2mat(q)[:, 2], dtype=np.float32)
    p = p - approaching * float(backoff)
    return make_pose_with_position(pose, p)


def _normalize_vec(v) -> np.ndarray | None:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= 1e-8:
        return None
    return (v / n).astype(np.float32)


def _rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize_vec(axis)
    if axis is None:
        return np.eye(3, dtype=np.float32)
    x, y, z = [float(v) for v in axis[:3]]
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    C = 1.0 - c
    return np.asarray(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float32,
    )


def _get_robot_base_world_position(demo) -> np.ndarray | None:
    try:
        pose = demo.env.unwrapped.agent.robot.pose
        return flatten_np(pose.p)[:3].astype(np.float32)
    except Exception:
        return None


def tilt_pose_toward_robot(demo, pose, angle_deg: float):
    from transforms3d.quaternions import quat2mat

    angle_deg = float(angle_deg)
    if abs(angle_deg) <= 1e-6:
        return None
    tcp_p = flatten_np(pose.p)[:3].astype(np.float32)
    q_wxyz = flatten_np(pose.q)[:4].astype(np.float32)
    robot_base_p = _get_robot_base_world_position(demo)
    if robot_base_p is None:
        return None

    to_robot_xy = robot_base_p - tcp_p
    to_robot_xy[2] = 0.0
    to_robot_xy = _normalize_vec(to_robot_xy)
    if to_robot_xy is None:
        return None

    R_tcp = quat2mat(q_wxyz).astype(np.float32)
    approaching = _normalize_vec(R_tcp[:, 2])
    if approaching is None:
        return None

    tilt_axis = _normalize_vec(np.cross(approaching, to_robot_xy))
    if tilt_axis is None:
        return None

    best_pose = None
    best_score = -np.inf
    angle_rad = np.deg2rad(abs(angle_deg))
    for sign in (1.0, -1.0):
        R_delta = _rotation_matrix_from_axis_angle(tilt_axis, sign * angle_rad)
        R_new = (R_delta @ R_tcp).astype(np.float32)
        approaching_new = _normalize_vec(R_new[:, 2])
        if approaching_new is None:
            continue
        approaching_xy = approaching_new.copy()
        approaching_xy[2] = 0.0
        approaching_xy = _normalize_vec(approaching_xy)
        score = -np.inf if approaching_xy is None else float(np.dot(approaching_xy, to_robot_xy))
        if score > best_score:
            best_score = score
            best_pose = Pose.create_from_pq(p=tcp_p, q=bridge_mod_mat2quat(R_new))
    return best_pose


def shift_pose_toward_robot_xy(demo, pose, distance_m: float):
    distance_m = float(distance_m)
    if abs(distance_m) <= 1e-6:
        return pose
    tcp_p = flatten_np(pose.p)[:3].astype(np.float32)
    q_wxyz = flatten_np(pose.q)[:4].astype(np.float32)
    robot_base_p = _get_robot_base_world_position(demo)
    if robot_base_p is None:
        return None
    to_robot_xy = robot_base_p - tcp_p
    to_robot_xy[2] = 0.0
    to_robot_xy = _normalize_vec(to_robot_xy)
    if to_robot_xy is None:
        return None
    shifted_p = tcp_p.copy()
    shifted_p[:2] += to_robot_xy[:2] * distance_m
    return Pose.create_from_pq(p=shifted_p, q=q_wxyz)


def rotate_pose_about_world_z(pose, yaw_deg: float):
    yaw_deg = float(yaw_deg)
    if abs(yaw_deg) <= 1e-6:
        return pose
    tcp_p = flatten_np(pose.p)[:3].astype(np.float32)
    q_wxyz = flatten_np(pose.q)[:4].astype(np.float32)
    R_tcp = quat2mat(q_wxyz).astype(np.float32)
    Rz = euler2mat(0.0, 0.0, np.deg2rad(yaw_deg)).astype(np.float32)
    R_new = (Rz @ R_tcp).astype(np.float32)
    return Pose.create_from_pq(p=tcp_p, q=bridge_mod_mat2quat(R_new))


def build_grasp_pose_variants(demo, grasp_pose, args):
    variants = [("grasp", grasp_pose)]
    grasp_mode = str(getattr(args, "grasp_mode", "object_normal") or "object_normal").strip().lower()
    if grasp_mode != "topdown_long_axis":
        return variants

    seen = set()

    def _key(pose):
        p = tuple(np.round(flatten_np(pose.p)[:3], 5).tolist())
        q = tuple(np.round(flatten_np(pose.q)[:4], 5).tolist())
        return p + q

    seen.add(_key(grasp_pose))
    tilt_degs = [float(v) for v in getattr(args, "topdown_tilt_toward_robot_deg", []) or [] if abs(float(v)) > 1e-6]
    shift_ds = [float(v) for v in getattr(args, "topdown_tilt_toward_robot_shift_m", []) or [] if float(v) >= 0.0]
    yaw_degs = [float(v) for v in getattr(args, "topdown_grasp_yaw_variant_deg", []) or []]
    if not shift_ds:
        shift_ds = [0.0]

    def _append_variant(label: str, pose):
        if pose is None:
            return
        key = _key(pose)
        if key in seen:
            return
        seen.add(key)
        variants.append((label, pose))

    for yaw_deg in yaw_degs:
        if abs(yaw_deg) <= 1e-6:
            continue
        _append_variant(
            f"grasp_yaw_{int(round(yaw_deg))}deg",
            rotate_pose_about_world_z(grasp_pose, yaw_deg),
        )

    for tilt_deg in tilt_degs:
        tilted_pose = tilt_pose_toward_robot(demo, grasp_pose, tilt_deg)
        if tilted_pose is None:
            continue
        for shift_d in shift_ds:
            shifted_pose = tilted_pose if shift_d <= 1e-6 else shift_pose_toward_robot_xy(demo, tilted_pose, shift_d)
            if shifted_pose is None:
                continue
            base_label = f"grasp_tilt_toward_robot_{int(round(abs(tilt_deg)))}deg"
            if shift_d > 1e-6:
                base_label += f"_shift_{int(round(1000.0 * shift_d))}mm"
            _append_variant(base_label, shifted_pose)
            for yaw_deg in yaw_degs:
                if abs(yaw_deg) <= 1e-6:
                    continue
                _append_variant(
                    f"{base_label}_yaw_{int(round(yaw_deg))}deg",
                    rotate_pose_about_world_z(shifted_pose, yaw_deg),
                )
    return variants


def plan_grasp_pose_with_fallbacks(demo, bridge_mod, grasp_pose, args, *, base_label: str = "grasp"):
    candidates = []
    seen = set()

    def _pose_key(pose):
        p = tuple(np.round(flatten_np(pose.p)[:3], 5).tolist())
        q = tuple(np.round(flatten_np(pose.q)[:4], 5).tolist())
        return p + q

    def _append(label, pose):
        key = _pose_key(pose)
        if key in seen:
            return
        seen.add(key)
        candidates.append((label, pose))

    _append(base_label, grasp_pose)
    z_lifts = [float(v) for v in getattr(args, "grasp_fallback_z_lifts", []) or [] if float(v) > 1e-6]
    backoffs = [float(v) for v in getattr(args, "grasp_fallback_approach_backoffs", []) or [] if float(v) > 1e-6]
    for dz in z_lifts:
        _append(f"{base_label}_raise_{int(round(dz * 1000))}mm", shift_pose_world_z(grasp_pose, dz))
    for backoff in backoffs:
        _append(f"{base_label}_backoff_{int(round(backoff * 1000))}mm", shift_pose_along_approach(grasp_pose, backoff))
    for dz in z_lifts:
        for backoff in backoffs:
            pose = shift_pose_world_z(shift_pose_along_approach(grasp_pose, backoff), dz)
            _append(f"{base_label}_raise_{int(round(dz * 1000))}mm_backoff_{int(round(backoff * 1000))}mm", pose)

    for label, pose in candidates:
        print(f"[planner] trying {label}")
        print(f"[planner] {label} pose p:", np.round(flatten_np(pose.p)[:3], 6))
        print(f"[planner] {label} pose q:", np.round(flatten_np(pose.q)[:4], 6))
        demo.preview_target_pose(pose)
        if args.render_mode == "human":
            bridge_mod.render_preview(demo.env, repeats=10)
        q = demo.plan_terminal_q(pose, variant_name=args.variant)
        if q is None:
            continue
        if label == base_label:
            return PlannedGraspExecution(
                label=label,
                approach_pose=pose,
                approach_q=np.asarray(q, dtype=np.float32).reshape(-1)[:7],
                final_grasp_pose=pose,
                final_grasp_path=None,
                require_runtime_true_grasp_planning=False,
            )

        print(f"[planner] {label} is reachable; deferring descent-to-true-grasp planning until after the fallback approach is executed")
        return PlannedGraspExecution(
            label=label,
            approach_pose=pose,
            approach_q=np.asarray(q, dtype=np.float32).reshape(-1)[:7],
            final_grasp_pose=grasp_pose,
            final_grasp_path=None,
            require_runtime_true_grasp_planning=True,
        )

    return None


def recapture_active_object_pose_from_foundationpose(demo, bridge_mod, args) -> bool:
    fp_rt = getattr(demo, "foundationpose_runtime", None)
    T_base_cam = getattr(demo, "T_base_cam", None)
    if fp_rt is None or T_base_cam is None:
        print("[warn] FoundationPose runtime is unavailable; cannot recapture the pushed object pose")
        return False
    try:
        T_cam_obj, _ = bridge_mod.capture_scene_poses_from_foundationpose(fp_rt, args)
    except Exception as exc:
        print(f"[warn] failed to recapture object pose from FoundationPose: {exc}")
        return False
    try:
        T_world_obj = bridge_mod.map_camera_pose_to_pick_world(T_cam_obj, T_base_cam, demo.env, args)
        bridge_mod.apply_pose_to_pick_object(demo.env, T_world_obj)
        demo.refresh_runtime_handles(rebuild_visual=False)
        lift_active_object_above_table_if_needed(
            demo,
            args,
            min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
        )
        print("[rearrangement] refreshed active object pose from FoundationPose after real execution")
        return True
    except Exception as exc:
        print(f"[warn] failed to apply refreshed object pose after real rearrangement: {exc}")
        return False


def build_foundationpose_scene_cache_key(args):
    return (
        str(Path(getattr(args, "foundationpose_root", DEFAULT_FOUNDATIONPOSE_ROOT)).expanduser()),
        str(Path(getattr(args, "camera_extrinsic_opencv_path", DEFAULT_CAMERA_EXTRINSIC)).expanduser()),
        bool(getattr(args, "use_direct_camera_extrinsic", False)),
        bool(getattr(args, "disable_object_spec_obstacles", False)),
        tuple(np.asarray(getattr(args, "foundationpose_position_offset", [0.0, 0.0, 0.0]), dtype=np.float32).round(6).tolist()),
        tuple(np.asarray(getattr(args, "foundationpose_local_rotation_offset_deg", [0.0, 0.0, 0.0]), dtype=np.float32).round(6).tolist()),
        int(getattr(args, "camera_width", 640)),
        int(getattr(args, "camera_height", 480)),
        int(getattr(args, "camera_fps", 30)),
        int(getattr(args, "foundationpose_stabilize_frames", 0)),
        int(getattr(args, "foundationpose_smoothing_window", 1)),
        int(getattr(args, "track_refine_iter", 2)),
        str(getattr(args, "camera_serial", None) or ""),
    )


def capture_or_reuse_foundationpose_scene(args, bridge_mod, scene_capture_cache=None):
    cache_key = build_foundationpose_scene_cache_key(args)
    target_name = normalize_object_name(getattr(args, "object_name", None))
    selected_obstacle_names = [
        normalize_object_name(name)
        for name in (getattr(args, "selected_obstacle_object_names", []) or [])
        if normalize_object_name(name) is not None
    ]
    if (
        bool(getattr(args, "reuse_foundationpose_scene_across_cycles", True))
        and isinstance(scene_capture_cache, dict)
        and scene_capture_cache.get("key") == cache_key
    ):
        cached_objects = dict(scene_capture_cache.get("objects", {}) or {})
        target_entry = cached_objects.get(target_name)
        if target_entry is not None and all(name in cached_objects for name in selected_obstacle_names):
            scene_obstacles = []
            for name in selected_obstacle_names:
                if name == target_name:
                    continue
                scene_obstacles.append(copy.deepcopy(cached_objects[name]))
            print("[foundationpose] reusing cached scene capture from the previous cycle")
            return (
                scene_capture_cache["fp_rt"],
                np.asarray(scene_capture_cache["T_base_cam"], dtype=np.float32),
                np.asarray(target_entry["T_cam_obj"], dtype=np.float32),
                scene_obstacles,
            )

    foundationpose_root = bridge_mod.resolve_foundationpose_root(args.foundationpose_root)
    fp_rt = bridge_mod.load_foundationpose_module(foundationpose_root)
    T_base_cam = bridge_mod.load_matrix(args.camera_extrinsic_opencv_path)
    fp_rt.set_logging_format()
    fp_rt.set_seed(args.seed)
    T_cam_obj, scene_obstacles = bridge_mod.capture_scene_poses_from_foundationpose(fp_rt, args)

    if isinstance(scene_capture_cache, dict):
        cached_objects = {}
        if target_name is not None:
            cached_objects[target_name] = {
                "object_name": target_name,
                "label": str(getattr(args, "target_object_name", "") or target_name),
                "score": 1.0,
                "box": np.zeros(4, dtype=np.float32),
                "T_cam_obj": np.asarray(T_cam_obj, dtype=np.float32),
                "object_args": argparse.Namespace(**vars(args).copy()),
            }
        for item in scene_obstacles:
            object_name = normalize_object_name(item.get("object_name"))
            if object_name is None:
                continue
            cached_objects[object_name] = copy.deepcopy(item)
        scene_capture_cache.clear()
        scene_capture_cache.update(
            {
                "key": cache_key,
                "fp_rt": fp_rt,
                "T_base_cam": np.asarray(T_base_cam, dtype=np.float32),
                "objects": cached_objects,
            }
        )
    return fp_rt, T_base_cam, T_cam_obj, scene_obstacles


def update_scene_capture_cache_object(demo, object_name: str, T_cam_obj: np.ndarray, object_args) -> None:
    scene_capture_cache = getattr(demo, "scene_capture_cache_ref", None)
    if not isinstance(scene_capture_cache, dict):
        return
    objects = scene_capture_cache.get("objects")
    if not isinstance(objects, dict):
        return
    normalized = normalize_object_name(object_name)
    if normalized is None:
        return
    old_entry = objects.get(normalized, {}) if isinstance(objects.get(normalized), dict) else {}
    objects[normalized] = {
        "object_name": normalized,
        "label": str(old_entry.get("label", getattr(object_args, "target_object_name", "") or normalized)),
        "score": float(old_entry.get("score", 1.0)),
        "box": np.asarray(old_entry.get("box", np.zeros(4, dtype=np.float32)), dtype=np.float32).reshape(4),
        "T_cam_obj": np.asarray(T_cam_obj, dtype=np.float32),
        "object_args": argparse.Namespace(**vars(object_args).copy()),
        "placed": False,
    }


def cache_successfully_placed_object_world_pose(demo, object_name: str, object_args) -> None:
    scene_capture_cache = getattr(demo, "scene_capture_cache_ref", None)
    if not isinstance(scene_capture_cache, dict):
        return
    objects = scene_capture_cache.get("objects")
    if not isinstance(objects, dict):
        return
    normalized = normalize_object_name(object_name)
    if normalized is None:
        return
    try:
        cached_pose = getattr(demo, "_scene_cache_placed_object_world_pose", None)
        if cached_pose is not None:
            T_world_obj = np.asarray(cached_pose, dtype=np.float32).reshape(4, 4)
        else:
            obj_pose = demo.base_env.obj.pose
            T_world_obj = pose_to_matrix(flatten_np(obj_pose.p)[:3], flatten_np(obj_pose.q)[:4]).astype(np.float32)
    except Exception as exc:
        print(f"[scene cache] failed to cache placed object {normalized}: {exc}")
        return
    old_entry = objects.get(normalized, {}) if isinstance(objects.get(normalized), dict) else {}
    objects[normalized] = {
        "object_name": normalized,
        "label": str(old_entry.get("label", getattr(object_args, "target_object_name", "") or normalized)),
        "score": float(old_entry.get("score", 1.0)),
        "box": np.asarray(old_entry.get("box", np.zeros(4, dtype=np.float32)), dtype=np.float32).reshape(4),
        "T_cam_obj": np.asarray(old_entry.get("T_cam_obj", np.eye(4, dtype=np.float32)), dtype=np.float32).reshape(4, 4),
        "T_world_obj": T_world_obj,
        "object_args": argparse.Namespace(**vars(object_args).copy()),
        "placed": True,
    }
    print(f"[scene cache] kept placed object {normalized} in the cached scene at world translation={np.round(T_world_obj[:3, 3], 6).tolist()}")


def apply_refined_target_pose_from_camera(demo, bridge_mod, T_cam_obj: np.ndarray, args) -> None:
    T_base_cam = getattr(demo, "T_base_cam", None)
    if T_base_cam is None:
        raise RuntimeError("T_base_cam is unavailable")
    bridge_mod.print_foundationpose_mapping_diagnostics(T_cam_obj, T_base_cam, demo.env, args, label="refined target")
    T_world_obj = bridge_mod.map_camera_pose_to_pick_world(T_cam_obj, T_base_cam, demo.env, args)
    bridge_mod.apply_pose_to_pick_object(demo.env, T_world_obj)
    demo.refresh_runtime_handles(rebuild_visual=False)
    lift_active_object_above_table_if_needed(
        demo,
        args,
        min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
    )
    if bool(getattr(args, "freeze_active_object_before_grasp", True)):
        set_pregrasp_object_freeze(demo, True)
        refresh_frozen_active_object_pose(demo)
    update_scene_capture_cache_object(demo, getattr(args, "object_name", None), T_cam_obj, args)
    print(f"[foundationpose refine] updated active object {normalize_object_name(getattr(args, 'object_name', None))}")


def apply_refined_scene_obstacle_pose_from_camera(demo, bridge_mod, object_name: str, T_cam_obj: np.ndarray, object_args) -> None:
    T_base_cam = getattr(demo, "T_base_cam", None)
    if T_base_cam is None:
        raise RuntimeError("T_base_cam is unavailable")
    normalized = normalize_object_name(object_name)
    if normalized is None:
        raise ValueError(f"Unknown obstacle name: {object_name}")
    scene_obstacles = list(getattr(demo, "scene_obstacles", []) or [])
    item = next((entry for entry in scene_obstacles if normalize_object_name(entry.get("object_name")) == normalized), None)
    if item is None:
        raise ValueError(f"Obstacle {normalized} is not present in the current scene")

    actor_name = str(item.get("actor_name", "") or "")
    actor = None
    for candidate in list(getattr(demo.env.unwrapped, "_scene_obstacle_actors", []) or []):
        if str(getattr(candidate, "name", "") or "") == actor_name:
            actor = candidate
            break
    if actor is None:
        raise RuntimeError(f"Scene obstacle actor {actor_name!r} was not found")

    if bool(getattr(object_args, "foundationpose_print_obstacle_mapping_diagnostics", False)):
        bridge_mod.print_foundationpose_mapping_diagnostics(
            T_cam_obj,
            T_base_cam,
            demo.env,
            object_args,
            label=f"refined scene obstacle {normalized}",
        )
    T_world_obj = bridge_mod.map_camera_pose_to_pick_world(T_cam_obj, T_base_cam, demo.env, object_args)
    pos = T_world_obj[:3, 3].astype(np.float32)
    quat = bridge_mod.mat2quat(T_world_obj[:3, :3]).astype(np.float32)
    actor.set_pose(Pose.create_from_pq(p=pos, q=quat))
    sync_scene_obstacle_planner_box_actor(demo.env, item, pos, quat)
    item["T_world_obj"] = np.asarray(T_world_obj, dtype=np.float32)
    try:
        if bool(item.get("planner_collision", False)):
            from mplib import collision_detection as mplib_cd

            planner_box_size = np.asarray(
                item.get("planner_box_size", item.get("visual_box_size")),
                dtype=np.float32,
            ).reshape(3)
            collision_object = mplib_cd.fcl.CollisionObject(
                mplib_cd.fcl.Box(planner_box_size.tolist()),
                pos.tolist(),
                quat.tolist(),
            )
            demo.planner.set_normal_object(actor_name, collision_object)
    except Exception as exc:
        print(f"[foundationpose refine] failed to refresh planner collision for {normalized}: {exc}")

    lift_scene_obstacles_above_table_if_needed(
        demo.env,
        demo,
        min_clearance=max(0.001, 0.5 * float(getattr(object_args, "min_object_center_z_margin", 0.0))),
    )
    update_scene_capture_cache_object(demo, normalized, T_cam_obj, object_args)
    print(f"[foundationpose refine] updated scene obstacle {normalized}")


def maybe_refine_foundationpose_scene_after_render(demo, bridge_mod, args) -> bool:
    if not bool(getattr(args, "foundationpose_refine_after_render", False)):
        return True
    if getattr(args, "render_mode", None) != "human":
        return True

    refinable_names = []
    target_name = normalize_object_name(getattr(args, "object_name", None))
    if target_name is not None:
        refinable_names.append(target_name)
    for item in list(getattr(demo, "scene_obstacles", []) or []):
        object_name = normalize_object_name(item.get("object_name"))
        if object_name is not None and object_name not in refinable_names:
            refinable_names.append(object_name)
    if not refinable_names:
        return True

    while True:
        print(f"[foundationpose refine] available objects: {refinable_names}")
        try:
            answer = input(
                "[foundationpose refine] Press Enter to continue, type object names separated by space/comma to re-register + live-track them, or type q to abort: "
            ).strip().lower()
        except EOFError:
            answer = "q"
        if answer == "":
            return True
        if answer.strip(" \t,，.。;；:：!?！？") == "":
            return True
        if answer == "q":
            return False
        try:
            selected_names = resolve_object_spec_name_list(answer.replace(",", " ").split(), available_names=refinable_names)
        except ValueError as exc:
            print(f"[foundationpose refine] {exc}")
            continue
        if not selected_names:
            print("[foundationpose refine] No valid object names were provided.")
            continue

        for object_name in selected_names:
            object_args = args if object_name == target_name else bridge_mod._make_object_specific_args(args, object_name)
            T_cam_obj = bridge_mod.interactive_refine_single_object_pose(
                getattr(demo, "foundationpose_runtime"),
                object_args,
                object_name=object_name,
                window_name=f"FoundationPose Refine: {object_name}",
            )
            if T_cam_obj is None:
                print(f"[foundationpose refine] skipped {object_name}")
                continue
            if object_name == target_name:
                apply_refined_target_pose_from_camera(demo, bridge_mod, T_cam_obj, args)
            else:
                apply_refined_scene_obstacle_pose_from_camera(demo, bridge_mod, object_name, T_cam_obj, object_args)
            if getattr(args, "render_mode", None) == "human":
                bridge_mod.render_preview(demo.env, repeats=5)


def create_demo(args, bridge_mod, planner_mod, scene_capture_cache=None):
    args.env_id = bridge_mod.ensure_pick_jiaobang_env_registered(args.env_id, args.extra_maniskill_package_root)
    fp_rt, T_base_cam, T_cam_obj, scene_obstacles = capture_or_reuse_foundationpose_scene(
        args,
        bridge_mod,
        scene_capture_cache=scene_capture_cache,
    )

    env = gym.make(
        args.env_id,
        robot_uids="RM75",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        max_episode_steps=args.max_episode_steps,
        object_asset_path=args.sim_asset_file,
        object_scale=args.sim_asset_scale,
    )

    initial_obs, initial_info = env.reset(seed=args.seed)
    bridge_mod.apply_pick_object_physics_profile(env, args)
    hide_env_goal_visual(env)
    bridge_mod.print_foundationpose_mapping_diagnostics(T_cam_obj, T_base_cam, env, args, label="target")
    T_world_obj = bridge_mod.map_camera_pose_to_pick_world(T_cam_obj, T_base_cam, env, args)
    print("Mapped sim object translation:", np.round(T_world_obj[:3, 3], 6).tolist())
    bridge_mod.apply_pose_to_pick_object(env, T_world_obj)
    if args.render_mode == "human":
        bridge_mod.render_preview(env, repeats=5)

    sim_urdf_path = planner_mod.find_existing_urdf(args.urdf_path)
    planning_urdf_path, srdf_path = planner_mod.resolve_planning_artifact_paths(sim_urdf_path, args)
    Path(planning_urdf_path).parent.mkdir(parents=True, exist_ok=True)
    Path(srdf_path).parent.mkdir(parents=True, exist_ok=True)
    planner_mod.generate_near_collision_free_planning_urdf(sim_urdf_path, planning_urdf_path)
    if args.srdf_path is None:
        planner_mod.write_permissive_srdf(planning_urdf_path, srdf_path)

    print("Sim URDF     :", sim_urdf_path)
    print("Planning URDF:", planning_urdf_path)
    print("SRDF         :", srdf_path)

    demo = planner_mod.RM75JiaobangPickMove(env, planning_urdf_path, srdf_path, args)
    demo.last_obs = initial_obs
    demo.last_info = initial_info if isinstance(initial_info, dict) else {}
    demo.last_terminated = False
    demo.last_truncated = False
    demo.foundationpose_runtime = fp_rt
    demo.T_base_cam = T_base_cam
    demo.scene_capture_cache_ref = scene_capture_cache
    demo.refresh_runtime_handles(rebuild_visual=False)
    demo.attached_box_visual = None
    demo.attached_box_size = None
    demo.attached_box_pose_tcp = None
    demo._attached_box_visual_visible = False
    demo._attached_object_visual_active = False
    demo._attached_object_collision_sphere_specs = []
    demo._freeze_active_object_before_grasp = False
    demo._frozen_active_object_pose = None
    snap_active_object_flat_on_table_if_needed(
        demo,
        args,
        min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
    )
    lift_active_object_above_table_if_needed(
        demo,
        args,
        min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
    )
    if bool(getattr(args, "freeze_active_object_before_grasp", True)):
        set_pregrasp_object_freeze(demo, True)
        refresh_frozen_active_object_pose(demo)

    applied_scene_obstacles = register_scene_obstacles(env, demo, bridge_mod, T_base_cam, scene_obstacles, args)
    snap_scene_obstacles_flat_on_table_if_needed(
        env,
        demo,
        args,
        min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
    )
    lift_scene_obstacles_above_table_if_needed(
        env,
        demo,
        min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
    )
    update_robot_collision_sphere_visuals(demo, args)
    print(f"Scene obstacles registered: {len(applied_scene_obstacles)}")
    if args.render_mode == "human" and applied_scene_obstacles:
        bridge_mod.render_preview(env, repeats=5)
    if not maybe_refine_foundationpose_scene_after_render(demo, bridge_mod, args):
        raise SystemExit(0)
    return env, demo


def remove_picked_object_from_scene_cache(scene_capture_cache, object_name: str | None) -> None:
    if not isinstance(scene_capture_cache, dict):
        return
    objects = scene_capture_cache.get("objects")
    if not isinstance(objects, dict):
        return
    normalized = normalize_object_name(object_name)
    if normalized is None:
        return
    if normalized in objects:
        objects.pop(normalized, None)
        print(f"[foundationpose] removed {normalized} from the cached scene after a successful pick-place")


def execute_stage(demo, bridge_mod, real_exec: RealmanJointExecutor | None, label: str, pose, q_target: np.ndarray, gripper_pos: float, args):
    q_target = np.asarray(q_target, dtype=np.float32).reshape(-1)[:7]
    if not confirm_planned_motion_or_skip(demo, bridge_mod, label, pose, q_target, args):
        print(f"[abort] user cancelled before executing {label}")
        return False, None
    if real_exec is not None:
        q_current = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
        if not _validate_real_waypoint_segment(demo, q_current, q_target, use_attach=False, label=label):
            return False, None
        q_sent = real_exec.move_linear(
            q_target,
            gripper_pos=gripper_pos,
            max_delta_per_step=args.real_max_delta_per_step,
            hz=args.real_control_hz,
            hold_steps=args.real_hold_steps,
            label=label,
            shadow_callback=lambda q_cmd: _render_real_shadow_step(demo, bridge_mod, args, q_cmd),
        )
        _render_real_shadow_step(demo, bridge_mod, args, q_sent)
    else:
        print(f"[dry-run] skipped real execution for {label}")
        sync_demo_arm_qpos(demo, q_target)
    return True, q_target


def confirm_joint_motion(demo, bridge_mod, label: str, q_target: np.ndarray, args) -> bool:
    q_target = np.asarray(q_target, dtype=np.float32).reshape(-1)[:7]
    if bool(getattr(args, "_skip_remaining_step_confirms_in_object", False)):
        if args.render_mode == "human":
            bridge_mod.render_preview(demo.env, repeats=3)
        print(f"\n[planned {label}]")
        print("planned arm q:", np.round(q_target, 6))
        print("[preview] joint-path confirmation keeps the current simulated state to avoid teleporting the grasped object")
        print(f"[confirm] auto-approved {label} because single-confirm-per-object is active")
        return True
    if args.render_mode == "human":
        bridge_mod.render_preview(demo.env, repeats=3)

    print(f"\n[planned {label}]")
    print("planned arm q:", np.round(q_target, 6))
    print("[preview] joint-path confirmation keeps the current simulated state to avoid teleporting the grasped object")

    if args.auto_execute:
        return True

    answer = bridge_mod.prompt_with_live_render(
        f"[confirm] {label} planned. Press Enter to execute, or type q then Enter to abort: ",
        demo.env,
        args,
    )

    return answer not in {"q", "quit", "n", "no"}


def execute_joint_stage(demo, bridge_mod, real_exec: RealmanJointExecutor | None, label: str, q_target: np.ndarray, gripper_pos: float, args):
    q_target = np.asarray(q_target, dtype=np.float32).reshape(-1)[:7]
    if not confirm_joint_motion(demo, bridge_mod, label, q_target, args):
        print(f"[abort] user cancelled before executing {label}")
        return False, None
    if real_exec is not None:
        q_sent = real_exec.move_linear(
            q_target,
            gripper_pos=gripper_pos,
            max_delta_per_step=args.real_max_delta_per_step,
            hz=args.real_control_hz,
            hold_steps=args.real_hold_steps,
            label=label,
        )
        sync_demo_arm_qpos(demo, q_sent)
    else:
        print(f"[dry-run] skipped real execution for {label}")
        sync_demo_arm_qpos(demo, q_target)
    return True, q_target


def inspect_failed_pose(demo, bridge_mod, label: str, args, *, pose=None, q_target=None, gripper_closed=None, use_attach: bool = False):
    if q_target is not None:
        sync_demo_arm_qpos(demo, q_target)
    if gripper_closed is not None:
        sync_demo_gripper_state(demo, bool(gripper_closed), steps=8)
    if pose is not None:
        try:
            demo.preview_target_pose(pose)
        except Exception:
            pass
    print_failure_diagnostics(demo, args, label, q_target=q_target, use_attach=use_attach)
    if args.render_mode == "human":
        bridge_mod.render_preview(demo.env, repeats=12)
    if args.auto_execute:
        return
    bridge_mod.prompt_with_live_render(
        f"[inspect] {label} failed. Viewer is showing the failed target state. Press Enter to end this cycle: ",
        demo.env,
        args,
    )


def run_real_episode(demo, bridge_mod, real_exec: RealmanJointExecutor | None, args) -> bool:
    print("\n[episode] planning from FoundationPose-initialized object pose")
    args._skip_remaining_step_confirms_in_object = False

    start_q = demo.current_arm_qpos()
    if real_exec is not None and bool(getattr(args, "single_confirm_per_object", False)):
        if not begin_single_confirm_window_for_object(demo, bridge_mod, args):
            print("[abort] user cancelled before executing this object's pick-place sequence")
            return False
    if real_exec is not None:
        print("\n[real robot setup]")
        if args.render_mode == "human":
            bridge_mod.render_preview(demo.env, repeats=5)
        ok, q_sent = align_real_robot_to_sim_start(demo, bridge_mod, real_exec, start_q, args)
        if not ok:
            print("[abort] failed to align the real robot to the simulation start pose")
            return False
        sync_demo_arm_qpos(demo, q_sent if q_sent is not None else start_q)
    else:
        print("\n[dry-run] --execute-real was not provided, so motions will only be planned and previewed")

    lift_active_object_above_table_if_needed(
        demo,
        args,
        min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
    )
    raw_grasp_pose = demo.build_topdown_grasp_pose()
    grasp_pose_variants = build_grasp_pose_variants(demo, raw_grasp_pose, args)
    selected_grasp_variant_label = None
    grasp_pose = None
    pregrasp_pose = None
    last_attempted_pregrasp_pose = None
    for grasp_variant_label, grasp_variant_pose in grasp_pose_variants:
        current_grasp_pose = grasp_variant_pose
        current_pregrasp_pose = demo.build_pregrasp_pose(current_grasp_pose)
        current_grasp_pose, current_pregrasp_pose, geometry_grasp_raise = enforce_topdown_grasp_insertion_limit(
            demo,
            args,
            current_grasp_pose,
            current_pregrasp_pose,
        )
        if geometry_grasp_raise > 0:
            print(
                f"[safety] raised grasp/pregrasp TCP z by {geometry_grasp_raise:.4f} m "
                f"to satisfy topdown_grasp_max_insertion_depth={args.topdown_grasp_max_insertion_depth:.4f}"
            )
        current_grasp_pose, current_pregrasp_pose, grasp_tcp_raise = enforce_min_grasp_tcp_z(
            current_grasp_pose,
            current_pregrasp_pose,
            args.min_grasp_tcp_z,
        )
        if grasp_tcp_raise > 0:
            print(
                f"[safety] raised grasp/pregrasp TCP z by {grasp_tcp_raise:.4f} m "
                f"to satisfy min_grasp_tcp_z={args.min_grasp_tcp_z:.4f}"
            )

        if grasp_variant_label != "grasp":
            print(f"[planner] trying pregrasp/grasp pose variant: {grasp_variant_label}")
        print("\n[poses]")
        print("object p:", np.round(demo.get_obj_pose()[0], 6), "object q:", np.round(demo.get_obj_pose()[1], 6))
        print("tcp grasp p:", np.round(flatten_np(current_grasp_pose.p)[:3], 6), "q:", np.round(flatten_np(current_grasp_pose.q)[:4], 6))
        print("tcp pregrasp p:", np.round(flatten_np(current_pregrasp_pose.p)[:3], 6), "q:", np.round(flatten_np(current_pregrasp_pose.q)[:4], 6))

        print("\n[move to pregrasp]")
        demo.preview_target_pose(current_pregrasp_pose)
        if args.render_mode == "human":
            bridge_mod.render_preview(demo.env, repeats=10)
        q_pre_path = plan_pose_path(
            demo,
            current_pregrasp_pose,
            variant_name=args.variant,
            label="pregrasp" if grasp_variant_label == "grasp" else f"pregrasp_{grasp_variant_label}",
        )
        last_attempted_pregrasp_pose = current_pregrasp_pose
        if q_pre_path is not None:
            ok, _ = execute_pose_path_stage(
                demo,
                bridge_mod,
                real_exec,
                "pregrasp" if grasp_variant_label == "grasp" else f"pregrasp_{grasp_variant_label}",
                current_pregrasp_pose,
                q_pre_path,
                args.real_gripper_open,
                args,
            )
            if not ok:
                ok, _ = retry_pregrasp_after_escape(
                    demo,
                    bridge_mod,
                    real_exec,
                    "pregrasp" if grasp_variant_label == "grasp" else f"pregrasp_{grasp_variant_label}",
                    current_pregrasp_pose,
                    args.real_gripper_open,
                    args,
                )
                if not ok:
                    return False
            grasp_pose = current_grasp_pose
            pregrasp_pose = current_pregrasp_pose
            selected_grasp_variant_label = grasp_variant_label
            break

        print("[planner] pregrasp pose-path planning failed; falling back to terminal joint target")
        q_pre = demo.plan_terminal_q(current_pregrasp_pose, variant_name=args.variant)
        if q_pre is None:
            print(
                f"[planner] pregrasp IK failed for "
                f"{'default top-down pose' if grasp_variant_label == 'grasp' else grasp_variant_label}"
            )
            continue
        ok, _ = execute_stage(
            demo,
            bridge_mod,
            real_exec,
            "pregrasp" if grasp_variant_label == "grasp" else f"pregrasp_{grasp_variant_label}",
            current_pregrasp_pose,
            q_pre,
            args.real_gripper_open,
            args,
        )
        if not ok:
            ok, _ = retry_pregrasp_after_escape(
                demo,
                bridge_mod,
                real_exec,
                "pregrasp" if grasp_variant_label == "grasp" else f"pregrasp_{grasp_variant_label}",
                current_pregrasp_pose,
                args.real_gripper_open,
                args,
            )
            if not ok:
                return False
        grasp_pose = current_grasp_pose
        pregrasp_pose = current_pregrasp_pose
        selected_grasp_variant_label = grasp_variant_label
        break

    if grasp_pose is None or pregrasp_pose is None:
        print("[FAIL] pregrasp planning failed")
        inspect_failed_pose(
            demo,
            bridge_mod,
            "pregrasp",
            args,
            pose=last_attempted_pregrasp_pose if last_attempted_pregrasp_pose is not None else raw_grasp_pose,
            gripper_closed=False,
        )
        return False
    retreat_pregrasp_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]

    print("\n[move to grasp]")
    grasp_plan = plan_grasp_pose_with_fallbacks(
        demo,
        bridge_mod,
        grasp_pose,
        args,
        base_label="grasp" if selected_grasp_variant_label in (None, "grasp") else selected_grasp_variant_label,
    )
    if grasp_plan is None:
        print("[FAIL] grasp planning failed")
        inspect_failed_pose(demo, bridge_mod, "grasp", args, pose=grasp_pose, gripper_closed=False)
        return False
    grasp_label = grasp_plan.label
    selected_grasp_pose = grasp_plan.final_grasp_pose
    if grasp_label != "grasp":
        print(f"[planner] selected grasp fallback approach: {grasp_label}")
    retreat_pregrasp_pose = demo.build_pregrasp_pose(selected_grasp_pose)
    _, retreat_pregrasp_pose, _ = enforce_min_grasp_tcp_z(
        selected_grasp_pose,
        retreat_pregrasp_pose,
        args.min_grasp_tcp_z,
    )

    if bool(getattr(grasp_plan, "require_runtime_true_grasp_planning", False)):
        approach_path = plan_pose_path(
            demo,
            grasp_plan.approach_pose,
            variant_name=args.variant,
            label=grasp_label or "grasp",
        )
        if approach_path is not None:
            ok, _ = execute_pose_path_stage(
                demo,
                bridge_mod,
                real_exec,
                grasp_label or "grasp",
                grasp_plan.approach_pose,
                approach_path,
                args.real_gripper_open,
                args,
            )
        else:
            print(f"[planner] {grasp_label or 'grasp'} pose-path planning failed; falling back to terminal joint target")
            ok, _ = execute_stage(
                demo,
                bridge_mod,
                real_exec,
                grasp_label or "grasp",
                grasp_plan.approach_pose,
                grasp_plan.approach_q,
                args.real_gripper_open,
                args,
            )
        if not ok:
            return False

        q_grasp_path = plan_pose_path(
            demo,
            selected_grasp_pose,
            variant_name=args.variant,
            label=f"{grasp_label}_to_true_grasp",
        )
        if q_grasp_path is not None:
            ok, _ = execute_pose_path_stage(
                demo,
                bridge_mod,
                real_exec,
                f"{grasp_label}_to_true_grasp",
                selected_grasp_pose,
                q_grasp_path,
                args.real_gripper_open,
                args,
            )
        else:
            print(f"[planner] {grasp_label}_to_true_grasp pose-path planning failed after executing the fallback approach; trying direct terminal joint target")
            q_grasp = demo.plan_terminal_q(selected_grasp_pose, variant_name=args.variant)
            if q_grasp is None:
                print(f"[FAIL] {grasp_label}_to_true_grasp planning failed from the executed fallback approach")
                inspect_failed_pose(demo, bridge_mod, "grasp", args, pose=selected_grasp_pose, gripper_closed=False)
                return False
            ok, _ = execute_stage(
                demo,
                bridge_mod,
                real_exec,
                f"{grasp_label}_to_true_grasp",
                selected_grasp_pose,
                q_grasp,
                args.real_gripper_open,
                args,
            )
    elif grasp_plan.final_grasp_path is not None:
        approach_path = plan_pose_path(
            demo,
            grasp_plan.approach_pose,
            variant_name=args.variant,
            label=grasp_label or "grasp",
        )
        if approach_path is not None:
            ok, _ = execute_pose_path_stage(
                demo,
                bridge_mod,
                real_exec,
                grasp_label or "grasp",
                grasp_plan.approach_pose,
                approach_path,
                args.real_gripper_open,
                args,
            )
        else:
            print(f"[planner] {grasp_label or 'grasp'} pose-path planning failed; falling back to terminal joint target")
            ok, _ = execute_stage(
                demo,
                bridge_mod,
                real_exec,
                grasp_label or "grasp",
                grasp_plan.approach_pose,
                grasp_plan.approach_q,
                args.real_gripper_open,
                args,
            )
        if not ok:
            return False
        ok, _ = execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            f"{grasp_label}_to_true_grasp",
            selected_grasp_pose,
            grasp_plan.final_grasp_path,
            args.real_gripper_open,
            args,
        )
    else:
        q_grasp_path = plan_pose_path(
            demo,
            selected_grasp_pose,
            variant_name=args.variant,
            label=grasp_label or "grasp",
        )
        if q_grasp_path is not None:
            ok, _ = execute_pose_path_stage(
                demo,
                bridge_mod,
                real_exec,
                grasp_label or "grasp",
                selected_grasp_pose,
                q_grasp_path,
                args.real_gripper_open,
                args,
            )
        else:
            print(f"[planner] {grasp_label or 'grasp'} pose-path planning failed; falling back to terminal joint target")
            ok, _ = execute_stage(
                demo,
                bridge_mod,
                real_exec,
                grasp_label or "grasp",
                selected_grasp_pose,
                grasp_plan.approach_q,
                args.real_gripper_open,
                args,
            )
    if not ok:
        return False
    print("\n[close gripper]")
    if not confirm_simple_action("close the real gripper", args, bridge_mod=bridge_mod, env=demo.env, repeats=6):
        print("[abort] user cancelled before closing the real gripper")
        return False
    if real_exec is not None:
        real_exec.set_gripper(args.real_gripper_close)
        sync_demo_gripper_state(demo, closed=True, steps=4)
        set_pregrasp_object_freeze(demo, False)
        gripper_pos, blocked = real_gripper_blocked_after_close(
            real_exec,
            close_cmd=args.real_gripper_close,
            blocked_margin=args.real_gripper_blocked_margin,
        )
        if gripper_pos is not None and blocked is not None:
            print(
                f"[real] gripper.pos after close: {gripper_pos:.4f} "
                f"blocked_before_full_close={blocked}"
            )
    else:
        print("[dry-run] skipped real gripper close")
        sync_demo_gripper_state(demo, closed=True, steps=4)
        set_pregrasp_object_freeze(demo, False)
    lift_active_object_above_table_if_needed(
        demo,
        args,
        min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
    )

    if args.skip_goal_motion:
        print("[done] skipped goal motion as requested")
        return True

    if args.use_sim_goal_motion:
        print("\n[move to goal]")
        goal_tcp_pose, q_goal = demo.refresh_goal_and_plan(announce=True)
        if q_goal is None:
            print("[FAIL] goal planning failed")
            inspect_failed_pose(demo, bridge_mod, "goal", args, pose=goal_tcp_pose, gripper_closed=True, use_attach=True)
            return False
        ok, _ = execute_stage(demo, bridge_mod, real_exec, "goal", goal_tcp_pose, q_goal, args.real_gripper_close, args)
        if not ok:
            return False
        print("[done] completed one planned goal motion on the current sampled goal pose")
        return True
    print("\n[move to fixed goal joints]")
    q_goal = np.deg2rad(np.asarray(args.fixed_goal_joints_deg, dtype=np.float32))
    target_box_size = get_asset_box_size(args.sim_asset_file, args.sim_asset_scale)
    attached_box_scale = float(np.clip(args.transport_attached_box_scale, 0.5, 2.0))
    attached_box_size = np.maximum(target_box_size * attached_box_scale, 1e-4).astype(np.float32)
    fixed_goal_use_attach = not bool(args.fixed_goal_no_attach)
    print(
        "[planner] fixed_goal raw target box size:",
        np.round(target_box_size, 6),
        f"(asset={Path(str(args.sim_asset_file)).name}, scale={float(args.sim_asset_scale):.6f})",
    )
    print(
        "[planner] fixed_goal attached box size:",
        np.round(attached_box_size, 6),
        f"(scale={attached_box_scale:.3f})",
    )
    print(f"[planner] fixed_goal use_attach={fixed_goal_use_attach}, linear_fallback={not args.disable_linear_joint_fallback}")

    def _register_transport_attached_box():
        attach_pose_local = make_attached_box_pose(demo, attached_box_size)
        try:
            demo.planner.update_attached_box(attached_box_size.tolist(), attach_pose_local.tolist())
            setup_attached_box_visual(demo, demo.env, attached_box_size, attach_pose_local)
        except Exception as exc:
            print(f"[warn] failed to register attached target box for fixed-goal planning: {exc}")
        return attach_pose_local

    def _run_post_grasp_escape(lift_height: float, label: str, retreat_distance: float = 0.0) -> bool:
        if float(lift_height) <= 1e-6:
            return True
        print(f"\n[{label}]")
        lift_pose = make_lifted_tcp_pose(demo, lift_height, retreat_distance=retreat_distance)
        lift_path = plan_lift_path(
            demo,
            lift_pose,
            variant_name=args.variant,
            use_attach=fixed_goal_use_attach,
            label=label,
            planning_time=min(float(args.fixed_goal_planning_time), 3.0),
            rrt_range=float(args.fixed_goal_rrt_range),
            start_q=np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7],
            allow_pose_rrt_fallback=True,
            max_segment_joint_delta=0.35,
            max_segment_joint7_delta=0.80,
            max_segment_norm_delta=0.60,
        )
        if lift_path is None:
            print(f"[warn] {label} planning failed")
            return False
        ok, _ = execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            label,
            lift_pose,
            lift_path,
            args.real_gripper_close,
            args,
            use_attach=fixed_goal_use_attach,
        )
        if not ok:
            return False
        lift_active_object_above_table_if_needed(
            demo,
            args,
            min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
        )
        _register_transport_attached_box()
        return True

    def _plan_fixed_goal_from_current(label: str):
        q_start_transport_local = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
        print("[planner] starting fixed-goal planning directly from the current post-grasp arm q")
        if bool(getattr(args, "fixed_goal_rrt_probe_no_attach", False)) and bool(getattr(args, "debug", 0)):
            run_fixed_goal_rrt_probe_suite(
                demo,
                q_start_transport_local,
                q_goal,
                planning_time=args.fixed_goal_planning_time,
                rrt_range=args.fixed_goal_rrt_range,
            )
        return plan_joint_path(
            demo,
            q_goal,
            use_attach=fixed_goal_use_attach,
            label=label,
            planning_time=args.fixed_goal_planning_time,
            rrt_range=args.fixed_goal_rrt_range,
            start_q=q_start_transport_local,
            allow_linear_fallback=not args.disable_linear_joint_fallback,
            allow_reverse_rrt_fallback=True,
            prefer_reverse_rrt=True,
            allow_shortcut=not (real_exec is not None and fixed_goal_use_attach),
        )

    attach_pose = _register_transport_attached_box()

    pre_transport_lift_height = float(max(getattr(args, "pre_transport_lift_height", 0.0), 0.0))
    pre_transport_retreat_distance = float(max(getattr(args, "post_grasp_retreat_distance", 0.0), 0.0))
    if pre_transport_lift_height > 1e-6:
        if not _run_post_grasp_escape(
            pre_transport_lift_height,
            "post_grasp_escape",
            retreat_distance=pre_transport_retreat_distance,
        ):
            print("[warn] post-grasp escape planning failed; continuing without the escape motion")

    q_goal_path = _plan_fixed_goal_from_current("fixed_goal")
    lift_pose = None
    lift_path = None
    if q_goal_path is None:
        stronger_escape_total = max(
            float(getattr(args, "transport_lift_height", 0.0)),
            pre_transport_lift_height + 0.02,
            0.05,
        )
        stronger_escape_delta = max(stronger_escape_total - pre_transport_lift_height, 0.02)
        print(
            "[planner] direct fixed-goal transport planning failed; "
            f"trying stronger post-grasp escape (+{stronger_escape_delta:.3f} m)"
        )
        stronger_escape_retreat = max(pre_transport_retreat_distance, 0.03)
        if _run_post_grasp_escape(
            stronger_escape_delta,
            "post_grasp_escape_retry",
            retreat_distance=stronger_escape_retreat,
        ):
            q_goal_path = _plan_fixed_goal_from_current("fixed_goal_after_escape_retry")

    if q_goal_path is None:
        print("[FAIL] direct fixed-goal transport planning failed from the current post-grasp state")
        print_failure_diagnostics(demo, args, "fixed_goal_direct", q_target=q_goal, use_attach=fixed_goal_use_attach)
        return False

    ok, _ = execute_joint_path_stage(
        demo,
        bridge_mod,
        real_exec,
        "fixed_goal",
        q_goal_path,
        args.real_gripper_close,
        args,
        use_attach=fixed_goal_use_attach,
    )
    if not ok:
        return False

    print("\n[open gripper at fixed goal]")
    if not confirm_simple_action("open the real gripper at the fixed goal", args, bridge_mod=bridge_mod, env=demo.env, repeats=6):
        print("[abort] user cancelled before opening the real gripper at the fixed goal")
        return False
    if real_exec is not None:
        real_exec.set_gripper(args.real_gripper_open)
        sync_demo_gripper_state(demo, closed=False, steps=4)
    else:
        print("[dry-run] skipped real gripper open at fixed goal")
    demo._attached_box_visual_visible = False
    demo._attached_object_visual_active = False
    update_attached_box_visual(demo, visible=False)

    print("[done] completed one fixed post-grasp joint motion and opened the gripper")
    return True


def close_env_quietly(env):
    if env is None:
        return
    try:
        env.close()
    except Exception as exc:
        print("close warning:", exc)


def prompt_cycle_object_name(base_args, cycle_idx: int, *, available_names: list[str] | None = None, default_name: str | None = None) -> str | None:
    names = list(available_names) if available_names is not None else list_object_spec_names()
    if not names:
        raise ValueError("No object specs are defined in object_specs.py")

    default_name = normalize_object_name(default_name) or normalize_object_name(base_args.object_name) or names[0]
    print(f"\n[cycle {cycle_idx}] available object specs:")
    for key in names:
        spec = get_object_spec(key)
        alias = "" if spec is None or normalize_object_name(spec.name) == key else f" (alias: {spec.name})"
        print(f"- {key}{alias}")

    while True:
        try:
            answer = input(f"[select] object for cycle {cycle_idx} (default: {default_name}, q to abort): ").strip()
        except EOFError:
            return None
        if not answer:
            return default_name
        if answer.lower() in {"q", "quit", "exit"}:
            return None

        normalized = normalize_object_name(answer)
        if normalized in names:
            return normalized

        for key in names:
            spec = get_object_spec(key)
            if spec is not None and normalize_object_name(spec.name) == normalized:
                return key

        print(f"[invalid] unknown object spec: {answer}")


def resolve_object_spec_name(token: str | None, *, available_names: list[str] | None = None) -> str | None:
    normalized = normalize_object_name(token)
    if normalized is None:
        return None
    names = list(available_names) if available_names is not None else list_object_spec_names()
    if normalized in names:
        return normalized
    for key in names:
        spec = get_object_spec(key)
        if spec is not None and normalize_object_name(spec.name) == normalized:
            return key
    return None


def resolve_object_spec_name_list(tokens, *, available_names: list[str] | None = None) -> list[str]:
    names = list(available_names) if available_names is not None else list_object_spec_names()
    resolved = []
    invalid = []
    for token in list(tokens or []):
        key = resolve_object_spec_name(token, available_names=names)
        if key is None:
            invalid.append(str(token))
            continue
        if key not in resolved:
            resolved.append(key)
    if invalid:
        raise ValueError(f"Unknown object spec(s): {', '.join(invalid)}")
    return resolved


def list_cached_scene_object_names(scene_capture_cache) -> list[str]:
    if not isinstance(scene_capture_cache, dict):
        return []
    objects = scene_capture_cache.get("objects")
    if not isinstance(objects, dict):
        return []
    names = [normalize_object_name(name) for name in objects.keys()]
    return sorted(name for name in names if name is not None)


def prompt_cycle_obstacle_names(base_args, cycle_idx: int, target_name: str) -> list[str] | None:
    target_name = normalize_object_name(target_name)
    names = [name for name in list_object_spec_names() if name != target_name]
    if not names:
        print(f"[cycle {cycle_idx}] no other object specs available for obstacles")
        return []

    print(f"\n[cycle {cycle_idx}] available obstacle specs (excluding target {target_name}):")
    for key in names:
        spec = get_object_spec(key)
        alias = "" if spec is None or normalize_object_name(spec.name) == key else f" (alias: {spec.name})"
        print(f"- {key}{alias}")

    while True:
        try:
            answer = input(f"[select] obstacles for cycle {cycle_idx} (comma-separated, empty for none, all for all, q to abort): ").strip()
        except EOFError:
            return None
        if not answer:
            return []
        lowered = answer.lower()
        if lowered in {"q", "quit", "exit"}:
            return None
        if lowered == "all":
            return names

        selected = []
        invalid = []
        for token in [part.strip() for part in answer.split(',') if part.strip()]:
            normalized = normalize_object_name(token)
            resolved = None
            if normalized in names:
                resolved = normalized
            else:
                for key in names:
                    spec = get_object_spec(key)
                    if spec is not None and normalize_object_name(spec.name) == normalized:
                        resolved = key
                        break
            if resolved is None:
                invalid.append(token)
                continue
            if resolved not in selected:
                selected.append(resolved)

        if invalid:
            print(f"[invalid] unknown obstacle spec(s): {', '.join(invalid)}")
            continue
        return selected


def make_cycle_args(base_args, object_name: str):
    cycle_args = argparse.Namespace(**vars(base_args).copy())
    cycle_args.object_name = object_name
    spec = apply_object_spec_defaults(cycle_args)
    if spec is None:
        raise ValueError(f"Unknown object spec: {object_name}")
    return cycle_args, spec


def main():
    args = parse_args()
    if int(args.repeat_count) < 1:
        raise ValueError("--repeat-count must be >= 1")
    maybe_print_and_exit_object_specs(args)
    base_args = argparse.Namespace(**vars(args).copy())
    base_args.object_name = resolve_object_spec_name(base_args.object_name) if base_args.object_name else None
    if base_args.selected_obstacle_object_names is not None:
        base_args.selected_obstacle_object_names = resolve_object_spec_name_list(base_args.selected_obstacle_object_names)
    cycle_object_sequence = resolve_object_spec_name_list(base_args.cycle_object_names) if base_args.cycle_object_names else []
    if cycle_object_sequence and not base_args.repeat_forever:
        base_args.repeat_count = max(int(base_args.repeat_count), len(cycle_object_sequence))
    bridge_mod = load_module_from_path("jiaobang_fp_bridge", args.bridge_script_path)
    planner_mod = load_module_from_path("jiaobang_planner_impl", args.pick_script_path)

    print(f"Using bridge script: {Path(args.bridge_script_path).resolve()}")
    print(f"Using planner script: {Path(args.pick_script_path).resolve()}")
    print(f"Using camera extrinsic from: {args.camera_extrinsic_opencv_path}")
    if args.use_direct_camera_extrinsic:
        print("Using camera extrinsic convention: T_base_obj = T_base_cam @ T_cam_obj")
    else:
        print("Using camera extrinsic convention: T_base_obj = inv(camera_extrinsic_opencv) @ T_cam_obj")
    if args.use_sim_goal_motion:
        print("Post-grasp goal mode: simulation-sampled goal")
    else:
        print(f"Post-grasp goal mode: fixed joint goal (deg) = {args.fixed_goal_joints_deg}")
    if args.repeat_forever:
        print("Repeat mode: forever")
    else:
        print(f"Repeat mode: {args.repeat_count} cycle(s)")

    real_exec = None
    final_ok = True
    cycle_idx = 0
    scene_capture_cache: dict | None = {} if bool(getattr(args, "reuse_foundationpose_scene_across_cycles", True)) else None
    try:
        if args.execute_real:
            real_exec = RealmanJointExecutor(args)
            if args.reset_real_before_start:
                print("\n[real robot pre-reset]")
                if not confirm_simple_action("reset the real robot to its hardware home pose before FoundationPose initialization", args):
                    print("[abort] user cancelled before the pre-FoundationPose real robot reset")
                    return
                real_exec.reset_robot(gripper_pos=args.real_gripper_open)

        while True:
            cycle_idx += 1
            env = None
            ok = False
            cached_scene_names = list_cached_scene_object_names(scene_capture_cache)
            if cycle_idx <= len(cycle_object_sequence):
                selected_name = cycle_object_sequence[cycle_idx - 1]
                print(f"\n[cycle {cycle_idx}] using CLI target object: {selected_name}")
            elif cycle_idx == 1 and base_args.object_name is not None:
                selected_name = base_args.object_name
                print(f"\n[cycle {cycle_idx}] using CLI target object: {selected_name}")
            elif cycle_idx > 1 and cached_scene_names:
                print(f"\n[cycle {cycle_idx}] reusing the cached tabletop scene; choose the next target from the remaining objects")
                selected_name = prompt_cycle_object_name(base_args, cycle_idx, available_names=cached_scene_names, default_name=cached_scene_names[0])
            else:
                selected_name = prompt_cycle_object_name(base_args, cycle_idx)
            if selected_name is None:
                final_ok = False
                print(f"[abort] user cancelled object selection for cycle {cycle_idx}")
                break
            cycle_args, spec = make_cycle_args(base_args, selected_name)
            if cycle_idx > 1 and cached_scene_names:
                selected_obstacles = [name for name in cached_scene_names if name != selected_name]
                print(f"[cycle {cycle_idx}] auto-selected obstacles from the cached scene: {selected_obstacles}")
            elif cycle_idx == 1 and base_args.selected_obstacle_object_names is not None:
                selected_obstacles = [name for name in base_args.selected_obstacle_object_names if name != selected_name]
                print(f"[cycle {cycle_idx}] using CLI obstacle objects: {selected_obstacles}")
            else:
                selected_obstacles = prompt_cycle_obstacle_names(cycle_args, cycle_idx, selected_name)
                if selected_obstacles is None:
                    final_ok = False
                    print(f"[abort] user cancelled obstacle selection for cycle {cycle_idx}")
                    break
            cycle_args.selected_obstacle_object_names = list(selected_obstacles)
            print(f"\n[cycle {cycle_idx}] using object spec: {spec.name}")
            print(f"[cycle {cycle_idx}] mesh file: {cycle_args.mesh_file}")
            print(f"[cycle {cycle_idx}] simulation asset file: {cycle_args.sim_asset_file}")
            print(f"[cycle {cycle_idx}] simulation asset scale: {cycle_args.sim_asset_scale}")
            print(f"[cycle {cycle_idx}] GroundingDINO target: {cycle_args.target_object_name}")
            print(f"[cycle {cycle_idx}] selected obstacles: {cycle_args.selected_obstacle_object_names}")
            print(f"\n================ cycle {cycle_idx} ================")
            try:
                env, demo = create_demo(cycle_args, bridge_mod, planner_mod, scene_capture_cache=scene_capture_cache)
                ok = run_real_episode(demo, bridge_mod, real_exec, cycle_args)
            finally:
                close_env_quietly(env)
                gc.collect()
            print(f"\ncycle {cycle_idx} success = {ok}")
            if not ok:
                final_ok = False
                break
            remove_picked_object_from_scene_cache(scene_capture_cache, cycle_args.object_name)
            if not args.repeat_forever and cycle_idx >= int(args.repeat_count):
                break
            if real_exec is not None:
                real_exec.set_gripper(args.real_gripper_open)
                print(f"\n[cycle {cycle_idx}] keeping the current real robot pose; the next cycle will continue planning from the current arm configuration")
            print(f"[cycle {cycle_idx}] ready for the next cycle")

        print("\nfinal success =", final_ok)
    finally:
        if real_exec is not None:
            real_exec.close()


if __name__ == "__main__":
    main()
