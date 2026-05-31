#!/usr/bin/env python3
"""临时可视化工具：在 ManiSkill 人眼窗口里显示 RM75 碰撞球。

功能:
- 强制挂载 portable maniskill env（用于你的 Two_finger_PickJiaobang-v1）
- 所有碰撞球绿色显示
- 目标索引（默认 1/11）红色显示
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import sapien
import torch
import yaml
from transforms3d.quaternions import quat2mat

from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


def _prepend_package_path(package: Any, path: Path) -> None:
    if not path.exists() or not hasattr(package, "__path__"):
        return
    path_str = str(path)
    existing = [str(item) for item in list(package.__path__)]
    existing = [item for item in existing if item != path_str]
    new_paths = [path_str, *existing]
    try:
        package.__path__[:] = new_paths
    except Exception:
        package.__path__ = new_paths


def _ensure_portable_maniskill_env(portable_root: Path):
    """Load the bundled portable ManiSkill task with same behavior as portable script."""
    if str(portable_root) not in sys.path:
        sys.path.insert(0, str(portable_root))
    maniskill_root = portable_root / "mani_skill"
    task_path = (
        portable_root
        / "mani_skill"
        / "envs"
        / "tasks"
        / "digital_twins"
        / "so101_arm_with_two_cameras"
        / "pick_jiaobang.py"
    )
    robots_pkg = importlib.import_module("mani_skill.agents.robots")
    dt_pkg = importlib.import_module("mani_skill.envs.tasks.digital_twins")
    _prepend_package_path(robots_pkg, maniskill_root / "agents" / "robots")
    _prepend_package_path(dt_pkg, maniskill_root / "envs" / "tasks" / "digital_twins")

    # Ensure portable RM75 implementation is used if base package differs.
    sys.modules.pop("mani_skill.agents.robots.realman", None)
    sys.modules.pop("mani_skill.agents.robots.realman.realman_with_gripper", None)
    importlib.import_module("mani_skill.agents.robots.realman")

    module_name = "jimu_portable_pick_jiaobang_env_tmp"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, str(task_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 portable 任务: {task_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if "Two_finger_PickJiaobang-v1" not in gym.envs.registry:
        raise RuntimeError("portable ManiSkill 任务未注册成功：Two_finger_PickJiaobang-v1")
    return module


def _flatten_np(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _build_collision_specs(curobo_cfg: Path, sphere_indices: set[int]) -> list[dict]:
    with curobo_cfg.open("r", encoding="utf-8") as f:
        robot_cfg = yaml.safe_load(f) or {}
    kinematics = (((robot_cfg.get("robot_cfg") or {}).get("kinematics") or {})
                  )
    spheres_value = kinematics.get("collision_spheres")
    if spheres_value in (None, "", {}):
        return []
    spheres_path = Path(spheres_value)
    if not spheres_path.is_absolute():
        spheres_path = (curobo_cfg.parent / spheres_path).resolve()
    with spheres_path.open("r", encoding="utf-8") as f:
        sphere_data = yaml.safe_load(f) or {}
    collision_spheres = (sphere_data or {}).get("collision_spheres") or {}
    buffer_cfg = kinematics.get("collision_sphere_buffer", 0.0)

    specs: list[dict] = []
    for link_name, entries in collision_spheres.items():
        if not isinstance(entries, list):
            continue
        if isinstance(buffer_cfg, dict):
            link_buffer = float(buffer_cfg.get(link_name, 0.0) or 0.0)
        else:
            link_buffer = float(buffer_cfg or 0.0)
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            center = np.asarray(entry.get("center", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
            radius = float(entry.get("radius", 0.0) or 0.0) + link_buffer
            if radius <= 0.0:
                continue
            specs.append(
                {
                    "index": len(specs),
                    "link_name": str(link_name),
                    "center": center,
                    "radius": float(radius),
                    "actor_name": f"robot_collision_sphere__{link_name}__{idx:02d}",
                    "color": (1.0, 0.0, 0.0, 0.75) if len(specs) in sphere_indices else (0.2, 1.0, 0.2, 0.45),
                }
            )
    return specs


def _create_actor(env, radius: float, actor_name: str, color=(0.2, 1.0, 0.2, 0.45)):
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


def _get_demo_robot_links_map(robot_or_demo) -> dict[str, Any]:
    demo_env = robot_or_demo.robot if hasattr(robot_or_demo, "robot") else robot_or_demo
    links_map = getattr(demo_env, "links_map", None)
    if isinstance(links_map, dict) and links_map:
        return {str(k): v for k, v in links_map.items()}
    try:
        links = list(demo_env.get_links())
    except Exception:
        links = []
    result: dict[str, Any] = {}
    for link in links:
        try:
            name = str(getattr(link, "name", "") or "")
        except Exception:
            name = ""
        if name:
            result[name] = link
    return result


def _update_collision_sphere_poses(demo_env, specs, actors: dict[str, Any]) -> None:
    link_map = _get_demo_robot_links_map(demo_env.agent.robot)
    hidden_pos = np.asarray([0.0, 0.0, -10.0], dtype=np.float32)
    hidden_pose = Pose.create_from_pq(p=hidden_pos, q=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    identity_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for spec in specs:
        actor = actors.get(spec["actor_name"])
        if actor is None:
            continue
        link = link_map.get(spec["link_name"])
        if link is None:
            actor.set_pose(hidden_pose)
            continue
        link_pose = getattr(link, "pose", None)
        if link_pose is None:
            actor.set_pose(hidden_pose)
            continue
        link_p = _flatten_np(link_pose.p)[:3].astype(np.float32)
        link_q = _flatten_np(getattr(link_pose, "q", np.array([1.0, 0.0, 0.0, 0.0])))
        link_R = None

        # 1) 优先尝试从完整变换/旋转读取
        if hasattr(link_pose, "to_transformation_matrix"):
            try:
                tm = np.asarray(link_pose.to_transformation_matrix(), dtype=np.float64)
                if tm.shape == (4, 4):
                    link_R = tm[:3, :3].astype(np.float32)
            except Exception:
                link_R = None
        if link_R is None and hasattr(link_pose, "rotation"):
            try:
                rot = _flatten_np(getattr(link_pose, "rotation"))
                if rot.shape == (3, 3):
                    link_R = rot.astype(np.float32)
                elif rot.shape == (4, 4):
                    link_R = rot[:3, :3].astype(np.float32)
            except Exception:
                link_R = None

        # 2) 回退到四元数
        if link_R is None:
            if link_q.ndim > 0:
                q = link_q.reshape(-1)
            else:
                q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            if q.size == 16:
                q = q.reshape(4, 4)
                link_R = q[:3, :3].astype(np.float32)
            elif q.size == 9:
                link_R = q.reshape(3, 3).astype(np.float32)
            elif q.size >= 4:
                try:
                    link_R = quat2mat(q[:4].astype(np.float64)).astype(np.float32)
                except Exception:
                    link_R = None

        if link_R is None or link_R.shape != (3, 3):
            link_R = np.eye(3, dtype=np.float32)
        if np.isnan(link_R).any() or np.isinf(link_R).any():
            link_R = np.eye(3, dtype=np.float32)

        center_local = np.asarray(spec["center"], dtype=np.float32)
        if center_local.shape != (3,):
            center_local = center_local.reshape(-1)[:3]
        center_world = link_p + link_R @ center_local
        actor.set_pose(Pose.create_from_pq(p=center_world.astype(np.float32), q=identity_quat))


def _parse_deg_list(text: str) -> list[float] | None:
    if not text:
        return None
    vals = [v.strip() for v in text.replace(",", " ").split() if v.strip()]
    if not vals:
        return None
    out: list[float] = []
    for item in vals:
        out.append(float(item))
    return out


def _patch_env_for_no_collision_objects(mod) -> None:
    """Keep task loadable without coacd by skipping object collision hull generation."""

    def _load_scene_no_coacd(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        this_dir = os.path.dirname(os.path.abspath(__file__))
        glb_path = self.object_asset_path or os.path.join(this_dir, "jiaobang.glb")
        glb_path = os.path.expanduser(glb_path)
        if not os.path.isabs(glb_path):
            glb_path = os.path.join(this_dir, glb_path)
        if not os.path.exists(glb_path):
            raise FileNotFoundError(glb_path)
        scale_vec = np.asarray(self.object_scale_vec, dtype=float)

        self._objs = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            builder.set_scene_idxs([i])
            builder.add_visual_from_file(glb_path, scale=scale_vec.tolist())
            self._objs.append(builder.build(name=f"jiaobang-{i}"))
            self.remove_from_state_dict_registry(self._objs[-1])
        self.obj = Actor.merge(self._objs, name="jiaobang")
        self.add_to_state_dict_registry(self.obj)

        goal_green_tint = sapien.render.RenderMaterial(
            base_color=[0.2, 0.88, 0.2, 0.5],
            roughness=0.4,
            specular=0.2,
        )
        self._goal_objs = []
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
            self._goal_objs.append(
                builder.build_kinematic(name=f"goal_jiaobang-{i}")
            )
            self.remove_from_state_dict_registry(self._goal_objs[-1])
        self.goal_jiaobang = Actor.merge(self._goal_objs, name="goal_jiaobang")
        self.add_to_state_dict_registry(self.goal_jiaobang)

        self.rest_qpos = torch.tensor(
            [np.pi / 2, 0, 0, -np.pi / 2, 0, -np.pi / 2, np.pi / 3, 0, 0, 0, 0, 0, 0],
            device=self.device,
        )

    def _after_reconfigure_no_collision(self, options: dict):
        self.object_zs = []
        obj_local_aabb_min_list = []
        obj_local_aabb_max_list = []
        for obj in self._objs:
            box = None
            cmesh = obj.get_first_collision_mesh()
            if cmesh is not None and getattr(cmesh, "bounding_box", None) is not None:
                box = cmesh.bounding_box
            if box is None:
                self.object_zs.append(0.0)
                obj_local_aabb_min_list.append(np.zeros(3, dtype=np.float32))
                obj_local_aabb_max_list.append(np.zeros(3, dtype=np.float32))
                continue
            bounds = np.asarray(box.bounds, dtype=np.float32)
            self.object_zs.append(-float(bounds[0][2]))
            obj_local_aabb_min_list.append(bounds[0].astype(np.float32))
            obj_local_aabb_max_list.append(bounds[1].astype(np.float32))

        self.object_zs = common.to_tensor(self.object_zs, device=self.device)
        self.obj_local_aabb_min = common.to_tensor(np.stack(obj_local_aabb_min_list), device=self.device)
        self.obj_local_aabb_max = common.to_tensor(np.stack(obj_local_aabb_max_list), device=self.device)
        self.object_initial_height = torch.full((self.num_envs,), -1.0, device=self.device, dtype=torch.float32)
        self.obj_xy_shortest_edge_vector = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )

    mod.PickJiaobangEnv._load_scene = _load_scene_no_coacd
    mod.PickJiaobangEnv._after_reconfigure = _after_reconfigure_no_collision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=str, default="")
    parser.add_argument("--curobo-robot-cfg", type=str, default="")
    parser.add_argument("--highlight-indices", type=int, nargs="*", default=[1, 11])
    parser.add_argument("--object-scale", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=2000000)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--start-joints-deg", type=str, default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    if str(repo_root / "pick_jiaobang") not in sys.path:
        sys.path.insert(0, str(repo_root / "pick_jiaobang"))
    portable_root = repo_root / "Beta_demo-codex-v0.9" / "jimu_portable_repro" / "maniskill_env"
    if str(portable_root) not in sys.path:
        sys.path.insert(0, str(portable_root))

    default_object_asset = (
        repo_root
        / "Beta_demo-codex-v0.9"
        / "jimu_portable_repro"
        / "assets"
        / "red_jimu_cube.glb"
    )
    default_curobo_cfg = (
        repo_root
        / "pick_jiaobang"
        / "curobo_rm75_config"
        / "rm75.yml"
    )
    object_asset = Path(args.asset) if args.asset else default_object_asset
    curobo_cfg = Path(args.curobo_robot_cfg) if args.curobo_robot_cfg else default_curobo_cfg

    task_module = _ensure_portable_maniskill_env(portable_root)
    _patch_env_for_no_collision_objects(task_module)

    env = gym.make(
        "Two_finger_PickJiaobang-v1",
        robot_uids="RM75",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="human",
        object_asset_path=str(object_asset.resolve()),
        object_scale=args.object_scale,
    )

    obs, info = env.reset(seed=0)
    print("[tmp] env reset ok")

    curobo_cfg = curobo_cfg.resolve()
    highlight = set(int(i) for i in args.highlight_indices)
    specs = _build_collision_specs(curobo_cfg, highlight)
    print(f"[tmp] collision sphere count={len(specs)}; highlight={sorted(highlight)}")
    if not specs:
        print("[tmp] 未读取到碰撞球配置，退出")
        env.close()
        return

    actors: dict[str, Any] = {}
    created_red = 0
    created_green = 0
    for spec in specs:
        actor = _create_actor(env, spec["radius"], spec["actor_name"], color=spec["color"])
        actors[spec["actor_name"]] = actor
        if spec["index"] in highlight:
            created_red += 1
        else:
            created_green += 1
    print(f"[tmp] created spheres: red={created_red}, green={created_green}")

    # optional initial joint pose
    start_deg = _parse_deg_list(args.start_joints_deg)
    if start_deg:
        q = np.array(start_deg, dtype=np.float64) / 180.0 * np.pi
        # RM75 env typically uses first 7 arm + hand joints; pad/truncate as needed.
        q = np.asarray(q, dtype=np.float64)
        current_q = np.array(env.unwrapped.agent.robot.get_qpos(), dtype=np.float64).reshape(-1)
        q_full = current_q.copy()
        n = min(len(q), len(q_full))
        q_full[:n] = q[:n]
        env.unwrapped.agent.robot.set_qpos(q_full)
        print(f"[tmp] 已设置起始关节(deg): {start_deg}")

    # keep the initial pose; update visuals in loop.
    dt = 1.0 / max(1e-6, float(args.fps))
    action = np.zeros(env.action_space.shape, dtype=np.float64)
    try:
        for step_idx in range(args.steps):
            _update_collision_sphere_poses(env.unwrapped, specs, actors)
            env.step(action)
            env.render()
            if env.unwrapped.control_mode != "pd_joint_pos":
                pass
            time.sleep(dt)
            if step_idx % 200 == 0:
                print(f"[tmp] rendered {step_idx} step(s)")
    except KeyboardInterrupt:
        print("[tmp] 用户中断，退出。")
    finally:
        env.close()
        print("[tmp] 结束")


if __name__ == "__main__":
    main()
