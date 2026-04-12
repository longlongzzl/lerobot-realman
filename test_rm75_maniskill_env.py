#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import mani_skill.envs  # noqa: F401  Ensures ManiSkill environments are registered.
import numpy as np
import sapien
from mani_skill.utils.visualization import tile_images
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke test for launching a ManiSkill RM75 environment."
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="RM75GraspCube_two_cameras-v1",
        help="Gym environment id to launch.",
    )
    parser.add_argument(
        "--env-kwargs-json-path",
        type=str,
        default="/home/zhangzhao/Desktop/lerobot/lerobot-sim2real/so101_env_config.json",
        help="Optional env kwargs JSON path.",
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="rgb+segmentation",
        help="Observation mode passed to gym.make.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        help="Render mode passed to gym.make. Use human or sensors.",
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="none",
        help="Reward mode passed to gym.make.",
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        default=None,
        help="Optional control mode override.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Number of random actions to run after reset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--hold-open",
        action="store_true",
        default=True,
        help="Keep the ManiSkill viewer open after stepping until you press Enter.",
    )
    parser.add_argument(
        "--random-motion",
        action="store_true",
        default=False,
        help="While holding the viewer open, keep stepping random actions so the scene moves.",
    )
    parser.add_argument(
        "--idle-render-fps",
        type=float,
        default=10.0,
        help="Viewer refresh rate while holding the window open.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Sensor width when sensor_configs is used.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Sensor height when sensor_configs is used.",
    )
    parser.add_argument(
        "--save-sensor-image",
        type=str,
        default="/home/zhangzhao/Desktop/rm75_base_camera.png",
        help="Where to save the reset sensor RGB image.",
    )
    parser.add_argument(
        "--no-show-sensor-image",
        action="store_true",
        help="Disable matplotlib display for the reset sensor RGB image.",
    )
    parser.add_argument(
        "--shader-pack",
        type=str,
        default="default",
        help="Optional ManiSkill viewer shader pack for human render mode.",
    )
    parser.add_argument(
        "--no-boost-human-lighting",
        action="store_true",
        help="Disable the extra ambient/directional lights added for human viewer debugging.",
    )
    parser.add_argument(
        "--no-wood-table",
        action="store_true",
        help="Keep the original table material instead of overriding it with a wood texture.",
    )
    parser.add_argument(
        "--foundationpose-export-json",
        type=str,
        # default="/home/zhangzhao/Desktop/lerobot/maniskill_export",
        default=None,
        help="Optional FoundationPose export json path or export dir. If set, load the custom mesh and place it with T_base_obj.",
    )
    parser.add_argument(
        "--hide-builtin-cube",
        action="store_true",
        default=True,
        help="Move the original cube far away when a custom FoundationPose mesh is loaded.",
    )
    parser.add_argument(
        "--foundationpose-position-offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="Optional xyz offset added in base frame to every imported FoundationPose pose.",
    )
    parser.add_argument(
        "--lock-foundationpose-z-to-cube",
        action="store_true",
        help="Override imported z with the environment cube z, useful when the imported object floats above the table.",
    )
    parser.add_argument(
        "--no-map-foundationpose-through-robot-base",
        action="store_true",
        help="Disable mapping imported FoundationPose base-frame poses through the simulated robot base pose.",
    )
    parser.add_argument(
        "--foundationpose-local-rotation-offset-deg",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("ROLL", "PITCH", "YAW"),
        help="Optional local mesh-frame rotation offset in degrees, applied after the imported pose.",
    )
    return parser.parse_args()


def maybe_load_camera_arrays(env_kwargs: dict):
    base_camera_settings = env_kwargs.get("base_camera_settings")
    if not isinstance(base_camera_settings, dict):
        return

    for key in ("extrinsics", "intrinsics"):
        value = base_camera_settings.get(key)
        if isinstance(value, str) and value.endswith(".npy"):
            path = Path(value)
            if path.exists():
                base_camera_settings[key] = np.load(path)


def load_env_kwargs(args) -> dict:
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        render_mode=args.render_mode,
        reward_mode=args.reward_mode,
        render_backend="cpu",
        sensor_configs=dict(width=args.width, height=args.height),
        domain_randomization=False,
    )
    if args.render_mode == "human":
        env_kwargs["viewer_camera_configs"] = dict(shader_pack=args.shader_pack)

    cfg_path = Path(args.env_kwargs_json_path)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        env_kwargs.update(loaded)
        maybe_load_camera_arrays(env_kwargs)

    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    return env_kwargs


def summarize_obs(obs):
    print("Observation summary:")
    if isinstance(obs, dict):
        for key, value in obs.items():
            if isinstance(value, dict):
                nested_keys = list(value.keys())
                print(f"  {key}: dict keys={nested_keys}")
            else:
                shape = getattr(value, "shape", None)
                dtype = getattr(value, "dtype", type(value))
                print(f"  {key}: shape={shape}, dtype={dtype}")
    else:
        shape = getattr(obs, "shape", None)
        dtype = getattr(obs, "dtype", type(obs))
        print(f"  obs: shape={shape}, dtype={dtype}")


def extract_first_rgb_frame(obs):
    sensor_data = obs.get("sensor_data")
    if not isinstance(sensor_data, dict):
        return None, None

    for camera_name, camera_data in sensor_data.items():
        if not isinstance(camera_data, dict):
            continue
        for key in ("rgb", "org_rgb"):
            rgb = camera_data.get(key)
            if rgb is None:
                continue
            if hasattr(rgb, "detach"):
                rgb = rgb.detach().cpu().numpy()
            rgb = np.asarray(rgb)
            if rgb.ndim == 4:
                rgb = rgb[0]
            if rgb.ndim == 3 and rgb.shape[-1] == 3:
                return camera_name, rgb.astype(np.uint8)
    return None, None


def save_and_optionally_show_sensor_image(obs, save_path: str, show: bool):
    camera_name, rgb = extract_first_rgb_frame(obs)
    if rgb is None:
        print("No sensor RGB image found in observation.")
        return

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, rgb)
    print(f"Saved sensor image from {camera_name} to {out_path}")

    if show:
        plt.figure(f"Sensor RGB: {camera_name}")
        plt.imshow(rgb)
        plt.title(f"{camera_name} sensor rgb")
        plt.axis("off")
        plt.show(block=False)


def extract_all_rgb_frames(obs):
    sensor_data = obs.get("sensor_data")
    if not isinstance(sensor_data, dict):
        return []

    frames = []
    for camera_name, camera_data in sensor_data.items():
        if not isinstance(camera_data, dict):
            continue
        rgb = camera_data.get("rgb")
        if rgb is None:
            continue
        if hasattr(rgb, "detach"):
            rgb = rgb.detach().cpu().numpy()
        rgb = np.asarray(rgb)
        if rgb.ndim == 4:
            rgb = rgb[0]
        if rgb.ndim == 3 and rgb.shape[-1] == 3:
            frames.append((camera_name, rgb.astype(np.uint8)))
    return frames


def tile_sensor_frames(obs):
    frames = extract_all_rgb_frames(obs)
    if not frames:
        return None, []
    tiled = tile_images([frame for _, frame in frames])
    if hasattr(tiled, "detach"):
        tiled = tiled.detach().cpu().numpy()
    return np.asarray(tiled), [name for name, _ in frames]


def boost_scene_lighting(env):
    scene = getattr(env.unwrapped, "scene", None)
    if scene is None:
        print("Scene handle not found; skipping lighting boost.")
        return

    try:
        scene.set_ambient_light([0.8, 0.8, 0.8])
        print("Applied ambient lighting boost.")
    except Exception as exc:
        print(f"Failed to boost scene lighting: {exc}")


def resolve_wood_texture_path() -> Path:
    candidates = [
        Path("/home/zhangzhao/anaconda3/envs/realman/lib/python3.11/site-packages/mani_skill/envs/tasks/tabletop/49038/images/texture_0.jpg"),
        Path("/home/zhangzhao/anaconda3/envs/realman/lib/python3.11/site-packages/mani_skill/envs/tasks/tabletop/49038/images/texture_1.jpg"),
        Path("/home/zhangzhao/anaconda3/envs/realman/lib/python3.11/site-packages/mani_skill/envs/tasks/tabletop/49038/textured_objs_fixed/material_0.png"),
    ]
    for path in candidates:
        if path.exists():
            return path

    fallback = Path("/tmp/rm75_debug_wood_texture.png")
    if not fallback.exists():
        generate_wood_texture(fallback)
    return fallback


def generate_wood_texture(texture_path: Path, size: int = 1024):
    texture_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    x = np.linspace(0, 1, size, dtype=np.float32)
    y = np.linspace(0, 1, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    base = 0.45 + 0.25 * np.sin(2 * np.pi * (xx * 10.0 + 0.15 * np.sin(yy * 18.0)))
    fine = 0.08 * np.sin(2 * np.pi * (xx * 45.0 + yy * 3.0))
    noise = rng.normal(0.0, 0.035, size=(size, size)).astype(np.float32)
    grain = np.clip(base + fine + noise, 0.0, 1.0)

    wood = np.zeros((size, size, 3), dtype=np.float32)
    wood[..., 0] = np.clip(0.30 + 0.45 * grain, 0.0, 1.0)
    wood[..., 1] = np.clip(0.18 + 0.28 * grain, 0.0, 1.0)
    wood[..., 2] = np.clip(0.08 + 0.14 * grain, 0.0, 1.0)

    plank_mask = ((yy * 6).astype(int) % 2 == 0).astype(np.float32)
    wood[..., 0] *= 0.94 + 0.08 * plank_mask
    wood[..., 1] *= 0.94 + 0.06 * plank_mask
    wood[..., 2] *= 0.94 + 0.04 * plank_mask

    plt.imsave(texture_path, np.clip(wood, 0.0, 1.0))


def apply_wood_table_material(env):
    table_scene = getattr(env.unwrapped, "table_scene", None)
    table = getattr(table_scene, "table", None)
    if table is None:
        print("Table handle not found; skipping table material override.")
        return

    changed = 0
    try:
        texture_path = resolve_wood_texture_path()
        texture = sapien.render.RenderTexture2D(str(texture_path))

        for obj in getattr(table, "_objs", []):
            render_body = obj.find_component_by_type(sapien.render.RenderBodyComponent)
            if render_body is None:
                continue
            for shape in render_body.render_shapes:
                for part in shape.parts:
                    material = part.material
                    material.set_base_color(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
                    material.set_base_color_texture(texture)
                    material.set_normal_texture(None)
                    material.set_emission_texture(None)
                    material.set_transmission_texture(None)
                    material.set_metallic_texture(None)
                    material.set_roughness_texture(None)
                    changed += 1
        print(f"Applied wood table material override from {texture_path} to {changed} render parts.")
    except Exception as exc:
        print(f"Failed to apply wood table material: {exc}")


def _normalize_foundationpose_payload(path: Path, payload: dict) -> dict:
    payload["mesh_file"] = str(payload["mesh_file"])
    payload["mesh_scale"] = float(payload.get("mesh_scale", 1.0))

    T_cam_obj = payload.get("T_cam_obj")
    T_cam_base = payload.get("T_base_cam")
    if T_cam_obj is not None:
        T_cam_obj = np.asarray(T_cam_obj, dtype=np.float32).reshape(4, 4)
        payload["T_cam_obj"] = T_cam_obj
    if T_cam_base is not None:
        T_cam_base = np.asarray(T_cam_base, dtype=np.float32).reshape(4, 4)
        payload["T_base_cam"] = T_cam_base

    if T_cam_obj is not None and T_cam_base is not None:
        payload["T_base_obj"] = np.linalg.inv(T_cam_base) @ T_cam_obj
    elif "T_base_obj" in payload:
        payload["T_base_obj"] = np.asarray(payload["T_base_obj"], dtype=np.float32).reshape(4, 4)
    else:
        raise ValueError(
            f"{path} does not contain enough pose data. Need T_cam_obj + T_base_cam, or T_base_obj."
        )
    return payload


def resolve_local_mesh_path(mesh_file: str) -> str:
    path = Path(mesh_file)
    if path.exists():
        return str(path)

    candidates = [
        Path("/home/zhangzhao/Desktop/lerobot/assets") / path.name,
        Path("/home/zhangzhao/Desktop/lerobot/FoundationPose/assets") / path.name,
        Path("/home/zhangzhao/Desktop/lerobot/maniskill_export") / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            print(f"Remapped mesh path: {mesh_file} -> {candidate}")
            return str(candidate)

    return mesh_file


def load_foundationpose_exports(path_like: str) -> list[dict]:
    path = Path(path_like)
    json_paths = []

    if path.is_dir():
        json_paths = sorted(
            p for p in path.glob("*.json") if p.name != "latest.json"
        )
        if not json_paths:
            subdirs = [p for p in path.iterdir() if p.is_dir()]
            subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for subdir in subdirs:
                json_paths = sorted(
                    p for p in subdir.glob("*.json") if p.name != "latest.json"
                )
                if json_paths:
                    print(f"Using latest FoundationPose export subdir: {subdir}")
                    break
        if not json_paths:
            latest = path / "latest.json"
            if latest.exists():
                json_paths = [latest]
    else:
        json_paths = [path]

    payloads = []
    for json_path in json_paths:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            continue
        if "mesh_file" not in payload and "T_cam_obj" not in payload and "T_base_obj" not in payload:
            continue
        normalized = _normalize_foundationpose_payload(json_path, payload)
        normalized["mesh_file"] = resolve_local_mesh_path(normalized["mesh_file"])
        payloads.append(normalized)

    if not payloads:
        raise FileNotFoundError(f"No FoundationPose export json found under {path}")

    return payloads


def load_foundationpose_export(json_path: str) -> dict:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    normalized = _normalize_foundationpose_payload(path, payload)
    normalized["mesh_file"] = resolve_local_mesh_path(normalized["mesh_file"])
    return normalized


def build_foundationpose_actor(env, mesh_file: str, mesh_scale: float, actor_name: str = "foundationpose_object"):
    scene = env.unwrapped.scene
    builder = scene.create_actor_builder()
    scale = [mesh_scale, mesh_scale, mesh_scale]
    collision_loaded = False

    try:
        builder.add_multiple_convex_collisions_from_file(
            mesh_file,
            decomposition="coacd",
            scale=scale,
            material=None,
            density=1000,
        )
        collision_loaded = True
    except Exception as exc:
        print(f"Custom mesh collision load failed, falling back to visual-only actor: {exc}")

    builder.add_visual_from_file(mesh_file, scale=scale)
    actor = builder.build_kinematic(name=actor_name)
    print(
        f"Loaded custom FoundationPose mesh: {mesh_file} "
        f"(scale={mesh_scale}, collision={'yes' if collision_loaded else 'no'})"
    )
    return actor


def set_actor_pose_from_matrix(actor, T_base_obj: np.ndarray):
    pos = T_base_obj[:3, 3].astype(np.float32)
    quat = mat2quat(T_base_obj[:3, :3]).astype(np.float32)
    actor.set_pose(Pose.create_from_pq(p=pos, q=quat))


def maybe_hide_builtin_cube(env):
    cube = getattr(env.unwrapped, "cube", None)
    if cube is None:
        cube = getattr(env.unwrapped, "object", None)
    if cube is None:
        print("Builtin cube/object handle not found; nothing to hide.")
        return
    cube.set_pose(Pose.create_from_pq(p=[10.0, 10.0, 10.0], q=[1.0, 0.0, 0.0, 0.0]))
    print("Moved builtin cube/object out of view.")


def get_reference_object_z(env):
    cube = getattr(env.unwrapped, "cube", None)
    if cube is None:
        cube = getattr(env.unwrapped, "object", None)
    if cube is None:
        return None
    pose = getattr(cube, "pose", None)
    if pose is None:
        return None
    p = getattr(pose, "p", None)
    if p is None:
        return None
    if hasattr(p, "detach"):
        p = p.detach().cpu().numpy()
    p = np.asarray(p)
    if p.ndim == 2:
        p = p[0]
    return float(p[2])


def get_robot_base_transform(env):
    agent = getattr(env.unwrapped, "agent", None)
    robot = getattr(agent, "robot", None)
    pose = getattr(robot, "pose", None)
    if pose is None:
        return None
    mat = pose.to_transformation_matrix()
    if hasattr(mat, "detach"):
        mat = mat.detach().cpu().numpy()
    mat = np.asarray(mat)
    if mat.ndim == 3:
        mat = mat[0]
    return mat.astype(np.float32)


def adjust_foundationpose_pose(T_base_obj: np.ndarray, env, args) -> np.ndarray:
    adjusted = np.array(T_base_obj, dtype=np.float32, copy=True)

    if not args.no_map_foundationpose_through_robot_base:
        robot_base_T = get_robot_base_transform(env)
        if robot_base_T is not None:
            adjusted = robot_base_T @ adjusted

    local_rpy_deg = np.asarray(args.foundationpose_local_rotation_offset_deg, dtype=np.float32)
    if np.any(np.abs(local_rpy_deg) > 1e-6):
        local_fix = np.eye(4, dtype=np.float32)
        local_fix[:3, :3] = euler2mat(*np.deg2rad(local_rpy_deg)).astype(np.float32)
        adjusted = adjusted @ local_fix

    adjusted[:3, 3] += np.asarray(args.foundationpose_position_offset, dtype=np.float32)

    if args.lock_foundationpose_z_to_cube:
        ref_z = get_reference_object_z(env)
        if ref_z is not None:
            adjusted[2, 3] = ref_z

    return adjusted


def main():
    args = parse_args()
    env_kwargs = load_env_kwargs(args)
    fp_payload = None
    fp_payloads = None
    fp_actor = None
    fp_frame_idx = 0

    if args.foundationpose_export_json is not None:
        fp_payloads = load_foundationpose_exports(args.foundationpose_export_json)
        fp_payload = fp_payloads[0]
        print("FoundationPose export:")
        print(f"  frames: {len(fp_payloads)}")
        print(f"  first_frame_id: {fp_payload.get('frame_id')}")
        print(f"  mesh_file: {fp_payload['mesh_file']}")
        print(f"  mesh_scale: {fp_payload['mesh_scale']}")
        print(f"  T_base_obj translation: {fp_payload['T_base_obj'][:3, 3].tolist()}")

    print(f"Launching env_id={args.env_id}")
    print("env_kwargs:")
    for key, value in env_kwargs.items():
        if key == "base_camera_settings" and isinstance(value, dict):
            printable = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    printable[k] = f"ndarray shape={v.shape}"
                else:
                    printable[k] = v
            print(f"  {key}: {printable}")
        else:
            print(f"  {key}: {value}")

    env = gym.make(args.env_id, **env_kwargs)
    try:
        obs, info = env.reset(seed=args.seed)
        if args.render_mode == "human" and not args.no_boost_human_lighting:
            boost_scene_lighting(env)
        if args.render_mode == "human" and not args.no_wood_table:
            apply_wood_table_material(env)
        if fp_payload is not None:
            fp_actor = build_foundationpose_actor(
                env,
                mesh_file=fp_payload["mesh_file"],
                mesh_scale=fp_payload["mesh_scale"],
            )
            set_actor_pose_from_matrix(
                fp_actor,
                adjust_foundationpose_pose(fp_payload["T_base_obj"], env, args),
            )
            if args.hide_builtin_cube:
                maybe_hide_builtin_cube(env)
        print("Reset succeeded.")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        summarize_obs(obs)
        save_and_optionally_show_sensor_image(obs, args.save_sensor_image, not args.no_show_sensor_image)
        print(f"Reset info keys: {list(info.keys()) if isinstance(info, dict) else type(info)}")

        for step_idx in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"step={step_idx} reward={reward} terminated={terminated} "
                f"truncated={truncated}"
            )
            if step_idx == 0:
                summarize_obs(obs)
            if terminated or truncated:
                print("Episode ended early; resetting.")
                obs, info = env.reset()
                fp_frame_idx = 0
                if fp_payload is not None:
                    set_actor_pose_from_matrix(
                        fp_actor,
                        adjust_foundationpose_pose(fp_payloads[fp_frame_idx]["T_base_obj"], env, args),
                    )
                    if args.hide_builtin_cube:
                        maybe_hide_builtin_cube(env)

        if args.hold_open:
            print("Display is being kept open. Press Ctrl+C to exit.")
            fig = None
            im = None
            if args.render_mode == "sensors":
                tiled_rgb, camera_names = tile_sensor_frames(obs)
                if tiled_rgb is not None:
                    fig = plt.figure("RM75 ManiSkill Sensors")
                    ax = fig.add_subplot(1, 1, 1)
                    im = ax.imshow(tiled_rgb)
                    ax.set_title(" | ".join(camera_names))
                    ax.axis("off")
                    plt.show(block=False)
            try:
                while True:
                    if fp_actor is not None and fp_payloads is not None:
                        fp_frame_idx = (fp_frame_idx + 1) % len(fp_payloads)
                        set_actor_pose_from_matrix(
                            fp_actor,
                            adjust_foundationpose_pose(fp_payloads[fp_frame_idx]["T_base_obj"], env, args),
                        )
                    if args.random_motion:
                        action = env.action_space.sample()
                        obs, reward, terminated, truncated, info = env.step(action)
                        if terminated or truncated:
                            obs, info = env.reset()
                            fp_frame_idx = 0
                            if fp_payload is not None:
                                set_actor_pose_from_matrix(
                                    fp_actor,
                                    adjust_foundationpose_pose(fp_payloads[fp_frame_idx]["T_base_obj"], env, args),
                                )
                                if args.hide_builtin_cube:
                                    maybe_hide_builtin_cube(env)
                    elif args.render_mode == "sensors":
                        obs = env.get_obs()

                    if args.render_mode == "sensors":
                        tiled_rgb, camera_names = tile_sensor_frames(obs)
                        if im is not None and tiled_rgb is not None:
                            im.set_data(tiled_rgb)
                            fig.axes[0].set_title(" | ".join(camera_names))
                            fig.canvas.draw_idle()
                            fig.canvas.flush_events()
                    else:
                        env.render()
                    if args.idle_render_fps > 0:
                        time.sleep(1.0 / args.idle_render_fps)
            except KeyboardInterrupt:
                print("Interrupted, closing display.")

        print("Smoke test finished successfully.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
