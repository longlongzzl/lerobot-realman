#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import cv2
import gymnasium as gym
import mani_skill.envs  # noqa: F401  Ensures ManiSkill environments are registered.
import matplotlib.pyplot as plt
import numpy as np

import test_rm75_maniskill_env as maniskill_debug


DEFAULT_FOUNDATIONPOSE_ROOT = "/home/zhangzhao/PycharmProjects/FoundationPose"
DEFAULT_CAMERA_EXTRINSIC = "/home/zhangzhao/Desktop/lerobot-sim2real/results/realman/realman_home/base_camera/camera_extrinsic_opencv.npy"
DEFAULT_EXTRA_MANISKILL_PACKAGE_ROOT = "/home/zhangzhao/anaconda3/envs/realman/lib/python3.11/site-packages/mani_skill"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Realtime bridge: stream RealSense RGB-D into FoundationPose and update a ManiSkill RM75 scene live."
    )
    parser.add_argument("--env-id", type=str, default="RM75GraspCube_two_cameras-v1")
    parser.add_argument(
        "--env-kwargs-json-path",
        type=str,
        default="/home/zhangzhao/Desktop/lerobot/lerobot-sim2real/so101_env_config.json",
    )
    parser.add_argument("--obs-mode", type=str, default="rgb+segmentation")
    parser.add_argument("--render-mode", type=str, default="human")
    parser.add_argument("--reward-mode", type=str, default="none")
    parser.add_argument("--control-mode", type=str, default=None)
    parser.add_argument("--extra-maniskill-package-root", type=str, default=DEFAULT_EXTRA_MANISKILL_PACKAGE_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=128, help="ManiSkill sensor width.")
    parser.add_argument("--height", type=int, default=128, help="ManiSkill sensor height.")
    parser.add_argument("--shader-pack", type=str, default="default")
    parser.add_argument("--idle-render-fps", type=float, default=20.0)
    parser.add_argument("--no-boost-human-lighting", action="store_true")
    parser.add_argument("--no-wood-table", action="store_true")

    parser.add_argument("--foundationpose-root", type=str, default=DEFAULT_FOUNDATIONPOSE_ROOT)
    parser.add_argument("--mesh-file", type=str, required=True)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--camera-extrinsic-opencv-path", type=str, default=DEFAULT_CAMERA_EXTRINSIC)
    parser.add_argument(
        "--use-direct-camera-extrinsic",
        action="store_true",
        help="Interpret camera_extrinsic_opencv as a direct T_base_cam matrix. Default matches the previous export/replay pipeline and applies inverse first.",
    )
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--debug-dir", type=str, default="/home/zhangzhao/Desktop/lerobot/debug_foundationpose_realtime")
    parser.add_argument("--est-refine-iter", dest="est_refine_iter", type=int, default=5)
    parser.add_argument("--track-refine-iter", dest="track_refine_iter", type=int, default=2)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--init-mask", type=str, default=None)

    parser.add_argument("--camera-width", type=int, default=640, help="RealSense color/depth width.")
    parser.add_argument("--camera-height", type=int, default=480, help="RealSense color/depth height.")
    parser.add_argument("--camera-fps", type=int, default=30, help="RealSense capture FPS.")
    parser.add_argument("--camera-serial", type=str, default=None, help="Optional RealSense serial to bind.")
    parser.add_argument("--enable-custom-mesh-collision", action="store_true", help="Build convex collision for the imported FoundationPose mesh. Disabled by default because coacd is optional.")

    parser.add_argument("--hide-builtin-cube", dest="hide_builtin_cube", action="store_true")
    parser.add_argument("--keep-builtin-cube", dest="hide_builtin_cube", action="store_false")
    parser.set_defaults(hide_builtin_cube=True)
    parser.add_argument(
        "--foundationpose-position-offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
    )
    parser.add_argument("--lock-foundationpose-z-to-cube", action="store_true")
    parser.add_argument("--no-map-foundationpose-through-robot-base", action="store_true")
    parser.add_argument(
        "--foundationpose-local-rotation-offset-deg",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("ROLL", "PITCH", "YAW"),
    )
    return parser.parse_args()



def resolve_foundationpose_root(path_like: str) -> Path:
    candidates = []
    if path_like:
        candidates.append(Path(path_like).expanduser())
    candidates.append(Path(DEFAULT_FOUNDATIONPOSE_ROOT))
    candidates.append(Path(__file__).resolve().parent / "FoundationPose")

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "run_realtime_demo.py").exists() and (candidate / "estimater.py").exists():
            return candidate

    raise FileNotFoundError(
        f"Failed to locate a usable FoundationPose root. Checked: {[str(p) for p in candidates]}"
    )



def load_foundationpose_module(root: Path):
    script_path = root / "run_realtime_demo.py"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    spec = importlib.util.spec_from_file_location("foundationpose_realtime_bridge_impl", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load FoundationPose realtime module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def normalize_env_id(env_id: str) -> str:
    if env_id == "RM75GraspCube_two_cameras":
        return "RM75GraspCube_two_cameras-v1"
    return env_id


def ensure_digital_twin_base_env_compat():
    from mani_skill.envs.tasks.digital_twins.base_env import BaseDigitalTwinEnv

    if not hasattr(BaseDigitalTwinEnv, "remove_object_from_greenscreen_table"):
        def _remove_object_from_greenscreen_table(self, obj):
            return self.remove_object_from_greenscreen(obj)
        BaseDigitalTwinEnv.remove_object_from_greenscreen_table = _remove_object_from_greenscreen_table




def ensure_env_registered(env_id: str, extra_maniskill_package_root: str):
    env_id = normalize_env_id(env_id)
    if env_id in gym.envs.registry:
        return env_id

    ensure_digital_twin_base_env_compat()

    package_root = Path(extra_maniskill_package_root).expanduser()
    if package_root.name != "mani_skill" and (package_root / "mani_skill").exists():
        package_root = package_root / "mani_skill"
    if not package_root.exists():
        raise FileNotFoundError(f"Extra ManiSkill package root not found: {package_root}")

    import mani_skill
    import mani_skill.agents.robots as robots_pkg
    import mani_skill.envs.tasks.digital_twins as dt_pkg

    robots_path = str(package_root / "agents" / "robots")
    dt_path = str(package_root / "envs" / "tasks" / "digital_twins")
    if robots_path not in robots_pkg.__path__:
        robots_pkg.__path__.append(robots_path)
    if dt_path not in dt_pkg.__path__:
        dt_pkg.__path__.append(dt_path)

    if env_id == "RM75GraspCube_two_cameras-v1":
        import importlib
        importlib.import_module("mani_skill.agents.robots.realman")
        importlib.import_module("mani_skill.envs.tasks.digital_twins.so101_arm_with_two_cameras.RM75_grasp_cube")

    if env_id not in gym.envs.registry:
        raise gym.error.NameNotFound(f"Environment {env_id!r} still not registered after extending ManiSkill paths")
    return env_id


def load_matrix(path_like: str) -> np.ndarray:
    path = Path(path_like).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        mat = np.load(path)
    else:
        mat = np.loadtxt(path)
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix from {path}, got {mat.shape}")
    return mat



def build_foundationpose_actor_compatible(env, mesh_file: str, mesh_scale: float, actor_name: str, enable_collision: bool):
    scene = env.unwrapped.scene
    builder = scene.create_actor_builder()
    scale = [mesh_scale, mesh_scale, mesh_scale]
    collision_loaded = False

    if enable_collision:
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




def load_trimesh_compatible(fp_rt, mesh_file, mesh_scale=1.0):
    try:
        return fp_rt.load_trimesh(mesh_file, mesh_scale)
    except Exception as exc:
        if "operands could not be broadcast together with shapes" not in str(exc):
            raise

        print(f"[create_tracker] fp_rt.load_trimesh failed: {exc}")
        print("[create_tracker] retrying with local trimesh compatibility fallback")
        trimesh_mod = fp_rt.trimesh
        mesh_or_scene = trimesh_mod.load(mesh_file, force="scene")
        if isinstance(mesh_or_scene, trimesh_mod.Scene):
            geometries = [g for g in mesh_or_scene.geometry.values() if isinstance(g, trimesh_mod.Trimesh)]
            if len(geometries) == 0:
                raise ValueError(f"No mesh geometry found in: {mesh_file}")
            mesh = trimesh_mod.util.concatenate(geometries)
        elif isinstance(mesh_or_scene, trimesh_mod.Trimesh):
            mesh = mesh_or_scene
        else:
            raise TypeError(f"Unsupported mesh type: {type(mesh_or_scene)}")

        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError(f"Empty mesh: {mesh_file}")
        if mesh_scale <= 0:
            raise ValueError(f"--mesh_scale must be positive, got {mesh_scale}")

        mesh.remove_unreferenced_vertices()
        if hasattr(mesh, "remove_degenerate_faces"):
            mesh.remove_degenerate_faces()
        elif hasattr(mesh, "nondegenerate_faces") and hasattr(mesh, "update_faces"):
            mesh.update_faces(mesh.nondegenerate_faces())

        if hasattr(mesh, "remove_duplicate_faces"):
            mesh.remove_duplicate_faces()
        elif hasattr(mesh, "unique_faces") and hasattr(mesh, "update_faces"):
            mesh.update_faces(mesh.unique_faces())

        try:
            mesh.process(validate=True)
        except Exception as inner_exc:
            print(f"[create_tracker] mesh.process(validate=True) failed: {inner_exc}")
            print("[create_tracker] retrying with validate=False for trimesh compatibility")
            mesh.remove_unreferenced_vertices()
            mesh.process(validate=False)

        mesh.apply_scale(mesh_scale)
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
        _ = mesh.vertex_normals
        return mesh


def create_tracker(fp_rt, args):
    mesh = load_trimesh_compatible(fp_rt, args.mesh_file, args.mesh_scale)
    to_origin, extents = fp_rt.trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = fp_rt.ScorePredictor()
    refiner = fp_rt.PoseRefinePredictor()
    glctx = fp_rt.dr.RasterizeCudaContext()
    est = fp_rt.FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=args.debug_dir,
        debug=args.debug,
        glctx=glctx,
    )

    reader = fp_rt.RealSenseRGBDReader(
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    )
    if args.camera_serial:
        reader.config.enable_device(args.camera_serial)
    reader.start()

    fp_rt.make_debug_dirs(args.debug_dir)
    for _ in range(max(args.warmup_frames, 0)):
        if reader.get_frame() is None:
            continue

    return mesh, est, reader, to_origin, bbox



def map_camera_pose_to_sim(T_cam_obj: np.ndarray, T_base_cam: np.ndarray, env, args) -> np.ndarray:
    T_cam_obj = np.asarray(T_cam_obj, dtype=np.float32)
    T_base_cam = np.asarray(T_base_cam, dtype=np.float32)

    if args.use_direct_camera_extrinsic:
        T_base_obj = T_base_cam @ T_cam_obj
    else:
        T_base_obj = np.linalg.inv(T_base_cam) @ T_cam_obj

    return maniskill_debug.adjust_foundationpose_pose(T_base_obj, env, args)



def render_sensor_window(obs, fig, im):
    tiled_rgb, camera_names = maniskill_debug.tile_sensor_frames(obs)
    if tiled_rgb is None:
        return fig, im

    if fig is None or im is None:
        fig = plt.figure("RM75 ManiSkill Sensors")
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(tiled_rgb)
        ax.set_title(" | ".join(camera_names))
        ax.axis("off")
        plt.show(block=False)
    else:
        im.set_data(tiled_rgb)
        fig.axes[0].set_title(" | ".join(camera_names))
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    return fig, im



def visualize_tracking(fp_rt, frame, pose_cam_obj, to_origin, bbox, sim_translation, instant_fps, track_avg_fps):
    center_pose = pose_cam_obj @ np.linalg.inv(to_origin)
    vis = fp_rt.draw_posed_3d_box(frame["K"], img=frame["color"], ob_in_cam=center_pose, bbox=bbox)
    vis = fp_rt.draw_xyz_axis(
        vis,
        ob_in_cam=center_pose,
        scale=0.1,
        K=frame["K"],
        thickness=3,
        transparency=0,
        is_input_rgb=True,
    )

    vis_text = vis.copy()
    cv2.putText(vis_text, f"FPS: {instant_fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(vis_text, f"TRACK AVG FPS: {track_avg_fps:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(
        vis_text,
        f"SIM XYZ: {sim_translation[0]:.3f}, {sim_translation[1]:.3f}, {sim_translation[2]:.3f}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        vis_text,
        "s: init frame  r: re-register  q: quit",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.imshow("FoundationPose Realtime", vis_text[..., ::-1])
    return vis



def main():
    args = parse_args()
    args.env_id = ensure_env_registered(args.env_id, args.extra_maniskill_package_root)
    foundationpose_root = resolve_foundationpose_root(args.foundationpose_root)
    fp_rt = load_foundationpose_module(foundationpose_root)
    T_base_cam = load_matrix(args.camera_extrinsic_opencv_path)

    fp_rt.set_logging_format()
    fp_rt.set_seed(args.seed)

    args.mesh_file = maniskill_debug.resolve_local_mesh_path(args.mesh_file)
    env_kwargs = maniskill_debug.load_env_kwargs(args)
    print(f"Using FoundationPose root: {foundationpose_root}")
    print(f"Using mesh file: {args.mesh_file}")
    print(f"Using camera extrinsic from: {args.camera_extrinsic_opencv_path}")
    if args.use_direct_camera_extrinsic:
        print("Using camera extrinsic convention: T_base_obj = T_base_cam @ T_cam_obj")
    else:
        print("Using camera extrinsic convention: T_base_obj = inv(camera_extrinsic_opencv) @ T_cam_obj")

    env = gym.make(args.env_id, **env_kwargs)
    reader = None
    fig = None
    im = None
    try:
        obs, info = env.reset(seed=args.seed)
        if args.render_mode == "human" and not args.no_boost_human_lighting:
            maniskill_debug.boost_scene_lighting(env)
        if args.render_mode == "human" and not args.no_wood_table:
            maniskill_debug.apply_wood_table_material(env)

        mesh, est, reader, to_origin, bbox = create_tracker(fp_rt, args)
        actor = build_foundationpose_actor_compatible(
            env,
            mesh_file=str(Path(args.mesh_file).expanduser()),
            mesh_scale=args.mesh_scale,
            actor_name="foundationpose_realtime_object",
            enable_collision=args.enable_custom_mesh_collision,
        )
        if args.hide_builtin_cube:
            maniskill_debug.maybe_hide_builtin_cube(env)

        print("Real camera preview is open. Press 's' there to capture an initialization frame.")
        init_frame = fp_rt.wait_for_registration_frame(reader)
        if init_frame is None:
            return

        pose_cam_obj = fp_rt.initialize_pose(est, init_frame, args)
        if pose_cam_obj is None:
            raise RuntimeError("FoundationPose registration failed")

        T_sim_obj = map_camera_pose_to_sim(pose_cam_obj, T_base_cam, env, args)
        maniskill_debug.set_actor_pose_from_matrix(actor, T_sim_obj)
        print("Initial realtime FoundationPose registration succeeded.")

        total_track_time = 0.0
        tracked_frames = 0

        while True:
            frame = reader.get_frame()
            if frame is None:
                continue

            frame_start = time.time()
            pose_cam_obj = est.track_one(
                rgb=frame["color"],
                depth=frame["depth"],
                K=frame["K"],
                iteration=args.track_refine_iter,
            )
            infer_time = time.time() - frame_start
            tracked_frames += 1
            total_track_time += infer_time
            instant_fps = 1.0 / infer_time if infer_time > 0 else 0.0
            track_avg_fps = tracked_frames / total_track_time if total_track_time > 0 else 0.0

            T_sim_obj = map_camera_pose_to_sim(pose_cam_obj, T_base_cam, env, args)
            maniskill_debug.set_actor_pose_from_matrix(actor, T_sim_obj)

            visualize_tracking(
                fp_rt,
                frame,
                pose_cam_obj,
                to_origin,
                bbox,
                T_sim_obj[:3, 3],
                instant_fps,
                track_avg_fps,
            )

            if args.render_mode == "sensors":
                obs = env.get_obs()
                fig, im = render_sensor_window(obs, fig, im)
            else:
                env.render()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                print("Re-registration requested. Press 's' on the real camera preview to pick a new init frame.")
                init_frame = fp_rt.wait_for_registration_frame(reader)
                if init_frame is None:
                    break
                pose_cam_obj = fp_rt.initialize_pose(est, init_frame, args)
                if pose_cam_obj is None:
                    print("Re-registration failed; keeping previous pose.")
                    continue
                T_sim_obj = map_camera_pose_to_sim(pose_cam_obj, T_base_cam, env, args)
                maniskill_debug.set_actor_pose_from_matrix(actor, T_sim_obj)
                total_track_time = 0.0
                tracked_frames = 0

            if args.idle_render_fps > 0:
                time.sleep(1.0 / args.idle_render_fps)
    finally:
        if reader is not None:
            reader.stop()
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
