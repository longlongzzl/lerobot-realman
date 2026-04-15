#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np

import rm75_jiaobang_pick_place_targeted as targeted
from curobo_rm75_planner import RM75CuRoboPlanner, RM75CuRoboPlannerConfig
from object_specs import normalize_object_name


_PLANNER_CACHE: dict[tuple[Any, ...], RM75CuRoboPlanner] = {}


def build_arg_parser():
    parser = targeted.build_arg_parser()
    parser.description = (
        "FoundationPose -> grasp -> targeted place pipeline with cuRobo replacing selected "
        "pose-planning stages in a new wrapper entrypoint."
    )
    parser.add_argument(
        "--curobo-use-for-targeted-place",
        dest="curobo_use_for_targeted_place",
        action="store_true",
        default=True,
        help="Use cuRobo for targeted-place pose planning stages such as pre_place and staging. Enabled by default in this wrapper.",
    )
    parser.add_argument(
        "--no-curobo-use-for-targeted-place",
        dest="curobo_use_for_targeted_place",
        action="store_false",
        help="Disable cuRobo interception and run the original targeted-place script unchanged.",
    )
    parser.add_argument(
        "--curobo-plan-label-prefixes",
        type=str,
        nargs="*",
        default=["pre_place", "pre_place_staging"],
        help="Only base.plan_pose_path calls whose label starts with one of these prefixes will be redirected to cuRobo.",
    )
    parser.add_argument(
        "--curobo-fallback-to-base",
        dest="curobo_fallback_to_base",
        action="store_true",
        default=True,
        help="If cuRobo planning fails for an intercepted label, fall back to the original planner. Enabled by default.",
    )
    parser.add_argument(
        "--no-curobo-fallback-to-base",
        dest="curobo_fallback_to_base",
        action="store_false",
        help="If cuRobo planning fails for an intercepted label, do not fall back to the original planner.",
    )
    parser.add_argument(
        "--curobo-enable-graph",
        action="store_true",
        help="Enable graph search inside cuRobo MotionGen for intercepted stages.",
    )
    parser.add_argument(
        "--curobo-max-attempts",
        type=int,
        default=2,
        help="Maximum MotionGen attempts for each intercepted stage.",
    )
    parser.add_argument(
        "--curobo-timeout",
        type=float,
        default=5.0,
        help="cuRobo MotionGen timeout in seconds for each intercepted stage.",
    )
    parser.add_argument(
        "--curobo-root",
        type=Path,
        default=Path("/home/zhangzhao/PycharmProjects/curobo"),
        help="Path to the local cuRobo repository root.",
    )
    parser.add_argument(
        "--curobo-rm75-robot-cfg",
        type=Path,
        default=None,
        help="Optional formal RM75 cuRobo robot config yaml. If omitted, this wrapper uses the embedded free-space RM75 config.",
    )
    parser.add_argument(
        "--curobo-rm75-urdf",
        type=Path,
        default=Path("/home/zhangzhao/Desktop/lerobot/RM75_gripper/RM75-B/urdf/RM75-B.urdf"),
        help="RM75 URDF used when --curobo-rm75-robot-cfg is not provided.",
    )
    parser.add_argument(
        "--curobo-ee-link",
        type=str,
        default="gripper_tcp",
        help="cuRobo end-effector link name.",
    )
    parser.add_argument(
        "--curobo-device",
        type=str,
        default="cuda:0",
        help="Torch device passed to cuRobo, for example cuda:0.",
    )
    parser.add_argument(
        "--curobo-torch-extensions-dir",
        type=Path,
        default=Path("/tmp/curobo_torch_extensions"),
        help="Writable directory for torch JIT CUDA extensions used by cuRobo.",
    )
    parser.add_argument(
        "--curobo-cuda-arch-list",
        type=str,
        default=None,
        help="Optional TORCH_CUDA_ARCH_LIST override for cuRobo compilation.",
    )
    parser.add_argument(
        "--curobo-num-ik-seeds",
        type=int,
        default=64,
        help="Number of IK seeds for intercepted cuRobo stages.",
    )
    parser.add_argument(
        "--curobo-num-trajopt-seeds",
        type=int,
        default=1,
        help="Number of trajectory-optimization seeds for intercepted cuRobo stages.",
    )
    parser.add_argument(
        "--curobo-num-graph-seeds",
        type=int,
        default=1,
        help="Number of graph-planning seeds for intercepted cuRobo stages.",
    )
    parser.add_argument(
        "--curobo-ik-position-threshold",
        type=float,
        default=0.005,
        help="IK position threshold in meters. Increase to relax IK acceptance during screening.",
    )
    parser.add_argument(
        "--curobo-ik-rotation-threshold",
        type=float,
        default=0.05,
        help="IK rotation threshold in radians. Increase to relax IK acceptance during screening.",
    )
    parser.add_argument(
        "--curobo-collision-activation-distance",
        type=float,
        default=0.02,
        help="Distance in meters at which cuRobo activates collision cost during trajectory optimization. "
        "Larger values make the robot stay further from obstacles (default 0.02 = 2cm clearance).",
    )
    parser.add_argument(
        "--curobo-debug",
        action="store_true",
        help="Print extra cuRobo interception diagnostics.",
    )
    parser.add_argument(
        "--curobo-world-mesh-object-names",
        type=str,
        nargs="*",
        default=["bitong"],
        help="Scene obstacle object names that should be represented as cuRobo world meshes instead of cuboids.",
    )
    parser.add_argument(
        "--curobo-diagnose-failures",
        action="store_true",
        help="When an intercepted cuRobo stage fails, run standalone IK and MotionGen-IK diagnostics and print the results.",
    )
    parser.add_argument(
        "--curobo-failure-dump-dir",
        type=Path,
        default=None,
        help="Optional directory where intercepted cuRobo failure cases will be dumped as json files.",
    )
    return parser


def parse_args():
    return build_arg_parser().parse_args()


def _planner_cache_key(args) -> tuple[Any, ...]:
    robot_cfg_path = None if args.curobo_rm75_robot_cfg is None else str(args.curobo_rm75_robot_cfg.expanduser().resolve())
    return (
        str(args.curobo_root.expanduser().resolve()),
        str(args.curobo_rm75_urdf.expanduser().resolve()),
        robot_cfg_path,
        str(args.curobo_ee_link),
        str(args.curobo_device),
        str(args.curobo_torch_extensions_dir.expanduser().resolve()),
        None if args.curobo_cuda_arch_list is None else str(args.curobo_cuda_arch_list),
        int(args.curobo_num_ik_seeds),
        int(args.curobo_num_trajopt_seeds),
        int(args.curobo_num_graph_seeds),
        float(args.curobo_ik_position_threshold),
        float(args.curobo_ik_rotation_threshold),
        float(args.curobo_collision_activation_distance),
    )


def _get_or_create_curobo_planner(args) -> RM75CuRoboPlanner:
    key = _planner_cache_key(args)
    planner = _PLANNER_CACHE.get(key)
    if planner is not None:
        return planner
    planner = RM75CuRoboPlanner(
        RM75CuRoboPlannerConfig(
            curobo_root=args.curobo_root,
            urdf=args.curobo_rm75_urdf,
            robot_cfg_path=args.curobo_rm75_robot_cfg,
            ee_link=str(args.curobo_ee_link),
            device=args.curobo_device,
            torch_extensions_dir=args.curobo_torch_extensions_dir,
            cuda_arch_list=args.curobo_cuda_arch_list,
            num_ik_seeds=int(args.curobo_num_ik_seeds),
            num_trajopt_seeds=int(args.curobo_num_trajopt_seeds),
            num_graph_seeds=int(args.curobo_num_graph_seeds),
            position_threshold=float(args.curobo_ik_position_threshold),
            rotation_threshold=float(args.curobo_ik_rotation_threshold),
            collision_activation_distance=float(args.curobo_collision_activation_distance),
        )
    )
    _PLANNER_CACHE[key] = planner
    return planner


def _label_matches_curobo_intercept(label: str, args) -> bool:
    prefixes = [str(x) for x in list(getattr(args, "curobo_plan_label_prefixes", []) or [])]
    if not bool(getattr(args, "curobo_use_for_targeted_place", True)):
        return False
    return any(str(label).startswith(prefix) for prefix in prefixes)


def _scene_obstacles_to_curobo_cuboids(demo) -> list[dict[str, Any]]:
    cuboids: list[dict[str, Any]] = []
    for item in list(getattr(demo, "scene_obstacles", []) or []):
        if not bool(item.get("planner_collision", False)):
            continue
        T_world_obj = item.get("T_world_obj")
        planner_box_size = item.get("planner_box_size", item.get("visual_box_size"))
        if T_world_obj is None or planner_box_size is None:
            continue
        T_world_obj = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4)
        dims = np.asarray(planner_box_size, dtype=np.float32).reshape(3)
        quat = targeted.base.bridge_mod_mat2quat(T_world_obj[:3, :3]).astype(np.float32)
        pose = np.concatenate([T_world_obj[:3, 3].astype(np.float32), quat]).astype(np.float32)
        cuboids.append(
            {
                "name": str(item.get("actor_name") or item.get("object_name") or f"obstacle_{len(cuboids):02d}"),
                "dims": dims.tolist(),
                "pose": pose.tolist(),
            }
        )
    return cuboids


def _curobo_world_mesh_name_set(args) -> set[str]:
    mesh_names: set[str] = set()
    for raw_name in list(getattr(args, "curobo_world_mesh_object_names", []) or []):
        normalized = normalize_object_name(raw_name)
        if normalized is not None:
            mesh_names.add(normalized)
    return mesh_names


def _scene_obstacles_to_curobo_world(
    demo,
    args,
    *,
    exclude_object_names: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cuboids: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []
    mesh_name_set = _curobo_world_mesh_name_set(args)
    for item in list(getattr(demo, "scene_obstacles", []) or []):
        if not bool(item.get("planner_collision", False)):
            continue
        if exclude_object_names:
            obj_name = normalize_object_name(item.get("object_name"))
            if obj_name in exclude_object_names:
                continue
        T_world_obj = item.get("T_world_obj")
        if T_world_obj is None:
            continue
        T_world_obj = np.asarray(T_world_obj, dtype=np.float32).reshape(4, 4)
        quat = targeted.base.bridge_mod_mat2quat(T_world_obj[:3, :3]).astype(np.float32)
        pose = np.concatenate([T_world_obj[:3, 3].astype(np.float32), quat]).astype(np.float32)
        obstacle_name = str(item.get("actor_name") or item.get("object_name") or f"obstacle_{len(cuboids):02d}")
        object_name = normalize_object_name(item.get("object_name"))
        asset_file = item.get("asset_file")
        asset_scale = item.get("asset_scale")
        if object_name in mesh_name_set and asset_file is not None:
            scale = float(asset_scale) if asset_scale is not None else 1.0
            meshes.append(
                {
                    "name": obstacle_name,
                    "file_path": str(asset_file),
                    "scale": [scale, scale, scale],
                    "pose": pose.tolist(),
                }
            )
            continue
        planner_box_size = item.get("planner_box_size", item.get("visual_box_size"))
        if planner_box_size is None:
            continue
        dims = np.asarray(planner_box_size, dtype=np.float32).reshape(3)
        cuboids.append(
            {
                "name": obstacle_name,
                "dims": dims.tolist(),
                "pose": pose.tolist(),
            }
        )
    return cuboids, meshes


def _get_robot_base_world_transform(demo) -> np.ndarray:
    return targeted.base.pose_to_matrix(
        targeted.base.flatten_np(demo.robot.pose.p)[:3],
        targeted.base.flatten_np(demo.robot.pose.q)[:4],
    )


def _transform_curobo_world_to_robot_base(
    cuboids: list[dict[str, Any]],
    meshes: list[dict[str, Any]],
    demo,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    T_world_base = _get_robot_base_world_transform(demo)
    T_base_world = np.linalg.inv(T_world_base)
    cuboids_in_base: list[dict[str, Any]] = []
    meshes_in_base: list[dict[str, Any]] = []
    for item in cuboids:
        pose = np.asarray(item["pose"], dtype=np.float32).reshape(7)
        T_world_obj = targeted.base.pose_to_matrix(pose[:3], pose[3:])
        T_base_obj = T_base_world @ T_world_obj
        pose_in_base = np.concatenate(
            [
                T_base_obj[:3, 3].astype(np.float32),
                targeted.base.bridge_mod_mat2quat(T_base_obj[:3, :3]).astype(np.float32),
            ]
        ).astype(np.float32)
        cuboids_in_base.append(
            {
                "name": str(item["name"]),
                "dims": list(item["dims"]),
                "pose": pose_in_base.tolist(),
            }
        )
    for item in meshes:
        pose = np.asarray(item["pose"], dtype=np.float32).reshape(7)
        T_world_obj = targeted.base.pose_to_matrix(pose[:3], pose[3:])
        T_base_obj = T_base_world @ T_world_obj
        pose_in_base = np.concatenate(
            [
                T_base_obj[:3, 3].astype(np.float32),
                targeted.base.bridge_mod_mat2quat(T_base_obj[:3, :3]).astype(np.float32),
            ]
        ).astype(np.float32)
        mesh_in_base = dict(item)
        mesh_in_base["pose"] = pose_in_base.tolist()
        meshes_in_base.append(mesh_in_base)
    return cuboids_in_base, meshes_in_base


def _make_curobo_pose_planner_wrapper(original_plan_pose_path, planner: RM75CuRoboPlanner, args):
    def _wrapped_plan_pose_path(
        demo,
        pose,
        *,
        variant_name: str = "array_wxyz",
        use_attach: bool = False,
        label: str = "pose_goal",
        planning_time: float = 4.0,
        rrt_range: float = 0.15,
        start_q=None,
        allow_start_in_collision: bool = False,
    ):
        if not _label_matches_curobo_intercept(label, args):
            return original_plan_pose_path(
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

        start_q_np = np.asarray(
            demo.current_arm_qpos() if start_q is None else start_q,
            dtype=np.float32,
        ).reshape(-1)[:7]

        if planner.collision_enabled:
            cuboids, meshes = _scene_obstacles_to_curobo_world(demo, args)
            cuboids_in_base, meshes_in_base = _transform_curobo_world_to_robot_base(cuboids, meshes, demo)
            planner.set_world_from_obstacles(cuboids=cuboids_in_base, meshes=meshes_in_base)
            if bool(getattr(args, "curobo_debug", False)):
                print(
                    f"[curobo] updated world with {len(cuboids_in_base)} cuboid and "
                    f"{len(meshes_in_base)} mesh obstacles for label={label}"
                )
        elif bool(getattr(args, "curobo_debug", False)):
            print(f"[curobo] planner is in embedded free-space mode for label={label}")

        result = planner.plan_to_pose(
            start_q_np,
            pose,
            enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
            max_attempts=int(getattr(args, "curobo_max_attempts", 2)),
            timeout=float(getattr(args, "curobo_timeout", 5.0)),
            num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
            num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
            num_graph_seeds=int(getattr(args, "curobo_num_graph_seeds", 1)),
        )
        if result.success and result.joint_path is not None:
            print(
                f"[curobo] label={label} success, waypoints={len(result.joint_path)}, "
                f"solve_time={result.solve_time:.3f}s, ik_time={result.ik_time:.3f}s, "
                f"trajopt_time={result.trajopt_time:.3f}s, use_attach={use_attach}"
            )
            return [np.asarray(q, dtype=np.float32) for q in result.joint_path]

        print(
            f"[curobo] label={label} failed with status={result.status}, "
            f"internal_ik={result.debug.get('motion_gen_internal_ik_successes')}"
        )
        if bool(getattr(args, "curobo_diagnose_failures", False)):
            diag = planner.diagnose_pose_goal(
                start_q_np,
                pose,
                num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
                num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
            )
            print(
                f"[curobo diagnose] label={label} standalone_ik_success={diag['standalone_ik_success']}, "
                f"standalone_pos_err={diag['standalone_ik_position_error']:.6f}, "
                f"standalone_rot_err={diag['standalone_ik_rotation_error']:.6f}, "
                f"motiongen_internal_ik_successes={diag['motiongen_internal_ik_successes']}, "
                f"collision_enabled={diag['collision_enabled']}, "
                f"embedded_cfg={diag['using_embedded_robot_cfg']}, "
                f"attached_object_collision_enabled={diag['attached_object_collision_enabled']}"
            )
            dump_dir = getattr(args, "curobo_failure_dump_dir", None)
            if dump_dir is not None:
                safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("_") or "pose"
                dump_path = dump_dir / f"{safe_label}.json"
                payload = {
                    "label": str(label),
                    "status": result.status,
                    "use_attach": bool(use_attach),
                    "diagnostics": diag,
                }
                written = planner.dump_diagnostics_json(dump_path, payload)
                print(f"[curobo diagnose] wrote failure case to {written}")
        if not bool(getattr(args, "curobo_fallback_to_base", True)):
            return None

        print(f"[curobo] label={label} falling back to the original planner")
        return original_plan_pose_path(
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

    return _wrapped_plan_pose_path


def run_targeted_place_episode_curobo(demo, bridge_mod, real_exec, args, scene_capture_cache, place_state_cache) -> bool:
    if not bool(getattr(args, "curobo_use_for_targeted_place", True)):
        return _ORIGINAL_RUN_TARGETED_PLACE_EPISODE(
            demo,
            bridge_mod,
            real_exec,
            args,
            scene_capture_cache,
            place_state_cache,
        )

    planner = _get_or_create_curobo_planner(args)
    original_plan_pose_path = targeted.base.plan_pose_path
    targeted.base.plan_pose_path = _make_curobo_pose_planner_wrapper(
        original_plan_pose_path,
        planner,
        args,
    )
    try:
        return _ORIGINAL_RUN_TARGETED_PLACE_EPISODE(
            demo,
            bridge_mod,
            real_exec,
            args,
            scene_capture_cache,
            place_state_cache,
        )
    finally:
        targeted.base.plan_pose_path = original_plan_pose_path


def main():
    targeted.parse_args = parse_args
    targeted.run_targeted_place_episode = run_targeted_place_episode_curobo
    targeted.main()


_ORIGINAL_RUN_TARGETED_PLACE_EPISODE = targeted.run_targeted_place_episode


if __name__ == "__main__":
    main()
