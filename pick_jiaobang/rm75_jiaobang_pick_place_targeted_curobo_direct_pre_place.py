#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np

import rm75_jiaobang_pick_place_targeted as targeted
import rm75_jiaobang_pick_place_targeted_curobo as curobo_wrapper


def build_arg_parser():
    parser = curobo_wrapper.build_arg_parser()
    parser.description = (
        "FoundationPose -> direct cuRobo grasp -> close gripper -> cuRobo candidate-selected "
        "pre-place -> short constrained release descent."
    )
    parser.set_defaults(
        targeted_place_staging=False,
        curobo_plan_label_prefixes=[],
        curobo_ee_link="gripper_tcp",
        insert_vertical_axial_spin_deg=[0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
        topdown_grasp_yaw_variant_deg=[0.0],
    )
    parser.add_argument(
        "--direct-release-approach-distance",
        type=float,
        default=0.04,
        help="For insert_vertical rules, build a short pre-place this far from the release pose along the target long axis.",
    )
    parser.add_argument(
        "--direct-release-approach-distances",
        type=float,
        nargs="*",
        default=[0.03, 0.04, 0.05, 0.06],
        help="For insert_vertical rules, also try these short pre-place approach distances along the target long axis.",
    )
    parser.add_argument(
        "--direct-pre-place-z-offsets",
        type=float,
        nargs="*",
        default=[0.0, 0.015],
        help="Also lift the short pre-place by these extra world-z offsets. This helps avoid under-table or desk-edge sweeps without changing the final release pose.",
    )
    parser.add_argument(
        "--direct-grasp-object-axis-shifts-m",
        type=float,
        nargs="*",
        default=[0.0, -0.012, 0.012, -0.024, 0.024],
        help="Also try shifting the grasp TCP along the source object's longest axis by these distances to search for grasp placements that yield a better downstream place chain.",
    )
    parser.add_argument(
        "--direct-elongated-grasp-tilt-toward-robot-deg",
        type=float,
        nargs="*",
        default=[0.0, 12.0, 20.0],
        help="For elongated-object grasp modes, also try these tilt-toward-robot angles when searching for grasp branches that better support downstream place.",
    )
    parser.add_argument(
        "--direct-elongated-grasp-tilt-toward-robot-shift-m",
        type=float,
        nargs="*",
        default=[0.0, 0.01, 0.02],
        help="For elongated-object tilt grasp variants, also try these small XY shifts toward the robot.",
    )
    parser.add_argument(
        "--direct-min-place-tcp-z",
        type=float,
        default=0.005,
        help="Reject any direct pre-place/place candidate whose TCP world z falls below this threshold.",
    )
    parser.add_argument(
        "--direct-pre-place-verticality-band",
        type=float,
        default=0.08,
        help="Among successful insert_vertical pre-place candidates, keep only those within this tcp_verticality band from the best vertical candidate before selecting the lowest-cost cuRobo path.",
    )
    parser.add_argument(
        "--direct-terminal-max-realized-pos-error",
        type=float,
        default=0.010,
        help="Reject a cuRobo candidate if the realized sim TCP position error after terminal snap exceeds this threshold.",
    )
    parser.add_argument(
        "--direct-terminal-max-realized-rot-error-deg",
        type=float,
        default=12.0,
        help="Reject a cuRobo candidate if the realized sim TCP orientation error after terminal snap exceeds this threshold.",
    )
    parser.add_argument(
        "--direct-terminal-snap-max-joint-delta",
        type=float,
        default=0.10,
        help="Maximum per-joint interpolation step for the short collision-checked terminal snap from the cuRobo endpoint to the sim terminal IK.",
    )
    parser.add_argument(
        "--direct-release-cartesian-step-m",
        type=float,
        default=0.005,
        help="Cartesian step size for the pure-cuRobo short constrained release descent.",
    )
    parser.add_argument(
        "--joint-search-max-grasp-candidates",
        type=int,
        default=0,
        help="During joint grasp->place search, only expand this many top-ranked grasp candidates with downstream place feasibility checks. Set <=0 to search all grasp candidates.",
    )
    parser.add_argument(
        "--joint-search-max-pre-place-candidates",
        type=int,
        default=0,
        help="During joint search, only expand this many heuristic-ranked pre-place candidates per grasp before running cuRobo. Set <=0 to search all pre-place candidates.",
    )
    parser.add_argument(
        "--joint-search-max-feasible-chains",
        type=int,
        default=1,
        help="Stop joint search after collecting this many feasible grasp->pre_place->release chains. Set <=0 to keep searching all candidates.",
    )
    parser.add_argument(
        "--joint-search-screen-timeout",
        type=float,
        default=0.0,
        help="Use this shorter cuRobo timeout during joint-search screening before the final selected chain is executed. Set <=0 to reuse the normal cuRobo timeout.",
    )
    parser.add_argument(
        "--joint-search-screen-max-attempts",
        type=int,
        default=0,
        help="Use this smaller max_attempts during joint-search screening. Set <=0 to reuse the normal cuRobo value.",
    )
    parser.add_argument(
        "--joint-search-screen-num-ik-seeds",
        type=int,
        default=0,
        help="Use this smaller IK seed count during joint-search screening. Set <=0 to reuse the normal cuRobo value.",
    )
    parser.add_argument(
        "--joint-search-screen-num-trajopt-seeds",
        type=int,
        default=0,
        help="Use this smaller trajopt seed count during joint-search screening. Set <=0 to reuse the normal cuRobo value.",
    )
    parser.add_argument(
        "--joint-search-goalset-max-winners",
        type=int,
        default=0,
        help="When using cuRobo goalset screening for pre-place during joint search, keep extracting at most this many successful winners before moving on to release checks. Set <=0 to exhaust the goalset.",
    )
    parser.add_argument(
        "--direct-grasp-goalset-max-winners",
        type=int,
        default=0,
        help="When screening grasp candidates with cuRobo goalset, keep extracting at most this many successful winners. Set <=0 to exhaust the goalset.",
    )
    parser.add_argument(
        "--curobo-batch-chunk-size",
        type=int,
        default=64,
        help="Evaluate multi-candidate cuRobo batches in chunks of this size to reduce peak VRAM use. Set <=0 to run all candidates in one batch.",
    )
    return parser


def parse_args():
    return build_arg_parser().parse_args()


def _skip_post_grasp_escape(demo, bridge_mod, real_exec, args, label: str, *, use_attach: bool) -> bool:
    print(
        f"[direct_pre_place] skipped {label}; planning directly from the current grasped pose "
        "to pre_place"
    )
    return True


def _normalize(vec):
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return None
    return (arr / norm).astype(np.float32)


def _unique_finite_float_list(values, *, min_value: float | None = None):
    cleaned = []
    seen = set()
    for value in list(values or []):
        try:
            f = float(value)
        except Exception:
            continue
        if not np.isfinite(f):
            continue
        if min_value is not None and f < float(min_value):
            continue
        key = round(f, 6)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(f)
    return cleaned


def _build_active_object_curobo_world(demo, args):
    asset_file = getattr(args, "sim_asset_file", None) or getattr(args, "mesh_file", None)
    asset_scale = getattr(args, "sim_asset_scale", None)
    if asset_scale is None:
        asset_scale = getattr(args, "mesh_scale", 1.0)
    try:
        obj_p, obj_q = demo.get_obj_pose()
    except Exception:
        return [], []
    pose = np.concatenate(
        [
            np.asarray(obj_p, dtype=np.float32).reshape(3),
            _normalize_quat_wxyz(obj_q),
        ]
    ).astype(np.float32)
    if asset_file is not None:
        asset_path = str(Path(str(asset_file)).expanduser())
        if Path(asset_path).exists():
            scale = float(asset_scale or 1.0)
            return [], [
                {
                    "name": "active_target_object",
                    "file_path": asset_path,
                    "scale": [scale, scale, scale],
                    "pose": pose.tolist(),
                }
            ]
    try:
        dims = targeted.base.get_asset_box_size(asset_file, float(asset_scale or 1.0))
    except Exception:
        return [], []
    return [
        {
            "name": "active_target_object",
            "dims": np.asarray(dims, dtype=np.float32).reshape(3).tolist(),
            "pose": pose.tolist(),
        }
    ], []


def _refresh_curobo_world(planner, demo, args, *, label: str, include_active_object: bool = False) -> None:
    if planner.collision_enabled:
        cuboids, meshes = curobo_wrapper._scene_obstacles_to_curobo_world(demo, args)
        if include_active_object:
            active_cuboids, active_meshes = _build_active_object_curobo_world(demo, args)
            cuboids.extend(active_cuboids)
            meshes.extend(active_meshes)
        cuboids_in_base, meshes_in_base = curobo_wrapper._transform_curobo_world_to_robot_base(
            cuboids,
            meshes,
            demo,
        )
        planner.set_world_from_obstacles(cuboids=cuboids_in_base, meshes=meshes_in_base)
        if bool(getattr(args, "curobo_debug", False)):
            print(
                f"[curobo] updated world with {len(cuboids_in_base)} cuboid and "
                f"{len(meshes_in_base)} mesh obstacles for {label}"
            )
    elif bool(getattr(args, "curobo_debug", False)):
        print(f"[curobo] planner is in embedded free-space mode for {label}")


def _path_metrics_and_score(q_start, q_path):
    metrics = targeted.base._joint_path_quality_metrics(q_start, q_path)
    score = targeted.base._joint_path_quality_score(metrics)
    return metrics, float(score)


def _candidate_sort_key(item):
    metrics = item["metrics"]
    return (
        float(item["score"]),
        float(metrics["total_motion"]),
        float(metrics["joint7_total_motion"]),
        int(metrics["waypoint_count"]),
    )


def _pre_place_screen_sort_key(item):
    label = "" if item.get("label") is None else str(item.get("label"))
    variant = "" if item.get("variant_label") is None else str(item.get("variant_label"))
    pose_z = float(_get_pose_position(item["pose"])[2]) if "pose" in item else 0.0
    place_z = float(_get_pose_position(item["place_pose"])[2]) if "place_pose" in item else 0.0
    return (
        -float(item.get("tcp_verticality", 0.0)),
        -pose_z,
        -place_z,
        variant,
        label,
    )


def _normalize_quat_wxyz(quat):
    arr = np.asarray(quat, dtype=np.float32).reshape(-1)[:4]
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (arr / norm).astype(np.float32)


def _quat_angle_rad_wxyz(q_a, q_b) -> float:
    q_a = _normalize_quat_wxyz(q_a)
    q_b = _normalize_quat_wxyz(q_b)
    dot = float(np.clip(np.abs(np.dot(q_a, q_b)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def _measure_realized_tcp_error(demo, q_arm, pose):
    q_saved = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    try:
        targeted.base.sync_demo_arm_qpos(demo, q_arm)
        actual_p = targeted.base.flatten_np(demo.tcp.pose.p)[:3].astype(np.float32)
        actual_q = _normalize_quat_wxyz(targeted.base.flatten_np(demo.tcp.pose.q)[:4])
    finally:
        targeted.base.sync_demo_arm_qpos(demo, q_saved)
    target_p = targeted.base.flatten_np(pose.p)[:3].astype(np.float32)
    target_q = _normalize_quat_wxyz(targeted.base.flatten_np(pose.q)[:4])
    pos_err = float(np.linalg.norm(actual_p - target_p))
    rot_err = _quat_angle_rad_wxyz(actual_q, target_q)
    return {
        "actual_p": actual_p,
        "actual_q": actual_q,
        "target_p": target_p,
        "target_q": target_q,
        "pos_err": pos_err,
        "rot_err_rad": rot_err,
        "rot_err_deg": float(np.degrees(rot_err)),
    }


def _get_pose_position(pose) -> np.ndarray:
    return targeted.base.flatten_np(pose.p)[:3].astype(np.float32)


def _terminal_error_within_limits(diag, args) -> bool:
    max_pos_err = float(max(getattr(args, "direct_terminal_max_realized_pos_error", 0.010), 0.0))
    max_rot_err_deg = float(max(getattr(args, "direct_terminal_max_realized_rot_error_deg", 12.0), 0.0))
    return float(diag["pos_err"]) <= max_pos_err and float(diag["rot_err_deg"]) <= max_rot_err_deg


def _get_robot_link_by_name(demo, link_name: str):
    links_map = getattr(demo.robot, "links_map", None)
    if isinstance(links_map, dict) and link_name in links_map:
        return links_map[link_name]
    for link in list(demo.robot.get_links()):
        get_name = getattr(link, "get_name", None)
        name = str(get_name()) if callable(get_name) else str(getattr(link, "name", ""))
        if name == link_name:
            return link
    return None


def _pose_to_matrix_from_pose_obj(pose) -> np.ndarray:
    p = targeted.base.flatten_np(pose.p)[:3].astype(np.float32)
    q = targeted.base.flatten_np(pose.q)[:4].astype(np.float32)
    return targeted.base.pose_to_matrix(p, q)


def _get_robot_base_world_transform(demo) -> np.ndarray:
    return _pose_to_matrix_from_pose_obj(demo.robot.pose)


def _get_link_name(link) -> str:
    get_name = getattr(link, "get_name", None)
    if callable(get_name):
        try:
            return str(get_name())
        except Exception:
            pass
    return str(getattr(link, "name", ""))


def _convert_demo_tcp_pose_to_curobo_ee_pose(demo, pose, *, ee_link_name: str):
    ee_link = _get_robot_link_by_name(demo, ee_link_name)
    T_world_base = _get_robot_base_world_transform(demo)
    T_base_world = np.linalg.inv(T_world_base)
    T_world_demo_tcp = _pose_to_matrix_from_pose_obj(demo.tcp.pose)
    T_world_ee_link = T_world_demo_tcp if ee_link is None else _pose_to_matrix_from_pose_obj(ee_link.pose)
    T_demo_tcp_to_ee_link = np.linalg.inv(T_world_demo_tcp) @ T_world_ee_link
    T_goal_demo_tcp = _pose_to_matrix_from_pose_obj(pose)
    T_world_goal_ee_link = T_goal_demo_tcp @ T_demo_tcp_to_ee_link
    T_base_goal_ee_link = T_base_world @ T_world_goal_ee_link
    return targeted._pose_from_matrix(T_base_goal_ee_link.astype(np.float32))


def _nlerp_quat_wxyz(q0, q1, alpha: float) -> np.ndarray:
    q0 = _normalize_quat_wxyz(q0)
    q1 = _normalize_quat_wxyz(q1)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    q = (1.0 - float(alpha)) * q0 + float(alpha) * q1
    return _normalize_quat_wxyz(q)


def _interpolate_demo_tcp_pose(pose_start, pose_goal, alpha: float):
    p0 = targeted.base.flatten_np(pose_start.p)[:3].astype(np.float32)
    p1 = targeted.base.flatten_np(pose_goal.p)[:3].astype(np.float32)
    q0 = targeted.base.flatten_np(pose_start.q)[:4].astype(np.float32)
    q1 = targeted.base.flatten_np(pose_goal.q)[:4].astype(np.float32)
    p = ((1.0 - float(alpha)) * p0 + float(alpha) * p1).astype(np.float32)
    q = _nlerp_quat_wxyz(q0, q1, alpha)
    return targeted.Pose.create_from_pq(p=p, q=q)


def _infer_goal_frame_free_linear_axis(pose_start, pose_goal):
    T_world_start = targeted.base.pose_to_matrix(
        targeted.base.flatten_np(pose_start.p)[:3],
        targeted.base.flatten_np(pose_start.q)[:4],
    ).astype(np.float32)
    T_world_goal = targeted.base.pose_to_matrix(
        targeted.base.flatten_np(pose_goal.p)[:3],
        targeted.base.flatten_np(pose_goal.q)[:4],
    ).astype(np.float32)
    T_goal_start = np.linalg.inv(T_world_goal) @ T_world_start
    delta_goal = np.asarray(T_goal_start[:3, 3], dtype=np.float32).reshape(3)
    free_axis = int(np.argmax(np.abs(delta_goal)))
    locked_delta = np.delete(delta_goal, free_axis)
    rot_trace = float(np.trace(T_goal_start[:3, :3]))
    cos_theta = float(np.clip((rot_trace - 1.0) * 0.5, -1.0, 1.0))
    rot_err_deg = float(np.degrees(np.arccos(cos_theta)))
    return free_axis, delta_goal, locked_delta, rot_err_deg


def _plan_release_with_motiongen_constraint(
    planner,
    demo,
    args,
    start_q,
    pose_start,
    pose_goal,
    *,
    label: str,
):
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    free_axis, delta_goal, locked_delta, rot_err_deg = _infer_goal_frame_free_linear_axis(pose_start, pose_goal)
    if rot_err_deg > 5.0 or float(np.max(np.abs(locked_delta))) > 0.01:
        print(
            f"[curobo] {label} cannot use constrained MotionGen release cleanly "
            f"(goal-frame rot_err={rot_err_deg:.2f} deg, locked_delta={np.round(locked_delta, 6)})"
        )
        return None

    hold_vec_weight = planner.tensor_args.to_device([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    hold_vec_weight[3 + free_axis] = 0.0
    metric = planner.mods["PoseCostMetric"](
        hold_partial_pose=True,
        hold_vec_weight=hold_vec_weight,
        project_to_goal_frame=True,
    )
    planner_pose = _convert_demo_tcp_pose_to_curobo_ee_pose(
        demo,
        pose_goal,
        ee_link_name=str(getattr(planner.config, "ee_link", "gripper_tcp")),
    )
    print(
        f"[curobo] {label} trying constrained MotionGen release "
        f"(free_goal_axis={free_axis}, goal_frame_delta={np.round(delta_goal, 6)})"
    )
    result = planner.plan_to_pose(
        start_q,
        planner_pose,
        enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
        max_attempts=int(getattr(args, "curobo_max_attempts", 2)),
        timeout=float(getattr(args, "curobo_timeout", 5.0)),
        num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
        num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
        num_graph_seeds=int(getattr(args, "curobo_num_graph_seeds", 1)),
        pose_cost_metric=metric,
    )
    if not result.success or result.joint_path is None:
        print(f"[curobo] {label} constrained MotionGen release failed with status={result.status}")
        return None
    q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in result.joint_path]
    final_diag = _measure_realized_tcp_error(demo, q_path[-1], pose_goal)
    print(
        f"[curobo] {label} constrained MotionGen realized tcp: "
        f"pos_err={final_diag['pos_err']:.4f} m, rot_err={final_diag['rot_err_deg']:.2f} deg"
    )
    if not _terminal_error_within_limits(final_diag, args):
        print(f"[curobo] {label} constrained MotionGen misses the release pose too much")
        return None
    return q_path


def _plan_short_curobo_cartesian_descent(
    planner,
    demo,
    args,
    start_q,
    pose_start,
    pose_goal,
    *,
    label: str,
):
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    motiongen_path = _plan_release_with_motiongen_constraint(
        planner,
        demo,
        args,
        start_q,
        pose_start,
        pose_goal,
        label=label,
    )
    if motiongen_path is not None:
        return motiongen_path

    p0 = targeted.base.flatten_np(pose_start.p)[:3].astype(np.float32)
    p1 = targeted.base.flatten_np(pose_goal.p)[:3].astype(np.float32)
    distance = float(np.linalg.norm(p1 - p0))
    step_m = float(max(getattr(args, "direct_release_cartesian_step_m", 0.005), 1e-3))
    num_segments = max(2, int(np.ceil(distance / step_m)))
    print(
        f"[curobo] {label} constrained cartesian descent with {num_segments} segments "
        f"(distance={distance:.4f} m)"
    )

    q_prev = start_q.copy()
    q_path = []
    max_joint_delta = 0.20
    max_joint7_delta = 0.25
    max_norm_delta = 0.35
    for seg_idx, alpha in enumerate(np.linspace(0.0, 1.0, num_segments + 1)[1:], start=1):
        interp_pose = _interpolate_demo_tcp_pose(pose_start, pose_goal, float(alpha))
        planner_pose = _convert_demo_tcp_pose_to_curobo_ee_pose(
            demo,
            interp_pose,
            ee_link_name=str(getattr(planner.config, "ee_link", "gripper_tcp")),
        )
        ik_result = planner.solve_ik(
            q_prev,
            planner_pose,
            num_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
        )
        print(f"[curobo] {label} segment={seg_idx}/{num_segments} ik_success={ik_result.success}")
        if not ik_result.success or ik_result.goal_joint is None:
            return None
        q_next = np.asarray(ik_result.goal_joint, dtype=np.float32).reshape(-1)[:7]
        dq = np.abs(q_next - q_prev)
        if (
            float(np.max(dq)) > max_joint_delta
            or float(dq[6]) > max_joint7_delta
            or float(np.linalg.norm(q_next - q_prev)) > max_norm_delta
        ):
            print(
                f"[curobo] {label} segment={seg_idx}/{num_segments} rejected non-local IK step "
                f"(max_joint_delta={float(np.max(dq)):.3f}, "
                f"joint7_delta={float(dq[6]):.3f}, "
                f"norm_delta={float(np.linalg.norm(q_next - q_prev)):.3f})"
            )
            return None
        q_path.append(q_next)
        q_prev = q_next

    final_diag = _measure_realized_tcp_error(demo, q_path[-1], pose_goal)
    print(
        f"[curobo] {label} realized tcp after constrained descent: "
        f"pos_err={final_diag['pos_err']:.4f} m, rot_err={final_diag['rot_err_deg']:.2f} deg"
    )
    if not _terminal_error_within_limits(final_diag, args):
        print(f"[curobo] {label} constrained descent misses the release pose too much")
        return None
    return q_path


def _validate_curobo_terminal_pose(demo, args, q_path, pose, *, label: str):
    q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in list(q_path or [])]
    if not q_path:
        return None

    realized = _measure_realized_tcp_error(demo, q_path[-1], pose)
    print(
        f"[curobo] {label} realized tcp: "
        f"pos_err={realized['pos_err']:.4f} m, "
        f"rot_err={realized['rot_err_deg']:.2f} deg"
    )
    if not _terminal_error_within_limits(realized, args):
        print(f"[curobo] {label} terminal pose misses the target pose too much; rejecting this candidate")
        return None

    return {
        "q_path": q_path,
        "realized_before": realized,
        "realized_after": realized,
        "snap_waypoints": 0,
        "terminal_q": q_path[-1],
    }


def _evaluate_curobo_pose_candidates(
    planner,
    demo,
    args,
    start_q,
    candidates,
    *,
    label: str,
    prefer_verticality: bool = False,
    use_attach: bool = False,
    timeout: float | None = None,
    max_attempts: int | None = None,
    num_ik_seeds: int | None = None,
    num_trajopt_seeds: int | None = None,
    num_graph_seeds: int | None = None,
    include_active_object: bool = False,
):
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    _refresh_curobo_world(planner, demo, args, label=label, include_active_object=include_active_object)
    if bool(getattr(args, "curobo_debug", False)):
        ee_link_name = str(getattr(planner.config, "ee_link", "gripper_tcp"))
        ee_link = _get_robot_link_by_name(demo, ee_link_name)
        base_pose = targeted.base.flatten_np(demo.robot.pose.p)[:3]
        print(
            f"[curobo] runtime frames for {label}: "
            f"planner_ee_link={ee_link_name}, "
            f"demo_tcp_link={_get_link_name(demo.tcp)}, "
            f"robot_base_world_p={np.round(base_pose, 6)}"
        )
        if ee_link is not None:
            print(
                f"[curobo] runtime link check for {label}: "
                f"ee_link_world_p={np.round(targeted.base.flatten_np(ee_link.pose.p)[:3], 6)}, "
                f"tcp_world_p={np.round(targeted.base.flatten_np(demo.tcp.pose.p)[:3], 6)}"
            )
    successes = []
    for candidate in list(candidates or []):
        candidate_label = str(candidate["label"])
        pose = candidate["pose"]
        planner_pose = _convert_demo_tcp_pose_to_curobo_ee_pose(
            demo,
            pose,
            ee_link_name=str(getattr(planner.config, "ee_link", "gripper_tcp")),
        )
        print(f"[curobo] trying {candidate_label}")
        print(f"[curobo] {candidate_label} pose p: {np.round(targeted.base.flatten_np(pose.p)[:3], 6)}")
        print(f"[curobo] {candidate_label} pose q: {np.round(targeted.base.flatten_np(pose.q)[:4], 6)}")
        if bool(getattr(args, "curobo_debug", False)):
            print(f"[curobo] {candidate_label} ee_link(base) pose p: {np.round(targeted.base.flatten_np(planner_pose.p)[:3], 6)}")
            print(f"[curobo] {candidate_label} ee_link(base) pose q: {np.round(targeted.base.flatten_np(planner_pose.q)[:4], 6)}")
        result = planner.plan_to_pose(
            start_q,
            planner_pose,
            enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
            max_attempts=int(getattr(args, "curobo_max_attempts", 2) if max_attempts is None else max_attempts),
            timeout=float(getattr(args, "curobo_timeout", 5.0) if timeout is None else timeout),
            num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64) if num_ik_seeds is None else num_ik_seeds),
            num_trajopt_seeds=int(
                getattr(args, "curobo_num_trajopt_seeds", 1) if num_trajopt_seeds is None else num_trajopt_seeds
            ),
            num_graph_seeds=int(
                getattr(args, "curobo_num_graph_seeds", 1) if num_graph_seeds is None else num_graph_seeds
            ),
        )
        if not result.success or result.joint_path is None:
            print(
                f"[curobo] {candidate_label} failed with status={result.status}, "
                f"internal_ik={result.debug.get('motion_gen_internal_ik_successes')}"
            )
            if str(result.status) == "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION":
                targeted.base.print_failure_diagnostics(
                    demo,
                    args,
                    f"{candidate_label}_start_state",
                    q_target=start_q,
                    use_attach=use_attach,
                )
                start_diag = planner.diagnose_start_state_world_collision(start_q)
                for item in list(start_diag.get("ablation", []) or []):
                    print(
                        f"[curobo] start-state ablation remove={item['removed']}: "
                        f"valid={item['valid']}, status={item['status']}"
                    )
            continue
        q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in result.joint_path]
        terminal_align = _validate_curobo_terminal_pose(
            demo,
            args,
            q_path,
            pose,
            label=candidate_label,
        )
        if terminal_align is None:
            continue
        q_path = terminal_align["q_path"]
        metrics, score = _path_metrics_and_score(start_q, q_path)
        item = dict(candidate)
        item["result"] = result
        item["q_path"] = q_path
        item["metrics"] = metrics
        item["score"] = score
        item["terminal_align"] = terminal_align
        item["planner_pose"] = planner_pose
        print(
            f"[curobo] {candidate_label} success: score={score:.3f}, "
            f"total_motion={metrics['total_motion']:.3f} rad, "
            f"joint7_total={metrics['joint7_total_motion']:.3f} rad, "
            f"joint7_excursion={metrics['joint7_max_excursion']:.3f} rad, "
            f"waypoints={metrics['waypoint_count']}, "
            f"realized_pos_err={terminal_align['realized_after']['pos_err']:.4f} m, "
            f"realized_rot_err={terminal_align['realized_after']['rot_err_deg']:.2f} deg"
        )
        successes.append(item)

    if not successes:
        return []

    if prefer_verticality:
        best_verticality = max(float(item.get("tcp_verticality", 0.0)) for item in successes)
        band = float(max(getattr(args, "direct_pre_place_verticality_band", 0.08), 0.0))
        filtered = [item for item in successes if float(item.get("tcp_verticality", 0.0)) >= best_verticality - band]
        if filtered:
            successes = filtered
        print(
            f"[direct_pre_place] kept {len(successes)} pre-place candidate(s) within tcp_verticality "
            f"{best_verticality - band:.3f}..{best_verticality:.3f}"
        )

    successes.sort(key=_candidate_sort_key)
    best = successes[0]
    print(
        f"[curobo] selected {best['label']}: score={best['score']:.3f}, "
        f"tcp_verticality={float(best.get('tcp_verticality', 0.0)):.3f}, "
        f"waypoints={best['metrics']['waypoint_count']}"
    )
    return successes


def _evaluate_curobo_pose_candidates_goalset(
    planner,
    demo,
    args,
    start_q,
    candidates,
    *,
    label: str,
    prefer_verticality: bool = False,
    use_attach: bool = False,
    timeout: float | None = None,
    max_attempts: int | None = None,
    num_ik_seeds: int | None = None,
    num_trajopt_seeds: int | None = None,
    num_graph_seeds: int | None = None,
    max_winners: int | None = None,
    include_active_object: bool = False,
):
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    _refresh_curobo_world(planner, demo, args, label=label, include_active_object=include_active_object)
    remaining = list(candidates or [])
    if not remaining:
        return []

    if bool(getattr(args, "curobo_debug", False)):
        ee_link_name = str(getattr(planner.config, "ee_link", "gripper_tcp"))
        ee_link = _get_robot_link_by_name(demo, ee_link_name)
        base_pose = targeted.base.flatten_np(demo.robot.pose.p)[:3]
        print(
            f"[curobo] runtime frames for {label}: "
            f"planner_ee_link={ee_link_name}, "
            f"demo_tcp_link={_get_link_name(demo.tcp)}, "
            f"robot_base_world_p={np.round(base_pose, 6)}"
        )
        if ee_link is not None:
            print(
                f"[curobo] runtime link check for {label}: "
                f"ee_link_world_p={np.round(targeted.base.flatten_np(ee_link.pose.p)[:3], 6)}, "
                f"tcp_world_p={np.round(targeted.base.flatten_np(demo.tcp.pose.p)[:3], 6)}"
            )

    if prefer_verticality:
        best_verticality = max(float(item.get("tcp_verticality", 0.0)) for item in remaining)
        band = float(max(getattr(args, "direct_pre_place_verticality_band", 0.08), 0.0))
        filtered = [item for item in remaining if float(item.get("tcp_verticality", 0.0)) >= best_verticality - band]
        if filtered:
            remaining = filtered
        print(
            f"[direct_pre_place] prefiltered {len(remaining)} goalset candidate(s) within tcp_verticality "
            f"{best_verticality - band:.3f}..{best_verticality:.3f}"
        )

    winners = []
    use_max_winners = int(len(remaining) if max_winners is None else max_winners)
    if use_max_winners <= 0:
        use_max_winners = len(remaining)
    saw_invalid_start = False
    chunk_size = int(getattr(args, "curobo_batch_chunk_size", 64))
    if chunk_size <= 0:
        chunk_size = len(remaining)
    num_chunks = max(1, (len(remaining) + chunk_size - 1) // chunk_size)
    print(
        f"[curobo] {label} evaluating {len(remaining)} candidate(s) "
        f"in {num_chunks} batch chunk(s) of size <= {chunk_size}"
    )
    ee_link_name = str(getattr(planner.config, "ee_link", "gripper_tcp"))

    for chunk_idx, start_idx in enumerate(range(0, len(remaining), chunk_size), start=1):
        chunk = remaining[start_idx : start_idx + chunk_size]
        planner_poses = [
            _convert_demo_tcp_pose_to_curobo_ee_pose(
                demo,
                item["pose"],
                ee_link_name=ee_link_name,
            )
            for item in chunk
        ]
        if num_chunks > 1:
            print(
                f"[curobo] {label} batch chunk {chunk_idx}/{num_chunks}: "
                f"{len(chunk)} candidate(s)"
            )
        batch_results = planner.plan_batch_to_poses(
            start_q,
            planner_poses,
            enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
            max_attempts=int(getattr(args, "curobo_max_attempts", 2) if max_attempts is None else max_attempts),
            timeout=float(getattr(args, "curobo_timeout", 5.0) if timeout is None else timeout),
            num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64) if num_ik_seeds is None else num_ik_seeds),
            num_trajopt_seeds=int(
                getattr(args, "curobo_num_trajopt_seeds", 1) if num_trajopt_seeds is None else num_trajopt_seeds
            ),
            num_graph_seeds=int(
                getattr(args, "curobo_num_graph_seeds", 1) if num_graph_seeds is None else num_graph_seeds
            ),
        )
        for candidate, planner_pose, result in zip(chunk, planner_poses, batch_results):
            candidate_label = str(candidate["label"])
            if not result.success or result.joint_path is None:
                print(f"[curobo] {candidate_label} batch failed with status={result.status}")
                if str(result.status) == "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION":
                    saw_invalid_start = True
                continue
            q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in result.joint_path]
            terminal_align = _validate_curobo_terminal_pose(
                demo,
                args,
                q_path,
                candidate["pose"],
                label=candidate_label,
            )
            if terminal_align is None:
                continue
            q_path = terminal_align["q_path"]
            metrics, score = _path_metrics_and_score(start_q, q_path)
            item = dict(candidate)
            item["result"] = result
            item["q_path"] = q_path
            item["metrics"] = metrics
            item["score"] = score
            item["terminal_align"] = terminal_align
            item["planner_pose"] = planner_pose
            print(
                f"[curobo] {candidate_label} success: score={score:.3f}, "
                f"total_motion={metrics['total_motion']:.3f} rad, "
                f"joint7_total={metrics['joint7_total_motion']:.3f} rad, "
                f"joint7_excursion={metrics['joint7_max_excursion']:.3f} rad, "
                f"waypoints={metrics['waypoint_count']}, "
                f"realized_pos_err={terminal_align['realized_after']['pos_err']:.4f} m, "
                f"realized_rot_err={terminal_align['realized_after']['rot_err_deg']:.2f} deg"
            )
            winners.append(item)
            if len(winners) >= use_max_winners:
                break
        if len(winners) >= use_max_winners:
            break

    if not winners and saw_invalid_start:
        targeted.base.print_failure_diagnostics(
            demo,
            args,
            f"{label}_start_state",
            q_target=start_q,
            use_attach=use_attach,
        )

    if not winners:
        return []
    winners.sort(key=_candidate_sort_key)
    best = winners[0]
    print(
        f"[curobo] selected {best['label']}: score={best['score']:.3f}, "
        f"tcp_verticality={float(best.get('tcp_verticality', 0.0)):.3f}, "
        f"waypoints={best['metrics']['waypoint_count']}"
    )
    return winners


def _filter_pre_place_candidates_by_verticality(candidates, args):
    remaining = list(candidates or [])
    if not remaining:
        return []
    best_verticality = max(float(item.get("tcp_verticality", 0.0)) for item in remaining)
    band = float(max(getattr(args, "direct_pre_place_verticality_band", 0.08), 0.0))
    filtered = [item for item in remaining if float(item.get("tcp_verticality", 0.0)) >= best_verticality - band]
    if filtered:
        remaining = filtered
    print(
        f"[direct_pre_place] prefiltered {len(remaining)} candidate(s) within tcp_verticality "
        f"{best_verticality - band:.3f}..{best_verticality:.3f}"
    )
    return remaining


def _evaluate_curobo_pose_candidates_multi_start(
    planner,
    demo,
    args,
    candidates,
    *,
    label: str,
    use_attach: bool = False,
    timeout: float | None = None,
    max_attempts: int | None = None,
    num_ik_seeds: int | None = None,
    num_trajopt_seeds: int | None = None,
    num_graph_seeds: int | None = None,
    max_winners: int | None = None,
    include_active_object: bool = False,
):
    _refresh_curobo_world(planner, demo, args, label=label, include_active_object=include_active_object)
    remaining = list(candidates or [])
    if not remaining:
        return []

    if bool(getattr(args, "curobo_debug", False)):
        ee_link_name = str(getattr(planner.config, "ee_link", "gripper_tcp"))
        ee_link = _get_robot_link_by_name(demo, ee_link_name)
        base_pose = targeted.base.flatten_np(demo.robot.pose.p)[:3]
        print(
            f"[curobo] runtime frames for {label}: "
            f"planner_ee_link={ee_link_name}, "
            f"demo_tcp_link={_get_link_name(demo.tcp)}, "
            f"robot_base_world_p={np.round(base_pose, 6)}"
        )
        if ee_link is not None:
            print(
                f"[curobo] runtime link check for {label}: "
                f"ee_link_world_p={np.round(targeted.base.flatten_np(ee_link.pose.p)[:3], 6)}, "
                f"tcp_world_p={np.round(targeted.base.flatten_np(demo.tcp.pose.p)[:3], 6)}"
            )

    winners = []
    use_max_winners = int(len(remaining) if max_winners is None else max_winners)
    if use_max_winners <= 0:
        use_max_winners = len(remaining)
    saw_invalid_start = False
    chunk_size = int(getattr(args, "curobo_batch_chunk_size", 64))
    if chunk_size <= 0:
        chunk_size = len(remaining)
    num_chunks = max(1, (len(remaining) + chunk_size - 1) // chunk_size)
    print(
        f"[curobo] {label} evaluating {len(remaining)} start-goal pair(s) "
        f"in {num_chunks} batch chunk(s) of size <= {chunk_size}"
    )
    ee_link_name = str(getattr(planner.config, "ee_link", "gripper_tcp"))

    for chunk_idx, start_idx in enumerate(range(0, len(remaining), chunk_size), start=1):
        chunk = remaining[start_idx : start_idx + chunk_size]
        planner_poses = [
            _convert_demo_tcp_pose_to_curobo_ee_pose(
                demo,
                item["pose"],
                ee_link_name=ee_link_name,
            )
            for item in chunk
        ]
        start_qs = [np.asarray(item["start_q"], dtype=np.float32).reshape(-1)[:7] for item in chunk]
        if num_chunks > 1:
            print(
                f"[curobo] {label} batch chunk {chunk_idx}/{num_chunks}: "
                f"{len(chunk)} pair(s)"
            )
        batch_results = planner.plan_batch_start_goal_pairs(
            start_qs,
            planner_poses,
            enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
            max_attempts=int(getattr(args, "curobo_max_attempts", 2) if max_attempts is None else max_attempts),
            timeout=float(getattr(args, "curobo_timeout", 5.0) if timeout is None else timeout),
            num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64) if num_ik_seeds is None else num_ik_seeds),
            num_trajopt_seeds=int(
                getattr(args, "curobo_num_trajopt_seeds", 1) if num_trajopt_seeds is None else num_trajopt_seeds
            ),
            num_graph_seeds=int(
                getattr(args, "curobo_num_graph_seeds", 1) if num_graph_seeds is None else num_graph_seeds
            ),
        )
        for candidate, planner_pose, result in zip(chunk, planner_poses, batch_results):
            candidate_label = str(candidate["label"])
            if not result.success or result.joint_path is None:
                print(f"[curobo] {candidate_label} batch failed with status={result.status}")
                if str(result.status) == "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION":
                    saw_invalid_start = True
                continue
            start_q = np.asarray(candidate["start_q"], dtype=np.float32).reshape(-1)[:7]
            q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in result.joint_path]
            terminal_align = _validate_curobo_terminal_pose(
                demo,
                args,
                q_path,
                candidate["pose"],
                label=candidate_label,
            )
            if terminal_align is None:
                continue
            q_path = terminal_align["q_path"]
            metrics, score = _path_metrics_and_score(start_q, q_path)
            item = dict(candidate)
            item["result"] = result
            item["q_path"] = q_path
            item["metrics"] = metrics
            item["score"] = score
            item["terminal_align"] = terminal_align
            item["planner_pose"] = planner_pose
            print(
                f"[curobo] {candidate_label} success: score={score:.3f}, "
                f"total_motion={metrics['total_motion']:.3f} rad, "
                f"joint7_total={metrics['joint7_total_motion']:.3f} rad, "
                f"joint7_excursion={metrics['joint7_max_excursion']:.3f} rad, "
                f"waypoints={metrics['waypoint_count']}, "
                f"realized_pos_err={terminal_align['realized_after']['pos_err']:.4f} m, "
                f"realized_rot_err={terminal_align['realized_after']['rot_err_deg']:.2f} deg"
            )
            winners.append(item)
            if len(winners) >= use_max_winners:
                break
        if len(winners) >= use_max_winners:
            break

    if not winners and saw_invalid_start and remaining:
        targeted.base.print_failure_diagnostics(
            demo,
            args,
            f"{label}_start_state",
            q_target=np.asarray(remaining[0]["start_q"], dtype=np.float32).reshape(-1)[:7],
            use_attach=use_attach,
        )

    winners.sort(key=_candidate_sort_key)
    return winners


def _build_direct_grasp_candidates(demo, args):
    raw_grasp_pose = demo.build_topdown_grasp_pose()
    grasp_mode = str(getattr(args, "grasp_mode", "object_normal") or "object_normal").strip().lower()
    object_axis_world = None
    try:
        extents = np.asarray(
            targeted.base.get_asset_box_size(args.sim_asset_file, args.sim_asset_scale),
            dtype=np.float32,
        ).reshape(3)
        axis_idx = int(np.argmax(extents))
        if float(np.max(extents)) >= 1.5 * float(max(np.sort(extents)[1], 1e-6)):
            T_world_obj = targeted.base.pose_to_matrix(*demo.get_obj_pose())
            axis_local = np.zeros(3, dtype=np.float32)
            axis_local[axis_idx] = 1.0
            object_axis_world = _normalize(T_world_obj[:3, :3] @ axis_local)
    except Exception:
        object_axis_world = None
    grasp_axis_shifts = _unique_finite_float_list(
        getattr(args, "direct_grasp_object_axis_shifts_m", [0.0]),
    )
    if not grasp_axis_shifts:
        grasp_axis_shifts = [0.0]

    elongated_modes = {"topdown_long_axis", "pen_topdown_insert_ready", "long_axis_adaptive"}

    def _pose_key(pose):
        return (
            tuple(np.round(targeted.base.flatten_np(pose.p)[:3], 5).tolist()),
            tuple(np.round(targeted.base.flatten_np(pose.q)[:4], 5).tolist()),
        )

    def _build_pose_variants():
        variants = []
        seen = set()

        def _append(label, pose):
            if pose is None:
                return
            key = _pose_key(pose)
            if key in seen:
                return
            seen.add(key)
            variants.append((str(label), pose))

        for label, pose in targeted.base.build_grasp_pose_variants(demo, raw_grasp_pose, args):
            _append(label, pose)

        if grasp_mode not in elongated_modes:
            return variants

        tilt_degs = _unique_finite_float_list(getattr(args, "direct_elongated_grasp_tilt_toward_robot_deg", [0.0]))
        shift_ds = _unique_finite_float_list(
            getattr(args, "direct_elongated_grasp_tilt_toward_robot_shift_m", [0.0]),
            min_value=0.0,
        )
        if not shift_ds:
            shift_ds = [0.0]

        for tilt_deg in tilt_degs:
            if abs(float(tilt_deg)) <= 1e-6:
                continue
            tilted_pose = targeted.base.tilt_pose_toward_robot(demo, raw_grasp_pose, float(tilt_deg))
            if tilted_pose is None:
                continue
            for shift_d in shift_ds:
                shifted_pose = tilted_pose if float(shift_d) <= 1e-6 else targeted.base.shift_pose_toward_robot_xy(
                    demo,
                    tilted_pose,
                    float(shift_d),
                )
                if shifted_pose is None:
                    continue
                base_label = f"grasp_elong_tilt_{int(round(abs(float(tilt_deg))))}deg"
                if float(shift_d) > 1e-6:
                    base_label += f"_shift_{int(round(1000.0 * float(shift_d)))}mm"
                _append(base_label, shifted_pose)
        return variants

    grasp_variants = _build_pose_variants()
    candidates = []
    seen = set()
    for grasp_variant_label, grasp_variant_pose in grasp_variants:
        for axis_shift in grasp_axis_shifts:
            current_grasp_pose = grasp_variant_pose
            if object_axis_world is not None and abs(float(axis_shift)) > 1e-6:
                shifted_p = (_get_pose_position(current_grasp_pose) + object_axis_world * float(axis_shift)).astype(np.float32)
                current_grasp_pose = targeted.base.make_pose_with_position(current_grasp_pose, shifted_p)
            current_pregrasp_pose = demo.build_pregrasp_pose(current_grasp_pose)
            current_grasp_pose, current_pregrasp_pose, geometry_grasp_raise = targeted.base.enforce_topdown_grasp_insertion_limit(
                demo,
                args,
                current_grasp_pose,
                current_pregrasp_pose,
            )
            if geometry_grasp_raise > 0:
                print(
                    f"[safety] raised grasp TCP z by {geometry_grasp_raise:.4f} m "
                    f"to satisfy topdown_grasp_max_insertion_depth={args.topdown_grasp_max_insertion_depth:.4f}"
                )
            current_grasp_pose, _, grasp_tcp_raise = targeted.base.enforce_min_grasp_tcp_z(
                current_grasp_pose,
                current_pregrasp_pose,
                args.min_grasp_tcp_z,
            )
            if grasp_tcp_raise > 0:
                print(
                    f"[safety] raised grasp TCP z by {grasp_tcp_raise:.4f} m "
                    f"to satisfy min_grasp_tcp_z={args.min_grasp_tcp_z:.4f}"
                )
            key = (
                tuple(np.round(targeted.base.flatten_np(current_grasp_pose.p)[:3], 5).tolist()),
                tuple(np.round(targeted.base.flatten_np(current_grasp_pose.q)[:4], 5).tolist()),
            )
            if key in seen:
                continue
            seen.add(key)
            label = "grasp_direct" if grasp_variant_label in (None, "grasp") else f"grasp_direct_{grasp_variant_label}"
            if abs(float(axis_shift)) > 1e-6:
                label = f"{label}_axis_{int(round(float(axis_shift) * 1000.0))}mm"
            candidates.append({"label": label, "pose": current_grasp_pose})
    print(
        f"[direct_grasp] built {len(candidates)} grasp candidate(s) "
        f"from {len(grasp_variants)} pose variant(s) x {len(grasp_axis_shifts)} axis shift(s); "
        f"grasp_mode={grasp_mode}"
    )
    return candidates


def _make_short_pre_place_pose(candidate, target_axis: np.ndarray | None, approach_distance: float):
    if target_axis is None:
        return candidate.pre_place_pose
    place_p = targeted.base.flatten_np(candidate.place_pose.p)[:3].astype(np.float32)
    pre_place_p = (place_p + target_axis * float(approach_distance)).astype(np.float32)
    return targeted.base.make_pose_with_position(candidate.place_pose, pre_place_p)


def _dedupe_place_candidates(candidates):
    deduped = []
    seen = set()
    for item in list(candidates or []):
        pose = item["pose"]
        place_pose = item["place_pose"]
        key = (
            tuple(np.round(targeted.base.flatten_np(pose.p)[:3], 5).tolist()),
            tuple(np.round(targeted.base.flatten_np(pose.q)[:4], 5).tolist()),
            tuple(np.round(targeted.base.flatten_np(place_pose.p)[:3], 5).tolist()),
            tuple(np.round(targeted.base.flatten_np(place_pose.q)[:4], 5).tolist()),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _build_direct_pre_place_candidates(demo, bridge_mod, scene_capture_cache, rule, place_state_cache, args):
    place_plan_candidates = targeted.build_targeted_place_plan_variants(
        demo,
        bridge_mod,
        scene_capture_cache,
        rule,
        place_state_cache,
        args,
    )
    if not place_plan_candidates:
        return []

    target_axis = None
    if rule.primitive == "insert_vertical":
        T_world_target = targeted._get_scene_object_world_transform(
            demo,
            bridge_mod,
            scene_capture_cache,
            rule.target_object_name,
        )
        if T_world_target is not None:
            target_axis = _normalize(np.asarray(T_world_target[:3, 1], dtype=np.float32).reshape(3))
            sign = 1.0 if float(getattr(rule.object_pose_local, "position", (0.0, 1.0, 0.0))[1]) >= 0.0 else -1.0
            if target_axis is not None:
                target_axis = (target_axis * sign).astype(np.float32)

    approach_distances = _unique_finite_float_list(
        getattr(args, "direct_release_approach_distances", []),
        min_value=0.005,
    )
    if not approach_distances:
        approach_distances = [float(max(getattr(args, "direct_release_approach_distance", 0.04), 0.005))]
    z_offsets = _unique_finite_float_list(getattr(args, "direct_pre_place_z_offsets", [0.0]))
    if not z_offsets:
        z_offsets = [0.0]
    min_place_tcp_z = float(getattr(args, "direct_min_place_tcp_z", 0.005))
    candidates = []
    for candidate in place_plan_candidates:
        place_z = float(_get_pose_position(candidate.place_pose)[2])
        if place_z < min_place_tcp_z:
            print(
                f"[direct_pre_place] reject {candidate.variant_label or 'base'} because place tcp z={place_z:.4f} "
                f"is below direct_min_place_tcp_z={min_place_tcp_z:.4f}"
            )
            continue
        for approach_distance in approach_distances:
            short_pre_place_pose = _make_short_pre_place_pose(candidate, target_axis, float(approach_distance))
            for z_offset in z_offsets:
                pose = short_pre_place_pose
                if abs(float(z_offset)) > 1e-6:
                    pose = targeted.base.make_pose_with_position(
                        pose,
                        (_get_pose_position(pose) + np.array([0.0, 0.0, float(z_offset)], dtype=np.float32)).astype(np.float32),
                    )
                pose_z = float(_get_pose_position(pose)[2])
                if pose_z < min_place_tcp_z:
                    print(
                        f"[direct_pre_place] reject {candidate.variant_label or 'base'} "
                        f"approach={float(approach_distance):.3f} z_offset={float(z_offset):.3f} "
                        f"because pre_place tcp z={pose_z:.4f} is below direct_min_place_tcp_z={min_place_tcp_z:.4f}"
                    )
                    continue
                label = "pre_place" if candidate.variant_label is None else f"pre_place_{candidate.variant_label}"
                if len(approach_distances) > 1:
                    label = f"{label}_approach_{int(round(float(approach_distance) * 1000.0))}mm"
                if abs(float(z_offset)) > 1e-6:
                    label = f"{label}_lift_{int(round(float(z_offset) * 1000.0))}mm"
                candidates.append(
                    {
                        "label": label,
                        "pose": pose,
                        "place_pose": candidate.place_pose,
                        "retreat_pose": pose,
                        "target_name": candidate.target_name,
                        "slot_name": candidate.slot_name,
                        "variant_label": candidate.variant_label,
                        "tcp_verticality": float(candidate.tcp_verticality),
                        "place_plan": candidate,
                    }
                )
    candidates = _dedupe_place_candidates(candidates)
    print(
        f"[direct_pre_place] built {len(candidates)} short pre-place candidate(s) "
        f"with approach_distances={np.round(np.asarray(approach_distances, dtype=np.float32), 4).tolist()} "
        f"and z_offsets={np.round(np.asarray(z_offsets, dtype=np.float32), 4).tolist()}"
    )
    return candidates


def _build_direct_place_candidates(demo, bridge_mod, scene_capture_cache, rule, place_state_cache, args):
    place_plan_candidates = targeted.build_targeted_place_plan_variants(
        demo,
        bridge_mod,
        scene_capture_cache,
        rule,
        place_state_cache,
        args,
    )
    if not place_plan_candidates:
        return []

    min_place_tcp_z = float(getattr(args, "direct_min_place_tcp_z", 0.005))
    candidates = []
    for candidate in place_plan_candidates:
        place_z = float(_get_pose_position(candidate.place_pose)[2])
        if place_z < min_place_tcp_z:
            print(
                f"[direct_place] reject {candidate.variant_label or 'base'} because place tcp z={place_z:.4f} "
                f"is below direct_min_place_tcp_z={min_place_tcp_z:.4f}"
            )
            continue
        label = "place_direct" if candidate.variant_label is None else f"place_direct_{candidate.variant_label}"
        candidates.append(
            {
                "label": label,
                "pose": candidate.place_pose,
                "place_pose": candidate.place_pose,
                "retreat_pose": candidate.pre_place_pose,
                "target_name": candidate.target_name,
                "slot_name": candidate.slot_name,
                "variant_label": candidate.variant_label,
                "tcp_verticality": float(candidate.tcp_verticality),
                "place_plan": candidate,
                "direct_place_mode": True,
            }
        )
    candidates = _dedupe_place_candidates(candidates)
    print(f"[direct_place] built {len(candidates)} direct place candidate(s)")
    return candidates


def _joint_chain_sort_key(item):
    return (
        float(item["total_score"]),
        float(item["grasp_choice"]["score"]),
        float(item["pre_place_choice"]["score"]),
        float(item["release_score"]),
    )


def _evaluate_joint_grasp_place_chains(
    planner,
    demo,
    bridge_mod,
    args,
    scene_capture_cache,
    place_state_cache,
    rule,
    grasp_successes,
):
    saved_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    targeted._register_transport_attached_box(
        demo,
        args,
        show_visual=False,
        activate_payload_visual=False,
    )
    chains = []
    grasp_candidates = list(grasp_successes or [])
    max_grasp_candidates = int(getattr(args, "joint_search_max_grasp_candidates", 8))
    if max_grasp_candidates > 0:
        grasp_candidates = grasp_candidates[:max_grasp_candidates]
    max_feasible_chains = int(getattr(args, "joint_search_max_feasible_chains", 1))
    screen_timeout = float(getattr(args, "joint_search_screen_timeout", 0.0))
    screen_max_attempts = int(getattr(args, "joint_search_screen_max_attempts", 0))
    screen_num_ik_seeds = int(getattr(args, "joint_search_screen_num_ik_seeds", 0))
    screen_num_trajopt_seeds = int(getattr(args, "joint_search_screen_num_trajopt_seeds", 0))
    screen_timeout_arg = None if screen_timeout <= 0.0 else screen_timeout
    screen_max_attempts_arg = None if screen_max_attempts <= 0 else screen_max_attempts
    screen_num_ik_seeds_arg = None if screen_num_ik_seeds <= 0 else screen_num_ik_seeds
    screen_num_trajopt_seeds_arg = None if screen_num_trajopt_seeds <= 0 else screen_num_trajopt_seeds
    try:
        for grasp_choice in grasp_candidates:
            grasp_label = str(grasp_choice["label"])
            grasp_terminal_q = np.asarray(grasp_choice["q_path"][-1], dtype=np.float32).reshape(-1)[:7]
            targeted.base.sync_demo_arm_qpos(demo, grasp_terminal_q)
            print(
                f"[joint_search] evaluating downstream place feasibility for {grasp_label} "
                f"(grasp_score={grasp_choice['score']:.3f})"
            )
            max_place_candidates = int(getattr(args, "joint_search_max_pre_place_candidates", 16))

            direct_place_candidates = _build_direct_place_candidates(
                demo,
                bridge_mod,
                scene_capture_cache,
                rule,
                place_state_cache,
                args,
            )
            if rule.primitive == "insert_vertical":
                direct_place_candidates = _filter_pre_place_candidates_by_verticality(direct_place_candidates, args)
            direct_place_candidates.sort(key=_pre_place_screen_sort_key)
            if max_place_candidates > 0:
                direct_place_candidates = direct_place_candidates[:max_place_candidates]
            if direct_place_candidates:
                direct_place_successes = _evaluate_curobo_pose_candidates(
                    planner,
                    demo,
                    args,
                    grasp_terminal_q,
                    direct_place_candidates,
                    label=f"joint_direct_place_for_{grasp_label}",
                    prefer_verticality=(rule.primitive == "insert_vertical"),
                    use_attach=True,
                    timeout=screen_timeout_arg,
                    max_attempts=screen_max_attempts_arg,
                    num_ik_seeds=screen_num_ik_seeds_arg,
                    num_trajopt_seeds=screen_num_trajopt_seeds_arg,
                )
            else:
                direct_place_successes = []
            if direct_place_successes:
                for candidate in direct_place_successes:
                    q_pre_place_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in candidate["q_path"]]
                    q_place_candidate = [np.asarray(q_pre_place_path[-1], dtype=np.float32).reshape(-1)[:7]]
                    release_metrics, release_score = _path_metrics_and_score(q_pre_place_path[-1], q_place_candidate)
                    total_score = float(grasp_choice["score"]) + float(candidate["score"]) + float(release_score)
                    chain = {
                        "grasp_choice": grasp_choice,
                        "pre_place_choice": candidate,
                        "q_pre_place_path": q_pre_place_path,
                        "q_place_path": q_place_candidate,
                        "release_metrics": release_metrics,
                        "release_score": float(release_score),
                        "total_score": total_score,
                        "direct_place_mode": True,
                    }
                    print(
                        f"[joint_search] feasible direct-place chain: grasp={grasp_label}, "
                        f"place={candidate['label']}, total_score={total_score:.3f}, "
                        f"waypoints={len(q_pre_place_path)}"
                    )
                    chains.append(chain)
                    if max_feasible_chains > 0 and len(chains) >= max_feasible_chains:
                        break
                if max_feasible_chains > 0 and len(chains) >= max_feasible_chains:
                    break
                continue

            pre_place_candidates = _build_direct_pre_place_candidates(
                demo,
                bridge_mod,
                scene_capture_cache,
                rule,
                place_state_cache,
                args,
            )
            if not pre_place_candidates:
                print(f"[joint_search] {grasp_label}: no targeted place candidate could be built")
                continue
            pre_place_candidates.sort(key=_pre_place_screen_sort_key)
            if max_place_candidates > 0:
                pre_place_candidates = pre_place_candidates[:max_place_candidates]
            if rule.primitive == "insert_vertical":
                pre_place_candidates = _filter_pre_place_candidates_by_verticality(pre_place_candidates, args)
            pre_place_successes = _evaluate_curobo_pose_candidates(
                planner,
                demo,
                args,
                grasp_terminal_q,
                pre_place_candidates,
                label=f"joint_pre_place_for_{grasp_label}",
                prefer_verticality=(rule.primitive == "insert_vertical"),
                use_attach=True,
                timeout=screen_timeout_arg,
                max_attempts=screen_max_attempts_arg,
                num_ik_seeds=screen_num_ik_seeds_arg,
                num_trajopt_seeds=screen_num_trajopt_seeds_arg,
            )
            if not pre_place_successes:
                print(f"[joint_search] {grasp_label}: no pre-place candidate passed cuRobo")
                continue

            pre_place_successes.sort(
                key=lambda item: (
                    float(grasp_choice["score"]) + float(item["score"]),
                    float(item["score"]),
                    float(grasp_choice["score"]),
                )
            )
            for candidate in pre_place_successes:
                q_pre_place_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in candidate["q_path"]]
                q_place_candidate = _plan_short_curobo_cartesian_descent(
                    planner,
                    demo,
                    args,
                    np.asarray(q_pre_place_path[-1], dtype=np.float32).reshape(-1)[:7],
                    candidate["pose"],
                    candidate["place_pose"],
                    label="release" if candidate["variant_label"] is None else f"release_{candidate['variant_label']}",
                )
                if q_place_candidate is None:
                    continue
                release_metrics, release_score = _path_metrics_and_score(q_pre_place_path[-1], q_place_candidate)
                total_score = float(grasp_choice["score"]) + float(candidate["score"]) + float(release_score)
                chain = {
                    "grasp_choice": grasp_choice,
                    "pre_place_choice": candidate,
                    "q_pre_place_path": q_pre_place_path,
                    "q_place_path": q_place_candidate,
                    "release_metrics": release_metrics,
                    "release_score": float(release_score),
                    "total_score": total_score,
                    "direct_place_mode": False,
                }
                print(
                    f"[joint_search] feasible chain: grasp={grasp_label}, "
                    f"pre_place={candidate['label']}, total_score={total_score:.3f}, "
                    f"release_waypoints={release_metrics['waypoint_count']}"
                )
                chains.append(chain)
                if max_feasible_chains > 0 and len(chains) >= max_feasible_chains:
                    break
            if max_feasible_chains > 0 and len(chains) >= max_feasible_chains:
                break
        if not chains:
            print("[joint_search] no direct-place or release descent chain stayed feasible")
    finally:
        targeted.base.sync_demo_arm_qpos(demo, saved_q)

    if not chains:
        return []

    chains.sort(key=_joint_chain_sort_key)
    best = chains[0]
    print(
        f"[joint_search] selected chain: grasp={best['grasp_choice']['label']}, "
        f"pre_place={best['pre_place_choice']['label']}, total_score={best['total_score']:.3f}"
    )
    return chains


def run_targeted_place_episode_curobo_direct(
    demo,
    bridge_mod,
    real_exec,
    args,
    scene_capture_cache,
    place_state_cache,
) -> bool:
    print("\n[episode] planning from FoundationPose-initialized object pose")
    args._skip_remaining_step_confirms_in_object = False

    start_q = demo.current_arm_qpos()
    if real_exec is not None and bool(getattr(args, "single_confirm_per_object", False)):
        if not targeted.base.begin_single_confirm_window_for_object(demo, bridge_mod, args):
            print("[abort] user cancelled before executing this object's pick-place sequence")
            return False
    if real_exec is not None:
        print("\n[real robot setup]")
        if args.render_mode == "human":
            bridge_mod.render_preview(demo.env, repeats=5)
        ok, q_sent = targeted.base.align_real_robot_to_sim_start(demo, bridge_mod, real_exec, start_q, args)
        if not ok:
            print("[abort] failed to align the real robot to the simulation start pose")
            return False
        targeted.base.sync_demo_arm_qpos(demo, q_sent if q_sent is not None else start_q)
    else:
        print("\n[dry-run] --execute-real was not provided, so motions will only be planned and previewed")

    planner = curobo_wrapper._get_or_create_curobo_planner(args)

    rule = None
    selected_joint_chain = None
    if not args.skip_goal_motion:
        rule = targeted.get_place_rule(args.object_name)
        if rule is None:
            print(f"[FAIL] no targeted-place rule is configured for source object {args.object_name}")
            return False

    targeted.base.lift_active_object_above_table_if_needed(
        demo,
        args,
        min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
    )

    print("\n[move to grasp directly with cuRobo]")
    grasp_candidates = _build_direct_grasp_candidates(demo, args)
    print("\n[poses]")
    print("object p:", np.round(demo.get_obj_pose()[0], 6), "object q:", np.round(demo.get_obj_pose()[1], 6))
    grasp_successes = _evaluate_curobo_pose_candidates_goalset(
        planner,
        demo,
        args,
        np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7],
        grasp_candidates,
        label="direct_grasp",
        prefer_verticality=False,
        use_attach=False,
        max_winners=int(getattr(args, "direct_grasp_goalset_max_winners", 12)),
        include_active_object=False,
    )
    if not grasp_successes:
        selected_pose = grasp_candidates[0]["pose"] if grasp_candidates else demo.build_topdown_grasp_pose()
        print("[FAIL] direct cuRobo grasp planning failed")
        targeted.base.inspect_failed_pose(demo, bridge_mod, "grasp", args, pose=selected_pose, gripper_closed=False)
        return False

    grasp_choice = grasp_successes[0]
    if rule is not None:
        joint_chains = _evaluate_joint_grasp_place_chains(
            planner,
            demo,
            bridge_mod,
            args,
            scene_capture_cache,
            place_state_cache,
            rule,
            grasp_successes,
        )
        if not joint_chains:
            print("[FAIL] no grasp candidate yielded a complete grasp->pre_place->release chain")
            targeted.base.inspect_failed_pose(
                demo,
                bridge_mod,
                "grasp",
                args,
                pose=grasp_choice["pose"],
                gripper_closed=False,
            )
            return False
        selected_joint_chain = joint_chains[0]
        grasp_choice = selected_joint_chain["grasp_choice"]

    ok, _ = targeted.base.execute_pose_path_stage(
        demo,
        bridge_mod,
        real_exec,
        grasp_choice["label"],
        grasp_choice["pose"],
        grasp_choice["q_path"],
        args.real_gripper_open,
        args,
    )
    if not ok:
        return False

    print("\n[close gripper]")
    if not targeted.base.confirm_simple_action("close the real gripper", args, bridge_mod=bridge_mod, env=demo.env, repeats=6):
        print("[abort] user cancelled before closing the real gripper")
        return False
    if real_exec is not None:
        real_exec.set_gripper(args.real_gripper_close)
        targeted.base.sync_demo_gripper_state(demo, closed=True, steps=4)
        targeted.base.set_pregrasp_object_freeze(demo, False)
        gripper_pos, blocked = targeted.base.real_gripper_blocked_after_close(
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
        targeted.base.sync_demo_gripper_state(demo, closed=True, steps=4)
        targeted.base.set_pregrasp_object_freeze(demo, False)

    targeted.base.lift_active_object_above_table_if_needed(
        demo,
        args,
        min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
    )

    if args.skip_goal_motion:
        print("[done] skipped place motion as requested")
        return True

    print("\n[move to targeted place]")
    try:
        targeted._ensure_target_registered_for_place(demo, rule.target_object_name)
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return False
    targeted._register_transport_attached_box(demo, args)
    _skip_post_grasp_escape(demo, bridge_mod, real_exec, args, "post_grasp_escape", use_attach=True)

    pre_place_successes = None
    relaxed_target_collision = False
    if selected_joint_chain is None:
        pre_place_candidates = _build_direct_pre_place_candidates(
            demo,
            bridge_mod,
            scene_capture_cache,
            rule,
            place_state_cache,
            args,
        )
        if not pre_place_candidates:
            print("[FAIL] no targeted place candidate could be built")
            return False

        pre_place_successes = _evaluate_curobo_pose_candidates(
            planner,
            demo,
            args,
            np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7],
            pre_place_candidates,
            label="direct_pre_place",
            prefer_verticality=(rule.primitive == "insert_vertical"),
            use_attach=True,
        )
        if not pre_place_successes:
            print("[FAIL] cuRobo pre-place planning failed for all allowed rotation candidates")
            targeted.base.inspect_failed_pose(
                demo,
                bridge_mod,
                "pre_place",
                args,
                pose=pre_place_candidates[0]["pose"],
                gripper_closed=True,
                use_attach=True,
            )
            return False

        place_choice = None
        q_pre_place_path = None
        q_place_path = None
        for candidate in pre_place_successes:
            slot_suffix = f", slot={candidate['slot_name']}" if candidate.get("slot_name") else ""
            variant_suffix = f", variant={candidate['variant_label']}" if candidate.get("variant_label") else ""
            print(
                f"[place] source={args.object_name}, primitive={rule.primitive}, "
                f"target={candidate['target_name']}{slot_suffix}{variant_suffix}, "
                f"tcp_verticality={candidate['tcp_verticality']:.3f}"
            )
            print("[place] pre_place p:", np.round(targeted.base.flatten_np(candidate["pose"].p)[:3], 6), "q:", np.round(targeted.base.flatten_np(candidate["pose"].q)[:4], 6))
            print("[place] place p:", np.round(targeted.base.flatten_np(candidate["place_pose"].p)[:3], 6), "q:", np.round(targeted.base.flatten_np(candidate["place_pose"].q)[:4], 6))
            q_pre_place_path = candidate["q_path"]
            q_place_candidate = _plan_short_curobo_cartesian_descent(
                planner,
                demo,
                args,
                np.asarray(q_pre_place_path[-1], dtype=np.float32).reshape(-1)[:7],
                candidate["pose"],
                candidate["place_pose"],
                label="release" if candidate["variant_label"] is None else f"release_{candidate['variant_label']}",
            )
            if q_place_candidate is None:
                print(f"[place] constrained release descent failed for variant={candidate['variant_label']}; trying the next cuRobo-selected candidate")
                continue
            place_choice = candidate
            q_place_path = q_place_candidate
            if rule.primitive == "insert_vertical":
                relaxed_target_collision = targeted._set_scene_obstacle_planner_box_scale(
                    demo,
                    candidate["target_name"],
                    float(args.place_insert_target_collision_scale),
                )
            break

        if place_choice is None or q_pre_place_path is None or q_place_path is None:
            print("[FAIL] no cuRobo pre-place candidate yielded a valid constrained release descent")
            targeted.base.inspect_failed_pose(
                demo,
                bridge_mod,
                "place",
                args,
                pose=pre_place_successes[0]["place_pose"],
                gripper_closed=True,
                use_attach=True,
            )
            return False
    else:
        place_choice = selected_joint_chain["pre_place_choice"]
        q_pre_place_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in selected_joint_chain["q_pre_place_path"]]
        q_place_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in selected_joint_chain["q_place_path"]]
        direct_place_mode = bool(selected_joint_chain.get("direct_place_mode", False))
        slot_suffix = f", slot={place_choice['slot_name']}" if place_choice.get("slot_name") else ""
        variant_suffix = f", variant={place_choice['variant_label']}" if place_choice.get("variant_label") else ""
        print(
            f"[place] source={args.object_name}, primitive={rule.primitive}, "
            f"target={place_choice['target_name']}{slot_suffix}{variant_suffix}, "
            f"tcp_verticality={place_choice['tcp_verticality']:.3f}"
        )
        print("[place] pre_place p:", np.round(targeted.base.flatten_np(place_choice["pose"].p)[:3], 6), "q:", np.round(targeted.base.flatten_np(place_choice["pose"].q)[:4], 6))
        print("[place] place p:", np.round(targeted.base.flatten_np(place_choice["place_pose"].p)[:3], 6), "q:", np.round(targeted.base.flatten_np(place_choice["place_pose"].q)[:4], 6))
        if direct_place_mode:
            print("[place] selected chain uses direct place motion; skipping separate release motion stage")
        if rule.primitive == "insert_vertical":
            relaxed_target_collision = targeted._set_scene_obstacle_planner_box_scale(
                demo,
                place_choice["target_name"],
                float(args.place_insert_target_collision_scale),
            )

    ok, _ = targeted.base.execute_pose_path_stage(
        demo,
        bridge_mod,
        real_exec,
        place_choice["label"],
        place_choice["pose"],
        q_pre_place_path,
        args.real_gripper_close,
        args,
        use_attach=True,
    )
    if not ok:
        if relaxed_target_collision:
            targeted._set_scene_obstacle_planner_box_scale(demo, place_choice["target_name"], 1.0)
        return False

    direct_place_mode = bool(selected_joint_chain.get("direct_place_mode", False)) if selected_joint_chain is not None else False
    if not direct_place_mode:
        ok, _ = targeted.base.execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            "release" if place_choice["variant_label"] is None else f"release_{place_choice['variant_label']}",
            place_choice["place_pose"],
            q_place_path,
            args.real_gripper_close,
            args,
            use_attach=True,
        )
        if not ok:
            if relaxed_target_collision:
                targeted._set_scene_obstacle_planner_box_scale(demo, place_choice["target_name"], 1.0)
            return False

    print("\n[open gripper at place]")
    if not targeted.base.confirm_simple_action("open the real gripper at the targeted place", args, bridge_mod=bridge_mod, env=demo.env, repeats=6):
        print("[abort] user cancelled before opening the real gripper at the targeted place")
        if relaxed_target_collision:
            targeted._set_scene_obstacle_planner_box_scale(demo, place_choice["target_name"], 1.0)
        return False
    if real_exec is not None:
        real_exec.set_gripper(args.real_gripper_open)
        targeted.base.sync_demo_gripper_state(demo, closed=False, steps=4)
    else:
        print("[dry-run] skipped real gripper open at the targeted place")
        targeted.base.sync_demo_gripper_state(demo, closed=False, steps=4)
    demo._attached_box_visual_visible = False
    demo._attached_object_visual_active = False
    targeted.base.update_attached_box_visual(demo, visible=False)
    targeted._mark_place_rule_success(rule, place_state_cache, place_choice["slot_name"])

    if direct_place_mode:
        q_retreat_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in reversed(q_pre_place_path[:-1])]
    else:
        q_retreat_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in reversed(q_place_path[:-1])]
    if q_retreat_path:
        ok, _ = targeted.base.execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            "post_place_retreat",
            place_choice["pose"],
            q_retreat_path,
            args.real_gripper_open,
            args,
            use_attach=False,
        )
        if not ok:
            if relaxed_target_collision:
                targeted._set_scene_obstacle_planner_box_scale(demo, place_choice["target_name"], 1.0)
            print("[warn] post-place retreat execution failed after the object was already released; keeping the placement result and ending this cycle without retreat")
            print("[done] completed one targeted place motion; post-place retreat was skipped")
            return True

    if relaxed_target_collision:
        targeted._set_scene_obstacle_planner_box_scale(demo, place_choice["target_name"], 1.0)
    print("[done] completed one targeted place motion and retreated after release")
    return True


def main():
    targeted.parse_args = parse_args
    targeted.run_targeted_place_episode = run_targeted_place_episode_curobo_direct
    targeted.main()


if __name__ == "__main__":
    main()
