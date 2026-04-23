#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import rm75_jiaobang_pick_place_targeted as targeted
import rm75_jiaobang_pick_place_targeted_curobo as curobo_wrapper


def build_arg_parser():
    parser = curobo_wrapper.build_arg_parser()
    parser.description = (
        "FoundationPose -> direct cuRobo grasp -> close gripper -> cuRobo direct place "
        "without pre-place fallback."
    )
    parser.set_defaults(
        targeted_place_staging=False,
        curobo_plan_label_prefixes=[],
        curobo_ee_link="gripper_tcp",
        curobo_rm75_robot_cfg=Path(__file__).resolve().parent / "curobo_rm75_config" / "rm75.yml",
        carry_sim_arm_across_cycles=False,
        insert_vertical_axial_spin_deg=[0.0, -15.0, 15.0, -35.0, 35.0, -55.0, 55.0, -75.0],
        targeted_place_allow_insert_axis_flip=False,
        tabletop_place_yaw_variant_deg=[0.0, -15.0, 15.0, -35.0, 35.0, -55.0, 55.0, -75.0],
        tabletop_place_tilt_toward_robot_deg=[0.0],
        tabletop_place_axial_spin_deg=[0.0],
        targeted_place_expand_orientation_invariant=False,
        targeted_place_hover_extra_height_m=[0.0, 0.01],
        topdown_grasp_yaw_variant_deg=[0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0, 180.0],
        topdown_tilt_toward_robot_deg=[12.0, 20.0, 30.0, 45.0],
        topdown_tilt_toward_robot_shift_m=[0.0, 0.02],
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
        default=[0.04, 0.05],
        help="For insert_vertical rules, also try these short pre-place approach distances along the target long axis.",
    )
    parser.add_argument(
        "--direct-pre-place-z-offsets",
        type=float,
        nargs="*",
        default=[0.0, 0.01],
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
        "--direct-grasp-z-lifts-m",
        type=float,
        nargs="*",
        default=[0.0, 0.01, 0.02],
        help="Also try lifting grasp TCP in world-z by these offsets to improve IK reachability screening.",
    )
    parser.add_argument(
        "--direct-grasp-diagnose-topk",
        type=int,
        default=3,
        help="When direct_grasp IK prefilter keeps zero candidates, run detailed diagnose_pose_goal for top-K closest IK candidates.",
    )
    parser.add_argument(
        "--direct-grasp-axis-shift-penalty-per-cm",
        type=float,
        default=0.10,
        help="Additional score penalty per 1cm absolute grasp-axis shift. Larger values prefer center-aligned grasps.",
    )
    parser.add_argument(
        "--direct-grasp-z-lift-penalty-per-cm",
        type=float,
        default=0.08,
        help="Additional score penalty per 1cm grasp TCP lift. Larger values prefer lower-lift grasps.",
    )
    parser.add_argument(
        "--direct-grasp-max-axis-shift-ratio",
        type=float,
        default=0.35,
        help="Max |axis_shift| as a fraction of the object's half-length along its longest axis. "
        "E.g. 0.35 on a 60mm object = max 10.5mm shift. Prevents grasp candidates that would "
        "overshoot the object edge.",
    )
    parser.add_argument(
        "--direct-min-place-tcp-z",
        type=float,
        default=0.005,
        help="Reject any direct pre-place/place candidate whose TCP world z falls below this threshold.",
    )
    parser.add_argument(
        "--direct-place-max-pad-tilt",
        type=float,
        default=0.25,
        help="Max |tcp_y_world_z| for place candidates — how much the pad opening axis can deviate "
        "from horizontal. 0.0 = pads must be perfectly level; 0.25 ≈ max ~14° tilt "
        "(~12mm height difference for 50mm pad span). Rejects poses where one pad is much "
        "lower than the other, causing unstable release.",
    )
    parser.add_argument(
        "--direct-place-max-abs-yaw-deg",
        type=float,
        default=0.0,
        help="Optional absolute-yaw filter for direct place candidates. "
        "Set <= 0 to disable and keep the full tabletop yaw variant set.",
    )
    parser.add_argument(
        "--direct-pre-place-verticality-band",
        type=float,
        default=0.08,
        help="Among successful insert_vertical pre-place candidates, keep only those within this tcp_verticality band from the best vertical candidate before selecting the lowest-cost cuRobo path.",
    )
    parser.add_argument(
        "--place-orientation-preference-weight",
        type=float,
        default=0.25,
        help="Additional penalty weight used to prefer place candidates whose object facing direction is oriented toward the robot. Set <=0 to disable this preference.",
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
    parser.add_argument(
        "--curobo-table-collision",
        dest="curobo_table_collision",
        action="store_true",
        default=True,
        help="Inject a virtual table cuboid at z=0 into cuRobo world for ground-plane collision avoidance.",
    )
    parser.add_argument(
        "--no-curobo-table-collision",
        dest="curobo_table_collision",
        action="store_false",
        help="Disable the virtual table cuboid in cuRobo world.",
    )
    parser.add_argument(
        "--curobo-table-z-offset",
        type=float,
        default=-0.01,
        help="Z offset of the virtual table cuboid top surface relative to z=0. "
        "Negative values lower the table below z=0, giving clearance for gripper approach. "
        "Default -0.01 means the table top is at z=-1cm.",
    )
    parser.add_argument(
        "--curobo-table-thickness",
        type=float,
        default=0.02,
        help="Thickness of the virtual table cuboid.",
    )
    parser.add_argument(
        "--curobo-table-size-x",
        type=float,
        default=1.2,
        help="X extent of the virtual table cuboid.",
    )
    parser.add_argument(
        "--curobo-table-size-y",
        type=float,
        default=1.2,
        help="Y extent of the virtual table cuboid.",
    )
    parser.add_argument(
        "--curobo-attach-object",
        dest="curobo_attach_object",
        action="store_true",
        default=True,
        help="After grasping, attach the target object to the robot in cuRobo so its collision spheres participate in transport planning.",
    )
    parser.add_argument(
        "--no-curobo-attach-object",
        dest="curobo_attach_object",
        action="store_false",
        help="Disable cuRobo attached object collision during transport.",
    )
    parser.add_argument(
        "--curobo-attach-world-z-offset-m",
        type=float,
        default=0.003,
        help="When attaching the grasped object into cuRobo, first apply this small world-frame +Z offset. "
        "Useful for objects that are still lightly touching the table at the instant of attach.",
    )
    parser.add_argument(
        "--curobo-ik-prefilter-position-threshold",
        type=float,
        default=0.01,
        help="Soft position threshold used by batch IK prefilter before MotionGen. "
        "Candidates that miss strict IK but stay within this bound are still sent to trajectory planning.",
    )
    parser.add_argument(
        "--curobo-ik-prefilter-rotation-threshold",
        type=float,
        default=0.25,
        help="Soft rotation threshold (rad) used by batch IK prefilter before MotionGen. "
        "Raise this when direct-place candidates are close in position but fail due to orientation error.",
    )
    parser.add_argument(
        "--transport-attached-box-scale-xy",
        type=float,
        default=None,
        help="Optional XY-only scale for the cuRobo attached-object collision box. "
        "If omitted, reuse --transport-attached-box-scale.",
    )
    parser.add_argument(
        "--transport-attached-box-scale-z",
        type=float,
        default=1.0,
        help="Optional Z-only scale for the cuRobo attached-object collision box during transport. "
        "Useful when the grasped object is scraping tabletop clutter even though the XY footprint is covered.",
    )
    parser.add_argument(
        "--curobo-demo-path-validation",
        dest="curobo_demo_path_validation",
        action="store_true",
        default=False,
        help="Run the extra demo-planner dense collision validation after a cuRobo plan succeeds. "
        "Disabled by default so candidate acceptance follows cuRobo directly.",
    )
    parser.add_argument(
        "--no-curobo-demo-path-validation",
        dest="curobo_demo_path_validation",
        action="store_false",
        help="Disable the extra demo-planner dense collision validation after cuRobo success.",
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


def _stabilize_post_grasp_attached_state(demo, args, grasp_choice) -> None:
    q_path = list(grasp_choice.get("q_path") or [])
    grasp_terminal_q = None
    if q_path:
        grasp_terminal_q = np.asarray(q_path[-1], dtype=np.float32).reshape(-1)[:7]
        targeted.base.sync_demo_arm_qpos(demo, grasp_terminal_q)
    T_tcp_obj = grasp_choice.get("T_tcp_obj")
    if T_tcp_obj is None:
        demo._transport_attached_T_tcp_obj = None
        print("[direct_pre_place] grasp choice has no saved T_tcp_obj; attached-state stabilization skipped")
        return
    demo._transport_attached_T_tcp_obj = np.asarray(T_tcp_obj, dtype=np.float32).reshape(4, 4)
    targeted._register_transport_attached_box(
        demo,
        args,
        show_visual=False,
        activate_payload_visual=False,
        T_tcp_obj_override=demo._transport_attached_T_tcp_obj,
    )
    if targeted.base.force_active_object_to_attached_pose(demo):
        print("[direct_pre_place] stabilized attached sim state from grasp-time TCP<->object transform")
    else:
        print("[direct_pre_place] failed to force active object to attached pose; continuing with current sim object state")


def _apply_deferred_two_step_final_approach(planner, demo, args, grasp_choice, pregrasp_lookup) -> dict:
    if not pregrasp_lookup:
        return grasp_choice
    pregrasp_success = pregrasp_lookup.get(str(grasp_choice.get("label", "")))
    if pregrasp_success is None:
        return grasp_choice
    pregrasp_q = np.asarray(pregrasp_success.get("deferred_pregrasp_q"), dtype=np.float32).reshape(-1)[:7]
    pregrasp_pose = pregrasp_success.get("deferred_pregrasp_pose")
    grasp_pose = pregrasp_success.get("deferred_grasp_pose", grasp_choice.get("pose"))
    if pregrasp_pose is None or grasp_pose is None:
        return grasp_choice
    final_approach_label = f"{grasp_choice.get('label', '?')}_final_approach"
    _refresh_curobo_world(
        planner,
        demo,
        args,
        label=final_approach_label,
        include_active_object=False,
        include_table=False,
    )
    final_approach_path = _plan_with_official_approach_metric(
        planner,
        demo,
        args,
        pregrasp_q,
        pregrasp_pose,
        grasp_pose,
        label=final_approach_label,
    )
    _refresh_curobo_world(
        planner,
        demo,
        args,
        label=final_approach_label,
        include_active_object=True,
        include_table=False,
    )
    if not final_approach_path:
        print(
            f"[two_step_grasp] deferred official final approach failed for "
            f"{grasp_choice.get('label', '?')}; candidate rejected"
        )
        return grasp_choice
    upgraded = dict(grasp_choice)
    upgraded["pose"] = grasp_pose
    upgraded["q_path"] = list(pregrasp_success["q_path"]) + list(final_approach_path[1:])
    upgraded["two_step_grasp"] = True
    upgraded["pregrasp_waypoints"] = len(pregrasp_success["q_path"])
    upgraded["approach_waypoints"] = len(final_approach_path)
    upgraded["approach_distance_m"] = float(pregrasp_success.get("approach_distance_m", 0.0))
    print(
        f"[two_step_grasp] using deferred official final approach for {grasp_choice.get('label', '?')}: "
        f"pregrasp={len(pregrasp_success['q_path'])} + approach={len(final_approach_path)} waypoints, "
        f"distance={upgraded['approach_distance_m']*1000:.1f}mm"
    )
    return upgraded


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


def _build_virtual_table_cuboid(args) -> dict:
    table_x = float(getattr(args, "curobo_table_center_x", 0.0))
    table_y = float(getattr(args, "curobo_table_center_y", 0.0))
    table_thickness = float(max(getattr(args, "curobo_table_thickness", 0.02), 1e-3))
    table_size_x = float(max(getattr(args, "curobo_table_size_x", 1.2), 0.1))
    table_size_y = float(max(getattr(args, "curobo_table_size_y", 1.2), 0.1))
    z_offset = float(getattr(args, "curobo_table_z_offset", -0.01))
    table_z = z_offset - 0.5 * table_thickness

    if bool(getattr(args, "curobo_debug", False)):
        print(
            f"[curobo] virtual table: center=({table_x:.3f}, {table_y:.3f}, {table_z:.3f}), "
            f"size=({table_size_x:.3f}, {table_size_y:.3f}, {table_thickness:.3f}), "
            f"top_surface_z={z_offset:.3f}"
        )

    return {
        "name": "virtual_table_plane",
        "dims": [table_size_x, table_size_y, table_thickness],
        "pose": [table_x, table_y, table_z, 1.0, 0.0, 0.0, 0.0],
    }


def _round_signature_values(values):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return tuple(float(np.round(v, 6)) for v in arr.tolist())


def _world_obstacle_signature(cuboids, meshes):
    cuboid_sig = []
    for item in list(cuboids or []):
        cuboid_sig.append(
            (
                str(item.get("name", "")),
                _round_signature_values(item.get("dims", [])),
                _round_signature_values(item.get("pose", item.get("xyz_quat", []))),
            )
        )
    mesh_sig = []
    for item in list(meshes or []):
        mesh_sig.append(
            (
                str(item.get("name", "")),
                str(item.get("file_path", item.get("asset_file", item.get("mesh_file", "")))),
                _round_signature_values(item.get("scale", item.get("asset_scale", [1.0, 1.0, 1.0]))),
                _round_signature_values(item.get("pose", item.get("xyz_quat", []))),
            )
        )
    cuboid_sig.sort()
    mesh_sig.sort()
    return (tuple(cuboid_sig), tuple(mesh_sig))


def _visualize_attached_spheres(planner, demo, args):
    """Print and optionally visualize the attached object collision spheres."""
    current_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    spheres = planner.get_attached_spheres_world(current_q)
    if not spheres:
        print("[curobo] no attached spheres to visualize")
        return
    base_p = targeted.base.flatten_np(demo.robot.pose.p)[:3].astype(np.float32)
    print(f"[curobo] attached object collision spheres ({len(spheres)} total):")
    for i, s in enumerate(spheres):
        world_center = np.asarray(s["center"], dtype=np.float32) + base_p
        print(
            f"  sphere[{i}]: world_center=[{world_center[0]:.4f}, {world_center[1]:.4f}, {world_center[2]:.4f}], "
            f"radius={s['radius']*1000:.1f}mm"
        )
    if not bool(getattr(args, "curobo_show_attached_spheres", True)):
        return
    try:
        from sapien.core import Pose
        env = demo.env
        sphere_actors = list(getattr(env.unwrapped, "_attached_sphere_actors", []) or [])
        for actor in sphere_actors:
            try:
                actor.remove_from_scene()
            except Exception:
                pass
        sphere_actors = []
        for i, s in enumerate(spheres):
            world_center = np.asarray(s["center"], dtype=np.float32) + base_p
            radius = float(s["radius"])
            builder = env.unwrapped.scene.create_actor_builder()
            builder.add_sphere_visual(radius=radius, material=None)
            actor = builder.build_static(name=f"attached_sphere_{i}")
            actor.set_pose(Pose(p=world_center.tolist()))
            try:
                for body in actor.get_visual_bodies():
                    for shape in body.get_render_shapes():
                        mat = shape.material
                        mat.set_base_color([1.0, 0.2, 0.2, 0.4])
                        shape.set_material(mat)
            except Exception:
                pass
            sphere_actors.append(actor)
        env.unwrapped._attached_sphere_actors = sphere_actors
        print(f"[curobo] visualized {len(sphere_actors)} attached collision spheres (red, semi-transparent)")
    except Exception as exc:
        print(f"[curobo] sphere visualization failed (non-critical): {exc}")


def _clear_visualized_attached_spheres(demo) -> None:
    env = getattr(demo, "env", None)
    if env is None:
        return
    sphere_actors = list(getattr(env.unwrapped, "_attached_sphere_actors", []) or [])
    for actor in sphere_actors:
        try:
            actor.remove_from_scene()
        except Exception:
            pass
    env.unwrapped._attached_sphere_actors = []


def _refresh_curobo_world(
    planner, demo, args, *, label: str, include_active_object: bool = False, include_table: bool = False,
    exclude_object_names: set[str] | None = None,
) -> None:
    if planner.collision_enabled:
        cuboids, meshes = curobo_wrapper._scene_obstacles_to_curobo_world(
            demo, args, exclude_object_names=exclude_object_names,
        )
        if include_active_object:
            active_cuboids, active_meshes = _build_active_object_curobo_world(demo, args)
            cuboids.extend(active_cuboids)
            meshes.extend(active_meshes)
        if include_table and bool(getattr(args, "curobo_table_collision", True)):
            cuboids.append(_build_virtual_table_cuboid(args))
        world_signature = _world_obstacle_signature(cuboids, meshes)
        world_changed = world_signature != getattr(planner, "_persistent_world_signature", None)
        if world_changed:
            targeted.base.sync_curobo_collision_world_visuals(
                demo.env,
                args,
                cuboids=cuboids,
                meshes=meshes,
                label=label,
            )
        if planner.attached_object_active:
            _visualize_attached_spheres(planner, demo, args)
        else:
            _clear_visualized_attached_spheres(demo)
        if world_changed:
            cuboids_in_base, meshes_in_base = curobo_wrapper._transform_curobo_world_to_robot_base(
                cuboids,
                meshes,
                demo,
            )
            planner.set_world_from_obstacles(cuboids=cuboids_in_base, meshes=meshes_in_base)
            planner._persistent_world_signature = world_signature
        if exclude_object_names:
            print(
                f"[curobo] {label} world: excluded scene obstacle(s): {sorted(exclude_object_names)}"
            )
        if bool(getattr(args, "curobo_debug", False)):
            if world_changed:
                print(
                    f"[curobo] updated world with {len(cuboids_in_base)} cuboid and "
                    f"{len(meshes_in_base)} mesh obstacles for {label}"
                )
            else:
                print(f"[curobo] reused persistent world for {label}")
            cuboid_names = [str(item.get("name", "")) for item in list(cuboids or [])]
            mesh_names = [str(item.get("name", "")) for item in list(meshes or [])]
            print(
                f"[curobo] {label} world members: "
                f"cuboids={cuboid_names}, meshes={mesh_names}"
            )
    elif bool(getattr(args, "curobo_debug", False)):
        print(f"[curobo] planner is in embedded free-space mode for {label}")
    else:
        _clear_visualized_attached_spheres(demo)


def _direct_grasp_target_contact_only_disabled_links(planner) -> list[str]:
    configured_links = set(getattr(planner, "configured_collision_links", []) or [])
    if not configured_links:
        return []
    candidate_disable_links = {
        "gripper_base_link",
        "gripper_Left_1_Link",
        "gripper_Left_Support_Link",
        "gripper_Left_2_Link",
        "gripper_Right_1_Link",
        "gripper_Right_Support_Link",
        "gripper_Right_2_Link",
        "left_pad",
        "right_pad",
    }
    return sorted(candidate_disable_links & configured_links)


def _direct_place_contact_tolerant_disabled_links(planner) -> list[str]:
    """
    放置时禁用的链路：允许夹爪的support link接触桌面，
    但保留pad的碰撞检测以确保不会夹到目标物体。
    """
    configured_links = set(getattr(planner, "configured_collision_links", []) or [])
    if not configured_links:
        return []
    candidate_disable_links = {
        "gripper_Left_Support_Link",
        "gripper_Right_Support_Link",
    }
    return sorted(candidate_disable_links & configured_links)


def _normalize_disabled_world_collision_links(planner, disabled_world_collision_links) -> list[str]:
    if not bool(getattr(planner, "collision_enabled", False)):
        return []
    configured_links = set(getattr(planner, "configured_collision_links", []) or [])
    normalized = []
    seen = set()
    for link_name in list(disabled_world_collision_links or []):
        name = str(link_name)
        if not name or name in seen:
            continue
        if configured_links and name not in configured_links:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


def _set_world_collision_for_links(planner, link_names, *, enabled: bool, label: str) -> list[str]:
    normalized = _normalize_disabled_world_collision_links(planner, link_names)
    if normalized:
        planner.set_world_collision_for_links(normalized, enabled=enabled)
        action = "re-enabled" if enabled else "temporarily disabled"
        print(f"[curobo] {label} {action} world collision for links: {normalized}")
    return normalized


def _path_metrics_and_score(q_start, q_path):
    metrics = targeted.base._joint_path_quality_metrics(q_start, q_path)
    score = targeted.base._joint_path_quality_score(metrics)
    return metrics, float(score)


def _candidate_selection_penalty(candidate, args) -> float:
    axis_shift_m = float(candidate.get("grasp_axis_shift_m", 0.0))
    z_lift_m = float(candidate.get("grasp_z_lift_m", 0.0))
    axis_penalty_per_cm = float(max(getattr(args, "direct_grasp_axis_shift_penalty_per_cm", 0.10), 0.0))
    z_lift_penalty_per_cm = float(max(getattr(args, "direct_grasp_z_lift_penalty_per_cm", 0.08), 0.0))
    axis_cm = abs(axis_shift_m) * 100.0
    z_lift_cm = max(z_lift_m, 0.0) * 100.0
    return float(axis_penalty_per_cm * axis_cm + z_lift_penalty_per_cm * z_lift_cm)


def _candidate_target_up_axis_world(candidate, demo) -> np.ndarray | None:
    place_plan = candidate.get("place_plan")
    target_name = None if place_plan is None else getattr(place_plan, "target_name", None)
    if target_name:
        scene_entry = targeted._find_scene_object_entry(demo, target_name)
        if scene_entry is not None and scene_entry.get("T_world_obj") is not None:
            T_world_target = np.asarray(scene_entry["T_world_obj"], dtype=np.float32).reshape(4, 4)
            return _normalize(T_world_target[:3, 1])
    return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)


def _candidate_place_orientation_penalty(candidate, demo, args) -> float:
    weight = float(max(getattr(args, "place_orientation_preference_weight", 0.25), 0.0))
    if weight <= 0.0:
        return 0.0
    place_plan = candidate.get("place_plan")
    if place_plan is None:
        return 0.0
    T_world_obj_desired = getattr(place_plan, "T_world_obj_desired", None)
    if T_world_obj_desired is None:
        return 0.0
    rule = getattr(place_plan, "rule", None)
    T_world_obj_desired = np.asarray(T_world_obj_desired, dtype=np.float32).reshape(4, 4)
    robot_pos = targeted.base.flatten_np(demo.robot.pose.p)[:3].astype(np.float32)
    obj_pos = T_world_obj_desired[:3, 3].astype(np.float32)
    up_axis = _candidate_target_up_axis_world(candidate, demo)
    if up_axis is None:
        up_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    toward_robot = robot_pos - obj_pos
    toward_robot = toward_robot - float(np.dot(toward_robot, up_axis)) * up_axis
    toward_robot = _normalize(toward_robot)
    if toward_robot is None:
        return 0.0

    R_world_obj = T_world_obj_desired[:3, :3].astype(np.float32)
    face_axis_local = None if rule is None else getattr(rule, "face_robot_axis_local", None)
    if face_axis_local is not None:
        face_axis_world = (R_world_obj @ np.asarray(face_axis_local, dtype=np.float32).reshape(3)).astype(np.float32)
        face_axis_world = face_axis_world - float(np.dot(face_axis_world, up_axis)) * up_axis
        face_axis_world = _normalize(face_axis_world)
        if face_axis_world is None:
            return 0.0
        alignment = float(np.clip(np.dot(face_axis_world, toward_robot), -1.0, 1.0))
    else:
        alignment = -1.0
        for axis_local in (
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        ):
            axis_world = (R_world_obj @ axis_local).astype(np.float32)
            axis_world = axis_world - float(np.dot(axis_world, up_axis)) * up_axis
            axis_world = _normalize(axis_world)
            if axis_world is None:
                continue
            alignment = max(alignment, float(np.clip(np.dot(axis_world, toward_robot), -1.0, 1.0)))
        if alignment < -0.5:
            return 0.0
    return float((1.0 - alignment) * 0.5 * weight)


def _candidate_sort_key(item):
    metrics = item["metrics"]
    return (
        float(item["score"]),
        float(metrics["total_motion"]),
        float(metrics["joint7_total_motion"]),
        int(metrics["waypoint_count"]),
    )


def _variant_abs_yaw_deg_from_labels(variant_label: str | None, label: str | None = None) -> float:
    """Parse tabletop yaw magnitude from variant/label, e.g. yaw_-60deg -> 60. No match -> 0."""
    text = " ".join(str(x) for x in (variant_label, label) if x is not None)
    m = re.search(r"yaw_([-+]?\d+(?:\.\d+)?)deg", text)
    if not m:
        return 0.0
    return abs(float(m.group(1)))


def _pre_place_screen_sort_key(item):
    label = "" if item.get("label") is None else str(item.get("label"))
    variant = "" if item.get("variant_label") is None else str(item.get("variant_label"))
    pose_z = float(_get_pose_position(item["pose"])[2]) if "pose" in item else 0.0
    place_z = float(_get_pose_position(item["place_pose"])[2]) if "place_pose" in item else 0.0
    yaw_abs = _variant_abs_yaw_deg_from_labels(variant, label)
    return (
        yaw_abs,
        -float(item.get("tcp_verticality", 0.0)),
        -pose_z,
        -place_z,
        variant,
        label,
    )


def _print_ik_error_summary(label: str, ik_records) -> None:
    records = list(ik_records or [])
    if not records:
        return
    pos = np.asarray(
        [float(x["position_error"]) for x in records if np.isfinite(float(x.get("position_error", np.nan)))],
        dtype=np.float32,
    )
    rot = np.asarray(
        [float(x["rotation_error"]) for x in records if np.isfinite(float(x.get("rotation_error", np.nan)))],
        dtype=np.float32,
    )
    if pos.size == 0 and rot.size == 0:
        return
    pos_txt = "n/a"
    rot_txt = "n/a"
    if pos.size > 0:
        pos_txt = f"min={float(np.min(pos)):.5f}, median={float(np.median(pos)):.5f}"
    if rot.size > 0:
        rot_txt = f"min={float(np.min(rot)):.5f}, median={float(np.median(rot)):.5f}"
    print(f"[curobo][diag] {label} IK error summary: pos({pos_txt}) rot({rot_txt})")


def _diagnose_failed_ik_candidates(planner, demo, args, label: str, start_q, ranked_candidates, *, topk: int = 3):
    use_topk = int(max(topk, 0))
    if use_topk <= 0:
        return
    for idx, item in enumerate(list(ranked_candidates or [])[:use_topk], start=1):
        candidate = item["candidate"]
        candidate_label = str(candidate["label"])
        planner_pose = item["planner_pose"]
        diag = planner.diagnose_pose_goal(
            start_q,
            planner_pose,
            num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
            num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
        )
        print(
            f"[curobo][diag] {label} top{idx}={candidate_label}: "
            f"standalone_ok={diag['standalone_ik_success']}, "
            f"pos_err={diag['standalone_ik_position_error']:.6f}, "
            f"rot_err={diag['standalone_ik_rotation_error']:.6f}, "
            f"goal_delta={diag['goal_translation_delta_norm']:.6f}, "
            f"motiongen_internal_ik={diag['motiongen_internal_ik_successes']}"
        )


def _print_start_state_self_collision_diagnostics(planner, start_q, *, label: str) -> None:
    start_diag = planner.diagnose_start_state_self_collision(start_q, top_k=10)
    if start_diag.get("error"):
        print(f"[curobo] {label} self-collision diagnosis failed: {start_diag['error']}")
        return
    for item in list(start_diag.get("link_pairs", []) or []):
        print(
            f"[curobo] {label} self-collision link_pair="
            f"{item['link_a']} <-> {item['link_b']} "
            f"overlap={float(item['overlap']) * 1000.0:.2f}mm"
        )
    for item in list(start_diag.get("pairs", []) or []):
        print(
            f"[curobo] {label} self-collision sphere_pair="
            f"{item['sphere_i']}({item['link_i']}) <-> {item['sphere_j']}({item['link_j']}) "
            f"overlap={float(item['overlap']) * 1000.0:.2f}mm "
            f"center_dist={float(item['center_distance']) * 1000.0:.2f}mm "
            f"threshold={float(item['threshold']) * 1000.0:.2f}mm"
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


def _quat_wxyz_to_rotmat(q_wxyz) -> np.ndarray:
    w, x, y, z = float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _tcp_pad_tilt_z(pose) -> float:
    """Return |tcp_y_axis_world[2]| — how tilted the pad opening direction is from horizontal.

    Pads open along TCP y-axis (URDF: gripper_tcp has 90-deg yaw from gripper_base_link,
    mapping the pad opening axis to TCP y).

    0.0 = pads perfectly level (both at the same height) — safe to release.
    1.0 = one pad directly above the other — object will slide off on release.
    """
    q_wxyz = _normalize_quat_wxyz(targeted.base.flatten_np(pose.q)[:4])
    R = _quat_wxyz_to_rotmat(q_wxyz)
    tcp_y_in_world = R[:, 1]
    return float(abs(tcp_y_in_world[2]))


def _get_pose_position(pose) -> np.ndarray:
    return targeted.base.flatten_np(pose.p)[:3].astype(np.float32)


def _copy_pose_obj(pose):
    return targeted.Pose.create_from_pq(
        p=targeted.base.flatten_np(pose.p)[:3].astype(np.float32),
        q=_normalize_quat_wxyz(targeted.base.flatten_np(pose.q)[:4]),
    )


def _get_demo_tcp_pose_for_joint_q(demo, q_arm):
    q_saved = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    try:
        targeted.base.sync_demo_arm_qpos(demo, q_arm)
        return _copy_pose_obj(demo.tcp.pose)
    finally:
        targeted.base.sync_demo_arm_qpos(demo, q_saved)


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


def _build_official_approach_metric(planner, args, pose_start, pose_goal, *, label: str):
    free_axis, delta_goal, locked_delta, rot_err_deg = _infer_goal_frame_free_linear_axis(pose_start, pose_goal)
    if rot_err_deg > 5.0 or float(np.max(np.abs(locked_delta))) > 0.01:
        print(
            f"[curobo] {label} cannot use official approach metric cleanly "
            f"(goal-frame rot_err={rot_err_deg:.2f} deg, locked_delta={np.round(locked_delta, 6)})"
        )
        return None
    offset = float(abs(delta_goal[free_axis]))
    if offset <= 1e-6:
        return None
    metric = planner.mods["PoseCostMetric"].create_grasp_approach_metric(
        offset_position=offset,
        linear_axis=free_axis,
        tstep_fraction=float(getattr(args, "curobo_approach_metric_tstep_fraction", 0.8)),
        tensor_args=planner.tensor_args,
    )
    metric.project_to_goal_frame = True
    return metric, free_axis, delta_goal


def _plan_with_official_approach_metric(
    planner,
    demo,
    args,
    start_q,
    pose_start,
    pose_goal,
    *,
    label: str,
):
    metric_info = _build_official_approach_metric(planner, args, pose_start, pose_goal, label=label)
    if metric_info is None:
        return None
    metric, free_axis, delta_goal = metric_info
    planner_pose = _convert_demo_tcp_pose_to_curobo_ee_pose(
        demo,
        pose_goal,
        ee_link_name=str(getattr(planner.config, "ee_link", "gripper_tcp")),
    )
    print(
        f"[curobo] {label} trying official MotionGen approach metric "
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
        print(f"[curobo] {label} official MotionGen approach metric failed with status={result.status}")
        return None
    q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in result.joint_path]
    final_diag = _measure_realized_tcp_error(demo, q_path[-1], pose_goal)
    print(
        f"[curobo] {label} official MotionGen realized tcp: "
        f"pos_err={final_diag['pos_err']:.4f} m, rot_err={final_diag['rot_err_deg']:.2f} deg"
    )
    if not _terminal_error_within_limits(final_diag, args):
        print(f"[curobo] {label} official MotionGen misses the target pose too much")
        return None
    return q_path


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
    return _plan_with_official_approach_metric(
        planner,
        demo,
        args,
        start_q,
        pose_start,
        pose_goal,
        label=label,
    )


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


def _validate_candidate_joint_path_with_demo_planner(demo, start_q, q_path, *, use_attach: bool, label: str) -> bool:
    if not bool(getattr(demo.args, "curobo_demo_path_validation", False)):
        return True
    dense_validate_delta = 0.01 if use_attach else 0.03
    ok = targeted.base.validate_joint_path_segments(
        demo,
        start_q,
        q_path,
        use_attach=use_attach,
        label=f"{label}_demo_validate",
        max_delta=dense_validate_delta,
    )
    if not ok:
        validation_mode = "attached-box" if use_attach else "scene"
        print(f"[curobo] {label} rejected by demo planner {validation_mode} collision validation")
    return bool(ok)


def _plan_and_execute_return_to_cycle_start(
    demo,
    bridge_mod,
    real_exec,
    args,
    start_q,
    *,
    use_attach: bool = False,
    gripper_pos: float | None = None,
) -> bool:
    label = "return_to_cycle_start"
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    q_current = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    target_pose = _get_demo_tcp_pose_for_joint_q(demo, start_q)

    planner = curobo_wrapper._get_or_create_curobo_planner(args)
    q_path = None
    if planner is not None:
        _refresh_curobo_world(planner, demo, args, label=label)
        goal_pose = _get_demo_tcp_pose_for_joint_q(demo, start_q)
        planner_pose = _convert_demo_tcp_pose_to_curobo_ee_pose(
            demo,
            goal_pose,
            ee_link_name=str(getattr(planner.config, "ee_link", "gripper_tcp")),
        )
        result = planner.plan_to_pose(
            q_current,
            planner_pose,
            enable_graph=bool(getattr(args, "curobo_enable_graph", False)),
            max_attempts=int(getattr(args, "curobo_max_attempts", 2)),
            timeout=float(getattr(args, "curobo_timeout", 5.0)),
            num_ik_seeds=int(getattr(args, "curobo_num_ik_seeds", 64)),
            num_trajopt_seeds=int(getattr(args, "curobo_num_trajopt_seeds", 1)),
            num_graph_seeds=int(getattr(args, "curobo_num_graph_seeds", 1)),
        )
        if result.success and result.joint_path is not None:
            q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in result.joint_path]
            # 检查终点joint configuration是否足够接近目标
            # 重要：检查每个关节的误差，防止关节1多转360度的情况
            q_final = q_path[-1]
            joint_errors = np.abs(q_final - start_q)
            max_joint_error_per_joint = float(getattr(args, "return_to_start_max_joint_error_rad", 0.1))  # 默认0.1弧度
            max_error = float(np.max(joint_errors))

            if max_error > max_joint_error_per_joint:
                # 找出哪个关节误差最大
                worst_joint_idx = int(np.argmax(joint_errors))
                print(
                    f"[curobo] {label} WARNING: joint[{worst_joint_idx}] error={joint_errors[worst_joint_idx]:.4f} rad "
                    f"exceeds max_allowed={max_joint_error_per_joint:.4f} rad, falling back to MPLib"
                )
                print(f"[curobo] {label} all joint errors (rad): {np.round(joint_errors, 4).tolist()}")
                q_path = None
            else:
                print(
                    f"[curobo] {label} planned successfully ({len(q_path)} waypoints), "
                    f"max_joint_error={max_error:.4f} rad"
                )
        else:
            print(f"[curobo] {label} cuRobo failed (status={result.status}), falling back to MPLib (no RRT)")

    if q_path is None:
        q_path = targeted.base.plan_joint_path(
            demo,
            start_q,
            use_attach=use_attach,
            label=label,
            start_q=q_current,
            allow_reverse_rrt_fallback=False,
        )
    if q_path is None:
        print(f"[FAIL] {label} planning failed")
        targeted.base.print_failure_diagnostics(
            demo,
            args,
            label,
            q_target=start_q,
            use_attach=use_attach,
        )
        return False
    ok, _ = targeted.base.execute_pose_path_stage(
        demo,
        bridge_mod,
        real_exec,
        label,
        target_pose,
        q_path,
        args.real_gripper_open if gripper_pos is None else float(gripper_pos),
        args,
        use_attach=use_attach,
    )
    if not ok:
        print(f"[FAIL] {label} execution failed")
        return False
    return True


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
    include_table: bool = False,
    exclude_object_names: set[str] | None = None,
    disabled_world_collision_links: list[str] | None = None,
):
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    _refresh_curobo_world(planner, demo, args, label=label, include_active_object=include_active_object, include_table=include_table, exclude_object_names=exclude_object_names)
    ee_link_name = str(getattr(planner.config, "ee_link", "gripper_tcp"))
    if bool(getattr(args, "curobo_debug", False)):
        ee_link = _get_robot_link_by_name(demo, ee_link_name)
        base_pose = targeted.base.flatten_np(demo.robot.pose.p)[:3]
        print(
            f"[curobo] runtime frames for {label}: "
            f"planner_ee_link={ee_link_name}, "
            f"demo_tcp_link={_get_link_name(demo.tcp)}, "
            f"robot_base_world_p={np.round(base_pose, 6)}"
        )
        if ee_link is not None:
            ee_world = targeted.base.flatten_np(ee_link.pose.p)[:3].astype(np.float32)
            tcp_world = targeted.base.flatten_np(demo.tcp.pose.p)[:3].astype(np.float32)
            print(
                f"[curobo] runtime link check for {label}: "
                f"ee_link_world_p={np.round(ee_world, 6)}, "
                f"tcp_world_p={np.round(tcp_world, 6)}, "
                f"ee_tcp_delta_norm={float(np.linalg.norm(ee_world - tcp_world)):.6f}"
            )
    successes = []
    disabled_world_collision_links = _set_world_collision_for_links(
        planner,
        disabled_world_collision_links,
        enabled=False,
        label=label,
    )
    try:
        for candidate in list(candidates or []):
            candidate_label = str(candidate["label"])
            pose = candidate["pose"]
            planner_pose = _convert_demo_tcp_pose_to_curobo_ee_pose(
                demo,
                pose,
                ee_link_name=ee_link_name,
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
                elif str(result.status) == "MotionGenStatus.INVALID_START_STATE_SELF_COLLISION":
                    _print_start_state_self_collision_diagnostics(
                        planner,
                        start_q,
                        label=f"{candidate_label}_start_state",
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
            if not _validate_candidate_joint_path_with_demo_planner(
                demo,
                start_q,
                q_path,
                use_attach=use_attach,
                label=candidate_label,
            ):
                continue
            metrics, score = _path_metrics_and_score(start_q, q_path)
            selection_penalty = _candidate_selection_penalty(candidate, args)
            place_pref_penalty = _candidate_place_orientation_penalty(candidate, demo, args)
            score = float(score) + float(selection_penalty) + float(place_pref_penalty)
            item = dict(candidate)
            item["result"] = result
            item["q_path"] = q_path
            item["metrics"] = metrics
            item["score"] = score
            item["selection_penalty"] = selection_penalty
            item["place_preference_penalty"] = place_pref_penalty
            item["terminal_align"] = terminal_align
            item["planner_pose"] = planner_pose
            print(
                f"[curobo] {candidate_label} success: score={score:.3f}, "
                f"selection_penalty={selection_penalty:.3f}, "
                f"place_pref_penalty={place_pref_penalty:.3f}, "
                f"total_motion={metrics['total_motion']:.3f} rad, "
                f"joint7_total={metrics['joint7_total_motion']:.3f} rad, "
                f"joint7_excursion={metrics['joint7_max_excursion']:.3f} rad, "
                f"waypoints={metrics['waypoint_count']}, "
                f"realized_pos_err={terminal_align['realized_after']['pos_err']:.4f} m, "
                f"realized_rot_err={terminal_align['realized_after']['rot_err_deg']:.2f} deg"
            )
            successes.append(item)
    finally:
        _set_world_collision_for_links(
            planner,
            disabled_world_collision_links,
            enabled=True,
            label=label,
        )

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


def _evaluate_two_step_grasp_candidates(
    planner,
    demo,
    args,
    start_q,
    candidates,
    *,
    label: str,
    max_winners: int | None = None,
    include_active_object: bool = False,
    disabled_world_collision_links: list[str] | None = None,
):
    """
    实现两步抓取：
    1. 第一步：规划到pregrasp pose（带碰撞检测）
    2. 第二步：从pregrasp沿夹爪轴向直线下降到grasp pose（简化碰撞检测）

    注意：pregrasp必须是通过demo.build_pregrasp_pose从grasp沿轴向回退得到的，
    这样第二步才能保证是直线下降而不是横向移动。
    """
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]

    # 准备pregrasp candidates
    pregrasp_candidates = []
    for cand in candidates:
        if "pregrasp_pose" not in cand:
            continue
        pregrasp_cand = dict(cand)
        pregrasp_cand["pose"] = cand["pregrasp_pose"]
        pregrasp_cand["original_grasp_pose"] = cand["pose"]
        pregrasp_candidates.append(pregrasp_cand)

    if not pregrasp_candidates:
        print(f"[{label}] no candidates have pregrasp_pose; explicit two-step grasp cannot proceed")
        return []

    print(f"[{label}] evaluating {len(pregrasp_candidates)} two-step grasp candidates")

    # 第一步：规划到pregrasp pose
    pregrasp_successes = _evaluate_curobo_pose_candidates_goalset(
        planner,
        demo,
        args,
        start_q,
        pregrasp_candidates,
        label=f"{label}_pregrasp",
        prefer_verticality=False,
        use_attach=False,
        max_winners=max_winners,
        include_active_object=include_active_object,
        disabled_world_collision_links=disabled_world_collision_links,
    )

    if not pregrasp_successes:
        print(f"[{label}] no pregrasp candidates succeeded")
        return []

    for pregrasp_success in pregrasp_successes:
        pregrasp_q = np.asarray(pregrasp_success["q_path"][-1], dtype=np.float32).reshape(-1)[:7]
        pregrasp_pose = pregrasp_success["pose"]
        grasp_pose = pregrasp_success["original_grasp_pose"]
        pregrasp_p = targeted.base.flatten_np(pregrasp_pose.p)[:3]
        grasp_p = targeted.base.flatten_np(grasp_pose.p)[:3]
        pregrasp_success["deferred_two_step_grasp"] = True
        pregrasp_success["deferred_pregrasp_q"] = pregrasp_q
        pregrasp_success["deferred_pregrasp_pose"] = pregrasp_pose
        pregrasp_success["deferred_grasp_pose"] = grasp_pose
        pregrasp_success["approach_distance_m"] = float(np.linalg.norm(grasp_p - pregrasp_p))

    print(
        f"[{label}] {len(pregrasp_successes)} pregrasp candidates succeeded; "
        "official final approach will be attempted only for the selected grasp chain"
    )
    return pregrasp_successes


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
    include_table: bool = False,
    exclude_object_names: set[str] | None = None,
    disabled_world_collision_links: list[str] | None = None,
):
    start_q = np.asarray(start_q, dtype=np.float32).reshape(-1)[:7]
    _refresh_curobo_world(planner, demo, args, label=label, include_active_object=include_active_object, include_table=include_table, exclude_object_names=exclude_object_names)
    remaining = list(candidates or [])
    if not remaining:
        return []

    ee_link_name = str(getattr(planner.config, "ee_link", "gripper_tcp"))
    if bool(getattr(args, "curobo_debug", False)):
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

    disabled_world_collision_links = _set_world_collision_for_links(
        planner,
        disabled_world_collision_links,
        enabled=False,
        label=f"{label}_ik+plan",
    )
    try:
        ik_screened = []
        ik_error_records = []
        ik_status_counts = {}
        ik_screen_total = len(remaining)
        ranked_failed_candidates = []
        for start_idx in range(0, len(remaining), chunk_size):
            chunk = remaining[start_idx : start_idx + chunk_size]
            planner_poses = [
                _convert_demo_tcp_pose_to_curobo_ee_pose(
                    demo,
                    item["pose"],
                    ee_link_name=ee_link_name,
                )
                for item in chunk
            ]
            start_qs = [start_q for _ in chunk]
            ik_results = planner.solve_batch_start_goal_ik(
                start_qs,
                planner_poses,
                num_seeds=int(getattr(args, "curobo_num_ik_seeds", 64) if num_ik_seeds is None else num_ik_seeds),
            )
            ik_details = planner.estimate_batch_start_goal_ik_errors(
                start_qs,
                planner_poses,
                num_seeds=int(getattr(args, "curobo_num_ik_seeds", 64) if num_ik_seeds is None else num_ik_seeds),
            )
            for candidate, planner_pose, ik_result, ik_detail in zip(chunk, planner_poses, ik_results, ik_details):
                status_key = str(ik_result.status)
                ik_status_counts[status_key] = int(ik_status_counts.get(status_key, 0)) + 1
                pos_err = float(ik_detail.get("position_error", np.nan))
                rot_err = float(ik_detail.get("rotation_error", np.nan))
                ik_error_records.append(
                    {
                        "label": str(candidate["label"]),
                        "position_error": pos_err,
                        "rotation_error": rot_err,
                        "success": bool(ik_result.success),
                    }
                )
                if ik_result.success:
                    ik_screened.append(candidate)
                    continue
                rank_key = (
                    pos_err if np.isfinite(pos_err) else np.inf,
                    rot_err if np.isfinite(rot_err) else np.inf,
                    str(candidate["label"]),
                )
                ranked_failed_candidates.append(
                    {
                        "candidate": candidate,
                        "planner_pose": planner_pose,
                        "rank_key": rank_key,
                    }
                )
        remaining = ik_screened
        status_summary = ", ".join(f"{k}={v}" for k, v in sorted(ik_status_counts.items()))
        print(
            f"[curobo][diag] {label} IK prefilter kept {len(remaining)}/{ik_screen_total} candidate(s); "
            f"statuses: {status_summary or 'none'}"
        )
        _print_ik_error_summary(label, ik_error_records)
        if not remaining:
            ranked_failed_candidates.sort(key=lambda x: x["rank_key"])
            _diagnose_failed_ik_candidates(
                planner,
                demo,
                args,
                label,
                start_q,
                ranked_failed_candidates,
                topk=int(getattr(args, "direct_grasp_diagnose_topk", 3)),
            )
            return []
        num_chunks = max(1, (len(remaining) + chunk_size - 1) // chunk_size)
        print(
            f"[curobo] {label} evaluating {len(remaining)} candidate(s) "
            f"in {num_chunks} batch chunk(s) of size <= {chunk_size}"
        )
        if use_max_winners == 1 and len(remaining) > 1:
            planner_poses = [
                _convert_demo_tcp_pose_to_curobo_ee_pose(
                    demo,
                    item["pose"],
                    ee_link_name=ee_link_name,
                )
                for item in remaining
            ]
            print(f"[curobo] {label} using official goalset fast-path for {len(remaining)} candidate(s)")
            goalset_result = planner.plan_goalset_to_poses(
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
            goal_idx = int((goalset_result.debug or {}).get("goalset_index", -1))
            if goalset_result.success and goalset_result.joint_path is not None and 0 <= goal_idx < len(remaining):
                candidate = remaining[goal_idx]
                candidate_label = str(candidate["label"])
                q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in goalset_result.joint_path]
                terminal_align = _validate_curobo_terminal_pose(
                    demo,
                    args,
                    q_path,
                    candidate["pose"],
                    label=candidate_label,
                )
                if terminal_align is not None:
                    q_path = terminal_align["q_path"]
                    if _validate_candidate_joint_path_with_demo_planner(
                        demo,
                        start_q,
                        q_path,
                        use_attach=use_attach,
                        label=candidate_label,
                    ):
                        metrics, score = _path_metrics_and_score(start_q, q_path)
                        selection_penalty = _candidate_selection_penalty(candidate, args)
                        place_pref_penalty = _candidate_place_orientation_penalty(candidate, demo, args)
                        score = float(score) + float(selection_penalty) + float(place_pref_penalty)
                        item = dict(candidate)
                        item["result"] = goalset_result
                        item["q_path"] = q_path
                        item["metrics"] = metrics
                        item["score"] = score
                        item["selection_penalty"] = selection_penalty
                        item["place_preference_penalty"] = place_pref_penalty
                        item["terminal_align"] = terminal_align
                        item["planner_pose"] = planner_poses[goal_idx]
                        winners.append(item)
                        print(
                            f"[curobo] {candidate_label} goalset success: score={score:.3f}, "
                            f"selection_penalty={selection_penalty:.3f}, "
                            f"place_pref_penalty={place_pref_penalty:.3f}, "
                            f"waypoints={metrics['waypoint_count']}"
                        )
                        return winners
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
                if not _validate_candidate_joint_path_with_demo_planner(
                    demo,
                    start_q,
                    q_path,
                    use_attach=use_attach,
                    label=candidate_label,
                ):
                    continue
                metrics, score = _path_metrics_and_score(start_q, q_path)
                selection_penalty = _candidate_selection_penalty(candidate, args)
                place_pref_penalty = _candidate_place_orientation_penalty(candidate, demo, args)
                score = float(score) + float(selection_penalty) + float(place_pref_penalty)
                item = dict(candidate)
                item["result"] = result
                item["q_path"] = q_path
                item["metrics"] = metrics
                item["score"] = score
                item["selection_penalty"] = selection_penalty
                item["place_preference_penalty"] = place_pref_penalty
                item["terminal_align"] = terminal_align
                item["planner_pose"] = planner_pose
                print(
                    f"[curobo] {candidate_label} success: score={score:.3f}, "
                    f"selection_penalty={selection_penalty:.3f}, "
                    f"place_pref_penalty={place_pref_penalty:.3f}, "
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
    finally:
        _set_world_collision_for_links(
            planner,
            disabled_world_collision_links,
            enabled=True,
            label=label,
        )

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
    include_table: bool = False,
    exclude_object_names: set[str] | None = None,
    disabled_world_collision_links: list[str] | None = None,
):
    _refresh_curobo_world(planner, demo, args, label=label, include_active_object=include_active_object, include_table=include_table, exclude_object_names=exclude_object_names)
    remaining = list(candidates or [])
    remaining.sort(key=_pre_place_screen_sort_key)
    if not remaining:
        return []

    ee_link_name = str(getattr(planner.config, "ee_link", "gripper_tcp"))
    if bool(getattr(args, "curobo_debug", False)):
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
    ik_prefilter_pos_thresh = float(getattr(args, "curobo_ik_prefilter_position_threshold", 0.01) or 0.0)
    ik_prefilter_rot_thresh = float(getattr(args, "curobo_ik_prefilter_rotation_threshold", 0.25) or 0.0)
    use_soft_prefilter = ik_prefilter_pos_thresh > 0 and ik_prefilter_rot_thresh > 0
    ik_screened = []
    ik_status_counts: dict[str, int] = {}
    ik_pos_errors: list[float] = []
    ik_rot_errors: list[float] = []
    strict_pass_count = 0
    soft_pass_count = 0
    ik_screen_total = len(remaining)
    for start_idx in range(0, len(remaining), chunk_size):
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
        ik_results = planner.solve_batch_start_goal_ik(
            start_qs,
            planner_poses,
            num_seeds=int(getattr(args, "curobo_num_ik_seeds", 64) if num_ik_seeds is None else num_ik_seeds),
        )
        for candidate, ik_result in zip(chunk, ik_results):
            status_key = str(ik_result.status)
            ik_status_counts[status_key] = ik_status_counts.get(status_key, 0) + 1
            dbg = getattr(ik_result, "debug", {}) or {}
            pos_err = dbg.get("position_error")
            rot_err = dbg.get("rotation_error")
            if pos_err is not None and np.isfinite(pos_err):
                ik_pos_errors.append(float(pos_err))
            if rot_err is not None and np.isfinite(rot_err):
                ik_rot_errors.append(float(rot_err))
            if ik_result.success:
                ik_screened.append(candidate)
                strict_pass_count += 1
            elif use_soft_prefilter and pos_err is not None and rot_err is not None:
                if float(pos_err) <= ik_prefilter_pos_thresh and float(rot_err) <= ik_prefilter_rot_thresh:
                    ik_screened.append(candidate)
                    soft_pass_count += 1
    remaining = ik_screened
    remaining.sort(key=_pre_place_screen_sort_key)
    status_str = ", ".join(f"{k}={v}" for k, v in sorted(ik_status_counts.items()))
    soft_info = ""
    if use_soft_prefilter:
        soft_info = (
            f" (strict={strict_pass_count}, soft={soft_pass_count}, "
            f"soft_thresholds: pos<={ik_prefilter_pos_thresh:.4f} rot<={ik_prefilter_rot_thresh:.4f})"
        )
    print(
        f"[curobo] {label} IK prefilter kept {len(remaining)}/{ik_screen_total} "
        f"start-goal pair(s); statuses: {status_str}{soft_info}"
    )
    if not remaining:
        if ik_pos_errors:
            pos_arr = np.array(ik_pos_errors)
            rot_arr = np.array(ik_rot_errors) if ik_rot_errors else np.array([float("nan")])
            print(
                f"[curobo][diag] {label} IK error summary across {len(ik_pos_errors)} failed pair(s): "
                f"pos_err min={float(pos_arr.min()):.6f} median={float(np.median(pos_arr)):.6f} max={float(pos_arr.max()):.6f}  "
                f"rot_err min={float(rot_arr.min()):.6f} median={float(np.median(rot_arr)):.6f} max={float(rot_arr.max()):.6f}"
            )
            pos_thresh = float(planner.config.position_threshold)
            rot_thresh = float(planner.config.rotation_threshold)
            near_pass = sum(1 for p, r in zip(ik_pos_errors, ik_rot_errors) if p < pos_thresh * 2.0 and r < rot_thresh * 2.0)
            print(
                f"[curobo][diag] {label} IK thresholds: position={pos_thresh:.4f} rotation={rot_thresh:.4f}  "
                f"candidates within 2x threshold: {near_pass}/{len(ik_pos_errors)}"
            )
        return []

    num_chunks = max(1, (len(remaining) + chunk_size - 1) // chunk_size)
    print(
        f"[curobo] {label} evaluating {len(remaining)} start-goal pair(s) "
        f"in {num_chunks} batch chunk(s) of size <= {chunk_size}"
    )

    if include_table and bool(getattr(args, "curobo_table_collision", True)):
        _refresh_curobo_world(
            planner, demo, args, label=f"{label}_trajopt",
            include_active_object=include_active_object,
            include_table=True,
            exclude_object_names=exclude_object_names,
        )

    disabled_world_collision_links = _set_world_collision_for_links(
        planner,
        disabled_world_collision_links,
        enabled=False,
        label=label,
    )
    try:
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
            batch_failures: list[tuple[str, str]] = []
            for candidate, planner_pose, result in zip(chunk, planner_poses, batch_results):
                candidate_label = str(candidate["label"])
                if not result.success or result.joint_path is None:
                    batch_failures.append((candidate_label, str(result.status)))
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
                if not _validate_candidate_joint_path_with_demo_planner(
                    demo,
                    start_q,
                    q_path,
                    use_attach=use_attach,
                    label=candidate_label,
                ):
                    continue
                metrics, score = _path_metrics_and_score(start_q, q_path)
                selection_penalty = _candidate_selection_penalty(candidate, args)
                place_pref_penalty = _candidate_place_orientation_penalty(candidate, demo, args)
                score = float(score) + float(selection_penalty) + float(place_pref_penalty)
                item = dict(candidate)
                item["result"] = result
                item["q_path"] = q_path
                item["metrics"] = metrics
                item["score"] = score
                item["selection_penalty"] = selection_penalty
                item["place_preference_penalty"] = place_pref_penalty
                item["terminal_align"] = terminal_align
                item["planner_pose"] = planner_pose
                print(
                    f"[curobo] {candidate_label} success: score={score:.3f}, "
                    f"selection_penalty={selection_penalty:.3f}, "
                    f"place_pref_penalty={place_pref_penalty:.3f}, "
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
            if batch_failures:
                by_status = Counter(st for _, st in batch_failures)
                by_label = Counter(lbl for lbl, _ in batch_failures)
                print(
                    f"[curobo] {label} batch chunk {chunk_idx}/{num_chunks}: "
                    f"{len(batch_failures)}/{len(chunk)} pair(s) failed | "
                    f"statuses={dict(by_status)} | top_labels={by_label.most_common(6)}"
                )
            if len(winners) >= use_max_winners:
                break
    finally:
        _set_world_collision_for_links(
            planner,
            disabled_world_collision_links,
            enabled=True,
            label=label,
        )

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
    obj_p0, obj_q0 = demo.get_obj_pose()
    T_world_obj0 = targeted.base.pose_to_matrix(obj_p0, obj_q0)
    grasp_mode = str(getattr(args, "grasp_mode", "object_normal") or "object_normal").strip().lower()
    place_rule = targeted.get_place_rule(getattr(args, "object_name", None))
    grasp_variant_args = SimpleNamespace(**vars(args))
    grasp_variant_args.topdown_grasp_yaw_variant_deg = [0.0]
    object_axis_world = None
    object_long_axis_half = None
    try:
        extents = np.asarray(
            targeted.base.get_asset_box_size(args.sim_asset_file, args.sim_asset_scale),
            dtype=np.float32,
        ).reshape(3)
        axis_idx = int(np.argmax(extents))
        object_long_axis_half = float(extents[axis_idx]) * 0.5

        # 检测球形物体：三个维度都接近
        extent_ratios = extents / (np.min(extents) + 1e-6)
        is_spherical = bool(np.max(extent_ratios) < 1.3)  # 最大维度不超过最小维度的1.3倍
        if is_spherical:
            print(
                f"[direct_grasp] detected spherical object (extents={np.round(extents * 1000, 1).tolist()}mm), "
                "disabling axis shifts, z lifts, and tilt variants"
            )
            object_axis_world = None
            object_long_axis_half = None
        elif float(np.max(extents)) >= 1.5 * float(max(np.sort(extents)[1], 1e-6)):
            T_world_obj = targeted.base.pose_to_matrix(*demo.get_obj_pose())
            axis_local = np.zeros(3, dtype=np.float32)
            axis_local[axis_idx] = 1.0
            object_axis_world = _normalize(T_world_obj[:3, :3] @ axis_local)
    except Exception:
        object_axis_world = None
        is_spherical = False

    rule_grasp_bias_variants = list(getattr(place_rule, "grasp_bias_variants", ()) or []) if place_rule is not None else []
    use_rule_bias_variants = bool(rule_grasp_bias_variants) and object_axis_world is not None

    if use_rule_bias_variants:
        grasp_axis_shifts = [float(v.axis_shift_m) for v in rule_grasp_bias_variants]
    else:
        grasp_axis_shifts = _unique_finite_float_list(
            getattr(args, "direct_grasp_object_axis_shifts_m", [0.0]),
        )
    if not grasp_axis_shifts:
        grasp_axis_shifts = [0.0]

    # 球形物体：强制只使用中心抓取
    if is_spherical:
        grasp_axis_shifts = [0.0]
    max_axis_shift_ratio = float(max(getattr(args, "direct_grasp_max_axis_shift_ratio", 0.35), 0.0))
    if object_long_axis_half is not None and max_axis_shift_ratio > 0:
        max_abs_shift = object_long_axis_half * max_axis_shift_ratio
        before_count = len(grasp_axis_shifts)
        grasp_axis_shifts = [s for s in grasp_axis_shifts if abs(float(s)) <= max_abs_shift + 1e-6]
        if not grasp_axis_shifts:
            grasp_axis_shifts = [0.0]
        if len(grasp_axis_shifts) < before_count:
            print(
                f"[direct_grasp] axis_shift filter: object half-length={object_long_axis_half * 1000:.1f}mm, "
                f"max_ratio={max_axis_shift_ratio:.2f} -> max_shift={max_abs_shift * 1000:.1f}mm, "
                f"kept {len(grasp_axis_shifts)}/{before_count} shift value(s)"
            )
    if use_rule_bias_variants:
        grasp_z_lifts = [float(v.z_lift_m) for v in rule_grasp_bias_variants]
    else:
        grasp_z_lifts = _unique_finite_float_list(
            getattr(args, "direct_grasp_z_lifts_m", [0.0]),
            min_value=0.0,
        )
    if not grasp_z_lifts:
        grasp_z_lifts = [0.0]

    # 球形物体：禁用z lift
    if is_spherical:
        grasp_z_lifts = [0.0]

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

        if use_rule_bias_variants:
            for bias in rule_grasp_bias_variants:
                pose = raw_grasp_pose
                tilt_deg = float(bias.tilt_toward_robot_deg)
                if abs(tilt_deg) > 1e-6:
                    pose = targeted.base.tilt_pose_toward_robot(
                        demo,
                        pose,
                        tilt_deg,
                        direction=str(getattr(bias, "tilt_direction", "toward_robot")),
                    )
                    if pose is None:
                        continue
                shift_d = float(bias.tilt_shift_m)
                if abs(shift_d) > 1e-6:
                    pose = targeted.base.shift_pose_toward_robot_xy(demo, pose, shift_d)
                    if pose is None:
                        continue
                label = str(bias.label) if getattr(bias, "label", None) else "grasp"
                _append(label, pose)
        else:
            for label, pose in targeted.base.build_grasp_pose_variants(demo, raw_grasp_pose, grasp_variant_args):
                _append(label, pose)

        if use_rule_bias_variants:
            tilt_degs = []
        else:
            tilt_degs = _unique_finite_float_list(getattr(args, "direct_grasp_tilt_toward_robot_deg", [12.0, 20.0, 30.0, 45.0]))
        # 球形物体：禁用tilt variants，因为倾斜抓取容易滑脱
        if is_spherical:
            tilt_degs = []

        if use_rule_bias_variants:
            shift_ds = [0.0]
        else:
            shift_ds = _unique_finite_float_list(
                getattr(args, "direct_grasp_tilt_toward_robot_shift_m", [0.0, 0.02]),
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
                base_label = f"grasp_tilt_{int(round(abs(float(tilt_deg))))}deg"
                if float(shift_d) > 1e-6:
                    base_label += f"_shift_{int(round(1000.0 * float(shift_d)))}mm"
                _append(base_label, shifted_pose)
        return variants

    grasp_variants = _build_pose_variants()
    candidates = []
    seen = set()
    if use_rule_bias_variants:
        bias_iter = []
        for grasp_variant_label, grasp_variant_pose in grasp_variants:
            matched_bias = next(
                (bias for bias in rule_grasp_bias_variants if str(getattr(bias, "label", "")) == str(grasp_variant_label)),
                None,
            )
            if matched_bias is None:
                continue
            bias_iter.append(((grasp_variant_label, grasp_variant_pose), matched_bias))
    else:
        bias_iter = [((grasp_variant_label, grasp_variant_pose), None) for grasp_variant_label, grasp_variant_pose in grasp_variants]

    for grasp_variant_item, rule_bias in bias_iter:
        grasp_variant_label, grasp_variant_pose = grasp_variant_item
        axis_shift_values = [float(rule_bias.axis_shift_m)] if rule_bias is not None else grasp_axis_shifts
        z_lift_values = [float(rule_bias.z_lift_m)] if rule_bias is not None else grasp_z_lifts
        for axis_shift in axis_shift_values:
            for z_lift in z_lift_values:
                current_grasp_pose = grasp_variant_pose
                if object_axis_world is not None and abs(float(axis_shift)) > 1e-6:
                    shifted_p = (_get_pose_position(current_grasp_pose) + object_axis_world * float(axis_shift)).astype(np.float32)
                    current_grasp_pose = targeted.base.make_pose_with_position(current_grasp_pose, shifted_p)
                if float(z_lift) > 1e-6:
                    shifted_p = (_get_pose_position(current_grasp_pose) + np.array([0.0, 0.0, float(z_lift)], dtype=np.float32)).astype(
                        np.float32
                    )
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
                if float(z_lift) > 1e-6:
                    label = f"{label}_lift_{int(round(float(z_lift) * 1000.0))}mm"
                candidates.append(
                    {
                        "label": label,
                        "pose": current_grasp_pose,
                        "pregrasp_pose": current_pregrasp_pose,  # 保存pregrasp pose用于两步抓取
                        "T_tcp_obj": np.linalg.inv(
                            targeted.base.pose_to_matrix(
                                targeted.base.flatten_np(current_grasp_pose.p)[:3],
                                targeted.base.flatten_np(current_grasp_pose.q)[:4],
                            )
                        ) @ T_world_obj0,
                        "grasp_axis_shift_m": float(axis_shift),
                        "grasp_z_lift_m": float(z_lift),
                    }
                )
    variant_labels = [label for label, _ in grasp_variants]
    tilt_variants = [l for l in variant_labels if "tilt" in l.lower()]
    non_tilt_cands = [c for c in candidates if "tilt" not in str(c["label"]).lower()]
    tilt_cands = [c for c in candidates if "tilt" in str(c["label"]).lower()]
    if tilt_cands and non_tilt_cands:
        interleaved = []
        ratio = max(1, len(non_tilt_cands) // max(len(tilt_cands), 1))
        ti = 0
        ni = 0
        while ni < len(non_tilt_cands) or ti < len(tilt_cands):
            for _ in range(min(ratio, len(non_tilt_cands) - ni)):
                interleaved.append(non_tilt_cands[ni])
                ni += 1
            if ti < len(tilt_cands):
                interleaved.append(tilt_cands[ti])
                ti += 1
        candidates = interleaved
    print(
        f"[direct_grasp] built {len(candidates)} grasp candidate(s) "
        f"from {len(grasp_variants)} pose variant(s)"
        f"{'' if not use_rule_bias_variants else ' (rule-paired bias variants)'}"
        f"{'' if use_rule_bias_variants else f' x {len(grasp_axis_shifts)} axis shift(s) x {len(grasp_z_lifts)} z lift(s)'}; "
        f"grasp_mode={grasp_mode}, tilt={len(tilt_cands)}, non_tilt={len(non_tilt_cands)}"
    )
    if use_rule_bias_variants:
        print(
            "[direct_grasp] place-rule grasp bias variants: "
            + ", ".join(
                f"{getattr(v, 'label', 'bias')}[axis={float(v.axis_shift_m):.3f}, tilt={float(v.tilt_toward_robot_deg):.1f}, tilt_shift={float(v.tilt_shift_m):.3f}, z_lift={float(v.z_lift_m):.3f}]"
                f"/dir={getattr(v, 'tilt_direction', 'toward_robot')}"
                for v in rule_grasp_bias_variants
            )
        )
    if tilt_variants:
        print(f"[direct_grasp] tilt variant labels: {tilt_variants[:8]}")
    return candidates


def _compute_object_lowest_point_offset(pose, object_dims):
    """
    计算物体在给定姿态下，最低点相对于中心点的Z偏移。
    当物体倾斜时，最低点会比中心点更低。

    Args:
        pose: 物体的放置姿态
        object_dims: 物体的尺寸 [x, y, z]

    Returns:
        最低点的Z偏移（负值表示低于中心）
    """
    try:
        # 获取旋转矩阵
        q = targeted.base.flatten_np(pose.q)[:4]
        from scipy.spatial.transform import Rotation
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()  # xyzw -> wxyz

        # 物体的8个顶点（在物体坐标系中）
        dims = np.asarray(object_dims, dtype=np.float32) / 2.0
        corners = np.array([
            [-dims[0], -dims[1], -dims[2]],
            [-dims[0], -dims[1], +dims[2]],
            [-dims[0], +dims[1], -dims[2]],
            [-dims[0], +dims[1], +dims[2]],
            [+dims[0], -dims[1], -dims[2]],
            [+dims[0], -dims[1], +dims[2]],
            [+dims[0], +dims[1], -dims[2]],
            [+dims[0], +dims[1], +dims[2]],
        ], dtype=np.float32)

        # 转换到世界坐标系
        corners_world = (R @ corners.T).T

        # 找到最低点的Z偏移
        min_z_offset = float(np.min(corners_world[:, 2]))
        return min_z_offset
    except Exception as e:
        print(f"[warning] failed to compute lowest point offset: {e}")
        return 0.0


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


def _build_direct_pre_place_candidates(demo, bridge_mod, scene_capture_cache, rule, place_state_cache, args, *, T_tcp_obj_override=None):
    place_plan_candidates = targeted.build_targeted_place_plan_variants(
        demo,
        bridge_mod,
        scene_capture_cache,
        rule,
        place_state_cache,
        args,
        T_tcp_obj_override=T_tcp_obj_override,
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

    if rule.primitive == "insert_vertical":
        approach_distances = [float(max(getattr(args, "direct_release_approach_distance", 0.04), 0.005))]
        z_offsets = [0.0]
    else:
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
    max_pad_tilt = float(max(getattr(args, "direct_place_max_pad_tilt", 0.25), 0.0))

    # 获取物体尺寸用于计算倾斜时的最低点偏移
    try:
        object_dims = targeted.base.get_asset_box_size(args.sim_asset_file, args.sim_asset_scale)
    except Exception:
        object_dims = None

    all_candidates = []
    level_candidates = []
    for candidate in place_plan_candidates:
        pad_tilt = _tcp_pad_tilt_z(candidate.place_pose)
        place_z = float(_get_pose_position(candidate.place_pose)[2])

        # 计算物体倾斜时的最低点偏移
        lowest_point_offset = 0.0
        if object_dims is not None:
            lowest_point_offset = _compute_object_lowest_point_offset(candidate.place_pose, object_dims)

        # 调整最小高度检查：考虑物体最低点
        effective_min_z = min_place_tcp_z - lowest_point_offset

        vlabel = candidate.variant_label or "base"
        if place_z < effective_min_z:
            if object_dims is not None:
                print(
                    f"[direct_pre_place] rejected {vlabel}: place_z={place_z:.4f}, "
                    f"lowest_offset={lowest_point_offset:.4f}, effective_min_z={effective_min_z:.4f}"
                )
            continue
        is_level = pad_tilt <= max_pad_tilt
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
                # 同样考虑物体最低点偏移
                if pose_z < effective_min_z:
                    continue
                label = "pre_place" if candidate.variant_label is None else f"pre_place_{candidate.variant_label}"
                if len(approach_distances) > 1:
                    label = f"{label}_approach_{int(round(float(approach_distance) * 1000.0))}mm"
                if abs(float(z_offset)) > 1e-6:
                    label = f"{label}_lift_{int(round(float(z_offset) * 1000.0))}mm"
                item = {
                    "label": label,
                    "pose": pose,
                    "place_pose": candidate.place_pose,
                    "retreat_pose": pose,
                    "target_name": candidate.target_name,
                    "slot_name": candidate.slot_name,
                    "variant_label": candidate.variant_label,
                    "tcp_verticality": float(candidate.tcp_verticality),
                    "pad_tilt": float(pad_tilt),
                    "place_plan": candidate,
                }
                all_candidates.append(item)
                if is_level:
                    level_candidates.append(item)
    if level_candidates:
        candidates = _dedupe_place_candidates(level_candidates)
    elif all_candidates:
        all_candidates.sort(key=lambda c: float(c["pad_tilt"]))
        candidates = _dedupe_place_candidates(all_candidates)
        print(
            f"[direct_pre_place] WARNING: no candidate passed pad-level filter "
            f"(max_pad_tilt={max_pad_tilt:.2f}), keeping all {len(candidates)} sorted by pad_tilt"
        )
    else:
        candidates = []
    print(
        f"[direct_pre_place] built {len(candidates)} short pre-place candidate(s) "
        f"with approach_distances={np.round(np.asarray(approach_distances, dtype=np.float32), 4).tolist()} "
        f"and z_offsets={np.round(np.asarray(z_offsets, dtype=np.float32), 4).tolist()}"
    )
    return candidates


def _build_direct_place_candidates(demo, bridge_mod, scene_capture_cache, rule, place_state_cache, args, *, T_tcp_obj_override=None):
    place_plan_candidates = targeted.build_targeted_place_plan_variants(
        demo,
        bridge_mod,
        scene_capture_cache,
        rule,
        place_state_cache,
        args,
        T_tcp_obj_override=T_tcp_obj_override,
    )
    if not place_plan_candidates:
        return []

    min_place_tcp_z = float(getattr(args, "direct_min_place_tcp_z", 0.005))
    max_pad_tilt = float(max(getattr(args, "direct_place_max_pad_tilt", 0.25), 0.0))
    max_abs_yaw = float(getattr(args, "direct_place_max_abs_yaw_deg", 0.0) or 0.0)
    all_candidates = []
    level_candidates = []
    for candidate in place_plan_candidates:
        pad_tilt = _tcp_pad_tilt_z(candidate.place_pose)
        place_z = float(_get_pose_position(candidate.place_pose)[2])
        vlabel = candidate.variant_label or "base"
        if place_z < min_place_tcp_z:
            continue
        label = "place_direct" if candidate.variant_label is None else f"place_direct_{candidate.variant_label}"
        if max_abs_yaw > 0.0:
            ydeg = _variant_abs_yaw_deg_from_labels(candidate.variant_label, label)
            if ydeg > max_abs_yaw + 1e-4:
                continue
        item = {
            "label": label,
            "pose": candidate.place_pose,
            "place_pose": candidate.place_pose,
            "retreat_pose": candidate.pre_place_pose,
            "target_name": candidate.target_name,
            "slot_name": candidate.slot_name,
            "variant_label": candidate.variant_label,
            "tcp_verticality": float(candidate.tcp_verticality),
            "pad_tilt": float(pad_tilt),
            "place_plan": candidate,
            "direct_place_mode": True,
        }
        all_candidates.append(item)
        if pad_tilt <= max_pad_tilt:
            level_candidates.append(item)
    if level_candidates:
        candidates = _dedupe_place_candidates(level_candidates)
        candidates.sort(key=_pre_place_screen_sort_key)
        print(
            f"[direct_place] built {len(candidates)} direct place candidate(s) "
            f"(pad-level filter kept {len(level_candidates)}/{len(all_candidates)}, max_pad_tilt={max_pad_tilt:.2f})"
        )
    elif all_candidates:
        all_candidates.sort(key=lambda c: float(c["pad_tilt"]))
        candidates = _dedupe_place_candidates(all_candidates)
        candidates.sort(key=_pre_place_screen_sort_key)
        best_tilt = float(candidates[0]["pad_tilt"]) if candidates else float("nan")
        print(
            f"[direct_place] WARNING: no candidate passed pad-level filter (max_pad_tilt={max_pad_tilt:.2f}), "
            f"keeping all {len(candidates)} candidate(s) sorted by pad_tilt (best={best_tilt:.3f}). "
            f"This place task may require a tilted gripper."
        )
    else:
        candidates = []
        print("[direct_place] built 0 direct place candidate(s) (all rejected by tcp_z threshold)")
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
    demo._last_joint_chain_failed_pose = None
    demo._last_joint_chain_failed_label = None
    demo._last_joint_chain_failed_candidate_poses = []
    targeted._register_transport_attached_box(
        demo,
        args,
        show_visual=False,
        activate_payload_visual=False,
    )
    chains = []
    grasp_candidates = list(grasp_successes or [])
    max_grasp_candidates = int(getattr(args, "joint_search_max_grasp_candidates", 8))
    if max_grasp_candidates > 0 and len(grasp_candidates) > max_grasp_candidates:
        non_tilt = [c for c in grasp_candidates if "tilt" not in str(c.get("label", "")).lower()]
        tilted = [c for c in grasp_candidates if "tilt" in str(c.get("label", "")).lower()]
        min_tilt_slots = min(max(max_grasp_candidates // 3, 1), len(tilted))
        non_tilt_slots = max_grasp_candidates - min_tilt_slots
        selected = non_tilt[:non_tilt_slots] + tilted[:min_tilt_slots]
        if len(selected) < max_grasp_candidates:
            remaining_pool = [c for c in grasp_candidates if c not in selected]
            selected.extend(remaining_pool[: max_grasp_candidates - len(selected)])
        grasp_candidates = selected
    all_labels = [str(c.get("label", "?")) for c in grasp_candidates]
    tilt_count = sum(1 for l in all_labels if "tilt" in l.lower())
    print(
        f"[joint_search] grasp winners for chain evaluation: {len(grasp_candidates)} "
        f"(tilted={tilt_count}): {all_labels}"
    )
    max_feasible_chains = int(getattr(args, "joint_search_max_feasible_chains", 1))
    screen_timeout = float(getattr(args, "joint_search_screen_timeout", 0.0))
    screen_max_attempts = int(getattr(args, "joint_search_screen_max_attempts", 0))
    screen_num_ik_seeds = int(getattr(args, "joint_search_screen_num_ik_seeds", 0))
    screen_num_trajopt_seeds = int(getattr(args, "joint_search_screen_num_trajopt_seeds", 0))
    screen_timeout_arg = None if screen_timeout <= 0.0 else screen_timeout
    screen_max_attempts_arg = None if screen_max_attempts <= 0 else screen_max_attempts
    screen_num_ik_seeds_arg = None if screen_num_ik_seeds <= 0 else screen_num_ik_seeds
    screen_num_trajopt_seeds_arg = None if screen_num_trajopt_seeds <= 0 else screen_num_trajopt_seeds
    direct_pair_candidates = []
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
                T_tcp_obj_override=grasp_choice.get("T_tcp_obj"),
            )
            if rule.primitive == "insert_vertical":
                direct_place_candidates = _filter_pre_place_candidates_by_verticality(direct_place_candidates, args)
            direct_place_candidates.sort(key=_pre_place_screen_sort_key)
            if max_place_candidates > 0:
                direct_place_candidates = direct_place_candidates[:max_place_candidates]
            if direct_place_candidates and demo._last_joint_chain_failed_pose is None:
                demo._last_joint_chain_failed_pose = direct_place_candidates[0]["pose"]
                demo._last_joint_chain_failed_label = str(direct_place_candidates[0]["label"])
            for candidate in direct_place_candidates:
                pair_item = dict(candidate)
                pair_item["start_q"] = grasp_terminal_q
                pair_item["grasp_choice"] = grasp_choice
                direct_pair_candidates.append(pair_item)
                demo._last_joint_chain_failed_candidate_poses.append(candidate["pose"])

        if direct_pair_candidates:
            print(
                f"[joint_search] direct-place pair candidate count: {len(direct_pair_candidates)} "
                f"from {len(grasp_candidates)} grasp winner(s)"
            )
            target_obj_name = curobo_wrapper.normalize_object_name(
                getattr(rule, "target_object_name", None)
            )
            exclude_names = {target_obj_name} if (rule.primitive == "insert_vertical" and target_obj_name) else None
            direct_place_disabled_links = _direct_place_contact_tolerant_disabled_links(planner)
            direct_place_successes = _evaluate_curobo_pose_candidates_multi_start(
                planner,
                demo,
                args,
                direct_pair_candidates,
                label="joint_direct_place_pairs",
                use_attach=True,
                timeout=screen_timeout_arg,
                max_attempts=screen_max_attempts_arg,
                num_ik_seeds=screen_num_ik_seeds_arg,
                num_trajopt_seeds=screen_num_trajopt_seeds_arg,
                include_table=True,
                exclude_object_names=exclude_names,
                disabled_world_collision_links=direct_place_disabled_links,
            )
            for candidate in direct_place_successes:
                grasp_choice = candidate["grasp_choice"]
                grasp_label = str(grasp_choice["label"])
                q_direct_place_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in candidate["q_path"]]
                release_metrics, release_score = _path_metrics_and_score(
                    q_direct_place_path[-1],
                    [np.asarray(q_direct_place_path[-1], dtype=np.float32).reshape(-1)[:7]],
                )
                total_score = float(grasp_choice["score"]) + float(candidate["score"]) + float(release_score)
                chain = {
                    "grasp_choice": grasp_choice,
                    "pre_place_choice": candidate,
                    "q_pre_place_path": q_direct_place_path,
                    "q_place_path": [np.asarray(q_direct_place_path[-1], dtype=np.float32).reshape(-1)[:7]],
                    "release_metrics": release_metrics,
                    "release_score": float(release_score),
                    "total_score": total_score,
                    "direct_place_mode": True,
                }
                print(
                    f"[joint_search] feasible direct-place chain: grasp={grasp_label}, "
                    f"place={candidate['label']}, total_score={total_score:.3f}, "
                    f"waypoints={len(q_direct_place_path)}"
                )
                chains.append(chain)
                if max_feasible_chains > 0 and len(chains) >= max_feasible_chains:
                    break
        if not chains:
            print("[joint_search] no direct-place chain stayed feasible")
    finally:
        targeted.base.sync_demo_arm_qpos(demo, saved_q)

    if not chains:
        return []

    demo._last_joint_chain_failed_pose = None
    demo._last_joint_chain_failed_label = None
    demo._last_joint_chain_failed_candidate_poses = []
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
    two_step_pregrasp_lookup = {}
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

    print("\n[move to grasp with two-step approach]")
    grasp_candidates = _build_direct_grasp_candidates(demo, args)
    print("\n[poses]")
    print("object p:", np.round(demo.get_obj_pose()[0], 6), "object q:", np.round(demo.get_obj_pose()[1], 6))
    direct_grasp_disabled_links: list[str] = []
    two_step_pregrasp_successes = _evaluate_two_step_grasp_candidates(
        planner,
        demo,
        args,
        np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7],
        grasp_candidates,
        label="two_step_grasp",
        max_winners=int(getattr(args, "direct_grasp_goalset_max_winners", 12)),
        include_active_object=True,
        disabled_world_collision_links=direct_grasp_disabled_links,
    )
    if two_step_pregrasp_successes:
        two_step_pregrasp_lookup = {
            str(item.get("label", "")): item for item in list(two_step_pregrasp_successes or [])
        }
        print(f"[grasp] explicit two-step grasp pregrasp succeeded for {len(two_step_pregrasp_lookup)} candidate(s)")
    else:
        print("[FAIL] two-step grasp pregrasp planning failed")
        selected_pose = grasp_candidates[0].get("pregrasp_pose", grasp_candidates[0]["pose"]) if grasp_candidates else demo.build_topdown_grasp_pose()
        targeted.base.inspect_failed_pose(
            demo,
            bridge_mod,
            "grasp_pregrasp",
            args,
            pose=selected_pose,
            gripper_closed=False,
            candidate_poses=[item.get("pregrasp_pose", item["pose"]) for item in grasp_candidates],
        )
        return False

    grasp_successes = []
    for pregrasp_success in list(two_step_pregrasp_successes or []):
        completed = _apply_deferred_two_step_final_approach(
            planner,
            demo,
            args,
            pregrasp_success,
            two_step_pregrasp_lookup,
        )
        if bool(completed.get("two_step_grasp", False)):
            grasp_successes.append(completed)
    grasp_successes.sort(key=_candidate_sort_key)
    if not grasp_successes:
        selected_pose = grasp_candidates[0]["pose"] if grasp_candidates else demo.build_topdown_grasp_pose()
        print("[FAIL] explicit two-step final approach failed for all grasp candidates")
        targeted.base.inspect_failed_pose(
            demo,
            bridge_mod,
            "grasp",
            args,
            pose=selected_pose,
            gripper_closed=False,
            candidate_poses=[item["pose"] for item in grasp_candidates],
        )
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
            failed_place_pose = getattr(demo, "_last_joint_chain_failed_pose", None)
            failed_place_label = str(getattr(demo, "_last_joint_chain_failed_label", "") or "place")
            failed_place_candidates = list(getattr(demo, "_last_joint_chain_failed_candidate_poses", []) or [])
            targeted.base.inspect_failed_pose(
                demo,
                bridge_mod,
                failed_place_label if failed_place_pose is not None else "grasp",
                args,
                pose=failed_place_pose if failed_place_pose is not None else grasp_choice["pose"],
                gripper_closed=True if failed_place_pose is not None else False,
                use_attach=True if failed_place_pose is not None else False,
                candidate_poses=failed_place_candidates if failed_place_pose is not None else [item["pose"] for item in grasp_candidates],
            )
            return False
        selected_joint_chain = joint_chains[0]
        grasp_choice = selected_joint_chain["grasp_choice"]

    pregrasp_waypoints = int(max(grasp_choice.get("pregrasp_waypoints", 0), 0))
    if pregrasp_waypoints > 0:
        q_pregrasp_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in grasp_choice["q_path"][:pregrasp_waypoints]]
        pregrasp_pose = grasp_choice.get("deferred_pregrasp_pose", grasp_choice.get("pregrasp_pose", grasp_choice["pose"]))
        ok, _ = targeted.base.execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            f"{grasp_choice['label']}_pregrasp",
            pregrasp_pose,
            q_pregrasp_path,
            args.real_gripper_open,
            args,
        )
        if not ok:
            return False

        q_approach_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in grasp_choice["q_path"][pregrasp_waypoints - 1 :]]
        ok, _ = targeted.base.execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            f"{grasp_choice['label']}_final_approach",
            grasp_choice["pose"],
            q_approach_path,
            args.real_gripper_open,
            args,
        )
        if not ok:
            return False
    else:
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

    _stabilize_post_grasp_attached_state(demo, args, grasp_choice)

    targeted.base.lift_active_object_above_table_if_needed(
        demo,
        args,
        min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
    )

    if bool(getattr(args, "curobo_attach_object", True)):
        try:
            attach_box_dims = targeted.base.get_asset_box_size(args.sim_asset_file, args.sim_asset_scale)
            attach_box_scale = float(np.clip(getattr(args, "transport_attached_box_scale", 1.0), 0.5, 2.0))

            # 检测球形物体
            attach_box_dims_arr = np.asarray(attach_box_dims, dtype=np.float32)
            extent_ratios = attach_box_dims_arr / (np.min(attach_box_dims_arr) + 1e-6)
            is_spherical = bool(np.max(extent_ratios) < 1.3)

            # 分别处理XY和Z维度：XY使用scale因子，Z保持原始尺寸避免影响垂直放置
            # 但对于球形物体，Z方向也需要放大以确保完全覆盖
            scale_xy_override = getattr(args, "transport_attached_box_scale_xy", None)
            if scale_xy_override is None:
                attach_box_scale_xy = attach_box_scale
            else:
                attach_box_scale_xy = float(np.clip(scale_xy_override, 0.5, 2.0))
            if is_spherical:
                # 球形物体：Z方向也使用XY的scale，确保完全覆盖
                attach_box_scale_z = attach_box_scale_xy
                print(f"[curobo] detected spherical object for transport, using uniform scale={attach_box_scale_xy:.2f}")
            else:
                attach_box_scale_z = float(np.clip(getattr(args, "transport_attached_box_scale_z", 1.0), 0.5, 2.0))

            attach_box_dims = attach_box_dims_arr.copy()
            attach_box_dims[0] *= attach_box_scale_xy  # X
            attach_box_dims[1] *= attach_box_scale_xy  # Y
            attach_box_dims[2] *= attach_box_scale_z   # Z
            attach_box_dims = np.maximum(attach_box_dims, 1e-4)

            print(
                f"[curobo] attaching object box: original_dims={np.round(attach_box_dims_arr * 1000, 1).tolist()}mm, "
                f"scaled_dims={np.round(attach_box_dims * 1000, 1).tolist()}mm, "
                f"scale_xy={attach_box_scale_xy:.2f}, scale_z={attach_box_scale_z:.2f}"
            )

            current_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
            obj_p_attach, obj_q_attach = demo.get_obj_pose()
            planner.attach_object_box_to_robot(
                current_q,
                attach_box_dims,
                object_pose_world=(obj_p_attach, obj_q_attach),
                world_z_offset=float(getattr(args, "curobo_attach_world_z_offset_m", 0.003)),
            )
            if planner.attached_object_active:
                _visualize_attached_spheres(planner, demo, args)
        except Exception as exc:
            print(f"[curobo] failed to attach object for transport collision: {exc}")

    if args.skip_goal_motion:
        print("[place] skipped place motion as requested; returning to the cycle start pose before finishing")
        targeted._register_transport_attached_box(
            demo,
            args,
            show_visual=False,
            activate_payload_visual=False,
            T_tcp_obj_override=getattr(demo, "_transport_attached_T_tcp_obj", None),
        )
        return _plan_and_execute_return_to_cycle_start(
            demo,
            bridge_mod,
            real_exec,
            args,
            start_q,
            use_attach=True,
            gripper_pos=args.real_gripper_close,
        )

    print("\n[move to targeted place]")
    try:
        targeted._ensure_target_registered_for_place(demo, rule.target_object_name)
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return False
    targeted._register_transport_attached_box(
        demo,
        args,
        T_tcp_obj_override=getattr(demo, "_transport_attached_T_tcp_obj", None),
    )
    _skip_post_grasp_escape(demo, bridge_mod, real_exec, args, "post_grasp_escape", use_attach=True)

    direct_place_successes = None
    relaxed_target_collision = False
    direct_place_mode = True
    if selected_joint_chain is not None:
        print(
            "[place] joint_search selected a grasp winner, but transport/place will be "
            "replanned after payload attach instead of reusing the pre-attach place path"
        )

    T_tcp_obj_transport = getattr(demo, "_transport_attached_T_tcp_obj", None)
    if T_tcp_obj_transport is None:
        T_tcp_obj_transport = grasp_choice.get("T_tcp_obj")
    direct_place_candidates = _build_direct_place_candidates(
        demo,
        bridge_mod,
        scene_capture_cache,
        rule,
        place_state_cache,
        args,
        T_tcp_obj_override=T_tcp_obj_transport,
    )
    if not direct_place_candidates:
        print("[FAIL] no direct targeted place candidate could be built")
        return False

    if rule.primitive == "insert_vertical":
        direct_place_candidates = _filter_pre_place_candidates_by_verticality(direct_place_candidates, args)
    if not direct_place_candidates:
        print("[FAIL] direct place candidates became empty after filtering")
        return False

    target_obj_name_place = curobo_wrapper.normalize_object_name(
        getattr(rule, "target_object_name", None)
    )
    exclude_names_place = (
        {target_obj_name_place}
        if (rule.primitive == "insert_vertical" and target_obj_name_place)
        else None
    )
    direct_place_disabled_links = _direct_place_contact_tolerant_disabled_links(planner)
    direct_place_successes = _evaluate_curobo_pose_candidates(
        planner,
        demo,
        args,
        np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7],
        direct_place_candidates,
        label="direct_place",
        prefer_verticality=(rule.primitive == "insert_vertical"),
        use_attach=True,
        include_table=bool(getattr(args, "curobo_table_collision", True)),
        exclude_object_names=exclude_names_place,
        disabled_world_collision_links=direct_place_disabled_links,
    )
    if not direct_place_successes:
        print("[FAIL] cuRobo direct-place planning failed and pre-place fallback is disabled")
        failed_pose = direct_place_candidates[0]["pose"] if direct_place_candidates else demo.tcp.pose
        targeted.base.inspect_failed_pose(
            demo,
            bridge_mod,
            "place",
            args,
            pose=failed_pose,
            gripper_closed=True,
            use_attach=True,
            candidate_poses=[item["pose"] for item in direct_place_candidates],
        )
        return False

    place_choice = direct_place_successes[0]
    q_pre_place_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in place_choice["q_path"]]
    q_place_path = [np.asarray(q_pre_place_path[-1], dtype=np.float32).reshape(-1)[:7]]
    slot_suffix = f", slot={place_choice['slot_name']}" if place_choice.get("slot_name") else ""
    variant_suffix = f", variant={place_choice['variant_label']}" if place_choice.get("variant_label") else ""
    print(
        f"[place] source={args.object_name}, primitive={rule.primitive}, "
        f"target={place_choice['target_name']}{slot_suffix}{variant_suffix}, "
        f"tcp_verticality={place_choice['tcp_verticality']:.3f}"
    )
    print("[place] direct place p:", np.round(targeted.base.flatten_np(place_choice["pose"].p)[:3], 6), "q:", np.round(targeted.base.flatten_np(place_choice["pose"].q)[:4], 6))
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
        if planner.attached_object_active:
            planner.detach_object_from_robot()
        if relaxed_target_collision:
            targeted._set_scene_obstacle_planner_box_scale(demo, place_choice["target_name"], 1.0)
        return False

    # direct-place-only mode: no separate release stage

    print("\n[open gripper at place]")
    if not targeted.base.confirm_simple_action("open the real gripper at the targeted place", args, bridge_mod=bridge_mod, env=demo.env, repeats=6):
        print("[abort] user cancelled before opening the real gripper at the targeted place")
        if planner.attached_object_active:
            planner.detach_object_from_robot()
        if relaxed_target_collision:
            targeted._set_scene_obstacle_planner_box_scale(demo, place_choice["target_name"], 1.0)
        return False
    if real_exec is not None:
        real_exec.set_gripper(args.real_gripper_open)
        targeted.base.sync_demo_gripper_state(demo, closed=False, steps=4)
    else:
        print("[dry-run] skipped real gripper open at the targeted place")
        targeted.base.sync_demo_gripper_state(demo, closed=False, steps=4)

    if planner.attached_object_active:
        planner.detach_object_from_robot()

    demo._attached_box_visual_visible = False
    demo._attached_object_visual_active = False
    targeted.base.update_attached_box_visual(demo, visible=False)
    clearance_pose = targeted.base.make_tcp_axis_retreat_pose(demo, 0.05)
    _refresh_curobo_world(
        planner,
        demo,
        args,
        label="post_place_clearance",
        include_active_object=False,
        include_table=bool(getattr(args, "curobo_table_collision", True)),
    )
    q_clearance_path = _plan_with_official_approach_metric(
        planner,
        demo,
        args,
        np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7],
        demo.tcp.pose,
        clearance_pose,
        label="post_place_clearance",
    )
    if not q_clearance_path:
        _, q_clearance_path = targeted.base.plan_post_place_clearance_path(
            demo,
            retreat_distance=0.05,
            label="post_place_clearance",
        )
    clearance_executed = False
    if q_clearance_path:
        ok, _ = targeted.base.execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            "post_place_clearance",
            clearance_pose,
            q_clearance_path,
            args.real_gripper_open,
            args,
            use_attach=False,
            skip_confirmation=True,
        )
        if not ok:
            print("[warn] post-place clearance execution failed after release; continuing to settle the object in place")
        else:
            clearance_executed = True
            print(
                "[place] post_place_clearance final q:",
                np.round(np.asarray(q_clearance_path[-1], dtype=np.float32).reshape(-1)[:7], 5).tolist(),
            )
    else:
        print("[warn] post-place clearance planning failed after release; settling the object without moving the arm away first")

    targeted.base.settle_released_active_object_for_scene_cache(demo, args)
    targeted._mark_place_rule_success(rule, place_state_cache, place_choice["slot_name"])

    if relaxed_target_collision:
        targeted._set_scene_obstacle_planner_box_scale(demo, place_choice["target_name"], 1.0)
        relaxed_target_collision = False
    if clearance_executed:
        print("[place] completed targeted place and clearance; returning to the cycle start pose")
    else:
        print("[place] completed targeted place without clearance; returning to the cycle start pose from the current release state")
    return _plan_and_execute_return_to_cycle_start(
        demo,
        bridge_mod,
        real_exec,
        args,
        start_q,
    )


def main():
    targeted.parse_args = parse_args
    targeted.run_targeted_place_episode = run_targeted_place_episode_curobo_direct
    targeted.main()


if __name__ == "__main__":
    main()
