"""Curobo2 pick-place entrypoint with cached-perception replay support."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rm75_app.execution.trajectory_executor import RecordingTrajectoryExecutor
from rm75_app.paths import DEFAULT_CAMERA_EXTRINSIC, RUNTIME_DIR
from rm75_app.pickplace.cached_scene import load_cached_known_object_scene, topdown_grasp_candidates
from rm75_app.pickplace.coordinator import PickPlaceCoordinator, PickPlaceTask
from rm75_app.planning.backends.curobo2 import Curobo2Backend, Curobo2BackendConfig
from rm75_app.planning.contracts import JointConfiguration, Pose, PoseCandidate


DEFAULT_CACHED_RESULT = (
    RUNTIME_DIR
    / "rrtrack_manual_init/20260808_203311_carriot_pid1169281/sam6d_pose_result.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the layered RM75 Curobo2 pick-place pipeline")
    parser.add_argument("--cached-pose-result", type=Path, default=DEFAULT_CACHED_RESULT)
    parser.add_argument("--camera-extrinsic", type=Path, default=DEFAULT_CAMERA_EXTRINSIC)
    parser.add_argument("--use-direct-camera-extrinsic", action="store_true")
    parser.add_argument("--grasp-z-offset", type=float, default=0.005)
    parser.add_argument("--approach-offset", type=float, default=-0.08)
    parser.add_argument("--lift-height", type=float, default=0.08)
    parser.add_argument("--place-clearance", type=float, default=0.08)
    parser.add_argument(
        "--place-surface-offset",
        type=float,
        default=0.02,
        help="Release TCP this far above the grasp-height surface pose to avoid gripper/table collision.",
    )
    parser.add_argument(
        "--place-xyz",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Explicit place TCP position. Default batch-screens nearby tabletop slots.",
    )
    parser.add_argument(
        "--smoke-place",
        action="store_true",
        help="Release at the lifted grasp pose; useful only for planner/executor smoke tests.",
    )
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--num-ik-seeds", type=int, default=64)
    parser.add_argument("--num-trajopt-seeds", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--use-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep cuRobo2 on its warmed CUDA-Graph fast path (default: enabled).",
    )
    parser.add_argument(
        "--attachment-num-spheres",
        type=int,
        default=64,
        help="Maximum slots for cuRobo2 automatic attached-object sphere fitting; use 0 only for diagnostics.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def _summary(result, scene, output_dir: Path) -> dict:
    refinement = result.held_object_refinement
    return {
        "schema_version": 1,
        "backend": "curobo2",
        "success": result.success,
        "failure_stage": result.failure_stage,
        "message": result.message,
        "selected_grasp": result.selected_grasp,
        "selected_place": result.selected_place,
        "object_name": scene.object_name,
        "cached_rgb": str(scene.rgb_path),
        "cached_depth": str(scene.depth_path),
        "cached_mask": str(scene.mask_path),
        "T_base_object": scene.T_base_object.tolist(),
        "held_object_refinement": None
        if refinement is None
        else {
            "accepted": refinement.accepted,
            "reason": refinement.reason,
            "source": refinement.source,
            "score": refinement.score,
            "translation_delta_m": refinement.translation_delta_m,
            "rotation_delta_deg": refinement.rotation_delta_deg,
            "T_tcp_object": None
            if refinement.T_tcp_object is None
            else refinement.T_tcp_object.tolist(),
            "metadata": refinement.metadata,
        },
        "stages": [
            {
                "name": stage.name,
                "candidate_id": stage.candidate_id,
                "end_joints": stage.end.positions.tolist(),
            }
            for stage in result.stages
        ],
        "execution_json": str((output_dir / "execution.json").resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else RUNTIME_DIR / "curobo2_pick_place" / time.strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = load_cached_known_object_scene(
        args.cached_pose_result,
        args.camera_extrinsic,
        use_direct_camera_extrinsic=bool(args.use_direct_camera_extrinsic),
    )
    grasps = topdown_grasp_candidates(scene, z_offset=float(args.grasp_z_offset))
    config = Curobo2BackendConfig(
        max_batch_size=int(args.max_batch_size),
        max_goalset=int(args.max_batch_size),
        num_ik_seeds=int(args.num_ik_seeds),
        num_trajopt_seeds=int(args.num_trajopt_seeds),
        use_cuda_graph=bool(args.use_cuda_graph),
        attachment_num_spheres=int(args.attachment_num_spheres),
    )
    executor = RecordingTrajectoryExecutor(output_dir / "execution.json")
    with Curobo2Backend(config) as planner:
        curobo = planner._ensure_planner()
        initial = curobo.default_joint_state.position.detach().cpu().numpy().reshape(-1)[:7]
        if args.smoke_place:
            position = grasps[1].pose.position.copy()
            position[2] += float(args.lift_height)
            places = (
                PoseCandidate(
                    "lift_hold",
                    Pose(position, grasps[1].pose.quaternion_wxyz),
                    score=1.0,
                ),
            )
        elif args.place_xyz is not None:
            places = (
                PoseCandidate(
                    "explicit_place",
                    Pose(args.place_xyz, grasps[1].pose.quaternion_wxyz),
                    score=1.0,
                ),
            )
        else:
            source = grasps[1].pose.position.copy()
            source[2] += float(args.place_surface_offset)
            slot_offsets = ((0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.0, -0.02))
            places = tuple(
                PoseCandidate(
                    f"table_slot_{index:02d}",
                    Pose(
                        [source[0] + dx, source[1] + dy, source[2]],
                        grasps[1].pose.quaternion_wxyz,
                    ),
                    score=1.0 - 0.05 * index,
                    metadata={
                        "source": "nearby_table_slot",
                        "offset_xy": [dx, dy],
                        "yaw_offset_deg": grasps[1].metadata["yaw_offset_deg"],
                    },
                )
                for index, (dx, dy) in enumerate(slot_offsets)
            )
        task = PickPlaceTask(
            object_name=scene.object_name,
            current=JointConfiguration(tuple(curobo.joint_names), initial),
            grasp_candidates=grasps,
            place_candidates=places,
            scene=scene.scene,
            grasp_approach_offset=float(args.approach_offset),
            lift_height=float(args.lift_height),
            place_clearance=float(args.place_clearance),
            max_attempts=int(args.max_attempts),
        )
        result = PickPlaceCoordinator(planner, executor).run(task)

    summary = _summary(result, scene, output_dir)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"result: {result_path.resolve()}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
