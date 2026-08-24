"""Interactive ManiSkill preview for a task scene and its target poses."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rm75_app.assets.object_specs import get_object_spec, resolve_object_spec_scales
from rm75_app.core.frames import robot_base_world_pose
from rm75_app.orchestration.multi_object_executor import load_task_scene
from rm75_app.pickplace.atom_task_builder import FixedSceneAtomTaskBuilder, _pose_to_matrix
from rm75_app.tasks.manipulation_plan import load_plan
from rm75_app.validation.three_gate import DEFAULT_CUROBO2_PYTHON
from rm75_app.validation.maniskill_gate import write_scene_spec


def _collision_proxy_payload(task_scene) -> list[dict]:
    planning_scene = FixedSceneAtomTaskBuilder()._planning_scene(task_scene)
    output = []
    for item in planning_scene.objects:
        pose = _pose_to_matrix(item.pose)
        kind = item.kind
        dimensions = None
        radius = item.radius
        # This mirrors Curobo2Backend._to_curobo_obstacle: analytic spheres
        # use an orientation-invariant conservative OBB in the actual planner.
        if kind == "sphere":
            kind = "cuboid"
            diameter = 2.0 * float(item.radius)
            dimensions = [diameter, diameter, diameter]
            pose[:3, :3] = np.eye(3)
            radius = None
        elif item.dimensions is not None:
            dimensions = np.asarray(item.dimensions, dtype=float).tolist()
        output.append(
            {
                "object_id": item.name,
                "kind": kind,
                "source_kind": item.kind,
                "pose": pose.tolist(),
                "dimensions": dimensions,
                "radius": radius,
                "proxy_scale": item.metadata.get("collision_proxy_scale"),
            }
        )
    return output


def write_preview_scene_spec(
    plan_file: str | Path,
    output_path: str | Path,
    *,
    robot_geometry_file: str | Path | None = None,
    candidate_id: str | None = None,
) -> Path:
    plan = load_plan(plan_file)
    task_scene = load_task_scene(plan.scene_file)
    scene_payload = json.loads(
        Path(plan.scene_file).expanduser().resolve().read_text(encoding="utf-8")
    )
    T_world_robot_base, robot_base_pose_source = robot_base_world_pose(scene_payload)
    path, _ = write_scene_spec(plan, output_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_asset = {str(item["object_id"]): item for item in payload.get("objects") or []}
    for item in payload.get("objects") or []:
        item["fixed"] = True
    targets = []
    for index, atom in enumerate(plan.atoms, start=1):
        source = by_asset.get(atom.object_id)
        if source is None:
            spec = get_object_spec(atom.object_asset)
            if spec is None:
                raise KeyError(f"target asset {atom.object_asset!r} is not registered")
            _, scale = resolve_object_spec_scales(spec)
            asset_file = str(Path(spec.sim_asset_file or spec.mesh_file).expanduser().resolve())
            scale = [float(scale)] * 3
        else:
            asset_file = source["asset_file"]
            scale = source["scale"]
        targets.append(
            {
                "step": index,
                "atom_id": atom.atom_id,
                "object_id": atom.object_id,
                "asset_file": asset_file,
                "scale": scale,
                "pose": np.asarray(atom.target_pose, dtype=float).reshape(4, 4).tolist(),
                "semantic_operator": atom.semantic_operator,
            }
        )
    payload["targets"] = targets
    grasp_previews = []
    if plan.atoms:
        atom = plan.atoms[0]
        scene_object = task_scene.objects.get(atom.object_id)
        if scene_object is not None:
            task_builder = FixedSceneAtomTaskBuilder()
            grasp_candidates = task_builder._grasp_candidates(
                atom.object_asset, scene_object.pose
            )
            if candidate_id is None:
                grasp = grasp_candidates[0]
            else:
                try:
                    grasp = next(
                        item for item in grasp_candidates if item.candidate_id == candidate_id
                    )
                except StopIteration as exc:
                    available = ", ".join(item.candidate_id for item in grasp_candidates)
                    raise ValueError(
                        f"unknown candidate_id {candidate_id!r}; available: {available}"
                    ) from exc
            T_world_grasp = _pose_to_matrix(grasp.pose)
            # Match cuRobo2 plan_grasp with approach_in_tool_frame=False.
            # Its approach displacement is applied on base/world +Z.
            world_approach_offset = abs(
                float(task_builder.config.grasp_approach_offset_m)
            )
            T_world_pregrasp = T_world_grasp.copy()
            T_world_pregrasp[2, 3] += world_approach_offset
            world_delta = T_world_grasp[:3, 3] - T_world_pregrasp[:3, 3]
            grasp_previews.append(
                {
                    "atom_id": atom.atom_id,
                    "object_id": atom.object_id,
                    "candidate_id": grasp.candidate_id,
                    "approach_axis": "world_z",
                    "approach_offset_m": world_approach_offset,
                    "pregrasp_pose": T_world_pregrasp.tolist(),
                    "grasp_pose": T_world_grasp.tolist(),
                    "world_delta_m": world_delta.tolist(),
                    "distance_m": float(np.linalg.norm(world_delta)),
                    "tool_z_world": T_world_grasp[:3, 2].tolist(),
                }
            )
    payload["grasp_previews"] = grasp_previews
    payload["collision_proxies"] = _collision_proxy_payload(task_scene)
    payload["robot_base_world_pose"] = T_world_robot_base.tolist()
    payload["robot_base_world_pose_source"] = robot_base_pose_source
    payload["robot_collision_models"] = []
    payload["collision_geometry_error"] = None
    if robot_geometry_file is not None:
        geometry_path = Path(robot_geometry_file).expanduser().resolve()
        if geometry_path.is_file():
            geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
            sphere_coordinate_frame = str(
                geometry.get("sphere_coordinate_frame") or "robot_base"
            )
            payload["robot_collision_models"] = [
                {
                    **dict(model),
                    "coordinate_frame": sphere_coordinate_frame,
                    "pose": (
                        T_world_robot_base.tolist()
                        if sphere_coordinate_frame == "robot_base"
                        else np.eye(4, dtype=float).tolist()
                    ),
                }
                for model in geometry.get("models") or []
            ]
            payload["collision_geometry_missing_ik_roles"] = list(
                geometry.get("missing_ik_roles") or []
            )
            payload["collision_geometry_fallback_roles"] = list(
                geometry.get("kinematic_fallback_roles") or []
            )
    payload["preview_views"] = True
    payload["plan_file"] = str(Path(plan_file).expanduser().resolve())
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open an interactive ManiSkill task-scene preview")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snapshot", action="store_true", help="Save a four-view GPU render instead of opening Viewer")
    parser.add_argument("--curobo2-python", default=DEFAULT_CUROBO2_PYTHON)
    parser.add_argument("--candidate-id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, viewer=None) -> int:
    args = parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = load_plan(args.plan)
    robot_geometry_file = output / "curobo2_preview_geometry.json"
    collision_geometry_error = None
    try:
        worker_command = [
                str(args.curobo2_python),
                "-m",
                "rm75_app.runtime.curobo2_preview_geometry",
                "--plan",
                str(args.plan.expanduser().resolve()),
                "--output",
                str(robot_geometry_file),
            ]
        if args.candidate_id:
            worker_command.extend(["--candidate-id", args.candidate_id])
        completed = subprocess.run(
            worker_command,
            cwd=str(Path(__file__).parents[2]),
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0 or not robot_geometry_file.is_file():
            collision_geometry_error = (
                completed.stderr or completed.stdout or f"worker exited {completed.returncode}"
            ).strip()[-2000:]
    except Exception as exc:
        collision_geometry_error = f"{type(exc).__name__}: {exc}"
    scene_spec = write_preview_scene_spec(
        args.plan,
        output / "maniskill_preview_scene.json",
        robot_geometry_file=robot_geometry_file,
        candidate_id=args.candidate_id,
    )
    if collision_geometry_error:
        payload = json.loads(scene_spec.read_text(encoding="utf-8"))
        payload["collision_geometry_error"] = collision_geometry_error
        scene_spec.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    import gymnasium as gym
    if viewer is None and not args.snapshot:
        from sapien.utils import Viewer

        viewer = Viewer(resolutions=(960, 720))
    from rm75_app.execution.maniskill_scene import ensure_pick_env_registered
    from rm75_app.runtime.curobo2_sim_replay import DEFAULT_EXTRA_MANISKILL_ROOT

    ensure_pick_env_registered("Two_finger_PickJiaobang-v1", DEFAULT_EXTRA_MANISKILL_ROOT)
    import rm75_app.simulation.maniskill_multi_object_env  # noqa: F401

    # SAPIEN 3.0 on this workstation can create a present-capable Viewer only
    # before ManiSkill allocates its offscreen RenderSystem.  Pre-create it and
    # attach the configured ManiSkill scene after reset.
    env = None
    try:
        env = gym.make(
            "RM75-MultiObjectTask-v1",
            scene_spec_file=str(scene_spec),
            robot_uids="RM75",
            obs_mode="none",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            max_episode_steps=100000,
        )
        env.reset(seed=int(args.seed))
        if args.snapshot:
            from PIL import Image

            rendered = env.render()
            image = np.asarray(rendered)
            while image.ndim > 3:
                image = image[0]
            if image.dtype != np.uint8:
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            image_path = output / "maniskill_scene_preview.png"
            Image.fromarray(image[..., :3]).save(image_path)
            preview_payload = json.loads(scene_spec.read_text(encoding="utf-8"))
            grasp_previews = preview_payload.get("grasp_previews") or []
            result = {
                "ok": True,
                "mode": "snapshot",
                "image_path": str(image_path),
                "scene_spec": str(scene_spec),
                "object_count": len(env.unwrapped.task_actors),
                "target_count": len(plan.atoms),
                "grasp_preview": grasp_previews[0] if grasp_previews else None,
                "collision_proxy_count": len(preview_payload.get("collision_proxies") or []),
                "collision_proxy_names": [
                    item.get("object_id")
                    for item in preview_payload.get("collision_proxies") or []
                ],
                "robot_collision_models": [
                    {
                        "role": item.get("role"),
                        "sphere_count": len(item.get("spheres") or []),
                        "solution_source": item.get("solution_source"),
                    }
                    for item in preview_payload.get("robot_collision_models") or []
                ],
                "collision_geometry_error": preview_payload.get("collision_geometry_error"),
            }
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return 0
        env.unwrapped._viewer = viewer
        env.unwrapped._setup_viewer()
        print(
            f"[web_ui] ManiSkill 场景已打开：{len(env.unwrapped.task_actors)} 个物体，"
            f"{len(plan.atoms)} 个目标；半透明模型是目标，红/绿/蓝分别为目标 X/Y/Z 轴。",
            flush=True,
        )
        for index, atom in enumerate(plan.atoms, start=1):
            xyz = np.asarray(atom.target_pose, dtype=float).reshape(4, 4)[:3, 3]
            print(
                f"[web_ui] 目标 {index}: {atom.object_id} {atom.semantic_operator} "
                f"xyz=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})",
                flush=True,
            )
        while True:
            env.unwrapped.render_human()
            if bool(getattr(viewer, "closed", False)):
                break
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        pass
    finally:
        if env is not None:
            env.close()
        try:
            if viewer is not None:
                viewer.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
