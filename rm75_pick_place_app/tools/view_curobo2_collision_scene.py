#!/usr/bin/env python3
"""Interactively display the exact cuRobo2 robot spheres and scene meshes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rm75_app.orchestration.multi_object_executor import load_task_scene
from rm75_app.pickplace.atom_task_builder import FixedSceneAtomTaskBuilder
from rm75_app.planning.backends.curobo2 import Curobo2Backend, Curobo2BackendConfig
from rm75_app.planning.contracts import BatchPlanningRequest
from rm75_app.tasks.manipulation_plan import load_plan


COLORS = {
    "tennis": "#d7e629",
    "bi": "#d627a8",
    "gluestick": "#ff7f0e",
    "lvmukuai": "#18a558",
    "shuazi": "#9467bd",
    "bitong": "#e0b83f",
    "carriot": "#f26b21",
    "desk": "#8c6d4f",
    "hongshupian": "#c73e3a",
}


def _mesh_from_file(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()
    meshes = []
    for node_name in loaded.graph.nodes_geometry:
        transform, geometry_name = loaded.graph[node_name]
        geometry = loaded.geometry[geometry_name].copy()
        geometry.apply_transform(transform)
        meshes.append(geometry)
    if not meshes:
        raise RuntimeError(f"mesh contains no geometry: {path}")
    return trimesh.util.concatenate(meshes)


def _solve_grasp_endpoint(plan_path: Path):
    plan = load_plan(plan_path)
    scene = load_task_scene(plan.scene_file)
    backend = Curobo2Backend(
        Curobo2BackendConfig(
            max_batch_size=4,
            max_goalset=4,
            num_ik_seeds=64,
            num_trajopt_seeds=4,
            self_collision_check=True,
        )
    )
    backend.update_scene(FixedSceneAtomTaskBuilder()._planning_scene(scene))
    planner = backend._ensure_planner()
    default = planner.default_joint_state.position.reshape(-1)[:7].detach().cpu().numpy()
    scene.set_joints(tuple(planner.joint_names), default)
    task = FixedSceneAtomTaskBuilder()(plan.atoms[0], scene)
    request = BatchPlanningRequest(
        current=task.current,
        candidates=task.grasp_candidates,
        scene=task.scene,
        tool_frame=task.tool_frame,
        max_attempts=task.max_attempts,
    )
    current, goals = backend._make_batch_inputs(request)
    names = [item.name for item in task.scene.objects]
    for name in names:
        backend._set_obstacle_enabled(name, False)
    planner.ik_solver.reset_seed()
    result = planner.ik_solver.solve_pose(
        goals,
        current_state=current,
        return_seeds=planner.trajopt_solver.config.num_seeds,
    )
    success = result.success.reshape(planner.batch_size, -1)
    candidate_idx = next(
        (idx for idx in range(len(task.grasp_candidates)) if bool(success[idx].any().item())),
        None,
    )
    if candidate_idx is None:
        backend.close()
        raise RuntimeError("no collision-free IK endpoint was found for visualization")
    seed_idx = int(
        success[candidate_idx]
        .to(dtype=backend._import_modules()["torch"].int32)
        .argmax().item()
    )
    q = result.solution[candidate_idx, seed_idx].reshape(1, -1)
    state = planner.kinematics.compute_kinematics(
        backend._import_modules()["JointState"].from_position(
            q, joint_names=list(planner.joint_names)
        )
    )
    spheres = state.robot_spheres.reshape(-1, 4).detach().cpu().numpy()
    sphere_to_link = (
        planner.kinematics.config.kinematics_config.link_sphere_idx_map
        .detach().cpu().numpy().reshape(-1)
    )
    idx_to_name = {
        int(value): key
        for key, value in planner.kinematics.config.kinematics_config.link_name_to_idx_map.items()
    }
    sphere_links = [idx_to_name.get(int(index), f"link_{int(index)}") for index in sphere_to_link]
    return backend, task, spheres, sphere_links, candidate_idx


def _draw_sphere_wireframe(axis, center, radius, color):
    u = np.linspace(0.0, 2.0 * np.pi, 13)
    v = np.linspace(0.0, np.pi, 8)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    axis.plot_wireframe(x, y, z, color=color, linewidth=0.45, alpha=0.42)


def _set_equal_axes(axis, points: np.ndarray) -> None:
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = 0.5 * (low + high)
    radius = max(float(np.max(high - low)) * 0.55, 0.25)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(max(-0.08, center[2] - radius), center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    backend, task, spheres, sphere_links, candidate_idx = _solve_grasp_endpoint(
        args.plan.expanduser().resolve()
    )
    try:
        figure = plt.figure(figsize=(13, 9))
        axis = figure.add_subplot(111, projection="3d")
        all_points = []
        rng = np.random.default_rng(0)
        for item in task.scene.objects:
            if item.kind == "cuboid":
                mesh = trimesh.creation.box(extents=np.asarray(item.dimensions, dtype=float))
            elif item.kind == "sphere":
                mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(item.radius))
            else:
                mesh = _mesh_from_file(Path(item.mesh_path))
                scale = np.ones(3) if item.scale is None else np.asarray(item.scale, dtype=float)
                mesh.vertices *= scale.reshape(1, 3)
            transform = np.eye(4)
            transform[:3, 3] = item.pose.position
            w, x, y, z = item.pose.quaternion_wxyz
            transform[:3, :3] = np.asarray([
                [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
            ])
            mesh.apply_transform(transform)
            faces = mesh.faces
            if len(faces) > 1800:
                faces = faces[rng.choice(len(faces), 1800, replace=False)]
            triangles = mesh.vertices[faces]
            color = COLORS.get(item.name, "#7f8c8d")
            axis.add_collection3d(
                Poly3DCollection(
                    triangles,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.12,
                    alpha=0.23,
                )
            )
            center = mesh.bounds.mean(axis=0)
            axis.text(*center, item.name, color=color, fontsize=9, weight="bold")
            all_points.append(mesh.vertices)

        valid = spheres[:, 3] > 0.0
        spheres = spheres[valid]
        sphere_links = [name for name, keep in zip(sphere_links, valid) if keep]
        for sphere in spheres:
            _draw_sphere_wireframe(axis, sphere[:3], float(sphere[3]), "#00b7ff")
        by_link = {}
        for sphere, link in zip(spheres, sphere_links):
            by_link.setdefault(link, []).append(sphere[:3])
        for link, centers in by_link.items():
            center = np.mean(centers, axis=0)
            axis.text(*center, link, color="#005a82", fontsize=7)
        all_points.append(spheres[:, :3])

        axis.scatter([0], [0], [0], color="black", marker="x", s=65)
        axis.quiver(0, 0, 0, 0.12, 0, 0, color="r", linewidth=2)
        axis.quiver(0, 0, 0, 0, 0.12, 0, color="g", linewidth=2)
        axis.quiver(0, 0, 0, 0, 0, 0.12, color="b", linewidth=2)
        _set_equal_axes(axis, np.concatenate(all_points, axis=0))
        axis.set_xlabel("world X / m")
        axis.set_ylabel("world Y / m")
        axis.set_zlabel("world Z / m")
        axis.set_title(
            "cuRobo2 collision scene: tennis grasp endpoint\n"
            f"candidate={task.grasp_candidates[candidate_idx].candidate_id}; drag to rotate, wheel to zoom"
        )
        legend = [Patch(facecolor="#00b7ff", alpha=0.35, label="RM75 collision spheres")]
        legend.extend(
            Patch(facecolor=COLORS.get(item.name, "#7f8c8d"), alpha=0.35, label=item.name)
            for item in task.scene.objects
        )
        axis.legend(handles=legend, loc="upper left", fontsize=8)
        figure.tight_layout()
        if args.save:
            args.save.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(args.save.expanduser().resolve(), dpi=180)
        print(
            f"Showing {len(spheres)} robot collision spheres and "
            f"{len(task.scene.objects)} object collision meshes.",
            flush=True,
        )
        plt.show()
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
