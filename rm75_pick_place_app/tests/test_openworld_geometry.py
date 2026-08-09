from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rm75_app.perception.openworld_geometry import DynamicGeometryConfig, DynamicGeometrySession, GeometryFrame
from rm75_app.perception.openworld_geometry.models import CompletionResult
from rm75_app.perception.openworld_geometry.pointcloud import (
    ConfidenceWeightedPointMap,
    masked_depth_to_points,
)
from rm75_app.planning.dynamic_geometry_world import DynamicGeometryWorld
from rm75_app.planning.contracts import PlanningScene


class CubeCompletionProvider:
    def complete(self, frame: GeometryFrame, output_dir: Path) -> CompletionResult:
        axis = np.linspace(-0.035, 0.035, 8)
        grid = np.array(np.meshgrid(axis, axis, axis)).reshape(3, -1).T
        surface = grid[np.any(np.isclose(np.abs(grid), 0.035), axis=1)]
        surface[:, 2] += 0.6
        return CompletionResult(
            points_model=surface,
            confidence=np.full(len(surface), 0.6),
            colors=np.tile([180, 120, 60], (len(surface), 1)),
            source="test_cube",
        )


def make_frame(index: int = 0, x_shift: int = 0) -> GeometryFrame:
    height, width = 48, 64
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 1] = 180
    depth = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    mask[14:35, 20 + x_shift : 43 + x_shift] = True
    ys, xs = np.where(mask)
    depth[ys, xs] = 0.59 + (xs - 31) * 0.0006 + (ys - 24) * 0.0003
    K = np.array([[90.0, 0.0, 31.5], [0.0, 90.0, 23.5], [0.0, 0.0, 1.0]])
    return GeometryFrame(rgb=rgb, depth_m=depth, mask=mask, K=K, frame_index=index)


def test_masked_depth_projection() -> None:
    frame = make_frame()
    points, colors = masked_depth_to_points(frame)
    assert len(points) == np.count_nonzero(frame.mask)
    assert points.shape == colors.shape
    assert np.all(points[:, 2] > 0.5)


def test_observation_overrides_near_completion() -> None:
    config = DynamicGeometryConfig(voxel_size_m=0.005, prediction_override_radius_m=0.012)
    point_map = ConfidenceWeightedPointMap(config)
    point_map.set_completion(np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))
    point_map.integrate_observation(np.array([[0.004, 0.0, 0.0]]))
    assert len(point_map.predicted_points) == 1
    assert np.allclose(point_map.predicted_points[0], [0.1, 0.0, 0.0])


def test_session_versions_mesh_and_emits_curobo_obstacle(tmp_path: Path) -> None:
    config = DynamicGeometryConfig(
        min_mask_pixels=20,
        min_depth_points=20,
        remesh_min_frame_interval=1,
        remesh_min_new_voxels=1,
        voxel_size_m=0.004,
    )
    session = DynamicGeometrySession(
        instance_id="unknown-cube-1",
        output_root=tmp_path,
        provider=CubeCompletionProvider(),
        config=config,
    )
    initial = session.initialize(make_frame(0), T_base_camera=np.eye(4))
    assert initial.version == 1
    assert initial.collision_mesh_path.exists()
    assert initial.visual_mesh_path.exists()

    camera_pose = np.eye(4)
    camera_pose[0, 3] = 0.015
    update = session.update(
        make_frame(1, x_shift=1),
        T_base_camera=camera_pose,
        trust_camera_pose=True,
        force_remesh=True,
    )
    assert update.accepted and update.remeshed
    assert update.snapshot is not None and update.snapshot.version == 2
    obstacle_path = tmp_path / "unknown-cube-1" / "latest_curobo_obstacle.json"
    obstacle = json.loads(obstacle_path.read_text(encoding="utf-8"))
    assert obstacle["geometry_version"] == 2
    assert Path(obstacle["file_path"]).exists()
    assert len(obstacle["pose"]) == 7


def test_rejected_frame_does_not_change_version(tmp_path: Path) -> None:
    config = DynamicGeometryConfig(min_mask_pixels=20, min_depth_points=20)
    session = DynamicGeometrySession(
        instance_id="reject-test",
        output_root=tmp_path,
        provider=CubeCompletionProvider(),
        config=config,
    )
    session.initialize(make_frame(0))
    bad = make_frame(1)
    bad = GeometryFrame(bad.rgb, np.zeros_like(bad.depth_m), bad.mask, bad.K, frame_index=1)
    result = session.update(bad, T_base_camera=np.eye(4), trust_camera_pose=True)
    assert not result.accepted
    assert result.snapshot is not None and result.snapshot.version == 1


def test_dynamic_world_applies_snapshot_at_replan_boundary(tmp_path: Path) -> None:
    config = DynamicGeometryConfig(min_mask_pixels=20, min_depth_points=20)
    session = DynamicGeometrySession(
        instance_id="world-object",
        output_root=tmp_path,
        provider=CubeCompletionProvider(),
        config=config,
    )
    snapshot = session.initialize(make_frame(0))

    class Planner:
        applied = None

        def set_world_from_obstacles(self, *, cuboids, meshes):
            self.applied = (cuboids, meshes)

    planner = Planner()
    world = DynamicGeometryWorld(static_cuboids=[{"name": "table"}])
    assert world.stage(snapshot)
    world.apply_before_replan(planner)
    assert planner.applied is not None
    assert planner.applied[0][0]["name"] == "table"
    assert planner.applied[1][0]["geometry_version"] == 1

    scene = world.apply_to_scene(PlanningScene(revision="cached-table"))
    assert scene.revision == "cached-table|world-object:v1"
    assert scene.objects[0].name == "openworld_world-object"
    assert scene.objects[0].metadata["geometry_version"] == 1
    assert scene.objects[0].mesh_path == snapshot.collision_mesh_path
