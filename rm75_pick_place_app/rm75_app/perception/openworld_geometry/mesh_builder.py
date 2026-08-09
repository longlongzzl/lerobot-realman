from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .models import DynamicGeometryConfig, RegistrationResult, as_transform


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _safe_mesh(points: np.ndarray, minimum_extent: float) -> trimesh.Trimesh:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(values) < 4:
        center = np.mean(values, axis=0) if len(values) else np.zeros(3)
        mesh = trimesh.creation.box(extents=np.full(3, minimum_extent))
        mesh.apply_translation(center)
        return mesh
    try:
        mesh = trimesh.points.PointCloud(values).convex_hull
        if mesh.is_empty or not np.all(np.isfinite(mesh.vertices)):
            raise ValueError("invalid convex hull")
        return mesh
    except (ValueError, RuntimeError):
        lower = np.min(values, axis=0)
        upper = np.max(values, axis=0)
        extents = np.maximum(upper - lower, minimum_extent)
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation((lower + upper) * 0.5)
        return mesh


class GeometryMeshBuilder:
    """Builds a detailed visual mesh and a conservative planning mesh."""

    def __init__(self, config: DynamicGeometryConfig) -> None:
        self.config = config

    def _visual_mesh(self, points: np.ndarray, colors: np.ndarray | None) -> trimesh.Trimesh:
        if len(points) < 100:
            return _safe_mesh(points, self.config.voxel_size_m * 2.0)
        try:
            import open3d as o3d

            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            if colors is not None and len(colors) == len(points):
                cloud.colors = o3d.utility.Vector3dVector(np.clip(colors / 255.0, 0.0, 1.0))
            radius = self.config.voxel_size_m * 4.0
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40))
            cloud.orient_normals_consistent_tangent_plane(min(30, max(3, len(points) // 20)))
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                cloud, depth=self.config.poisson_depth
            )
            density = np.asarray(densities)
            if density.size:
                mesh.remove_vertices_by_mask(density < np.quantile(density, 0.02))
            mesh = mesh.crop(cloud.get_axis_aligned_bounding_box())
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            vertex_colors = np.asarray(mesh.vertex_colors)
            if len(vertices) and len(faces):
                visual = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                if len(vertex_colors) == len(vertices):
                    visual.visual.vertex_colors = np.clip(vertex_colors * 255.0, 0, 255).astype(np.uint8)
                return visual
        except (ImportError, RuntimeError, ValueError):
            pass
        return _safe_mesh(points, self.config.voxel_size_m * 2.0)

    def _collision_mesh(self, points: np.ndarray) -> trimesh.Trimesh:
        collision = _safe_mesh(points, self.config.voxel_size_m * 2.0)
        center = np.mean(collision.vertices, axis=0)
        collision.vertices = center + (collision.vertices - center) * self.config.collision_inflation
        if len(collision.faces) > self.config.collision_max_faces:
            try:
                collision = collision.simplify_quadric_decimation(face_count=self.config.collision_max_faces)
            except (ImportError, RuntimeError, ValueError):
                pass
        collision.remove_unreferenced_vertices()
        return collision

    def build(
        self,
        *,
        instance_dir: Path,
        version: int,
        instance_id: str,
        T_base_model: np.ndarray,
        observed_points: np.ndarray,
        observed_colors: np.ndarray | None,
        completion_points: np.ndarray,
        combined_points: np.ndarray,
        combined_colors: np.ndarray | None,
        registration: RegistrationResult | None,
        completion_source: str,
    ) -> dict[str, Any]:
        version_dir = instance_dir / f"v{version:04d}"
        version_dir.mkdir(parents=True, exist_ok=False)
        visual_path = version_dir / "visual_mesh.ply"
        collision_path = version_dir / "collision_mesh.obj"
        observed_path = version_dir / "observed_points.ply"
        combined_path = version_dir / "combined_points.ply"

        visual = self._visual_mesh(combined_points, combined_colors)
        collision = self._collision_mesh(combined_points)
        visual.export(visual_path, file_type="ply")
        collision.export(collision_path, file_type="obj")
        trimesh.points.PointCloud(
            observed_points,
            colors=None if observed_colors is None else np.clip(observed_colors, 0, 255).astype(np.uint8),
        ).export(observed_path, file_type="ply")
        trimesh.points.PointCloud(
            combined_points,
            colors=None if combined_colors is None else np.clip(combined_colors, 0, 255).astype(np.uint8),
        ).export(combined_path, file_type="ply")

        observed_count = int(len(observed_points))
        completion_count = int(len(completion_points))
        evidence_ratio = observed_count / max(observed_count + completion_count, 1)
        registration_quality = 1.0 if registration is None else float(np.clip(registration.fitness, 0.0, 1.0))
        geometry_score = float(np.clip(0.25 + 0.55 * evidence_ratio + 0.20 * registration_quality, 0.0, 1.0))
        manifest_path = version_dir / "geometry_manifest.json"
        manifest = {
            "schema_version": 1,
            "instance_id": instance_id,
            "geometry_version": int(version),
            "T_base_model": as_transform(T_base_model).tolist(),
            "completion_source": completion_source,
            "observed_point_count": observed_count,
            "completion_point_count": completion_count,
            "geometry_score": geometry_score,
            "visual_mesh_path": str(visual_path.resolve()),
            "collision_mesh_path": str(collision_path.resolve()),
            "observed_points_path": str(observed_path.resolve()),
            "combined_points_path": str(combined_path.resolve()),
            "registration": None
            if registration is None
            else {
                "accepted": registration.accepted,
                "fitness": registration.fitness,
                "inlier_rmse_m": registration.inlier_rmse_m,
                "reason": registration.reason,
            },
        }
        _atomic_json(manifest_path, manifest)
        _atomic_json(instance_dir / "latest.json", manifest)
        return {
            "visual_mesh_path": visual_path.resolve(),
            "collision_mesh_path": collision_path.resolve(),
            "observed_points_path": observed_path.resolve(),
            "combined_points_path": combined_path.resolve(),
            "manifest_path": manifest_path.resolve(),
            "observed_point_count": observed_count,
            "completion_point_count": completion_count,
            "geometry_score": geometry_score,
        }
