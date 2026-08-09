from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .models import DynamicGeometryConfig, RegistrationResult, as_transform
from .pointcloud import transform_points


def rotation_angle_deg(rotation: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(np.asarray(rotation, dtype=np.float64).reshape(3, 3)).magnitude()))


def viewpoint_angle_deg(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None:
        return 180.0
    old = as_transform(previous)
    new = as_transform(current)
    relative = old[:3, :3].T @ new[:3, :3]
    return rotation_angle_deg(relative)


class GeometryRegistrar:
    def __init__(self, config: DynamicGeometryConfig) -> None:
        self.config = config

    def refine(
        self,
        source_points_camera: np.ndarray,
        target_points_model: np.ndarray,
        initial_T_model_camera: np.ndarray,
        *,
        trust_initial: bool = False,
    ) -> RegistrationResult:
        initial = as_transform(initial_T_model_camera)
        source = np.asarray(source_points_camera, dtype=np.float64).reshape(-1, 3)
        target = np.asarray(target_points_model, dtype=np.float64).reshape(-1, 3)
        if len(source) < self.config.min_depth_points or len(target) < self.config.min_depth_points:
            return RegistrationResult(False, initial, 0.0, np.inf, 0.0, 0.0, "insufficient_points")
        if trust_initial:
            return RegistrationResult(True, initial, 1.0, 0.0, 0.0, 0.0, "trusted_transform")
        try:
            import open3d as o3d
        except ImportError as exc:
            return RegistrationResult(False, initial, 0.0, np.inf, 0.0, 0.0, f"open3d_missing:{exc}")

        transformed = transform_points(source, initial)
        source_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(transformed))
        target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
        voxel = self.config.registration_voxel_m
        source_cloud = source_cloud.voxel_down_sample(voxel)
        target_cloud = target_cloud.voxel_down_sample(voxel)
        if len(source_cloud.points) < 20 or len(target_cloud.points) < 20:
            return RegistrationResult(False, initial, 0.0, np.inf, 0.0, 0.0, "insufficient_downsampled_points")
        source_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4.0, max_nn=40))
        target_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4.0, max_nn=40))
        result = o3d.pipelines.registration.registration_icp(
            source_cloud,
            target_cloud,
            self.config.registration_max_correspondence_m,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
        )
        delta = np.asarray(result.transformation, dtype=np.float64)
        refined = delta @ initial
        translation_delta = float(np.linalg.norm(delta[:3, 3]))
        rotation_delta = rotation_angle_deg(delta[:3, :3])
        accepted = (
            float(result.fitness) >= self.config.registration_min_fitness
            and float(result.inlier_rmse) <= self.config.registration_max_rmse_m
            and translation_delta <= self.config.registration_max_translation_delta_m
            and rotation_delta <= self.config.registration_max_rotation_delta_deg
        )
        reason = "accepted" if accepted else "registration_gate_failed"
        return RegistrationResult(
            accepted,
            refined if accepted else initial,
            float(result.fitness),
            float(result.inlier_rmse),
            translation_delta,
            rotation_delta,
            reason,
            metadata={
                "source_downsampled": len(source_cloud.points),
                "target_downsampled": len(target_cloud.points),
            },
        )
