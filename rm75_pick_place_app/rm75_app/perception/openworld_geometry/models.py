from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def as_transform(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("transform contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("transform must be homogeneous")
    return matrix.copy()


@dataclass(frozen=True)
class GeometryFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    mask: np.ndarray
    K: np.ndarray
    frame_index: int = 0
    timestamp_s: float | None = None
    source: str = "camera"

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth_m)
        mask = np.asarray(self.mask)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must have shape HxWx3")
        if depth.shape != rgb.shape[:2] or mask.shape != rgb.shape[:2]:
            raise ValueError("depth and mask must match RGB dimensions")
        if np.asarray(self.K).shape != (3, 3):
            raise ValueError("K must have shape 3x3")


@dataclass(frozen=True)
class CompletionResult:
    points_model: np.ndarray
    confidence: np.ndarray | None = None
    colors: np.ndarray | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = np.asarray(self.points_model)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("completion points must have shape Nx3")
        if self.confidence is not None and np.asarray(self.confidence).reshape(-1).shape[0] != points.shape[0]:
            raise ValueError("completion confidence must have one value per point")
        if self.colors is not None and np.asarray(self.colors).shape != points.shape:
            raise ValueError("completion colors must have shape Nx3")


@dataclass(frozen=True)
class RegistrationResult:
    accepted: bool
    T_model_camera: np.ndarray
    fitness: float
    inlier_rmse_m: float
    translation_delta_m: float
    rotation_delta_deg: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DynamicGeometryConfig:
    voxel_size_m: float = 0.003
    observed_weight: float = 10.0
    completion_weight: float = 0.75
    prediction_override_radius_m: float = 0.006
    min_mask_pixels: int = 64
    min_depth_points: int = 80
    min_depth_m: float = 0.05
    max_depth_m: float = 2.0
    max_points_per_frame: int = 30000
    max_observed_voxels: int = 160000
    max_completion_voxels: int = 160000
    registration_voxel_m: float = 0.006
    registration_max_correspondence_m: float = 0.018
    registration_min_fitness: float = 0.28
    registration_max_rmse_m: float = 0.012
    registration_max_translation_delta_m: float = 0.08
    registration_max_rotation_delta_deg: float = 35.0
    remesh_min_new_voxels: int = 120
    remesh_min_view_angle_deg: float = 15.0
    remesh_min_frame_interval: int = 3
    poisson_depth: int = 7
    collision_inflation: float = 1.025
    collision_max_faces: int = 4000

    def __post_init__(self) -> None:
        positive = (
            "voxel_size_m",
            "observed_weight",
            "completion_weight",
            "prediction_override_radius_m",
            "min_mask_pixels",
            "min_depth_points",
            "max_points_per_frame",
            "max_observed_voxels",
            "max_completion_voxels",
            "registration_voxel_m",
            "registration_max_correspondence_m",
            "remesh_min_frame_interval",
            "collision_inflation",
        )
        for name in positive:
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class DynamicGeometrySnapshot:
    instance_id: str
    version: int
    T_base_model: np.ndarray
    visual_mesh_path: Path
    collision_mesh_path: Path
    observed_points_path: Path
    combined_points_path: Path
    manifest_path: Path
    observed_point_count: int
    completion_point_count: int
    geometry_score: float
    registration: RegistrationResult | None = None

    def curobo_mesh_obstacle(self, *, name: str | None = None) -> dict[str, Any]:
        pose = transform_to_pose_wxyz(self.T_base_model)
        return {
            "name": str(name or f"openworld_{self.instance_id}_v{self.version}"),
            "file_path": str(self.collision_mesh_path),
            "pose": pose.tolist(),
            "scale": [1.0, 1.0, 1.0],
            "geometry_version": int(self.version),
        }


@dataclass(frozen=True)
class DynamicGeometryUpdate:
    accepted: bool
    remeshed: bool
    snapshot: DynamicGeometrySnapshot | None
    registration: RegistrationResult
    new_observed_voxels: int
    view_angle_deg: float
    reason: str


def transform_to_pose_wxyz(T: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    transform = as_transform(T)
    q_xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return np.asarray([*transform[:3, 3], q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
