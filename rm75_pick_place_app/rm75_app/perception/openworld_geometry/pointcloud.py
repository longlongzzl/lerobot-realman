from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .models import DynamicGeometryConfig, GeometryFrame, as_transform


def masked_depth_to_points(frame: GeometryFrame) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(frame.depth_m, dtype=np.float64)
    mask = np.asarray(frame.mask, dtype=bool)
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    ys, xs = np.where(valid)
    if xs.size == 0:
        return np.empty((0, 3), np.float64), np.empty((0, 3), np.uint8)
    z = depth[ys, xs]
    K = np.asarray(frame.K, dtype=np.float64)
    x = (xs.astype(np.float64) - K[0, 2]) * z / K[0, 0]
    y = (ys.astype(np.float64) - K[1, 2]) * z / K[1, 1]
    return np.column_stack((x, y, z)), np.asarray(frame.rgb, dtype=np.uint8)[ys, xs]


def transform_points(points: np.ndarray, T_target_source: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    transform = as_transform(T_target_source)
    return values @ transform[:3, :3].T + transform[:3, 3]


def voxel_reduce(
    points: np.ndarray,
    colors: np.ndarray | None,
    weights: np.ndarray | float,
    voxel_size_m: float,
    *,
    max_voxels: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return points, None if colors is None else np.empty((0, 3), np.float64), np.empty((0,), np.float64)
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    color_values = None if colors is None else np.asarray(colors, dtype=np.float64).reshape(-1, 3)[finite]
    weight_values = np.broadcast_to(np.asarray(weights, dtype=np.float64), (finite.shape[0],))[finite]
    keys = np.floor(points / float(voxel_size_m)).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    total_weight = np.bincount(inverse, weights=weight_values)
    reduced = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis] * weight_values) for axis in range(3)]
    ) / np.maximum(total_weight[:, None], 1e-12)
    reduced_colors = None
    if color_values is not None:
        reduced_colors = np.column_stack(
            [np.bincount(inverse, weights=color_values[:, axis] * weight_values) for axis in range(3)]
        ) / np.maximum(total_weight[:, None], 1e-12)
    if max_voxels is not None and len(reduced) > int(max_voxels):
        order = np.argsort(total_weight)[-int(max_voxels) :]
        reduced = reduced[order]
        total_weight = total_weight[order]
        if reduced_colors is not None:
            reduced_colors = reduced_colors[order]
    return reduced, reduced_colors, total_weight


class ConfidenceWeightedPointMap:
    """Canonical object map where real depth dominates generated completion."""

    def __init__(self, config: DynamicGeometryConfig) -> None:
        self.config = config
        self.predicted_points = np.empty((0, 3), np.float64)
        self.predicted_colors: np.ndarray | None = None
        self.predicted_weights = np.empty((0,), np.float64)
        self.observed_points = np.empty((0, 3), np.float64)
        self.observed_colors: np.ndarray | None = None
        self.observed_weights = np.empty((0,), np.float64)

    def set_completion(
        self,
        points: np.ndarray,
        *,
        confidence: np.ndarray | None = None,
        colors: np.ndarray | None = None,
    ) -> None:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        weights = (
            np.full(len(values), self.config.completion_weight, np.float64)
            if confidence is None
            else np.clip(np.asarray(confidence, dtype=np.float64).reshape(-1), 0.01, 1.0)
            * self.config.completion_weight
        )
        self.predicted_points, self.predicted_colors, self.predicted_weights = voxel_reduce(
            values,
            colors,
            weights,
            self.config.voxel_size_m,
            max_voxels=self.config.max_completion_voxels,
        )

    def integrate_observation(self, points_model: np.ndarray, colors: np.ndarray | None = None) -> int:
        new_points, new_colors, new_weights = voxel_reduce(
            points_model,
            colors,
            self.config.observed_weight,
            self.config.voxel_size_m,
        )
        if len(new_points) == 0:
            return 0
        before = len(self.observed_points)
        merged_points = np.vstack((self.observed_points, new_points))
        merged_weights = np.concatenate((self.observed_weights, new_weights))
        if self.observed_colors is None and new_colors is None:
            merged_colors = None
        else:
            old_colors = self.observed_colors
            if old_colors is None:
                old_colors = np.zeros((len(self.observed_points), 3), np.float64)
            if new_colors is None:
                new_colors = np.zeros((len(new_points), 3), np.float64)
            merged_colors = np.vstack((old_colors, new_colors))
        self.observed_points, self.observed_colors, self.observed_weights = voxel_reduce(
            merged_points,
            merged_colors,
            merged_weights,
            self.config.voxel_size_m,
            max_voxels=self.config.max_observed_voxels,
        )
        self._remove_predictions_near_observations(new_points)
        return max(0, len(self.observed_points) - before)

    def _remove_predictions_near_observations(self, new_points: np.ndarray) -> None:
        if len(self.predicted_points) == 0 or len(new_points) == 0:
            return
        distances, _ = cKDTree(new_points).query(self.predicted_points, k=1)
        keep = distances > self.config.prediction_override_radius_m
        self.predicted_points = self.predicted_points[keep]
        self.predicted_weights = self.predicted_weights[keep]
        if self.predicted_colors is not None:
            self.predicted_colors = self.predicted_colors[keep]

    def combined(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        points = np.vstack((self.observed_points, self.predicted_points))
        weights = np.concatenate((self.observed_weights, self.predicted_weights))
        if self.observed_colors is None and self.predicted_colors is None:
            colors = None
        else:
            observed_colors = self.observed_colors
            predicted_colors = self.predicted_colors
            if observed_colors is None:
                observed_colors = np.zeros((len(self.observed_points), 3), np.float64)
            if predicted_colors is None:
                predicted_colors = np.zeros((len(self.predicted_points), 3), np.float64)
            colors = np.vstack((observed_colors, predicted_colors))
        return points, colors, weights
