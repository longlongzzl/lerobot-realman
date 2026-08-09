from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from .mesh_builder import GeometryMeshBuilder
from .models import (
    DynamicGeometryConfig,
    DynamicGeometrySnapshot,
    DynamicGeometryUpdate,
    GeometryFrame,
    RegistrationResult,
    as_transform,
)
from .pointcloud import ConfidenceWeightedPointMap, masked_depth_to_points, transform_points
from .providers import GeometryCompletionProvider
from .registration import GeometryRegistrar, viewpoint_angle_deg


def _safe_instance_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._")
    if not cleaned:
        raise ValueError("instance_id must contain at least one safe character")
    return cleaned


class DynamicGeometrySession:
    """One rigid object's generated prior, observed map and versioned planning mesh."""

    def __init__(
        self,
        *,
        instance_id: str,
        output_root: str | Path,
        provider: GeometryCompletionProvider,
        config: DynamicGeometryConfig | None = None,
        registrar: GeometryRegistrar | None = None,
        mesh_builder: GeometryMeshBuilder | None = None,
    ) -> None:
        self.instance_id = _safe_instance_id(instance_id)
        self.instance_dir = Path(output_root).expanduser().resolve() / self.instance_id
        self.instance_dir.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.config = config or DynamicGeometryConfig()
        self.registrar = registrar or GeometryRegistrar(self.config)
        self.mesh_builder = mesh_builder or GeometryMeshBuilder(self.config)
        self.point_map = ConfidenceWeightedPointMap(self.config)
        self.T_base_model: np.ndarray | None = None
        self.version = 0
        self.accepted_frames = 0
        self.last_mesh_frame = -1
        self.last_mesh_view: np.ndarray | None = None
        self.completion_source = "unknown"
        self.latest_snapshot: DynamicGeometrySnapshot | None = None

    def _frame_points(self, frame: GeometryFrame) -> tuple[np.ndarray, np.ndarray]:
        mask_pixels = int(np.count_nonzero(frame.mask))
        if mask_pixels < self.config.min_mask_pixels:
            raise ValueError(f"mask has only {mask_pixels} pixels")
        points, colors = masked_depth_to_points(frame)
        valid = (
            np.all(np.isfinite(points), axis=1)
            & (points[:, 2] >= self.config.min_depth_m)
            & (points[:, 2] <= self.config.max_depth_m)
        )
        points, colors = points[valid], colors[valid]
        if len(points) < self.config.min_depth_points:
            raise ValueError(f"frame has only {len(points)} valid object depth points")
        if len(points) > self.config.max_points_per_frame:
            indices = np.linspace(0, len(points) - 1, self.config.max_points_per_frame, dtype=np.int64)
            points, colors = points[indices], colors[indices]
        return points, colors

    def initialize(
        self,
        frame: GeometryFrame,
        *,
        T_base_camera: np.ndarray | None = None,
        T_base_model: np.ndarray | None = None,
    ) -> DynamicGeometrySnapshot:
        if self.latest_snapshot is not None:
            raise RuntimeError("dynamic geometry session is already initialized")
        camera_pose = as_transform(np.eye(4) if T_base_camera is None else T_base_camera)
        observed_camera, observed_colors = self._frame_points(frame)
        if T_base_model is None:
            camera_model = np.eye(4)
            camera_model[:3, 3] = np.median(observed_camera, axis=0)
            self.T_base_model = camera_pose @ camera_model
        else:
            self.T_base_model = as_transform(T_base_model)
        T_model_camera = np.linalg.inv(self.T_base_model) @ camera_pose

        completion = self.provider.complete(frame, self.instance_dir / "initialization")
        completion_points = transform_points(completion.points_model, T_model_camera)
        self.point_map.set_completion(
            completion_points,
            confidence=completion.confidence,
            colors=completion.colors,
        )
        self.point_map.integrate_observation(transform_points(observed_camera, T_model_camera), observed_colors)
        self.completion_source = completion.source
        self.accepted_frames = 1
        registration = RegistrationResult(True, T_model_camera, 1.0, 0.0, 0.0, 0.0, "initial_frame")
        return self._remesh(frame.frame_index, T_model_camera, registration)

    def update(
        self,
        frame: GeometryFrame,
        *,
        T_base_camera: np.ndarray,
        predicted_T_base_model: np.ndarray | None = None,
        trust_camera_pose: bool = False,
        force_remesh: bool = False,
    ) -> DynamicGeometryUpdate:
        if self.T_base_model is None or self.latest_snapshot is None:
            raise RuntimeError("initialize must be called before update")
        try:
            source, colors = self._frame_points(frame)
        except ValueError as exc:
            model_prediction = self.T_base_model if predicted_T_base_model is None else as_transform(predicted_T_base_model)
            initial = np.linalg.inv(model_prediction) @ as_transform(T_base_camera)
            rejected = RegistrationResult(False, initial, 0.0, np.inf, 0.0, 0.0, str(exc))
            return DynamicGeometryUpdate(False, False, self.latest_snapshot, rejected, 0, 0.0, str(exc))

        camera_pose = as_transform(T_base_camera)
        model_prediction = self.T_base_model if predicted_T_base_model is None else as_transform(predicted_T_base_model)
        initial = np.linalg.inv(model_prediction) @ camera_pose
        target = self.point_map.observed_points
        if len(target) < self.config.min_depth_points:
            target = self.point_map.combined()[0]
        registration = self.registrar.refine(source, target, initial, trust_initial=trust_camera_pose)
        if not registration.accepted:
            self._append_event(frame, registration, 0, False)
            return DynamicGeometryUpdate(False, False, self.latest_snapshot, registration, 0, 0.0, registration.reason)

        self.T_base_model = camera_pose @ np.linalg.inv(registration.T_model_camera)
        new_voxels = self.point_map.integrate_observation(
            transform_points(source, registration.T_model_camera), colors
        )
        self.accepted_frames += 1
        angle = viewpoint_angle_deg(self.last_mesh_view, registration.T_model_camera)
        interval = int(frame.frame_index) - self.last_mesh_frame
        should_remesh = force_remesh or (
            interval >= self.config.remesh_min_frame_interval
            and (
                new_voxels >= self.config.remesh_min_new_voxels
                or angle >= self.config.remesh_min_view_angle_deg
            )
        )
        if should_remesh:
            snapshot = self._remesh(frame.frame_index, registration.T_model_camera, registration)
        else:
            snapshot = replace(
                self.latest_snapshot,
                T_base_model=self.T_base_model.copy(),
                registration=registration,
            )
            self.latest_snapshot = snapshot
            self._publish_state(snapshot)
        self._append_event(frame, registration, new_voxels, should_remesh)
        return DynamicGeometryUpdate(True, should_remesh, snapshot, registration, new_voxels, angle, "accepted")

    def _remesh(
        self,
        frame_index: int,
        T_model_camera: np.ndarray,
        registration: RegistrationResult,
    ) -> DynamicGeometrySnapshot:
        assert self.T_base_model is not None
        combined_points, combined_colors, _ = self.point_map.combined()
        if len(combined_points) < 4:
            raise RuntimeError("not enough geometry to build a mesh")
        self.version += 1
        built = self.mesh_builder.build(
            instance_dir=self.instance_dir,
            version=self.version,
            instance_id=self.instance_id,
            T_base_model=self.T_base_model,
            observed_points=self.point_map.observed_points,
            observed_colors=self.point_map.observed_colors,
            completion_points=self.point_map.predicted_points,
            combined_points=combined_points,
            combined_colors=combined_colors,
            registration=registration,
            completion_source=self.completion_source,
        )
        self.last_mesh_frame = int(frame_index)
        self.last_mesh_view = as_transform(T_model_camera)
        self.latest_snapshot = DynamicGeometrySnapshot(
            instance_id=self.instance_id,
            version=self.version,
            T_base_model=self.T_base_model.copy(),
            registration=registration,
            **built,
        )
        self._publish_state(self.latest_snapshot)
        return self.latest_snapshot

    def _publish_state(self, snapshot: DynamicGeometrySnapshot) -> None:
        obstacle = snapshot.curobo_mesh_obstacle()
        state = {
            "instance_id": snapshot.instance_id,
            "geometry_version": snapshot.version,
            "T_base_model": snapshot.T_base_model.tolist(),
            "collision_mesh_path": str(snapshot.collision_mesh_path),
            "geometry_score": snapshot.geometry_score,
            "curobo_obstacle": obstacle,
        }
        for name, payload in (("latest_curobo_obstacle.json", obstacle), ("latest_state.json", state)):
            path = self.instance_dir / name
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, path)

    def _append_event(
        self,
        frame: GeometryFrame,
        registration: RegistrationResult,
        new_voxels: int,
        remeshed: bool,
    ) -> None:
        event = {
            "frame_index": int(frame.frame_index),
            "timestamp_s": frame.timestamp_s,
            "accepted": registration.accepted,
            "registration_reason": registration.reason,
            "registration_fitness": registration.fitness,
            "registration_rmse_m": registration.inlier_rmse_m,
            "new_observed_voxels": int(new_voxels),
            "remeshed": bool(remeshed),
            "geometry_version": int(self.version),
        }
        with (self.instance_dir / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
