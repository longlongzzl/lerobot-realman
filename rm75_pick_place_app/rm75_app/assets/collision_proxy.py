"""Automatic planner collision proxies derived from registered visual assets."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import trimesh

from rm75_app.assets.object_specs import ObjectSpec, resolve_object_spec_scales
from rm75_app.planning.contracts import CollisionObject, Pose


DEFAULT_SCENE_PROXY_SCALE = 0.9


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.asarray([
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        ])
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.asarray([
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            ])
        elif axis == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = np.asarray([
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            ])
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = np.asarray([
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            ])
    return quat / np.linalg.norm(quat)


@lru_cache(maxsize=128)
def _asset_geometry(mesh_path: str, mesh_scale: float) -> tuple[np.ndarray, np.ndarray, bool]:
    loaded = trimesh.load(Path(mesh_path), force="scene", process=False)
    bounds = np.asarray(loaded.bounds, dtype=np.float64)
    raw_extents = bounds[1] - bounds[0]
    if np.any(raw_extents <= 1e-9):
        raise ValueError(f"asset has degenerate collision bounds: {mesh_path}")
    local_center = 0.5 * (bounds[0] + bounds[1]) * float(mesh_scale)
    extents = raw_extents * float(mesh_scale)

    # A near-isotropic sphere fills about pi/6 of its AABB. Cubes fill almost
    # the whole AABB, so this separates balls from blocks without asset names.
    hull_fill = float(loaded.convex_hull.volume / np.prod(raw_extents))
    anisotropy = float(np.max(raw_extents) / np.min(raw_extents))
    sphere_like = anisotropy <= 1.08 and 0.35 <= hull_fill <= 0.72
    return local_center, extents, sphere_like


def build_automatic_collision_proxy(
    object_name: str,
    spec: ObjectSpec,
    T_world_object: np.ndarray,
    *,
    global_scale: float = DEFAULT_SCENE_PROXY_SCALE,
    metadata: Mapping[str, Any] | None = None,
) -> CollisionObject:
    """Build a Sphere or oriented Cuboid while retaining attachment mesh metadata."""
    transform = np.asarray(T_world_object, dtype=np.float64).reshape(4, 4)
    mesh_scale, _ = resolve_object_spec_scales(spec)
    mesh_path = str(Path(spec.mesh_file).expanduser().resolve())
    local_center, extents, sphere_like = _asset_geometry(mesh_path, float(mesh_scale))
    object_scale = float(spec.scene_obstacle_box_scale or 1.0)
    proxy_scale = float(global_scale) * object_scale
    if not np.isfinite(proxy_scale) or proxy_scale <= 0.0:
        raise ValueError("collision proxy scale must be positive")

    proxy_transform = transform.copy()
    proxy_transform[:3, 3] = transform[:3, :3] @ local_center + transform[:3, 3]
    proxy_pose = Pose(
        proxy_transform[:3, 3],
        _matrix_to_quaternion_wxyz(proxy_transform[:3, :3]),
    )
    proxy_metadata = {
        "asset_name": spec.name,
        "collision_proxy": "sphere" if sphere_like else "cuboid",
        "collision_proxy_scale": proxy_scale,
        "visual_mesh_path": mesh_path,
        "visual_mesh_scale": [float(mesh_scale)] * 3,
        "object_world_pose": transform.tolist(),
        "proxy_local_center": local_center.tolist(),
        # Optional escape hatch only; None selects cuRobo2 automatic count.
        "attachment_num_spheres": spec.attachment_num_spheres,
        **dict(metadata or {}),
    }
    if sphere_like:
        return CollisionObject(
            object_name,
            "sphere",
            proxy_pose,
            radius=0.5 * float(np.max(extents)) * proxy_scale,
            metadata=proxy_metadata,
        )
    return CollisionObject(
        object_name,
        "cuboid",
        proxy_pose,
        dimensions=(extents * proxy_scale).tolist(),
        metadata=proxy_metadata,
    )
