"""Shared coordinate-frame conventions for the RM75 tabletop workspace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


DEFAULT_ROBOT_BASE_WORLD_XYZ_M = (-0.615, 0.0, 0.0)


def robot_base_world_pose(
    scene_payload: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, str]:
    """Resolve ``T_world_robot_base`` from a scene, with the RM75 fallback."""

    payload = dict(scene_payload or {})
    for key in ("T_world_robot_base", "robot_base_world_transform"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            transform = np.asarray(value, dtype=np.float64).reshape(4, 4)
            if np.all(np.isfinite(transform)):
                return transform.copy(), key
        except (TypeError, ValueError):
            pass

    for key in (
        "robot_base_world_xyz_m",
        "robot_base_world_p",
        "robot_base_world_position",
        "robot_base_position",
    ):
        value = payload.get(key)
        if value is None:
            continue
        try:
            xyz = np.asarray(value, dtype=np.float64).reshape(-1)
            if xyz.size >= 3 and np.all(np.isfinite(xyz[:3])):
                transform = np.eye(4, dtype=np.float64)
                transform[:3, 3] = xyz[:3]
                return transform, key
        except (TypeError, ValueError):
            pass

    pose = payload.get("robot_base_world_pose")
    if isinstance(pose, Mapping):
        for key in ("position", "p", "xyz"):
            try:
                xyz = np.asarray(pose.get(key), dtype=np.float64).reshape(-1)
                if xyz.size >= 3 and np.all(np.isfinite(xyz[:3])):
                    transform = np.eye(4, dtype=np.float64)
                    transform[:3, 3] = xyz[:3]
                    return transform, f"robot_base_world_pose.{key}"
            except (TypeError, ValueError):
                pass

    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = DEFAULT_ROBOT_BASE_WORLD_XYZ_M
    return transform, "rm75_default"
