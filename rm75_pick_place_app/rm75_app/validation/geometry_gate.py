"""Fast CPU-only geometry gate before GPU planning or simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from rm75_app.assets.object_specs import get_object_spec, resolve_object_spec_scales
from rm75_app.orchestration.multi_object_executor import SceneObjectState, load_task_scene
from rm75_app.tasks.manipulation_plan import ManipulationPlan
from rm75_app.validation.contracts import GateResult, GateStatus


@dataclass(frozen=True)
class GeometryGateConfig:
    robot_base_xyz: tuple[float, float, float] = (-0.615, 0.0, 0.0)
    max_reach_m: float = 1.10
    min_world_z_m: float = -0.05
    max_world_z_m: float = 1.20
    collision_penetration_m: float = 0.004
    collision_overlap_ratio: float = 0.12


class GeometryGate:
    def __init__(self, config: GeometryGateConfig | None = None):
        self.config = config or GeometryGateConfig()
        self._bounds_cache: dict[tuple[str, tuple[float, float, float]], np.ndarray] = {}

    def run(self, plan: ManipulationPlan) -> GateResult:
        scene = load_task_scene(plan.scene_file)
        checks: list[dict[str, Any]] = []
        failed = False
        for atom in plan.atoms:
            source = scene.objects[atom.object_id]
            target = np.asarray(atom.target_pose, dtype=np.float64).reshape(4, 4)
            center = target[:3, 3]
            distance = float(np.linalg.norm(center - np.asarray(self.config.robot_base_xyz)))
            reach_ok = bool(distance <= self.config.max_reach_m)
            height_ok = bool(self.config.min_world_z_m <= center[2] <= self.config.max_world_z_m)
            atom_check: dict[str, Any] = {
                "atom_id": atom.atom_id,
                "object_id": atom.object_id,
                "target_xyz_m": center.tolist(),
                "robot_distance_m": distance,
                "reach_ok": reach_ok,
                "height_ok": height_ok,
                "collisions": [],
                "warnings": [],
            }
            if not reach_ok or not height_ok:
                failed = True

            target_bounds = self._world_aabb(source, target)
            for other_id, other in scene.objects.items():
                if other_id in {atom.object_id, atom.support_object_id}:
                    continue
                other_bounds = self._world_aabb(other, other.pose)
                penetration = np.minimum(target_bounds[1], other_bounds[1]) - np.maximum(
                    target_bounds[0], other_bounds[0]
                )
                if np.any(penetration <= 0.0):
                    continue
                intersection = float(np.prod(penetration))
                target_volume = max(float(np.prod(target_bounds[1] - target_bounds[0])), 1e-12)
                other_volume = max(float(np.prod(other_bounds[1] - other_bounds[0])), 1e-12)
                ratio = intersection / min(target_volume, other_volume)
                collision = {
                    "other_object_id": other_id,
                    "penetration_xyz_m": penetration.tolist(),
                    "overlap_ratio": ratio,
                }
                atom_check["collisions"].append(collision)
                obvious = bool(
                    np.min(penetration) >= self.config.collision_penetration_m
                    and ratio >= self.config.collision_overlap_ratio
                )
                if obvious:
                    collision["severity"] = "error"
                    failed = True
                else:
                    collision["severity"] = "warning"
                    atom_check["warnings"].append(
                        f"AABB overlap with {other_id}; defer exact collision to Curobo2"
                    )
            checks.append(atom_check)
            # Update the predicted world so later atoms are checked against the
            # preceding placements rather than the initial snapshot.
            scene.commit_object_pose(atom.object_id, target)

        status = GateStatus.FAILED if failed else GateStatus.PASSED
        return GateResult(
            "geometry",
            status,
            "geometry gate rejected at least one target" if failed else "all targets passed fast geometry checks",
            tuple(checks),
            artifacts={"final_predicted_scene_revision": scene.revision},
        )

    def _world_aabb(self, state: SceneObjectState, transform: np.ndarray) -> np.ndarray:
        local = self._local_bounds(state)
        corners = trimesh.bounds.corners(local)
        homogeneous = np.concatenate([corners, np.ones((len(corners), 1))], axis=1)
        world = (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]
        return np.stack([world.min(axis=0), world.max(axis=0)])

    def _local_bounds(self, state: SceneObjectState) -> np.ndarray:
        path, scale = self._asset_geometry(state)
        key = (str(path), tuple(float(item) for item in scale))
        if key not in self._bounds_cache:
            loaded = trimesh.load(path, force="scene", process=False)
            self._bounds_cache[key] = np.asarray(loaded.bounds, dtype=np.float64) * np.asarray(scale)
        return self._bounds_cache[key]

    @staticmethod
    def _asset_geometry(state: SceneObjectState) -> tuple[Path, np.ndarray]:
        spec = get_object_spec(state.asset_name)
        if spec is not None:
            mesh_scale, _ = resolve_object_spec_scales(spec)
            return Path(spec.mesh_file).expanduser().resolve(), np.repeat(mesh_scale, 3)
        raw = dict(state.metadata.get("source_scene_entry") or {})
        path = raw.get("collision_mesh_path") or raw.get("mesh_file") or raw.get("sim_asset_file")
        if not path:
            raise KeyError(f"object {state.object_id!r} has no geometry asset")
        scale = np.asarray(raw.get("mesh_scale") or raw.get("scale") or [1.0, 1.0, 1.0], dtype=float).reshape(-1)
        if scale.size == 1:
            scale = np.repeat(scale, 3)
        return Path(path).expanduser().resolve(), scale
