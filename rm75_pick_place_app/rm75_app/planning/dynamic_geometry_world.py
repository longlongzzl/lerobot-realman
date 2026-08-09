from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rm75_app.perception.openworld_geometry.models import DynamicGeometrySnapshot
from rm75_app.planning.contracts import CollisionObject, PlanningScene, Pose


class DynamicGeometryWorld:
    """Stages versioned object meshes and applies them only at a planning boundary."""

    def __init__(
        self,
        *,
        static_cuboids: Iterable[Mapping[str, Any]] = (),
        static_meshes: Iterable[Mapping[str, Any]] = (),
    ) -> None:
        self.static_cuboids = [dict(item) for item in static_cuboids]
        self.static_meshes = [dict(item) for item in static_meshes]
        self._objects: dict[str, DynamicGeometrySnapshot] = {}

    def stage(self, snapshot: DynamicGeometrySnapshot) -> bool:
        previous = self._objects.get(snapshot.instance_id)
        self._objects[snapshot.instance_id] = snapshot
        return previous is None or previous.version != snapshot.version or not (
            previous.T_base_model == snapshot.T_base_model
        ).all()

    def remove(self, instance_id: str) -> None:
        self._objects.pop(str(instance_id), None)

    def mesh_obstacles(self) -> list[dict[str, Any]]:
        dynamic = [
            self._objects[key].curobo_mesh_obstacle(name=f"openworld_{key}")
            for key in sorted(self._objects)
        ]
        return [*self.static_meshes, *dynamic]

    def collision_objects(self) -> tuple[CollisionObject, ...]:
        """Expose dynamic meshes through the planner-independent v2 contract."""
        objects = []
        for key in sorted(self._objects):
            snapshot = self._objects[key]
            pose = snapshot.curobo_mesh_obstacle(name=f"openworld_{key}")["pose"]
            objects.append(
                CollisionObject(
                    name=f"openworld_{key}",
                    kind="mesh",
                    pose=Pose(pose[:3], pose[3:]),
                    mesh_path=snapshot.collision_mesh_path,
                    scale=[1.0, 1.0, 1.0],
                    metadata={
                        "source": "openworld_dynamic_geometry",
                        "instance_id": snapshot.instance_id,
                        "geometry_version": snapshot.version,
                        "geometry_score": snapshot.geometry_score,
                    },
                )
            )
        return tuple(objects)

    def apply_to_scene(self, base_scene: PlanningScene) -> PlanningScene:
        """Create an immutable scene revision at a replanning boundary."""
        versions = ",".join(
            f"{key}:v{self._objects[key].version}" for key in sorted(self._objects)
        )
        revision = f"{base_scene.revision or 'scene'}|{versions or 'no-dynamic-geometry'}"
        return PlanningScene((*base_scene.objects, *self.collision_objects()), revision=revision)

    def apply_before_replan(self, planner: Any) -> None:
        """Swap the staged world in one planner update; never call during execution."""
        planner.set_world_from_obstacles(cuboids=self.static_cuboids, meshes=self.mesh_obstacles())
