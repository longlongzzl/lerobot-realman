"""Compile one resolved manipulation atom into a PickPlaceTask."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh

from rm75_app.assets.object_specs import get_object_spec, resolve_object_spec_scales
from rm75_app.assets.collision_proxy import (
    DEFAULT_SCENE_PROXY_SCALE,
    build_automatic_collision_proxy,
)
from rm75_app.orchestration.multi_object_executor import TaskSceneState
from rm75_app.pickplace.cached_scene import matrix_to_quaternion_wxyz
from rm75_app.pickplace.coordinator import PickPlaceTask
from rm75_app.planning.contracts import (
    CollisionObject,
    JointConfiguration,
    PlanningScene,
    Pose,
    PoseCandidate,
)
from rm75_app.tasks.manipulation_plan import ManipulationAtom


JointStateProvider = Callable[[TaskSceneState], JointConfiguration]


@dataclass(frozen=True)
class AtomTaskBuilderConfig:
    grasp_z_offset_m: float = 0.005
    grasp_yaw_offsets_deg: tuple[float, ...] = (0.0, -12.0, 12.0, 90.0)
    grasp_tilt_toward_base_deg: tuple[float, ...] = (0.0, 20.0, 30.0, 45.0)
    # Keep a fixed 8 x 8 relation lattice for one batch64 IK query.  Yaw is
    # physically interchangeable for a sphere but changes arm kinematics;
    # shortlist IK-feasible rows before trajectory planning to bound GPU memory.
    sphere_grasp_yaw_offsets_deg: tuple[float, ...] = (
        0.0,
        -45.0,
        45.0,
        -90.0,
        90.0,
        -135.0,
        135.0,
        180.0,
    )
    sphere_grasp_tilt_toward_base_deg: tuple[float, ...] = (
        0.0,
        15.0,
        30.0,
        40.0,
        45.0,
        60.0,
        75.0,
        90.0,
    )
    pen_grasp_axis_shifts_m: tuple[float, ...] = (
        0.0,
        0.02,
        -0.02,
        0.04,
        -0.04,
        0.06,
        -0.06,
    )
    pen_grasp_tilt_toward_base_deg: tuple[float, ...] = (0.0, 20.0)
    pen_refinement_axis_step_m: float = 0.005
    pen_refinement_tilt_step_deg: float = 5.0
    pen_refinement_yaw_step_deg: float = 6.0
    axial_place_spin_deg: tuple[float, ...] = (0.0, -90.0, 90.0, 180.0)
    grasp_approach_offset_m: float = -0.08
    lift_height_m: float = 0.08
    place_clearance_m: float = 0.08
    max_attempts: int = 3
    scene_collision_proxy_scale: float = DEFAULT_SCENE_PROXY_SCALE
    sphere_release_tilt_toward_base_deg: tuple[float, ...] = (0.0, 15.0, 30.0, 45.0)
    sphere_release_roll_deg: tuple[float, ...] = (0.0, -90.0, 90.0, 180.0)
    sphere_release_lift_m: float = 0.03
    # Multi-object/ManiSkill scenes use a world frame whose RM75 base is at
    # x=-0.615 m. Real perception inputs are already base-referenced and keep
    # this unset. cuRobo itself always plans in robot-base coordinates.
    robot_base_world_xyz_m: tuple[float, float, float] | None = None


class FixedSceneAtomTaskBuilder:
    """Build collision, grasp, and paired place candidates from task state."""

    def __init__(
        self,
        joint_state_provider: JointStateProvider | None = None,
        config: AtomTaskBuilderConfig | None = None,
    ):
        self.joint_state_provider = joint_state_provider
        self.config = config or AtomTaskBuilderConfig()

    def __call__(self, atom: ManipulationAtom, scene: TaskSceneState) -> PickPlaceTask:
        if atom.object_id not in scene.objects:
            raise KeyError(atom.object_id)
        current = self._joint_state(scene)
        planning_scene = self._planning_scene(scene)
        T_world_object = scene.objects[atom.object_id].pose
        T_planning_object = self._to_planning_pose(T_world_object)
        spec = get_object_spec(atom.object_asset)
        if spec is None:
            raise KeyError(f"object asset {atom.object_asset!r} is not registered")
        grasps = self._grasp_candidates(atom.object_asset, T_planning_object)
        place_by_grasp: dict[str, tuple[PoseCandidate, ...]] = {}
        all_places: dict[str, PoseCandidate] = {}
        T_world_goal_object = np.asarray(atom.target_pose, dtype=np.float64).reshape(4, 4)
        T_planning_goal_object = self._to_planning_pose(T_world_goal_object)
        shared_sphere_places: tuple[PoseCandidate, ...] | None = None
        if spec.orientation_symmetry == "spherical":
            shared_sphere_places = self._sphere_release_candidates(
                grasps[0],
                T_planning_object,
                T_planning_goal_object,
                T_world_goal_object=T_world_goal_object,
                loaded_asset=atom.object_asset,
                atom_id=atom.atom_id,
            )
        for grasp in grasps:
            T_world_tcp = _pose_to_matrix(grasp.pose)
            T_tcp_object = np.linalg.inv(T_world_tcp) @ T_planning_object
            if shared_sphere_places is not None:
                places = shared_sphere_places
            elif spec.orientation_symmetry in {"axial", "axial_bidirectional"}:
                places = self._axial_release_candidates(
                    grasp,
                    T_tcp_object,
                    T_planning_goal_object,
                    T_world_goal_object,
                    symmetry_axis_local=spec.symmetry_axis_local,
                    bidirectional=spec.orientation_symmetry == "axial_bidirectional",
                    atom_id=atom.atom_id,
                )
            else:
                T_world_goal_tcp = T_planning_goal_object @ np.linalg.inv(T_tcp_object)
                places = (
                    PoseCandidate(
                        f"place_for_{grasp.candidate_id}",
                        _matrix_to_pose(T_world_goal_tcp),
                        score=grasp.score,
                        metadata={
                            "paired_grasp_id": grasp.candidate_id,
                            "T_tcp_object": T_tcp_object.tolist(),
                            "target_object_pose": T_world_goal_object.tolist(),
                            "planning_target_object_pose": T_planning_goal_object.tolist(),
                            "atom_id": atom.atom_id,
                        },
                    ),
                )
            place_by_grasp[grasp.candidate_id] = places
            for place in places:
                all_places[place.candidate_id] = place
        return PickPlaceTask(
            object_name=atom.object_id,
            current=current,
            grasp_candidates=grasps,
            place_candidates=tuple(all_places.values()),
            place_candidates_by_grasp=place_by_grasp,
            scene=planning_scene,
            grasp_approach_offset=float(self.config.grasp_approach_offset_m),
            lift_height=float(self.config.lift_height_m),
            place_clearance=float(self.config.place_clearance_m),
            max_attempts=int(self.config.max_attempts),
            place_contact_object_name=atom.support_object_id,
        )

    def _joint_state(self, scene: TaskSceneState) -> JointConfiguration:
        if self.joint_state_provider is not None:
            return self.joint_state_provider(scene)
        if not scene.joint_names or scene.joint_positions is None:
            raise ValueError("task scene has no robot joint state and no joint_state_provider was configured")
        return JointConfiguration(scene.joint_names, scene.joint_positions)

    def _planning_scene(self, scene: TaskSceneState) -> PlanningScene:
        objects: list[CollisionObject] = []
        for state in scene.objects.values():
            spec = get_object_spec(state.asset_name)
            if spec is None:
                raise KeyError(f"scene asset {state.asset_name!r} is not registered")
            proxy = build_automatic_collision_proxy(
                state.object_id,
                spec,
                self._to_planning_pose(state.pose),
                global_scale=float(self.config.scene_collision_proxy_scale),
                metadata={"movable": state.movable},
            )
            objects.append(proxy)
        return PlanningScene(tuple(objects), revision=f"task-scene-{scene.revision}")

    def _to_planning_pose(self, T_world: np.ndarray) -> np.ndarray:
        """Convert a world pose to cuRobo's robot-base planning frame."""

        transform = np.asarray(T_world, dtype=np.float64).reshape(4, 4).copy()
        base_xyz = self.config.robot_base_world_xyz_m
        if base_xyz is not None:
            transform[:3, 3] -= np.asarray(base_xyz, dtype=np.float64).reshape(3)
        return transform

    def _grasp_candidates(
        self,
        asset_name: str,
        T_world_object: np.ndarray,
    ) -> tuple[PoseCandidate, ...]:
        spec = get_object_spec(asset_name)
        if spec is None:
            raise KeyError(f"object asset {asset_name!r} is not registered")
        mesh_scale, _ = resolve_object_spec_scales(spec)
        loaded = trimesh.load(Path(spec.mesh_file), force="scene", process=False)
        extents = (np.asarray(loaded.bounds[1]) - np.asarray(loaded.bounds[0])) * mesh_scale
        grasp_axis = np.zeros(3, dtype=np.float64)
        if spec.grasp_mode == "topdown_symmetric":
            # RM75 default TCP points down with its closing axis along world X.
            # Symmetric assets should not inherit yaw from an arbitrary mesh
            # principal axis or a noisy observed object rotation.
            base_closing = np.asarray([1.0, 0.0, 0.0])
            yaw_offsets = self.config.sphere_grasp_yaw_offsets_deg
            tilt_offsets = self.config.sphere_grasp_tilt_toward_base_deg
            axis_shift_offsets = (0.0,)
        else:
            local_long_axis = np.zeros(3)
            local_long_axis[int(np.argmax(extents))] = 1.0
            long_axis = np.asarray(T_world_object[:3, :3]) @ local_long_axis
            long_xy = np.asarray([long_axis[0], long_axis[1], 0.0])
            if np.linalg.norm(long_xy) < 1e-8:
                long_xy = np.asarray([1.0, 0.0, 0.0])
            long_xy /= np.linalg.norm(long_xy)
            base_closing = np.asarray([-long_xy[1], long_xy[0], 0.0])
            yaw_offsets = self.config.grasp_yaw_offsets_deg
            if spec.grasp_mode == "pen_topdown_insert_ready":
                # A centered pick can force the TCP too low at a leaned or
                # inserted target. Preserve 16 relations by varying the grasp
                # point along the pen instead of adding more orientations.
                tilt_offsets = self.config.pen_grasp_tilt_toward_base_deg
                half_length = 0.5 * float(np.max(extents))
                axis_shift_offsets = tuple(
                    float(np.clip(value, -0.70 * half_length, 0.70 * half_length))
                    for value in self.config.pen_grasp_axis_shifts_m
                )
                grasp_axis = _normalized(long_axis)
            else:
                tilt_offsets = self.config.grasp_tilt_toward_base_deg
                axis_shift_offsets = (0.0,)
        approaching = np.asarray([0.0, 0.0, -1.0])
        # FoundationPose returns the asset frame origin, which is not
        # guaranteed to be the geometric center.  Compute the grasp point
        # from the mesh's AABB after transforming it into the world frame.
        # This also makes the vertical grasp height follow world Z when an
        # elongated object is lying down or its asset frame is rotated.
        center = _world_aabb_center(loaded, mesh_scale, T_world_object)
        grasp_depth = float(spec.grasp_z_offset or self.config.grasp_z_offset_m)
        position = center - approaching * grasp_depth
        toward_base = np.asarray([-position[0], -position[1], 0.0])
        if np.linalg.norm(toward_base) < 1e-8:
            toward_base = np.asarray([1.0, 0.0, 0.0])
        toward_base /= np.linalg.norm(toward_base)
        output: list[PoseCandidate] = []
        index = 0

        def append_candidate(
            yaw_deg: float,
            tilt_deg: float,
            axis_shift_m: float,
            *,
            search_tier: int | None = None,
            refinement_parent_id: str | None = None,
        ) -> PoseCandidate:
            nonlocal index
            angle = np.deg2rad(yaw_deg)
            c, s = np.cos(angle), np.sin(angle)
            yaw_closing = np.asarray(
                [c * base_closing[0] - s * base_closing[1], s * base_closing[0] + c * base_closing[1], 0.0]
            )
            tilt_rad = np.deg2rad(abs(float(tilt_deg)))
            # gripper_tcp lies in front of the wrist along local +Z.
            # To lean the gripper body/wrist toward the base while keeping the
            # TCP on the object, local +Z points horizontally away from base.
            approach = _normalized(
                np.cos(tilt_rad) * approaching - np.sin(tilt_rad) * toward_base
            )
            orthogonal = _normalized(np.cross(yaw_closing, approach))
            closing = _normalized(np.cross(approach, orthogonal))
            T_world_tcp = np.eye(4)
            T_world_tcp[:3, :3] = np.stack([orthogonal, closing, approach], axis=1)
            candidate_position = center - approach * grasp_depth
            if abs(float(axis_shift_m)) > 1e-9:
                candidate_position += grasp_axis * float(axis_shift_m)
            T_world_tcp[:3, 3] = candidate_position
            label = f"grasp_{index:02d}_{yaw_deg:+g}deg"
            if abs(float(tilt_deg)) > 1e-6:
                label += f"_tilt_{tilt_deg:+g}deg"
            if abs(float(axis_shift_m)) > 1e-9:
                label += f"_axis_{axis_shift_m * 1000:+.0f}mm"
            if refinement_parent_id is not None:
                label += "_refined"
            candidate = PoseCandidate(
                label,
                _matrix_to_pose(T_world_tcp),
                score=1.0
                - 0.002 * abs(float(yaw_deg))
                - 0.001 * abs(float(tilt_deg))
                - 0.1 * abs(float(axis_shift_m)),
                metadata={
                    "relation_index": index,
                    "yaw_offset_deg": float(yaw_deg),
                    "tilt_toward_base_deg": float(tilt_deg),
                    "axis_shift_m": float(axis_shift_m),
                    "search_tier": (
                        int(search_tier)
                        if search_tier is not None
                        else int(
                            abs(float(tilt_deg)) > 20.0
                            or abs(float(axis_shift_m)) > 0.0201
                        )
                    ),
                    "refinement_parent_id": refinement_parent_id,
                    "source": "task_scene_known_mesh",
                },
            )
            output.append(candidate)
            index += 1
            return candidate

        coarse_candidates: list[PoseCandidate] = []
        for yaw_deg in yaw_offsets:
            for tilt_deg in tilt_offsets:
                for axis_shift_m in axis_shift_offsets:
                    coarse_candidates.append(
                        append_candidate(yaw_deg, tilt_deg, axis_shift_m)
                    )

        if spec.grasp_mode == "pen_topdown_insert_ready":
            half_length = 0.5 * float(np.max(extents))
            seen = {
                (
                    round(float(item.metadata["yaw_offset_deg"]), 6),
                    round(float(item.metadata["tilt_toward_base_deg"]), 6),
                    round(float(item.metadata["axis_shift_m"]), 6),
                )
                for item in coarse_candidates
            }
            for parent in coarse_candidates:
                parent_shift = float(parent.metadata["axis_shift_m"])
                if abs(parent_shift) < 0.035:
                    continue
                parent_yaw = float(parent.metadata["yaw_offset_deg"])
                parent_tilt = float(parent.metadata["tilt_toward_base_deg"])
                toward_center = parent_shift - np.sign(parent_shift) * float(
                    self.config.pen_refinement_axis_step_m
                )
                toward_center = float(
                    np.clip(toward_center, -0.70 * half_length, 0.70 * half_length)
                )
                variants = (
                    (parent_yaw, parent_tilt, toward_center),
                    (
                        parent_yaw,
                        max(0.0, parent_tilt - self.config.pen_refinement_tilt_step_deg),
                        toward_center,
                    ),
                    (
                        parent_yaw,
                        parent_tilt + self.config.pen_refinement_tilt_step_deg,
                        toward_center,
                    ),
                    (
                        parent_yaw - self.config.pen_refinement_yaw_step_deg,
                        parent_tilt,
                        toward_center,
                    ),
                    (
                        parent_yaw + self.config.pen_refinement_yaw_step_deg,
                        parent_tilt,
                        toward_center,
                    ),
                )
                for yaw_deg, tilt_deg, axis_shift_m in variants:
                    key = (
                        round(float(yaw_deg), 6),
                        round(float(tilt_deg), 6),
                        round(float(axis_shift_m), 6),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    append_candidate(
                        yaw_deg,
                        tilt_deg,
                        axis_shift_m,
                        search_tier=2,
                        refinement_parent_id=parent.candidate_id,
                    )
        return tuple(output)

    def _sphere_release_candidates(
        self,
        grasp: PoseCandidate,
        T_planning_object: np.ndarray,
        T_planning_goal_object: np.ndarray,
        *,
        T_world_goal_object: np.ndarray,
        loaded_asset: str,
        atom_id: str,
    ) -> tuple[PoseCandidate, ...]:
        """Generate base-referenced tilt/roll release poses for a sphere.

        A sphere's observed and target rotations are physically irrelevant.
        Keeping a rigid object-frame transform here unnecessarily forces a
        vertical wrist pose that RM75 cannot reach near the rear-left bin.
        Preserve the held center distance while expanding only release TCP
        orientation, matching the proven legacy tennis release strategy.
        """
        spec = get_object_spec(loaded_asset)
        if spec is None:
            raise KeyError(loaded_asset)
        mesh_scale, _ = resolve_object_spec_scales(spec)
        loaded = trimesh.load(Path(spec.mesh_file), force="scene", process=False)
        source_center = _world_aabb_center(loaded, mesh_scale, T_planning_object)
        goal_center = _world_aabb_center(loaded, mesh_scale, T_planning_goal_object)
        grasp_transform = _pose_to_matrix(grasp.pose)
        tcp_center_distance = float(
            np.linalg.norm(source_center - grasp_transform[:3, 3])
        )
        toward_base = np.asarray([-goal_center[0], -goal_center[1], 0.0])
        if np.linalg.norm(toward_base) < 1e-8:
            toward_base = np.asarray([1.0, 0.0, 0.0])
        toward_base /= np.linalg.norm(toward_base)
        down = np.asarray([0.0, 0.0, -1.0])
        output: list[PoseCandidate] = []
        for tilt_deg in self.config.sphere_release_tilt_toward_base_deg:
            tilt_rad = np.deg2rad(abs(float(tilt_deg)))
            shift_direction = toward_base * (1.0 if tilt_deg >= 0.0 else -1.0)
            approach = _normalized(
                np.cos(tilt_rad) * down - np.sin(tilt_rad) * shift_direction
            )
            closing = _normalized(np.cross(shift_direction, down))
            orthogonal = _normalized(np.cross(closing, approach))
            closing = _normalized(np.cross(approach, orthogonal))
            base_rotation = np.stack([orthogonal, closing, approach], axis=1)
            tcp_position = goal_center - approach * tcp_center_distance
            tcp_position[2] += float(self.config.sphere_release_lift_m)
            for roll_deg in self.config.sphere_release_roll_deg:
                angle = np.deg2rad(float(roll_deg))
                c, s = np.cos(angle), np.sin(angle)
                local_roll = np.asarray(
                    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
                )
                transform = np.eye(4, dtype=np.float64)
                transform[:3, :3] = base_rotation @ local_roll
                transform[:3, 3] = tcp_position
                output.append(
                    PoseCandidate(
                        f"sphere_place_tilt_{tilt_deg:+.0f}_roll_{roll_deg:+.0f}",
                        _matrix_to_pose(transform),
                        score=grasp.score
                        - 0.001 * abs(float(tilt_deg))
                        - 0.00001 * abs(float(roll_deg)),
                        metadata={
                            "paired_grasp_id": grasp.candidate_id,
                            "sphere_release": True,
                            "release_tilt_toward_base_deg": float(tilt_deg),
                            "release_roll_deg": float(roll_deg),
                            "tcp_center_distance_m": tcp_center_distance,
                            "sphere_release_lift_m": float(
                                self.config.sphere_release_lift_m
                            ),
                            "target_object_pose": T_world_goal_object.tolist(),
                            "planning_target_object_pose": T_planning_goal_object.tolist(),
                            "search_tier": int(abs(float(tilt_deg)) > 15.0),
                            "atom_id": atom_id,
                        },
                    )
                )
        return tuple(output)

    def _axial_release_candidates(
        self,
        grasp: PoseCandidate,
        T_tcp_object: np.ndarray,
        T_planning_goal_object: np.ndarray,
        T_world_goal_object: np.ndarray,
        *,
        symmetry_axis_local: tuple[float, float, float] | None,
        bidirectional: bool,
        atom_id: str,
    ) -> tuple[PoseCandidate, ...]:
        """Expand only the physically irrelevant spin around an object's axis."""

        axis = np.asarray(
            symmetry_axis_local or (0.0, 1.0, 0.0), dtype=np.float64
        ).reshape(3)
        axis = _normalized(axis)
        output: list[PoseCandidate] = []
        flip_axis = _perpendicular_axis(axis)
        flip_options = (False, True) if bidirectional else (False,)
        for flipped in flip_options:
            flip_rotation = (
                _axis_angle_rotation(flip_axis, np.pi)
                if flipped
                else np.eye(3, dtype=np.float64)
            )
            for spin_deg in self.config.axial_place_spin_deg:
                spin_abs = abs(float(spin_deg))
                if flipped:
                    symmetry_search_tier = 3
                elif spin_abs <= 1e-6:
                    symmetry_search_tier = 0
                elif spin_abs <= 90.0 + 1e-6:
                    symmetry_search_tier = 1
                else:
                    symmetry_search_tier = 2
                equivalent_local = np.eye(4, dtype=np.float64)
                equivalent_local[:3, :3] = (
                    _axis_angle_rotation(axis, np.deg2rad(float(spin_deg)))
                    @ flip_rotation
                )
                planning_target = T_planning_goal_object @ equivalent_local
                world_target = T_world_goal_object @ equivalent_local
                T_planning_tcp = planning_target @ np.linalg.inv(T_tcp_object)
                flip_label = "_flip" if flipped else ""
                output.append(
                    PoseCandidate(
                        f"place_for_{grasp.candidate_id}__axial_{spin_deg:+.0f}{flip_label}",
                        _matrix_to_pose(T_planning_tcp),
                        score=float(grasp.score)
                        - 0.0001 * abs(float(spin_deg))
                        - 0.00005 * int(flipped),
                        metadata={
                            "paired_grasp_id": grasp.candidate_id,
                            "orientation_symmetry": (
                                "axial_bidirectional" if bidirectional else "axial"
                            ),
                            "axial_spin_deg": float(spin_deg),
                            "axis_flipped": bool(flipped),
                            "refinement_parent_id": grasp.metadata.get(
                                "refinement_parent_id"
                            ),
                            "T_tcp_object": T_tcp_object.tolist(),
                            "target_object_pose": world_target.tolist(),
                            "canonical_target_object_pose": T_world_goal_object.tolist(),
                            "planning_target_object_pose": planning_target.tolist(),
                            "search_tier": max(
                                int(grasp.metadata.get("search_tier", 0)),
                                symmetry_search_tier,
                            ),
                            "atom_id": atom_id,
                        },
                    )
                )
        return tuple(output)


def _world_aabb_center(
    loaded: trimesh.Scene,
    mesh_scale: float,
    T_world_object: np.ndarray,
) -> np.ndarray:
    world_mesh = loaded.copy()
    scale_transform = np.eye(4, dtype=np.float64)
    scale_transform[:3, :3] *= float(mesh_scale)
    world_mesh.apply_transform(
        np.asarray(T_world_object, dtype=np.float64).reshape(4, 4) @ scale_transform
    )
    bounds = np.asarray(world_mesh.bounds, dtype=np.float64).reshape(2, 3)
    return 0.5 * (bounds[0] + bounds[1])


def _normalized(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError("cannot normalize a zero-length grasp axis")
    return vector / norm


def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    x, y, z = _normalized(axis)
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _perpendicular_axis(axis: np.ndarray) -> np.ndarray:
    axis = _normalized(axis)
    seed = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(axis, seed))) > 0.9:
        seed = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return _normalized(seed - float(np.dot(seed, axis)) * axis)


def _matrix_to_pose(transform: np.ndarray) -> Pose:
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return Pose(matrix[:3, 3], matrix_to_quaternion_wxyz(matrix[:3, :3]))


def _pose_to_matrix(pose: Pose) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    w, x, y, z = np.asarray(pose.quaternion_wxyz, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    transform[:3, 3] = pose.position
    return transform
