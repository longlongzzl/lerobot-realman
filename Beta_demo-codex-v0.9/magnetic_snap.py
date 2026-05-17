from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import sapien
from transforms3d.quaternions import mat2quat, quat2mat


@dataclass(frozen=True)
class LockedPanelPose:
    role: str
    actor: Any
    position: np.ndarray
    quaternion: np.ndarray


@dataclass(frozen=True)
class MagneticConnection:
    parent: str
    parent_edge: str
    child: str
    child_edge: str
    mode: str
    parent_lane: str = "rim"
    child_lane: str = "rim"


@dataclass
class ActiveMagneticConnection:
    connection: MagneticConnection
    drives: list[Any]
    active: bool = True
    active_steps: int = 0


@dataclass(frozen=True)
class EdgeSpec:
    start: np.ndarray
    end: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return ((self.start + self.end) * 0.5).astype(np.float32)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def points(self) -> list[np.ndarray]:
        return [self.start, self.center, self.end]


def _quat_from_columns(x_axis: list[float], y_axis: list[float], z_axis: list[float]) -> np.ndarray:
    matrix = np.asarray([x_axis, y_axis, z_axis], dtype=np.float64).T
    return mat2quat(matrix).astype(np.float32)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        raise ValueError("cannot normalize a zero-length vector")
    return (vec / norm).astype(np.float32)


def _triangle_roof_quat(base_dir: list[float], apex_dir: list[float]) -> np.ndarray:
    x_axis = _normalize(np.asarray(base_dir, dtype=np.float32))
    z_axis = _normalize(np.asarray(apex_dir, dtype=np.float32))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    x_axis = _normalize(np.cross(y_axis, z_axis))
    return _quat_from_columns(x_axis.tolist(), y_axis.tolist(), z_axis.tolist())


class MagneticSnapManager:
    BASE_STRUCTURE_ROLES = {"floor", "right_wall", "back_wall", "left_wall", "front_wall"}

    def __init__(
        self,
        *,
        plate_size: float,
        plate_thickness: float,
        triangle_height: float,
        triangle_width: float,
        triangle_thickness: float,
        magnet_mode: str = "edge_pair_drive",
        position_stiffness: float = 900.0,
        position_damping: float = 55.0,
        rotation_stiffness: float = 0.08,
        rotation_damping: float = 0.015,
        drive_stiffness: float = 60.0,
        drive_damping: float = 10.0,
        drive_force_limit: float = 1.0,
        drive_angular_stiffness: float = 0.0,
        drive_angular_damping: float = 0.0,
        drive_angular_force_limit: float = 0.0,
        attract_distance: float = 0.03,
        attract_stiffness: float = 1.6,
        attract_force_limit: float = 0.65,
        attract_torque_stiffness: float = 0.055,
        attract_torque_limit: float = 0.035,
        attract_normal_torque_stiffness: float = 0.16,
        attract_normal_torque_limit: float = 0.12,
        active_magnet_stiffness: float = 0.0,
        active_magnet_damping: float = 0.0,
        active_magnet_force_limit: float = 0.0,
        active_edge_torque_scale: float = 0.0,
        active_edge_torque_min_scale: float = 0.0,
        active_normal_torque_scale: float = 1.0,
        active_normal_torque_min_scale: float = 0.0,
        active_torque_delay_steps: int = 0,
        active_torque_ramp_steps: int = 0,
        active_torque_max_point_error: float = 0.0,
        floor_support_score: float = 8.0,
        support_connection_score_scale: float = 1.4,
        multi_connection_support_bonus: float = 2.0,
        upright_torque_damping: float = 0.10,
        attach_distance: float = 0.018,
        detach_distance: float = 0.035,
        base_connection_detach_distance: float = 0.0,
        max_length_error: float = 0.006,
        edge_sample_half_span: float = 0.0,
        connection_edge_sample_half_span: float = 0.0,
        edge_sample_offsets: list[float] | tuple[float, ...] | None = None,
        connection_edge_sample_offsets: list[float] | tuple[float, ...] | None = None,
        dynamic_update_period: int = 5,
    ):
        self.plate_size = float(plate_size)
        self.plate_thickness = float(plate_thickness)
        self.triangle_height = float(triangle_height)
        self.triangle_width = float(triangle_width)
        self.triangle_thickness = float(triangle_thickness)
        self.magnet_mode = magnet_mode
        self.position_stiffness = float(position_stiffness)
        self.position_damping = float(position_damping)
        self.rotation_stiffness = float(rotation_stiffness)
        self.rotation_damping = float(rotation_damping)
        self.drive_stiffness = float(drive_stiffness)
        self.drive_damping = float(drive_damping)
        self.drive_force_limit = float(drive_force_limit)
        self.drive_angular_stiffness = float(drive_angular_stiffness)
        self.drive_angular_damping = float(drive_angular_damping)
        self.drive_angular_force_limit = float(drive_angular_force_limit)
        self.attract_distance = float(attract_distance)
        self.attract_stiffness = float(attract_stiffness)
        self.attract_force_limit = float(attract_force_limit)
        self.attract_torque_stiffness = float(attract_torque_stiffness)
        self.attract_torque_limit = float(attract_torque_limit)
        self.attract_normal_torque_stiffness = float(attract_normal_torque_stiffness)
        self.attract_normal_torque_limit = float(attract_normal_torque_limit)
        self.active_magnet_stiffness = float(active_magnet_stiffness)
        self.active_magnet_damping = float(active_magnet_damping)
        self.active_magnet_force_limit = float(active_magnet_force_limit)
        self.active_edge_torque_scale = float(active_edge_torque_scale)
        self.active_edge_torque_min_scale = float(active_edge_torque_min_scale)
        self.active_normal_torque_scale = float(active_normal_torque_scale)
        self.active_normal_torque_min_scale = float(active_normal_torque_min_scale)
        self.active_torque_delay_steps = int(active_torque_delay_steps)
        self.active_torque_ramp_steps = int(active_torque_ramp_steps)
        self.active_torque_max_point_error = float(active_torque_max_point_error)
        self.floor_support_score = float(floor_support_score)
        self.support_connection_score_scale = float(support_connection_score_scale)
        self.multi_connection_support_bonus = float(multi_connection_support_bonus)
        self.upright_torque_damping = float(upright_torque_damping)
        self.attach_distance = float(attach_distance)
        self.detach_distance = float(detach_distance)
        self.base_connection_detach_distance = float(base_connection_detach_distance)
        self.max_length_error = float(max_length_error)
        self.edge_sample_half_span = float(edge_sample_half_span)
        self.connection_edge_sample_half_span = float(connection_edge_sample_half_span)
        self.edge_sample_offsets = list(edge_sample_offsets or [])
        self.connection_edge_sample_offsets = list(connection_edge_sample_offsets or [])
        self.dynamic_update_period = max(int(dynamic_update_period), 1)
        self._dynamic_update_counter = 0
        self._step_pose_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._locked_panel_by_role: dict[str, LockedPanelPose] = {}
        self._edge_spec_cache: dict[tuple[str, str, str], EdgeSpec] = {}
        self.locked_panel_poses: list[LockedPanelPose] = []
        self.connections: list[MagneticConnection] = []
        self.active_connections: list[ActiveMagneticConnection] = []
        self.drives: list[Any] = []
        self.disabled_roles: set[str] = set()
        self.suspended_roles: set[str] = set()
        self.desired_active_connections_by_role: dict[str, int] = {}
        self.allowed_edges_by_role: dict[str, set[str]] = {}
        self.allowed_connection_keys: set[tuple[tuple[str, str, str], tuple[str, str, str]]] | None = None

    def arrange_open_cube(self, plates: list[Any], center_xy: tuple[float, float] = (0.0, 0.0)) -> None:
        if len(plates) < 5:
            raise ValueError("open cube assembly requires at least 5 square plates")

        self._disable_all_drives()
        self._dynamic_update_counter = 0
        self.locked_panel_poses.clear()
        self.connections.clear()
        self.active_connections.clear()
        self.drives.clear()
        self.disabled_roles.clear()
        self.suspended_roles.clear()
        self.desired_active_connections_by_role.clear()
        self.allowed_edges_by_role.clear()
        self.allowed_connection_keys = None
        self._refresh_locked_panel_map()
        cx, cy = center_xy
        size = self.plate_size
        thickness = self.plate_thickness
        half_size = size / 2.0
        half_thickness = thickness / 2.0
        floor_top_z = thickness
        wall_center_z = floor_top_z + half_size

        q_floor = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        q_right = _quat_from_columns([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
        q_left = _quat_from_columns([0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0])
        q_back = _quat_from_columns([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
        q_front = _quat_from_columns([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0])

        role_specs = [
            (
                "floor",
                [cx, cy, half_thickness],
                q_floor,
            ),
            (
                "right_wall",
                [cx + half_size + half_thickness, cy, wall_center_z],
                q_right,
            ),
            (
                "left_wall",
                [cx - half_size - half_thickness, cy, wall_center_z],
                q_left,
            ),
            (
                "back_wall",
                [cx, cy + half_size + half_thickness, wall_center_z],
                q_back,
            ),
            (
                "front_wall",
                [cx, cy - half_size - half_thickness, wall_center_z],
                q_front,
            ),
        ]

        for actor_group_start in range(0, len(plates), 5):
            group = plates[actor_group_start : actor_group_start + 5]
            if len(group) < 5:
                break
            for actor, (role, position, quaternion) in zip(group, role_specs):
                self.locked_panel_poses.append(
                    LockedPanelPose(
                        role=role,
                        actor=actor,
                        position=np.asarray(position, dtype=np.float32),
                        quaternion=np.asarray(quaternion, dtype=np.float32),
                    )
                )

        self.connections.extend(
            [
                MagneticConnection("floor", "right_edge", "right_wall", "bottom_edge", "right_angle", "face_b", "face_a"),
                MagneticConnection("floor", "left_edge", "left_wall", "bottom_edge", "right_angle", "face_b", "face_a"),
                MagneticConnection("floor", "top_edge", "back_wall", "bottom_edge", "right_angle", "face_b", "face_a"),
                MagneticConnection("floor", "bottom_edge", "front_wall", "bottom_edge", "right_angle", "face_b", "face_a"),
                MagneticConnection("right_wall", "right_edge", "back_wall", "left_edge", "right_angle", "face_a", "face_a"),
                MagneticConnection("back_wall", "right_edge", "left_wall", "left_edge", "right_angle", "face_a", "face_a"),
                MagneticConnection("left_wall", "right_edge", "front_wall", "left_edge", "right_angle", "face_a", "face_a"),
                MagneticConnection("front_wall", "right_edge", "right_wall", "left_edge", "right_angle", "face_a", "face_a"),
            ]
        )
        self.apply_initial_poses()

    def register_free_triangles(self, triangles: list[Any], role_prefix: str = "free_triangle") -> None:
        for index, triangle in enumerate(triangles):
            role = f"{role_prefix}_{index}"
            if any(locked.role == role for locked in self.locked_panel_poses):
                continue
            self.locked_panel_poses.append(
                LockedPanelPose(
                    role=role,
                    actor=triangle,
                    position=self._actor_position(triangle),
                    quaternion=self._actor_quaternion(triangle),
                )
            )
        self._refresh_locked_panel_map()

    def arrange_house(
        self,
        plates: list[Any],
        triangles: list[Any],
        center_xy: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        if len(triangles) < 4:
            raise ValueError("house assembly requires at least 4 triangular roof panels")
        self.arrange_open_cube(plates, center_xy=center_xy)

        cx, cy = center_xy
        size = self.plate_size
        half_size = size / 2.0
        wall_top_z = self.plate_thickness + size
        roof_base_offset = half_size + self.plate_thickness / 2.0
        roof_run = roof_base_offset
        roof_rise = float(np.sqrt(max(self.triangle_height**2 - roof_run**2, 0.0)))
        peak_z = wall_top_z + roof_rise

        roof_specs = [
            (
                "right_roof",
                [cx + roof_base_offset, cy, wall_top_z],
                [0.0, 1.0, 0.0],
                [cx, cy, peak_z],
                "right_wall",
                "top_edge",
            ),
            (
                "left_roof",
                [cx - roof_base_offset, cy, wall_top_z],
                [0.0, -1.0, 0.0],
                [cx, cy, peak_z],
                "left_wall",
                "top_edge",
            ),
            (
                "back_roof",
                [cx, cy + roof_base_offset, wall_top_z],
                [-1.0, 0.0, 0.0],
                [cx, cy, peak_z],
                "back_wall",
                "top_edge",
            ),
            (
                "front_roof",
                [cx, cy - roof_base_offset, wall_top_z],
                [1.0, 0.0, 0.0],
                [cx, cy, peak_z],
                "front_wall",
                "top_edge",
            ),
        ]

        for actor_group_start in range(0, len(triangles), 4):
            group = triangles[actor_group_start : actor_group_start + 4]
            if len(group) < 4:
                break
            for actor, (role, base_position, base_dir, peak_position, parent, parent_edge) in zip(group, roof_specs):
                apex_dir = np.asarray(peak_position, dtype=np.float32) - np.asarray(base_position, dtype=np.float32)
                self.locked_panel_poses.append(
                    LockedPanelPose(
                        role=role,
                        actor=actor,
                        position=np.asarray(base_position, dtype=np.float32),
                        quaternion=_triangle_roof_quat(base_dir, apex_dir),
                    )
                )
                self.connections.append(
                    MagneticConnection(parent, parent_edge, role, "base_edge", "roof_base_snap")
                )

        self.connections.extend(
            [
                MagneticConnection("right_roof", "right_slope_edge", "back_roof", "left_slope_edge", "roof_ridge"),
                MagneticConnection("back_roof", "right_slope_edge", "left_roof", "left_slope_edge", "roof_ridge"),
                MagneticConnection("left_roof", "right_slope_edge", "front_roof", "left_slope_edge", "roof_ridge"),
                MagneticConnection("front_roof", "right_slope_edge", "right_roof", "left_slope_edge", "roof_ridge"),
            ]
        )
        self._refresh_locked_panel_map()
        self.apply_initial_poses()

    def arrange_triangle_vertical(
        self,
        plates: list[Any],
        triangles: list[Any],
        center_xy: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        if not plates or not triangles:
            raise ValueError("triangle vertical assembly requires at least 1 square plate and 1 triangle")

        self._disable_all_drives()
        self._dynamic_update_counter = 0
        self.locked_panel_poses.clear()
        self.connections.clear()
        self.active_connections.clear()
        self.drives.clear()
        self.disabled_roles.clear()
        self.suspended_roles.clear()
        self.desired_active_connections_by_role.clear()
        self.allowed_edges_by_role.clear()
        self.allowed_connection_keys = None
        self._refresh_locked_panel_map()

        cx, cy = center_xy
        half_size = self.plate_size / 2.0
        half_thickness = self.plate_thickness / 2.0
        floor_pose = LockedPanelPose(
            role="floor",
            actor=plates[0],
            position=np.asarray([cx, cy, half_thickness], dtype=np.float32),
            quaternion=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        triangle_pose = LockedPanelPose(
            role="vertical_triangle",
            actor=triangles[0],
            position=np.asarray([cx, cy + half_size, half_thickness], dtype=np.float32),
            quaternion=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        self.locked_panel_poses.extend([floor_pose, triangle_pose])
        self._refresh_locked_panel_map()
        self.connections.append(
            MagneticConnection(
                "floor",
                "top_edge",
                "vertical_triangle",
                "base_edge",
                "single_support_base",
            )
        )
        self.apply_initial_poses()

    def clear(self) -> None:
        self._disable_all_drives()
        self._dynamic_update_counter = 0
        self._step_pose_cache = {}
        self.locked_panel_poses.clear()
        self.connections.clear()
        self.active_connections.clear()
        self.drives.clear()
        self.disabled_roles.clear()
        self.suspended_roles.clear()
        self.desired_active_connections_by_role.clear()
        self.allowed_edges_by_role.clear()
        self.allowed_connection_keys = None
        self._refresh_locked_panel_map()

    def apply_initial_poses(self) -> None:
        for locked in self.locked_panel_poses:
            locked.actor.set_pose(
                sapien.Pose(
                    p=locked.position.tolist(),
                    q=locked.quaternion.tolist(),
                )
            )
            self._set_velocity(locked.actor, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))

    def apply(self) -> None:
        if self.magnet_mode == "hard_lock":
            self.apply_initial_poses()
            return
        if self.magnet_mode == "soft_magnetic":
            self.apply_soft_magnetic_forces()
            return
        if self.magnet_mode in {"edge_pair_drive", "edge_drive"}:
            self._step_pose_cache = {
                id(locked.actor): self._actor_pose_data(locked.actor)
                for locked in self.locked_panel_poses
            }
            try:
                if self.dynamic_update_period <= 1 or self._dynamic_update_counter % self.dynamic_update_period == 0:
                    self.update_dynamic_connections()
                else:
                    self._apply_active_connection_fast_update()
                self._dynamic_update_counter += 1
            finally:
                self._step_pose_cache = {}

    def ensure_magnetic_drives(self, scene: Any) -> None:
        if self.drives:
            return
        self.scene = scene
        if self.magnet_mode in {"drive_magnetic", "relative_drive"}:
            self._create_relative_pose_drives(scene)
        elif self.magnet_mode == "edge_drive":
            self._create_edge_drives(scene, points_per_edge=1)
        elif self.magnet_mode == "edge_pair_drive":
            self._create_edge_drives(scene, points_per_edge=2)

    def _create_relative_pose_drives(self, scene: Any) -> None:
        locked_by_role = self._locked_by_role()
        for connection in self.connections:
            parent = locked_by_role.get(connection.parent)
            child = locked_by_role.get(connection.child)
            if parent is None or child is None:
                continue
            parent_pose = sapien.Pose(parent.position.tolist(), parent.quaternion.tolist())
            child_pose = sapien.Pose(child.position.tolist(), child.quaternion.tolist())
            pose_in_parent = sapien.Pose()
            pose_in_child = child_pose.inv() * parent_pose
            drive = scene.create_drive(parent.actor, pose_in_parent, child.actor, pose_in_child)
            self._configure_drive(drive)
            self.drives.append(drive)
            self.active_connections.append(
                ActiveMagneticConnection(
                    connection=connection,
                    drives=[drive],
                    active=True,
                )
            )

    def _create_edge_drives(self, scene: Any, *, points_per_edge: int) -> None:
        locked_by_role = self._locked_by_role()
        for connection in self.connections:
            parent = locked_by_role.get(connection.parent)
            child = locked_by_role.get(connection.child)
            if parent is None or child is None:
                continue
            parent_points, child_points = self._matched_connection_points(
                connection,
                parent,
                child,
                points_per_edge=points_per_edge,
                use_locked_pose=True,
            )
            parent_pose_world = sapien.Pose(parent.position.tolist(), parent.quaternion.tolist())
            child_pose_world = sapien.Pose(child.position.tolist(), child.quaternion.tolist())
            drives = []
            for parent_point, child_point in zip(parent_points, child_points):
                pose_in_parent = sapien.Pose(p=parent_point.tolist())
                desired_world = parent_pose_world * pose_in_parent
                pose_in_child = child_pose_world.inv() * desired_world
                pose_in_child = sapien.Pose(p=child_point.tolist(), q=pose_in_child.q)
                drive = scene.create_drive(parent.actor, pose_in_parent, child.actor, pose_in_child)
                self._configure_drive(drive)
                self.drives.append(drive)
                drives.append(drive)
            self.active_connections.append(
                ActiveMagneticConnection(
                    connection=connection,
                    drives=drives,
                    active=True,
                )
            )

    def _matched_edge_points(
        self,
        parent: LockedPanelPose,
        parent_edge: EdgeSpec,
        child: LockedPanelPose,
        child_edge: EdgeSpec,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        parent_points = self._sampled_edge_points(
            parent_edge,
            half_span=self.edge_sample_half_span,
            include_center=True,
            offsets=self.edge_sample_offsets,
        )
        child_points = self._sampled_edge_points(
            child_edge,
            half_span=self.edge_sample_half_span,
            include_center=True,
            offsets=self.edge_sample_offsets,
        )
        p_world = self._current_world_points_array(parent.actor, parent_points)
        c_world = self._current_world_points_array(child.actor, child_points)
        direct = sum(float(np.linalg.norm(a - b)) for a, b in zip(p_world, c_world))
        flipped = sum(float(np.linalg.norm(a - b)) for a, b in zip(p_world, reversed(c_world)))
        if flipped < direct:
            child_points = list(reversed(child_points))
        return parent_points, child_points

    def _matched_connection_edge_points(
        self,
        parent: LockedPanelPose,
        parent_edge: EdgeSpec,
        child: LockedPanelPose,
        child_edge: EdgeSpec,
        *,
        use_locked_pose: bool,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        parent_points = self._sampled_edge_points(
            parent_edge,
            half_span=self.connection_edge_sample_half_span,
            include_center=False,
            offsets=self.connection_edge_sample_offsets,
        )
        child_points = self._sampled_edge_points(
            child_edge,
            half_span=self.connection_edge_sample_half_span,
            include_center=False,
            offsets=self.connection_edge_sample_offsets,
        )
        p_world = self._world_points_for_pose(parent, parent_points, use_locked_pose=use_locked_pose)
        c_world = self._world_points_for_pose(child, child_points, use_locked_pose=use_locked_pose)
        direct = sum(float(np.linalg.norm(a - b)) for a, b in zip(p_world, c_world))
        flipped = sum(float(np.linalg.norm(a - b)) for a, b in zip(p_world, reversed(c_world)))
        if flipped < direct:
            child_points = list(reversed(child_points))
        return parent_points, child_points

    def _sampled_edge_points(
        self,
        edge: EdgeSpec,
        *,
        half_span: float,
        include_center: bool,
        offsets: list[float] | tuple[float, ...] | None = None,
    ) -> list[np.ndarray]:
        offset_values = self._expanded_edge_sample_offsets(offsets, edge.length)
        if offset_values:
            direction = (edge.end - edge.start).astype(np.float32)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-8:
                return edge.points if include_center else [edge.start, edge.end]
            direction = direction / norm
            center = edge.center
            return [(center + direction * offset).astype(np.float32) for offset in offset_values]
        span = float(half_span)
        half_length = edge.length * 0.5
        if span <= 0.0 or half_length <= 1e-8:
            return edge.points if include_center else [edge.start, edge.end]
        clamped_span = min(span, half_length)
        direction = (edge.end - edge.start).astype(np.float32)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            return edge.points if include_center else [edge.start, edge.end]
        direction = direction / norm
        center = edge.center
        if include_center:
            return [
                (center - direction * clamped_span).astype(np.float32),
                center,
                (center + direction * clamped_span).astype(np.float32),
            ]
        return [
            (center - direction * clamped_span).astype(np.float32),
            (center + direction * clamped_span).astype(np.float32),
        ]

    def _expanded_edge_sample_offsets(
        self,
        offsets: list[float] | tuple[float, ...] | None,
        edge_length: float,
    ) -> list[float]:
        half_length = float(edge_length) * 0.5
        magnitudes = sorted({abs(float(item)) for item in list(offsets or []) if abs(float(item)) > 1e-8}, reverse=True)
        if not magnitudes or half_length <= 1e-8:
            return []
        signed: list[float] = []
        for magnitude in magnitudes:
            clamped = min(float(magnitude), half_length)
            signed.append(-clamped)
        for magnitude in reversed(magnitudes):
            clamped = min(float(magnitude), half_length)
            signed.append(clamped)
        return signed

    def _min_pairwise_point_distance(
        self,
        parent_world: list[np.ndarray],
        child_world: list[np.ndarray],
    ) -> float:
        parent_points = np.asarray(parent_world, dtype=np.float32)
        child_points = np.asarray(child_world, dtype=np.float32)
        if parent_points.size == 0 or child_points.size == 0:
            return float("inf")
        deltas = parent_points[:, None, :] - child_points[None, :, :]
        return float(np.min(np.linalg.norm(deltas, axis=2)))

    def _matched_point_distances(
        self,
        parent_world: list[np.ndarray],
        child_world: list[np.ndarray],
    ) -> np.ndarray:
        parent_points = np.asarray(parent_world, dtype=np.float32)
        child_points = np.asarray(child_world, dtype=np.float32)
        if parent_points.size == 0 or child_points.size == 0:
            return np.asarray([], dtype=np.float32)
        count = min(parent_points.shape[0], child_points.shape[0])
        if count <= 0:
            return np.asarray([], dtype=np.float32)
        return np.linalg.norm(parent_points[:count] - child_points[:count], axis=1).astype(np.float32)

    def _max_matched_point_distance(
        self,
        parent_world: list[np.ndarray],
        child_world: list[np.ndarray],
    ) -> float:
        distances = self._matched_point_distances(parent_world, child_world)
        if distances.size == 0:
            return 0.0
        return float(np.max(distances))

    def _matched_connection_points(
        self,
        connection: MagneticConnection,
        parent: LockedPanelPose,
        child: LockedPanelPose,
        *,
        points_per_edge: int,
        use_locked_pose: bool,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        parent_edge = self._edge_spec(parent.role, connection.parent_edge, connection.parent_lane)
        child_edge = self._edge_spec(child.role, connection.child_edge, connection.child_lane)
        if points_per_edge == 1:
            return [parent_edge.center], [child_edge.center]
        if self._use_magnetic_strip_points(connection):
            strip_points = self._matched_strip_points(connection, parent, child, use_locked_pose=use_locked_pose)
            if strip_points is not None:
                return strip_points
        return self._matched_connection_edge_points(
            parent,
            parent_edge,
            child,
            child_edge,
            use_locked_pose=use_locked_pose,
        )

    def _use_magnetic_strip_points(self, connection: MagneticConnection) -> bool:
        if self.magnet_mode != "edge_pair_drive":
            return False
        if connection.parent_lane != "rim" or connection.child_lane != "rim":
            return False
        return connection.mode in {"single_support_base", "roof_base_snap", "reattach_parallel"}

    def _matched_strip_points(
        self,
        connection: MagneticConnection,
        parent: LockedPanelPose,
        child: LockedPanelPose,
        *,
        use_locked_pose: bool,
    ) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
        lane_options = [
            [("face_a", "face_a"), ("face_b", "face_b")],
            [("face_a", "face_b"), ("face_b", "face_a")],
        ]
        best: tuple[float, list[np.ndarray], list[np.ndarray]] | None = None
        for lane_pairs in lane_options:
            parent_points: list[np.ndarray] = []
            child_points: list[np.ndarray] = []
            score = 0.0
            for parent_lane, child_lane in lane_pairs:
                parent_edge = self._edge_spec(parent.role, connection.parent_edge, parent_lane)
                child_edge = self._edge_spec(child.role, connection.child_edge, child_lane)
                parent_pair, child_pair = self._matched_edge_endpoints(
                    parent,
                    parent_edge,
                    child,
                    child_edge,
                    use_locked_pose=use_locked_pose,
                )
                parent_points.extend(parent_pair)
                child_points.extend(child_pair)
                parent_world = self._world_points_for_pose(parent, parent_pair, use_locked_pose=use_locked_pose)
                child_world = self._world_points_for_pose(child, child_pair, use_locked_pose=use_locked_pose)
                score += sum(float(np.linalg.norm(a - b)) for a, b in zip(parent_world, child_world))
            if best is None or score < best[0]:
                best = (score, parent_points, child_points)
        if best is None:
            return None
        return best[1], best[2]

    def _matched_edge_endpoints(
        self,
        parent: LockedPanelPose,
        parent_edge: EdgeSpec,
        child: LockedPanelPose,
        child_edge: EdgeSpec,
        *,
        use_locked_pose: bool,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        parent_sampled = self._sampled_edge_points(
            parent_edge,
            half_span=self.connection_edge_sample_half_span,
            include_center=False,
            offsets=self.connection_edge_sample_offsets,
        )
        child_sampled = self._sampled_edge_points(
            child_edge,
            half_span=self.connection_edge_sample_half_span,
            include_center=False,
            offsets=self.connection_edge_sample_offsets,
        )
        parent_points = list(parent_sampled)
        child_points = list(child_sampled)
        parent_world = self._world_points_for_pose(parent, parent_points, use_locked_pose=use_locked_pose)
        child_world = self._world_points_for_pose(child, child_points, use_locked_pose=use_locked_pose)
        direct = sum(float(np.linalg.norm(a - b)) for a, b in zip(parent_world, child_world))
        flipped = sum(float(np.linalg.norm(a - b)) for a, b in zip(parent_world, reversed(child_world)))
        if flipped < direct:
            child_points = list(reversed(child_sampled))
        return parent_points, child_points

    def _world_points_for_pose(
        self,
        locked: LockedPanelPose,
        local_points: list[np.ndarray],
        *,
        use_locked_pose: bool,
    ) -> list[np.ndarray]:
        if use_locked_pose:
            world = self._transform_local_points(locked.position, locked.quaternion, local_points)
        else:
            world = self._current_world_points_array(locked.actor, local_points)
        return [np.asarray(point, dtype=np.float32) for point in world]

    def _configure_drive(self, drive: Any) -> None:
        drive.set_drive_property_x(
            self.drive_stiffness,
            self.drive_damping,
            self.drive_force_limit,
        )
        drive.set_drive_property_y(
            self.drive_stiffness,
            self.drive_damping,
            self.drive_force_limit,
        )
        drive.set_drive_property_z(
            self.drive_stiffness,
            self.drive_damping,
            self.drive_force_limit,
        )
        for item in getattr(drive, "_objs", []):
            item.set_drive_property_slerp(
                self.drive_angular_stiffness,
                self.drive_angular_damping,
                self.drive_angular_force_limit,
            )

    def refresh_drive_properties(self) -> None:
        for active_connection in self.active_connections:
            for drive in active_connection.drives:
                if active_connection.active:
                    self._configure_drive(drive)
                else:
                    self._disable_drive(drive)

    def _disable_drive(self, drive: Any) -> None:
        drive.set_drive_property_x(0.0, 0.0, 0.0)
        drive.set_drive_property_y(0.0, 0.0, 0.0)
        drive.set_drive_property_z(0.0, 0.0, 0.0)
        for item in getattr(drive, "_objs", []):
            item.set_drive_property_slerp(0.0, 0.0, 0.0)

    def _disable_all_drives(self) -> None:
        for drive in self.drives:
            self._disable_drive(drive)

    def _connection_support_score(self, role: str) -> float:
        if role == "floor":
            return max(float(self.floor_support_score), 1.0)
        active_count = self._active_connection_count(role)
        score = 1.0 + float(active_count) * max(float(self.support_connection_score_scale), 0.0)
        if active_count >= 2:
            score += max(float(self.multi_connection_support_bonus), 0.0)
        if self._has_single_supporting_edge(role):
            score += 0.5
        if role in self.suspended_roles:
            score *= 0.6
        return max(score, 0.5)

    def _pair_correction_weights(self, parent_role: str, child_role: str) -> tuple[float, float]:
        parent_score = self._connection_support_score(parent_role)
        child_score = self._connection_support_score(child_role)
        total = parent_score + child_score
        if total <= 1e-6:
            return 0.5, 0.5
        return child_score / total, parent_score / total

    def _active_error_scale(self, max_distance: float) -> float:
        detach_distance = max(float(self.detach_distance), float(self.attach_distance), 1e-6)
        if max_distance >= detach_distance:
            return 0.0
        return max((detach_distance - max_distance) / detach_distance, 0.0)

    def _active_torque_ramp(self, active_connection: ActiveMagneticConnection) -> float:
        delay_steps = max(int(self.active_torque_delay_steps), 0)
        ramp_steps = max(int(self.active_torque_ramp_steps), 0)
        age = max(int(getattr(active_connection, "active_steps", 0)), 0)
        if age <= delay_steps:
            return 0.0
        if ramp_steps <= 0:
            return 1.0
        return float(np.clip((age - delay_steps) / max(float(ramp_steps), 1.0), 0.0, 1.0))

    def _increment_active_connection_steps(self) -> None:
        for active_connection in self.active_connections:
            if active_connection.active:
                active_connection.active_steps = int(getattr(active_connection, "active_steps", 0)) + 1

    def update_dynamic_connections(self) -> None:
        self._detach_overstretched_connections()
        self._drop_underconstrained_pieces()
        self._reactivate_nearby_inactive_connections()
        self._drop_underconstrained_pieces()
        if not self._all_enabled_roles_connected():
            self._apply_magnetic_edge_attraction()
            self._attach_nearby_free_edges()
        self._apply_active_connection_magnetic_forces()
        self._apply_active_connection_normal_torques()
        self._apply_active_connection_upright_torques()
        self._increment_active_connection_steps()

    def _apply_active_connection_fast_update(self) -> None:
        self._detach_overstretched_connections()
        self._drop_underconstrained_pieces()
        self._apply_active_connection_magnetic_forces()
        self._apply_active_connection_normal_torques()
        self._apply_active_connection_upright_torques()
        self._increment_active_connection_steps()

    def _all_enabled_roles_connected(self) -> bool:
        for locked in self.locked_panel_poses:
            role = locked.role
            if role in self.disabled_roles:
                continue
            required_connections = max(int(self.desired_active_connections_by_role.get(role, 1)), 1)
            if self._active_connection_count(role) < required_connections:
                return False
        return not self.suspended_roles

    def _detach_overstretched_connections(self) -> None:
        for active_connection in self.active_connections:
            if not active_connection.active:
                continue
            max_error = self._connection_point_error(active_connection.connection)
            if max_error > self._detach_distance_for_connection(active_connection.connection):
                active_connection.active = False
                for drive in active_connection.drives:
                    self._disable_drive(drive)

    def _detach_distance_for_connection(self, connection: MagneticConnection) -> float:
        if (
            self.base_connection_detach_distance > 0.0
            and connection.parent in self.BASE_STRUCTURE_ROLES
            and connection.child in self.BASE_STRUCTURE_ROLES
        ):
            return max(float(self.detach_distance), float(self.base_connection_detach_distance))
        return float(self.detach_distance)

    def _reactivate_nearby_inactive_connections(self) -> None:
        occupied_edges = self._occupied_edges()
        for active_connection in self.active_connections:
            if active_connection.active:
                continue
            connection = active_connection.connection
            if connection.parent in self.disabled_roles or connection.child in self.disabled_roles:
                continue
            if not self._connection_allowed(connection):
                continue
            if not self._edge_allowed(connection.parent, connection.parent_edge):
                continue
            if not self._edge_allowed(connection.child, connection.child_edge):
                continue
            parent_key = (connection.parent, connection.parent_edge, connection.parent_lane)
            child_key = (connection.child, connection.child_edge, connection.child_lane)
            if self._edge_key_occupied(occupied_edges, parent_key) or self._edge_key_occupied(occupied_edges, child_key):
                continue
            if self._connection_point_error(connection) > self.attach_distance:
                continue
            active_connection.active = True
            active_connection.active_steps = 0
            for drive in active_connection.drives:
                self._configure_drive(drive)
            self._add_occupied_edge(occupied_edges, parent_key)
            self._add_occupied_edge(occupied_edges, child_key)
            if self._active_connection_count(connection.parent) >= 2 or self._has_single_supporting_edge(connection.parent):
                self.suspended_roles.discard(connection.parent)
            if self._active_connection_count(connection.child) >= 2 or self._has_single_supporting_edge(connection.child):
                self.suspended_roles.discard(connection.child)
        scene = getattr(self, "scene", None)
        if scene is None:
            return
        locked_by_role = self._locked_by_role()
        for connection in list(self.connections):
            if self._find_active_connection(connection) is not None or self._find_inactive_connection(connection) is not None:
                continue
            if connection.parent in self.disabled_roles or connection.child in self.disabled_roles:
                continue
            if not self._connection_allowed(connection):
                continue
            if not self._edge_allowed(connection.parent, connection.parent_edge):
                continue
            if not self._edge_allowed(connection.child, connection.child_edge):
                continue
            parent_key = (connection.parent, connection.parent_edge, connection.parent_lane)
            child_key = (connection.child, connection.child_edge, connection.child_lane)
            if self._edge_key_occupied(occupied_edges, parent_key) or self._edge_key_occupied(occupied_edges, child_key):
                continue
            parent = locked_by_role.get(connection.parent)
            child = locked_by_role.get(connection.child)
            if parent is None or child is None:
                continue
            parent_points, child_points = self._matched_connection_points(
                connection,
                parent,
                child,
                points_per_edge=2,
                use_locked_pose=False,
            )
            parent_world = self._current_world_points_array(parent.actor, parent_points)
            child_world = self._current_world_points_array(child.actor, child_points)
            if not self._dynamic_edge_pair_allowed(connection, parent, child, parent_world, child_world):
                continue
            if self._connection_point_error(connection) > self.attach_distance:
                continue
            self._create_runtime_edge_connection(scene, parent, child, connection)
            self._add_occupied_edge(occupied_edges, parent_key)
            self._add_occupied_edge(occupied_edges, child_key)
            if self._active_connection_count(connection.parent) >= 2 or self._has_single_supporting_edge(connection.parent):
                self.suspended_roles.discard(connection.parent)
            if self._active_connection_count(connection.child) >= 2 or self._has_single_supporting_edge(connection.child):
                self.suspended_roles.discard(connection.child)

    def _attach_nearby_free_edges(self) -> None:
        scene = getattr(self, "scene", None)
        if scene is None:
            return
        occupied_edges = self._occupied_edges()
        locked_by_role = self._locked_by_role()
        reattachable_suspended_roles = self._reattachable_suspended_roles(
            locked_by_role,
            occupied_edges,
        )
        roles = list(locked_by_role)
        for i, parent_role in enumerate(roles):
            if parent_role in self.disabled_roles:
                continue
            if parent_role in self.suspended_roles and parent_role not in reattachable_suspended_roles:
                continue
            for child_role in roles[i + 1 :]:
                if child_role in self.disabled_roles:
                    continue
                if child_role in self.suspended_roles and child_role not in reattachable_suspended_roles:
                    continue
                if self._has_active_pair_connection(parent_role, child_role):
                    continue
                parent = locked_by_role[parent_role]
                child = locked_by_role[child_role]
                attached_pair = False
                for parent_edge_name in self._edge_names(parent_role):
                    if not self._edge_allowed(parent_role, parent_edge_name):
                        continue
                    if attached_pair:
                        break
                    parent_edge = self._edge_spec(parent_role, parent_edge_name)
                    for parent_lane in self._dynamic_lane_names(parent_role, parent_edge_name):
                        parent_key = (parent_role, parent_edge_name, parent_lane)
                        if self._edge_key_occupied(occupied_edges, parent_key):
                            continue
                        for child_edge_name in self._edge_names(child_role):
                            if not self._edge_allowed(child_role, child_edge_name):
                                continue
                            child_edge = self._edge_spec(child_role, child_edge_name)
                            if abs(parent_edge.length - child_edge.length) > self.max_length_error:
                                continue
                            for child_lane in self._dynamic_lane_names(child_role, child_edge_name):
                                child_key = (child_role, child_edge_name, child_lane)
                                if self._edge_key_occupied(occupied_edges, child_key):
                                    continue
                                parent_points, child_points = self._matched_edge_points(
                                    parent,
                                    parent_edge,
                                    child,
                                    child_edge,
                                )
                                parent_world = self._current_world_points_array(parent.actor, parent_points)
                                child_world = self._current_world_points_array(child.actor, child_points)
                                max_distance = self._max_matched_point_distance(parent_world, child_world)
                                connection = MagneticConnection(
                                    parent_role,
                                    parent_edge_name,
                                    child_role,
                                    child_edge_name,
                                    self._dynamic_connection_mode(parent_role, parent_edge_name, child_role, child_edge_name, parent, child),
                                    parent_lane,
                                    child_lane,
                                )
                                if not self._dynamic_edge_pair_allowed(connection, parent, child, parent_world, child_world):
                                    continue
                                if not self._connection_allowed(connection):
                                    continue
                                if self._connection_point_error(connection) > self.attach_distance:
                                    continue
                                self._create_runtime_edge_connection(scene, parent, child, connection)
                                self._add_occupied_edge(occupied_edges, parent_key)
                                self._add_occupied_edge(occupied_edges, child_key)
                                if self._active_connection_count(parent_role) >= 2 or self._has_single_supporting_edge(parent_role):
                                    self.suspended_roles.discard(parent_role)
                                if self._active_connection_count(child_role) >= 2 or self._has_single_supporting_edge(child_role):
                                    self.suspended_roles.discard(child_role)
                                attached_pair = True
                                break
                            if attached_pair:
                                break
                        if attached_pair:
                            break

    def _apply_magnetic_edge_attraction(self) -> None:
        occupied_edges = self._occupied_edges()
        locked_by_role = self._locked_by_role()
        reattachable_suspended_roles = self._reattachable_suspended_roles(
            locked_by_role,
            occupied_edges,
            distance_threshold=self.attract_distance,
        )
        attracted_edges: set[tuple[str, str, str]] = set()
        roles = list(locked_by_role)
        for i, parent_role in enumerate(roles):
            if parent_role in self.disabled_roles:
                continue
            if parent_role in self.suspended_roles and parent_role not in reattachable_suspended_roles:
                continue
            parent = locked_by_role[parent_role]
            for child_role in roles[i + 1 :]:
                if child_role in self.disabled_roles:
                    continue
                if child_role in self.suspended_roles and child_role not in reattachable_suspended_roles:
                    continue
                if self._has_active_pair_connection(parent_role, child_role):
                    continue
                child = locked_by_role[child_role]
                best: tuple[float, float, str, str, str, str, list[np.ndarray], list[np.ndarray]] | None = None
                for parent_edge_name in self._edge_names(parent_role):
                    if not self._edge_allowed(parent_role, parent_edge_name):
                        continue
                    parent_edge = self._edge_spec(parent_role, parent_edge_name)
                    for parent_lane in self._dynamic_lane_names(parent_role, parent_edge_name):
                        parent_key = (parent_role, parent_edge_name, parent_lane)
                        if self._edge_key_occupied(occupied_edges, parent_key) or self._edge_key_occupied(attracted_edges, parent_key):
                            continue
                        for child_edge_name in self._edge_names(child_role):
                            if not self._edge_allowed(child_role, child_edge_name):
                                continue
                            child_edge = self._edge_spec(child_role, child_edge_name)
                            if abs(parent_edge.length - child_edge.length) > self.max_length_error:
                                continue
                            for child_lane in self._dynamic_lane_names(child_role, child_edge_name):
                                child_key = (child_role, child_edge_name, child_lane)
                                if self._edge_key_occupied(occupied_edges, child_key) or self._edge_key_occupied(attracted_edges, child_key):
                                    continue
                                parent_points, child_points = self._matched_edge_points(
                                    parent,
                                    parent_edge,
                                    child_edge=child_edge,
                                    child=child,
                                )
                                parent_world = self._current_world_points_array(parent.actor, parent_points)
                                child_world = self._current_world_points_array(child.actor, child_points)
                                max_distance = self._max_matched_point_distance(parent_world, child_world)
                                min_distance = self._min_pairwise_point_distance(parent_world, child_world)
                                if max_distance > self.attract_distance and min_distance > self.attach_distance:
                                    continue
                                connection = MagneticConnection(
                                    parent_role,
                                    parent_edge_name,
                                    child_role,
                                    child_edge_name,
                                    self._dynamic_connection_mode(parent_role, parent_edge_name, child_role, child_edge_name, parent, child),
                                    parent_lane,
                                    child_lane,
                                )
                                if not self._dynamic_edge_pair_allowed(connection, parent, child, parent_world, child_world):
                                    continue
                                if not self._connection_allowed(connection):
                                    continue
                                force_distance = max_distance if max_distance <= self.attract_distance else min_distance
                                score = max_distance if max_distance <= self.attract_distance else self.attract_distance + min_distance
                                if best is None or score < best[0]:
                                    best = (
                                        score,
                                        force_distance,
                                        parent_edge_name,
                                        parent_lane,
                                        child_edge_name,
                                        child_lane,
                                        parent_world,
                                        child_world,
                                    )
                if best is None:
                    continue
                _, force_distance, parent_edge_name, parent_lane, child_edge_name, child_lane, parent_world, child_world = best
                parent_weight, child_weight = self._pair_correction_weights(parent.role, child.role)
                self._apply_edge_pair_force(
                    parent.actor,
                    child.actor,
                    parent_world,
                    child_world,
                    force_distance,
                    parent_weight=parent_weight,
                    child_weight=child_weight,
                )
                self._apply_edge_alignment_torque(
                    parent.actor,
                    child.actor,
                    parent_world,
                    child_world,
                    force_distance,
                    parent_weight=parent_weight,
                    child_weight=child_weight,
                )
                self._apply_face_normal_alignment_torque(
                    parent,
                    child,
                    parent_world,
                    child_world,
                    force_distance,
                    parent_weight=parent_weight,
                    child_weight=child_weight,
                )
                self._add_occupied_edge(attracted_edges, (parent_role, parent_edge_name, parent_lane))
                self._add_occupied_edge(attracted_edges, (child_role, child_edge_name, child_lane))

    def _apply_edge_pair_force(
        self,
        parent_actor: Any,
        child_actor: Any,
        parent_world: list[np.ndarray],
        child_world: list[np.ndarray],
        max_distance: float,
        *,
        parent_weight: float = 0.5,
        child_weight: float = 0.5,
        distance_scale_floor: float = 0.0,
        torque_scale: float = 1.0,
    ) -> None:
        if max_distance <= 1e-6:
            return
        scale = max(self.attract_distance - max_distance, 0.0) / max(self.attract_distance, 1e-6)
        force_magnitude = min(self.attract_stiffness * scale, self.attract_force_limit)
        if force_magnitude <= 0.0:
            return
        for parent_point, child_point in zip(parent_world, child_world):
            delta = child_point - parent_point
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-6:
                continue
            force = (delta / distance * (force_magnitude / len(parent_world))).astype(np.float32)
            self._add_force_at_point(parent_actor, force * float(parent_weight), parent_point)
            self._add_force_at_point(child_actor, -force * float(child_weight), child_point)

    def _apply_active_connection_magnetic_forces(self) -> None:
        if self.active_magnet_force_limit <= 0.0:
            return
        locked_by_role = self._locked_by_role()
        for active_connection in self.active_connections:
            if not active_connection.active:
                continue
            connection = active_connection.connection
            parent = locked_by_role.get(connection.parent)
            child = locked_by_role.get(connection.child)
            if parent is None or child is None:
                continue
            parent_edge = self._edge_spec(parent.role, connection.parent_edge, connection.parent_lane)
            child_edge = self._edge_spec(child.role, connection.child_edge, connection.child_lane)
            parent_points, child_points = self._matched_edge_points(parent, parent_edge, child, child_edge)
            parent_world = self._current_world_points_array(parent.actor, parent_points)
            child_world = self._current_world_points_array(child.actor, child_points)
            max_distance = self._max_matched_point_distance(parent_world, child_world)
            active_scale = self._active_error_scale(max_distance)
            if active_scale <= 0.0:
                continue
            parent_weight, child_weight = self._pair_correction_weights(parent.role, child.role)
            for parent_point, child_point in zip(parent_world, child_world):
                delta = child_point - parent_point
                distance = float(np.linalg.norm(delta))
                if distance <= 1e-6:
                    continue
                parent_velocity = self._point_velocity(parent.actor, parent_point)
                child_velocity = self._point_velocity(child.actor, child_point)
                force = (
                    self.active_magnet_stiffness * delta
                    + self.active_magnet_damping * (child_velocity - parent_velocity)
                ) * active_scale
                magnitude = float(np.linalg.norm(force))
                if magnitude <= 1e-6:
                    continue
                limit = self.active_magnet_force_limit * active_scale / max(len(parent_world), 1)
                if magnitude > limit:
                    force = force / magnitude * limit
                force = force.astype(np.float32)
                self._add_force_at_point(parent.actor, force * float(parent_weight), parent_point)
                self._add_force_at_point(child.actor, -force * float(child_weight), child_point)

    def _apply_edge_alignment_torque(
        self,
        parent_actor: Any,
        child_actor: Any,
        parent_world: list[np.ndarray],
        child_world: list[np.ndarray],
        max_distance: float,
        *,
        parent_weight: float = 0.5,
        child_weight: float = 0.5,
        distance_scale_floor: float = 0.0,
        torque_scale: float = 1.0,
    ) -> None:
        if len(parent_world) < 2 or len(child_world) < 2:
            return
        parent_dir = parent_world[1] - parent_world[0]
        child_dir = child_world[1] - child_world[0]
        parent_norm = float(np.linalg.norm(parent_dir))
        child_norm = float(np.linalg.norm(child_dir))
        if parent_norm <= 1e-6 or child_norm <= 1e-6:
            return
        parent_dir = parent_dir / parent_norm
        child_dir = child_dir / child_norm
        axis = np.cross(child_dir, parent_dir)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-6:
            return
        dot = float(np.clip(np.dot(child_dir, parent_dir), -1.0, 1.0))
        angle = float(np.arccos(dot))
        distance_scale = max(self.attract_distance - max_distance, 0.0) / max(self.attract_distance, 1e-6)
        distance_scale = max(distance_scale, max(float(distance_scale_floor), 0.0))
        torque_magnitude = min(
            self.attract_torque_stiffness * angle * distance_scale * max(float(torque_scale), 0.0),
            self.attract_torque_limit,
        )
        if torque_magnitude <= 0.0:
            return
        torque = (axis / axis_norm * torque_magnitude).astype(np.float32)
        self._add_torque(child_actor, torque * float(child_weight))
        self._add_torque(parent_actor, -torque * float(parent_weight))

    def _apply_face_normal_alignment_torque(
        self,
        parent: LockedPanelPose,
        child: LockedPanelPose,
        parent_world: list[np.ndarray],
        child_world: list[np.ndarray],
        max_distance: float,
        target_mode: str | None = None,
        reciprocal_torque: bool = True,
        parent_weight: float = 0.5,
        child_weight: float = 0.5,
        distance_scale_floor: float = 0.0,
        torque_scale: float = 1.0,
    ) -> None:
        if len(parent_world) < 2 or len(child_world) < 2:
            return
        parent_normal = self._world_face_normal(parent.actor, parent.role)
        child_normal = self._world_face_normal(child.actor, child.role)
        parent_normal = _normalize(parent_normal)
        child_normal = _normalize(child_normal)
        edge_tangent = _normalize(parent_world[1] - parent_world[0])
        dot = float(np.clip(np.dot(child_normal, parent_normal), -1.0, 1.0))
        if target_mode == "parallel" or (target_mode is None and abs(dot) >= 0.707):
            desired_child_normal = parent_normal if dot >= 0.0 else -parent_normal
        else:
            candidate = np.cross(edge_tangent, parent_normal)
            candidate_norm = float(np.linalg.norm(candidate))
            if candidate_norm <= 1e-6:
                return
            candidate = (candidate / candidate_norm).astype(np.float32)
            desired_child_normal = candidate if float(np.dot(child_normal, candidate)) >= 0.0 else -candidate
        axis = np.cross(child_normal, desired_child_normal)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-6:
            return
        angle = float(np.arccos(np.clip(np.dot(child_normal, desired_child_normal), -1.0, 1.0)))
        distance_scale = max(self.attract_distance - max_distance, 0.0) / max(self.attract_distance, 1e-6)
        distance_scale = max(distance_scale, max(float(distance_scale_floor), 0.0))
        torque_magnitude = min(
            self.attract_normal_torque_stiffness * angle * distance_scale * max(float(torque_scale), 0.0),
            self.attract_normal_torque_limit,
        )
        if torque_magnitude <= 0.0:
            return
        torque = (axis / axis_norm * torque_magnitude).astype(np.float32)
        self._add_torque(child.actor, torque * float(child_weight))
        if reciprocal_torque:
            self._add_torque(parent.actor, -torque * float(parent_weight))

    def _apply_active_connection_normal_torques(self) -> None:
        locked_by_role = self._locked_by_role()
        for active_connection in self.active_connections:
            if not active_connection.active:
                continue
            connection = active_connection.connection
            target_mode = self._target_normal_mode(connection.mode)
            if target_mode is None:
                continue
            parent = locked_by_role.get(connection.parent)
            child = locked_by_role.get(connection.child)
            if parent is None or child is None:
                continue
            parent_edge = self._edge_spec(parent.role, connection.parent_edge, connection.parent_lane)
            child_edge = self._edge_spec(child.role, connection.child_edge, connection.child_lane)
            parent_points, child_points = self._matched_edge_points(parent, parent_edge, child, child_edge)
            parent_world = self._current_world_points_array(parent.actor, parent_points)
            child_world = self._current_world_points_array(child.actor, child_points)
            max_distance = self._max_matched_point_distance(parent_world, child_world)
            if self.active_torque_max_point_error > 0.0 and max_distance > self.active_torque_max_point_error:
                continue
            parent_count = self._active_connection_count(parent.role)
            child_count = self._active_connection_count(child.role)
            parent_weight, child_weight = self._pair_correction_weights(parent.role, child.role)
            ramp = self._active_torque_ramp(active_connection)
            edge_torque_scale = self.active_edge_torque_scale * ramp
            edge_distance_scale_floor = self.active_edge_torque_min_scale * ramp
            normal_torque_scale = 1.0 + (self.active_normal_torque_scale - 1.0) * ramp
            normal_distance_scale_floor = self.active_normal_torque_min_scale * ramp
            if parent_count < child_count:
                self._apply_edge_alignment_torque(
                    child.actor,
                    parent.actor,
                    child_world,
                    parent_world,
                    min(max_distance, self.attach_distance),
                    parent_weight=child_weight,
                    child_weight=parent_weight,
                    distance_scale_floor=edge_distance_scale_floor,
                    torque_scale=edge_torque_scale,
                )
                self._apply_face_normal_alignment_torque(
                    child,
                    parent,
                    child_world,
                    parent_world,
                    min(max_distance, self.attach_distance),
                    target_mode=target_mode,
                    parent_weight=child_weight,
                    child_weight=parent_weight,
                    distance_scale_floor=normal_distance_scale_floor,
                    torque_scale=normal_torque_scale,
                )
                continue
            self._apply_edge_alignment_torque(
                parent.actor,
                child.actor,
                parent_world,
                child_world,
                min(max_distance, self.attach_distance),
                parent_weight=parent_weight,
                child_weight=child_weight,
                distance_scale_floor=edge_distance_scale_floor,
                torque_scale=edge_torque_scale,
            )
            self._apply_face_normal_alignment_torque(
                parent,
                child,
                parent_world,
                child_world,
                min(max_distance, self.attach_distance),
                target_mode=target_mode,
                parent_weight=parent_weight,
                child_weight=child_weight,
                distance_scale_floor=normal_distance_scale_floor,
                torque_scale=normal_torque_scale,
            )

    def _apply_active_connection_upright_torques(self) -> None:
        locked_by_role = self._locked_by_role()
        world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        for active_connection in self.active_connections:
            if not active_connection.active:
                continue
            connection = active_connection.connection
            if connection.mode != "single_support_base":
                continue
            parent = locked_by_role.get(connection.parent)
            child = locked_by_role.get(connection.child)
            if parent is None or child is None:
                continue
            if "triangle" not in child.role and "roof" not in child.role:
                continue
            parent_edge = self._edge_spec(parent.role, connection.parent_edge, connection.parent_lane)
            parent_world = self._current_world_points_array(parent.actor, [parent_edge.start, parent_edge.end])
            edge_tangent = parent_world[1] - parent_world[0]
            edge_norm = float(np.linalg.norm(edge_tangent))
            if edge_norm <= 1e-6:
                continue
            edge_tangent = (edge_tangent / edge_norm).astype(np.float32)
            desired_apex = world_up - edge_tangent * float(np.dot(world_up, edge_tangent))
            desired_norm = float(np.linalg.norm(desired_apex))
            if desired_norm <= 1e-6:
                continue
            desired_apex = (desired_apex / desired_norm).astype(np.float32)
            child_apex = quat2mat(self._actor_quaternion(child.actor)) @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
            child_apex = _normalize(child_apex.astype(np.float32))
            child_projected = child_apex - edge_tangent * float(np.dot(child_apex, edge_tangent))
            child_projected_norm = float(np.linalg.norm(child_projected))
            if child_projected_norm <= 1e-6:
                continue
            child_projected = (child_projected / child_projected_norm).astype(np.float32)
            signed_sin = float(np.dot(edge_tangent, np.cross(child_projected, desired_apex)))
            signed_cos = float(np.clip(np.dot(child_projected, desired_apex), -1.0, 1.0))
            angle = float(np.arctan2(signed_sin, signed_cos))
            relative_angular_velocity_along_edge = float(
                np.dot(self._angular_velocity(child.actor) - self._angular_velocity(parent.actor), edge_tangent)
            )
            torque_scalar = (
                self.attract_normal_torque_stiffness * angle
                - self.upright_torque_damping * relative_angular_velocity_along_edge
            )
            torque_scalar = float(
                np.clip(
                    torque_scalar,
                    -self.attract_normal_torque_limit,
                    self.attract_normal_torque_limit,
                )
            )
            if abs(torque_scalar) <= 1e-7:
                continue
            torque = (edge_tangent * torque_scalar).astype(np.float32)
            parent_weight, child_weight = self._pair_correction_weights(parent.role, child.role)
            self._add_torque(child.actor, torque * float(child_weight))
            self._add_torque(parent.actor, -torque * float(parent_weight))

    def _target_normal_mode(self, connection_mode: str) -> str | None:
        if connection_mode in {"right_angle", "floor_support", "reattach_perpendicular", "direct_vertical_snap", "upright_square_roof_snap"}:
            return "perpendicular"
        if connection_mode in {"reattach_parallel", "single_support_base", "roof_base_snap"}:
            return "parallel"
        return None

    def _reattach_mode(self, parent: LockedPanelPose, child: LockedPanelPose) -> str:
        parent_normal = self._world_face_normal(parent.actor, parent.role)
        child_normal = self._world_face_normal(child.actor, child.role)
        dot = float(np.clip(np.dot(parent_normal, child_normal), -1.0, 1.0))
        if abs(dot) >= 0.707:
            return "reattach_parallel"
        return "reattach_perpendicular"

    def _dynamic_connection_mode(
        self,
        parent_role: str,
        parent_edge_name: str,
        child_role: str,
        child_edge_name: str,
        parent: LockedPanelPose,
        child: LockedPanelPose,
    ) -> str:
        if child_edge_name == "base_edge" and ("triangle" in child_role or "roof" in child_role):
            return "single_support_base"
        if parent_edge_name == "base_edge" and ("triangle" in parent_role or "roof" in parent_role):
            return "single_support_base"
        return self._reattach_mode(parent, child)

    def _has_active_pair_connection(self, role_a: str, role_b: str) -> bool:
        target = frozenset([role_a, role_b])
        for active_connection in self.active_connections:
            if not active_connection.active:
                continue
            connection = active_connection.connection
            if frozenset([connection.parent, connection.child]) == target:
                return True
        return False

    def _edge_allowed(self, role: str, edge_name: str) -> bool:
        return True

    def _connection_allowed(self, connection: MagneticConnection) -> bool:
        if self.allowed_connection_keys is not None and self._connection_key(connection) not in self.allowed_connection_keys:
            return False
        if connection.mode.startswith("reattach"):
            if self._has_defined_physical_connection(connection):
                return False
            if self._has_defined_physical_pair_connection(connection):
                return False
        return True

    def _has_defined_physical_connection(self, connection: MagneticConnection) -> bool:
        target = frozenset(
            [
                (connection.parent, connection.parent_edge),
                (connection.child, connection.child_edge),
            ]
        )
        for item in self.connections:
            if item is connection:
                continue
            if item.mode.startswith("reattach"):
                continue
            if frozenset([(item.parent, item.parent_edge), (item.child, item.child_edge)]) == target:
                return True
        return False

    def _has_defined_physical_pair_connection(self, connection: MagneticConnection) -> bool:
        target = frozenset([connection.parent, connection.child])
        for item in self.connections:
            if item is connection:
                continue
            if item.mode.startswith("reattach"):
                continue
            if frozenset([item.parent, item.child]) == target:
                return True
        return False

    def _dynamic_edge_pair_allowed(
        self,
        connection: MagneticConnection,
        parent: LockedPanelPose,
        child: LockedPanelPose,
        parent_world: list[np.ndarray],
        child_world: list[np.ndarray],
    ) -> bool:
        if len(parent_world) < 2 or len(child_world) < 2:
            return False
        parent_dir = parent_world[-1] - parent_world[0]
        child_dir = child_world[-1] - child_world[0]
        parent_norm = float(np.linalg.norm(parent_dir))
        child_norm = float(np.linalg.norm(child_dir))
        if parent_norm <= 1e-6 or child_norm <= 1e-6:
            return False
        parent_dir = parent_dir / parent_norm
        child_dir = child_dir / child_norm
        if abs(float(np.dot(parent_dir, child_dir))) < 0.82:
            return False
        parent_normal = self._world_face_normal(parent.actor, parent.role)
        child_normal = self._world_face_normal(child.actor, child.role)
        normal_dot = abs(float(np.clip(np.dot(parent_normal, child_normal), -1.0, 1.0)))
        target_mode = self._target_normal_mode(connection.mode)
        if target_mode == "parallel" and normal_dot < 0.50:
            return False
        if target_mode == "perpendicular" and normal_dot > 0.70:
            return False
        return True

    def _add_force_at_point(self, actor: Any, force: np.ndarray, point: np.ndarray) -> None:
        body = self._body(actor)
        if body is None:
            return
        if getattr(actor, "px_body_type", None) != "dynamic":
            return
        body.add_force_at_point(force.astype(np.float32), point.astype(np.float32))

    def _add_torque(self, actor: Any, torque: np.ndarray) -> None:
        body = self._body(actor)
        if body is None:
            return
        if getattr(actor, "px_body_type", None) != "dynamic":
            return
        body.add_force_torque(np.zeros(3, dtype=np.float32), torque.astype(np.float32))

    def _drop_underconstrained_pieces(self) -> None:
        for locked in self.locked_panel_poses:
            role = locked.role
            if role == "floor" or role in self.disabled_roles:
                continue
            if self._active_connection_count(role) >= 2:
                continue
            if self._has_single_supporting_edge(role):
                continue
            if self._has_nearby_attract_candidate(role):
                continue
            self.suspended_roles.add(role)
            for active_connection in self.active_connections:
                if not active_connection.active:
                    continue
                connection = active_connection.connection
                if connection.parent == role or connection.child == role:
                    active_connection.active = False
                    for drive in active_connection.drives:
                        self._disable_drive(drive)

    def _has_nearby_attract_candidate(self, role: str) -> bool:
        locked_by_role = self._locked_by_role()
        current = locked_by_role.get(role)
        if current is None:
            return False
        for connection in self.connections:
            if connection.parent != role and connection.child != role:
                continue
            other_role = connection.child if connection.parent == role else connection.parent
            if other_role in self.disabled_roles:
                continue
            if not self._connection_allowed(connection):
                continue
            if not self._edge_allowed(connection.parent, connection.parent_edge):
                continue
            if not self._edge_allowed(connection.child, connection.child_edge):
                continue
            parent = locked_by_role.get(connection.parent)
            child = locked_by_role.get(connection.child)
            if parent is None or child is None:
                continue
            parent_edge = self._edge_spec(parent.role, connection.parent_edge, connection.parent_lane)
            child_edge = self._edge_spec(child.role, connection.child_edge, connection.child_lane)
            parent_points, child_points = self._matched_edge_points(parent, parent_edge, child, child_edge)
            parent_world = self._current_world_points_array(parent.actor, parent_points)
            child_world = self._current_world_points_array(child.actor, child_points)
            if self._min_pairwise_point_distance(list(parent_world), list(child_world)) <= self.attach_distance:
                return True
            if self._max_matched_point_distance(list(parent_world), list(child_world)) <= self.attract_distance:
                return True
        return False

    def _active_connection_count(self, role: str) -> int:
        keys = set()
        for active_connection in self.active_connections:
            if not active_connection.active:
                continue
            connection = active_connection.connection
            if connection.parent == role or connection.child == role:
                keys.add(self._connection_key(connection))
        return len(keys)

    def _has_single_supporting_edge(self, role: str) -> bool:
        support_edges = {"bottom_edge", "base_edge"}
        support_connections = 0
        for active_connection in self.active_connections:
            if not active_connection.active:
                continue
            connection = active_connection.connection
            if connection.parent == role and connection.parent_edge in support_edges:
                support_connections += 1
            if connection.child == role and connection.child_edge in support_edges:
                support_connections += 1
        return support_connections == 1

    def _reattachable_suspended_roles(
        self,
        locked_by_role: dict[str, LockedPanelPose],
        occupied_edges: set[tuple[str, str, str]],
        distance_threshold: float | None = None,
    ) -> set[str]:
        threshold = self.attach_distance if distance_threshold is None else float(distance_threshold)
        candidate_counts = {role: 0 for role in self.suspended_roles}
        support_candidates: set[str] = set()
        roles = list(locked_by_role)
        for i, parent_role in enumerate(roles):
            if parent_role in self.disabled_roles:
                continue
            parent = locked_by_role[parent_role]
            for child_role in roles[i + 1 :]:
                if child_role in self.disabled_roles:
                    continue
                child = locked_by_role[child_role]
                for parent_edge_name in self._edge_names(parent_role):
                    if not self._edge_allowed(parent_role, parent_edge_name):
                        continue
                    parent_edge = self._edge_spec(parent_role, parent_edge_name)
                    for parent_lane in self._dynamic_lane_names(parent_role, parent_edge_name):
                        parent_key = (parent_role, parent_edge_name, parent_lane)
                        if self._edge_key_occupied(occupied_edges, parent_key):
                            continue
                        for child_edge_name in self._edge_names(child_role):
                            if not self._edge_allowed(child_role, child_edge_name):
                                continue
                            child_edge = self._edge_spec(child_role, child_edge_name)
                            if abs(parent_edge.length - child_edge.length) > self.max_length_error:
                                continue
                            for child_lane in self._dynamic_lane_names(child_role, child_edge_name):
                                child_key = (child_role, child_edge_name, child_lane)
                                if self._edge_key_occupied(occupied_edges, child_key):
                                    continue
                                parent_points, child_points = self._matched_edge_points(
                                    parent,
                                    parent_edge,
                                    child,
                                    child_edge,
                                )
                                parent_world = self._current_world_points_array(parent.actor, parent_points)
                                child_world = self._current_world_points_array(child.actor, child_points)
                                max_distance = self._max_matched_point_distance(parent_world, child_world)
                                min_pairwise_distance = self._min_pairwise_point_distance(parent_world, child_world)
                                if max_distance > threshold and min_pairwise_distance > self.attach_distance:
                                    continue
                                connection = MagneticConnection(
                                    parent_role,
                                    parent_edge_name,
                                    child_role,
                                    child_edge_name,
                                    self._dynamic_connection_mode(parent_role, parent_edge_name, child_role, child_edge_name, parent, child),
                                    parent_lane,
                                    child_lane,
                                )
                                if not self._dynamic_edge_pair_allowed(connection, parent, child, parent_world, child_world):
                                    continue
                                if not self._connection_allowed(connection):
                                    continue
                                if parent_role in candidate_counts:
                                    candidate_counts[parent_role] += 1
                                    if parent_edge_name in {"bottom_edge", "base_edge"}:
                                        support_candidates.add(parent_role)
                                if child_role in candidate_counts:
                                    candidate_counts[child_role] += 1
                                    if child_edge_name in {"bottom_edge", "base_edge"}:
                                        support_candidates.add(child_role)
        return {role for role, count in candidate_counts.items() if count >= 2 or role in support_candidates}

    def _create_runtime_edge_connection(
        self,
        scene: Any,
        parent: LockedPanelPose,
        child: LockedPanelPose,
        connection: MagneticConnection,
    ) -> None:
        if not self._connection_allowed(connection):
            return
        if self._find_active_connection(connection) is not None:
            return
        reusable = self._find_inactive_connection(connection)
        if reusable is not None:
            reusable.active = True
            reusable.active_steps = 0
            for drive in reusable.drives:
                self._configure_drive(drive)
            return
        drives = []
        parent_points, child_points = self._matched_connection_points(
            connection,
            parent,
            child,
            points_per_edge=1 if self.magnet_mode == "edge_drive" else 2,
            use_locked_pose=False,
        )
        parent_pose_world = sapien.Pose(self._actor_position(parent.actor).tolist(), self._actor_quaternion(parent.actor).tolist())
        child_pose_world = sapien.Pose(self._actor_position(child.actor).tolist(), self._actor_quaternion(child.actor).tolist())
        for parent_point, child_point in zip(parent_points, child_points):
            pose_in_parent = sapien.Pose(p=parent_point.tolist())
            desired_world = parent_pose_world * pose_in_parent
            pose_in_child = child_pose_world.inv() * desired_world
            pose_in_child = sapien.Pose(p=child_point.tolist(), q=pose_in_child.q)
            drive = scene.create_drive(parent.actor, pose_in_parent, child.actor, pose_in_child)
            self._configure_drive(drive)
            self.drives.append(drive)
            drives.append(drive)
        self.connections.append(connection)
        self.active_connections.append(
            ActiveMagneticConnection(
                connection=connection,
                drives=drives,
                active=True,
            )
        )

    def _find_active_connection(self, connection: MagneticConnection) -> ActiveMagneticConnection | None:
        target_key = self._connection_key(connection)
        for active_connection in self.active_connections:
            if not active_connection.active:
                continue
            if self._connection_key(active_connection.connection) == target_key:
                return active_connection
        return None

    def _find_inactive_connection(self, connection: MagneticConnection) -> ActiveMagneticConnection | None:
        target_key = self._connection_key(connection)
        for active_connection in self.active_connections:
            if active_connection.active:
                continue
            if self._connection_key(active_connection.connection) == target_key:
                return active_connection
        return None

    def _connection_key(self, connection: MagneticConnection) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
        left = (connection.parent, connection.parent_edge, connection.parent_lane)
        right = (connection.child, connection.child_edge, connection.child_lane)
        return tuple(sorted([left, right]))

    def _normal_alignment_error(
        self,
        connection: MagneticConnection,
        parent: LockedPanelPose,
        child: LockedPanelPose,
        parent_world: list[np.ndarray],
    ) -> tuple[str | None, float | None]:
        target_mode = self._target_normal_mode(connection.mode)
        if target_mode is None or len(parent_world) < 2:
            return target_mode, None
        parent_normal = _normalize(self._world_face_normal(parent.actor, parent.role))
        child_normal = _normalize(self._world_face_normal(child.actor, child.role))
        dot = float(np.clip(np.dot(child_normal, parent_normal), -1.0, 1.0))
        if target_mode == "parallel":
            angle = float(np.arccos(abs(dot)))
            return target_mode, float(np.degrees(angle))
        edge_tangent = _normalize(parent_world[1] - parent_world[0])
        candidate = np.cross(edge_tangent, parent_normal)
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm <= 1e-6:
            return target_mode, None
        candidate = (candidate / candidate_norm).astype(np.float32)
        desired_child_normal = candidate if float(np.dot(child_normal, candidate)) >= 0.0 else -candidate
        angle = float(np.arccos(np.clip(np.dot(child_normal, desired_child_normal), -1.0, 1.0)))
        return target_mode, float(np.degrees(angle))

    def _occupied_edges(self) -> set[tuple[str, str, str]]:
        occupied = set()
        for active_connection in self.active_connections:
            if not active_connection.active:
                continue
            connection = active_connection.connection
            self._add_occupied_edge(occupied, (connection.parent, connection.parent_edge, connection.parent_lane))
            self._add_occupied_edge(occupied, (connection.child, connection.child_edge, connection.child_lane))
        return occupied

    def _add_occupied_edge(self, occupied: set[tuple[str, str, str]], key: tuple[str, str, str]) -> None:
        occupied.add(key)

    def _edge_key_occupied(self, occupied: set[tuple[str, str, str]], key: tuple[str, str, str]) -> bool:
        role, edge_name, _lane = key
        return any(item_role == role and item_edge == edge_name for item_role, item_edge, _ in occupied)

    def deactivate_piece(self, role: str) -> None:
        self.disabled_roles.add(role)
        self.suspended_roles.discard(role)
        for active_connection in self.active_connections:
            connection = active_connection.connection
            if not active_connection.active:
                continue
            if connection.parent == role or connection.child == role:
                active_connection.active = False
                for drive in active_connection.drives:
                    self._disable_drive(drive)

    def _connection_point_error(self, connection: MagneticConnection) -> float:
        locked_by_role = self._locked_by_role()
        parent = locked_by_role.get(connection.parent)
        child = locked_by_role.get(connection.child)
        if parent is None or child is None:
            return 0.0
        parent_points, child_points = self._matched_connection_points(
            connection,
            parent,
            child,
            points_per_edge=1 if self.magnet_mode == "edge_drive" else 2,
            use_locked_pose=False,
        )
        parent_world = self._current_world_points_array(parent.actor, parent_points)
        child_world = self._current_world_points_array(child.actor, child_points)
        return self._max_matched_point_distance(parent_world, child_world)

    def connection_errors(self) -> dict[str, Any]:
        locked_by_role = self._locked_by_role()
        errors = []
        for active_connection in self.active_connections:
            if not active_connection.active:
                continue
            connection = active_connection.connection
            parent = locked_by_role.get(connection.parent)
            child = locked_by_role.get(connection.child)
            if parent is None or child is None:
                continue
            parent_points, child_points = self._matched_connection_points(
                connection,
                parent,
                child,
                points_per_edge=1 if self.magnet_mode == "edge_drive" else 2,
                use_locked_pose=False,
            )
            parent_world = self._current_world_points_array(parent.actor, parent_points)
            child_world = self._current_world_points_array(child.actor, child_points)
            point_errors = self._matched_point_distances(parent_world, child_world).astype(np.float32).tolist()
            parent_edge = self._edge_spec(parent.role, connection.parent_edge, connection.parent_lane)
            child_edge = self._edge_spec(child.role, connection.child_edge, connection.child_lane)
            normal_target_mode, normal_angle_error_deg = self._normal_alignment_error(
                connection,
                parent,
                child,
                parent_world,
            )
            edge_angle_error_deg = None
            if len(parent_world) >= 2 and len(child_world) >= 2:
                parent_dir = _normalize(parent_world[1] - parent_world[0])
                child_dir = _normalize(child_world[1] - child_world[0])
                edge_angle = float(np.arccos(np.clip(abs(float(np.dot(parent_dir, child_dir))), -1.0, 1.0)))
                edge_angle_error_deg = float(np.degrees(edge_angle))
            errors.append(
                {
                    "parent": connection.parent,
                    "parent_edge": connection.parent_edge,
                    "child": connection.child,
                    "child_edge": connection.child_edge,
                    "mode": connection.mode,
                    "normal_target_mode": normal_target_mode,
                    "normal_angle_error_deg": normal_angle_error_deg,
                    "edge_angle_error_deg": edge_angle_error_deg,
                    "max_point_error": max(point_errors),
                    "mean_point_error": float(np.mean(point_errors)),
                    "length_error": abs(parent_edge.length - child_edge.length),
                }
            )
        if not errors:
            return {
                "connections": [],
                "max_point_error": 0.0,
                "mean_point_error": 0.0,
                "max_normal_angle_error_deg": 0.0,
                "max_edge_angle_error_deg": 0.0,
            }
        normal_errors = [
            float(item["normal_angle_error_deg"])
            for item in errors
            if item.get("normal_angle_error_deg") is not None
        ]
        edge_errors = [
            float(item["edge_angle_error_deg"])
            for item in errors
            if item.get("edge_angle_error_deg") is not None
        ]
        return {
            "connections": errors,
            "max_point_error": max(item["max_point_error"] for item in errors),
            "mean_point_error": float(np.mean([item["mean_point_error"] for item in errors])),
            "max_normal_angle_error_deg": max(normal_errors) if normal_errors else 0.0,
            "max_edge_angle_error_deg": max(edge_errors) if edge_errors else 0.0,
        }

    def apply_soft_magnetic_forces(self) -> None:
        for locked in self.locked_panel_poses:
            body = self._body(locked.actor)
            if body is None:
                continue
            pos = self._actor_position(locked.actor)
            quat = self._actor_quaternion(locked.actor)
            linear_velocity = self._linear_velocity(locked.actor)
            angular_velocity = self._angular_velocity(locked.actor)

            position_error = locked.position - pos
            mass = max(float(getattr(body, "mass", 1.0)), 1e-4)
            force = mass * (
                self.position_stiffness * position_error
                - self.position_damping * linear_velocity
                + np.asarray([0.0, 0.0, 9.81], dtype=np.float32)
            )
            torque = self._orientation_torque(
                current_quat=quat,
                target_quat=locked.quaternion,
                angular_velocity=angular_velocity,
            )
            body.add_force_torque(force.astype(np.float32), torque.astype(np.float32))

    def _orientation_torque(
        self,
        *,
        current_quat: np.ndarray,
        target_quat: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> np.ndarray:
        current = current_quat.astype(np.float32)
        target = target_quat.astype(np.float32)
        if float(np.dot(current, target)) < 0.0:
            target = -target
        error = _quat_multiply(target, _quat_conjugate(current))
        if error[0] < 0.0:
            error = -error
        axis_error = 2.0 * error[1:4]
        return self.rotation_stiffness * axis_error - self.rotation_damping * angular_velocity

    def _body(self, actor: Any) -> Any | None:
        bodies = getattr(actor, "_bodies", None)
        if not bodies:
            return None
        return bodies[0]

    def _actor_position(self, actor: Any) -> np.ndarray:
        return self._actor_pose_data(actor)[0]

    def _actor_quaternion(self, actor: Any) -> np.ndarray:
        return self._actor_pose_data(actor)[1]

    def _actor_pose_data(self, actor: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._step_pose_cache.get(id(actor))
        if cached is not None:
            return cached
        position = self._to_numpy(actor.pose.p).reshape(-1, 3)[0].astype(np.float32)
        quat = self._to_numpy(actor.pose.q).reshape(-1, 4)[0].astype(np.float32)
        norm = float(np.linalg.norm(quat))
        if norm <= 1e-8:
            quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            quat = quat / norm
        rotation = quat2mat(quat).astype(np.float32)
        return position, quat, rotation

    def _transform_local_points(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        local_points: list[np.ndarray],
    ) -> np.ndarray:
        if len(local_points) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        points = np.asarray(local_points, dtype=np.float32)
        rotation = quat2mat(quaternion.astype(np.float32)).astype(np.float32)
        return (points @ rotation.T + position.astype(np.float32)).astype(np.float32)

    def _current_world_points_array(self, actor: Any, local_points: list[np.ndarray]) -> np.ndarray:
        position, _quat, rotation = self._actor_pose_data(actor)
        if len(local_points) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        points = np.asarray(local_points, dtype=np.float32)
        return (points @ rotation.T + position).astype(np.float32)

    def _current_world_points(self, actor: Any, local_points: list[np.ndarray]) -> list[np.ndarray]:
        return [np.asarray(point, dtype=np.float32) for point in self._current_world_points_array(actor, local_points)]

    def _world_face_normal(self, actor: Any, role: str) -> np.ndarray:
        local = np.asarray([0.0, 1.0, 0.0] if ("roof" in role or "triangle" in role) else [0.0, 0.0, 1.0], dtype=np.float32)
        _position, _quat, rotation = self._actor_pose_data(actor)
        normal = rotation @ local
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-8:
            return local
        return (normal / norm).astype(np.float32)

    def _point_velocity(self, actor: Any, point: np.ndarray) -> np.ndarray:
        linear = self._linear_velocity(actor)
        angular = self._angular_velocity(actor)
        radius = point.astype(np.float32) - self._actor_position(actor)
        return (linear + np.cross(angular, radius)).astype(np.float32)

    def _edge_spec(self, role: str, edge_name: str, lane_name: str = "rim") -> EdgeSpec:
        role_kind = "triangle" if ("roof" in role or "triangle" in role) else "square"
        key = (role_kind, edge_name, lane_name)
        cached = self._edge_spec_cache.get(key)
        if cached is not None:
            return cached
        if "roof" in role or "triangle" in role:
            spec = self._triangle_edge_spec(edge_name, lane_name)
        else:
            spec = self._square_edge_spec(edge_name, lane_name)
        self._edge_spec_cache[key] = spec
        return spec

    def _edge_names(self, role: str) -> list[str]:
        if "roof" in role or "triangle" in role:
            return ["base_edge", "left_slope_edge", "right_slope_edge"]
        return ["left_edge", "right_edge", "top_edge", "bottom_edge"]

    def _lane_names(self, role: str, edge_name: str) -> list[str]:
        return ["face_a", "face_b", "rim"]

    def _dynamic_lane_names(self, role: str, edge_name: str) -> list[str]:
        return ["rim"]

    def _square_edge_spec(self, edge_name: str, lane_name: str = "rim") -> EdgeSpec:
        half = self.plate_size / 2.0
        lane_offsets = {
            "center": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            "rim": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            "face_a": np.asarray([0.0, 0.0, -self.plate_thickness / 2.0], dtype=np.float32),
            "face_b": np.asarray([0.0, 0.0, self.plate_thickness / 2.0], dtype=np.float32),
        }
        if lane_name not in lane_offsets:
            raise KeyError(f"unknown square magnetic lane: {lane_name}")
        aliases = {
            "back_edge": "top_edge",
            "front_edge": "bottom_edge",
        }
        name = aliases.get(edge_name, edge_name)
        specs = {
            "right_edge": ([half, -half, 0.0], [half, half, 0.0]),
            "left_edge": ([-half, half, 0.0], [-half, -half, 0.0]),
            "top_edge": ([-half, half, 0.0], [half, half, 0.0]),
            "bottom_edge": ([half, -half, 0.0], [-half, -half, 0.0]),
        }
        if name not in specs:
            raise KeyError(f"unknown square magnetic edge: {edge_name}")
        start, end = specs[name]
        offset = lane_offsets[lane_name]
        return EdgeSpec(
            start=np.asarray(start, dtype=np.float32) + offset,
            end=np.asarray(end, dtype=np.float32) + offset,
        )

    def _triangle_edge_spec(self, edge_name: str, lane_name: str = "rim") -> EdgeSpec:
        half_width = self.triangle_width / 2.0
        lane_offsets = {
            "center": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            "rim": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            "face_a": np.asarray([0.0, -self.triangle_thickness / 2.0, 0.0], dtype=np.float32),
            "face_b": np.asarray([0.0, self.triangle_thickness / 2.0, 0.0], dtype=np.float32),
        }
        if lane_name not in lane_offsets:
            raise KeyError(f"unknown triangular magnetic lane: {lane_name}")
        specs = {
            "base_edge": ([-half_width, 0.0, 0.0], [half_width, 0.0, 0.0]),
            "left_slope_edge": ([-half_width, 0.0, 0.0], [0.0, 0.0, self.triangle_height]),
            "right_slope_edge": ([half_width, 0.0, 0.0], [0.0, 0.0, self.triangle_height]),
        }
        if edge_name not in specs:
            raise KeyError(f"unknown triangular magnetic edge: {edge_name}")
        start, end = specs[edge_name]
        offset = lane_offsets[lane_name]
        return EdgeSpec(
            start=np.asarray(start, dtype=np.float32) + offset,
            end=np.asarray(end, dtype=np.float32) + offset,
        )

    def _linear_velocity(self, actor: Any) -> np.ndarray:
        if hasattr(actor, "get_linear_velocity"):
            return self._to_numpy(actor.get_linear_velocity()).reshape(-1, 3)[0].astype(np.float32)
        body = self._body(actor)
        if body is None:
            return np.zeros(3, dtype=np.float32)
        return self._to_numpy(body.get_linear_velocity()).reshape(3).astype(np.float32)

    def _angular_velocity(self, actor: Any) -> np.ndarray:
        if hasattr(actor, "get_angular_velocity"):
            return self._to_numpy(actor.get_angular_velocity()).reshape(-1, 3)[0].astype(np.float32)
        body = self._body(actor)
        if body is None:
            return np.zeros(3, dtype=np.float32)
        return self._to_numpy(body.get_angular_velocity()).reshape(3).astype(np.float32)

    def _set_velocity(self, actor: Any, linear: np.ndarray, angular: np.ndarray) -> None:
        if getattr(actor, "px_body_type", None) != "dynamic":
            return
        actor.set_linear_velocity(linear)
        actor.set_angular_velocity(angular)

    def _to_numpy(self, value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _refresh_locked_panel_map(self) -> None:
        self._locked_panel_by_role = {locked.role: locked for locked in self.locked_panel_poses}

    def _locked_by_role(self) -> dict[str, LockedPanelPose]:
        if not self._locked_panel_by_role:
            self._refresh_locked_panel_map()
        return self._locked_panel_by_role

    def report(self) -> dict[str, Any]:
        active_connections = [
            item.connection
            for item in self.active_connections
            if item.active
        ]
        return {
            "locked_panel_count": len(self.locked_panel_poses),
            "magnet_mode": self.magnet_mode,
            "position_stiffness": self.position_stiffness,
            "rotation_stiffness": self.rotation_stiffness,
            "drive_count": len(self.drives),
            "active_connection_count": len({self._connection_key(item) for item in active_connections}),
            "drive_force_limit": self.drive_force_limit,
            "detach_distance": self.detach_distance,
            "base_connection_detach_distance": self.base_connection_detach_distance,
            "dynamic_update_period": self.dynamic_update_period,
            "active_edge_torque_scale": self.active_edge_torque_scale,
            "active_edge_torque_min_scale": self.active_edge_torque_min_scale,
            "active_normal_torque_scale": self.active_normal_torque_scale,
            "active_normal_torque_min_scale": self.active_normal_torque_min_scale,
            "active_torque_delay_steps": self.active_torque_delay_steps,
            "active_torque_ramp_steps": self.active_torque_ramp_steps,
            "active_torque_max_point_error": self.active_torque_max_point_error,
            "floor_support_score": self.floor_support_score,
            "support_connection_score_scale": self.support_connection_score_scale,
            "multi_connection_support_bonus": self.multi_connection_support_bonus,
            "edge_sample_half_span": self.edge_sample_half_span,
            "connection_edge_sample_half_span": self.connection_edge_sample_half_span,
            "edge_sample_offsets": list(self._expanded_edge_sample_offsets(self.edge_sample_offsets, self.plate_size)),
            "connection_edge_sample_offsets": list(
                self._expanded_edge_sample_offsets(self.connection_edge_sample_offsets, self.plate_size)
            ),
            "connection_error": self.connection_errors(),
            "connections": [
                {
                    "parent": item.parent,
                    "parent_edge": item.parent_edge,
                    "child": item.child,
                    "child_edge": item.child_edge,
                    "mode": item.mode,
                    "parent_lane": item.parent_lane,
                    "child_lane": item.child_lane,
                    "active_steps": next(
                        (
                            int(getattr(active_connection, "active_steps", 0))
                            for active_connection in self.active_connections
                            if active_connection.active and self._connection_key(active_connection.connection) == self._connection_key(item)
                        ),
                        0,
                    ),
                }
                for item in active_connections
            ],
        }


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    return np.asarray([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float32)


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )
