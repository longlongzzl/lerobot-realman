from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrajectoryMetrics:
    curobo_success: bool
    robot_collision: bool = False
    held_object_displacement_m: float = 0.0
    release_edge_error_m: float = 0.0
    release_center_error_m: float = 0.0
    release_edge_angle_deg: float = 0.0
    release_normal_angle_deg: float = 0.0
    magnetic_connection_active: bool = False
    stable_after_release: bool = False
    built_structure_motion_m: float = 0.0
    path_joint_distance: float = 0.0
    execution_steps: int = 0

    @classmethod
    def from_report(cls, report: dict[str, Any]) -> TrajectoryMetrics:
        alignment = report.get("alignment") if isinstance(report.get("alignment"), dict) else {}
        active = report.get("active_connection") if isinstance(report.get("active_connection"), dict) else {}
        return cls(
            curobo_success=bool(report.get("curobo_success", report.get("success", False))),
            robot_collision=bool(report.get("robot_collision", False)),
            held_object_displacement_m=float(report.get("held_object_displacement_m", 0.0) or 0.0),
            release_edge_error_m=float(alignment.get("max_point_error_m", report.get("release_edge_error_m", 0.0)) or 0.0),
            release_center_error_m=float(alignment.get("center_error_m", report.get("release_center_error_m", 0.0)) or 0.0),
            release_edge_angle_deg=float(alignment.get("edge_parallel_error_deg", report.get("release_edge_angle_deg", 0.0)) or 0.0),
            release_normal_angle_deg=float(report.get("release_normal_angle_deg", 0.0) or 0.0),
            magnetic_connection_active=bool(active.get("active_count", 0) or report.get("magnetic_connection_active", False)),
            stable_after_release=bool(report.get("stable_after_release", False)),
            built_structure_motion_m=float(report.get("built_structure_motion_m", 0.0) or 0.0),
            path_joint_distance=float(report.get("path_joint_distance", 0.0) or 0.0),
            execution_steps=int(report.get("execution_steps", 0) or 0),
        )


@dataclass(frozen=True)
class TrajectoryScoreWeights:
    curobo_failure: float = -10000.0
    robot_collision: float = -8000.0
    held_object_displacement_per_m: float = -30000.0
    release_edge_error_per_m: float = -50000.0
    release_center_error_per_m: float = -25000.0
    release_edge_angle_per_deg: float = -50.0
    release_normal_angle_per_deg: float = -40.0
    magnetic_connection: float = 3000.0
    stable_after_release: float = 5000.0
    built_structure_motion_per_m: float = -60000.0
    path_joint_distance_per_rad: float = -10.0
    execution_step_penalty: float = -0.25


def score_trajectory(metrics: TrajectoryMetrics, weights: TrajectoryScoreWeights | None = None) -> dict[str, Any]:
    weights = weights or TrajectoryScoreWeights()
    components: dict[str, float] = {}
    if not metrics.curobo_success:
        components["curobo_failure"] = weights.curobo_failure
    if metrics.robot_collision:
        components["robot_collision"] = weights.robot_collision
    components["held_object_displacement"] = weights.held_object_displacement_per_m * max(metrics.held_object_displacement_m, 0.0)
    components["release_edge_error"] = weights.release_edge_error_per_m * max(metrics.release_edge_error_m, 0.0)
    components["release_center_error"] = weights.release_center_error_per_m * max(metrics.release_center_error_m, 0.0)
    components["release_edge_angle"] = weights.release_edge_angle_per_deg * abs(metrics.release_edge_angle_deg)
    components["release_normal_angle"] = weights.release_normal_angle_per_deg * abs(metrics.release_normal_angle_deg)
    if metrics.magnetic_connection_active:
        components["magnetic_connection"] = weights.magnetic_connection
    if metrics.stable_after_release:
        components["stable_after_release"] = weights.stable_after_release
    components["built_structure_motion"] = weights.built_structure_motion_per_m * max(metrics.built_structure_motion_m, 0.0)
    components["path_joint_distance"] = weights.path_joint_distance_per_rad * max(metrics.path_joint_distance, 0.0)
    components["execution_steps"] = weights.execution_step_penalty * max(metrics.execution_steps, 0)
    total = float(sum(components.values()))
    return {
        "score": total,
        "components": components,
        "metrics": {
            "curobo_success": bool(metrics.curobo_success),
            "robot_collision": bool(metrics.robot_collision),
            "held_object_displacement_m": float(metrics.held_object_displacement_m),
            "release_edge_error_m": float(metrics.release_edge_error_m),
            "release_center_error_m": float(metrics.release_center_error_m),
            "release_edge_angle_deg": float(metrics.release_edge_angle_deg),
            "release_normal_angle_deg": float(metrics.release_normal_angle_deg),
            "magnetic_connection_active": bool(metrics.magnetic_connection_active),
            "stable_after_release": bool(metrics.stable_after_release),
            "built_structure_motion_m": float(metrics.built_structure_motion_m),
            "path_joint_distance": float(metrics.path_joint_distance),
            "execution_steps": int(metrics.execution_steps),
        },
    }
