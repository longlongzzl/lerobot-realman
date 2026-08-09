"""Small execution adapters for the new pick-place coordinator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rm75_app.planning.contracts import JointTrajectory


def downsample_joint_path(positions: np.ndarray, max_points: int) -> np.ndarray:
    path = np.asarray(positions, dtype=np.float64)
    if path.ndim != 2:
        raise ValueError("joint path must be two-dimensional")
    if len(path) <= max_points:
        return path
    indices = np.unique(np.linspace(0, len(path) - 1, max_points).round().astype(int))
    return path[indices]


class RecordingTrajectoryExecutor:
    """Headless executor that also emits portable per-stage trajectory arrays.

    The JSON manifest stays small and inspectable; dense positions/timestamps are
    stored in NPZ files beside it so a simulator or robot process can replay the
    result without importing the planning backend.
    """

    def __init__(self, output_path: str | Path | None = None):
        self.output_path = None if output_path is None else Path(output_path).expanduser().resolve()
        self.events: list[dict[str, Any]] = []

    def _save(self) -> None:
        if self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.events, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.output_path)

    def execute_trajectory(self, stage: str, trajectory: JointTrajectory) -> None:
        trajectory_file = None
        if self.output_path is not None:
            safe_stage = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(stage))
            trajectory_path = self.output_path.parent / f"{len(self.events):02d}_{safe_stage}.npz"
            payload: dict[str, np.ndarray] = {
                "positions": np.asarray(trajectory.positions, dtype=np.float64),
                "joint_names": np.asarray(trajectory.joint_names, dtype=np.str_),
            }
            if trajectory.dt is not None:
                payload["dt"] = np.asarray(trajectory.dt, dtype=np.float64)
            np.savez_compressed(trajectory_path, **payload)
            trajectory_file = trajectory_path.name
        self.events.append(
            {
                "type": "trajectory",
                "stage": str(stage),
                "joint_names": list(trajectory.joint_names),
                "points": int(len(trajectory.positions)),
                "start": trajectory.positions[0].tolist(),
                "end": trajectory.positions[-1].tolist(),
                "trajectory_file": trajectory_file,
            }
        )
        self._save()

    def set_gripper(self, closed: bool) -> None:
        self.events.append({"type": "gripper", "closed": bool(closed)})
        self._save()


class ManiSkillTrajectoryExecutor:
    """Execute planned joints through an existing RM75 ManiSkill demo instance."""

    def __init__(
        self,
        demo: Any,
        *,
        gripper_open: float = -1.0,
        gripper_closed: float = 1.0,
        gripper_steps: int = 20,
        max_path_points: int = 400,
    ):
        self.demo = demo
        self.gripper_open = float(gripper_open)
        self.gripper_closed = float(gripper_closed)
        self.gripper_steps = int(gripper_steps)
        self.max_path_points = int(max_path_points)
        self._gripper_value = self.gripper_open

    def execute_trajectory(self, stage: str, trajectory: JointTrajectory) -> None:
        for position in downsample_joint_path(trajectory.positions, self.max_path_points):
            action = self.demo.compose_action(position, self._gripper_value)
            self.demo.step_and_render(action, tag=str(stage))

    def set_gripper(self, closed: bool) -> None:
        self._gripper_value = self.gripper_closed if closed else self.gripper_open
        self.demo.hold_current_and_set_gripper(self._gripper_value, steps=self.gripper_steps)
