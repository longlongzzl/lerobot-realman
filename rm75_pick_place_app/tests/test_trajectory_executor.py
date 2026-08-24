from __future__ import annotations

import json

import numpy as np

from rm75_app.execution.trajectory_executor import (
    ManiSkillTrajectoryExecutor,
    RecordingTrajectoryExecutor,
    downsample_joint_path,
)
from rm75_app.planning.contracts import JointTrajectory


def test_downsample_preserves_endpoints() -> None:
    path = np.arange(200, dtype=np.float64).reshape(100, 2)
    sampled = downsample_joint_path(path, 9)
    assert len(sampled) == 9
    np.testing.assert_array_equal(sampled[0], path[0])
    np.testing.assert_array_equal(sampled[-1], path[-1])


def test_recording_executor_writes_replay_artifact(tmp_path) -> None:
    output = tmp_path / "execution.json"
    executor = RecordingTrajectoryExecutor(output)
    executor.begin_atom("atom_01")
    executor.execute_trajectory(
        "lift", JointTrajectory(("j1", "j2"), np.asarray([[0.0, 0.0], [1.0, 2.0]]))
    )
    executor.set_gripper(True)
    executor.end_atom("atom_01", success=True)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved[0] == {"type": "atom_start", "atom_id": "atom_01"}
    assert saved[1]["stage"] == "lift"
    assert saved[1]["end"] == [1.0, 2.0]
    trajectory_path = tmp_path / saved[1]["trajectory_file"]
    with np.load(trajectory_path) as trajectory:
        np.testing.assert_array_equal(trajectory["positions"], [[0.0, 0.0], [1.0, 2.0]])
        assert trajectory["joint_names"].tolist() == ["j1", "j2"]
    assert saved[2] == {"type": "gripper", "closed": True}
    assert saved[3] == {"type": "atom_end", "atom_id": "atom_01", "success": True}


def test_maniskill_rm75_gripper_uses_positive_open_negative_closed() -> None:
    class FakeDemo:
        def __init__(self):
            self.gripper_commands = []

        def hold_current_and_set_gripper(self, value, steps):
            self.gripper_commands.append((value, steps))

    demo = FakeDemo()
    executor = ManiSkillTrajectoryExecutor(demo, gripper_steps=7)

    assert executor.gripper_open == 1.0
    assert executor.gripper_closed == -1.0
    executor.set_gripper(True)
    executor.set_gripper(False)

    assert demo.gripper_commands == [(-1.0, 7), (1.0, 7)]
