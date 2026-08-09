from __future__ import annotations

import json

import numpy as np

from rm75_app.execution.trajectory_executor import RecordingTrajectoryExecutor, downsample_joint_path
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
    executor.execute_trajectory(
        "lift", JointTrajectory(("j1", "j2"), np.asarray([[0.0, 0.0], [1.0, 2.0]]))
    )
    executor.set_gripper(True)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved[0]["stage"] == "lift"
    assert saved[0]["end"] == [1.0, 2.0]
    trajectory_path = tmp_path / saved[0]["trajectory_file"]
    with np.load(trajectory_path) as trajectory:
        np.testing.assert_array_equal(trajectory["positions"], [[0.0, 0.0], [1.0, 2.0]])
        assert trajectory["joint_names"].tolist() == ["j1", "j2"]
    assert saved[1] == {"type": "gripper", "closed": True}
