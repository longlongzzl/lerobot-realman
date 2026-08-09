from __future__ import annotations

import json

import numpy as np

from rm75_app.runtime.curobo2_sim_replay import load_replay_events


def test_load_replay_events_is_planner_independent(tmp_path) -> None:
    np.savez_compressed(
        tmp_path / "00_approach.npz",
        positions=np.asarray([[0.0, 1.0], [2.0, 3.0]]),
        joint_names=np.asarray(["j1", "j2"]),
        dt=np.asarray(0.05),
    )
    manifest = tmp_path / "execution.json"
    manifest.write_text(
        json.dumps(
            [
                {"type": "trajectory", "stage": "approach", "trajectory_file": "00_approach.npz"},
                {"type": "gripper", "closed": True},
            ]
        ),
        encoding="utf-8",
    )
    events = load_replay_events(manifest)
    assert events[0]["trajectory"].joint_names == ("j1", "j2")
    np.testing.assert_array_equal(events[0]["trajectory"].positions[-1], [2.0, 3.0])
    assert float(events[0]["trajectory"].dt) == 0.05
    assert events[1]["closed"] is True
