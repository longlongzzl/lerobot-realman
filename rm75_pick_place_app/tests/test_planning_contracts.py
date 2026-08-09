from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rm75_app.planning.backends.curobo2 import load_curobo2_robot_config
from rm75_app.planning.contracts import (
    BatchPlanningRequest,
    BatchPlanningResult,
    CandidatePlan,
    CollisionObject,
    JointConfiguration,
    PlanningScene,
    Pose,
    PoseCandidate,
)


def test_pose_normalizes_quaternion() -> None:
    pose = Pose([0.1, 0.2, 0.3], [2.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(pose.quaternion_wxyz, [1.0, 0.0, 0.0, 0.0])
    assert pose.as_curobo_list() == [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]


def test_scene_rejects_duplicate_names() -> None:
    pose = Pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    item = CollisionObject("table", "cuboid", pose, dimensions=[1.0, 1.0, 0.1])
    with pytest.raises(ValueError, match="unique"):
        PlanningScene((item, item))


def test_batch_result_selects_best_feasible_candidate() -> None:
    pose = Pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    candidates = (
        PoseCandidate("low", pose, score=0.1),
        PoseCandidate("failed", pose, score=1.0),
        PoseCandidate("high", pose, score=0.8),
    )
    request = BatchPlanningRequest(
        JointConfiguration(("j1",), [0.0]), candidates
    )
    result = BatchPlanningResult(
        (
            CandidatePlan("low", True),
            CandidatePlan("failed", False),
            CandidatePlan("high", True),
        ),
        backend="fake",
    )
    assert result.best(request.candidates).candidate_id == "high"


def test_v1_rm75_config_is_translated_without_changing_source() -> None:
    path = Path(__file__).parents[1] / "assets/curobo_rm75_config/rm75.yml"
    source = path.read_text(encoding="utf-8")
    converted = load_curobo2_robot_config(path)
    kinematics = converted["robot_cfg"]["kinematics"]

    assert kinematics["format_version"] == 2.0
    assert kinematics["tool_frames"] == ["gripper_tcp"]
    assert "ee_link" not in kinematics
    assert "link_names" not in kinematics
    assert "default_joint_position" in kinematics["cspace"]
    assert "retract_config" not in kinematics["cspace"]
    assert isinstance(kinematics["collision_spheres"], dict)
    assert path.read_text(encoding="utf-8") == source
