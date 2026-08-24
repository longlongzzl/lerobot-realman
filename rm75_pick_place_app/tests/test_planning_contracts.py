from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rm75_app.planning.backends.curobo2 import (
    Curobo2Backend,
    Curobo2BackendConfig,
    load_curobo2_robot_config,
)
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


def test_curobo2_routes_analytic_sphere_through_conservative_obb() -> None:
    class Record:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    backend = Curobo2Backend()
    backend._import_modules = lambda: {
        "Cuboid": Record,
        "Mesh": Record,
        "Scene": Record,
    }
    sphere = CollisionObject(
        "ball",
        "sphere",
        Pose([0.3, -0.2, 0.1], [0.5, 0.5, 0.5, 0.5]),
        radius=0.04,
    )

    converted = backend._to_curobo_scene(PlanningScene((sphere,)))

    assert converted.mesh == []
    assert len(converted.cuboid) == 1
    obstacle = converted.cuboid[0]
    assert obstacle.name == "ball"
    assert obstacle.pose == [0.3, -0.2, 0.1, 1.0, 0.0, 0.0, 0.0]
    assert obstacle.dims == [0.08, 0.08, 0.08]


def test_curobo2_fast_path_is_the_default() -> None:
    config = Curobo2BackendConfig()

    assert config.use_cuda_graph is True
    assert config.coarse_ik_use_cuda_graph is True
    assert config.warmup_planner is True
    assert config.enable_graph_attempt == 1
    assert config.interpolation_buffer_size == 500
    assert config.collision_cache_obb == 32
    assert config.collision_cache_mesh == 16
    assert config.isolate_graph_seed_failures is False


def test_coarse_screening_keeps_both_fixed_shape_solvers_resident() -> None:
    backend = Curobo2Backend()
    formal = object()
    coarse = object()
    backend._planner = formal
    backend._coarse_ik_solver = coarse

    backend.begin_coarse_screening()
    backend.end_coarse_screening()

    assert backend._planner is formal
    assert backend._coarse_ik_solver is coarse


def test_official_graph_seeding_is_default_but_workaround_remains_opt_in() -> None:
    planner = object()
    backend = Curobo2Backend()
    backend._install_isolated_graph_seeding = lambda *_args: pytest.fail(
        "compatibility override should not be installed"
    )
    backend._maybe_install_isolated_graph_seeding(planner)()

    opted_in = Curobo2Backend(
        Curobo2BackendConfig(isolate_graph_seed_failures=True)
    )
    calls = []
    opted_in._install_isolated_graph_seeding = lambda *args: (
        calls.append(args) or (lambda: None)
    )
    opted_in._maybe_install_isolated_graph_seeding(planner)()
    assert calls == [(planner, None)]


def test_grasp_stage_attempts_preserve_official_planner_routing() -> None:
    class FakePlanner:
        def __init__(self):
            self.graph_planner = object()
            self.calls = []

        def plan_pose(self, *args, **kwargs):
            self.calls.append((self.graph_planner, kwargs.get("max_attempts")))
            return len(self.calls)

    planner = FakePlanner()
    graph_planner = planner.graph_planner
    restore = Curobo2Backend._install_grasp_stage_attempts(planner, max_attempts=3)
    try:
        assert planner.plan_pose() == 1
        assert planner.plan_pose(max_attempts=5) == 2
    finally:
        restore()

    assert planner.calls == [(graph_planner, 3), (graph_planner, 5)]
    assert planner.graph_planner is graph_planner
    assert "plan_pose" not in planner.__dict__


def test_grasp_linear_scale_changes_only_non_terminal_weights() -> None:
    class FakeTensor:
        def __init__(self, values):
            self.values = list(values)

        def __ne__(self, other):
            return FakeTensor([value != other for value in self.values])

        def any(self):
            return self

        def item(self):
            return any(self.values)

        def mul_(self, scale):
            self.values = [value * scale for value in self.values]

    class FakeCriterion:
        def __init__(self, terminal, non_terminal):
            self.terminal_pose_axes_weight_factor = FakeTensor(terminal)
            self.non_terminal_pose_axes_weight_factor = FakeTensor(non_terminal)

        def clone(self):
            return FakeCriterion(
                self.terminal_pose_axes_weight_factor.values,
                self.non_terminal_pose_axes_weight_factor.values,
            )

    class FakePlanner:
        def __init__(self):
            self.received = []

        def update_tool_pose_criteria(self, criteria):
            self.received.append(criteria)

    planner = FakePlanner()
    linear = FakeCriterion([1.0] * 6, [1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    standard = FakeCriterion([1.0] * 6, [0.0] * 6)
    restore = Curobo2Backend._install_grasp_linear_scale(planner, scale=0.5)
    try:
        planner.update_tool_pose_criteria({"tcp": linear})
        planner.update_tool_pose_criteria({"tcp": standard})
    finally:
        restore()

    scaled = planner.received[0]["tcp"]
    untouched = planner.received[1]["tcp"]
    assert scaled is not linear
    assert scaled.terminal_pose_axes_weight_factor.values == [1.0] * 6
    assert scaled.non_terminal_pose_axes_weight_factor.values == [0.5, 0.5, 0.0, 0.5, 0.5, 0.5]
    assert untouched is standard
    assert "update_tool_pose_criteria" not in planner.__dict__


def test_grasp_padding_winner_maps_to_last_semantic_candidate() -> None:
    assert Curobo2Backend._grasp_winner_indices(
        np.asarray([False, False, False, True]), 1
    ) == [3]
    assert Curobo2Backend._grasp_winner_indices(
        np.asarray([True, False, True, False]), 3
    ) == [0, 1, 2]


def test_official_grasp_status_distinguishes_all_pipeline_stages() -> None:
    status = Curobo2Backend._official_grasp_status
    assert "goalset_failed" in status(
        "Goalset planning returned None.",
        success=False,
        goalset_success=False,
        approach_success=False,
        grasp_success=False,
    )
    assert "approach_failed" in status(
        "Planning to approach pose failed.",
        success=False,
        goalset_success=True,
        approach_success=False,
        grasp_success=False,
    )
    assert "grasp_failed" in status(
        "Planning to grasp pose failed.",
        success=False,
        goalset_success=True,
        approach_success=True,
        grasp_success=False,
    )
