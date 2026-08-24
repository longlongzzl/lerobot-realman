from __future__ import annotations

import numpy as np

from rm75_app.pickplace.coordinator import PickPlaceCoordinator, PickPlaceTask, _target_object_pose
from rm75_app.perception.held_object_refinement import HeldObjectRefinementUpdate
from rm75_app.planning.contracts import (
    BatchPlanningResult,
    CandidatePlan,
    CollisionObject,
    JointConfiguration,
    JointTrajectory,
    PlanningScene,
    Pose,
    PoseCandidate,
)


JOINTS = ("j1", "j2")


def trajectory(start: float, end: float) -> JointTrajectory:
    return JointTrajectory(JOINTS, np.asarray([[start, start], [end, end]]))


class FakePlanner:
    name = "fake"

    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.next_value = 2.0
        self.linear_fail_ids = {"grasp_bad"}

    def update_scene(self, scene) -> None:
        self.events.append(("scene", scene.revision))

    def plan_candidates(self, request):
        identifiers = tuple(item.candidate_id for item in request.candidates)
        self.events.append(("plan", identifiers, tuple(request.current.positions)))
        value = self.next_value
        self.next_value += 1.0
        start = float(request.current.positions[0])
        return BatchPlanningResult(
            tuple(
                CandidatePlan(
                    item.candidate_id, True, trajectory(start, value)
                )
                for item in request.candidates
            ),
            self.name,
        )

    def plan_linear_candidates(self, request, **kwargs):
        identifiers = tuple(item.candidate_id for item in request.candidates)
        self.events.append(
            ("linear", identifiers, tuple(request.current.positions), dict(kwargs))
        )
        value = self.next_value
        self.next_value += 1.0
        start = float(request.current.positions[0])
        return BatchPlanningResult(
            tuple(
                CandidatePlan(
                    item.candidate_id,
                    item.candidate_id not in self.linear_fail_ids,
                    trajectory=(
                        trajectory(start, value)
                        if item.candidate_id not in self.linear_fail_ids
                        else None
                    ),
                    status=(
                        "linear_success"
                        if item.candidate_id not in self.linear_fail_ids
                        else "linear_collision"
                    ),
                )
                for item in request.candidates
            ),
            self.name,
        )

    def attach_object(self, object_name, grasp):
        self.events.append(("attach", object_name, tuple(grasp.positions)))

    def update_attached_object_pose(self, object_name, current, T_tcp_object):
        self.events.append(
            (
                "update_attachment",
                object_name,
                tuple(current.positions),
                tuple(np.asarray(T_tcp_object)[:3, 3]),
            )
        )

    def detach_object(self, object_name, released_pose=None):
        self.events.append(("detach", object_name, released_pose is not None))

    def enable_object_collision(self, object_name):
        self.events.append(("enable", object_name))

    def close(self):
        pass


class FakeExecutor:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def execute_trajectory(self, stage, trajectory):
        self.events.append((stage, tuple(trajectory.positions[-1])))

    def set_gripper(self, closed):
        self.events.append(("gripper", bool(closed)))


def test_coordinator_keeps_attachment_boundary_and_state_chain() -> None:
    planner = FakePlanner()
    executor = FakeExecutor()
    pose = Pose([0.4, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0])
    scene = PlanningScene(
        (CollisionObject("carrot", "cuboid", pose, dimensions=[0.15, 0.03, 0.03]),),
        revision="cached-scene",
    )
    task = PickPlaceTask(
        object_name="carrot",
        current=JointConfiguration(JOINTS, [0.0, 0.0]),
        grasp_candidates=(
            PoseCandidate("grasp_bad", pose, score=1.0),
            PoseCandidate("grasp_good", pose, score=0.8),
        ),
        place_candidates=(PoseCandidate("bin_a", pose, score=1.0),),
        scene=scene,
    )

    result = PickPlaceCoordinator(planner, executor).run(task)

    assert result.success
    assert [stage.name for stage in result.stages] == [
        "approach",
        "grasp",
        "lift",
        "preplace",
        "place",
        "retreat",
    ]
    assert next(event for event in planner.events if event[0] == "attach")[0] == "attach"
    assert ("detach", "carrot", True) in planner.events
    assert planner.events[-1] == ("enable", "carrot")
    assert [event[0] for event in executor.events] == [
        "approach",
        "grasp",
        "gripper",
        "lift",
        "preplace",
        "place",
        "gripper",
        "retreat",
    ]
    plan_starts = [event[2] for event in planner.events if event[0] == "plan"]
    assert plan_starts == [(0.0, 0.0), (0.0, 0.0), (6.0, 6.0)]
    linear_ids = [event[1] for event in planner.events if event[0] == "linear"]
    assert linear_ids == [
        ("grasp_bad",),
        ("grasp_good",),
        ("lift:grasp_good",),
        ("bin_a",),
    ]
    assert "plan_grasps" not in [event[0] for event in planner.events]
    assert executor.events[-1] == ("retreat", (7.0, 7.0))
    place_linear = next(
        event for event in planner.events
        if event[0] == "linear" and event[1] == ("bin_a",)
    )
    assert place_linear[3]["disable_collision_links"] == ("left_pad", "right_pad")


class FakeHeldObjectRefiner:
    def __init__(self, accepted: bool = True) -> None:
        self.accepted = accepted
        self.calls: list[tuple[str, tuple[float, ...]]] = []

    def refine_after_lift(self, object_name, current):
        self.calls.append((object_name, tuple(current.positions)))
        transform = np.eye(4)
        transform[:3, 3] = [0.01, -0.002, 0.08]
        return HeldObjectRefinementUpdate(
            self.accepted,
            object_name,
            transform if self.accepted else None,
            0.01,
            1.0,
            "accepted" if self.accepted else "translation_gate",
            source="fake_wrist",
        )


def _simple_task() -> PickPlaceTask:
    pose = Pose([0.4, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0])
    return PickPlaceTask(
        object_name="carrot",
        current=JointConfiguration(JOINTS, [0.0, 0.0]),
        grasp_candidates=(PoseCandidate("grasp_good", pose),),
        place_candidates=(PoseCandidate("bin_a", pose),),
        scene=PlanningScene(
            (
                CollisionObject(
                    "carrot", "cuboid", pose, dimensions=[0.15, 0.03, 0.03]
                ),
            )
        ),
    )


def test_accepted_held_object_refinement_updates_attachment_after_lift() -> None:
    planner = FakePlanner()
    refiner = FakeHeldObjectRefiner()

    result = PickPlaceCoordinator(
        planner, FakeExecutor(), held_object_refiner=refiner
    ).run(_simple_task())

    assert result.success
    assert result.held_object_refinement is not None
    assert result.held_object_refinement.accepted
    assert refiner.calls == [("carrot", (4.0, 4.0))]
    event_names = [event[0] for event in planner.events]
    update_index = event_names.index("update_attachment")
    assert event_names[update_index - 1] == "attach"
    assert event_names[update_index + 1] == "detach"
    assert "plan_grasps" not in event_names


def test_rejected_held_object_refinement_keeps_original_attachment() -> None:
    planner = FakePlanner()
    refiner = FakeHeldObjectRefiner(accepted=False)

    result = PickPlaceCoordinator(
        planner, FakeExecutor(), held_object_refiner=refiner
    ).run(_simple_task())

    assert result.success
    assert result.held_object_refinement is not None
    assert not result.held_object_refinement.accepted
    assert "update_attachment" not in [event[0] for event in planner.events]


def test_selected_grasp_uses_only_its_paired_place_candidates() -> None:
    planner = FakePlanner()
    pose = Pose([0.4, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0])
    paired = PoseCandidate("paired_place", pose)
    unrelated = PoseCandidate("unrelated_place", pose)
    task = PickPlaceTask(
        object_name="carrot",
        current=JointConfiguration(JOINTS, [0.0, 0.0]),
        grasp_candidates=(
            PoseCandidate("grasp_bad", pose),
            PoseCandidate("grasp_good", pose),
        ),
        place_candidates=(unrelated, paired),
        place_candidates_by_grasp={
            "grasp_bad": (unrelated,),
            "grasp_good": (paired,),
        },
        scene=PlanningScene(
            (CollisionObject("carrot", "cuboid", pose, dimensions=[0.15, 0.03, 0.03]),)
        ),
    )
    result = PickPlaceCoordinator(planner, FakeExecutor()).run(task)
    assert result.success
    assert result.selected_grasp == "grasp_good"
    assert result.selected_place == "paired_place"
    linear_ids = [event[1] for event in planner.events if event[0] == "linear"]
    assert ("paired_place",) in linear_ids
    assert ("unrelated_place",) not in linear_ids
    assert "plan_grasps" not in [event[0] for event in planner.events]


def test_release_updates_scene_with_object_pose_not_tcp_pose() -> None:
    planner = FakePlanner()
    tcp_pose = Pose([0.4, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0])
    object_transform = np.eye(4)
    object_transform[:3, 3] = [0.3, -0.1, 0.05]
    place = PoseCandidate(
        "place",
        tcp_pose,
        metadata={"target_object_pose": object_transform.tolist()},
    )
    task = PickPlaceTask(
        object_name="carrot",
        current=JointConfiguration(JOINTS, [0.0, 0.0]),
        grasp_candidates=(PoseCandidate("grasp_good", tcp_pose),),
        place_candidates=(place,),
        scene=PlanningScene(
            (CollisionObject("carrot", "cuboid", tcp_pose, dimensions=[0.1, 0.03, 0.03]),)
        ),
    )

    result = PickPlaceCoordinator(planner, FakeExecutor()).run(task)

    assert result.success
    detaches = [event for event in planner.events if event[0] == "detach"]
    assert detaches[-1] == ("detach", "carrot", True)
    np.testing.assert_allclose(_target_object_pose(place).position, object_transform[:3, 3])


def test_failed_grasp_preserves_structured_collision_diagnostics() -> None:
    planner = FakePlanner()

    def fail_linear(request, **kwargs):
        del kwargs
        return BatchPlanningResult(
            tuple(
                CandidatePlan(
                    candidate.candidate_id,
                    False,
                    status="Start or End state in collision",
                    diagnostics={
                        "collision_diagnostics": [
                            {
                                "collision_type": "world",
                                "state": "grasp_goal",
                                "robot_link": "link6",
                                "world_object": "tennis",
                                "penetration_m": 0.003,
                            }
                        ]
                    },
                )
                for candidate in request.candidates
            ),
            planner.name,
        )

    planner.plan_linear_candidates = fail_linear
    result = PickPlaceCoordinator(planner, FakeExecutor()).run(_simple_task())

    assert not result.success
    failures = result.diagnostics["candidate_failures"]
    collision = failures[0]["candidates"][0]["collision_diagnostics"][0]
    assert collision["robot_link"] == "link6"
    assert collision["world_object"] == "tennis"


def test_four_endpoint_relation_screen_selects_complete_lower_score_chain() -> None:
    class RelationPlanner(FakePlanner):
        def __init__(self):
            super().__init__()
            self.prepared = []

        def prepare_pose_candidates(self, candidates, scene, **kwargs):
            del scene, kwargs
            self.prepared.append(tuple(item.candidate_id for item in candidates))

        def feasible_pose_candidate_ids(self, candidates):
                return frozenset(
                    item.candidate_id
                    for item in candidates
                    if "low_place" in item.candidate_id
                    or "grasp_low" in item.candidate_id
            )


    planner = RelationPlanner()
    pose = Pose([0.4, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0])
    task = PickPlaceTask(
        object_name="carrot",
        current=JointConfiguration(JOINTS, [0.0, 0.0]),
        grasp_candidates=(
            PoseCandidate("grasp_high", pose, score=1.0),
            PoseCandidate("grasp_low", pose, score=0.5),
        ),
        place_candidates=(
            PoseCandidate("high_place", pose),
            PoseCandidate("low_place", pose),
        ),
        place_candidates_by_grasp={
            "grasp_high": (PoseCandidate("high_place", pose),),
            "grasp_low": (PoseCandidate("low_place", pose),),
        },
        scene=PlanningScene(
            (CollisionObject("carrot", "cuboid", pose, dimensions=[0.1, 0.03, 0.03]),)
        ),
    )

    result = PickPlaceCoordinator(planner, FakeExecutor()).run(task)

    assert result.success
    assert result.selected_grasp == "grasp_low"
    assert result.selected_place == "low_place"
    assert planner.prepared[0] == (
        "preplace:high_place",
        "preplace:low_place",
    )
    assert planner.prepared[2] == ("pregrasp:grasp_low",)
    assert planner.prepared[3] == ("grasp_low",)
    linear_ids = [event[1] for event in planner.events if event[0] == "linear"]
    assert ("grasp_low",) in linear_ids
    assert ("low_place",) in linear_ids
    assert ("grasp_high",) not in linear_ids


def test_failed_place_line_falls_back_to_next_matching_relation() -> None:
    class PairPlanner(FakePlanner):
        def __init__(self):
            super().__init__()
            self.linear_fail_ids = {"place_a"}

    planner = PairPlanner()
    pose = Pose([0.4, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0])
    place_a = PoseCandidate("place_a", pose, score=1.0)
    place_b = PoseCandidate("place_b", pose, score=0.9)
    task = PickPlaceTask(
        object_name="carrot",
        current=JointConfiguration(JOINTS, [0.0, 0.0]),
        grasp_candidates=(
            PoseCandidate("grasp_a", pose, score=1.0),
            PoseCandidate("grasp_b", pose, score=0.9),
        ),
        place_candidates=(place_a, place_b),
        place_candidates_by_grasp={
            "grasp_a": (place_a,),
            "grasp_b": (place_b,),
        },
        scene=PlanningScene(
            (CollisionObject("carrot", "cuboid", pose, dimensions=[0.1, 0.03, 0.03]),)
        ),
    )

    result = PickPlaceCoordinator(planner, FakeExecutor()).run(task)

    assert result.success
    assert result.selected_grasp == "grasp_b"
    assert result.selected_place == "place_b"
    linear_ids = [event[1] for event in planner.events if event[0] == "linear"]
    assert ("place_a",) in linear_ids
    assert ("place_b",) in linear_ids
    assert linear_ids.index(("place_a",)) < linear_ids.index(("grasp_b",))
    assert "attach_batch" not in [event[0] for event in planner.events]
