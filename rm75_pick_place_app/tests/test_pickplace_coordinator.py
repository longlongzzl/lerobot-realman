from __future__ import annotations

import numpy as np

from rm75_app.pickplace.coordinator import PickPlaceCoordinator, PickPlaceTask
from rm75_app.perception.held_object_refinement import HeldObjectRefinementUpdate
from rm75_app.planning.contracts import (
    BatchPlanningResult,
    CandidatePlan,
    CollisionObject,
    GraspCandidatePlan,
    GraspPlanningResult,
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

    def update_scene(self, scene) -> None:
        self.events.append(("scene", scene.revision))

    def plan_grasps(self, request):
        self.events.append(("plan_grasps", tuple(x.candidate_id for x in request.planning.candidates)))
        plans = []
        for item in request.planning.candidates:
            ok = item.candidate_id == "grasp_good"
            plans.append(
                GraspCandidatePlan(
                    item.candidate_id,
                    ok,
                    approach=trajectory(0.0, 0.5) if ok else None,
                    grasp=trajectory(0.5, 1.0) if ok else None,
                )
            )
        return GraspPlanningResult(tuple(plans), self.name)

    def plan_candidates(self, request):
        identifiers = tuple(item.candidate_id for item in request.candidates)
        self.events.append(("plan", identifiers, tuple(request.current.positions)))
        value = self.next_value
        self.next_value += 1.0
        return BatchPlanningResult(
            tuple(CandidatePlan(item.candidate_id, True, trajectory(value - 1.0, value)) for item in request.candidates),
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
    assert planner.events[1][0] == "attach"
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
    assert plan_starts == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]


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
    assert refiner.calls == [("carrot", (2.0, 2.0))]
    event_names = [event[0] for event in planner.events]
    update_index = event_names.index("update_attachment")
    assert event_names[update_index - 1] == "plan"  # lift planning
    assert event_names[update_index + 1] == "plan"  # pre-place planning


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
