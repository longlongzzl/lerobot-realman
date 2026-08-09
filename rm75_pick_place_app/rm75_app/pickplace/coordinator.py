"""Planner-independent pick-place state machine.

The coordinator owns sequencing; perception, cuRobo and simulation remain
replaceable adapters.  In particular, attachment is an explicit planning
boundary and can no longer be hidden in a monolithic runtime script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rm75_app.planning.contracts import (
    BatchPlanningRequest,
    CandidatePlan,
    GraspCandidatePlan,
    GraspPlanningRequest,
    JointConfiguration,
    JointTrajectory,
    PlanningScene,
    Pose,
    PoseCandidate,
)
from rm75_app.planning.interfaces import PlanningBackend
from rm75_app.perception.held_object_refinement import (
    HeldObjectRefinementHook,
    HeldObjectRefinementUpdate,
)


class TrajectoryExecutor(Protocol):
    def execute_trajectory(self, stage: str, trajectory: JointTrajectory) -> None: ...

    def set_gripper(self, closed: bool) -> None: ...


@dataclass(frozen=True)
class PickPlaceTask:
    object_name: str
    current: JointConfiguration
    grasp_candidates: tuple[PoseCandidate, ...]
    place_candidates: tuple[PoseCandidate, ...]
    scene: PlanningScene
    tool_frame: str = "gripper_tcp"
    grasp_approach_offset: float = -0.10
    lift_height: float = 0.10
    place_clearance: float = 0.10
    max_attempts: int = 2

    def __post_init__(self) -> None:
        if not self.object_name:
            raise ValueError("object_name must not be empty")
        if self.object_name not in {item.name for item in self.scene.objects}:
            raise ValueError(f"picked object {self.object_name!r} is missing from the scene")
        if not self.grasp_candidates or not self.place_candidates:
            raise ValueError("grasp and place candidates must not be empty")


@dataclass(frozen=True)
class ExecutedStage:
    name: str
    candidate_id: str | None
    end: JointConfiguration


@dataclass(frozen=True)
class PickPlaceRunResult:
    success: bool
    stages: tuple[ExecutedStage, ...]
    selected_grasp: str | None = None
    selected_place: str | None = None
    failure_stage: str | None = None
    message: str | None = None
    held_object_refinement: HeldObjectRefinementUpdate | None = None


def _offset_candidates(
    candidates: tuple[PoseCandidate, ...], dz: float, prefix: str
) -> tuple[PoseCandidate, ...]:
    return tuple(
        PoseCandidate(
            candidate_id=f"{prefix}:{candidate.candidate_id}",
            pose=Pose(
                [
                    candidate.pose.position[0],
                    candidate.pose.position[1],
                    candidate.pose.position[2] + dz,
                ],
                candidate.pose.quaternion_wxyz,
            ),
            score=candidate.score,
            metadata=candidate.metadata,
        )
        for candidate in candidates
    )


class PickPlaceCoordinator:
    def __init__(
        self,
        planner: PlanningBackend,
        executor: TrajectoryExecutor,
        held_object_refiner: HeldObjectRefinementHook | None = None,
    ):
        self.planner = planner
        self.executor = executor
        self.held_object_refiner = held_object_refiner

    @staticmethod
    def _end_configuration(trajectory: JointTrajectory) -> JointConfiguration:
        return JointConfiguration(trajectory.joint_names, trajectory.positions[-1])

    def _execute(self, stage: str, plan: CandidatePlan | GraspCandidatePlan) -> ExecutedStage:
        trajectory = plan.grasp if isinstance(plan, GraspCandidatePlan) else plan.trajectory
        if trajectory is None:
            raise RuntimeError(f"successful {stage} plan has no trajectory")
        self.executor.execute_trajectory(stage, trajectory)
        return ExecutedStage(stage, plan.candidate_id, self._end_configuration(trajectory))

    def _plan_pose_stage(
        self,
        *,
        stage: str,
        current: JointConfiguration,
        candidates: tuple[PoseCandidate, ...],
        task: PickPlaceTask,
        prefer_unbiased_ik: bool = True,
    ) -> CandidatePlan | None:
        result = self.planner.plan_candidates(
            BatchPlanningRequest(
                current=current,
                candidates=candidates,
                scene=task.scene,
                tool_frame=task.tool_frame,
                max_attempts=task.max_attempts,
                prefer_unbiased_ik=prefer_unbiased_ik,
            )
        )
        best = result.best(candidates)
        if best is None:
            diagnostics = ", ".join(
                f"{plan.candidate_id}={plan.status}" for plan in result.plans
            )
            print(f"[{stage}] no feasible candidate: {diagnostics}")
        return best

    def run(self, task: PickPlaceTask) -> PickPlaceRunResult:
        stages: list[ExecutedStage] = []
        attached = False
        released_collision_disabled = False
        selected_grasp: str | None = None
        selected_place: str | None = None
        held_refinement: HeldObjectRefinementUpdate | None = None
        try:
            preplace_candidates = _offset_candidates(
                task.place_candidates, task.place_clearance, "preplace"
            )
            prepare = getattr(self.planner, "prepare_pose_candidates", None)
            if callable(prepare):
                # IK is independent of the grasp state. Solve both goal sets
                # before attachment, then retain attached-object collision for
                # the actual transport/descent trajectories.
                prepare(
                    preplace_candidates,
                    task.scene,
                    tool_frame=task.tool_frame,
                    ignore_object_name=task.object_name,
                )
                prepare(
                    task.place_candidates,
                    task.scene,
                    tool_frame=task.tool_frame,
                    ignore_object_name=task.object_name,
                )
            grasp_result = self.planner.plan_grasps(
                GraspPlanningRequest(
                    BatchPlanningRequest(
                        current=task.current,
                        candidates=task.grasp_candidates,
                        scene=task.scene,
                        tool_frame=task.tool_frame,
                        max_attempts=task.max_attempts,
                    ),
                    target_object_name=task.object_name,
                    approach_offset=task.grasp_approach_offset,
                )
            )
            grasp = grasp_result.best(task.grasp_candidates)
            if grasp is None or grasp.approach is None or grasp.grasp is None:
                return PickPlaceRunResult(False, tuple(stages), failure_stage="grasp", message="no feasible grasp")
            selected_grasp = grasp.candidate_id

            self.executor.execute_trajectory("approach", grasp.approach)
            stages.append(
                ExecutedStage(
                    "approach", grasp.candidate_id, self._end_configuration(grasp.approach)
                )
            )
            self.executor.execute_trajectory("grasp", grasp.grasp)
            current = self._end_configuration(grasp.grasp)
            stages.append(ExecutedStage("grasp", grasp.candidate_id, current))
            self.executor.set_gripper(True)

            self.planner.attach_object(task.object_name, current)
            attached = True

            grasp_by_id = {item.candidate_id: item for item in task.grasp_candidates}
            selected_pose = grasp_by_id[grasp.candidate_id]
            lift_candidates = _offset_candidates((selected_pose,), task.lift_height, "lift")
            lift = self._plan_pose_stage(
                stage="lift", current=current, candidates=lift_candidates, task=task
            )
            if lift is None:
                return PickPlaceRunResult(
                    False, tuple(stages), selected_grasp, failure_stage="lift", message="no feasible lift"
                )
            stage = self._execute("lift", lift)
            stages.append(stage)
            current = stage.end

            if self.held_object_refiner is not None:
                try:
                    held_refinement = self.held_object_refiner.refine_after_lift(
                        task.object_name, current
                    )
                    if held_refinement.accepted and held_refinement.T_tcp_object is not None:
                        self.planner.update_attached_object_pose(
                            task.object_name,
                            current,
                            held_refinement.T_tcp_object,
                        )
                except Exception as exc:
                    # Perception refinement is optional and fail-open: retain
                    # the grasp-time attachment estimate and continue safely.
                    held_refinement = HeldObjectRefinementUpdate(
                        False,
                        task.object_name,
                        None,
                        0.0,
                        0.0,
                        f"refinement_error:{type(exc).__name__}",
                        metadata={"error": str(exc)},
                    )

            preplace = self._plan_pose_stage(
                stage="preplace",
                current=current,
                candidates=preplace_candidates,
                task=task,
            )
            if preplace is None:
                return PickPlaceRunResult(
                    False,
                    tuple(stages),
                    selected_grasp,
                    failure_stage="preplace",
                    message="no feasible preplace",
                    held_object_refinement=held_refinement,
                )
            selected_place = preplace.candidate_id.split(":", 1)[-1]
            stage = self._execute("preplace", preplace)
            stages.append(stage)
            current = stage.end

            place_lookup = {item.candidate_id: item for item in task.place_candidates}
            place_candidates = (place_lookup[selected_place],)
            place = self._plan_pose_stage(
                stage="place", current=current, candidates=place_candidates, task=task
            )
            if place is None:
                return PickPlaceRunResult(
                    False,
                    tuple(stages),
                    selected_grasp,
                    selected_place,
                    failure_stage="place",
                    message="no feasible place descent",
                    held_object_refinement=held_refinement,
                )
            stage = self._execute("place", place)
            stages.append(stage)
            current = stage.end

            self.executor.set_gripper(False)
            self.planner.detach_object(task.object_name, place_candidates[0].pose)
            attached = False
            released_collision_disabled = True

            retreat_candidates = _offset_candidates(place_candidates, task.place_clearance, "retreat")
            retreat = self._plan_pose_stage(
                stage="retreat", current=current, candidates=retreat_candidates, task=task
            )
            if retreat is None:
                return PickPlaceRunResult(
                    False,
                    tuple(stages),
                    selected_grasp,
                    selected_place,
                    failure_stage="retreat",
                    message="object released but retreat planning failed",
                    held_object_refinement=held_refinement,
                )
            stages.append(self._execute("retreat", retreat))
            self.planner.enable_object_collision(task.object_name)
            released_collision_disabled = False
            return PickPlaceRunResult(
                True,
                tuple(stages),
                selected_grasp=selected_grasp,
                selected_place=selected_place,
                held_object_refinement=held_refinement,
            )
        finally:
            if attached:
                self.planner.detach_object(task.object_name)
            elif released_collision_disabled:
                self.planner.enable_object_collision(task.object_name)
