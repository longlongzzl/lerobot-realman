"""Planner-independent pick-place state machine.

The coordinator owns sequencing; perception, cuRobo and simulation remain
replaceable adapters.  In particular, attachment is an explicit planning
boundary and can no longer be hidden in a monolithic runtime script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any, Mapping, Protocol

import numpy as np

from rm75_app.pickplace.cached_scene import matrix_to_quaternion_wxyz
from rm75_app.planning.contracts import (
    BatchPlanningRequest,
    CandidatePlan,
    JointConfiguration,
    JointTrajectory,
    PlanningScene,
    Pose,
    PoseCandidate,
)
from rm75_app.planning.interfaces import PlanningBackend
if TYPE_CHECKING:
    from rm75_app.perception.held_object_refinement import (
        HeldObjectRefinementHook,
        HeldObjectRefinementUpdate,
    )
else:
    HeldObjectRefinementHook = Any
    HeldObjectRefinementUpdate = Any


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
    place_contact_object_name: str | None = None
    max_motion_candidates: int = 8
    place_candidates_by_grasp: Mapping[str, tuple[PoseCandidate, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.object_name:
            raise ValueError("object_name must not be empty")
        if self.object_name not in {item.name for item in self.scene.objects}:
            raise ValueError(f"picked object {self.object_name!r} is missing from the scene")
        if not self.grasp_candidates or not self.place_candidates:
            raise ValueError("grasp and place candidates must not be empty")
        grasp_ids = {item.candidate_id for item in self.grasp_candidates}
        unknown = set(self.place_candidates_by_grasp) - grasp_ids
        if unknown:
            raise ValueError(f"place mapping contains unknown grasp ids: {sorted(unknown)}")
        if any(not candidates for candidates in self.place_candidates_by_grasp.values()):
            raise ValueError("per-grasp place candidate lists must not be empty")
        if self.max_motion_candidates < 1:
            raise ValueError("max_motion_candidates must be positive")
        object.__setattr__(
            self,
            "place_candidates_by_grasp",
            {str(key): tuple(value) for key, value in self.place_candidates_by_grasp.items()},
        )

    def places_for_grasp(self, grasp_id: str) -> tuple[PoseCandidate, ...]:
        return tuple(self.place_candidates_by_grasp.get(grasp_id, self.place_candidates))


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
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


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


def _target_object_pose(place_candidate: PoseCandidate) -> Pose:
    value = place_candidate.metadata.get(
        "planning_target_object_pose",
        place_candidate.metadata.get("target_object_pose"),
    )
    if value is None:
        # Compatibility for externally constructed tasks that predate explicit
        # TCP/object pose separation.
        return place_candidate.pose
    transform = np.asarray(value, dtype=np.float64).reshape(4, 4)
    return Pose(
        transform[:3, 3],
        matrix_to_quaternion_wxyz(transform[:3, :3]),
    )


def _reverse_trajectory(trajectory: JointTrajectory) -> JointTrajectory:
    """Return the exact time-reversed path for a verified contact retreat."""

    dt = trajectory.dt
    if isinstance(dt, np.ndarray):
        dt = np.asarray(dt)[::-1].copy()
    return JointTrajectory(
        trajectory.joint_names,
        trajectory.positions[::-1].copy(),
        dt=dt,
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
        self._last_plan_failure: dict[str, Any] = {}

    @staticmethod
    def _end_configuration(trajectory: JointTrajectory) -> JointConfiguration:
        return JointConfiguration(trajectory.joint_names, trajectory.positions[-1])

    def _execute(self, stage: str, plan: CandidatePlan) -> ExecutedStage:
        trajectory = plan.trajectory
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
        batch_size = max(1, int(getattr(self.planner, "max_batch_size", len(candidates))))
        plans: list[CandidatePlan] = []
        for start in range(0, len(candidates), batch_size):
            chunk = tuple(candidates[start : start + batch_size])
            result = self.planner.plan_candidates(
                BatchPlanningRequest(
                    current=current,
                    candidates=chunk,
                    scene=task.scene,
                    tool_frame=task.tool_frame,
                    max_attempts=task.max_attempts,
                    prefer_unbiased_ik=prefer_unbiased_ik,
                )
            )
            plans.extend(result.plans)
        score_by_id = {item.candidate_id: item.score for item in candidates}
        feasible = [item for item in plans if item.success]
        best = (
            None
            if not feasible
            else max(
                feasible,
                key=lambda item: score_by_id.get(item.candidate_id, float("-inf")),
            )
        )
        if best is None:
            diagnostics = ", ".join(
                f"{plan.candidate_id}={plan.status}" for plan in plans
            )
            print(f"[{stage}] no feasible candidate: {diagnostics}")
            self._last_plan_failure = {
                "stage": stage,
                "candidates": [
                    {
                        "candidate_id": plan.candidate_id,
                        "status": plan.status,
                        **dict(plan.diagnostics),
                    }
                    for plan in plans
                ],
            }
        else:
            self._last_plan_failure = {}
        return best

    def _plan_linear_stage(
        self,
        *,
        stage: str,
        current: JointConfiguration,
        candidate: PoseCandidate,
        task: PickPlaceTask,
        ignore_object_name: str | None = None,
        disable_collision_links: tuple[str, ...] | None = None,
    ) -> CandidatePlan | None:
        request = BatchPlanningRequest(
            current=current,
            candidates=(candidate,),
            scene=task.scene,
            tool_frame=task.tool_frame,
            max_attempts=task.max_attempts,
        )
        linear = getattr(self.planner, "plan_linear_candidates", None)
        if callable(linear):
            result = linear(
                request,
                axis="z",
                project_distance_to_goal=False,
                ignore_object_name=ignore_object_name,
                disable_collision_links=disable_collision_links,
            )
            best = result.best((candidate,))
        else:
            result = self.planner.plan_candidates(request)
            best = result.best((candidate,))
        if best is None:
            self._last_plan_failure = {
                "stage": stage,
                "candidates": [
                    {
                        "candidate_id": item.candidate_id,
                        "status": item.status,
                        **dict(item.diagnostics),
                    }
                    for item in result.plans
                ],
            }
        else:
            self._last_plan_failure = {}
        return best

    def _run_segmented_chain(
        self,
        task: PickPlaceTask,
        relation_grasp_candidates: tuple[PoseCandidate, ...],
        complete_places_by_grasp: Mapping[str, tuple[PoseCandidate, ...]],
        grasp_scores: Mapping[str, float],
        relation_screen: Mapping[str, Any],
    ) -> PickPlaceRunResult:
        """Run the proven segmented chain without cuRobo2 PlanGrasp.

        Free-space legs use normal MotionGen.  Contact-adjacent legs use the
        independent world-Z linear primitive, and broad reachability has
        already been screened by the fixed batch64 IK solver.
        """

        planning_started = time.perf_counter()
        ranked_grasps = sorted(
            relation_grasp_candidates,
            key=lambda item: grasp_scores.get(item.candidate_id, item.score),
            reverse=True,
        )[: task.max_motion_candidates]
        selected_chain = None
        failures: list[dict[str, Any]] = []
        attached = False
        for grasp_candidate in ranked_grasps:
            pregrasp_candidate = _offset_candidates(
                (grasp_candidate,),
                abs(float(task.grasp_approach_offset)),
                "pregrasp",
            )[0]
            pregrasp = self._plan_pose_stage(
                stage="pregrasp",
                current=task.current,
                candidates=(pregrasp_candidate,),
                task=task,
            )
            if pregrasp is None or pregrasp.trajectory is None:
                failures.append(dict(self._last_plan_failure))
                continue
            pregrasp_end = self._end_configuration(pregrasp.trajectory)
            grasp = self._plan_linear_stage(
                stage="grasp",
                current=pregrasp_end,
                candidate=grasp_candidate,
                task=task,
                ignore_object_name=task.object_name,
            )
            if grasp is None or grasp.trajectory is None:
                failures.append(dict(self._last_plan_failure))
                continue
            grasp_end = self._end_configuration(grasp.trajectory)
            self.planner.attach_object(task.object_name, grasp_end)
            attached = True
            try:
                lift_candidate = _offset_candidates(
                    (grasp_candidate,), abs(float(task.lift_height)), "lift"
                )[0]
                lift = self._plan_linear_stage(
                    stage="lift",
                    current=grasp_end,
                    candidate=lift_candidate,
                    task=task,
                )
                if lift is None or lift.trajectory is None:
                    failures.append(dict(self._last_plan_failure))
                    continue
                lift_end = self._end_configuration(lift.trajectory)
                ranked_places = sorted(
                    complete_places_by_grasp[grasp_candidate.candidate_id],
                    key=lambda item: item.score,
                    reverse=True,
                )[: task.max_motion_candidates]
                for place_candidate in ranked_places:
                    preplace_candidate = _offset_candidates(
                        (place_candidate,), task.place_clearance, "preplace"
                    )[0]
                    preplace = self._plan_pose_stage(
                        stage="preplace",
                        current=lift_end,
                        candidates=(preplace_candidate,),
                        task=task,
                    )
                    if preplace is None or preplace.trajectory is None:
                        failures.append(dict(self._last_plan_failure))
                        continue
                    preplace_end = self._end_configuration(preplace.trajectory)
                    place = self._plan_linear_stage(
                        stage="place",
                        current=preplace_end,
                        candidate=place_candidate,
                        task=task,
                        ignore_object_name=task.place_contact_object_name,
                        disable_collision_links=("left_pad", "right_pad"),
                    )
                    if place is None or place.trajectory is None:
                        failures.append(dict(self._last_plan_failure))
                        continue
                    selected_chain = (
                        grasp_candidate,
                        place_candidate,
                        pregrasp,
                        grasp,
                        lift,
                        preplace,
                        place,
                    )
                    break
                if selected_chain is not None:
                    break
            finally:
                if attached:
                    self.planner.detach_object(task.object_name)
                    attached = False

        planning_time_s = time.perf_counter() - planning_started
        if selected_chain is None:
            return PickPlaceRunResult(
                False,
                (),
                failure_stage="segmented_chain",
                message="no complete segmented pick-place chain",
                diagnostics={
                    "relation_screen": dict(relation_screen),
                    "candidate_failures": failures,
                    "timing": {"segmented_plan_time_s": planning_time_s},
                },
            )

        (
            grasp_candidate,
            place_candidate,
            pregrasp,
            grasp,
            lift,
            preplace,
            place,
        ) = selected_chain
        stages: list[ExecutedStage] = []
        released_collision_disabled = False
        held_refinement = None
        try:
            stage = self._execute("approach", pregrasp)
            stages.append(stage)
            stage = self._execute("grasp", grasp)
            stages.append(stage)
            current = stage.end
            self.executor.set_gripper(True)
            self.planner.attach_object(task.object_name, current)
            attached = True

            stage = self._execute("lift", lift)
            stages.append(stage)
            current = stage.end
            if self.held_object_refiner is not None:
                try:
                    held_refinement = self.held_object_refiner.refine_after_lift(
                        task.object_name, current
                    )
                    if (
                        held_refinement.accepted
                        and held_refinement.T_tcp_object is not None
                    ):
                        self.planner.update_attached_object_pose(
                            task.object_name,
                            current,
                            held_refinement.T_tcp_object,
                        )
                except Exception as exc:
                    from rm75_app.perception.held_object_refinement import (
                        HeldObjectRefinementUpdate as _Update,
                    )

                    held_refinement = _Update(
                        False,
                        task.object_name,
                        None,
                        0.0,
                        0.0,
                        f"refinement_error:{type(exc).__name__}",
                        metadata={"error": str(exc)},
                    )

            stage = self._execute("preplace", preplace)
            stages.append(stage)
            stage = self._execute("place", place)
            stages.append(stage)
            self.executor.set_gripper(False)
            released_pose = _target_object_pose(place_candidate)
            self.planner.detach_object(task.object_name, released_pose)
            attached = False
            released_collision_disabled = True

            retreat = CandidatePlan(
                f"retreat:{place_candidate.candidate_id}",
                True,
                trajectory=_reverse_trajectory(place.trajectory),
                status="reverse_validated_place_line",
            )
            stages.append(self._execute("retreat", retreat))
            self.planner.enable_object_collision(task.object_name)
            released_collision_disabled = False
            return PickPlaceRunResult(
                True,
                tuple(stages),
                selected_grasp=grasp_candidate.candidate_id,
                selected_place=place_candidate.candidate_id,
                held_object_refinement=held_refinement,
                diagnostics={
                    "relation_screen": dict(relation_screen),
                    "planner_mode": "batch64_ik_segmented_motiongen",
                    "timing": {"segmented_plan_time_s": planning_time_s},
                },
            )
        finally:
            if attached:
                self.planner.detach_object(task.object_name)
            elif released_collision_disabled:
                self.planner.enable_object_collision(task.object_name)

    def run(self, task: PickPlaceTask) -> PickPlaceRunResult:
        coarse_screening_active = False
        try:
            begin_coarse = getattr(self.planner, "begin_coarse_screening", None)
            end_coarse = getattr(self.planner, "end_coarse_screening", None)
            if callable(begin_coarse):
                begin_coarse()
                coarse_screening_active = callable(end_coarse)
            all_place_candidates = tuple(task.place_candidates)
            if task.place_candidates_by_grasp:
                deduplicated: dict[str, PoseCandidate] = {
                    item.candidate_id: item for item in all_place_candidates
                }
                for candidates in task.place_candidates_by_grasp.values():
                    for item in candidates:
                        deduplicated[item.candidate_id] = item
                all_place_candidates = tuple(deduplicated.values())
            all_preplace_candidates = _offset_candidates(
                all_place_candidates, task.place_clearance, "preplace"
            )
            preplace_by_place_id = {
                place.candidate_id: preplace
                for place, preplace in zip(
                    all_place_candidates, all_preplace_candidates, strict=True
                )
            }
            prepare = getattr(self.planner, "prepare_pose_candidates", None)
            prepare_coarse = getattr(
                self.planner, "prepare_pose_candidates_coarse", prepare
            )
            feasible_pose_ids = getattr(
                self.planner, "feasible_pose_candidate_ids", None
            )
            contact_ignores = tuple(
                name
                for name in (task.object_name, task.place_contact_object_name)
                if name
            )
            grasp_ignores = (task.object_name,)
            max_tier = max(
                [int(item.metadata.get("search_tier", 0)) for item in task.grasp_candidates]
                + [int(item.metadata.get("search_tier", 0)) for item in all_place_candidates]
                + [0]
            )
            tiers = range(max_tier + 1) if callable(prepare) else (max_tier,)
            active_refinement_parent_ids: set[str] | None = None
            available_refinement_parent_ids = {
                str(parent)
                for item in task.grasp_candidates
                if (parent := item.metadata.get("refinement_parent_id")) is not None
            }
            refinement_tier_by_parent: dict[str, int] = {}
            for item in task.grasp_candidates:
                parent = item.metadata.get("refinement_parent_id")
                if parent is None:
                    continue
                parent_id = str(parent)
                refinement_tier_by_parent[parent_id] = max(
                    refinement_tier_by_parent.get(parent_id, 0),
                    int(item.metadata.get("search_tier", 0)),
                )

            def tier_enabled(candidate: PoseCandidate, tier: int) -> bool:
                if int(candidate.metadata.get("search_tier", 0)) > tier:
                    return False
                parent = candidate.metadata.get("refinement_parent_id")
                return parent is None or (
                    active_refinement_parent_ids is not None
                    and str(parent) in active_refinement_parent_ids
                )
            screened_place_ids: set[str] = set()
            screened_grasp_ids: set[str] = set()
            preplace_feasible: frozenset[str] = frozenset()
            place_feasible: frozenset[str] = frozenset()
            pregrasp_feasible: frozenset[str] = frozenset()
            grasp_feasible: frozenset[str] = frozenset()
            complete_places_by_grasp: dict[str, tuple[PoseCandidate, ...]] = {}
            relation_grasp_candidates: tuple[PoseCandidate, ...] = ()
            selected_search_tier = max_tier
            screen_started = time.perf_counter()
            for tier in tiers:
                cumulative_places = tuple(
                    item
                    for item in all_place_candidates
                    if tier_enabled(item, tier)
                )
                new_places = tuple(
                    item
                    for item in cumulative_places
                    if item.candidate_id not in screened_place_ids
                )
                if callable(prepare_coarse) and new_places:
                    prepare_coarse(
                        tuple(preplace_by_place_id[item.candidate_id] for item in new_places),
                        task.scene,
                        tool_frame=task.tool_frame,
                        ignore_object_names=contact_ignores,
                    )
                    prepare_coarse(
                        new_places,
                        task.scene,
                        tool_frame=task.tool_frame,
                        ignore_object_names=contact_ignores,
                    )
                screened_place_ids.update(item.candidate_id for item in new_places)
                cumulative_preplaces = tuple(
                    preplace_by_place_id[item.candidate_id] for item in cumulative_places
                )
                if callable(feasible_pose_ids):
                    preplace_feasible = feasible_pose_ids(cumulative_preplaces)
                    place_feasible = feasible_pose_ids(cumulative_places)
                else:
                    preplace_feasible = frozenset(
                        item.candidate_id for item in cumulative_preplaces
                    )
                    place_feasible = frozenset(item.candidate_id for item in cumulative_places)

                place_ready_grasps = tuple(
                    grasp
                    for grasp in task.grasp_candidates
                    if tier_enabled(grasp, tier)
                    and any(
                        place.candidate_id in place_feasible
                        and f"preplace:{place.candidate_id}" in preplace_feasible
                        for place in task.places_for_grasp(grasp.candidate_id)
                    )
                )
                new_grasps = tuple(
                    item
                    for item in place_ready_grasps
                    if item.candidate_id not in screened_grasp_ids
                )
                if callable(prepare_coarse) and new_grasps:
                    new_pregrasps = _offset_candidates(
                        new_grasps, abs(float(task.grasp_approach_offset)), "pregrasp"
                    )
                    prepare_coarse(
                        new_pregrasps,
                        task.scene,
                        tool_frame=task.tool_frame,
                        ignore_object_names=grasp_ignores,
                    )
                    prepare_coarse(
                        new_grasps,
                        task.scene,
                        tool_frame=task.tool_frame,
                        ignore_object_names=grasp_ignores,
                    )
                screened_grasp_ids.update(item.candidate_id for item in new_grasps)
                cumulative_pregrasps = _offset_candidates(
                    place_ready_grasps,
                    abs(float(task.grasp_approach_offset)),
                    "pregrasp",
                )
                if callable(feasible_pose_ids):
                    pregrasp_feasible = feasible_pose_ids(cumulative_pregrasps)
                    grasp_feasible = feasible_pose_ids(place_ready_grasps)
                else:
                    pregrasp_feasible = frozenset(
                        item.candidate_id for item in cumulative_pregrasps
                    )
                    grasp_feasible = frozenset(
                        item.candidate_id for item in place_ready_grasps
                    )

                complete_places_by_grasp = {}
                for grasp_candidate in place_ready_grasps:
                    grasp_id = grasp_candidate.candidate_id
                    if (
                        grasp_id not in grasp_feasible
                        or f"pregrasp:{grasp_id}" not in pregrasp_feasible
                    ):
                        complete_places_by_grasp[grasp_id] = ()
                        continue
                    complete_places_by_grasp[grasp_id] = tuple(
                        place
                        for place in task.places_for_grasp(grasp_id)
                        if tier_enabled(place, tier)
                        and place.candidate_id in place_feasible
                        and f"preplace:{place.candidate_id}" in preplace_feasible
                    )
                relation_grasp_candidates = tuple(
                    candidate
                    for candidate in place_ready_grasps
                    if complete_places_by_grasp.get(candidate.candidate_id)
                )
                if relation_grasp_candidates:
                    boundary_relations = tuple(
                        candidate
                        for candidate in relation_grasp_candidates
                        if abs(float(candidate.metadata.get("axis_shift_m", 0.0)))
                        >= 0.035
                        and candidate.candidate_id in available_refinement_parent_ids
                        and refinement_tier_by_parent.get(candidate.candidate_id, 0)
                        > tier
                    )
                    if tier < max_tier and boundary_relations:
                        active_refinement_parent_ids = {
                            item.candidate_id
                            for item in sorted(
                                boundary_relations,
                                key=lambda item: item.score,
                                reverse=True,
                            )[:4]
                        }
                        continue
                    selected_search_tier = tier
                    break
                if tier < max_tier:
                    metrics_reader = getattr(
                        self.planner, "pose_candidate_metrics", None
                    )
                    if callable(metrics_reader) and place_ready_grasps:
                        grasp_metrics = metrics_reader(place_ready_grasps)
                        pregrasp_metrics = metrics_reader(cumulative_pregrasps)

                        def residual(candidate: PoseCandidate) -> float:
                            grasp_metric = grasp_metrics.get(candidate.candidate_id, {})
                            pregrasp_metric = pregrasp_metrics.get(
                                f"pregrasp:{candidate.candidate_id}", {}
                            )
                            total = 0.0
                            for metric in (grasp_metric, pregrasp_metric):
                                if not metric:
                                    total += 1.0e9
                                    continue
                                if not bool(metric.get("constraint_feasible", False)):
                                    total += 1.0e6
                                total += float(metric.get("normalized_pose_gap", 1.0e8))
                            return total

                        refinable = tuple(
                            item
                            for item in place_ready_grasps
                            if item.candidate_id in available_refinement_parent_ids
                            and refinement_tier_by_parent.get(item.candidate_id, 0)
                            > tier
                        )
                        nearest = sorted(refinable, key=residual)[:4]
                        active_refinement_parent_ids = {
                            item.candidate_id for item in nearest
                        }
            screen_time_s = time.perf_counter() - screen_started
            relation_screen = {
                "candidate_count": len(task.grasp_candidates),
                "unique_place_candidate_count": len(all_place_candidates),
                "screened_grasp_candidate_count": len(screened_grasp_ids),
                "screened_place_candidate_count": len(screened_place_ids),
                "search_tier": selected_search_tier,
                "grasp_feasible_count": len(grasp_feasible),
                "pregrasp_feasible_count": len(pregrasp_feasible),
                "preplace_feasible_count": len(preplace_feasible),
                "place_feasible_count": len(place_feasible),
                "complete_relation_count": len(relation_grasp_candidates),
                "refinement_parent_ids": sorted(
                    active_refinement_parent_ids or ()
                ),
                "screen_time_s": screen_time_s,
            }
            if coarse_screening_active:
                end_coarse()
                coarse_screening_active = False
            if not relation_grasp_candidates:
                return PickPlaceRunResult(
                    False,
                    (),
                    failure_stage="relation_screen",
                    message="no grasp relation has feasible preplace and place endpoints",
                    diagnostics={
                        "relation_screen": relation_screen
                    },
                )
            grasp_scores = {
                item.candidate_id: float(item.score)
                + max(
                    (
                        0.1 * float(place.score)
                        for place in complete_places_by_grasp[item.candidate_id]
                    ),
                    default=float("-inf"),
                )
                for item in relation_grasp_candidates
            }
            return self._run_segmented_chain(
                task,
                relation_grasp_candidates,
                complete_places_by_grasp,
                grasp_scores,
                relation_screen,
            )
        finally:
            if coarse_screening_active:
                end_coarse()
