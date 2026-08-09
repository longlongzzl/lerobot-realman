"""cuRobo v2 adapter behind planner-independent application contracts."""

from __future__ import annotations

import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rm75_app.paths import APP_ROOT

from ..contracts import (
    BatchPlanningRequest,
    BatchPlanningResult,
    CandidatePlan,
    CollisionObject,
    GraspCandidatePlan,
    GraspPlanningRequest,
    GraspPlanningResult,
    JointConfiguration,
    JointTrajectory,
    PlanningScene,
)


DEFAULT_CUROBO2_ROOT = Path("/home/zhangzhao/PycharmProjects/curobo2/curobo")
DEFAULT_RM75_CONFIG = APP_ROOT / "assets/curobo_rm75_config/rm75.yml"


@dataclass(frozen=True)
class Curobo2BackendConfig:
    curobo_root: Path = DEFAULT_CUROBO2_ROOT
    robot_config: Path = DEFAULT_RM75_CONFIG
    device: str = "cuda:0"
    max_batch_size: int = 32
    max_goalset: int = 32
    num_ik_seeds: int = 32
    num_trajopt_seeds: int = 4
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.05
    collision_activation_distance: float = 0.01
    use_cuda_graph: bool = False
    multi_env: bool = False
    attachment_num_spheres: int = 16


def load_curobo2_robot_config(path: str | Path) -> dict[str, Any]:
    """Translate the app's v1 RM75 YAML into the v2 robot schema in memory."""

    config_path = Path(path).expanduser().resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid robot config: {config_path}")
    result = copy.deepcopy(data)
    robot = result.get("robot_cfg", result)
    kinematics = robot["kinematics"]

    for stale_key in (
        "use_usd_kinematics",
        "usd_path",
        "usd_robot_root",
        "isaac_usd_path",
        "usd_flip_joints",
        "usd_flip_joint_limits",
    ):
        kinematics.pop(stale_key, None)

    ee_link = str(kinematics.pop("ee_link", "gripper_tcp"))
    # v1 ``link_names`` meant additional FK outputs.  In v2 every
    # ``tool_frames`` entry is a required goal constraint, so carrying that
    # list across would incorrectly require callers to target every finger.
    kinematics.pop("link_names", None)
    kinematics["tool_frames"] = [ee_link]
    kinematics["format_version"] = 2.0
    kinematics.setdefault(
        "grasp_contact_link_names",
        ["left_pad", "right_pad", "attached_object"],
    )

    cspace = kinematics["cspace"]
    if "default_joint_position" not in cspace and "retract_config" in cspace:
        cspace["default_joint_position"] = cspace.pop("retract_config")

    spheres = kinematics.get("collision_spheres")
    if isinstance(spheres, str):
        sphere_path = Path(spheres).expanduser()
        if not sphere_path.is_absolute():
            sphere_path = config_path.parent / sphere_path
        sphere_data = yaml.safe_load(sphere_path.resolve().read_text(encoding="utf-8"))
        kinematics["collision_spheres"] = sphere_data["collision_spheres"]
    return result


class Curobo2Backend:
    """Batch candidate planner using the public cuRobo v2 API."""

    name = "curobo2"

    def __init__(self, config: Curobo2BackendConfig | None = None):
        self.config = config or Curobo2BackendConfig()
        self._planner = None
        self._modules: dict[str, Any] | None = None
        self._scene = PlanningScene()
        self._attachment_active = False
        self._pose_ik_cache: dict[tuple[float, ...], np.ndarray] = {}

    @staticmethod
    def _pose_cache_key(candidate: Any) -> tuple[float, ...]:
        values = np.concatenate(
            [candidate.pose.position, candidate.pose.quaternion_wxyz]
        )
        return tuple(np.round(values, decimals=8).tolist())

    def _import_modules(self) -> dict[str, Any]:
        if self._modules is not None:
            return self._modules
        root = str(self.config.curobo_root.expanduser().resolve())
        loaded = sys.modules.get("curobo")
        if loaded is not None:
            loaded_path = str(Path(loaded.__file__).resolve())
            if not loaded_path.startswith(root):
                raise RuntimeError(
                    "a different cuRobo version is already loaded; launch the Curobo2 "
                    f"entrypoint in a clean process (loaded {loaded_path})"
                )
        if root not in sys.path:
            sys.path.insert(0, root)

        import torch
        from curobo.batch_motion_planner import BatchMotionPlanner
        from curobo.motion_planner import MotionPlannerCfg
        from curobo.scene import Cuboid, Mesh, Scene, Sphere
        from curobo.types import DeviceCfg, Pose as CuroboPose
        from curobo._src.state.state_joint import JointState
        from curobo._src.state.state_joint_trajectory_ops import get_joint_state_at_horizon_index
        from curobo._src.types.tool_pose import GoalToolPose

        self._modules = {
            "torch": torch,
            "BatchMotionPlanner": BatchMotionPlanner,
            "MotionPlannerCfg": MotionPlannerCfg,
            "Cuboid": Cuboid,
            "Mesh": Mesh,
            "Scene": Scene,
            "Sphere": Sphere,
            "DeviceCfg": DeviceCfg,
            "CuroboPose": CuroboPose,
            "JointState": JointState,
            "get_joint_state_at_horizon_index": get_joint_state_at_horizon_index,
            "GoalToolPose": GoalToolPose,
        }
        return self._modules

    def _ensure_planner(self) -> Any:
        if self._planner is not None:
            return self._planner
        modules = self._import_modules()
        device_cfg = modules["DeviceCfg"](device=self.config.device)
        robot_config = load_curobo2_robot_config(self.config.robot_config)
        kinematics = robot_config["robot_cfg"]["kinematics"]
        extra_spheres = kinematics.setdefault("extra_collision_spheres", {})
        extra_spheres["attached_object"] = max(
            int(extra_spheres.get("attached_object", 0)),
            int(self.config.attachment_num_spheres),
        )
        planner_cfg = modules["MotionPlannerCfg"].create(
            robot=robot_config,
            scene_model=self._to_curobo_scene(self._scene),
            collision_cache={"obb": 128, "mesh": 128},
            device_cfg=device_cfg,
            max_batch_size=self.config.max_batch_size,
            max_goalset=self.config.max_goalset,
            multi_env=self.config.multi_env,
            num_ik_seeds=self.config.num_ik_seeds,
            num_trajopt_seeds=self.config.num_trajopt_seeds,
            position_tolerance=self.config.position_tolerance,
            orientation_tolerance=self.config.orientation_tolerance,
            optimizer_collision_activation_distance=self.config.collision_activation_distance,
            use_cuda_graph=self.config.use_cuda_graph,
        )
        self._planner = modules["BatchMotionPlanner"](planner_cfg)
        return self._planner

    def _to_curobo_obstacle(self, item: CollisionObject) -> Any:
        modules = self._import_modules()
        common = {"name": item.name, "pose": item.pose.as_curobo_list()}
        if item.kind == "cuboid":
            return modules["Cuboid"](**common, dims=item.dimensions.tolist())
        if item.kind == "sphere":
            return modules["Sphere"](**common, radius=float(item.radius))
        return modules["Mesh"](
            **common,
            file_path=str(item.mesh_path),
            scale=None if item.scale is None else item.scale.tolist(),
        )

    def _to_curobo_scene(self, scene: PlanningScene) -> Any:
        modules = self._import_modules()
        grouped: dict[str, list[Any]] = {"cuboid": [], "sphere": [], "mesh": []}
        for item in scene.objects:
            grouped[item.kind].append(self._to_curobo_obstacle(item))
        return modules["Scene"](**grouped)

    def update_scene(self, scene: PlanningScene) -> None:
        self._scene = scene
        if self._planner is not None:
            self._planner.update_world(self._to_curobo_scene(scene))

    def _set_obstacle_enabled(self, name: str, enabled: bool) -> None:
        """Work around cuRobo2's mixed-type enable lookup raising ValueError."""
        planner = self._ensure_planner()
        collision = planner.scene_collision_checker
        if collision is None:
            raise RuntimeError("collision scene is not configured")
        for store_name in ("cuboids", "meshes", "voxels"):
            store = getattr(collision.data, store_name, None)
            if store is None:
                continue
            names = getattr(store, "names", None)
            if names and str(name) in names[0]:
                store.set_enabled(str(name), bool(enabled), 0)
                return
        raise KeyError(f"obstacle {name!r} is not present in the cuRobo scene")

    def _update_obstacle_pose(self, name: str, pose: Any) -> None:
        planner = self._ensure_planner()
        collision = planner.scene_collision_checker
        for store_name in ("cuboids", "meshes", "voxels"):
            store = getattr(collision.data, store_name, None)
            names = None if store is None else getattr(store, "names", None)
            if names and str(name) in names[0]:
                store.update_pose(str(name), w_obj_pose=pose, env_idx=0)
                return
        raise KeyError(f"obstacle {name!r} is not present in the cuRobo scene")

    def _attachment_manager(self) -> Any:
        planner = self._ensure_planner()
        # cuRobo 0.8's public property currently points one level too high;
        # SolverCore owns the manager in this checkout.
        manager = getattr(planner.trajopt_solver, "attachment_manager", None)
        if manager is None:
            manager = planner.trajopt_solver.core.attachment_manager
        return manager

    @staticmethod
    def _batch_values(value: Any, count: int, default: float | None = None) -> list[Any]:
        if value is None:
            return [default] * count
        array = value.detach().cpu().numpy()
        if array.ndim > 1:
            array = array[:, 0]
        return array.reshape(-1)[:count].tolist()

    def plan_candidates(self, request: BatchPlanningRequest) -> BatchPlanningResult:
        count = len(request.candidates)
        if count > self.config.max_batch_size:
            raise ValueError(
                f"received {count} candidates, maximum batch size is {self.config.max_batch_size}"
            )
        if request.scene is not self._scene:
            self.update_scene(request.scene)
        planner = self._ensure_planner()
        modules = self._import_modules()
        torch = modules["torch"]
        device = planner.device_cfg.device
        solver_batch_size = planner.batch_size

        current_position = torch.as_tensor(
            request.current.positions, dtype=planner.device_cfg.dtype, device=device
        ).unsqueeze(0).repeat(solver_batch_size, 1)
        current = modules["JointState"].from_position(
            current_position, joint_names=list(request.current.names)
        )
        candidate_positions = np.stack([item.pose.position for item in request.candidates])
        candidate_quaternions = np.stack(
            [item.pose.quaternion_wxyz for item in request.candidates]
        )
        # cuRobo v2 advertises short batches, but its graph-seeding path still
        # reshapes to max_batch_size. Pad with the final candidate and discard
        # those results to keep one genuine solver invocation.
        if count < solver_batch_size:
            padding = solver_batch_size - count
            candidate_positions = np.concatenate(
                [candidate_positions, np.repeat(candidate_positions[-1:], padding, axis=0)]
            )
            candidate_quaternions = np.concatenate(
                [candidate_quaternions, np.repeat(candidate_quaternions[-1:], padding, axis=0)]
            )
        positions = torch.as_tensor(
            candidate_positions,
            dtype=planner.device_cfg.dtype,
            device=device,
        )[:, None, None, None, :]
        quaternions = torch.as_tensor(
            candidate_quaternions,
            dtype=planner.device_cfg.dtype,
            device=device,
        )[:, None, None, None, :]
        goals = modules["GoalToolPose"](
            tool_frames=[request.tool_frame], position=positions, quaternion=quaternions
        )

        started = time.perf_counter()
        raw = None
        unbiased_ik_success = None

        def plan_with_unbiased_ik() -> Any:
            nonlocal unbiased_ik_success
            # cuRobo2 can over-regularize IK around a supplied near-goal
            # current state (common for lift/descent). Reuse one unbiased
            # batched IK solve, then plan all resulting joint goals in one
            # batched C-space call.
            cached = [self._pose_ik_cache.get(self._pose_cache_key(item)) for item in request.candidates]
            if all(item is not None for item in cached):
                cached_positions = np.stack(cached)
                if count < solver_batch_size:
                    cached_positions = np.concatenate(
                        [
                            cached_positions,
                            np.repeat(cached_positions[-1:], solver_batch_size - count, axis=0),
                        ]
                    )
                goal_positions = torch.as_tensor(
                    cached_positions, dtype=planner.device_cfg.dtype, device=device
                ).contiguous()
                unbiased_ik_success = torch.ones(
                    solver_batch_size, dtype=torch.bool, device=device
                )
                goal_states = modules["JointState"].from_position(
                    goal_positions, joint_names=list(request.current.names)
                )
                return planner.plan_cspace(
                    goal_states, current, max_attempts=request.max_attempts
                )

            planner.ik_solver.reset_seed()
            suspended_spheres: list[tuple[Any, Any, Any]] = []
            if self._attachment_active:
                parameter_sets = [
                    self._attachment_manager().kinematics_params,
                    planner.ik_solver.kinematics.kinematics_config,
                ]
                seen: set[int] = set()
                for params in parameter_sets:
                    if id(params) in seen:
                        continue
                    seen.add(id(params))
                    indices = params.get_sphere_index_from_link_name("attached_object")
                    saved = params.link_spheres[:, indices, :].clone()
                    params.link_spheres[:, indices, 3] = -100.0
                    suspended_spheres.append((params, indices, saved))
            try:
                ik_result = planner.ik_solver.solve_pose(
                    goals,
                    return_seeds=planner.trajopt_solver.config.num_seeds,
                    current_state=None,
                )
            finally:
                for params, indices, saved in suspended_spheres:
                    params.link_spheres[:, indices, :] = saved
            ik_success = ik_result.success.reshape(solver_batch_size, -1)
            unbiased_ik_success = ik_success.any(dim=-1)
            if bool(unbiased_ik_success.any().item()):
                # Each candidate owns a set of returned IK seeds. Selecting
                # seed zero after ``any(success)`` silently feeds an invalid
                # joint target to TrajOpt whenever a later seed is the winner.
                solutions = ik_result.solution.reshape(
                    solver_batch_size, -1, ik_result.solution.shape[-1]
                )
                winning_seed = ik_success.to(dtype=torch.int64).argmax(dim=-1)
                batch_index = torch.arange(solver_batch_size, device=device)
                goal_positions = solutions[batch_index, winning_seed].contiguous()
                goal_states = modules["JointState"].from_position(
                    goal_positions, joint_names=list(request.current.names)
                )
                return planner.plan_cspace(
                    goal_states,
                    current,
                    max_attempts=request.max_attempts,
                )
            return None

        if request.prefer_unbiased_ik:
            raw = plan_with_unbiased_ik()
        if raw is None and not request.prefer_unbiased_ik:
            raw = planner.plan_pose(goals, current, max_attempts=request.max_attempts)
        if raw is None and unbiased_ik_success is None:
            raw = plan_with_unbiased_ik()
        elapsed = time.perf_counter() - started
        if raw is None:
            return BatchPlanningResult(
                plans=tuple(
                    CandidatePlan(item.candidate_id, False, status="ik_failed")
                    for item in request.candidates
                ),
                backend=self.name,
                total_time=elapsed,
            )

        success_tensor = raw.success
        if unbiased_ik_success is not None:
            success_tensor = raw.success.any(dim=-1) & unbiased_ik_success
        success_array = success_tensor.detach().cpu().numpy()
        if success_array.ndim > 1:
            success_array = success_array.any(axis=tuple(range(1, success_array.ndim)))
        success_array = success_array.reshape(-1)
        winner_indices: list[int] = []
        for candidate_index in range(count):
            pool = [candidate_index]
            if candidate_index == count - 1:
                pool = list(range(candidate_index, solver_batch_size))
            winner_indices.append(next((idx for idx in pool if success_array[idx]), candidate_index))
        success = np.asarray([success_array[idx] for idx in winner_indices], dtype=bool)
        position_error_all = self._batch_values(
            getattr(raw, "position_error", None), solver_batch_size
        )
        rotation_error_all = self._batch_values(
            getattr(raw, "rotation_error", None), solver_batch_size
        )
        trajectory_state = raw.interpolated_trajectory
        if trajectory_state is None:
            trajectory_state = raw.js_solution
        positions_np = None
        trajectory_joint_names = tuple(request.current.names)
        if trajectory_state is not None:
            positions_np = trajectory_state.position.detach().cpu().numpy()
            raw_joint_names = tuple(getattr(trajectory_state, "joint_names", ()) or ())
            if raw_joint_names:
                trajectory_joint_names = raw_joint_names
            if positions_np.ndim == 4:
                positions_np = positions_np[:, 0]

        plans: list[CandidatePlan] = []
        for index, candidate in enumerate(request.candidates):
            result_index = winner_indices[index]
            trajectory = None
            if bool(success[index]) and positions_np is not None:
                path = np.asarray(positions_np[result_index])
                # cuRobo keeps a return-seed dimension in some result modes.
                # The solver has already ranked seeds, so consume the first
                # remaining seed until the contract is [time, joints].
                while path.ndim > 2:
                    path = path[0]
                output_names = trajectory_joint_names
                if path.shape[-1] != len(output_names):
                    raise RuntimeError(
                        "cuRobo trajectory joint metadata does not match its data: "
                        f"{path.shape[-1]} columns versus {len(output_names)} names"
                    )
                if output_names != tuple(request.current.names):
                    active_indices = [output_names.index(name) for name in request.current.names]
                    path = path[:, active_indices]
                    output_names = tuple(request.current.names)
                trajectory = JointTrajectory(output_names, path)
            if bool(success[index]):
                status = "success"
            elif unbiased_ik_success is not None and not bool(
                unbiased_ik_success[result_index].item()
            ):
                status = "ik_failed"
            else:
                status = str(getattr(raw, "status", None) or "trajectory_failed")
            plans.append(
                CandidatePlan(
                    candidate_id=candidate.candidate_id,
                    success=bool(success[index]),
                    trajectory=trajectory,
                    position_error=position_error_all[result_index],
                    orientation_error=rotation_error_all[result_index],
                    solve_time=float(getattr(raw, "total_time", elapsed) or elapsed),
                    status=status,
                )
            )
        return BatchPlanningResult(tuple(plans), backend=self.name, total_time=elapsed)

    def prepare_pose_candidates(
        self,
        candidates: tuple[Any, ...],
        scene: PlanningScene,
        *,
        tool_frame: str = "gripper_tcp",
        ignore_object_name: str | None = None,
    ) -> None:
        """Batch IK before attachment; later transport planning reuses the joint goals."""
        if scene is not self._scene:
            self.update_scene(scene)
        planner = self._ensure_planner()
        modules = self._import_modules()
        torch = modules["torch"]
        count = len(candidates)
        if count > planner.batch_size:
            raise ValueError(f"received {count} candidates, maximum batch size is {planner.batch_size}")
        positions = np.stack([item.pose.position for item in candidates])
        quaternions = np.stack([item.pose.quaternion_wxyz for item in candidates])
        if count < planner.batch_size:
            positions = np.concatenate(
                [positions, np.repeat(positions[-1:], planner.batch_size - count, axis=0)]
            )
            quaternions = np.concatenate(
                [quaternions, np.repeat(quaternions[-1:], planner.batch_size - count, axis=0)]
            )
        goals = modules["GoalToolPose"](
            tool_frames=[tool_frame],
            position=torch.as_tensor(
                positions, dtype=planner.device_cfg.dtype, device=planner.device_cfg.device
            )[:, None, None, None, :],
            quaternion=torch.as_tensor(
                quaternions, dtype=planner.device_cfg.dtype, device=planner.device_cfg.device
            )[:, None, None, None, :],
        )
        if ignore_object_name:
            self._set_obstacle_enabled(ignore_object_name, False)
        try:
            planner.ik_solver.reset_seed()
            result = planner.ik_solver.solve_pose(
                goals,
                current_state=None,
                return_seeds=planner.trajopt_solver.config.num_seeds,
            )
        finally:
            if ignore_object_name:
                self._set_obstacle_enabled(ignore_object_name, True)
        success = result.success.reshape(planner.batch_size, -1)
        solutions = result.solution.reshape(
            planner.batch_size, -1, result.solution.shape[-1]
        )
        winner = success.to(dtype=torch.int64).argmax(dim=-1)
        batch_index = torch.arange(planner.batch_size, device=planner.device_cfg.device)
        selected = solutions[batch_index, winner].detach().cpu().numpy()
        for index, candidate in enumerate(candidates):
            if bool(success[index].any().item()):
                self._pose_ik_cache[self._pose_cache_key(candidate)] = selected[index]

    def _make_batch_inputs(self, request: BatchPlanningRequest) -> tuple[Any, Any]:
        planner = self._ensure_planner()
        modules = self._import_modules()
        torch = modules["torch"]
        count = len(request.candidates)
        batch_size = planner.batch_size
        if count > batch_size:
            raise ValueError(f"received {count} candidates, maximum batch size is {batch_size}")

        current_position = torch.as_tensor(
            request.current.positions,
            dtype=planner.device_cfg.dtype,
            device=planner.device_cfg.device,
        ).unsqueeze(0).repeat(batch_size, 1)
        current = modules["JointState"].from_position(
            current_position, joint_names=list(request.current.names)
        )
        positions = np.stack([item.pose.position for item in request.candidates])
        quaternions = np.stack([item.pose.quaternion_wxyz for item in request.candidates])
        if count < batch_size:
            padding = batch_size - count
            positions = np.concatenate([positions, np.repeat(positions[-1:], padding, axis=0)])
            quaternions = np.concatenate(
                [quaternions, np.repeat(quaternions[-1:], padding, axis=0)]
            )
        goals = modules["GoalToolPose"](
            tool_frames=[request.tool_frame],
            position=torch.as_tensor(
                positions, dtype=planner.device_cfg.dtype, device=planner.device_cfg.device
            )[:, None, None, None, :],
            quaternion=torch.as_tensor(
                quaternions, dtype=planner.device_cfg.dtype, device=planner.device_cfg.device
            )[:, None, None, None, :],
        )
        return current, goals

    @staticmethod
    def _trajectory_for_candidate(
        state: Any, index: int, active_joint_names: tuple[str, ...]
    ) -> JointTrajectory | None:
        if state is None:
            return None
        path = np.asarray(state.position.detach().cpu().numpy()[index])
        while path.ndim > 2:
            path = path[0]
        names = tuple(getattr(state, "joint_names", ()) or active_joint_names)
        if path.ndim != 2 or path.shape[-1] != len(names):
            raise RuntimeError(
                f"cuRobo trajectory shape {path.shape} does not match {len(names)} joint names"
            )
        if names != active_joint_names:
            indices = [names.index(name) for name in active_joint_names]
            path = path[:, indices]
        return JointTrajectory(active_joint_names, path)

    def plan_grasps(self, request: GraspPlanningRequest) -> GraspPlanningResult:
        """Batch-screen approach and grasp stages in one cuRobo v2 call.

        Lift is deliberately deferred: the coordinator must attach the selected
        object collision geometry at the grasp state before planning lift.
        """

        planning = request.planning
        if planning.scene is not self._scene:
            self.update_scene(planning.scene)
        planner = self._ensure_planner()
        modules = self._import_modules()
        current, goals = self._make_batch_inputs(planning)
        started = time.perf_counter()
        target_name = request.target_object_name
        fallback_grasp_result = None
        if target_name and planner.scene_collision_checker is not None:
            self._set_obstacle_enabled(target_name, False)
        try:
            raw = planner.plan_grasp(
                goals,
                current,
                grasp_approach_axis=request.approach_axis,
                grasp_approach_offset=request.approach_offset,
                grasp_approach_in_tool_frame=request.approach_in_tool_frame,
                plan_approach_to_grasp=True,
                plan_grasp_to_lift=False,
            )
            # v2's linear-motion criterion is intentionally strict and can
            # reject otherwise collision-free tabletop insertions. Keep the
            # batched approach result, then perform one batched unconstrained
            # grasp solve from each candidate's own approach endpoint.
            if (
                not bool(raw.success.any().item())
                and getattr(raw, "approach_result", None) is not None
                and raw.approach_result.js_solution is not None
            ):
                approach_end = modules["get_joint_state_at_horizon_index"](
                    raw.approach_result.js_solution, -1
                ).squeeze(1)
                approach_end = planner.kinematics.get_active_js(approach_end)
                grasp_end = modules["get_joint_state_at_horizon_index"](
                    raw.goalset_result.js_solution, -1
                ).squeeze(1)
                grasp_end = planner.kinematics.get_active_js(grasp_end)
                fallback_grasp_result = planner.plan_cspace(
                    grasp_end,
                    approach_end,
                    max_attempts=planning.max_attempts,
                    enable_graph_attempt=planning.max_attempts + 1,
                )
        finally:
            if target_name and planner.scene_collision_checker is not None:
                self._set_obstacle_enabled(target_name, True)
        elapsed = time.perf_counter() - started
        count = len(planning.candidates)
        success_source = raw.success
        if fallback_grasp_result is not None:
            fallback_success = fallback_grasp_result.success.any(dim=-1)
            success_source = fallback_success & raw.approach_success
        success = np.asarray(self._batch_values(success_source, count, False), dtype=bool)
        approach_state = raw.approach_interpolated_trajectory
        if approach_state is None:
            approach_state = raw.approach_trajectory
        grasp_state = raw.grasp_interpolated_trajectory
        if grasp_state is None:
            grasp_state = raw.grasp_trajectory
        if grasp_state is None and fallback_grasp_result is not None:
            grasp_state = fallback_grasp_result.interpolated_trajectory
            if grasp_state is None:
                grasp_state = fallback_grasp_result.js_solution
        names = tuple(planning.current.names)
        plans = tuple(
            GraspCandidatePlan(
                candidate_id=candidate.candidate_id,
                success=bool(success[index]),
                approach=self._trajectory_for_candidate(approach_state, index, names),
                grasp=self._trajectory_for_candidate(grasp_state, index, names),
                status=(
                    "approach + batched grasp fallback"
                    if fallback_grasp_result is not None and bool(success[index])
                    else raw.status
                ),
            )
            for index, candidate in enumerate(planning.candidates)
        )
        return GraspPlanningResult(plans, backend=self.name, total_time=elapsed)

    def attach_object(self, object_name: str, grasp: JointConfiguration) -> None:
        planner = self._ensure_planner()
        modules = self._import_modules()
        torch = modules["torch"]
        position = torch.as_tensor(
            grasp.positions, dtype=planner.device_cfg.dtype, device=planner.device_cfg.device
        ).unsqueeze(0)
        state = modules["JointState"].from_position(position, joint_names=list(grasp.names))
        object_world_pose = self._scene_object_pose(object_name)
        self._attach_at_world_pose(object_name, state, object_world_pose)

    def _scene_object_pose(self, object_name: str) -> Any:
        planner = self._ensure_planner()
        modules = self._import_modules()
        scene_model = planner.scene_collision_checker.scene_model
        if isinstance(scene_model, list):
            scene_model = scene_model[0]
        obstacle = scene_model.get_obstacle(str(object_name))
        if obstacle is None:
            raise KeyError(f"obstacle {object_name!r} is missing from the scene")
        return modules["CuroboPose"].from_list(
            list(obstacle.pose), planner.device_cfg
        )

    def _attach_at_world_pose(self, object_name: str, state: Any, object_world_pose: Any) -> None:
        planner = self._ensure_planner()
        scene_model = planner.scene_collision_checker.scene_model
        if isinstance(scene_model, list):
            scene_model = scene_model[0]
        obstacle = scene_model.get_obstacle(str(object_name))
        if obstacle is None:
            raise KeyError(f"obstacle {object_name!r} is missing from the scene")
        local_obstacle = copy.deepcopy(obstacle)
        local_obstacle.pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        if self.config.attachment_num_spheres > 0:
            self._attachment_manager().attach(
                state,
                [local_obstacle],
                link_name="attached_object",
                num_spheres=int(self.config.attachment_num_spheres),
                world_objects_pose_offset=object_world_pose,
            )
            self._attachment_active = True
        self._set_obstacle_enabled(str(object_name), False)

    def update_attached_object_pose(
        self,
        object_name: str,
        current: JointConfiguration,
        T_tcp_object: np.ndarray,
    ) -> None:
        """Re-fit the held collision geometry using a wrist-refined TCP-relative pose."""
        if not self._attachment_active:
            raise RuntimeError("cannot refine an object that is not attached")
        transform = np.asarray(T_tcp_object, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("T_tcp_object must be a finite 4x4 transform")
        planner = self._ensure_planner()
        modules = self._import_modules()
        torch = modules["torch"]
        position = torch.as_tensor(
            current.positions,
            dtype=planner.device_cfg.dtype,
            device=planner.device_cfg.device,
        ).unsqueeze(0)
        state = modules["JointState"].from_position(
            position, joint_names=list(current.names)
        )
        kinematics = planner.compute_kinematics(state)
        tcp_pose = kinematics.tool_poses.get_link_pose(planner.tool_frames[0])
        relative_pose = modules["CuroboPose"].from_matrix(
            torch.as_tensor(
                transform,
                dtype=planner.device_cfg.dtype,
                device=planner.device_cfg.device,
            )
        )
        object_world_pose = tcp_pose.multiply(relative_pose)
        self._attachment_manager().detach(link_name="attached_object")
        self._attachment_active = False
        self._update_obstacle_pose(str(object_name), object_world_pose)
        self._attach_at_world_pose(object_name, state, object_world_pose)

    def detach_object(self, object_name: str, released_pose: Any | None = None) -> None:
        if self._planner is not None:
            if self._attachment_active:
                self._attachment_manager().detach(link_name="attached_object")
                self._attachment_active = False
            if released_pose is not None:
                modules = self._import_modules()
                self._update_obstacle_pose(
                    str(object_name),
                    modules["CuroboPose"].from_list(
                        released_pose.as_curobo_list(), self._planner.device_cfg
                    ),
                )
            if released_pose is None:
                self._set_obstacle_enabled(str(object_name), True)

    def enable_object_collision(self, object_name: str) -> None:
        self._set_obstacle_enabled(str(object_name), True)

    def close(self) -> None:
        if self._planner is not None:
            self._planner.destroy()
            self._planner = None

    def __enter__(self) -> "Curobo2Backend":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
