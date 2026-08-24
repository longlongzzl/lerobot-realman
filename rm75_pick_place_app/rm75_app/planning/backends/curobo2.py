"""cuRobo v2 adapter behind planner-independent application contracts."""

from __future__ import annotations

import copy
import hashlib
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rm75_app.paths import APP_ROOT, RUNTIME_DIR

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
    # MotionGen plans only the shortlisted free-space and short-linear rows.
    # Keep this batch small on the workstation's 8 GB GPU; broad endpoint
    # screening uses the separate fixed batch64 IK solver below.
    max_batch_size: int = 4
    max_goalset: int = 4
    num_ik_seeds: int = 32
    num_trajopt_seeds: int = 4
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.05
    collision_activation_distance: float = 0.01
    self_collision_check: bool = True
    # cuRobo2's advertised motion-generation latency assumes a warmed,
    # persistent CUDA Graph. Fixed batch shapes make that the normal RM75
    # execution path; callers can still disable it for diagnostics.
    use_cuda_graph: bool = True
    warmup_planner: bool = True
    warmup_iterations: int = 2
    # Match NVIDIA's benchmark routing: try direct TrajOpt first and only
    # invoke PRM on the next attempt when the local optimizer cannot solve.
    enable_graph_attempt: int = 1
    # At 25 ms per waypoint this still covers 12.5 seconds, without carrying
    # cuRobo2's default 5000-point fixed buffer through every batch stage.
    interpolation_buffer_size: int = 500
    collision_cache_obb: int = 32
    collision_cache_mesh: int = 16
    multi_env: bool = False
    # A separate fixed-shape solver accelerates broad reachability screening.
    # Motion planning keeps max_batch_size=4 to stay within the 8 GB budget.
    coarse_ik_batch_size: int = 64
    coarse_ik_num_seeds: int = 32
    coarse_ik_use_cuda_graph: bool = True
    # Compatibility workaround for a cuRobo2 mixed-feasibility batch bug.
    # The segmented mainline pads one shortlisted row to a fixed batch, so it
    # should use the official graph-seeding implementation by default.
    isolate_graph_seed_failures: bool = False
    # Maximum allocated slots. The actual sphere count is estimated by
    # cuRobo2's MORPHIT fitter for every asset.
    attachment_num_spheres: int = 64
    attachment_surface_radius: float = 0.002
    attachment_cache_dir: Path = RUNTIME_DIR / "curobo2_attachment_spheres"
    # cuRobo2 2.x hard-codes 1.0 for the non-terminal linear grasp pose
    # criterion.  On RM75 this over-constrains an otherwise feasible world-Z
    # descent.  0.5 is the strongest value that converges in the fixed-scene
    # validation while retaining the straight approach corridor.
    grasp_linear_non_terminal_scale: float = 0.5


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
        self._coarse_ik_solver = None
        self._modules: dict[str, Any] | None = None
        self._scene = PlanningScene()
        self._attachment_active = False
        self._attachment_fit_report: dict[str, Any] | None = None
        self._pose_ik_cache: dict[tuple[float, ...], np.ndarray] = {}
        self._pose_ik_metrics: dict[tuple[float, ...], dict[str, Any]] = {}

    @property
    def max_batch_size(self) -> int:
        return int(self.config.max_batch_size)

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
        from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
        from curobo._src.geom.sphere_fit import SphereFitType
        from curobo._src.geom.collision import create_collision_checker
        from curobo.motion_planner import MotionPlannerCfg
        from curobo._src.solver.solver_ik import IKSolver
        from curobo.scene import Cuboid, Mesh, Scene, Sphere
        from curobo.types import DeviceCfg, Pose as CuroboPose
        from curobo._src.state.state_joint import JointState
        from curobo._src.state.state_joint_trajectory_ops import get_joint_state_at_horizon_index
        from curobo._src.types.tool_pose import GoalToolPose

        self._modules = {
            "torch": torch,
            "BatchMotionPlanner": BatchMotionPlanner,
            "ToolPoseCriteria": ToolPoseCriteria,
            "SphereFitType": SphereFitType,
            "create_collision_checker": create_collision_checker,
            "MotionPlannerCfg": MotionPlannerCfg,
            "IKSolver": IKSolver,
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
            # Keep planner construction deterministic. In cuRobo2 2.x a mesh
            # supplied at construction can produce different goalset IK results
            # from the same mesh loaded through the public update_world path.
            scene_model=self._to_formal_curobo_scene_dict(PlanningScene()),
            collision_cache={
                "obb": int(self.config.collision_cache_obb),
                "mesh": int(self.config.collision_cache_mesh),
            },
            device_cfg=device_cfg,
            max_batch_size=self.config.max_batch_size,
            max_goalset=self.config.max_goalset,
            multi_env=self.config.multi_env,
            num_ik_seeds=self.config.num_ik_seeds,
            num_trajopt_seeds=self.config.num_trajopt_seeds,
            position_tolerance=self.config.position_tolerance,
            orientation_tolerance=self.config.orientation_tolerance,
            optimizer_collision_activation_distance=self.config.collision_activation_distance,
            self_collision_check=self.config.self_collision_check,
            use_cuda_graph=self.config.use_cuda_graph,
        )
        planner_cfg.trajopt_solver_config.interpolation_buffer_size = max(
            1, int(self.config.interpolation_buffer_size)
        )
        self._planner = modules["BatchMotionPlanner"](planner_cfg)
        if self._scene.objects:
            self._update_formal_world(self._planner, self._scene)
        if self.config.warmup_planner:
            self._planner.warmup(
                enable_graph=True,
                num_warmup_iterations=max(1, int(self.config.warmup_iterations)),
            )
        return self._planner

    def _ensure_coarse_ik_solver(self) -> Any:
        """Create the fixed large-batch solver used only for coarse screening."""
        if self._coarse_ik_solver is not None:
            return self._coarse_ik_solver
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
            scene_model=self._to_curobo_scene(PlanningScene()),
            collision_cache={
                "obb": int(self.config.collision_cache_obb),
                "mesh": int(self.config.collision_cache_mesh),
            },
            device_cfg=device_cfg,
            max_batch_size=int(self.config.coarse_ik_batch_size),
            max_goalset=int(self.config.max_goalset),
            multi_env=False,
            num_ik_seeds=int(self.config.coarse_ik_num_seeds),
            num_trajopt_seeds=self.config.num_trajopt_seeds,
            position_tolerance=self.config.position_tolerance,
            orientation_tolerance=self.config.orientation_tolerance,
            optimizer_collision_activation_distance=self.config.collision_activation_distance,
            self_collision_check=self.config.self_collision_check,
            use_cuda_graph=bool(self.config.coarse_ik_use_cuda_graph),
        )
        collision_checker = modules["create_collision_checker"](
            planner_cfg.scene_collision_cfg
        )
        self._coarse_ik_solver = modules["IKSolver"](
            planner_cfg.ik_solver_config, scene_collision_checker=collision_checker
        )
        if self._scene.objects:
            self._coarse_ik_solver.update_world(self._to_curobo_scene(self._scene))
        return self._coarse_ik_solver

    def begin_coarse_screening(self) -> None:
        """Enter coarse screening without discarding the warmed formal planner."""
        if self._attachment_active:
            raise RuntimeError("cannot screen poses while an object is attached")

    def end_coarse_screening(self) -> None:
        """Leave both fixed-shape solvers resident for the next atom."""

    def _maybe_install_isolated_graph_seeding(
        self,
        planner: Any,
        terminal_records: list[dict[str, Any]] | None = None,
    ) -> Any:
        if not self.config.isolate_graph_seed_failures:
            return lambda: None
        return self._install_isolated_graph_seeding(planner, terminal_records)

    def _to_curobo_obstacle(self, item: CollisionObject) -> Any:
        modules = self._import_modules()
        common = {"name": item.name, "pose": item.pose.as_curobo_list()}
        if item.kind == "cuboid":
            return modules["Cuboid"](**common, dims=item.dimensions.tolist())
        if item.kind == "sphere":
            # SceneCfg exposes Sphere, but cuRobo2 2.x SceneData only stores
            # cuboid/mesh/voxel obstacles.  Its Warp SDF currently classifies
            # points outside an in-memory icosphere as penetrating, which made
            # a tennis ball 0.4 m from the base collide with base_link.  Use the
            # sphere's conservative AABB through the supported OBB path.  A
            # sphere is orientation invariant, so discard its observed rotation.
            diameter = 2.0 * float(item.radius)
            return modules["Cuboid"](
                name=item.name,
                pose=[*item.pose.position.tolist(), 1.0, 0.0, 0.0, 0.0],
                dims=[diameter, diameter, diameter],
            )
        return modules["Mesh"](
            **common,
            file_path=str(item.mesh_path),
            scale=None if item.scale is None else item.scale.tolist(),
        )

    def _to_curobo_scene(self, scene: PlanningScene) -> Any:
        modules = self._import_modules()
        grouped: dict[str, list[Any]] = {"cuboid": [], "sphere": [], "mesh": []}
        for item in scene.objects:
            storage_kind = "cuboid" if item.kind == "sphere" else item.kind
            grouped[storage_kind].append(self._to_curobo_obstacle(item))
        return modules["Scene"](**grouped)

    def _to_formal_curobo_scene_dict(self, scene: PlanningScene) -> Any:
        grouped: dict[str, dict[str, Any]] = {"cuboid": {}, "mesh": {}}
        for item in scene.objects:
            obstacle = self._to_curobo_obstacle(item)
            storage_kind = "cuboid" if item.kind == "sphere" else item.kind
            payload: dict[str, Any] = {"pose": list(obstacle.pose)}
            if storage_kind == "cuboid":
                payload["dims"] = list(obstacle.dims)
            else:
                payload["file_path"] = str(obstacle.file_path)
                if obstacle.scale is not None:
                    payload["scale"] = list(obstacle.scale)
            grouped[storage_kind][str(obstacle.name)] = payload
        converted = {kind: values for kind, values in grouped.items() if values}
        if not self.config.multi_env:
            return converted
        return [copy.deepcopy(converted) for _ in range(self.config.max_batch_size)]

    def _update_formal_world(self, planner: Any, scene: PlanningScene) -> None:
        converted = self._to_curobo_scene(scene)
        if not self.config.multi_env:
            planner.update_world(converted)
            return
        collision = planner.scene_collision_checker
        for env_idx in range(self.config.max_batch_size):
            collision.load_collision_model(copy.deepcopy(converted), env_idx=env_idx)

    def update_scene(self, scene: PlanningScene) -> None:
        self._scene = scene
        self._pose_ik_cache.clear()
        self._pose_ik_metrics.clear()
        if self._planner is not None:
            self._update_formal_world(self._planner, scene)
        if self._coarse_ik_solver is not None:
            self._coarse_ik_solver.update_world(self._to_curobo_scene(scene))

    @staticmethod
    def _set_collision_obstacle_enabled(collision: Any, name: str, enabled: bool) -> None:
        if collision is None:
            raise RuntimeError("collision scene is not configured")
        for store_name in ("cuboids", "meshes", "voxels"):
            store = getattr(collision.data, store_name, None)
            if store is None:
                continue
            names = getattr(store, "names", None)
            matching_envs = [
                env_idx
                for env_idx, env_names in enumerate(names or ())
                if str(name) in env_names
            ]
            if matching_envs:
                for env_idx in matching_envs:
                    store.set_enabled(str(name), bool(enabled), env_idx)
                return
        raise KeyError(f"obstacle {name!r} is not present in the cuRobo scene")

    @staticmethod
    def _collision_obstacle_enabled(collision: Any, name: str) -> bool:
        if collision is None:
            raise RuntimeError("collision scene is not configured")
        for store_name in ("cuboids", "meshes", "voxels"):
            store = getattr(collision.data, store_name, None)
            names = None if store is None else getattr(store, "names", None)
            matching = [
                (env_idx, env_names.index(str(name)))
                for env_idx, env_names in enumerate(names or ())
                if str(name) in env_names
            ]
            if matching:
                return all(
                    bool(store.enable[env_idx, obstacle_idx].item())
                    for env_idx, obstacle_idx in matching
                )
        raise KeyError(f"obstacle {name!r} is not present in the cuRobo scene")

    def _set_obstacle_enabled(self, name: str, enabled: bool) -> None:
        """Work around cuRobo2's mixed-type enable lookup raising ValueError."""
        planner = self._ensure_planner()
        collision = planner.scene_collision_checker
        self._set_collision_obstacle_enabled(collision, name, enabled)

    def _obstacle_enabled(self, name: str) -> bool:
        planner = self._ensure_planner()
        collision = planner.scene_collision_checker
        return self._collision_obstacle_enabled(collision, name)

    def _update_obstacle_pose(self, name: str, pose: Any) -> None:
        planner = self._ensure_planner()
        collision = planner.scene_collision_checker
        for store_name in ("cuboids", "meshes", "voxels"):
            store = getattr(collision.data, store_name, None)
            names = None if store is None else getattr(store, "names", None)
            matching_envs = [
                env_idx
                for env_idx, env_names in enumerate(names or ())
                if str(name) in env_names
            ]
            if matching_envs:
                for env_idx in matching_envs:
                    store.update_pose(str(name), w_obj_pose=pose, env_idx=env_idx)
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

    def _install_isolated_graph_seeding(
        self,
        planner: Any,
        terminal_records: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Keep one infeasible PRM terminal pair from rejecting the whole batch.

        cuRobo2 2.x checks ``mask.all()`` in ``PRMGraphPlanner._find_path_impl``.
        BatchMotionPlanner flattens every candidate and seed into one PRM call,
        so one colliding pair otherwise prevents graph seeds for every valid
        candidate. Filter those pairs before the official PRM call and scatter
        its paths back into the original batch layout.
        """
        graph_planner = planner.graph_planner
        if graph_planner is None:
            return lambda: None
        attribute = "_get_graph_seed_trajectories"
        had_instance_override = attribute in planner.__dict__
        previous_instance_value = planner.__dict__.get(attribute)

        def isolated_graph_seeds(current_state: Any, seed_config: Any) -> Any:
            from curobo._src.graph_planner.graph_planner_prm import TrajInterpolationType

            torch = self._import_modules()["torch"]
            batch_size = int(current_state.position.shape[0])
            dof = int(seed_config.shape[-1])
            goal_count = int(seed_config.numel() // dof)
            if goal_count % batch_size:
                raise RuntimeError(
                    f"graph seed count {goal_count} is not divisible by batch {batch_size}"
                )
            num_seeds = goal_count // batch_size
            starts = current_state.position.reshape(batch_size, 1, dof)
            starts = starts.repeat(1, num_seeds, 1).reshape(-1, dof)
            goals = seed_config.reshape(-1, dof)
            terminals = torch.stack((starts, goals), dim=1).reshape(-1, dof)
            feasible = torch.as_tensor(
                graph_planner.check_samples_feasibility(terminals),
                dtype=torch.bool,
                device=terminals.device,
            ).reshape(-1)
            if terminal_records is not None and not bool(feasible.all().item()):
                terminal_records.append(
                    {
                        "positions": terminals.detach().clone(),
                        "feasible": feasible.detach().clone(),
                        "num_seeds": num_seeds,
                    }
                )
            pair_feasible = feasible.reshape(-1, 2).all(dim=-1)
            valid_indices = pair_feasible.nonzero().reshape(-1)
            if int(valid_indices.numel()) == 0:
                return None
            graph_result = graph_planner.find_path(
                starts[valid_indices].clone(),
                goals[valid_indices].clone(),
                interpolate_waypoints=True,
                interpolation_steps=planner.trajopt_solver.action_horizon,
                interpolation_type=TrajInterpolationType.LINEAR,
                validate_interpolated_trajectory=False,
            )
            if not bool(graph_result.success.any().item()):
                return None
            paths = graph_result.interpolated_waypoints
            if paths is None:
                return None
            full_paths = torch.zeros(
                (starts.shape[0], paths.shape[1], dof),
                dtype=paths.dtype,
                device=paths.device,
            )
            full_paths[valid_indices] = paths
            return full_paths.reshape(batch_size, num_seeds, paths.shape[1], dof)

        setattr(planner, attribute, isolated_graph_seeds)

        def restore() -> None:
            if had_instance_override:
                setattr(planner, attribute, previous_instance_value)
            else:
                delattr(planner, attribute)

        return restore

    @staticmethod
    def _install_grasp_stage_attempts(planner: Any, *, max_attempts: int) -> Any:
        """Apply task retries to each official plan_grasp pose stage.

        cuRobo2's public ``plan_grasp`` API currently does not expose the
        ``max_attempts`` argument accepted by its three internal ``plan_pose``
        calls.  Repeating each stage is important for batched grasp planning:
        goalset, approach, and grasp must all converge in the same batch row.
        This wrapper changes no routing or constraints; PRM and TrajOpt remain
        enabled exactly as in the official pipeline.
        """
        attribute = "plan_pose"
        had_instance_override = attribute in planner.__dict__
        previous_instance_value = planner.__dict__.get(attribute)
        original_plan_pose = planner.plan_pose
        def plan_pose_with_attempts(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("max_attempts", max_attempts)
            return original_plan_pose(*args, **kwargs)

        setattr(planner, attribute, plan_pose_with_attempts)

        def restore() -> None:
            if had_instance_override:
                setattr(planner, attribute, previous_instance_value)
            else:
                delattr(planner, attribute)

        return restore

    @staticmethod
    def _install_grasp_linear_scale(planner: Any, *, scale: float) -> Any:
        """Scale only plan_grasp's non-terminal linear pose criterion.

        The public cuRobo2 ``plan_grasp`` API exposes approach axis and
        distance but currently hard-codes ``non_terminal_scale=1.0``.  Keep
        using the official pipeline and intercept its criterion update on this
        planner instance; terminal pose weights and standard criteria remain
        unchanged.
        """
        value = float(scale)
        if not 0.0 < value <= 1.0:
            raise ValueError("grasp linear non-terminal scale must be in (0, 1]")
        attribute = "update_tool_pose_criteria"
        had_instance_override = attribute in planner.__dict__
        previous_instance_value = planner.__dict__.get(attribute)
        original_update = planner.update_tool_pose_criteria

        def update_with_scale(criteria_by_frame: dict[str, Any]) -> Any:
            adjusted: dict[str, Any] = {}
            for frame, criterion in criteria_by_frame.items():
                weights = getattr(
                    criterion, "non_terminal_pose_axes_weight_factor", None
                )
                if weights is not None and bool((weights != 0).any().item()):
                    criterion = criterion.clone()
                    criterion.non_terminal_pose_axes_weight_factor.mul_(value)
                adjusted[frame] = criterion
            return original_update(adjusted)

        setattr(planner, attribute, update_with_scale)

        def restore() -> None:
            if had_instance_override:
                setattr(planner, attribute, previous_instance_value)
            else:
                delattr(planner, attribute)

        return restore

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
        goal_states_for_diagnostics = None
        restore_graph_seeding = self._maybe_install_isolated_graph_seeding(planner)

        def plan_with_unbiased_ik() -> Any:
            nonlocal unbiased_ik_success, goal_states_for_diagnostics
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
                goal_states_for_diagnostics = goal_states
                return planner.plan_cspace(
                    goal_states,
                    current,
                    max_attempts=request.max_attempts,
                    enable_graph_attempt=int(self.config.enable_graph_attempt),
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
                goal_states_for_diagnostics = goal_states
                return planner.plan_cspace(
                    goal_states,
                    current,
                    max_attempts=request.max_attempts,
                    enable_graph_attempt=int(self.config.enable_graph_attempt),
                )
            return None

        planner.reset_seed()
        try:
            if request.prefer_unbiased_ik:
                raw = plan_with_unbiased_ik()
            if raw is None and not request.prefer_unbiased_ik:
                raw = planner.plan_pose(
                    goals,
                    current,
                    max_attempts=request.max_attempts,
                    enable_graph_attempt=int(self.config.enable_graph_attempt),
                )
            if raw is None and unbiased_ik_success is None:
                raw = plan_with_unbiased_ik()
        finally:
            restore_graph_seeding()
            planner.reset_seed()
        elapsed = time.perf_counter() - started
        if raw is None:
            contacts_by_candidate: dict[str, list[dict[str, Any]]] = {
                item.candidate_id: [] for item in request.candidates
            }
            diagnostic_error = None
            if goal_states_for_diagnostics is not None:
                try:
                    contacts = self._collision_diagnostics_for_states(
                        planner,
                        {"goal": goal_states_for_diagnostics},
                    )
                    for index, item in enumerate(request.candidates):
                        owned_indices = (
                            [index]
                            if index < count - 1
                            else list(range(index, solver_batch_size))
                        )
                        contacts_by_candidate[item.candidate_id] = [
                            {
                                key: value
                                for key, value in contact.items()
                                if key != "candidate_index"
                            }
                            for contact in contacts
                            if int(contact["candidate_index"]) in owned_indices
                        ]
                except Exception as exc:
                    diagnostic_error = f"{type(exc).__name__}: {exc}"
            ik_reachable = (
                unbiased_ik_success is not None
                and bool(unbiased_ik_success.any().item())
            )
            return BatchPlanningResult(
                plans=tuple(
                    CandidatePlan(
                        item.candidate_id,
                        False,
                        status=(
                            "trajectory_failed"
                            if ik_reachable
                            else "ik_failed"
                        ),
                        diagnostics={
                            "collision_diagnostics": contacts_by_candidate[
                                item.candidate_id
                            ],
                            "collision_summary": (
                                f"检测到 {len(contacts_by_candidate[item.candidate_id])} 个目标状态碰撞对"
                                if contacts_by_candidate[item.candidate_id]
                                else "目标 IK 可达但轨迹规划失败"
                                if ik_reachable
                                else "目标 IK 未找到"
                            ),
                            **(
                                {}
                                if diagnostic_error is None
                                else {"diagnostic_error": diagnostic_error}
                            ),
                        },
                    )
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
        last_tsteps = self._batch_values(
            getattr(raw, "interpolated_last_tstep", None), solver_batch_size
        )

        plans: list[CandidatePlan] = []
        for index, candidate in enumerate(request.candidates):
            result_index = winner_indices[index]
            trajectory = None
            if bool(success[index]) and trajectory_state is not None:
                trajectory = self._trajectory_for_candidate(
                    trajectory_state,
                    result_index,
                    tuple(request.current.names),
                    end_tstep=last_tsteps[result_index],
                )
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
                    diagnostics={},
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
        ignore_object_names: tuple[str, ...] = (),
    ) -> None:
        """Batch IK before attachment; later transport planning reuses the joint goals."""
        if scene is not self._scene:
            self.update_scene(scene)
        planner = self._ensure_planner()
        self._prepare_pose_candidates_with_solver(
            candidates,
            solver=planner.ik_solver,
            batch_size=int(planner.batch_size),
            scene=scene,
            tool_frame=tool_frame,
            ignore_object_name=ignore_object_name,
            ignore_object_names=ignore_object_names,
            screen_kind="stable",
        )

    def prepare_pose_candidates_coarse(
        self,
        candidates: tuple[Any, ...],
        scene: PlanningScene,
        *,
        tool_frame: str = "gripper_tcp",
        ignore_object_name: str | None = None,
        ignore_object_names: tuple[str, ...] = (),
    ) -> None:
        """High-throughput first pass; stable IK can supplement its failures."""
        if scene is not self._scene:
            self.update_scene(scene)
        self._prepare_pose_candidates_with_solver(
            candidates,
            solver=self._ensure_coarse_ik_solver(),
            batch_size=int(self.config.coarse_ik_batch_size),
            scene=scene,
            tool_frame=tool_frame,
            ignore_object_name=ignore_object_name,
            ignore_object_names=ignore_object_names,
            screen_kind="coarse",
        )

    def _prepare_pose_candidates_with_solver(
        self,
        candidates: tuple[Any, ...],
        *,
        solver: Any,
        batch_size: int,
        scene: PlanningScene,
        tool_frame: str,
        ignore_object_name: str | None,
        ignore_object_names: tuple[str, ...],
        screen_kind: str,
    ) -> None:
        if not candidates:
            return
        modules = self._import_modules()
        torch = modules["torch"]
        device_cfg = modules["DeviceCfg"](device=self.config.device)
        ignored_names = tuple(
            dict.fromkeys(
                [
                    name
                    for name in (ignore_object_name, *ignore_object_names)
                    if name
                ]
            )
        )
        collision = solver.scene_collision_checker
        ignored_states = {
            name: self._collision_obstacle_enabled(collision, name)
            for name in ignored_names
        }
        for name in ignored_names:
            self._set_collision_obstacle_enabled(collision, name, False)
        try:
            for start in range(0, len(candidates), batch_size):
                chunk = tuple(candidates[start : start + batch_size])
                count = len(chunk)
                positions = np.stack([item.pose.position for item in chunk])
                quaternions = np.stack([item.pose.quaternion_wxyz for item in chunk])
                if count < batch_size:
                    positions = np.concatenate(
                        [positions, np.repeat(positions[-1:], batch_size - count, axis=0)]
                    )
                    quaternions = np.concatenate(
                        [quaternions, np.repeat(quaternions[-1:], batch_size - count, axis=0)]
                    )
                goals = modules["GoalToolPose"](
                    tool_frames=[tool_frame],
                    position=torch.as_tensor(
                        positions,
                        dtype=device_cfg.dtype,
                        device=device_cfg.device,
                    )[:, None, None, None, :],
                    quaternion=torch.as_tensor(
                        quaternions,
                        dtype=device_cfg.dtype,
                        device=device_cfg.device,
                    )[:, None, None, None, :],
                )
                solver.reset_seed()
                result = solver.solve_pose(
                    goals,
                    current_state=None,
                    return_seeds=int(self.config.num_trajopt_seeds),
                )
                success = result.success.reshape(batch_size, -1)
                solutions = result.solution.reshape(
                    batch_size, -1, result.solution.shape[-1]
                )
                position_error = result.position_error.reshape(batch_size, -1)
                rotation_error = result.rotation_error.reshape(batch_size, -1)
                feasible = result.feasible.reshape(batch_size, -1)
                normalized_gap = (
                    position_error / float(self.config.position_tolerance)
                    + rotation_error / float(self.config.orientation_tolerance)
                )
                ranked_gap = normalized_gap + (~feasible).to(normalized_gap.dtype) * 1.0e6
                success_winner = success.to(dtype=torch.int64).argmax(dim=-1)
                nearest_winner = ranked_gap.argmin(dim=-1)
                has_success = success.any(dim=-1)
                winner = torch.where(has_success, success_winner, nearest_winner)
                batch_index = torch.arange(batch_size, device=device_cfg.device)
                selected = solutions[batch_index, winner].detach().cpu().numpy()
                success_cpu = has_success.detach().cpu().numpy()
                feasible_cpu = feasible[batch_index, winner].detach().cpu().numpy()
                position_cpu = position_error[batch_index, winner].detach().cpu().numpy()
                rotation_cpu = rotation_error[batch_index, winner].detach().cpu().numpy()
                gap_cpu = normalized_gap[batch_index, winner].detach().cpu().numpy()
                for index, candidate in enumerate(chunk):
                    key = self._pose_cache_key(candidate)
                    self._pose_ik_metrics[key] = {
                        "candidate_id": str(candidate.candidate_id),
                        "screen_kind": screen_kind,
                        "success": bool(success_cpu[index]),
                        "constraint_feasible": bool(feasible_cpu[index]),
                        "position_error_m": float(position_cpu[index]),
                        "orientation_error_rad": float(rotation_cpu[index]),
                        "normalized_pose_gap": float(gap_cpu[index]),
                    }
                    if bool(success_cpu[index]):
                        self._pose_ik_cache[key] = selected[index]
        finally:
            for name, was_enabled in ignored_states.items():
                self._set_collision_obstacle_enabled(collision, name, bool(was_enabled))

    def pose_candidate_metrics(self, candidates: tuple[Any, ...]) -> dict[str, dict[str, Any]]:
        """Return the latest continuous IK residuals for application-level ranking."""
        return {
            str(candidate.candidate_id): dict(metrics)
            for candidate in candidates
            if (metrics := self._pose_ik_metrics.get(self._pose_cache_key(candidate))) is not None
        }

    def feasible_pose_candidate_ids(
        self, candidates: tuple[Any, ...]
    ) -> frozenset[str]:
        """Return candidates accepted by the latest collision-checked IK screen."""

        return frozenset(
            str(candidate.candidate_id)
            for candidate in candidates
            if self._pose_cache_key(candidate) in self._pose_ik_cache
        )

    def _make_batch_inputs(self, request: BatchPlanningRequest) -> tuple[Any, Any]:
        planner = self._ensure_planner()
        modules = self._import_modules()
        torch = modules["torch"]
        count = len(request.candidates)
        batch_size = planner.batch_size
        if count > batch_size:
            raise ValueError(f"received {count} candidates, maximum batch size is {batch_size}")

        current_positions = np.stack([
            request.current_by_candidate.get(item.candidate_id, request.current).positions
            for item in request.candidates
        ])
        if count < batch_size:
            current_positions = np.concatenate([
                current_positions,
                np.repeat(current_positions[-1:], batch_size - count, axis=0),
            ])
        current_position = torch.as_tensor(
            current_positions,
            dtype=planner.device_cfg.dtype,
            device=planner.device_cfg.device,
        )
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
        state: Any,
        index: int,
        active_joint_names: tuple[str, ...],
        *,
        end_tstep: int | float | None = None,
    ) -> JointTrajectory | None:
        if state is None:
            return None
        position = state.position[index]
        while position.ndim > 2:
            position = position[0]
        if end_tstep is not None and int(end_tstep) > 0:
            position = position[: int(end_tstep)]
        path = np.asarray(position.detach().cpu().numpy())
        names = tuple(getattr(state, "joint_names", ()) or active_joint_names)
        if path.ndim != 2 or path.shape[-1] != len(names):
            raise RuntimeError(
                f"cuRobo trajectory shape {path.shape} does not match {len(names)} joint names"
            )
        if names != active_joint_names:
            indices = [names.index(name) for name in active_joint_names]
            path = path[:, indices]
        return JointTrajectory(active_joint_names, path)

    def _collision_diagnostics_for_states(
        self,
        planner: Any,
        states: dict[str, Any],
        *,
        ignored_world_objects: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return exact link/object and link/link contacts for batched joint states."""
        from curobo._src.cost.cost_self_collision import SelfCollisionCost
        from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
        from curobo._src.geom.collision.buffer_collision import CollisionBuffer

        modules = self._import_modules()
        torch = modules["torch"]
        ignored = {str(name) for name in (ignored_world_objects or set())}
        scene_names = [item.name for item in self._scene.objects]
        kinematics_config = planner.kinematics.config.kinematics_config
        sphere_to_link = kinematics_config.link_sphere_idx_map
        idx_to_name = {
            int(value): str(key)
            for key, value in kinematics_config.link_name_to_idx_map.items()
        }
        output: list[dict[str, Any]] = []
        enabled_snapshot = {name: self._obstacle_enabled(name) for name in scene_names}

        # Query one obstacle at a time because cuRobo2's public distance API
        # aggregates over obstacles but retains the colliding robot sphere.
        for name in scene_names:
            self._set_obstacle_enabled(name, False)
        try:
            for state_name, joint_state in states.items():
                if joint_state is None:
                    continue
                active = planner.kinematics.get_active_js(joint_state)
                batch_size = int(active.position.shape[0])
                kin_state = planner.kinematics.compute_kinematics(active)
                robot_spheres = kin_state.robot_spheres.reshape(batch_size, 1, -1, 4)

                for obstacle_name in scene_names:
                    if obstacle_name in ignored:
                        continue
                    self._set_obstacle_enabled(obstacle_name, True)
                    buffer = CollisionBuffer.from_shape(robot_spheres.shape, planner.device_cfg)
                    distance = planner.scene_collision_checker.get_sphere_distance_raw(
                        robot_spheres,
                        buffer,
                        torch.ones(1, device=robot_spheres.device),
                        torch.zeros(1, device=robot_spheres.device),
                    )
                    for batch_index in range(batch_size):
                        sphere_indices = (distance[batch_index, 0] > 0).nonzero().reshape(-1)
                        for sphere_index in sphere_indices.detach().cpu().tolist():
                            penetration = float(distance[batch_index, 0, sphere_index].item())
                            link_idx = int(sphere_to_link[int(sphere_index)].item())
                            output.append(
                                {
                                    "collision_type": "world",
                                    "state": state_name,
                                    "candidate_index": batch_index,
                                    "robot_link": idx_to_name.get(link_idx, f"link_index_{link_idx}"),
                                    "world_object": obstacle_name,
                                    "penetration_m": penetration,
                                }
                            )
                    self._set_obstacle_enabled(obstacle_name, False)

                self_cost = SelfCollisionCost(
                    SelfCollisionCostCfg(
                        weight=1.0,
                        self_collision_kin_config=planner.kinematics.get_self_collision_config(),
                        store_pair_distance=True,
                    )
                )
                self_cost.setup_batch_tensors(batch_size, 1)
                self_cost.forward(robot_spheres)
                pair_distance = self_cost._pair_distance.reshape(batch_size, 1, -1)
                collision_pairs = planner.kinematics.get_self_collision_config().collision_pairs
                for batch_index in range(batch_size):
                    pair_indices = (pair_distance[batch_index, 0] > 0).nonzero().reshape(-1)
                    for pair_index in pair_indices.detach().cpu().tolist():
                        sphere_pair = collision_pairs[int(pair_index)].to(dtype=torch.int32)
                        link_pair = sphere_to_link[sphere_pair].detach().cpu().tolist()
                        output.append(
                            {
                                "collision_type": "self",
                                "state": state_name,
                                "candidate_index": batch_index,
                                "robot_link": idx_to_name.get(int(link_pair[0]), f"link_index_{link_pair[0]}"),
                                "other_robot_link": idx_to_name.get(int(link_pair[1]), f"link_index_{link_pair[1]}"),
                                "penetration_m": float(
                                    pair_distance[batch_index, 0, int(pair_index)].item()
                                ),
                            }
                        )
        finally:
            for name in scene_names:
                self._set_obstacle_enabled(name, enabled_snapshot[name])
        # Several robot spheres can report the same semantic contact. Keep the
        # deepest penetration for each link/object (or link/link) pair so the
        # report stays useful to a human and stable across sphere refits.
        aggregated: dict[tuple[Any, ...], dict[str, Any]] = {}
        for item in output:
            if item["collision_type"] == "world":
                key = (
                    item["state"],
                    item["candidate_index"],
                    "world",
                    item["robot_link"],
                    item["world_object"],
                )
            else:
                links = sorted((item["robot_link"], item["other_robot_link"]))
                key = (
                    item["state"],
                    item["candidate_index"],
                    "self",
                    *links,
                )
            previous = aggregated.get(key)
            if previous is None or item["penetration_m"] > previous["penetration_m"]:
                aggregated[key] = item
        return list(aggregated.values())

    def _diagnose_grasp_endpoints(
        self,
        planner: Any,
        raw: Any,
        *,
        target_name: str | None,
        candidate_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        modules = self._import_modules()
        extract = modules["get_joint_state_at_horizon_index"]
        states: dict[str, Any] = {}
        approach_result = getattr(raw, "approach_result", None)
        goalset_result = getattr(raw, "goalset_result", None)
        if approach_result is not None and approach_result.js_solution is not None:
            states["approach_end"] = extract(approach_result.js_solution, -1).squeeze(1)
        if goalset_result is not None and goalset_result.js_solution is not None:
            states["grasp_goal"] = extract(goalset_result.js_solution, -1).squeeze(1)
        if not states:
            return {}
        contacts = self._collision_diagnostics_for_states(
            planner,
            states,
            ignored_world_objects=set() if target_name is None else {target_name},
        )
        by_candidate: dict[str, dict[str, Any]] = {}
        for index, candidate_id in enumerate(candidate_ids):
            owned_indices = (
                [index]
                if index < len(candidate_ids) - 1
                else list(range(index, planner.batch_size))
            )
            candidate_contacts = [
                {key: value for key, value in item.items() if key != "candidate_index"}
                for item in contacts
                if int(item["candidate_index"]) in owned_indices
            ]
            by_candidate[candidate_id] = {
                "collision_diagnostics": candidate_contacts,
                "collision_summary": (
                    "未检测到起终点碰撞"
                    if not candidate_contacts
                    else f"检测到 {len(candidate_contacts)} 个起终点碰撞对"
                ),
                "ignored_world_objects": [] if target_name is None else [target_name],
            }
        return by_candidate

    def _diagnose_graph_terminals(
        self,
        planner: Any,
        terminal_record: dict[str, Any] | None,
        *,
        target_name: str | None,
        candidate_ids: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Resolve the PRM warning's flattened start/goal seed states to contacts."""
        if not terminal_record:
            return {}
        positions = terminal_record["positions"]
        feasible = terminal_record["feasible"].reshape(-1)
        num_seeds = int(terminal_record["num_seeds"])
        modules = self._import_modules()
        states = modules["JointState"].from_position(
            positions,
            joint_names=list(planner.joint_names),
        )
        contacts = self._collision_diagnostics_for_states(
            planner,
            {"graph_terminal": states},
            ignored_world_objects=set() if target_name is None else {target_name},
        )
        aggregated: dict[tuple[Any, ...], dict[str, Any]] = {}
        for contact in contacts:
            terminal_index = int(contact.pop("candidate_index"))
            if terminal_index >= int(feasible.numel()) or bool(feasible[terminal_index].item()):
                continue
            problem_index, endpoint_index = divmod(terminal_index, 2)
            candidate_index, seed_index = divmod(problem_index, num_seeds)
            if not candidate_ids:
                continue
            candidate_index = min(candidate_index, len(candidate_ids) - 1)
            candidate_id = candidate_ids[candidate_index]
            contact.update(
                {
                    "state": "graph_start" if endpoint_index == 0 else "graph_goal",
                    "seed_index": seed_index,
                    "source": "curobo2_prm_terminal_check",
                }
            )
            if contact["collision_type"] == "world":
                pair = (contact["robot_link"], contact["world_object"])
            else:
                pair = tuple(sorted((contact["robot_link"], contact["other_robot_link"])))
            key = (candidate_id, contact["collision_type"], contact["state"], *pair)
            previous = aggregated.get(key)
            if previous is None or contact["penetration_m"] > previous["penetration_m"]:
                aggregated[key] = {"candidate_id": candidate_id, **contact}
        result: dict[str, list[dict[str, Any]]] = {}
        for contact in aggregated.values():
            candidate_id = contact.pop("candidate_id")
            result.setdefault(candidate_id, []).append(contact)
        return result

    @staticmethod
    def _grasp_winner_indices(success: np.ndarray, count: int) -> list[int]:
        """Map padded solver rows back to their owning semantic candidates."""
        winners: list[int] = []
        for candidate_index in range(count):
            pool = [candidate_index]
            if candidate_index == count - 1:
                pool = list(range(candidate_index, len(success)))
            winners.append(next((index for index in pool if bool(success[index])), candidate_index))
        return winners

    @staticmethod
    def _official_grasp_status(
        raw_status: Any,
        *,
        success: bool,
        goalset_success: bool,
        approach_success: bool,
        grasp_success: bool,
    ) -> str:
        detail = str(raw_status)
        if success:
            return "official_plan_grasp:success"
        if not goalset_success:
            return f"official_plan_grasp:goalset_failed ({detail})"
        if not approach_success:
            return f"official_plan_grasp:approach_failed ({detail})"
        if not grasp_success:
            return f"official_plan_grasp:grasp_failed ({detail})"
        return f"official_plan_grasp:failed ({detail})"

    def _fresh_linear_grasp_retry(
        self,
        request: GraspPlanningRequest,
        candidate: PoseCandidate,
        approach: JointTrajectory,
    ) -> GraspCandidatePlan | None:
        """Retry only the final linear leg on a clean cuRobo2 solver.

        cuRobo2 2.x can leave its batched solver in a state where the third
        internal ``plan_grasp`` stage stalls, while the identical constrained
        solve succeeds from a fresh solver.  The official approach trajectory
        is retained; this retry still uses ``ToolPoseCriteria.linear_motion``
        in the requested world/tool frame and never falls back to an arbitrary
        unconstrained path.
        """
        planning = request.planning
        previous = self._planner
        if previous is not None:
            previous.destroy()
        self._planner = None
        planner = self._ensure_planner()
        modules = self._import_modules()
        current = JointConfiguration(approach.joint_names, approach.positions[-1])
        retry_planning = replace(
            planning,
            current=current,
            candidates=(candidate,),
        )
        current_state, grasp_goals = self._make_batch_inputs(retry_planning)
        contact_links = list(
            request.disable_collision_links
            if request.disable_collision_links is not None
            else (
                planner.kinematics.config.kinematics_config.grasp_contact_link_names
                or []
            )
        )
        sphere_snapshot = (
            planner.kinematics.config.kinematics_config.link_spheres.clone()
        )
        target_name = request.target_object_name
        target_was_enabled = None
        restore_graph_seeding = self._maybe_install_isolated_graph_seeding(planner)
        started = time.perf_counter()
        raw = None
        try:
            if target_name and planner.scene_collision_checker is not None:
                target_was_enabled = self._obstacle_enabled(target_name)
                self._set_obstacle_enabled(target_name, False)
            criterion = modules["ToolPoseCriteria"].linear_motion(
                axis=request.approach_axis,
                non_terminal_scale=self.config.grasp_linear_non_terminal_scale,
                project_distance_to_goal=request.approach_in_tool_frame,
            )
            planner.update_tool_pose_criteria({planning.tool_frame: criterion})
            planner.disable_link_collision(contact_links)
            planner.reset_seed()
            raw = planner.plan_pose(
                grasp_goals,
                current_state,
                max_attempts=planning.max_attempts,
                enable_graph_attempt=int(self.config.enable_graph_attempt),
            )
        finally:
            planner.kinematics.config.kinematics_config.link_spheres.copy_(
                sphere_snapshot
            )
            planner.update_tool_pose_criteria(
                {planning.tool_frame: modules["ToolPoseCriteria"]()}
            )
            if target_name and target_was_enabled is not None:
                self._set_obstacle_enabled(target_name, bool(target_was_enabled))
            restore_graph_seeding()
            planner.reset_seed()
        if raw is None:
            return None
        success = raw.success.reshape(planner.batch_size, -1).any(dim=-1)
        feasible = success.nonzero().reshape(-1)
        if int(feasible.numel()) == 0:
            return None
        winner = int(feasible[0].item())
        grasp_state = raw.interpolated_trajectory
        if grasp_state is None:
            grasp_state = raw.js_solution
        grasp = self._trajectory_for_candidate(
            grasp_state,
            winner,
            tuple(planning.current.names),
        )
        lift = None
        if request.plan_lift:
            if request.lift_in_tool_frame:
                return None
            axis_index = {"x": 0, "y": 1, "z": 2}[request.lift_axis]
            lift_position = grasp_goals.position.clone()
            lift_position[..., axis_index] += float(request.lift_offset)
            lift_goals = modules["GoalToolPose"](
                tool_frames=list(grasp_goals.tool_frames),
                position=lift_position,
                quaternion=grasp_goals.quaternion.clone(),
            )
            lift_start_position = modules["torch"].as_tensor(
                grasp.positions[-1],
                dtype=planner.device_cfg.dtype,
                device=planner.device_cfg.device,
            ).unsqueeze(0).repeat(planner.batch_size, 1)
            lift_start = modules["JointState"].from_position(
                lift_start_position,
                joint_names=list(planning.current.names),
            )
            raw_lift = None
            target_was_enabled = None
            try:
                if target_name and planner.scene_collision_checker is not None:
                    target_was_enabled = self._obstacle_enabled(target_name)
                    self._set_obstacle_enabled(target_name, False)
                lift_criterion = modules["ToolPoseCriteria"].linear_motion(
                    axis=request.lift_axis,
                    non_terminal_scale=self.config.grasp_linear_non_terminal_scale,
                    project_distance_to_goal=False,
                )
                planner.update_tool_pose_criteria({planning.tool_frame: lift_criterion})
                planner.disable_link_collision(contact_links)
                planner.reset_seed()
                raw_lift = planner.plan_pose(
                    lift_goals,
                    lift_start,
                    max_attempts=planning.max_attempts,
                    enable_graph_attempt=int(self.config.enable_graph_attempt),
                )
            finally:
                planner.update_tool_pose_criteria(
                    {planning.tool_frame: modules["ToolPoseCriteria"]()}
                )
                if target_name and target_was_enabled is not None:
                    self._set_obstacle_enabled(target_name, bool(target_was_enabled))
                planner.reset_seed()
            if raw_lift is None or not bool(raw_lift.success.any().item()):
                return None
            lift_raw_state = raw_lift.interpolated_trajectory
            if lift_raw_state is None:
                lift_raw_state = raw_lift.js_solution
            lift_success = raw_lift.success.reshape(planner.batch_size, -1).any(dim=-1)
            lift_winner = int(lift_success.nonzero().reshape(-1)[0].item())
            lift = self._trajectory_for_candidate(
                lift_raw_state, lift_winner, tuple(planning.current.names)
            )
        return GraspCandidatePlan(
            candidate_id=candidate.candidate_id,
            success=True,
            approach=approach,
            grasp=grasp,
            lift=lift,
            status="official_plan_grasp:fresh_world_linear_retry_success",
            diagnostics={
                "fresh_linear_retry": True,
                "approach_frame": (
                    "tool" if request.approach_in_tool_frame else "world"
                ),
                "linear_non_terminal_scale": self.config.grasp_linear_non_terminal_scale,
                "retry_time_s": time.perf_counter() - started,
            },
        )

    def plan_linear_candidates(
        self,
        request: BatchPlanningRequest,
        *,
        axis: str = "z",
        project_distance_to_goal: bool = False,
        ignore_object_name: str | None = None,
        disable_collision_links: tuple[str, ...] | None = None,
    ) -> BatchPlanningResult:
        """Plan one independent collision-checked Cartesian line segment.

        This is the stable pre-cuRobo2 pick-place primitive: broad endpoint IK
        screening happens separately, while MotionGen receives only shortlisted
        rows and one explicit linear pose criterion.
        """

        if len(request.candidates) > self.config.max_batch_size:
            plans: list[CandidatePlan] = []
            total_time = 0.0
            for start in range(0, len(request.candidates), self.config.max_batch_size):
                chunk = tuple(request.candidates[start : start + self.config.max_batch_size])
                result = self.plan_linear_candidates(
                    replace(
                        request,
                        candidates=chunk,
                        current_by_candidate={
                            item.candidate_id: request.current_by_candidate[item.candidate_id]
                            for item in chunk
                            if item.candidate_id in request.current_by_candidate
                        },
                    ),
                    axis=axis,
                    project_distance_to_goal=project_distance_to_goal,
                    ignore_object_name=ignore_object_name,
                    disable_collision_links=disable_collision_links,
                )
                plans.extend(result.plans)
                total_time += float(result.total_time)
            return BatchPlanningResult(tuple(plans), backend=self.name, total_time=total_time)
        if axis not in {"x", "y", "z"}:
            raise ValueError(f"unsupported linear axis {axis!r}")
        if request.scene is not self._scene:
            self.update_scene(request.scene)
        planner = self._ensure_planner()
        modules = self._import_modules()
        current, goals = self._make_batch_inputs(request)
        contact_links = list(
            disable_collision_links
            if disable_collision_links is not None
            else (
                planner.kinematics.config.kinematics_config.grasp_contact_link_names
                or []
            )
        )
        sphere_snapshot = planner.kinematics.config.kinematics_config.link_spheres.clone()
        ignored_was_enabled = None
        restore_graph_seeding = self._maybe_install_isolated_graph_seeding(planner)
        raw = None
        started = time.perf_counter()
        try:
            if ignore_object_name and planner.scene_collision_checker is not None:
                ignored_was_enabled = self._obstacle_enabled(ignore_object_name)
                self._set_obstacle_enabled(ignore_object_name, False)
            criterion = modules["ToolPoseCriteria"].linear_motion(
                axis=axis,
                non_terminal_scale=self.config.grasp_linear_non_terminal_scale,
                project_distance_to_goal=project_distance_to_goal,
            )
            planner.update_tool_pose_criteria({request.tool_frame: criterion})
            if contact_links:
                planner.disable_link_collision(contact_links)
            planner.reset_seed()
            raw = planner.plan_pose(
                goals,
                current,
                max_attempts=request.max_attempts,
                enable_graph_attempt=int(self.config.enable_graph_attempt),
            )
        finally:
            planner.kinematics.config.kinematics_config.link_spheres.copy_(
                sphere_snapshot
            )
            planner.update_tool_pose_criteria(
                {request.tool_frame: modules["ToolPoseCriteria"]()}
            )
            if ignore_object_name and ignored_was_enabled is not None:
                self._set_obstacle_enabled(
                    ignore_object_name, bool(ignored_was_enabled)
                )
            restore_graph_seeding()
            planner.reset_seed()
        elapsed = time.perf_counter() - started
        count = len(request.candidates)
        if raw is None:
            return BatchPlanningResult(
                tuple(
                    CandidatePlan(item.candidate_id, False, status="linear_planner_failed")
                    for item in request.candidates
                ),
                backend=self.name,
                total_time=elapsed,
            )
        raw_success = np.asarray(
            self._batch_values(raw.success, planner.batch_size, False), dtype=bool
        )
        winner_indices = self._grasp_winner_indices(raw_success, count)
        state = raw.interpolated_trajectory
        if state is None:
            state = raw.js_solution
        last_tsteps = self._batch_values(
            getattr(raw, "interpolated_last_tstep", None), planner.batch_size
        )
        plans = tuple(
            CandidatePlan(
                candidate.candidate_id,
                bool(raw_success[winner_indices[index]]),
                trajectory=(
                    self._trajectory_for_candidate(
                        state,
                        winner_indices[index],
                        tuple(request.current.names),
                        end_tstep=last_tsteps[winner_indices[index]],
                    )
                    if bool(raw_success[winner_indices[index]])
                    else None
                ),
                solve_time=elapsed,
                status=(
                    "linear_success"
                    if bool(raw_success[winner_indices[index]])
                    else f"linear_failed ({raw.status})"
                ),
                diagnostics={
                    "linear_axis": axis,
                    "project_distance_to_goal": project_distance_to_goal,
                    "ignored_object": ignore_object_name,
                },
            )
            for index, candidate in enumerate(request.candidates)
        )
        return BatchPlanningResult(plans, backend=self.name, total_time=elapsed)

    def plan_grasps(self, request: GraspPlanningRequest) -> GraspPlanningResult:
        """Use cuRobo2's official goalset -> approach -> grasp pipeline.

        Lift remains an application boundary: the selected object must first be
        attached to the robot collision model, then planned as carried geometry.
        """

        planning = request.planning
        if len(planning.candidates) > self.config.max_batch_size:
            plans: list[GraspCandidatePlan] = []
            total_time = 0.0
            for start in range(0, len(planning.candidates), self.config.max_batch_size):
                chunk = tuple(
                    planning.candidates[start : start + self.config.max_batch_size]
                )
                result = self.plan_grasps(
                    replace(
                        request,
                        planning=replace(
                            planning,
                            candidates=chunk,
                            current_by_candidate={
                                item.candidate_id: planning.current_by_candidate[item.candidate_id]
                                for item in chunk
                                if item.candidate_id in planning.current_by_candidate
                            },
                        ),
                    )
                )
                plans.extend(result.plans)
                total_time += float(result.total_time)
            return GraspPlanningResult(
                tuple(plans), backend=self.name, total_time=total_time
            )
        if planning.scene is not self._scene:
            self.update_scene(planning.scene)
        planner = self._ensure_planner()
        modules = self._import_modules()
        current, grasp_goals = self._make_batch_inputs(planning)
        contact_links = list(
            request.disable_collision_links
            if request.disable_collision_links is not None
            else (
                planner.kinematics.config.kinematics_config.grasp_contact_link_names
                or []
            )
        )
        sphere_snapshot = (
            planner.kinematics.config.kinematics_config.link_spheres.clone()
        )
        target_name = request.target_object_name
        target_disabled = False
        target_was_enabled = None
        graph_terminal_records: list[dict[str, Any]] = []
        restore_graph_seeding = self._maybe_install_isolated_graph_seeding(
            planner, graph_terminal_records
        )
        restore_grasp_attempts = self._install_grasp_stage_attempts(
            planner, max_attempts=planning.max_attempts
        )
        restore_grasp_linear_scale = self._install_grasp_linear_scale(
            planner,
            scale=self.config.grasp_linear_non_terminal_scale,
        )
        planner.reset_seed()
        started = time.perf_counter()
        endpoint_diagnostics: dict[str, dict[str, Any]] = {}
        try:
            if target_name and planner.scene_collision_checker is not None:
                target_was_enabled = self._obstacle_enabled(target_name)
                self._set_obstacle_enabled(target_name, False)
                target_disabled = True
            raw = planner.plan_grasp(
                grasp_goals,
                current,
                grasp_approach_axis=request.approach_axis,
                grasp_approach_offset=request.approach_offset,
                grasp_approach_in_tool_frame=request.approach_in_tool_frame,
                plan_approach_to_grasp=True,
                grasp_lift_axis=request.lift_axis,
                grasp_lift_offset=request.lift_offset,
                grasp_lift_in_tool_frame=request.lift_in_tool_frame,
                plan_grasp_to_lift=request.plan_lift,
                disable_collision_links=contact_links,
            )
            raw_success = (
                raw.success.reshape(planner.batch_size, -1)
                .any(dim=-1)
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )
            diagnostic_winners = self._grasp_winner_indices(
                raw_success, len(planning.candidates)
            )
            if not all(bool(raw_success[index]) for index in diagnostic_winners):
                # plan_grasp restores grasp-contact links before returning.
                # Reapply the strict grasp-stage collision mask while checking
                # its approach end and goalset solution, then let the sphere
                # snapshot below restore the planner completely.
                planner.disable_link_collision(contact_links)
                try:
                    endpoint_diagnostics = self._diagnose_grasp_endpoints(
                        planner,
                        raw,
                        target_name=target_name,
                        candidate_ids=[item.candidate_id for item in planning.candidates],
                    )
                    graph_contacts = self._diagnose_graph_terminals(
                        planner,
                        graph_terminal_records[-1] if graph_terminal_records else None,
                        target_name=target_name,
                        candidate_ids=[item.candidate_id for item in planning.candidates],
                    )
                    graph_collision_sources = sorted(graph_contacts)
                    for candidate_id, contacts in graph_contacts.items():
                        value = endpoint_diagnostics.setdefault(
                            candidate_id,
                            {
                                "collision_diagnostics": [],
                                "ignored_world_objects": [] if target_name is None else [target_name],
                            },
                        )
                        value["collision_diagnostics"].extend(contacts)
                        value["collision_summary"] = (
                            f"检测到 {len(value['collision_diagnostics'])} 个碰撞对"
                        )
                    for value in endpoint_diagnostics.values():
                        value["planner_status"] = str(raw.status)
                        value["batch_collision_source_candidates"] = graph_collision_sources
                        if (
                            graph_collision_sources
                            and not value.get("collision_diagnostics")
                        ):
                            value["collision_summary"] = (
                                "本候选未检测到碰撞；同批候选 "
                                + ", ".join(graph_collision_sources)
                                + " 触发了 PRM 整批拒绝"
                            )
                except Exception as exc:
                    endpoint_diagnostics = {
                        item.candidate_id: {
                            "collision_summary": "碰撞复核接口执行失败",
                            "collision_diagnostics": [],
                            "diagnostic_error": f"{type(exc).__name__}: {exc}",
                            "planner_status": str(raw.status),
                        }
                        for item in planning.candidates
                    }
        finally:
            restore_grasp_linear_scale()
            restore_grasp_attempts()
            restore_graph_seeding()
            # cuRobo2 2.x can leave collision spheres / linear pose criteria
            # modified when its grasp stage returns early. Restore the complete
            # planner state so retries and later atoms remain deterministic.
            planner.kinematics.config.kinematics_config.link_spheres.copy_(
                sphere_snapshot
            )
            planner.update_tool_pose_criteria(
                {planning.tool_frame: modules["ToolPoseCriteria"]()}
            )
            if target_disabled:
                self._set_obstacle_enabled(target_name, bool(target_was_enabled))
            planner.reset_seed()

        elapsed = time.perf_counter() - started
        count = len(planning.candidates)
        raw_success = np.asarray(
            self._batch_values(raw.success, planner.batch_size, False), dtype=bool
        )
        winner_indices = self._grasp_winner_indices(raw_success, count)
        success = np.asarray([raw_success[index] for index in winner_indices], dtype=bool)
        approach_success_all = self._batch_values(
            raw.approach_success, planner.batch_size, False
        )
        grasp_success_all = self._batch_values(raw.grasp_success, planner.batch_size, False)
        goalset_result = getattr(raw, "goalset_result", None)
        goalset_success_all = self._batch_values(
            None if goalset_result is None else goalset_result.success,
            planner.batch_size,
            False,
        )
        approach_state = raw.approach_interpolated_trajectory
        if approach_state is None:
            approach_state = raw.approach_trajectory
        grasp_state = raw.grasp_interpolated_trajectory
        if grasp_state is None:
            grasp_state = raw.grasp_trajectory
        lift_state = raw.lift_interpolated_trajectory
        if lift_state is None:
            lift_state = raw.lift_trajectory
        names = tuple(planning.current.names)
        plans = tuple(
            GraspCandidatePlan(
                candidate_id=candidate.candidate_id,
                success=bool(success[index]),
                approach=self._trajectory_for_candidate(
                    approach_state, winner_indices[index], names
                ),
                grasp=self._trajectory_for_candidate(
                    grasp_state, winner_indices[index], names
                ),
                lift=(
                    self._trajectory_for_candidate(lift_state, winner_indices[index], names)
                    if request.plan_lift else None
                ),
                status=self._official_grasp_status(
                    raw.status,
                    success=bool(success[index]),
                    goalset_success=bool(goalset_success_all[winner_indices[index]]),
                    approach_success=bool(approach_success_all[winner_indices[index]]),
                    grasp_success=bool(grasp_success_all[winner_indices[index]]),
                ),
                diagnostics=endpoint_diagnostics.get(candidate.candidate_id, {}),
            )
            for index, candidate in enumerate(planning.candidates)
        )
        if not bool(success.any()) and count == 1:
            valid_approach = np.asarray(approach_success_all, dtype=bool)
            feasible_approach = np.flatnonzero(valid_approach)
            if len(feasible_approach):
                approach = self._trajectory_for_candidate(
                    approach_state,
                    int(feasible_approach[0]),
                    names,
                )
                if approach is not None:
                    fresh_retry = self._fresh_linear_grasp_retry(
                        request,
                        planning.candidates[0],
                        approach,
                    )
                    if fresh_retry is not None:
                        return GraspPlanningResult(
                            (fresh_retry,),
                            backend=self.name,
                            total_time=elapsed
                            + float(fresh_retry.diagnostics.get("retry_time_s", 0.0)),
                        )
        if not bool(success.any()) and not self._attachment_active:
            # reset_seed() is insufficient after an early grasp-stage return
            # in the current cuRobo2 implementation: its next goalset solve can
            # fail even for the default FK pose. Rebuild only on the failure
            # path so retries start from a known-clean solver state.
            planner.destroy()
            self._planner = None
        return GraspPlanningResult(plans, backend=self.name, total_time=elapsed)

    def attach_object(self, object_name: str, grasp: JointConfiguration) -> None:
        self.attach_object_batch(object_name, (grasp,))

    def attach_object_batch(
        self, object_name: str, grasps: tuple[JointConfiguration, ...]
    ) -> None:
        """Install one grasp-relative object transform per formal planner row."""
        if not grasps:
            raise ValueError("grasps must not be empty")
        if len(grasps) > 1 and not self.config.multi_env:
            raise RuntimeError("per-candidate attachments require multi_env=True")
        planner = self._ensure_planner()
        if len(grasps) > int(planner.batch_size):
            raise ValueError(
                f"received {len(grasps)} attachment states, maximum batch size is "
                f"{planner.batch_size}"
            )
        names = grasps[0].names
        if any(item.names != names for item in grasps):
            raise ValueError("all attachment states must use identical joint names")
        positions = np.stack([item.positions for item in grasps])
        if self.config.multi_env and len(grasps) < int(planner.batch_size):
            positions = np.concatenate([
                positions,
                np.repeat(positions[-1:], int(planner.batch_size) - len(grasps), axis=0),
            ])
        modules = self._import_modules()
        position = modules["torch"].as_tensor(
            positions, dtype=planner.device_cfg.dtype, device=planner.device_cfg.device
        )
        state = modules["JointState"].from_position(position, joint_names=list(names))
        object_world_pose = self._scene_object_pose(object_name)
        self._attach_at_world_pose(object_name, state, object_world_pose)

    def _scene_object_pose(self, object_name: str) -> Any:
        planner = self._ensure_planner()
        modules = self._import_modules()
        collision_object = next(
            (item for item in self._scene.objects if item.name == str(object_name)),
            None,
        )
        if collision_object is not None:
            object_pose = collision_object.metadata.get("object_world_pose")
            if object_pose is not None:
                transform = np.asarray(object_pose, dtype=np.float64).reshape(4, 4)
                return modules["CuroboPose"].from_matrix(
                    modules["torch"].as_tensor(
                        transform,
                        dtype=planner.device_cfg.dtype,
                        device=planner.device_cfg.device,
                    )
                )
        scene_model = planner.scene_collision_checker.scene_model
        if isinstance(scene_model, list):
            scene_model = scene_model[0]
        obstacle = scene_model.get_obstacle(str(object_name))
        if obstacle is None:
            raise KeyError(f"obstacle {object_name!r} is missing from the scene")
        return modules["CuroboPose"].from_list(
            list(obstacle.pose), planner.device_cfg
        )

    def _object_pose_to_proxy_pose(self, object_name: str, object_world_pose: Any) -> Any:
        collision_object = next(
            (item for item in self._scene.objects if item.name == str(object_name)),
            None,
        )
        center = (
            None
            if collision_object is None
            else collision_object.metadata.get("proxy_local_center")
        )
        if center is None:
            return object_world_pose
        local_center_pose = self._import_modules()["CuroboPose"].from_list(
            [*np.asarray(center, dtype=np.float64).reshape(3).tolist(), 1.0, 0.0, 0.0, 0.0],
            self._ensure_planner().device_cfg,
        )
        return object_world_pose.multiply(local_center_pose)

    def _attach_at_world_pose(self, object_name: str, state: Any, object_world_pose: Any) -> None:
        planner = self._ensure_planner()
        modules = self._import_modules()
        scene_model = planner.scene_collision_checker.scene_model
        if isinstance(scene_model, list):
            scene_model = scene_model[0]
        obstacle = scene_model.get_obstacle(str(object_name))
        if obstacle is None:
            raise KeyError(f"obstacle {object_name!r} is missing from the scene")
        collision_object = next(
            (item for item in self._scene.objects if item.name == str(object_name)),
            None,
        )
        local_obstacle = copy.deepcopy(obstacle)
        if collision_object is not None and collision_object.metadata.get("visual_mesh_path"):
            local_obstacle = modules["Mesh"](
                name=f"{object_name}__attachment_mesh",
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                file_path=str(collision_object.metadata["visual_mesh_path"]),
                scale=list(collision_object.metadata.get("visual_mesh_scale") or [1.0] * 3),
            )
        else:
            local_obstacle.pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        configured_spheres = (
            None
            if collision_object is None
            else collision_object.metadata.get("attachment_num_spheres")
        )
        max_spheres = int(self.config.attachment_num_spheres)
        if max_spheres > 0:
            manager = self._attachment_manager()
            requested_spheres = (
                None if configured_spheres is None else min(int(configured_spheres), max_spheres)
            )
            cache_path = None
            if collision_object is not None and collision_object.metadata.get("visual_mesh_path"):
                mesh_path = Path(collision_object.metadata["visual_mesh_path"]).resolve()
                stat = mesh_path.stat()
                cache_key = "|".join([
                    str(mesh_path),
                    str(stat.st_mtime_ns),
                    str(stat.st_size),
                    repr(collision_object.metadata.get("visual_mesh_scale")),
                    str(requested_spheres),
                    str(max_spheres),
                    str(float(self.config.attachment_surface_radius)),
                    "morphit-v2",
                ])
                cache_path = (
                    Path(self.config.attachment_cache_dir).expanduser().resolve()
                    / f"{hashlib.sha256(cache_key.encode('utf-8')).hexdigest()}.npz"
                )
            cache_hit = bool(cache_path is not None and cache_path.is_file())
            fit_debug: dict[str, Any] = {}
            if cache_hit:
                with np.load(cache_path) as cached:
                    cached_spheres = np.asarray(cached["spheres"], dtype=np.float32)
                sphere_tensor = modules["torch"].as_tensor(
                    cached_spheres,
                    dtype=planner.device_cfg.dtype,
                    device=planner.device_cfg.device,
                )
            else:
                sphere_tensor = manager.fit_spheres(
                    [local_obstacle],
                    num_spheres=requested_spheres,
                    surface_radius=float(self.config.attachment_surface_radius),
                    sphere_fit_type=modules["SphereFitType"].MORPHIT,
                )
                if sphere_tensor.shape[0] > max_spheres:
                    sphere_tensor = manager.fit_spheres(
                        [local_obstacle],
                        num_spheres=max_spheres,
                        surface_radius=float(self.config.attachment_surface_radius),
                        sphere_fit_type=modules["SphereFitType"].MORPHIT,
                    )
                fit_debug = dict(
                    getattr(getattr(manager, "_last_fit_result", None), "debug_info", {}) or {}
                )
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        cache_path,
                        spheres=sphere_tensor.detach().cpu().numpy(),
                    )
            fitted_sphere_count = int(sphere_tensor.shape[0])
            if fitted_sphere_count == 1:
                # cuRobo2 AttachmentManager.update currently squeezes a
                # transformed single center from [1, 1, 3] to [3], which
                # cannot concatenate with its [1, 1] radius tensor. Duplicate
                # the official fitted sphere; the collision union is identical.
                sphere_tensor = sphere_tensor.repeat(2, 1)
            object_pose_batch = object_world_pose
            state_batch_size = int(state.position.shape[0])
            if (
                int(getattr(object_pose_batch, "batch_size", 1)) == 1
                and state_batch_size > 1
            ):
                object_pose_batch = object_pose_batch.repeat(state_batch_size)
            manager.update(
                sphere_tensor,
                state,
                link_name="attached_object",
                world_objects_pose_offset=object_pose_batch,
            )
            self._attachment_fit_report = {
                "object_name": str(object_name),
                "fit_type": "morphit",
                "automatic_count": configured_spheres is None,
                "fitted_spheres": fitted_sphere_count,
                "stored_spheres": int(sphere_tensor.shape[0]),
                "max_spheres": max_spheres,
                "cache_hit": cache_hit,
                "cache_file": None if cache_path is None else str(cache_path),
                "used_convex_hull": bool(fit_debug.get("used_convex_hull", False)),
                "fallback_used": bool(fit_debug.get("fallback_used", False)),
            }
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
        self._update_obstacle_pose(
            str(object_name),
            self._object_pose_to_proxy_pose(object_name, object_world_pose),
        )
        self._attach_at_world_pose(object_name, state, object_world_pose)

    def detach_object(self, object_name: str, released_pose: Any | None = None) -> None:
        if self._planner is not None:
            if self._attachment_active:
                self._attachment_manager().detach(link_name="attached_object")
                self._attachment_active = False
                self._attachment_fit_report = None
            if released_pose is not None:
                modules = self._import_modules()
                object_world_pose = modules["CuroboPose"].from_list(
                    released_pose.as_curobo_list(), self._planner.device_cfg
                )
                self._update_obstacle_pose(
                    str(object_name),
                    self._object_pose_to_proxy_pose(object_name, object_world_pose),
                )
            if released_pose is None:
                self._set_obstacle_enabled(str(object_name), True)

    def enable_object_collision(self, object_name: str) -> None:
        self._set_obstacle_enabled(str(object_name), True)

    def close(self) -> None:
        if self._coarse_ik_solver is not None:
            self._coarse_ik_solver.destroy()
            self._coarse_ik_solver = None
        if self._planner is not None:
            self._planner.destroy()
            self._planner = None

    def __enter__(self) -> "Curobo2Backend":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
