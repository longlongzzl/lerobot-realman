#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


DEFAULT_CUROBO_ROOT = Path("/home/zhangzhao/PycharmProjects/curobo")
DEFAULT_RM75_URDF = Path("/home/zhangzhao/Desktop/lerobot/RM75_gripper/RM75-B/urdf/RM75-B.urdf")
DEFAULT_TORCH_EXTENSIONS_DIR = Path("/tmp/curobo_torch_extensions")

ARM_JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
]

GRIPPER_JOINT_NAMES = [
    "gripper_Left_1_Joint",
    "gripper_Left_Support_Joint",
    "gripper_Left_2_Joint",
    "gripper_Right_1_Joint",
    "gripper_Right_Support_Joint",
    "gripper_Right_2_Joint",
]

TRACKED_LINK_NAMES = [
    "gripper_base_link",
    "gripper_Left_1_Link",
    "gripper_Left_Support_Link",
    "gripper_Left_2_Link",
    "gripper_Right_1_Link",
    "gripper_Right_Support_Link",
    "gripper_Right_2_Link",
    "left_pad",
    "right_pad",
]

DEFAULT_RETRACT_CONFIG = [
    float(np.pi / 2.0),
    0.0,
    0.0,
    float(-np.pi / 2.0),
    0.0,
    float(-np.pi / 2.0),
    float(np.pi / 3.0),
]


@dataclass
class CuRoboPlanResult:
    success: bool
    status: Optional[str]
    goal_joint: Optional[np.ndarray] = None
    joint_path: Optional[np.ndarray] = None
    solve_time: float = 0.0
    ik_time: float = 0.0
    trajopt_time: float = 0.0
    raw_result: Any = None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class RM75CuRoboPlannerConfig:
    curobo_root: Path = DEFAULT_CUROBO_ROOT
    urdf: Path = DEFAULT_RM75_URDF
    robot_cfg_path: Optional[Path] = None
    base_link: str = "base_link"
    ee_link: str = "gripper_tcp"
    device: str = "cuda:0"
    torch_extensions_dir: Path = DEFAULT_TORCH_EXTENSIONS_DIR
    cuda_arch_list: Optional[str] = None
    gripper_lock: float = 0.6
    num_ik_seeds: int = 64
    num_trajopt_seeds: int = 1
    num_graph_seeds: int = 1
    interpolation_dt: float = 0.02
    position_threshold: float = 0.005
    rotation_threshold: float = 0.05
    use_cuda_graph: bool = False
    self_collision_check: bool = False
    self_collision_opt: bool = False


class RM75CuRoboPlanner:
    """Standalone cuRobo adapter for RM75.

    This file is meant for incremental integration:
    - keep your existing pose-generation logic
    - call `solve_ik()` or `plan_to_pose()` from the old pipeline
    - later swap `robot_cfg_path` to a formal rm75.yml with collision spheres
    """

    def __init__(self, config: Optional[RM75CuRoboPlannerConfig] = None):
        self.config = config or RM75CuRoboPlannerConfig()
        self._ensure_curobo_on_path(self.config.curobo_root)
        self._prepare_runtime_env()
        self.mods = self._import_curobo_modules()
        self.tensor_args = self._make_tensor_args()
        self.robot_cfg_dict = self._load_robot_cfg_dict()
        self.robot_cfg = self.mods["RobotConfig"].from_dict(
            self.robot_cfg_dict,
            tensor_args=self.tensor_args,
        )
        self._collision_enabled = self._robot_cfg_has_collision_model(self.robot_cfg_dict)
        self._empty_world = self.mods["WorldConfig"]()
        self._world = self._empty_world
        self.ik_solver = self._build_ik_solver()
        self.motion_gen = self._build_motion_gen()

    @property
    def joint_names(self) -> list[str]:
        return list(ARM_JOINT_NAMES)

    @property
    def collision_enabled(self) -> bool:
        return self._collision_enabled

    @property
    def using_embedded_robot_cfg(self) -> bool:
        return self.config.robot_cfg_path is None

    @property
    def configured_collision_links(self) -> list[str]:
        robot_cfg_dict: Mapping[str, Any] = self.robot_cfg_dict
        if "robot_cfg" in robot_cfg_dict:
            robot_cfg_dict = robot_cfg_dict["robot_cfg"]
        kinematics = robot_cfg_dict.get("kinematics", {})
        collision_link_names = kinematics.get("collision_link_names") or []
        return [str(x) for x in list(collision_link_names)]

    def fk(self, q: Sequence[float]) -> dict[str, list[float]]:
        start_state = self._make_start_state(q)
        ee_pose = self.motion_gen.compute_kinematics(start_state).ee_pose
        return self._pose_to_dict(ee_pose)

    def check_start_state(self, q: Sequence[float]) -> tuple[bool, Optional[str]]:
        start_state = self._make_start_state(q)
        valid, status = self.motion_gen.check_start_state(start_state)
        return bool(valid), None if status is None else str(status)

    def set_world_collision_for_links(
        self,
        collision_link_names: Sequence[str],
        *,
        enabled: bool,
    ) -> list[str]:
        link_names = [str(x) for x in list(collision_link_names or []) if str(x)]
        if not link_names:
            return []
        self.motion_gen.toggle_link_collision(link_names, bool(enabled))
        try:
            ik_kinematics = getattr(self.ik_solver, "kinematics", None)
            ik_kin_cfg = getattr(ik_kinematics, "kinematics_config", None)
            if ik_kin_cfg is not None:
                for link_name in link_names:
                    if enabled:
                        ik_kin_cfg.enable_link_spheres(link_name)
                    else:
                        ik_kin_cfg.disable_link_spheres(link_name)
        except Exception:
            pass
        return link_names

    def diagnose_start_state_world_collision(self, q: Sequence[float]) -> dict[str, Any]:
        q_np = self._normalize_q(q)
        valid, status = self.check_start_state(q_np)
        world = self._world
        obstacle_names = [
            str(getattr(obj, "name", f"obstacle_{idx:02d}"))
            for idx, obj in enumerate(list(getattr(world, "objects", []) or []))
        ]
        diagnosis: dict[str, Any] = {
            "valid": bool(valid),
            "status": status,
            "world_obstacle_names": obstacle_names,
            "ablation": [],
        }
        if valid or status != "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION" or not obstacle_names:
            return diagnosis

        original_world = world
        try:
            for name in obstacle_names:
                ablated_world = self._world_without_obstacles(original_world, excluded_names={name})
                self.motion_gen.update_world(ablated_world)
                self.ik_solver.update_world(ablated_world)
                ablated_valid, ablated_status = self.check_start_state(q_np)
                diagnosis["ablation"].append(
                    {
                        "removed": str(name),
                        "valid": bool(ablated_valid),
                        "status": ablated_status,
                    }
                )
        finally:
            self.motion_gen.update_world(original_world)
            self.ik_solver.update_world(original_world)
            self._world = original_world
        return diagnosis

    def solve_ik(
        self,
        start_q: Sequence[float],
        goal_pose: Any,
        *,
        num_seeds: Optional[int] = None,
    ) -> CuRoboPlanResult:
        start_q_np = self._normalize_q(start_q)
        start_state = self._make_start_state(start_q_np)
        goal = self._make_pose(goal_pose)
        use_num_seeds = self.config.num_ik_seeds if num_seeds is None else int(num_seeds)
        self.ik_solver.reset_seed()
        result = self.ik_solver.solve_single(
            goal,
            retract_config=start_state.position.clone(),
            seed_config=start_state.position.view(1, 1, -1).clone(),
            return_seeds=use_num_seeds,
            num_seeds=use_num_seeds,
            use_nn_seed=False,
        )
        best_q = self._nearest_success_solution(result.solution, result.success, start_q_np)
        success = best_q is not None
        return CuRoboPlanResult(
            success=success,
            status="Success" if success else "IK_FAIL",
            goal_joint=None if best_q is None else np.asarray(best_q, dtype=np.float32),
            solve_time=float(result.solve_time),
            ik_time=float(result.solve_time),
            raw_result=result,
            debug={
                "position_error": float(result.position_error.reshape(-1)[0].item()),
                "rotation_error": float(result.rotation_error.reshape(-1)[0].item()),
                "ik_success_count": int(self.mods["torch"].count_nonzero(result.success).item()),
            },
        )

    def solve_pose_ik(
        self,
        start_q: Sequence[float],
        goal_pose: Any,
        *,
        num_seeds: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        result = self.solve_ik(start_q, goal_pose, num_seeds=num_seeds)
        return result.goal_joint

    def plan_to_pose(
        self,
        start_q: Sequence[float],
        goal_pose: Any,
        *,
        enable_graph: bool = False,
        max_attempts: int = 2,
        timeout: float = 5.0,
        num_ik_seeds: Optional[int] = None,
        num_trajopt_seeds: Optional[int] = None,
        num_graph_seeds: Optional[int] = None,
        pose_cost_metric: Optional[Any] = None,
    ) -> CuRoboPlanResult:
        start_q_np = self._normalize_q(start_q)
        start_state = self._make_start_state(start_q_np)
        goal = self._make_pose(goal_pose)

        use_num_ik_seeds = self.config.num_ik_seeds if num_ik_seeds is None else int(num_ik_seeds)
        use_num_trajopt_seeds = (
            self.config.num_trajopt_seeds if num_trajopt_seeds is None else int(num_trajopt_seeds)
        )
        use_num_graph_seeds = (
            self.config.num_graph_seeds if num_graph_seeds is None else int(num_graph_seeds)
        )

        self.motion_gen.reset_seed()
        motion_ik_result = self.motion_gen.ik_solver.solve_single(
            goal,
            retract_config=start_state.position.clone(),
            seed_config=start_state.position.view(1, 1, -1).clone(),
            return_seeds=use_num_trajopt_seeds,
            num_seeds=use_num_ik_seeds,
            use_nn_seed=False,
        )
        motion_ik_successes = int(self.mods["torch"].count_nonzero(motion_ik_result.success).item())

        plan_config = self.mods["MotionGenPlanConfig"](
            enable_graph=bool(enable_graph),
            enable_opt=True,
            max_attempts=int(max_attempts),
            timeout=float(timeout),
            num_ik_seeds=use_num_ik_seeds,
            num_graph_seeds=use_num_graph_seeds,
            num_trajopt_seeds=use_num_trajopt_seeds,
            pose_cost_metric=pose_cost_metric,
        )

        self.motion_gen.reset_seed()
        result = self.motion_gen.plan_single(start_state, goal, plan_config)
        success = bool(result.success.reshape(-1)[0].item())
        status = None if result.status is None else str(result.status)
        if not success:
            return CuRoboPlanResult(
                success=False,
                status=status,
                solve_time=float(result.solve_time),
                ik_time=float(result.ik_time),
                trajopt_time=float(result.trajopt_time),
                raw_result=result,
                debug={"motion_gen_internal_ik_successes": motion_ik_successes},
            )

        traj = result.get_interpolated_plan()
        q_path = self._to_numpy(traj.position).astype(np.float32)
        return CuRoboPlanResult(
            success=True,
            status=status,
            goal_joint=np.asarray(q_path[-1], dtype=np.float32),
            joint_path=q_path,
            solve_time=float(result.solve_time),
            ik_time=float(result.ik_time),
            trajopt_time=float(result.trajopt_time),
            raw_result=result,
            debug={
                "motion_gen_internal_ik_successes": motion_ik_successes,
                "interpolation_dt": float(result.interpolation_dt),
            },
        )

    def plan_goalset_to_poses(
        self,
        start_q: Sequence[float],
        goal_poses: Sequence[Any],
        *,
        enable_graph: bool = False,
        max_attempts: int = 2,
        timeout: float = 5.0,
        num_ik_seeds: Optional[int] = None,
        num_trajopt_seeds: Optional[int] = None,
        num_graph_seeds: Optional[int] = None,
    ) -> CuRoboPlanResult:
        goal_poses = list(goal_poses or [])
        if len(goal_poses) == 0:
            return CuRoboPlanResult(success=False, status="EMPTY_GOALSET")

        start_q_np = self._normalize_q(start_q)
        start_state = self._make_start_state(start_q_np)
        goal = self._make_goalset_pose(goal_poses)

        use_num_ik_seeds = self.config.num_ik_seeds if num_ik_seeds is None else int(num_ik_seeds)
        use_num_trajopt_seeds = (
            self.config.num_trajopt_seeds if num_trajopt_seeds is None else int(num_trajopt_seeds)
        )
        use_num_graph_seeds = (
            self.config.num_graph_seeds if num_graph_seeds is None else int(num_graph_seeds)
        )

        plan_config = self.mods["MotionGenPlanConfig"](
            enable_graph=bool(enable_graph),
            enable_opt=True,
            max_attempts=int(max_attempts),
            timeout=float(timeout),
            num_ik_seeds=use_num_ik_seeds,
            num_graph_seeds=use_num_graph_seeds,
            num_trajopt_seeds=use_num_trajopt_seeds,
        )

        self.motion_gen.reset_seed()
        result = self.motion_gen.plan_goalset(start_state, goal, plan_config)
        success = bool(result.success.reshape(-1)[0].item())
        status = None if result.status is None else str(result.status)
        if not success:
            return CuRoboPlanResult(
                success=False,
                status=status,
                solve_time=float(result.solve_time),
                ik_time=float(result.ik_time),
                trajopt_time=float(result.trajopt_time),
                raw_result=result,
                debug={},
            )

        traj = result.get_interpolated_plan()
        q_path = self._to_numpy(traj.position).astype(np.float32)
        goal_index = None
        if getattr(result, "goalset_index", None) is not None:
            goal_index = int(self._to_numpy(result.goalset_index).reshape(-1)[0])
        return CuRoboPlanResult(
            success=True,
            status=status,
            goal_joint=np.asarray(q_path[-1], dtype=np.float32),
            joint_path=q_path,
            solve_time=float(result.solve_time),
            ik_time=float(result.ik_time),
            trajopt_time=float(result.trajopt_time),
            raw_result=result,
            debug={
                "goalset_index": goal_index,
                "interpolation_dt": float(result.interpolation_dt),
            },
        )

    def plan_batch_to_poses(
        self,
        start_q: Sequence[float],
        goal_poses: Sequence[Any],
        *,
        enable_graph: bool = False,
        max_attempts: int = 2,
        timeout: float = 5.0,
        num_ik_seeds: Optional[int] = None,
        num_trajopt_seeds: Optional[int] = None,
        num_graph_seeds: Optional[int] = None,
    ) -> list[CuRoboPlanResult]:
        goal_poses = list(goal_poses or [])
        if len(goal_poses) == 0:
            return []
        if len(goal_poses) == 1:
            single_result = self.plan_to_pose(
                start_q,
                goal_poses[0],
                enable_graph=enable_graph,
                max_attempts=max_attempts,
                timeout=timeout,
                num_ik_seeds=num_ik_seeds,
                num_trajopt_seeds=num_trajopt_seeds,
                num_graph_seeds=num_graph_seeds,
            )
            return [single_result]

        start_q_np = self._normalize_q(start_q)
        start_state = self._make_batched_start_state(start_q_np, len(goal_poses))
        goal = self._make_batch_pose(goal_poses)

        use_num_ik_seeds = self.config.num_ik_seeds if num_ik_seeds is None else int(num_ik_seeds)
        use_num_trajopt_seeds = (
            self.config.num_trajopt_seeds if num_trajopt_seeds is None else int(num_trajopt_seeds)
        )
        use_num_graph_seeds = (
            self.config.num_graph_seeds if num_graph_seeds is None else int(num_graph_seeds)
        )

        plan_config = self.mods["MotionGenPlanConfig"](
            enable_graph=bool(enable_graph),
            enable_opt=True,
            max_attempts=int(max_attempts),
            timeout=float(timeout),
            num_ik_seeds=use_num_ik_seeds,
            num_graph_seeds=use_num_graph_seeds,
            num_trajopt_seeds=use_num_trajopt_seeds,
        )

        self.motion_gen.reset_seed()
        result = self.motion_gen.plan_batch(start_state, goal, plan_config)
        success_arr = self._to_numpy(result.success).reshape(-1).astype(bool)
        status = None if result.status is None else str(result.status)
        paths = self._extract_batch_paths(result, len(goal_poses))
        outputs: list[CuRoboPlanResult] = []
        for idx, success in enumerate(success_arr.tolist()):
            q_path = None
            goal_joint = None
            if success and idx < len(paths) and paths[idx] is not None:
                q_path = self._to_numpy(paths[idx].position).astype(np.float32)
                if q_path.ndim == 1:
                    q_path = q_path.reshape(1, -1)
                goal_joint = np.asarray(q_path[-1], dtype=np.float32)
            outputs.append(
                CuRoboPlanResult(
                    success=bool(success),
                    status=status,
                    goal_joint=goal_joint,
                    joint_path=q_path,
                    solve_time=float(result.solve_time),
                    ik_time=float(result.ik_time),
                    trajopt_time=float(result.trajopt_time),
                    raw_result=result,
                    debug={
                        "batch_index": int(idx),
                        "interpolation_dt": float(result.interpolation_dt),
                    },
                )
        )
        return outputs

    def plan_batch_start_goal_pairs(
        self,
        start_qs: Sequence[Sequence[float]],
        goal_poses: Sequence[Any],
        *,
        enable_graph: bool = False,
        max_attempts: int = 2,
        timeout: float = 5.0,
        num_ik_seeds: Optional[int] = None,
        num_trajopt_seeds: Optional[int] = None,
        num_graph_seeds: Optional[int] = None,
    ) -> list[CuRoboPlanResult]:
        start_qs = list(start_qs or [])
        goal_poses = list(goal_poses or [])
        if len(start_qs) == 0 or len(goal_poses) == 0:
            return []
        if len(start_qs) != len(goal_poses):
            raise ValueError(
                f"start_qs and goal_poses must have the same length, got {len(start_qs)} and {len(goal_poses)}"
            )
        if len(goal_poses) == 1:
            single_result = self.plan_to_pose(
                start_qs[0],
                goal_poses[0],
                enable_graph=enable_graph,
                max_attempts=max_attempts,
                timeout=timeout,
                num_ik_seeds=num_ik_seeds,
                num_trajopt_seeds=num_trajopt_seeds,
                num_graph_seeds=num_graph_seeds,
            )
            return [single_result]

        start_state = self._make_multi_start_state(start_qs)
        goal = self._make_batch_pose(goal_poses)

        use_num_ik_seeds = self.config.num_ik_seeds if num_ik_seeds is None else int(num_ik_seeds)
        use_num_trajopt_seeds = (
            self.config.num_trajopt_seeds if num_trajopt_seeds is None else int(num_trajopt_seeds)
        )
        use_num_graph_seeds = (
            self.config.num_graph_seeds if num_graph_seeds is None else int(num_graph_seeds)
        )

        plan_config = self.mods["MotionGenPlanConfig"](
            enable_graph=bool(enable_graph),
            enable_opt=True,
            max_attempts=int(max_attempts),
            timeout=float(timeout),
            num_ik_seeds=use_num_ik_seeds,
            num_graph_seeds=use_num_graph_seeds,
            num_trajopt_seeds=use_num_trajopt_seeds,
        )

        self.motion_gen.reset_seed()
        result = self.motion_gen.plan_batch(start_state, goal, plan_config)
        success_arr = self._to_numpy(result.success).reshape(-1).astype(bool)
        status = None if result.status is None else str(result.status)
        paths = self._extract_batch_paths(result, len(goal_poses))
        outputs: list[CuRoboPlanResult] = []
        for idx, success in enumerate(success_arr.tolist()):
            q_path = None
            goal_joint = None
            if success and idx < len(paths) and paths[idx] is not None:
                q_path = self._to_numpy(paths[idx].position).astype(np.float32)
                if q_path.ndim == 1:
                    q_path = q_path.reshape(1, -1)
                goal_joint = np.asarray(q_path[-1], dtype=np.float32)
            outputs.append(
                CuRoboPlanResult(
                    success=bool(success),
                    status=status,
                    goal_joint=goal_joint,
                    joint_path=q_path,
                    solve_time=float(result.solve_time),
                    ik_time=float(result.ik_time),
                    trajopt_time=float(result.trajopt_time),
                    raw_result=result,
                    debug={
                        "batch_index": int(idx),
                        "interpolation_dt": float(result.interpolation_dt),
                    },
                )
            )
        return outputs

    def plan_pose_path(
        self,
        start_q: Sequence[float],
        goal_pose: Any,
        *,
        enable_graph: bool = False,
        max_attempts: int = 2,
        timeout: float = 5.0,
        num_ik_seeds: Optional[int] = None,
        num_trajopt_seeds: Optional[int] = None,
        num_graph_seeds: Optional[int] = None,
    ) -> Optional[list[np.ndarray]]:
        result = self.plan_to_pose(
            start_q,
            goal_pose,
            enable_graph=enable_graph,
            max_attempts=max_attempts,
            timeout=timeout,
            num_ik_seeds=num_ik_seeds,
            num_trajopt_seeds=num_trajopt_seeds,
            num_graph_seeds=num_graph_seeds,
        )
        if not result.success or result.joint_path is None:
            return None
        return [np.asarray(q, dtype=np.float32) for q in result.joint_path]

    def diagnose_pose_goal(
        self,
        start_q: Sequence[float],
        goal_pose: Any,
        *,
        num_ik_seeds: Optional[int] = None,
        num_trajopt_seeds: Optional[int] = None,
    ) -> dict[str, Any]:
        start_q_np = self._normalize_q(start_q)
        start_state = self._make_start_state(start_q_np)
        goal = self._make_pose(goal_pose)

        use_num_ik_seeds = self.config.num_ik_seeds if num_ik_seeds is None else int(num_ik_seeds)
        use_num_trajopt_seeds = (
            self.config.num_trajopt_seeds if num_trajopt_seeds is None else int(num_trajopt_seeds)
        )

        start_fk = self.motion_gen.compute_kinematics(start_state).ee_pose

        self.ik_solver.reset_seed()
        standalone_result = self.ik_solver.solve_single(
            goal,
            retract_config=start_state.position.clone(),
            seed_config=start_state.position.view(1, 1, -1).clone(),
            return_seeds=1,
            num_seeds=use_num_ik_seeds,
            use_nn_seed=False,
        )
        standalone_q = self._first_success_solution(standalone_result.solution, standalone_result.success)

        self.motion_gen.reset_seed()
        motiongen_ik_result = self.motion_gen.ik_solver.solve_single(
            goal,
            retract_config=start_state.position.clone(),
            seed_config=start_state.position.view(1, 1, -1).clone(),
            return_seeds=use_num_trajopt_seeds,
            num_seeds=use_num_ik_seeds,
            use_nn_seed=False,
        )
        motiongen_ik_q = self._first_success_solution(
            motiongen_ik_result.solution,
            motiongen_ik_result.success,
        )
        motiongen_ik_successes = int(
            self.mods["torch"].count_nonzero(motiongen_ik_result.success).item()
        )

        goal_position = self._to_numpy(goal.position).reshape(-1, 3)[0]
        start_fk_position = self._to_numpy(start_fk.position).reshape(-1, 3)[0]

        return {
            "start_q": start_q_np.tolist(),
            "start_fk_position": np.round(start_fk_position, 9).tolist(),
            "start_fk_quaternion": np.round(
                self._to_numpy(start_fk.quaternion).reshape(-1, 4)[0],
                9,
            ).tolist(),
            "goal_position": np.round(goal_position, 9).tolist(),
            "goal_quaternion": np.round(
                self._to_numpy(goal.quaternion).reshape(-1, 4)[0],
                9,
            ).tolist(),
            "goal_translation_delta_norm": float(np.linalg.norm(goal_position - start_fk_position)),
            "collision_enabled": bool(self.collision_enabled),
            "using_embedded_robot_cfg": bool(self.using_embedded_robot_cfg),
            "attached_object_collision_enabled": False,
            "joint_names": list(self.joint_names),
            "num_ik_seeds": int(use_num_ik_seeds),
            "num_trajopt_seeds": int(use_num_trajopt_seeds),
            "standalone_ik_success": standalone_q is not None,
            "standalone_ik_solution": None if standalone_q is None else np.round(standalone_q, 9).tolist(),
            "standalone_ik_solve_time": float(standalone_result.solve_time),
            "standalone_ik_position_error": float(standalone_result.position_error.reshape(-1)[0].item()),
            "standalone_ik_rotation_error": float(standalone_result.rotation_error.reshape(-1)[0].item()),
            "motiongen_internal_ik_successes": motiongen_ik_successes,
            "motiongen_internal_ik_solution": None if motiongen_ik_q is None else np.round(motiongen_ik_q, 9).tolist(),
            "motiongen_internal_ik_solve_time": float(motiongen_ik_result.solve_time),
            "motiongen_internal_ik_position_error_min": float(
                self._to_numpy(motiongen_ik_result.position_error).reshape(-1).min()
            ),
            "motiongen_internal_ik_rotation_error_min": float(
                self._to_numpy(motiongen_ik_result.rotation_error).reshape(-1).min()
            ),
        }

    def dump_diagnostics_json(self, path: Path, payload: Mapping[str, Any]) -> Path:
        out_path = path.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
        return out_path

    def set_world_from_cuboids(self, cuboids: Sequence[Mapping[str, Any]]) -> None:
        self.set_world_from_obstacles(cuboids=cuboids, meshes=())

    def set_world_from_obstacles(
        self,
        *,
        cuboids: Sequence[Mapping[str, Any]] = (),
        meshes: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        if not self.collision_enabled:
            raise RuntimeError(
                "This planner is running with an embedded free-space RM75 config. "
                "World collision updates need a formal robot config with collision_spheres."
            )
        world_cfg = self.build_world_from_obstacles(cuboids=cuboids, meshes=meshes)
        self.motion_gen.update_world(world_cfg)
        self.ik_solver.update_world(world_cfg)
        self._world = world_cfg

    def clear_world(self) -> None:
        if not self.collision_enabled:
            return
        self.motion_gen.update_world(self._empty_world)
        self.ik_solver.update_world(self._empty_world)
        self._world = self._empty_world

    def build_world_from_cuboids(self, cuboids: Sequence[Mapping[str, Any]]):
        return self.build_world_from_obstacles(cuboids=cuboids, meshes=())

    def build_world_from_obstacles(
        self,
        *,
        cuboids: Sequence[Mapping[str, Any]] = (),
        meshes: Sequence[Mapping[str, Any]] = (),
    ):
        if len(cuboids) == 0 and len(meshes) == 0:
            return self.mods["WorldConfig"]()
        world_dict: dict[str, dict[str, Any]] = {}
        if len(cuboids) > 0:
            world_dict["cuboid"] = {}
        for idx, cuboid in enumerate(cuboids):
            if not isinstance(cuboid, Mapping):
                raise TypeError(f"cuboid[{idx}] must be a mapping, got {type(cuboid)!r}")
            name = str(cuboid.get("name", f"obstacle_{idx:02d}"))
            dims = self._as_float_array(cuboid.get("dims"), expected=3, name=f"{name}.dims")
            pose = self._cuboid_mapping_to_pose_array(cuboid)
            world_dict["cuboid"][name] = {
                "dims": dims.tolist(),
                "pose": pose.tolist(),
            }
        if len(meshes) > 0:
            world_dict["mesh"] = {}
        for idx, mesh in enumerate(meshes):
            if not isinstance(mesh, Mapping):
                raise TypeError(f"mesh[{idx}] must be a mapping, got {type(mesh)!r}")
            name = str(mesh.get("name", f"mesh_{idx:02d}"))
            pose = self._mesh_mapping_to_pose_array(mesh)
            mesh_entry: dict[str, Any] = {"pose": pose.tolist()}
            file_path = mesh.get("file_path", mesh.get("asset_file", mesh.get("mesh_file")))
            if file_path is not None:
                mesh_entry["file_path"] = str(file_path)
            scale_value = mesh.get("scale", mesh.get("asset_scale", [1.0, 1.0, 1.0]))
            mesh_entry["scale"] = self._scale_to_xyz_array(scale_value, name=f"{name}.scale").tolist()
            if mesh_entry.get("file_path") is None:
                vertices = mesh.get("vertices")
                faces = mesh.get("faces")
                if vertices is None or faces is None:
                    raise ValueError(
                        f"{name} requires file_path or both vertices and faces for cuRobo mesh world"
                    )
                mesh_entry["vertices"] = np.asarray(vertices, dtype=np.float32).tolist()
                mesh_entry["faces"] = np.asarray(faces, dtype=np.int32).tolist()
            world_dict["mesh"][name] = mesh_entry
        return self.mods["WorldConfig"].from_dict(world_dict)

    def _build_ik_solver(self):
        ik_config = self.mods["IKSolverConfig"].load_from_robot_config(
            self.robot_cfg,
            self._world,
            tensor_args=self.tensor_args,
            num_seeds=int(self.config.num_ik_seeds),
            position_threshold=float(self.config.position_threshold),
            rotation_threshold=float(self.config.rotation_threshold),
            use_cuda_graph=bool(self.config.use_cuda_graph),
            self_collision_check=bool(self.config.self_collision_check),
            self_collision_opt=bool(self.config.self_collision_opt),
        )
        return self.mods["IKSolver"](ik_config)

    def _build_motion_gen(self):
        motion_gen_config = self.mods["MotionGenConfig"].load_from_robot_config(
            self.robot_cfg,
            self._world,
            tensor_args=self.tensor_args,
            num_ik_seeds=int(self.config.num_ik_seeds),
            num_graph_seeds=int(self.config.num_graph_seeds),
            num_trajopt_seeds=int(self.config.num_trajopt_seeds),
            interpolation_dt=float(self.config.interpolation_dt),
            position_threshold=float(self.config.position_threshold),
            rotation_threshold=float(self.config.rotation_threshold),
            use_cuda_graph=bool(self.config.use_cuda_graph),
            self_collision_check=bool(self.config.self_collision_check),
            self_collision_opt=bool(self.config.self_collision_opt),
        )
        return self.mods["MotionGen"](motion_gen_config)

    def _world_without_obstacles(self, world, *, excluded_names: set[str]):
        excluded_names = {str(x) for x in excluded_names}
        return self.mods["WorldConfig"](
            cuboid=[x for x in list(getattr(world, "cuboid", []) or []) if str(getattr(x, "name", "")) not in excluded_names],
            sphere=[x for x in list(getattr(world, "sphere", []) or []) if str(getattr(x, "name", "")) not in excluded_names],
            capsule=[x for x in list(getattr(world, "capsule", []) or []) if str(getattr(x, "name", "")) not in excluded_names],
            cylinder=[x for x in list(getattr(world, "cylinder", []) or []) if str(getattr(x, "name", "")) not in excluded_names],
            mesh=[x for x in list(getattr(world, "mesh", []) or []) if str(getattr(x, "name", "")) not in excluded_names],
            blox=[x for x in list(getattr(world, "blox", []) or []) if str(getattr(x, "name", "")) not in excluded_names],
            voxel=[x for x in list(getattr(world, "voxel", []) or []) if str(getattr(x, "name", "")) not in excluded_names],
        )

    def _load_robot_cfg_dict(self) -> dict[str, Any]:
        if self.config.robot_cfg_path is not None:
            robot_cfg_path = self.config.robot_cfg_path.expanduser().resolve()
            if not robot_cfg_path.is_file():
                raise FileNotFoundError(f"robot cfg yaml was not found: {robot_cfg_path}")
            data = self.mods["load_yaml"](str(robot_cfg_path))
            if not isinstance(data, dict):
                raise TypeError(f"robot cfg yaml must load as a mapping, got {type(data)!r}")
            return data
        return self._build_embedded_rm75_robot_cfg_dict()

    def _build_embedded_rm75_robot_cfg_dict(self) -> dict[str, Any]:
        urdf_path = self.config.urdf.expanduser().resolve()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"RM75 URDF was not found: {urdf_path}")
        lock_joints = {
            joint_name: float(self.config.gripper_lock)
            for joint_name in GRIPPER_JOINT_NAMES
        }
        robot_root = urdf_path.parent.parent
        return {
            "robot_cfg": {
                "kinematics": {
                    "use_usd_kinematics": False,
                    "urdf_path": str(urdf_path),
                    "asset_root_path": str(robot_root),
                    "base_link": str(self.config.base_link),
                    "ee_link": str(self.config.ee_link),
                    "link_names": [str(self.config.ee_link)] + list(TRACKED_LINK_NAMES),
                    "lock_joints": lock_joints,
                    "extra_links": None,
                    "collision_link_names": None,
                    "collision_spheres": None,
                    "collision_sphere_buffer": 0.0,
                    "extra_collision_spheres": None,
                    "self_collision_ignore": None,
                    "self_collision_buffer": None,
                    "use_global_cumul": True,
                    "mesh_link_names": None,
                    "cspace": {
                        "joint_names": list(ARM_JOINT_NAMES),
                        "retract_config": list(DEFAULT_RETRACT_CONFIG),
                        "null_space_weight": [1.0] * len(ARM_JOINT_NAMES),
                        "cspace_distance_weight": [1.0] * len(ARM_JOINT_NAMES),
                        "max_acceleration": 12.0,
                        "max_jerk": 500.0,
                    },
                }
            }
        }

    def _prepare_runtime_env(self) -> None:
        torch_extensions_dir = self.config.torch_extensions_dir.expanduser().resolve()
        torch_extensions_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(torch_extensions_dir))
        if self.config.cuda_arch_list:
            os.environ["TORCH_CUDA_ARCH_LIST"] = str(self.config.cuda_arch_list)
        elif "TORCH_CUDA_ARCH_LIST" not in os.environ:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                arch_list = []
                for line in result.stdout.splitlines():
                    arch = line.strip()
                    if arch and arch not in arch_list:
                        arch_list.append(arch)
                if arch_list:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)
            except Exception:
                pass

    def _ensure_curobo_on_path(self, curobo_root: Path) -> None:
        src_dir = curobo_root.expanduser().resolve() / "src"
        if not src_dir.is_dir():
            raise FileNotFoundError(f"cuRobo src directory was not found: {src_dir}")
        src_str = str(src_dir)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)

    def _import_curobo_modules(self) -> dict[str, Any]:
        import torch
        from curobo.geom.types import WorldConfig
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose
        from curobo.types.robot import RobotConfig
        from curobo.types.state import JointState
        from curobo.util_file import load_yaml
        from curobo.rollout.cost.pose_cost import PoseCostMetric
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

        return {
            "torch": torch,
            "WorldConfig": WorldConfig,
            "TensorDeviceType": TensorDeviceType,
            "Pose": Pose,
            "RobotConfig": RobotConfig,
            "JointState": JointState,
            "load_yaml": load_yaml,
            "PoseCostMetric": PoseCostMetric,
            "IKSolver": IKSolver,
            "IKSolverConfig": IKSolverConfig,
            "MotionGen": MotionGen,
            "MotionGenConfig": MotionGenConfig,
            "MotionGenPlanConfig": MotionGenPlanConfig,
        }

    def _make_tensor_args(self):
        device = self.mods["torch"].device(self.config.device)
        return self.mods["TensorDeviceType"](device=device)

    def _make_start_state(self, q: Sequence[float]):
        q_tensor = self.mods["torch"].as_tensor(
            self._normalize_q(q),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        ).view(1, -1)
        return self.mods["JointState"].from_position(q_tensor, joint_names=list(ARM_JOINT_NAMES))

    def _make_batched_start_state(self, q: Sequence[float], batch_size: int):
        q_tensor = self.mods["torch"].as_tensor(
            self._normalize_q(q),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        ).view(1, -1).repeat(int(batch_size), 1)
        return self.mods["JointState"].from_position(q_tensor, joint_names=list(ARM_JOINT_NAMES))

    def _make_multi_start_state(self, qs: Sequence[Sequence[float]]):
        q_tensor = self.mods["torch"].as_tensor(
            np.asarray([self._normalize_q(q) for q in list(qs or [])], dtype=np.float32),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        ).view(len(qs), -1)
        return self.mods["JointState"].from_position(q_tensor, joint_names=list(ARM_JOINT_NAMES))

    def _make_pose(self, pose_like: Any):
        position, quaternion = self._extract_pose_components(pose_like)
        pos_tensor = self.mods["torch"].as_tensor(
            position,
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        ).view(1, 3)
        quat_tensor = self.mods["torch"].as_tensor(
            quaternion,
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        ).view(1, 4)
        return self.mods["Pose"](position=pos_tensor, quaternion=quat_tensor)

    def _make_goalset_pose(self, goal_poses: Sequence[Any]):
        positions = []
        quaternions = []
        for pose_like in list(goal_poses or []):
            position, quaternion = self._extract_pose_components(pose_like)
            positions.append(position)
            quaternions.append(quaternion)
        pos_tensor = self.mods["torch"].as_tensor(
            np.asarray(positions, dtype=np.float32),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        ).view(1, len(positions), 3)
        quat_tensor = self.mods["torch"].as_tensor(
            np.asarray(quaternions, dtype=np.float32),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        ).view(1, len(quaternions), 4)
        return self.mods["Pose"](position=pos_tensor, quaternion=quat_tensor)

    def _make_batch_pose(self, goal_poses: Sequence[Any]):
        positions = []
        quaternions = []
        for pose_like in list(goal_poses or []):
            position, quaternion = self._extract_pose_components(pose_like)
            positions.append(position)
            quaternions.append(quaternion)
        pos_tensor = self.mods["torch"].as_tensor(
            np.asarray(positions, dtype=np.float32),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        ).view(len(positions), 3)
        quat_tensor = self.mods["torch"].as_tensor(
            np.asarray(quaternions, dtype=np.float32),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        ).view(len(quaternions), 4)
        return self.mods["Pose"](position=pos_tensor, quaternion=quat_tensor)

    def _extract_pose_components(self, pose_like: Any) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(pose_like, "position") and hasattr(pose_like, "quaternion"):
            position = self._as_float_array(pose_like.position, expected=3, name="pose.position")
            quaternion = self._as_float_array(
                pose_like.quaternion,
                expected=4,
                name="pose.quaternion",
            )
            return position, quaternion

        if hasattr(pose_like, "p") and hasattr(pose_like, "q"):
            position = self._as_float_array(pose_like.p, expected=3, name="pose.p")
            quaternion = self._as_float_array(pose_like.q, expected=4, name="pose.q")
            return position, quaternion

        if isinstance(pose_like, Mapping):
            if "pose" in pose_like:
                pose_array = self._as_float_array(pose_like["pose"], expected=7, name="pose")
                return pose_array[:3], pose_array[3:7]
            position_key = "position" if "position" in pose_like else "p"
            quat_key = "quaternion" if "quaternion" in pose_like else "q"
            if position_key in pose_like and quat_key in pose_like:
                position = self._as_float_array(pose_like[position_key], expected=3, name=position_key)
                quaternion = self._as_float_array(pose_like[quat_key], expected=4, name=quat_key)
                return position, quaternion

        pose_array = self._as_float_array(pose_like, expected=7, name="pose")
        return pose_array[:3], pose_array[3:7]

    def _cuboid_mapping_to_pose_array(self, cuboid: Mapping[str, Any]) -> np.ndarray:
        if "pose" in cuboid:
            return self._as_float_array(cuboid["pose"], expected=7, name="cuboid.pose")
        if "position" in cuboid or "center" in cuboid:
            position_key = "position" if "position" in cuboid else "center"
            position = self._as_float_array(cuboid[position_key], expected=3, name=position_key)
            quat_value = (
                cuboid.get("quaternion")
                or cuboid.get("q")
                or cuboid.get("orientation")
                or [1.0, 0.0, 0.0, 0.0]
            )
            quaternion = self._as_float_array(quat_value, expected=4, name="cuboid.quaternion")
            return np.concatenate([position, quaternion]).astype(np.float32)
        raise KeyError("cuboid mapping must contain either pose or position/center + quaternion")

    def _mesh_mapping_to_pose_array(self, mesh: Mapping[str, Any]) -> np.ndarray:
        if "pose" in mesh:
            return self._as_float_array(mesh["pose"], expected=7, name="mesh.pose")
        if "position" in mesh or "center" in mesh:
            position_key = "position" if "position" in mesh else "center"
            position = self._as_float_array(mesh[position_key], expected=3, name=position_key)
            quat_value = (
                mesh.get("quaternion")
                or mesh.get("q")
                or mesh.get("orientation")
                or [1.0, 0.0, 0.0, 0.0]
            )
            quaternion = self._as_float_array(quat_value, expected=4, name="mesh.quaternion")
            return np.concatenate([position, quaternion]).astype(np.float32)
        raise KeyError("mesh mapping must contain either pose or position/center + quaternion")

    def _scale_to_xyz_array(self, scale_like: Any, *, name: str) -> np.ndarray:
        arr = np.asarray(scale_like, dtype=np.float32).reshape(-1)
        if arr.size == 1:
            arr = np.repeat(arr, 3)
        if arr.size != 3:
            raise ValueError(f"{name} must have 1 or 3 values, got shape {arr.shape!r}")
        return arr.astype(np.float32)

    def _pose_to_dict(self, pose) -> dict[str, list[float]]:
        position = np.round(self._to_numpy(pose.position).reshape(-1, 3)[0], 9)
        quaternion = np.round(self._to_numpy(pose.quaternion).reshape(-1, 4)[0], 9)
        return {
            "position": position.tolist(),
            "quaternion": quaternion.tolist(),
        }

    def _normalize_q(self, q: Sequence[float]) -> np.ndarray:
        q_np = self._as_float_array(q, expected=7, name="q")
        return q_np.astype(np.float32)

    def _first_success_solution(self, solution, success) -> Optional[np.ndarray]:
        if solution is None or success is None:
            return None
        success_idx = self.mods["torch"].nonzero(success.reshape(-1), as_tuple=False).view(-1)
        if success_idx.numel() == 0:
            return None
        flat_solution = solution.reshape(-1, solution.shape[-1])
        return self._to_numpy(flat_solution[int(success_idx[0])]).reshape(-1)[:7]

    def _nearest_success_solution(
        self,
        solution,
        success,
        reference_q: Sequence[float],
    ) -> Optional[np.ndarray]:
        if solution is None or success is None:
            return None
        success_idx = self.mods["torch"].nonzero(success.reshape(-1), as_tuple=False).view(-1)
        if success_idx.numel() == 0:
            return None
        flat_solution = solution.reshape(-1, solution.shape[-1])
        ref = self._as_float_array(reference_q, expected=7, name="reference_q")
        best_q = None
        best_dist = None
        for idx in success_idx.tolist():
            q = self._to_numpy(flat_solution[int(idx)]).reshape(-1)[:7].astype(np.float32)
            dist = float(np.linalg.norm(q - ref))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_q = q
        return best_q

    def _extract_batch_paths(self, result: Any, expected_count: int) -> list[Any]:
        expected_count = int(max(expected_count, 0))
        paths: list[Any] = [None] * expected_count
        interpolated_plan = getattr(result, "interpolated_plan", None)
        if interpolated_plan is None or expected_count == 0:
            return paths
        try:
            raw_paths = result.get_paths()
            for idx, path in enumerate(list(raw_paths or [])[:expected_count]):
                paths[idx] = path
            return paths
        except Exception:
            pass

        last_tsteps = getattr(result, "path_buffer_last_tstep", None)
        for idx in range(expected_count):
            try:
                path = interpolated_plan[idx]
            except Exception:
                continue
            position = getattr(path, "position", None)
            if position is None:
                continue
            if len(getattr(position, "shape", ())) < 2:
                continue
            try:
                end_idx = None
                if last_tsteps is not None and idx < len(last_tsteps):
                    end_idx = last_tsteps[idx]
                if end_idx is not None:
                    path = path.trim_trajectory(0, end_idx)
            except Exception:
                pass
            paths[idx] = path
        return paths

    def _robot_cfg_has_collision_model(self, robot_cfg_dict: Mapping[str, Any]) -> bool:
        if "robot_cfg" in robot_cfg_dict:
            robot_cfg_dict = robot_cfg_dict["robot_cfg"]
        kinematics = robot_cfg_dict.get("kinematics", {})
        collision_spheres = kinematics.get("collision_spheres")
        collision_link_names = kinematics.get("collision_link_names")
        return collision_spheres is not None and collision_link_names not in (None, [])

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        return np.asarray(x)

    @staticmethod
    def _as_float_array(values: Any, *, expected: int, name: str) -> np.ndarray:
        if values is None:
            raise ValueError(f"{name} is required")
        if hasattr(values, "detach"):
            values = values.detach()
        if hasattr(values, "cpu"):
            values = values.cpu()
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.size < expected:
            raise ValueError(f"{name} must have at least {expected} values, got {array.size}")
        return array[:expected]


__all__ = [
    "ARM_JOINT_NAMES",
    "GRIPPER_JOINT_NAMES",
    "RM75CuRoboPlannerConfig",
    "CuRoboPlanResult",
    "RM75CuRoboPlanner",
]
