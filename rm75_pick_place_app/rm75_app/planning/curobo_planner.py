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

from rm75_app.paths import DEFAULT_RM75_URDF, RUNTIME_DIR


DEFAULT_CUROBO_ROOT = Path("/home/zhangzhao/PycharmProjects/curobo")
DEFAULT_TORCH_EXTENSIONS_DIR = RUNTIME_DIR / "curobo_torch_extensions"
DEFAULT_CUDA_ARCH_LIST = "12.0"

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

CUROBO_EXTENSION_MODULES = (
    "kinematics_fused_cu",
    "geom_cu",
    "lbfgs_step_cu",
    "line_search_cu",
    "tensor_step_cu",
)

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
    cuda_arch_list: Optional[str] = DEFAULT_CUDA_ARCH_LIST
    gripper_lock: float = 0.6
    num_ik_seeds: int = 64
    num_trajopt_seeds: int = 1
    num_graph_seeds: int = 1
    interpolation_dt: float = 0.02
    position_threshold: float = 0.005
    rotation_threshold: float = 0.05
    use_cuda_graph: bool = False
    self_collision_check: bool = True
    self_collision_opt: bool = True
    collision_activation_distance: float = 0.02
    profile_motiongen_internal_ik: bool = False


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
        self._mesh_world_initialized = False
        self._disabled_collision_links: set[str] = set()
        self._cuda_graph_batch_ik_solvers: dict[tuple[int, int], Any] = {}
        self._cuda_graph_batch_ik_disabled_reason: Optional[str] = None
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
        disabled_before = set(self._disabled_collision_links)
        if enabled:
            self._disabled_collision_links.difference_update(link_names)
        else:
            self._disabled_collision_links.update(link_names)
        disabled_changed = disabled_before != set(self._disabled_collision_links)
        self._set_solver_world_collision_for_links(self.ik_solver, link_names, enabled=enabled)
        if disabled_changed:
            for solver in list(self._cuda_graph_batch_ik_solvers.values()):
                self._set_solver_world_collision_for_links(solver, link_names, enabled=enabled)
        return link_names

    @staticmethod
    def _set_solver_world_collision_for_links(solver, link_names: Sequence[str], *, enabled: bool) -> None:
        try:
            ik_kinematics = getattr(solver, "kinematics", None)
            ik_kin_cfg = getattr(ik_kinematics, "kinematics_config", None)
            if ik_kin_cfg is not None:
                for link_name in list(link_names or []):
                    if enabled:
                        ik_kin_cfg.enable_link_spheres(str(link_name))
                    else:
                        ik_kin_cfg.disable_link_spheres(str(link_name))
        except Exception:
            pass

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

    def _compute_world_link_spheres(self, q: Sequence[float]) -> np.ndarray:
        q_np = self._normalize_q(q)
        joint_state = self._make_start_state(q_np)
        kin_state = self.motion_gen.compute_kinematics(joint_state)
        link_spheres = getattr(kin_state, "link_spheres_tensor", None)
        if link_spheres is None and hasattr(kin_state, "get_link_spheres"):
            link_spheres = kin_state.get_link_spheres()
        if link_spheres is None:
            q_tensor = getattr(joint_state, "position", None)
            if q_tensor is None:
                raise RuntimeError("joint state does not expose position tensor")
            kin_state = self.motion_gen.kinematics.get_state(q_tensor)
            link_spheres = getattr(kin_state, "link_spheres_tensor", None)
            if link_spheres is None and hasattr(kin_state, "get_link_spheres"):
                link_spheres = kin_state.get_link_spheres()
        if link_spheres is None:
            raise RuntimeError("kinematics state does not expose link spheres")
        if hasattr(link_spheres, "detach"):
            link_spheres = link_spheres.detach().cpu().numpy()
        link_spheres = np.asarray(link_spheres, dtype=np.float32)
        if link_spheres.ndim == 3:
            link_spheres = link_spheres[0]
        return link_spheres.reshape(-1, 4)

    def _collision_sphere_link_names(self) -> list[str]:
        kin_cfg = self.motion_gen.robot_cfg.kinematics.kinematics_config
        link_idx_map = getattr(kin_cfg, "link_name_to_idx_map", {}) or {}
        idx_to_name = {int(v): str(k) for k, v in link_idx_map.items()}
        sphere_link_idx = getattr(kin_cfg, "link_sphere_idx_map", None)
        if sphere_link_idx is None:
            return []
        if hasattr(sphere_link_idx, "detach"):
            sphere_link_idx = sphere_link_idx.detach().cpu().numpy()
        sphere_link_idx = np.asarray(sphere_link_idx).reshape(-1)
        return [idx_to_name.get(int(idx), f"link_idx_{int(idx)}") for idx in sphere_link_idx.tolist()]

    def _self_collision_ignore_pairs(self) -> set[tuple[str, str]]:
        robot_cfg_dict: Mapping[str, Any] = self.robot_cfg_dict
        if "robot_cfg" in robot_cfg_dict:
            robot_cfg_dict = robot_cfg_dict["robot_cfg"]
        ignore_cfg = (robot_cfg_dict.get("kinematics", {}) or {}).get("self_collision_ignore") or {}
        ignore_pairs: set[tuple[str, str]] = set()
        for link_a, ignored_links in dict(ignore_cfg).items():
            a = str(link_a)
            for link_b in list(ignored_links or []):
                b = str(link_b)
                if not a or not b:
                    continue
                ignore_pairs.add(tuple(sorted((a, b))))
        return ignore_pairs

    def _self_collision_buffers(self) -> dict[str, float]:
        robot_cfg_dict: Mapping[str, Any] = self.robot_cfg_dict
        if "robot_cfg" in robot_cfg_dict:
            robot_cfg_dict = robot_cfg_dict["robot_cfg"]
        buffer_cfg = (robot_cfg_dict.get("kinematics", {}) or {}).get("self_collision_buffer") or {}
        return {str(k): float(v) for k, v in dict(buffer_cfg).items()}

    def diagnose_start_state_self_collision(self, q: Sequence[float], *, top_k: int = 10) -> dict[str, Any]:
        q_np = self._normalize_q(q)
        valid, status = self.check_start_state(q_np)
        diagnosis: dict[str, Any] = {
            "valid": bool(valid),
            "status": status,
            "pairs": [],
            "link_pairs": [],
        }
        if valid or status != "MotionGenStatus.INVALID_START_STATE_SELF_COLLISION":
            return diagnosis
        try:
            spheres = self._compute_world_link_spheres(q_np)
            sphere_link_names = self._collision_sphere_link_names()
        except Exception as exc:
            diagnosis["error"] = str(exc)
            return diagnosis
        if len(sphere_link_names) != spheres.shape[0]:
            diagnosis["error"] = (
                f"sphere/link count mismatch: n_spheres={spheres.shape[0]} "
                f"n_link_names={len(sphere_link_names)}"
            )
            return diagnosis

        ignore_pairs = self._self_collision_ignore_pairs()
        buffer_by_link = self._self_collision_buffers()
        pair_records: list[dict[str, Any]] = []
        link_pair_best: dict[tuple[str, str], dict[str, Any]] = {}
        for i in range(spheres.shape[0]):
            c_i = spheres[i, :3]
            r_i = float(spheres[i, 3])
            link_i = sphere_link_names[i]
            for j in range(i + 1, spheres.shape[0]):
                link_j = sphere_link_names[j]
                if link_i == link_j:
                    continue
                pair_key = tuple(sorted((link_i, link_j)))
                if pair_key in ignore_pairs:
                    continue
                c_j = spheres[j, :3]
                r_j = float(spheres[j, 3])
                center_dist = float(np.linalg.norm(c_i - c_j))
                threshold = (
                    r_i
                    + r_j
                    + float(buffer_by_link.get(link_i, 0.0))
                    + float(buffer_by_link.get(link_j, 0.0))
                )
                overlap = threshold - center_dist
                if overlap <= 0.0:
                    continue
                record = {
                    "sphere_i": int(i),
                    "sphere_j": int(j),
                    "link_i": link_i,
                    "link_j": link_j,
                    "overlap": float(overlap),
                    "center_distance": float(center_dist),
                    "threshold": float(threshold),
                }
                pair_records.append(record)
                current_best = link_pair_best.get(pair_key)
                if current_best is None or float(record["overlap"]) > float(current_best["overlap"]):
                    link_pair_best[pair_key] = {
                        "link_a": pair_key[0],
                        "link_b": pair_key[1],
                        "overlap": float(record["overlap"]),
                        "count": 1,
                    }
                else:
                    current_best["count"] = int(current_best.get("count", 1)) + 1

        pair_records.sort(key=lambda x: (-float(x["overlap"]), str(x["link_i"]), str(x["link_j"])))
        link_pair_records = sorted(
            link_pair_best.values(),
            key=lambda x: (-float(x["overlap"]), str(x["link_a"]), str(x["link_b"])),
        )
        use_top_k = max(int(top_k), 0)
        diagnosis["pairs"] = pair_records if use_top_k <= 0 else pair_records[:use_top_k]
        diagnosis["link_pairs"] = link_pair_records if use_top_k <= 0 else link_pair_records[:use_top_k]
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

    def solve_batch_start_goal_ik(
        self,
        start_qs: Sequence[Sequence[float]],
        goal_poses: Sequence[Any],
        *,
        num_seeds: Optional[int] = None,
        use_cuda_graph_batch: bool = False,
        cuda_graph_batch_size: Optional[int] = None,
        cuda_graph_fixed_batch_size: Optional[int] = None,
    ) -> list[CuRoboPlanResult]:
        start_qs = list(start_qs or [])
        goal_poses = list(goal_poses or [])
        if len(start_qs) == 0 or len(goal_poses) == 0:
            return []
        if len(start_qs) != len(goal_poses):
            raise ValueError(
                f"start_qs and goal_poses must have the same length, got {len(start_qs)} and {len(goal_poses)}"
            )
        if (
            use_cuda_graph_batch
            and len(goal_poses) > 1
            and not self.attached_object_active
            and self._cuda_graph_batch_ik_disabled_reason is None
        ):
            try:
                return self._solve_batch_start_goal_ik_cuda_graph(
                    start_qs,
                    goal_poses,
                    num_seeds=num_seeds,
                    max_batch_size=cuda_graph_batch_size,
                    fixed_batch_size=cuda_graph_fixed_batch_size,
                )
            except Exception as exc:
                if "exceeds CUDA graph max batch" not in str(exc):
                    self._cuda_graph_batch_ik_disabled_reason = str(exc)
                print(f"[curobo] CUDA graph batch IK failed; falling back to eager batch IK: {exc}")
        if len(goal_poses) == 1:
            return [self.solve_ik(start_qs[0], goal_poses[0], num_seeds=num_seeds)]

        start_state = self._make_multi_start_state(start_qs)
        goal = self._make_batch_pose(goal_poses)
        use_num_seeds = self.config.num_ik_seeds if num_seeds is None else int(num_seeds)

        self.ik_solver.reset_seed()
        result = self.ik_solver.solve_batch(
            goal,
            retract_config=start_state.position.clone(),
            seed_config=start_state.position.unsqueeze(1).clone(),
            return_seeds=1,
            num_seeds=use_num_seeds,
            use_nn_seed=False,
        )

        success_arr = self._to_numpy(result.success).reshape(-1).astype(bool)
        flat_solution = None if result.solution is None else self._to_numpy(result.solution).reshape(-1, result.solution.shape[-1])
        position_error = None if getattr(result, "position_error", None) is None else self._to_numpy(result.position_error).reshape(-1)
        rotation_error = None if getattr(result, "rotation_error", None) is None else self._to_numpy(result.rotation_error).reshape(-1)
        outputs: list[CuRoboPlanResult] = []
        for idx, success in enumerate(success_arr.tolist()):
            goal_joint = None
            if bool(success) and flat_solution is not None and idx < flat_solution.shape[0]:
                goal_joint = np.asarray(flat_solution[idx], dtype=np.float32).reshape(-1)[:7]
            outputs.append(
                CuRoboPlanResult(
                    success=bool(success),
                    status="Success" if bool(success) else "IK_FAIL",
                    goal_joint=goal_joint,
                    solve_time=float(result.solve_time),
                    ik_time=float(result.solve_time),
                    raw_result=result,
                    debug={
                        "batch_index": int(idx),
                        "position_error": None if position_error is None or idx >= position_error.shape[0] else float(position_error[idx]),
                        "rotation_error": None if rotation_error is None or idx >= rotation_error.shape[0] else float(rotation_error[idx]),
                        "ik_success_count": int(self.mods["torch"].count_nonzero(result.success).item()),
                    },
                )
            )
        return outputs

    def _solve_batch_start_goal_ik_cuda_graph(
        self,
        start_qs: Sequence[Sequence[float]],
        goal_poses: Sequence[Any],
        *,
        num_seeds: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        fixed_batch_size: Optional[int] = None,
    ) -> list[CuRoboPlanResult]:
        if self._cuda_graph_batch_ik_disabled_reason:
            raise RuntimeError(self._cuda_graph_batch_ik_disabled_reason)
        start_qs = list(start_qs or [])
        goal_poses = list(goal_poses or [])
        requested_batch = len(goal_poses)
        if requested_batch <= 1:
            return self.solve_batch_start_goal_ik(start_qs, goal_poses, num_seeds=num_seeds)
        use_num_seeds = self.config.num_ik_seeds if num_seeds is None else int(num_seeds)
        fixed_batch = self._select_cuda_graph_ik_batch_size(
            requested_batch,
            max_batch_size=max_batch_size,
            fixed_batch_size=fixed_batch_size,
        )
        if fixed_batch < requested_batch:
            if fixed_batch <= 1:
                raise ValueError(f"requested IK batch {requested_batch} exceeds CUDA graph max batch {fixed_batch}")
            outputs: list[CuRoboPlanResult] = []
            chunk_count = int((requested_batch + fixed_batch - 1) // fixed_batch)
            for chunk_index, start_idx in enumerate(range(0, requested_batch, fixed_batch)):
                chunk_start_qs = start_qs[start_idx : start_idx + fixed_batch]
                chunk_goal_poses = goal_poses[start_idx : start_idx + fixed_batch]
                outputs.extend(
                    self._solve_batch_start_goal_ik_cuda_graph_once(
                        chunk_start_qs,
                        chunk_goal_poses,
                        fixed_batch=fixed_batch,
                        num_seeds=use_num_seeds,
                        total_requested_batch=requested_batch,
                        chunk_index=chunk_index,
                        chunk_count=chunk_count,
                    )
                )
            return outputs
        return self._solve_batch_start_goal_ik_cuda_graph_once(
            start_qs,
            goal_poses,
            fixed_batch=fixed_batch,
            num_seeds=use_num_seeds,
            total_requested_batch=requested_batch,
            chunk_index=0,
            chunk_count=1,
        )

    def _solve_batch_start_goal_ik_cuda_graph_once(
        self,
        start_qs: Sequence[Sequence[float]],
        goal_poses: Sequence[Any],
        *,
        fixed_batch: int,
        num_seeds: int,
        total_requested_batch: int,
        chunk_index: int,
        chunk_count: int,
    ) -> list[CuRoboPlanResult]:
        start_qs = list(start_qs or [])
        goal_poses = list(goal_poses or [])
        requested_batch = len(goal_poses)
        if requested_batch <= 0:
            return []
        fixed_batch = int(fixed_batch)
        use_num_seeds = int(num_seeds)
        padded_start_qs = list(start_qs)
        padded_goal_poses = list(goal_poses)
        while len(padded_goal_poses) < fixed_batch:
            padded_start_qs.append(padded_start_qs[-1])
            padded_goal_poses.append(padded_goal_poses[-1])

        solver = self._get_cuda_graph_batch_ik_solver(fixed_batch, use_num_seeds)
        start_state = self._make_multi_start_state(padded_start_qs)
        goal = self._make_batch_pose(padded_goal_poses)
        solver.reset_seed()
        result = solver.solve_batch(
            goal,
            retract_config=start_state.position.clone(),
            seed_config=start_state.position.unsqueeze(1).clone(),
            return_seeds=1,
            num_seeds=use_num_seeds,
            use_nn_seed=False,
        )

        success_arr = self._to_numpy(result.success).reshape(-1).astype(bool)[:requested_batch]
        flat_solution = None if result.solution is None else self._to_numpy(result.solution).reshape(-1, result.solution.shape[-1])
        position_error = None if getattr(result, "position_error", None) is None else self._to_numpy(result.position_error).reshape(-1)
        rotation_error = None if getattr(result, "rotation_error", None) is None else self._to_numpy(result.rotation_error).reshape(-1)
        outputs: list[CuRoboPlanResult] = []
        for idx, success in enumerate(success_arr.tolist()):
            goal_joint = None
            if bool(success) and flat_solution is not None and idx < flat_solution.shape[0]:
                goal_joint = np.asarray(flat_solution[idx], dtype=np.float32).reshape(-1)[:7]
            outputs.append(
                CuRoboPlanResult(
                    success=bool(success),
                    status="Success" if bool(success) else "IK_FAIL",
                    goal_joint=goal_joint,
                    solve_time=float(result.solve_time),
                    ik_time=float(result.solve_time),
                    raw_result=result,
                    debug={
                        "batch_index": int(idx),
                        "position_error": None if position_error is None or idx >= position_error.shape[0] else float(position_error[idx]),
                        "rotation_error": None if rotation_error is None or idx >= rotation_error.shape[0] else float(rotation_error[idx]),
                        "ik_success_count": int(np.count_nonzero(success_arr)),
                        "cuda_graph_batch": True,
                        "requested_batch_size": int(requested_batch),
                        "fixed_batch_size": int(fixed_batch),
                        "total_requested_batch_size": int(total_requested_batch),
                        "cuda_graph_chunk_index": int(chunk_index),
                        "cuda_graph_chunk_count": int(chunk_count),
                    },
                )
            )
        return outputs

    @staticmethod
    def _select_cuda_graph_ik_batch_size(
        requested_batch: int,
        *,
        max_batch_size: Optional[int] = None,
        fixed_batch_size: Optional[int] = None,
    ) -> int:
        requested = max(int(requested_batch), 1)
        fixed_batch = int(fixed_batch_size or 0)
        if fixed_batch > 0:
            return fixed_batch
        max_batch = 128 if max_batch_size is None or int(max_batch_size) <= 0 else int(max_batch_size)
        buckets = [2, 4, 8, 16, 32, 64, 128, 256]
        buckets = [x for x in buckets if x <= max_batch]
        if not buckets or requested > buckets[-1]:
            return max_batch
        for bucket in buckets:
            if requested <= bucket:
                return bucket
        return buckets[-1]

    def _get_cuda_graph_batch_ik_solver(self, batch_size: int, num_seeds: int):
        key = (int(batch_size), int(num_seeds))
        solver = self._cuda_graph_batch_ik_solvers.get(key)
        if solver is not None:
            return solver
        solver = self._build_ik_solver(
            use_cuda_graph=True,
            num_seeds=int(num_seeds),
            collision_cache=self._cuda_graph_ik_collision_cache(),
        )
        self._apply_disabled_collision_links_to_solver(solver)
        self._cuda_graph_batch_ik_solvers[key] = solver
        print(f"[curobo] created CUDA graph batch IK solver: batch={int(batch_size)}, seeds={int(num_seeds)}")
        return solver

    def _invalidate_cuda_graph_batch_ik_solvers(self) -> None:
        self._cuda_graph_batch_ik_solvers.clear()

    def _update_cuda_graph_batch_ik_world(self, world) -> None:
        if not self._cuda_graph_batch_ik_solvers:
            return
        try:
            for solver in list(self._cuda_graph_batch_ik_solvers.values()):
                solver.update_world(world)
        except Exception as exc:
            self._cuda_graph_batch_ik_solvers.clear()
            self._cuda_graph_batch_ik_disabled_reason = str(exc)
            print(f"[curobo] disabled CUDA graph batch IK after world update failed: {exc}")

    def _cuda_graph_ik_collision_cache(self) -> dict[str, int]:
        world = self._world
        cuboid_count = len(list(getattr(world, "cuboid", []) or []))
        mesh_count = len(list(getattr(world, "mesh", []) or []))
        return {
            "obb": max(32, cuboid_count + 8),
            "mesh": max(8, mesh_count + 4),
        }

    def _apply_disabled_collision_links_to_solver(self, solver) -> None:
        disabled = [str(x) for x in sorted(getattr(self, "_disabled_collision_links", set()) or set()) if str(x)]
        if not disabled:
            return
        self._set_solver_world_collision_for_links(solver, disabled, enabled=False)

    def estimate_batch_start_goal_ik_errors(
        self,
        start_qs: Sequence[Sequence[float]],
        goal_poses: Sequence[Any],
        *,
        num_seeds: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Return per-candidate IK error estimates for diagnostics."""
        start_qs = list(start_qs or [])
        goal_poses = list(goal_poses or [])
        if len(start_qs) == 0 or len(goal_poses) == 0:
            return []
        if len(start_qs) != len(goal_poses):
            raise ValueError(
                f"start_qs and goal_poses must have the same length, got {len(start_qs)} and {len(goal_poses)}"
            )
        if len(goal_poses) == 1:
            single = self.solve_ik(start_qs[0], goal_poses[0], num_seeds=num_seeds)
            return [
                {
                    "success": bool(single.success),
                    "status": str(single.status),
                    "position_error": float(single.debug.get("position_error", np.nan)),
                    "rotation_error": float(single.debug.get("rotation_error", np.nan)),
                }
            ]

        start_state = self._make_multi_start_state(start_qs)
        goal = self._make_batch_pose(goal_poses)
        use_num_seeds = self.config.num_ik_seeds if num_seeds is None else int(num_seeds)
        self.ik_solver.reset_seed()
        result = self.ik_solver.solve_batch(
            goal,
            retract_config=start_state.position.clone(),
            seed_config=start_state.position.unsqueeze(1).clone(),
            return_seeds=1,
            num_seeds=use_num_seeds,
            use_nn_seed=False,
        )
        success_arr = self._to_numpy(result.success).reshape(-1).astype(bool)
        pos_err = None if getattr(result, "position_error", None) is None else self._to_numpy(result.position_error).reshape(-1)
        rot_err = None if getattr(result, "rotation_error", None) is None else self._to_numpy(result.rotation_error).reshape(-1)
        outputs: list[dict[str, Any]] = []
        for idx, success in enumerate(success_arr.tolist()):
            outputs.append(
                {
                    "success": bool(success),
                    "status": "Success" if bool(success) else "IK_FAIL",
                    "position_error": float(np.nan if pos_err is None or idx >= pos_err.shape[0] else pos_err[idx]),
                    "rotation_error": float(np.nan if rot_err is None or idx >= rot_err.shape[0] else rot_err[idx]),
                }
            )
        return outputs

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

        motion_ik_successes = None
        if bool(getattr(self.config, "profile_motiongen_internal_ik", False)):
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
        debug: dict[str, Any] = {}
        if motion_ik_successes is not None:
            debug["motion_gen_internal_ik_successes"] = motion_ik_successes
        if not success:
            return CuRoboPlanResult(
                success=False,
                status=status,
                solve_time=float(result.solve_time),
                ik_time=float(result.ik_time),
                trajopt_time=float(result.trajopt_time),
                raw_result=result,
                debug=debug,
            )

        traj = result.get_interpolated_plan()
        q_path = self._to_numpy(traj.position).astype(np.float32)
        debug["interpolation_dt"] = float(result.interpolation_dt)
        return CuRoboPlanResult(
            success=True,
            status=status,
            goal_joint=np.asarray(q_path[-1], dtype=np.float32),
            joint_path=q_path,
            solve_time=float(result.solve_time),
            ik_time=float(result.ik_time),
            trajopt_time=float(result.trajopt_time),
            raw_result=result,
            debug=debug,
        )

    def plan_to_joint_state(
        self,
        start_q: Sequence[float],
        goal_q: Sequence[float],
        *,
        enable_graph: bool = False,
        max_attempts: int = 2,
        timeout: float = 5.0,
        num_trajopt_seeds: Optional[int] = None,
        num_graph_seeds: Optional[int] = None,
    ) -> CuRoboPlanResult:
        start_q_np = self._normalize_q(start_q)
        goal_q_np = self._normalize_q(goal_q)
        start_state = self._make_start_state(start_q_np)
        goal_state = self._make_start_state(goal_q_np)

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
            num_graph_seeds=use_num_graph_seeds,
            num_trajopt_seeds=use_num_trajopt_seeds,
        )

        self.motion_gen.reset_seed()
        result = self.motion_gen.plan_single_js(start_state, goal_state, plan_config)
        success = bool(result.success.reshape(-1)[0].item())
        status = None if result.status is None else str(result.status)
        if not success:
            return CuRoboPlanResult(
                success=False,
                status=status,
                solve_time=float(result.solve_time),
                ik_time=float(getattr(result, "ik_time", 0.0) or 0.0),
                trajopt_time=float(result.trajopt_time),
                raw_result=result,
            )

        traj = result.get_interpolated_plan()
        q_path = self._to_numpy(traj.position).astype(np.float32)
        return CuRoboPlanResult(
            success=True,
            status=status,
            goal_joint=np.asarray(q_path[-1], dtype=np.float32),
            joint_path=q_path,
            solve_time=float(result.solve_time),
            ik_time=float(getattr(result, "ik_time", 0.0) or 0.0),
            trajopt_time=float(result.trajopt_time),
            raw_result=result,
            debug={
                "interpolation_dt": float(result.interpolation_dt),
                "target_joint_error": float(np.max(np.abs(np.asarray(q_path[-1], dtype=np.float32) - goal_q_np))),
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
        if bool(enable_graph):
            print(
                "[curobo] plan_batch_to_poses requested graph planning; "
                "falling back to per-goal planning because cuRobo batch graph supports only one graph seed"
            )
            return [
                self.plan_to_pose(
                    start_q,
                    goal_pose,
                    enable_graph=enable_graph,
                    max_attempts=max_attempts,
                    timeout=timeout,
                    num_ik_seeds=num_ik_seeds,
                    num_trajopt_seeds=num_trajopt_seeds,
                    num_graph_seeds=num_graph_seeds,
                )
                for goal_pose in goal_poses
            ]

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
        try:
            result = self.motion_gen.plan_batch(start_state, goal, plan_config)
        except Exception as exc:
            print(
                "[curobo] plan_batch_to_poses failed inside motion_gen.plan_batch; "
                f"falling back to per-goal planning ({type(exc).__name__}: {exc})"
            )
            outputs: list[CuRoboPlanResult] = []
            for goal_pose in goal_poses:
                outputs.append(
                    self.plan_to_pose(
                        start_q,
                        goal_pose,
                        enable_graph=enable_graph,
                        max_attempts=max_attempts,
                        timeout=timeout,
                        num_ik_seeds=num_ik_seeds,
                        num_trajopt_seeds=num_trajopt_seeds,
                        num_graph_seeds=num_graph_seeds,
                    )
                )
            return outputs
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
        if bool(enable_graph):
            print(
                "[curobo] plan_batch_start_goal_pairs requested graph planning; "
                "falling back to per-pair planning because cuRobo batch graph supports only one graph seed"
            )
            return [
                self.plan_to_pose(
                    start_q,
                    goal_pose,
                    enable_graph=enable_graph,
                    max_attempts=max_attempts,
                    timeout=timeout,
                    num_ik_seeds=num_ik_seeds,
                    num_trajopt_seeds=num_trajopt_seeds,
                    num_graph_seeds=num_graph_seeds,
                )
                for start_q, goal_pose in zip(start_qs, goal_poses)
            ]

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
        try:
            result = self.motion_gen.plan_batch(start_state, goal, plan_config)
        except Exception as exc:
            print(
                "[curobo] plan_batch_start_goal_pairs failed inside motion_gen.plan_batch; "
                f"falling back to per-pair planning ({type(exc).__name__}: {exc})"
            )
            outputs: list[CuRoboPlanResult] = []
            for start_q, goal_pose in zip(start_qs, goal_poses):
                outputs.append(
                    self.plan_to_pose(
                        start_q,
                        goal_pose,
                        enable_graph=enable_graph,
                        max_attempts=max_attempts,
                        timeout=timeout,
                        num_ik_seeds=num_ik_seeds,
                        num_trajopt_seeds=num_trajopt_seeds,
                        num_graph_seeds=num_graph_seeds,
                    )
                )
            return outputs
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
        needs_mesh_world = len(list(meshes or [])) > 0
        if needs_mesh_world and not self._mesh_world_initialized:
            self._world = world_cfg
            self._invalidate_cuda_graph_batch_ik_solvers()
            self.ik_solver = self._build_ik_solver()
            self.motion_gen = self._build_motion_gen()
            self._mesh_world_initialized = True
            return
        self.motion_gen.update_world(world_cfg)
        self.ik_solver.update_world(world_cfg)
        self._world = world_cfg
        self._update_cuda_graph_batch_ik_world(world_cfg)

    def clear_world(self) -> None:
        if not self.collision_enabled:
            return
        self.motion_gen.update_world(self._empty_world)
        self.ik_solver.update_world(self._empty_world)
        self._world = self._empty_world
        self._update_cuda_graph_batch_ik_world(self._empty_world)

    @staticmethod
    def _quat_wxyz_to_rotmat(quaternion: Sequence[float]) -> np.ndarray:
        q = np.asarray(quaternion, dtype=np.float32).reshape(4)
        norm = float(np.linalg.norm(q))
        if norm <= 1e-8:
            return np.eye(3, dtype=np.float32)
        w, x, y, z = (q / norm).tolist()
        return np.asarray(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )

    def _build_linear_attached_sphere_tensor(
        self,
        *,
        q: Sequence[float],
        box_dims: Sequence[float],
        object_pose_world: Any,
        link_name: str,
        sphere_count: int,
        world_z_offset: float = 0.0,
        radius_scale: float = 0.48,
        end_cover_margin_scale: float = 0.14,
        length_scale: float = 1.0,
    ):
        torch = self.mods["torch"]
        dims = np.asarray(box_dims, dtype=np.float32).reshape(3)
        dims = dims.copy()
        axis_idx = int(np.argmax(dims))
        if float(length_scale) > 0.0:
            dims[axis_idx] *= float(length_scale)
        short_axes = [i for i in range(3) if i != axis_idx]
        long_dim = float(dims[axis_idx])
        short_dim = float(max(dims[short_axes[0]], dims[short_axes[1]]))
        sphere_radius = max(0.0025, float(radius_scale) * short_dim)

        end_cover_margin = max(0.0, float(end_cover_margin_scale) * short_dim)
        center_limit = max(0.0, 0.5 * long_dim + end_cover_margin - sphere_radius)
        if sphere_count <= 1 or center_limit <= 1e-6:
            axis_positions = np.zeros((sphere_count,), dtype=np.float32)
        else:
            axis_positions = np.linspace(-center_limit, center_limit, sphere_count, dtype=np.float32)

        local_centers = np.zeros((sphere_count, 3), dtype=np.float32)
        local_centers[:, axis_idx] = axis_positions

        obj_p, obj_q = self._extract_pose_components(object_pose_world)
        obj_p = obj_p.astype(np.float32).copy()
        obj_p[2] += float(world_z_offset)
        obj_R = self._quat_wxyz_to_rotmat(obj_q)
        world_centers = (obj_R @ local_centers.T).T + obj_p.reshape(1, 3)

        q_np = self._normalize_q(q)
        joint_state = self._make_start_state(q_np)
        kin_state = self.motion_gen.compute_kinematics(joint_state)
        ee_p = self._to_numpy(kin_state.ee_pose.position).reshape(-1, 3)[0].astype(np.float32)
        ee_q = self._to_numpy(kin_state.ee_pose.quaternion).reshape(-1, 4)[0].astype(np.float32)
        ee_R = self._quat_wxyz_to_rotmat(ee_q)
        ee_centers = ((ee_R.T) @ (world_centers - ee_p.reshape(1, 3)).T).T

        max_spheres = int(self.motion_gen.robot_cfg.kinematics.kinematics_config.get_number_of_spheres(link_name))
        sphere_tensor = np.zeros((max_spheres, 4), dtype=np.float32)
        sphere_tensor[:, 3] = -10.0
        fill_count = min(int(sphere_count), max_spheres)
        sphere_tensor[:fill_count, :3] = ee_centers[:fill_count]
        sphere_tensor[:fill_count, 3] = float(sphere_radius)
        covered_length = 2.0 * (float(center_limit) + float(sphere_radius)) if fill_count > 0 else 0.0
        return (
            torch.as_tensor(sphere_tensor, device=self.tensor_args.device, dtype=self.tensor_args.dtype),
            sphere_radius,
            covered_length,
            fill_count,
        )

    def _build_single_attached_sphere_tensor(
        self,
        *,
        q: Sequence[float],
        object_pose_world: Any,
        link_name: str,
        sphere_radius: float,
        world_z_offset: float = 0.0,
    ):
        torch = self.mods["torch"]
        obj_p, _ = self._extract_pose_components(object_pose_world)
        obj_p = obj_p.astype(np.float32).copy()
        obj_p[2] += float(world_z_offset)

        q_np = self._normalize_q(q)
        joint_state = self._make_start_state(q_np)
        kin_state = self.motion_gen.compute_kinematics(joint_state)
        ee_p = self._to_numpy(kin_state.ee_pose.position).reshape(-1, 3)[0].astype(np.float32)
        ee_q = self._to_numpy(kin_state.ee_pose.quaternion).reshape(-1, 4)[0].astype(np.float32)
        ee_R = self._quat_wxyz_to_rotmat(ee_q)
        ee_center = (ee_R.T @ (obj_p - ee_p)).astype(np.float32)

        max_spheres = int(self.motion_gen.robot_cfg.kinematics.kinematics_config.get_number_of_spheres(link_name))
        sphere_tensor = np.zeros((max_spheres, 4), dtype=np.float32)
        sphere_tensor[:, 3] = -10.0
        sphere_tensor[0, :3] = ee_center
        sphere_tensor[0, 3] = float(max(sphere_radius, 0.0025))
        return (
            torch.as_tensor(sphere_tensor, device=self.tensor_args.device, dtype=self.tensor_args.dtype),
            float(max(sphere_radius, 0.0025)),
            1,
        )

    def attach_object_box_to_robot(
        self,
        q: Sequence[float],
        box_dims: Sequence[float],
        *,
        link_name: str = "attached_object",
        surface_sphere_radius: float | None = None,
        object_pose_world: Any | None = None,
        world_z_offset: float = 0.0,
        linear_sphere_count: int | None = None,
        linear_sphere_radius_scale: float = 0.48,
        linear_end_cover_margin_scale: float = 0.14,
        linear_length_scale: float = 1.0,
        single_sphere_radius: float | None = None,
    ) -> bool:
        """Attach a box-shaped object to the robot in cuRobo collision checking.

        Uses cuRobo's Cuboid obstacle type and attach_external_objects_to_robot.
        If surface_sphere_radius is None, it is auto-computed as half of the object's
        smallest dimension so the spheres actually cover the cross-section.
        """
        if not self.collision_enabled:
            return False
        self._invalidate_cuda_graph_batch_ik_solvers()
        torch = self.mods["torch"]
        q_np = self._normalize_q(q)
        joint_state = self._make_start_state(q_np)
        dims = np.asarray(box_dims, dtype=np.float32).reshape(3)
        extent_ratio = float(np.max(dims) / max(1e-6, float(np.min(dims))))
        max_spheres = int(self.motion_gen.robot_cfg.kinematics.kinematics_config.get_number_of_spheres(link_name))
        if surface_sphere_radius is None:
            surface_sphere_radius = float(dims.min()) * 0.5
            surface_sphere_radius = max(surface_sphere_radius, 0.003)
        prefer_linear_long_axis = bool(extent_ratio >= 2.0 and max_spheres >= 4 and object_pose_world is not None)
        try:
            from curobo.geom.types import Cuboid as CuRoboCuboid
        except ImportError:
            print("[curobo] could not import Cuboid from curobo.geom.types; attach skipped")
            return False
        if object_pose_world is None:
            obstacle_pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        else:
            position, quaternion = self._extract_pose_components(object_pose_world)
            obstacle_pose = np.concatenate([position, quaternion]).astype(np.float32).tolist()
        box_obstacle = CuRoboCuboid(
            name="attached_payload",
            pose=obstacle_pose,
            dims=dims.tolist(),
        )
        try:
            if single_sphere_radius is not None and object_pose_world is not None:
                sphere_tensor, manual_radius, filled_spheres = self._build_single_attached_sphere_tensor(
                    q=q_np,
                    object_pose_world=object_pose_world,
                    link_name=str(link_name),
                    sphere_radius=float(single_sphere_radius),
                    world_z_offset=float(world_z_offset),
                )
                self.motion_gen.attach_spheres_to_robot(
                    sphere_radius=0.0,
                    sphere_tensor=sphere_tensor,
                    link_name=str(link_name),
                )
                try:
                    self.ik_solver.attach_object_to_robot(
                        sphere_radius=0.0,
                        sphere_tensor=sphere_tensor.clone(),
                        link_name=str(link_name),
                    )
                except Exception as exc:
                    print(f"[curobo] failed to mirror single attached sphere to ik_solver: {exc}")
                self._attached_object_active = True
                enabled_spheres = self.get_attached_sphere_count(link_name=str(link_name))
                print(
                    f"[curobo] attached single-sphere object {np.round(dims, 4).tolist()} "
                    f"to link={link_name}, enabled_spheres={enabled_spheres}, "
                    f"sphere_radius={float(manual_radius)*1000:.1f}mm, "
                    f"world_object_z_offset={float(world_z_offset)*1000:.1f}mm, "
                    f"object_world_p={np.round(np.asarray(obstacle_pose[:3], dtype=np.float32), 4).tolist()}"
                )
                return True
            if prefer_linear_long_axis:
                use_sphere_count = min(
                    max_spheres,
                    max(1, int(linear_sphere_count)) if linear_sphere_count is not None else min(6, max_spheres),
                )
                sphere_tensor, manual_radius, covered_length, filled_spheres = self._build_linear_attached_sphere_tensor(
                    q=q_np,
                    box_dims=dims,
                    object_pose_world=object_pose_world,
                    link_name=str(link_name),
                    sphere_count=use_sphere_count,
                    world_z_offset=float(world_z_offset),
                    radius_scale=float(linear_sphere_radius_scale),
                    end_cover_margin_scale=float(linear_end_cover_margin_scale),
                    length_scale=float(linear_length_scale),
                )
                self.motion_gen.attach_spheres_to_robot(
                    sphere_radius=0.0,
                    sphere_tensor=sphere_tensor,
                    link_name=str(link_name),
                )
                try:
                    self.ik_solver.attach_object_to_robot(
                        sphere_radius=0.0,
                        sphere_tensor=sphere_tensor.clone(),
                        link_name=str(link_name),
                    )
                except Exception as exc:
                    print(f"[curobo] failed to mirror manual attached spheres to ik_solver: {exc}")
                self._attached_object_active = True
                enabled_spheres = self.get_attached_sphere_count(link_name=str(link_name))
                print(
                    f"[curobo] attached long-axis object {np.round(dims, 4).tolist()} "
                    f"to link={link_name} with {filled_spheres}/{max_spheres} manual spheres, "
                    f"enabled_spheres={enabled_spheres}, "
                    f"manual_sphere_radius={float(manual_radius)*1000:.1f}mm, "
                    f"covered_length={float(covered_length)*1000:.1f}mm/{float(np.max(dims))*1000:.1f}mm, "
                    f"radius_scale={float(linear_sphere_radius_scale):.2f}, "
                    f"length_scale={float(linear_length_scale):.2f}, "
                    f"world_object_z_offset={float(world_z_offset)*1000:.1f}mm, "
                    f"object_world_p={np.round(np.asarray(obstacle_pose[:3], dtype=np.float32), 4).tolist()}"
                )
                return True
            world_objects_pose_offset = None
            if abs(float(world_z_offset)) > 1e-8:
                try:
                    Pose = self.mods["Pose"]
                except Exception:
                    from curobo.types.math import Pose  # type: ignore
                world_objects_pose_offset = Pose.from_list([0.0, 0.0, float(world_z_offset), 1.0, 0.0, 0.0, 0.0])
            ok = self.motion_gen.attach_external_objects_to_robot(
                joint_state=joint_state,
                external_objects=[box_obstacle],
                surface_sphere_radius=float(surface_sphere_radius),
                link_name=str(link_name),
                world_objects_pose_offset=world_objects_pose_offset,
            )
            self._attached_object_active = bool(ok)
            if ok:
                try:
                    sphere_tensor = (
                        self.motion_gen.robot_cfg.kinematics.kinematics_config.get_link_spheres(str(link_name))
                        .clone()
                    )
                    self.ik_solver.attach_object_to_robot(
                        sphere_radius=0.0,
                        sphere_tensor=sphere_tensor,
                        link_name=str(link_name),
                    )
                except Exception as exc:
                    print(f"[curobo] failed to mirror attached object spheres to ik_solver: {exc}")
                n_spheres = self.motion_gen.robot_cfg.kinematics.kinematics_config.get_number_of_spheres(link_name)
                enabled_spheres = self.get_attached_sphere_count(link_name=str(link_name))
                print(
                    f"[curobo] attached object box {np.round(dims, 4).tolist()} "
                    f"to link={link_name} with {n_spheres} spheres, "
                    f"enabled_spheres={enabled_spheres}, "
                    f"surface_sphere_radius={float(surface_sphere_radius)*1000:.1f}mm, "
                    f"world_object_z_offset={float(world_z_offset)*1000:.1f}mm, "
                    f"object_world_p={np.round(np.asarray(obstacle_pose[:3], dtype=np.float32), 4).tolist()}"
                )
            else:
                print("[curobo] attach_external_objects_to_robot returned False")
            return bool(ok)
        except Exception as exc:
            print(f"[curobo] attach_external_objects_to_robot failed: {exc}")
            self._attached_object_active = False
            return False

    def detach_object_from_robot(self, *, link_name: str = "attached_object") -> None:
        """Detach any attached object from the robot in cuRobo collision checking."""
        if not self.collision_enabled:
            return
        try:
            self._invalidate_cuda_graph_batch_ik_solvers()
            self.motion_gen.detach_object_from_robot(link_name=str(link_name))
            try:
                self.ik_solver.detach_object_from_robot(link_name=str(link_name))
            except Exception as exc:
                print(f"[curobo] ik_solver detach_object_from_robot failed: {exc}")
            self._attached_object_active = False
            print(f"[curobo] detached object from link={link_name}")
        except Exception as exc:
            print(f"[curobo] detach_object_from_robot failed: {exc}")

    @property
    def attached_object_active(self) -> bool:
        return bool(getattr(self, "_attached_object_active", False))

    def get_attached_sphere_count(self, *, link_name: str = "attached_object") -> int:
        """Return the number of enabled collision spheres on the attached-object link."""
        try:
            spheres = (
                self.motion_gen.robot_cfg.kinematics.kinematics_config.get_link_spheres(str(link_name))
                .detach()
                .cpu()
                .numpy()
            )
            return int(np.count_nonzero(np.asarray(spheres, dtype=np.float32)[:, 3] > 0.0))
        except Exception:
            return 0

    def get_attached_spheres_world(self, q: Sequence[float], *, link_name: str = "attached_object") -> list[dict]:
        """Return the world-frame positions and radii of attached object collision spheres."""
        if not self.attached_object_active:
            return []
        try:
            q_np = self._normalize_q(q)
            joint_state = self._make_start_state(q_np)
            kin_state = self.motion_gen.compute_kinematics(joint_state)
            link_spheres = getattr(kin_state, "link_spheres_tensor", None)
            if link_spheres is None and hasattr(kin_state, "get_link_spheres"):
                link_spheres = kin_state.get_link_spheres()
            if link_spheres is None:
                q_tensor = getattr(joint_state, "position", None)
                if q_tensor is None:
                    return []
                kin_state = self.motion_gen.kinematics.get_state(q_tensor)
                link_spheres = getattr(kin_state, "link_spheres_tensor", None)
                if link_spheres is None and hasattr(kin_state, "get_link_spheres"):
                    link_spheres = kin_state.get_link_spheres()
            kin_cfg = self.motion_gen.robot_cfg.kinematics.kinematics_config
            link_sphere_index = kin_cfg.get_sphere_index_from_link_name(str(link_name))
            if link_sphere_index.numel() == 0:
                return []
            if link_spheres.ndim == 3:
                link_spheres = link_spheres[0]
            spheres_tensor = link_spheres[link_sphere_index, :].detach().cpu().numpy()
            result = []
            for i in range(spheres_tensor.shape[0]):
                if float(spheres_tensor[i, 3]) <= 0.0:
                    continue
                result.append({
                    "center": spheres_tensor[i, :3].tolist(),
                    "radius": float(spheres_tensor[i, 3]),
                })
            return result
        except Exception as exc:
            print(f"[curobo] get_attached_spheres_world failed: {exc}")
            return []

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

    def _build_ik_solver(
        self,
        *,
        use_cuda_graph: bool = False,
        num_seeds: Optional[int] = None,
        collision_cache: Optional[dict[str, int]] = None,
    ):
        use_num_seeds = int(self.config.num_ik_seeds if num_seeds is None else num_seeds)
        ik_config = self.mods["IKSolverConfig"].load_from_robot_config(
            self.robot_cfg,
            self._world,
            tensor_args=self.tensor_args,
            num_seeds=use_num_seeds,
            position_threshold=float(self.config.position_threshold),
            rotation_threshold=float(self.config.rotation_threshold),
            # IKSolver in this local cuRobo build cannot safely switch between
            # solve_single/solve_batch goal types under cuda graph capture on
            # CUDA < 12 graph reset support. Keep MotionGen graph-enabled, but
            # leave the shared IK solver in eager mode to avoid "changing goal
            # type" crashes. Dedicated fixed-size batch IK solvers can opt into
            # cuda graph capture safely.
            use_cuda_graph=bool(use_cuda_graph),
            self_collision_check=bool(self.config.self_collision_check),
            self_collision_opt=bool(self.config.self_collision_opt),
            collision_cache=collision_cache,
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
            collision_activation_distance=float(self.config.collision_activation_distance),
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
        os.environ["TORCH_EXTENSIONS_DIR"] = str(torch_extensions_dir)
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
        self._register_cached_curobo_extensions(torch_extensions_dir)

    def _register_cached_curobo_extensions(self, torch_extensions_dir: Path) -> None:
        """Let cuRobo's package import find project-local JIT extension .so files.

        cuRobo first tries `from curobo.curobolib import *_cu` and only then
        falls back to `torch.utils.cpp_extension.load()`.  Adding each cached
        extension directory to `curobo.curobolib.__path__` makes that first
        import succeed, so warm caches do not print misleading "JIT compiling"
        messages or take the JIT fallback path.
        """
        try:
            import curobo.curobolib as curobolib
        except Exception:
            return
        package_path = getattr(curobolib, "__path__", None)
        if package_path is None:
            return
        for module_name in CUROBO_EXTENSION_MODULES:
            module_dir = torch_extensions_dir / module_name
            module_so = module_dir / f"{module_name}.so"
            if not module_so.is_file():
                continue
            module_dir_str = str(module_dir)
            if module_dir_str not in package_path:
                package_path.append(module_dir_str)

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

        if isinstance(pose_like, (tuple, list)) and len(pose_like) == 2:
            position = self._as_float_array(pose_like[0], expected=3, name="pose[0]")
            quaternion = self._as_float_array(pose_like[1], expected=4, name="pose[1]")
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
