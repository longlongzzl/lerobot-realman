#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy
from pyroki import collision, costs


def _as_f32(x, shape: int | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def _normalize_quat_wxyz(q) -> np.ndarray:
    q = _as_f32(q, 4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def _quat_angle_deg(q0, q1) -> float:
    q0 = _normalize_quat_wxyz(q0)
    q1 = _normalize_quat_wxyz(q1)
    dot = float(np.clip(abs(float(np.dot(q0, q1))), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _build_seed_path(start_q: np.ndarray, goal_q_hint: np.ndarray | None, knot_count: int) -> np.ndarray:
    start_q = _as_f32(start_q)
    goal_q = start_q if goal_q_hint is None else _as_f32(goal_q_hint)
    alphas = np.linspace(0.0, 1.0, int(max(knot_count, 2)), dtype=np.float32)[:, None]
    return ((1.0 - alphas) * start_q[None, :] + alphas * goal_q[None, :]).astype(np.float32)


def _make_world_box_geom(boxes: list[dict]) -> collision.Box | None:
    if not boxes:
        return None
    extents = np.asarray([box["size"] for box in boxes], dtype=np.float32).reshape(-1, 3)
    positions = np.asarray([box["position"] for box in boxes], dtype=np.float32).reshape(-1, 3)
    quats = np.asarray([_normalize_quat_wxyz(box["quat_wxyz"]) for box in boxes], dtype=np.float32).reshape(-1, 4)
    return collision.Box.from_extent(extent=extents, position=positions, wxyz=quats)


def _forward_tcp_pose(robot: pk.Robot, q: np.ndarray, link_index: int) -> tuple[np.ndarray, np.ndarray]:
    fk = np.asarray(robot.forward_kinematics(_as_f32(q)[None, :])[0, int(link_index)], dtype=np.float32)
    quat = _normalize_quat_wxyz(fk[:4])
    pos = fk[4:7].astype(np.float32)
    return pos, quat


def _pose_goal_errors(robot: pk.Robot, q: np.ndarray, link_index: int, target_pos: np.ndarray, target_quat: np.ndarray) -> tuple[float, float]:
    pos, quat = _forward_tcp_pose(robot, q, link_index)
    pos_err = float(np.linalg.norm(pos - target_pos))
    ori_err_deg = _quat_angle_deg(quat, target_quat)
    return pos_err, ori_err_deg


def solve_request(request: dict) -> dict:
    urdf_path = str(request["planning_urdf_path"])
    tip_link_name = str(request.get("tip_link_name", "gripper_tcp"))
    start_q = _as_f32(request["start_q"])
    goal_mode = str(request["goal_mode"]).strip().lower()
    knot_count = int(max(request.get("knot_count", 8), 2))
    allow_start_in_collision = bool(request.get("allow_start_in_collision", False))
    linear_solver = str(request.get("linear_solver", "dense_cholesky"))
    max_iterations = int(max(request.get("max_iterations", 60), 1))
    verbose = bool(request.get("verbose", False))

    weights = dict(request.get("weights", {}))
    smoothness_weight = float(weights.get("smoothness", 1.0))
    limit_weight = float(weights.get("limit", 0.1))
    init_path_rest_weight = float(weights.get("init_path_rest", 0.02))
    start_anchor_weight = float(weights.get("start_anchor", 50.0))
    goal_anchor_weight = float(weights.get("goal_anchor", 5.0))
    pose_pos_weight = float(weights.get("pose_pos", 40.0))
    pose_ori_weight = float(weights.get("pose_ori", 10.0))
    self_collision_weight = float(weights.get("self_collision", 2.0))
    world_collision_weight = float(weights.get("world_collision", 3.0))
    self_collision_margin = float(weights.get("self_collision_margin", 0.005))
    world_collision_margin = float(weights.get("world_collision_margin", 0.01))

    accept_pos_error_m = float(request.get("accept_pos_error_m", 0.02))
    accept_ori_error_deg = float(request.get("accept_ori_error_deg", 12.0))
    accept_joint_error_rad = float(request.get("accept_joint_error_rad", 0.15))

    urdf = yourdfpy.URDF.load(
        urdf_path,
        build_collision_scene_graph=False,
        load_collision_meshes=True,
        load_meshes=False,
    )
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=start_q)
    robot_coll = collision.RobotCollision.from_urdf(urdf)
    joint_var_cls = robot.joint_var_cls
    link_names = list(robot.links.names)
    if tip_link_name not in link_names:
        raise ValueError(f"Tip link not found in PyRoki robot: {tip_link_name}")
    tip_link_index = int(link_names.index(tip_link_name))

    goal_q = None
    goal_q_hint = request.get("goal_q_hint")
    goal_q_hint = None if goal_q_hint is None else _as_f32(goal_q_hint)
    target_pos = None
    target_quat = None
    target_pose = None
    if goal_mode == "joint":
        goal_q = _as_f32(request["goal_q"])
        if goal_q_hint is None:
            goal_q_hint = goal_q.copy()
    elif goal_mode == "pose":
        target_pos = _as_f32(request["goal_pose"]["position"], 3)
        target_quat = _normalize_quat_wxyz(request["goal_pose"]["quat_wxyz"])
        target_pose = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(target_quat),
            target_pos,
        )
    else:
        raise ValueError(f"Unsupported goal_mode: {goal_mode}")

    seed_path = _build_seed_path(start_q, goal_q_hint, knot_count)
    vars_ = [joint_var_cls(i) for i in range(knot_count)]
    world_boxes = _make_world_box_geom(list(request.get("world_boxes", []) or []))

    cost_terms = []
    collision_start_index = 1 if allow_start_in_collision else 0
    for i, joint_var in enumerate(vars_):
        cost_terms.append(costs.limit_cost(robot=robot, joint_var=joint_var, weight=limit_weight))
        if init_path_rest_weight > 0.0:
            cost_terms.append(costs.rest_cost(joint_var=joint_var, rest_pose=seed_path[i], weight=init_path_rest_weight))
        if i == 0 and start_anchor_weight > 0.0:
            cost_terms.append(costs.rest_cost(joint_var=joint_var, rest_pose=start_q, weight=start_anchor_weight))
        if i >= collision_start_index:
            if self_collision_weight > 0.0:
                cost_terms.append(
                    costs.self_collision_cost(
                        robot=robot,
                        robot_coll=robot_coll,
                        joint_var=joint_var,
                        margin=self_collision_margin,
                        weight=self_collision_weight,
                    )
                )
            if world_boxes is not None and world_collision_weight > 0.0:
                cost_terms.append(
                    costs.world_collision_cost(
                        robot=robot,
                        robot_coll=robot_coll,
                        joint_var=joint_var,
                        world_geom=world_boxes,
                        margin=world_collision_margin,
                        weight=world_collision_weight,
                    )
                )
        if i > 0 and smoothness_weight > 0.0:
            cost_terms.append(
                costs.smoothness_cost(
                    curr_joint_var=joint_var,
                    past_joint_var=vars_[i - 1],
                    weight=smoothness_weight,
                )
            )

    if goal_mode == "joint":
        assert goal_q is not None
        if goal_anchor_weight > 0.0:
            cost_terms.append(costs.rest_cost(joint_var=vars_[-1], rest_pose=goal_q, weight=goal_anchor_weight))
    else:
        assert target_pose is not None
        cost_terms.append(
            costs.pose_cost(
                robot=robot,
                joint_var=vars_[-1],
                target_pose=target_pose,
                target_link_index=np.asarray(tip_link_index, dtype=np.int32),
                pos_weight=pose_pos_weight,
                ori_weight=pose_ori_weight,
            )
        )
        if goal_q_hint is not None and goal_anchor_weight > 0.0:
            cost_terms.append(costs.rest_cost(joint_var=vars_[-1], rest_pose=goal_q_hint, weight=goal_anchor_weight))

    problem = jaxls.LeastSquaresProblem(costs=cost_terms, variables=vars_).analyze(use_onp=True)
    initial_vals = jaxls.VarValues.make([var.with_value(seed_path[i]) for i, var in enumerate(vars_)])
    solution, summary = problem.solve(
        initial_vals,
        linear_solver=linear_solver,
        termination=jaxls.TerminationConfig(max_iterations=max_iterations),
        verbose=verbose,
        return_summary=True,
    )

    path = [np.asarray(solution[var], dtype=np.float32).reshape(-1).tolist() for var in vars_]
    result: dict[str, object] = {
        "ok": True,
        "path": path[1:],
        "iterations": int(summary.iterations),
        "cost_history_last": float(np.asarray(summary.cost_history)[max(int(summary.iterations) - 1, 0)]),
    }

    final_q = np.asarray(path[-1], dtype=np.float32)
    if goal_mode == "joint":
        assert goal_q is not None
        joint_err = float(np.max(np.abs(final_q - goal_q)))
        result["joint_error_rad"] = joint_err
        if joint_err > accept_joint_error_rad:
            result["ok"] = False
            result["reason"] = (
                f"trajopt final joint error {joint_err:.4f} rad exceeds "
                f"accept_joint_error_rad={accept_joint_error_rad:.4f}"
            )
    else:
        assert target_pos is not None and target_quat is not None
        pos_err, ori_err_deg = _pose_goal_errors(robot, final_q, tip_link_index, target_pos, target_quat)
        result["pose_position_error_m"] = pos_err
        result["pose_orientation_error_deg"] = ori_err_deg
        if pos_err > accept_pos_error_m or ori_err_deg > accept_ori_error_deg:
            result["ok"] = False
            result["reason"] = (
                f"trajopt final pose error too large: pos={pos_err:.4f} m "
                f"(limit {accept_pos_error_m:.4f}), ori={ori_err_deg:.2f} deg "
                f"(limit {accept_ori_error_deg:.2f})"
            )
    return result


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: pyroki_trajopt_bridge.py <request.json> <response.json>", file=sys.stderr)
        return 2
    request_path = Path(argv[1])
    response_path = Path(argv[2])
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        response = solve_request(request)
    except Exception as exc:  # pragma: no cover - subprocess bridge
        response = {
            "ok": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    response_path.write_text(json.dumps(response), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
