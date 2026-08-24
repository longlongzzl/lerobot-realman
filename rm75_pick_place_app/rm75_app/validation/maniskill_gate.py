"""GPU physics gate for a complete manipulation plan."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rm75_app.assets.object_specs import get_object_spec, resolve_object_spec_scales
from rm75_app.execution.maniskill_task_bridge import ManiSkillTaskBridge
from rm75_app.execution.trajectory_executor import ManiSkillTrajectoryExecutor
from rm75_app.orchestration.multi_object_executor import AtomExecution, TaskSceneState, load_task_scene
from rm75_app.planning.contracts import JointConfiguration
from rm75_app.runtime.curobo2_sim_replay import load_replay_events
from rm75_app.tasks.manipulation_plan import ManipulationPlan
from rm75_app.validation.contracts import GateResult, GateStatus


class ManiSkillJointAdapter:
    def __init__(self, env: Any):
        self.env = env
        robot = env.unwrapped.agent.robot
        names = [joint.get_name() for joint in robot.get_active_joints()]
        self.arm_indices = [names.index(f"joint_{index}") for index in range(1, 8)]
        self.arm_names = tuple(names[index] for index in self.arm_indices)
        self.action_dim = int(np.prod(env.action_space.shape))
        self.gripper_value = -1.0

    def current_arm_qpos(self) -> np.ndarray:
        qpos = self.env.unwrapped.agent.robot.get_qpos()
        if hasattr(qpos, "detach"):
            qpos = qpos.detach().cpu().numpy()
        return np.asarray(qpos, dtype=np.float64).reshape(-1)[self.arm_indices]

    def set_arm_qpos(self, arm_qpos: np.ndarray) -> None:
        robot = self.env.unwrapped.agent.robot
        qpos = robot.get_qpos()
        is_tensor = hasattr(qpos, "detach")
        raw = qpos.detach().cpu().numpy() if is_tensor else np.asarray(qpos)
        raw = np.asarray(raw, dtype=np.float32).copy()
        flat = raw.reshape(-1)
        flat[self.arm_indices] = np.asarray(arm_qpos, dtype=np.float32).reshape(7)
        robot.set_qpos(raw)

    def joint_configuration(self, scene: TaskSceneState | None = None) -> JointConfiguration:
        del scene
        return JointConfiguration(self.arm_names, self.current_arm_qpos())

    def compose_action(self, arm_target_q: np.ndarray, gripper_value: float) -> np.ndarray:
        self.gripper_value = float(gripper_value)
        action = np.zeros(self.action_dim, dtype=np.float32)
        action[:7] = np.asarray(arm_target_q, dtype=np.float32).reshape(-1)[:7]
        if self.action_dim > 7:
            action[7] = self.gripper_value
        return action

    def step_and_render(self, action: np.ndarray, tag: str = "") -> None:
        del tag
        self.env.step(action)

    def hold_current_and_set_gripper(self, value: float, steps: int = 20) -> None:
        action = self.compose_action(self.current_arm_qpos(), value)
        for _ in range(int(steps)):
            self.env.step(action)

    def hold_action(self) -> np.ndarray:
        return self.compose_action(self.current_arm_qpos(), self.gripper_value)


def write_scene_spec(plan: ManipulationPlan, output_path: str | Path) -> tuple[Path, TaskSceneState]:
    state = load_task_scene(plan.scene_file)
    objects = []
    for item in state.objects.values():
        spec = get_object_spec(item.asset_name)
        if spec is None:
            raw = dict(item.metadata.get("source_scene_entry") or {})
            asset_file = raw.get("sim_asset_file") or raw.get("collision_mesh_path") or raw.get("mesh_file")
            scale = raw.get("sim_asset_scale") or raw.get("mesh_scale") or raw.get("scale") or 1.0
        else:
            _, sim_scale = resolve_object_spec_scales(spec)
            asset_file = spec.sim_asset_file or spec.mesh_file
            scale = sim_scale
        if not asset_file:
            raise KeyError(f"object {item.object_id!r} has no simulation asset")
        scale_array = np.asarray(scale, dtype=float).reshape(-1)
        if scale_array.size == 1:
            scale_array = np.repeat(scale_array, 3)
        objects.append(
            {
                "object_id": item.object_id,
                "asset_name": item.asset_name,
                "asset_file": str(Path(asset_file).expanduser().resolve()),
                "scale": scale_array.tolist(),
                "pose": item.pose.tolist(),
                "fixed": not item.movable,
            }
        )
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"objects": objects}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, state


def run_maniskill_gate(
    plan: ManipulationPlan,
    execution_file: str | Path,
    output_dir: str | Path,
    *,
    render_mode: str = "rgb_array",
    settle_steps: int | None = None,
) -> GateResult:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene_spec, state = write_scene_spec(plan, output / "maniskill_scene.json")
    events = load_replay_events(execution_file)

    import gymnasium as gym
    from mani_skill.utils.wrappers.record import RecordEpisode
    import rm75_app.simulation.maniskill_multi_object_env  # noqa: F401

    env = gym.make(
        "RM75-MultiObjectTask-v1",
        scene_spec_file=str(scene_spec),
        robot_uids="RM75",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode=render_mode,
        max_episode_steps=10000,
    )
    env = RecordEpisode(
        env,
        output_dir=str(output / "video"),
        save_trajectory=False,
        save_video=True,
        source_type="motionplanning",
        source_desc="RM75 three-gate multi-object physics validation",
        video_fps=20,
    )
    try:
        env.reset(seed=0)
        joint_adapter = ManiSkillJointAdapter(env)
        first_trajectory = next((item["trajectory"] for item in events if item.get("type") == "trajectory"), None)
        if first_trajectory is None:
            raise ValueError("Curobo2 package contains no trajectory")
        joint_adapter.set_arm_qpos(first_trajectory.positions[0])
        state.set_joints(joint_adapter.arm_names, joint_adapter.current_arm_qpos())
        bridge = ManiSkillTaskBridge(
            env,
            env.unwrapped.task_actors,
            settle_steps=settle_steps,
            hold_action=joint_adapter.hold_action,
        )
        motion_executor = ManiSkillTrajectoryExecutor(joint_adapter)
        atoms = {atom.atom_id: atom for atom in plan.atoms}
        active_atom = None
        checks = []
        failed = False
        for event in events:
            event_type = event.get("type")
            if event_type == "atom_start":
                active_atom = atoms[str(event["atom_id"])]
            elif event_type == "trajectory":
                motion_executor.execute_trajectory(str(event["stage"]), event["trajectory"])
            elif event_type == "gripper":
                motion_executor.set_gripper(bool(event["closed"]))
            elif event_type == "atom_end":
                if active_atom is None or active_atom.atom_id != str(event["atom_id"]):
                    raise ValueError("trajectory package has mismatched atom markers")
                observed_pose = bridge.observe_object_pose(active_atom.object_id)
                execution = AtomExecution(
                    True,
                    final_object_pose=observed_pose,
                    joint_names=joint_adapter.arm_names,
                    joint_positions=joint_adapter.current_arm_qpos(),
                )
                validation = bridge.validate_atom(active_atom, execution, state)
                checks.append(
                    {
                        "atom_id": active_atom.atom_id,
                        "status": "passed" if validation.success else "failed",
                        "position_error_m": validation.position_error_m,
                        "orientation_error_deg": validation.orientation_error_deg,
                        "observed_pose": observed_pose.tolist(),
                        "message": validation.message,
                    }
                )
                committed_pose = (
                    observed_pose
                    if validation.observed_object_pose is None
                    else validation.observed_object_pose
                )
                state.commit_object_pose(active_atom.object_id, committed_pose)
                state.set_joints(joint_adapter.arm_names, joint_adapter.current_arm_qpos())
                if not validation.success:
                    failed = True
                    break
                active_atom = None
        result_file = output / "maniskill_gate_result.json"
        result_file.write_text(
            json.dumps(
                {"success": not failed, "checks": checks, "final_scene": state.as_dict()},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return GateResult(
            "maniskill",
            GateStatus.FAILED if failed else GateStatus.PASSED,
            "physics execution did not satisfy the task" if failed else "physics execution and settled poses passed",
            tuple(checks),
            artifacts={"scene_spec": str(scene_spec), "result_file": str(result_file), "video_dir": str(output / "video")},
        )
    finally:
        for method in ("flush_video", "flush", "close"):
            try:
                getattr(env, method)()
            except Exception:
                pass
