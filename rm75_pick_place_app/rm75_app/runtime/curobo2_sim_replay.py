"""Replay a planner-independent Curobo2 trajectory package in ManiSkill."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rm75_app.assets.object_specs import get_object_spec, normalize_object_name, resolve_object_spec_scales
from rm75_app.execution.trajectory_executor import ManiSkillTrajectoryExecutor
from rm75_app.planning.contracts import JointTrajectory


DEFAULT_EXTRA_MANISKILL_ROOT = "/home/zhangzhao/anaconda3/envs/realman/lib/python3.11/site-packages/mani_skill"


def load_replay_events(manifest_path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(manifest_path).expanduser().resolve()
    events = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(events, list):
        raise ValueError("execution manifest must contain a list")
    loaded: list[dict[str, Any]] = []
    for event in events:
        item = dict(event)
        if item.get("type") == "trajectory":
            relative = item.get("trajectory_file")
            if not relative:
                raise ValueError(f"trajectory event {item.get('stage')!r} has no trajectory_file")
            trajectory_path = (manifest.parent / str(relative)).resolve()
            if trajectory_path.parent != manifest.parent:
                raise ValueError(f"trajectory file escapes package directory: {relative}")
            with np.load(trajectory_path, allow_pickle=False) as data:
                item["trajectory"] = JointTrajectory(
                    tuple(str(name) for name in data["joint_names"].tolist()),
                    np.asarray(data["positions"], dtype=np.float64),
                    dt=np.asarray(data["dt"], dtype=np.float64) if "dt" in data else None,
                )
        loaded.append(item)
    return loaded


class _ManiSkillJointAdapter:
    def __init__(self, env: Any):
        self.env = env
        robot = env.unwrapped.agent.robot
        names = [joint.get_name() for joint in robot.get_active_joints()]
        self.arm_indices = [names.index(f"joint_{index}") for index in range(1, 8)]
        self.action_dim = int(np.prod(env.action_space.shape))

    def current_arm_qpos(self) -> np.ndarray:
        qpos = self.env.unwrapped.agent.robot.get_qpos()
        if hasattr(qpos, "detach"):
            qpos = qpos.detach().cpu().numpy()
        return np.asarray(qpos, dtype=np.float64).reshape(-1)[self.arm_indices]

    def compose_action(self, arm_target_q: np.ndarray, gripper_value: float) -> np.ndarray:
        action = np.zeros(self.action_dim, dtype=np.float32)
        action[:7] = np.asarray(arm_target_q, dtype=np.float32).reshape(-1)[:7]
        if self.action_dim > 7:
            action[7] = float(gripper_value)
        return action

    def step_and_render(self, action: np.ndarray, tag: str = "") -> None:
        del tag
        self.env.step(action)

    def hold_current_and_set_gripper(self, value: float, steps: int = 20) -> None:
        action = self.compose_action(self.current_arm_qpos(), value)
        for _ in range(int(steps)):
            self.env.step(action)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a Curobo2 execution package in ManiSkill")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, default=None)
    parser.add_argument("--env-id", default="Two_finger_PickJiaobang-v1")
    parser.add_argument("--extra-maniskill-package-root", default=DEFAULT_EXTRA_MANISKILL_ROOT)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--render-mode", choices=["rgb_array", "human"], default="rgb_array")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-path-points", type=int, default=400)
    parser.add_argument("--gripper-steps", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=5000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = args.manifest.expanduser().resolve()
    result_path = (
        args.result_json.expanduser().resolve()
        if args.result_json is not None
        else manifest.parent / "result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    object_name = normalize_object_name(result.get("object_name")) or str(result["object_name"])
    object_spec = get_object_spec(object_name)
    if object_spec is None:
        raise KeyError(f"simulation asset is not registered for {object_name!r}")
    _, sim_scale = resolve_object_spec_scales(object_spec)
    sim_asset = str(Path(object_spec.sim_asset_file or object_spec.mesh_file).expanduser().resolve())
    events = load_replay_events(manifest)

    import gymnasium as gym
    from mani_skill.utils.wrappers.record import RecordEpisode
    from rm75_app.execution import maniskill_scene

    maniskill_scene.ensure_pick_env_registered(args.env_id, args.extra_maniskill_package_root)
    video_dir = (
        args.video_dir.expanduser().resolve()
        if args.video_dir is not None
        else manifest.parent / "maniskill_replay"
    )
    video_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(
        args.env_id,
        robot_uids="RM75",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        max_episode_steps=int(args.max_episode_steps),
        object_asset_path=sim_asset,
        object_scale=float(sim_scale),
    )
    env = RecordEpisode(
        env,
        output_dir=str(video_dir),
        save_trajectory=False,
        save_video=True,
        source_type="motionplanning",
        source_desc="Curobo2 portable trajectory replay",
        video_fps=20,
    )
    try:
        env.reset(seed=int(args.seed))
        maniskill_scene.apply_object_physics_profile(env, object_name)
        T_base_object = np.asarray(result["T_base_object"], dtype=np.float32).reshape(4, 4)
        T_world_base = maniskill_scene.robot_base_transform(env)
        T_world_object = T_base_object if T_world_base is None else T_world_base @ T_base_object
        maniskill_scene.set_object_pose(env, T_world_object)
        initial_object_pose = maniskill_scene.object_pose_matrix(env)

        executor = ManiSkillTrajectoryExecutor(
            _ManiSkillJointAdapter(env),
            gripper_steps=int(args.gripper_steps),
            max_path_points=int(args.max_path_points),
        )
        for event in events:
            if event.get("type") == "trajectory":
                executor.execute_trajectory(str(event["stage"]), event["trajectory"])
            elif event.get("type") == "gripper":
                executor.set_gripper(bool(event["closed"]))
        final_object_pose = maniskill_scene.object_pose_matrix(env)
        displacement = None
        if initial_object_pose is not None and final_object_pose is not None:
            displacement = float(
                np.linalg.norm(final_object_pose[:3, 3] - initial_object_pose[:3, 3])
            )
        report = {
            "success": True,
            "backend": "maniskill",
            "object_name": object_name,
            "manifest": str(manifest),
            "video_dir": str(video_dir),
            "events_replayed": len(events),
            "object_displacement_m": displacement,
            "final_object_pose": None if final_object_pose is None else final_object_pose.tolist(),
        }
        (video_dir / "replay_result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        for method in ("flush_video", "flush", "close"):
            try:
                getattr(env, method)()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
