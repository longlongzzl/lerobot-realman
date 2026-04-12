#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import time
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import rm75_jiaobang_pick_real_with_foundationpose as real_mod


DEFAULT_MANIDREAMS_ROOT = Path("~/PycharmProjects/ManiDreams").expanduser()


def ensure_manidreams_paths(root: str | Path) -> Path:
    root_path = Path(root).expanduser().resolve()
    for path in [root_path, root_path / "src"]:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
    return root_path


ensure_manidreams_paths(DEFAULT_MANIDREAMS_ROOT)

try:
    from manidreams.base.dris import DRIS
    from manidreams.base.tsip import TSIPBase
except Exception:
    @dataclass
    class DRIS:  # type: ignore[override]
        observation: Any
        state_space: Any = None
        context: dict[str, Any] | None = None
        representation_type: str = "state"
        metadata: dict[str, Any] = field(default_factory=dict)

    class TSIPBase:  # type: ignore[override]
        def __init__(self, name: str | None = None):
            self.name = name or self.__class__.__name__

        def next(self, dris, action, cage=None):
            raise NotImplementedError

        def reset(self) -> None:
            return None


@dataclass(frozen=True)
class PushPrimitive:
    name: str
    direction_xy: np.ndarray
    push_distance: float
    contact_lateral_bias: float = 0.0
    wrist_roll_deg: float = 0.0
    backoff_scale: float = 1.0
    push_distance_scale: float = 1.0
    contact_height_offset: float = 0.0
    family: str = "generic"


@dataclass
class PushStagePlan:
    label: str
    pose: Any
    q_path: list[np.ndarray]


@dataclass
class PushPlanningDiagnostics:
    reachability_ratio: float
    screen_reason: str
    geometry_attempt: int | None
    contact_lateral_offset: float
    push_distance_used: float
    preview_stage_labels: list[str] = field(default_factory=list)
    preview_stage_terminal_qs: list[np.ndarray | None] = field(default_factory=list)


@dataclass
class PushPlan:
    primitive: PushPrimitive
    stages: list[PushStagePlan]
    reference_pose: Any
    diagnostics: PushPlanningDiagnostics


@dataclass
class RearrangementTraceSample:
    stage_label: str
    waypoint_index: int
    obj_p: np.ndarray
    obj_q: np.ndarray
    frame_bgr: np.ndarray | None = None


@dataclass
class GraspReadiness:
    feasible: bool
    pregrasp_feasible: bool
    grasp_label: str | None
    selected_grasp_pose: Any | None
    grasp_pose: Any
    pregrasp_pose: Any
    q_pre: np.ndarray | None
    q_grasp: np.ndarray | None
    score: float
    reason: str
    object_bottom_z: float | None
    min_obstacle_dist_xy: float | None
    approach_corridor_clearance_xy: float | None
    approach_blocker_count: int


@dataclass
class RearrangementEvaluation:
    primitive: PushPrimitive
    plan: PushPlan | None
    grasp: GraspReadiness
    score: float
    reason: str
    initial_obj_pose: tuple[np.ndarray, np.ndarray]
    final_obj_pose: tuple[np.ndarray, np.ndarray]
    object_motion_xy: float
    clearance_gain_xy: float
    corridor_clearance_gain_xy: float
    blocker_count_delta: int
    reachability_ratio: float
    screen_reason: str
    trace_samples: list[RearrangementTraceSample] = field(default_factory=list)


@dataclass
class RearrangementSearchResult:
    baseline: GraspReadiness
    baseline_dris: DRIS
    evaluations: list[RearrangementEvaluation]
    selected: RearrangementEvaluation | None
    scene_obstacles: list[dict[str, Any]] = field(default_factory=list)
    baseline_frame_bgr: np.ndarray | None = None
    search_tag: str = "search"


@dataclass
class SimGraspCandidate:
    label: str
    grasp_pose: Any
    pregrasp_pose: Any


@dataclass
class SimGraspEvaluation:
    candidate: SimGraspCandidate
    score: float
    reason: str
    feasible: bool
    grasped_after_close: bool
    retained_after_lift: bool
    object_lift_z: float
    close_gap: float | None
    lift_gap: float | None
    q_pre_path: list[np.ndarray] = field(default_factory=list)
    q_grasp_path: list[np.ndarray] = field(default_factory=list)
    q_lift_path: list[np.ndarray] = field(default_factory=list)
    trace_samples: list[RearrangementTraceSample] = field(default_factory=list)


flatten_np = real_mod.flatten_np


def parse_args():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--manidreams-root", type=str, default=str(DEFAULT_MANIDREAMS_ROOT))
    extra.add_argument(
        "--force-rearrangement",
        action="store_true",
        help="Search for a tabletop rearrangement action even if the current scene already looks directly graspable.",
    )
    extra.add_argument(
        "--max-rearrangement-steps",
        type=int,
        default=1,
        help="Maximum number of rearrangement actions before falling back to the normal grasp pipeline.",
    )
    extra.add_argument(
        "--rearrangement-push-distance",
        type=float,
        default=0.04,
        help="Horizontal sweep length in meters for each push primitive.",
    )
    extra.add_argument(
        "--rearrangement-backoff",
        type=float,
        default=0.02,
        help="Extra distance behind the contact point before starting the push.",
    )
    extra.add_argument(
        "--rearrangement-contact-margin",
        type=float,
        default=0.008,
        help="Extra xy offset added outside the object footprint when choosing the contact point.",
    )
    extra.add_argument(
        "--rearrangement-hover-height",
        type=float,
        default=0.05,
        help="Vertical lift above the pushing height for entering and leaving the push.",
    )
    extra.add_argument(
        "--rearrangement-contact-height",
        type=float,
        default=0.012,
        help="Minimum world z for the pushing contact point.",
    )
    extra.add_argument(
        "--rearrangement-contact-lift",
        type=float,
        default=0.003,
        help="Extra upward bias added above the estimated object bottom when building push contact poses.",
    )
    extra.add_argument(
        "--rearrangement-push-tcp-z-margin",
        type=float,
        default=0.012,
        help="Extra upward margin above the validated topdown grasp TCP z when building push contact poses.",
    )
    extra.add_argument(
        "--rearrangement-max-object-drop",
        type=float,
        default=0.02,
        help="Reject rearrangement candidates that lower the object too much relative to the baseline scene.",
    )
    extra.add_argument(
        "--rearrangement-table-penetration-tolerance",
        type=float,
        default=0.005,
        help="Allow a small negative bottom-z tolerance when validating rearrangement results, to absorb conservative oriented-box bounds on thin tabletop objects.",
    )
    extra.add_argument(
        "--rearrangement-stage-planning-time",
        type=float,
        default=3.0,
        help="RRT planning budget per push stage.",
    )
    extra.add_argument(
        "--rearrangement-stage-rrt-range",
        type=float,
        default=0.12,
        help="RRT range per push stage.",
    )
    extra.add_argument(
        "--disable-rearrangement-local-refine",
        action="store_true",
        help="Disable the coarse-to-fine refinement pass around the best coarse push primitive.",
    )
    extra.add_argument(
        "--rearrangement-local-refine-topk",
        type=int,
        default=1,
        help="How many coarse candidates seed the second-round local refinement search.",
    )
    extra.add_argument(
        "--rearrangement-local-refine-lateral-step",
        type=float,
        default=0.22,
        help="Extra lateral-bias delta used in the second-round local refinement pass.",
    )
    extra.add_argument(
        "--rearrangement-local-refine-push-scale-step",
        type=float,
        default=0.20,
        help="Extra push-distance scale delta used in the second-round local refinement pass.",
    )
    extra.add_argument(
        "--rearrangement-local-refine-contact-z-step",
        type=float,
        default=0.003,
        help="Extra TCP contact-height offset in meters used in the second-round local refinement pass.",
    )
    extra.add_argument(
        "--rearrangement-local-refine-backoff-scale-step",
        type=float,
        default=0.35,
        help="Extra backoff-scale delta used in the second-round local refinement pass.",
    )
    extra.add_argument(
        "--rearrangement-local-refine-wrist-roll-step-deg",
        type=float,
        default=15.0,
        help="Extra wrist-roll delta in degrees used in the second-round local refinement pass.",
    )
    extra.add_argument(
        "--rearrangement-sim-max-delta-per-step",
        type=float,
        default=0.02,
        help="Joint interpolation limit used while simulating a candidate push.",
    )
    extra.add_argument(
        "--rearrangement-sim-hold-steps",
        type=int,
        default=2,
        help="Extra hold steps after the last waypoint in each simulated push stage.",
    )
    extra.add_argument(
        "--rearrangement-sim-gripper-open",
        type=float,
        default=-1.0,
        help="Simulation gripper value used during push rollout.",
    )
    extra.add_argument(
        "--rearrangement-sim-gripper-value",
        type=float,
        default=None,
        help="Simulation gripper value used during rearrangement pushes. Defaults to sim_gripper_close.",
    )
    extra.add_argument(
        "--rearrangement-real-gripper-value",
        type=float,
        default=None,
        help="Real gripper value used during rearrangement pushes. Defaults to real_gripper_close.",
    )
    extra.add_argument(
        "--sim-exec-max-delta-per-step",
        type=float,
        default=0.02,
        help="Joint interpolation limit used for the full simulation execution path when --execute-real is not set.",
    )
    extra.add_argument(
        "--sim-exec-hold-steps",
        type=int,
        default=4,
        help="Extra hold steps at the end of each simulated motion stage in the full simulation execution path.",
    )
    extra.add_argument(
        "--sim-grasp-hold-steps",
        type=int,
        default=15,
        help="Extra hold steps at the grasp pose before closing the gripper in the full simulation execution path.",
    )
    extra.add_argument(
        "--sim-render-sleep",
        type=float,
        default=0.04,
        help="Wall-clock sleep in seconds after each simulated env.step when render_mode=human during executed motions.",
    )
    extra.add_argument(
        "--sim-gripper-open",
        type=float,
        default=-1.0,
        help="Simulation gripper command used for the open state during the full simulation execution path.",
    )
    extra.add_argument(
        "--sim-gripper-close",
        type=float,
        default=1.0,
        help="Simulation gripper command used for the closed state during the full simulation execution path.",
    )
    extra.add_argument(
        "--sim-gripper-settle-steps",
        type=int,
        default=24,
        help="Simulation settling steps after opening or closing the gripper in the full simulation execution path.",
    )
    extra.add_argument(
        "--sim-gripper-blocked-gap-threshold",
        type=float,
        default=0.02,
        help="Treat the simulated gripper as blocked by an object when the finger-pad gap after a close remains above this threshold in meters.",
    )
    extra.add_argument(
        "--disable-sim-grasp-check",
        action="store_true",
        help="Do not fail the full simulation execution path when the gripper-opening proxy says the gripper fully closed after closing or transport.",
    )
    extra.add_argument(
        "--disable-sim-grasp-retries",
        action="store_true",
        help="Disable simulated grasp-pose retries after the first close attempt fails.",
    )
    extra.add_argument(
        "--rearrangement-include-diagonals",
        action="store_true",
        help="Include diagonal push primitives in addition to the four cardinal directions.",
    )
    extra.add_argument(
        "--rearrangement-preview-rollout",
        action="store_true",
        help="Keep rendering enabled during simulated push rollout when render_mode=human.",
    )
    extra.add_argument(
        "--rearrangement-min-improvement",
        type=float,
        default=20.0,
        help="Minimum score gain over the no-action baseline before a push is executed.",
    )
    extra.add_argument(
        "--rearrangement-min-improvement-when-ungraspable",
        type=float,
        default=0.0,
        help="Minimum score gain required to execute a push when the baseline scene is not directly graspable.",
    )
    extra.add_argument(
        "--rearrangement-primitive-grasp-bonus",
        type=float,
        default=100.0,
        help="Score bonus when the rearranged scene becomes directly graspable.",
    )
    extra.add_argument(
        "--rearrangement-pregrasp-bonus",
        type=float,
        default=25.0,
        help="Score bonus when the rearranged scene reaches a valid pregrasp.",
    )
    extra.add_argument(
        "--rearrangement-clearance-scale",
        type=float,
        default=25.0,
        help="Reward scale for increased xy clearance between target and known obstacles.",
    )
    extra.add_argument(
        "--rearrangement-clearance-gain-scale",
        type=float,
        default=35.0,
        help="Extra reward scale for increasing target-obstacle xy clearance relative to the baseline scene.",
    )
    extra.add_argument(
        "--rearrangement-corridor-half-width",
        type=float,
        default=0.05,
        help="Half-width in meters of the approximate top-down grasp approach corridor used for clutter-aware scoring.",
    )
    extra.add_argument(
        "--rearrangement-corridor-clearance-scale",
        type=float,
        default=20.0,
        help="Reward scale for larger lateral clearance inside the target grasp-approach corridor.",
    )
    extra.add_argument(
        "--rearrangement-blocker-penalty",
        type=float,
        default=12.0,
        help="Penalty per obstacle center that remains inside the approximate top-down grasp corridor.",
    )
    extra.add_argument(
        "--rearrangement-blocker-gain-scale",
        type=float,
        default=16.0,
        help="Reward scale for reducing the number of obstacle centers inside the grasp corridor after a push.",
    )
    extra.add_argument(
        "--rearrangement-reachability-scale",
        type=float,
        default=8.0,
        help="Reward scale applied to the robot-aware IK reachability pre-screen ratio for each push candidate.",
    )
    extra.add_argument(
        "--rearrangement-motion-penalty-scale",
        type=float,
        default=5.0,
        help="Penalty factor for larger object motion during the simulated push.",
    )
    extra.add_argument(
        "--rearrangement-export-grid-video",
        action="store_true",
        help="Export a 4x4 candidate-evaluation video with rollout progress and scoring overlays for each rearrangement search.",
    )
    extra.add_argument(
        "--rearrangement-video-dir",
        type=str,
        default=str(Path("~/Desktop/lerobot/pick_jiaobang/tsip_eval_videos").expanduser()),
        help="Directory used for candidate evaluation videos and JSON summaries.",
    )
    extra.add_argument(
        "--rearrangement-video-fps",
        type=int,
        default=8,
        help="FPS for exported rearrangement evaluation videos.",
    )
    extra.add_argument(
        "--rearrangement-video-tile-size",
        type=int,
        default=320,
        help="Per-tile pixel size for the exported 4x4 rearrangement evaluation grid video.",
    )
    extra.add_argument(
        "--rearrangement-video-sample-repeat",
        type=int,
        default=2,
        help="How many video frames to repeat each recorded rollout sample in the rearrangement evaluation export.",
    )
    extra.add_argument(
        "--rearrangement-only",
        action="store_true",
        help="Execute at most one rearrangement action and stop before the final grasp stage.",
    )
    extra.add_argument(
        "--disable-sim-grasp-tsip-selection",
        action="store_true",
        help="Disable TSIP-style candidate evaluation for simulated grasp-pose selection and use the original rule-based grasp directly.",
    )
    extra.add_argument(
        "--sim-grasp-tsip-planning-time",
        type=float,
        default=2.5,
        help="Planning budget used while internally evaluating simulated grasp candidates.",
    )
    extra.add_argument(
        "--sim-grasp-tsip-rrt-range",
        type=float,
        default=0.12,
        help="RRT range used while internally evaluating simulated grasp candidates.",
    )
    extra.add_argument(
        "--sim-grasp-tsip-lift-height",
        type=float,
        default=0.03,
        help="Vertical lift in meters used to score grasp retention for simulated grasp candidates.",
    )
    extra.add_argument(
        "--sim-grasp-tsip-yaw-offsets-deg",
        type=float,
        nargs="*",
        default=[-12.0, 0.0, 12.0],
        help="Approach-axis yaw offsets in degrees used to generate simulated grasp candidates.",
    )
    extra.add_argument(
        "--sim-grasp-tsip-lateral-offsets",
        type=float,
        nargs="*",
        default=[-0.004, 0.0, 0.004],
        help="Offsets in meters along the TCP ortho axis used to generate simulated grasp candidates.",
    )
    extra.add_argument(
        "--sim-grasp-tsip-closing-offsets",
        type=float,
        nargs="*",
        default=[-0.003, 0.0, 0.003],
        help="Offsets in meters along the TCP closing axis used to generate simulated grasp candidates.",
    )
    extra.add_argument(
        "--sim-grasp-tsip-z-offsets",
        type=float,
        nargs="*",
        default=[0.0, 0.004],
        help="World-z offsets in meters used to generate simulated grasp candidates.",
    )
    extra.add_argument(
        "--sim-grasp-tsip-export-grid-video",
        action="store_true",
        help="Export a 4x4 candidate-evaluation video for simulated grasp-pose selection.",
    )
    extra.add_argument(
        "--sim-grasp-tsip-video-dir",
        type=str,
        default=str(Path("~/Desktop/lerobot/pick_jiaobang/grasp_tsip_eval_videos").expanduser()),
        help="Directory used for simulated grasp candidate videos and JSON summaries.",
    )

    known_args, remaining = extra.parse_known_args()
    ensure_manidreams_paths(known_args.manidreams_root)

    saved_argv = sys.argv[:]
    sys.argv = [saved_argv[0]] + remaining
    try:
        base_args = real_mod.parse_args()
    finally:
        sys.argv = saved_argv

    merged = argparse.Namespace(**vars(base_args))
    for key, value in vars(known_args).items():
        setattr(merged, key, value)
    return merged



def clone_state_tree(obj):
    if isinstance(obj, dict):
        return {key: clone_state_tree(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clone_state_tree(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(clone_state_tree(value) for value in obj)
    if hasattr(obj, "clone"):
        try:
            return obj.clone()
        except Exception:
            pass
    if isinstance(obj, np.ndarray):
        return obj.copy()
    return obj


RUNTIME_ATTR_CANDIDATES = (
    "elapsed_steps",
    "_elapsed_steps",
    "episode_step",
    "_episode_step",
    "step_count",
    "_step_count",
    "_elapsed_episode_steps",
    "elapsed_step",
    "_elapsed_step",
)



def snapshot_runtime_attrs(obj) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name in RUNTIME_ATTR_CANDIDATES:
        if hasattr(obj, name):
            try:
                state[name] = clone_state_tree(getattr(obj, name))
            except Exception:
                pass
    return state



def restore_runtime_attrs(obj, state: dict[str, Any] | None):
    if not state:
        return
    for name, value in state.items():
        try:
            setattr(obj, name, clone_state_tree(value))
        except Exception:
            pass



def snapshot_env_state(env):
    base_env = env.unwrapped
    if hasattr(base_env, "get_state_dict"):
        return clone_state_tree(base_env.get_state_dict())
    if hasattr(base_env.scene, "get_sim_state"):
        return clone_state_tree(base_env.scene.get_sim_state())
    raise RuntimeError("Environment does not support state snapshot via get_state_dict() or scene.get_sim_state().")



def snapshot_demo_state(demo):
    return {
        "__snapshot_kind__": "demo_state_v1",
        "physics": snapshot_env_state(demo.env),
        "runtime": {
            "base_env": snapshot_runtime_attrs(demo.env.unwrapped),
            "demo": {
                "step_counter": int(getattr(demo, "step_counter", 0)),
                "last_reward": clone_state_tree(getattr(demo, "last_reward", None)),
                "last_terminated": clone_state_tree(getattr(demo, "last_terminated", False)),
                "last_truncated": clone_state_tree(getattr(demo, "last_truncated", False)),
                "last_info": clone_state_tree(getattr(demo, "last_info", {})),
            },
        },
    }



def restore_demo_state(demo, state_dict):
    base_env = demo.env.unwrapped
    runtime_state = {}
    restored = clone_state_tree(state_dict)
    if isinstance(restored, dict) and restored.get("__snapshot_kind__") == "demo_state_v1":
        runtime_state = dict(restored.get("runtime", {}) or {})
        restored = restored.get("physics")
    if hasattr(base_env, "set_state_dict"):
        base_env.set_state_dict(restored)
    elif hasattr(base_env.scene, "set_sim_state"):
        base_env.scene.set_sim_state(restored)
    else:
        raise RuntimeError("Environment does not support state restore via set_state_dict() or scene.set_sim_state().")

    try:
        for _ in range(2):
            base_env.scene.step()
    except Exception:
        pass

    try:
        demo.refresh_runtime_handles(rebuild_visual=False)
    except Exception:
        pass
    restore_runtime_attrs(base_env, runtime_state.get("base_env"))
    demo_runtime = dict(runtime_state.get("demo", {}) or {})
    if "step_counter" in demo_runtime:
        try:
            demo.step_counter = int(demo_runtime["step_counter"])
        except Exception:
            pass
    for name in ("last_reward", "last_terminated", "last_truncated", "last_info"):
        if name in demo_runtime:
            setattr(demo, name, clone_state_tree(demo_runtime[name]))
    real_mod.sync_planner_qpos_from_demo(demo)
    real_mod.update_attached_box_visual(demo)



def normalize_xy(direction_xy: np.ndarray) -> np.ndarray:
    vec = np.asarray(direction_xy, dtype=np.float32).reshape(2)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        raise ValueError(f"Degenerate push direction: {vec}")
    return (vec / norm).astype(np.float32)



def make_push_primitives(args) -> list[PushPrimitive]:
    base_dirs = [
        ("push_pos_x", np.array([1.0, 0.0], dtype=np.float32)),
        ("push_neg_x", np.array([-1.0, 0.0], dtype=np.float32)),
        ("push_pos_y", np.array([0.0, 1.0], dtype=np.float32)),
        ("push_neg_y", np.array([0.0, -1.0], dtype=np.float32)),
    ]
    if args.rearrangement_include_diagonals:
        base_dirs.extend(
            [
                ("push_pos_x_pos_y", np.array([1.0, 1.0], dtype=np.float32)),
                ("push_pos_x_neg_y", np.array([1.0, -1.0], dtype=np.float32)),
                ("push_neg_x_pos_y", np.array([-1.0, 1.0], dtype=np.float32)),
                ("push_neg_x_neg_y", np.array([-1.0, -1.0], dtype=np.float32)),
            ]
        )

    variant_specs = [
        ("center", 0.0, 0.0),
        ("flip", 0.0, 180.0),
        ("left", 0.65, 0.0),
        ("right", -0.65, 0.0),
    ]

    primitives: list[PushPrimitive] = []
    for base_name, direction_xy in base_dirs:
        direction_xy = normalize_xy(direction_xy)
        for variant_name, lateral_bias, wrist_roll_deg in variant_specs:
            primitives.append(
                PushPrimitive(
                    name=f"{base_name}_{variant_name}",
                    direction_xy=direction_xy,
                    push_distance=float(args.rearrangement_push_distance),
                    contact_lateral_bias=float(lateral_bias),
                    wrist_roll_deg=float(wrist_roll_deg),
                    family=variant_name,
                )
            )
    return primitives


def make_local_refinement_primitives(base_primitive: PushPrimitive, args) -> list[PushPrimitive]:
    lateral_step = float(max(getattr(args, "rearrangement_local_refine_lateral_step", 0.22), 0.0))
    push_scale_step = float(max(getattr(args, "rearrangement_local_refine_push_scale_step", 0.20), 0.0))
    contact_z_step = float(max(getattr(args, "rearrangement_local_refine_contact_z_step", 0.003), 0.0))
    backoff_scale_step = float(max(getattr(args, "rearrangement_local_refine_backoff_scale_step", 0.35), 0.0))
    wrist_roll_step_deg = float(max(getattr(args, "rearrangement_local_refine_wrist_roll_step_deg", 15.0), 0.0))

    variants: list[tuple[str, dict[str, float]]] = [
        ("ref_base", {}),
        ("ref_lat_p", {"contact_lateral_bias": base_primitive.contact_lateral_bias + lateral_step}),
        ("ref_lat_m", {"contact_lateral_bias": base_primitive.contact_lateral_bias - lateral_step}),
        ("ref_push_p", {"push_distance_scale": base_primitive.push_distance_scale + push_scale_step}),
        ("ref_push_m", {"push_distance_scale": max(base_primitive.push_distance_scale - push_scale_step, 0.35)}),
        ("ref_z_p", {"contact_height_offset": base_primitive.contact_height_offset + contact_z_step}),
        ("ref_z_m", {"contact_height_offset": base_primitive.contact_height_offset - contact_z_step}),
        ("ref_backoff_p", {"backoff_scale": base_primitive.backoff_scale + backoff_scale_step}),
        ("ref_backoff_m", {"backoff_scale": max(base_primitive.backoff_scale - backoff_scale_step, 0.15)}),
        ("ref_roll_p", {"wrist_roll_deg": base_primitive.wrist_roll_deg + wrist_roll_step_deg}),
        ("ref_roll_m", {"wrist_roll_deg": base_primitive.wrist_roll_deg - wrist_roll_step_deg}),
    ]

    seen_names: set[str] = set()
    refined: list[PushPrimitive] = []
    for suffix, updates in variants:
        primitive = PushPrimitive(
            name=f"{base_primitive.name}_{suffix}",
            direction_xy=np.asarray(base_primitive.direction_xy, dtype=np.float32).copy(),
            push_distance=float(base_primitive.push_distance),
            contact_lateral_bias=float(np.clip(updates.get("contact_lateral_bias", base_primitive.contact_lateral_bias), -1.25, 1.25)),
            wrist_roll_deg=float(updates.get("wrist_roll_deg", base_primitive.wrist_roll_deg)),
            backoff_scale=float(max(updates.get("backoff_scale", base_primitive.backoff_scale), 0.05)),
            push_distance_scale=float(max(updates.get("push_distance_scale", base_primitive.push_distance_scale), 0.25)),
            contact_height_offset=float(updates.get("contact_height_offset", base_primitive.contact_height_offset)),
            family=f"{base_primitive.family}_refined",
        )
        if primitive.name in seen_names:
            continue
        seen_names.add(primitive.name)
        refined.append(primitive)
    return refined


def candidate_rank_key(evaluation: RearrangementEvaluation) -> tuple[float, int, float, int, int]:
    return (
        float(evaluation.score),
        int(evaluation.plan is not None),
        float(evaluation.reachability_ratio),
        int(evaluation.grasp.pregrasp_feasible),
        int(evaluation.grasp.feasible),
    )


def select_rearrangement_candidate(
    evaluations: list[RearrangementEvaluation],
    baseline: GraspReadiness,
    args,
) -> RearrangementEvaluation | None:
    if not evaluations:
        return None
    candidate = max(evaluations, key=candidate_rank_key)
    if candidate.plan is None:
        return None
    if baseline.feasible:
        required_gain = float(args.rearrangement_min_improvement)
    else:
        required_gain = float(getattr(args, "rearrangement_min_improvement_when_ungraspable", 0.0))
    if candidate.score >= baseline.score + required_gain:
        return candidate
    return None



def pose_world_xyz(pose) -> np.ndarray:
    return np.asarray(flatten_np(pose.p)[:3], dtype=np.float32)



def compute_min_obstacle_dist_xy(demo) -> float | None:
    scene_obstacles = list(getattr(demo, "scene_obstacles", []) or [])
    if not scene_obstacles:
        return None
    obj_p, _ = demo.get_obj_pose()
    obj_xy = np.asarray(obj_p[:2], dtype=np.float32)
    dists = []
    for item in scene_obstacles:
        T_world_obj = np.asarray(item.get("T_world_obj"), dtype=np.float32)
        if T_world_obj.shape != (4, 4):
            continue
        dists.append(float(np.linalg.norm(T_world_obj[:2, 3] - obj_xy)))
    if not dists:
        return None
    return float(min(dists))



def compute_approach_corridor_metrics(demo, pregrasp_pose, grasp_pose, args) -> tuple[float | None, int]:
    scene_obstacles = list(getattr(demo, "scene_obstacles", []) or [])
    if not scene_obstacles:
        return None, 0

    pre_xy = pose_world_xyz(pregrasp_pose)[:2]
    grasp_xy = pose_world_xyz(grasp_pose)[:2]
    segment = grasp_xy - pre_xy
    segment_norm = float(np.linalg.norm(segment))
    if segment_norm < 1e-6:
        return None, 0
    direction = segment / segment_norm
    half_width = float(max(getattr(args, "rearrangement_corridor_half_width", 0.05), 1e-3))
    along_margin = 0.03
    blocker_count = 0
    min_lateral = None

    for item in scene_obstacles:
        T_world_obj = np.asarray(item.get("T_world_obj"), dtype=np.float32)
        if T_world_obj.shape != (4, 4):
            continue
        obs_xy = T_world_obj[:2, 3]
        rel = obs_xy - pre_xy
        along = float(np.dot(rel, direction))
        if along < -along_margin or along > segment_norm + along_margin:
            continue
        lateral = abs(float(direction[0] * rel[1] - direction[1] * rel[0]))
        if min_lateral is None or lateral < min_lateral:
            min_lateral = lateral
        if lateral <= half_width:
            blocker_count += 1

    return min_lateral, blocker_count



def assess_direct_grasp(demo, bridge_mod, args, *, preview: bool = False) -> GraspReadiness:
    scoring_args = args
    if not preview:
        scoring_args = argparse.Namespace(**vars(args).copy())
        scoring_args.render_mode = "none"

    grasp_pose = demo.build_topdown_grasp_pose()
    pregrasp_pose = demo.build_pregrasp_pose(grasp_pose)
    grasp_pose, pregrasp_pose, _ = real_mod.enforce_min_grasp_tcp_z(
        grasp_pose,
        pregrasp_pose,
        args.min_grasp_tcp_z,
    )

    q_pre = demo.plan_terminal_q(pregrasp_pose, variant_name=args.variant)
    grasp_plan = real_mod.plan_grasp_pose_with_fallbacks(
        demo,
        bridge_mod,
        grasp_pose,
        scoring_args,
    )

    pregrasp_feasible = q_pre is not None
    grasp_label = grasp_plan.label if grasp_plan is not None else None
    selected_grasp_pose = grasp_plan.final_grasp_pose if grasp_plan is not None else None
    grasp_feasible = grasp_plan is not None
    q_grasp = None
    if grasp_plan is not None:
        if grasp_plan.final_grasp_path:
            q_grasp = np.asarray(grasp_plan.final_grasp_path[-1], dtype=np.float32).reshape(-1)[:7]
        else:
            q_grasp = np.asarray(grasp_plan.approach_q, dtype=np.float32).reshape(-1)[:7]
    object_bottom_z = real_mod.get_object_bottom_z(demo, args)
    min_obstacle_dist_xy = compute_min_obstacle_dist_xy(demo)
    corridor_clearance_xy, approach_blocker_count = compute_approach_corridor_metrics(
        demo,
        pregrasp_pose,
        selected_grasp_pose or grasp_pose,
        args,
    )

    score = 0.0
    if pregrasp_feasible:
        score += float(args.rearrangement_pregrasp_bonus)
    if grasp_feasible:
        score += float(args.rearrangement_primitive_grasp_bonus)
    if min_obstacle_dist_xy is not None:
        score += float(min(min_obstacle_dist_xy, 0.20) * args.rearrangement_clearance_scale)
    if corridor_clearance_xy is not None:
        score += float(min(corridor_clearance_xy, 0.12) * args.rearrangement_corridor_clearance_scale)
    score -= float(approach_blocker_count) * float(args.rearrangement_blocker_penalty)
    if object_bottom_z is not None and object_bottom_z < -1e-3:
        score -= 100.0

    reason_parts = []
    if not pregrasp_feasible:
        reason_parts.append("pregrasp_unreachable")
    if not grasp_feasible:
        reason_parts.append("grasp_unreachable")
    if approach_blocker_count > 0 and not grasp_feasible:
        reason_parts.append(f"approach_blocked_{approach_blocker_count}")
    if object_bottom_z is not None and object_bottom_z < -1e-3:
        reason_parts.append("object_penetrates_table")
    if not reason_parts:
        reason_parts.append("direct_grasp_feasible")

    return GraspReadiness(
        feasible=grasp_feasible,
        pregrasp_feasible=pregrasp_feasible,
        grasp_label=grasp_label,
        selected_grasp_pose=selected_grasp_pose,
        grasp_pose=grasp_pose,
        pregrasp_pose=pregrasp_pose,
        q_pre=None if q_pre is None else np.asarray(q_pre, dtype=np.float32).reshape(-1)[:7],
        q_grasp=None if q_grasp is None else np.asarray(q_grasp, dtype=np.float32).reshape(-1)[:7],
        score=float(score),
        reason=",".join(reason_parts),
        object_bottom_z=object_bottom_z,
        min_obstacle_dist_xy=min_obstacle_dist_xy,
        approach_corridor_clearance_xy=corridor_clearance_xy,
        approach_blocker_count=int(approach_blocker_count),
    )



def invalidate_grasp_readiness(grasp: GraspReadiness, reason: str) -> GraspReadiness:
    return GraspReadiness(
        feasible=False,
        pregrasp_feasible=False,
        grasp_label=grasp.grasp_label,
        selected_grasp_pose=grasp.selected_grasp_pose,
        grasp_pose=grasp.grasp_pose,
        pregrasp_pose=grasp.pregrasp_pose,
        q_pre=grasp.q_pre,
        q_grasp=grasp.q_grasp,
        score=grasp.score,
        reason=reason,
        object_bottom_z=grasp.object_bottom_z,
        min_obstacle_dist_xy=grasp.min_obstacle_dist_xy,
        approach_corridor_clearance_xy=grasp.approach_corridor_clearance_xy,
        approach_blocker_count=grasp.approach_blocker_count,
    )



def estimate_object_half_extent_xy_along_direction(demo, args, direction_xy: np.ndarray) -> float:
    from transforms3d.quaternions import quat2mat

    size = real_mod.get_asset_box_size(args.sim_asset_file, args.sim_asset_scale)
    _, obj_q = demo.get_obj_pose()
    half_extents = np.asarray(size, dtype=np.float32).reshape(3) * 0.5
    direction_xy = normalize_xy(direction_xy).astype(np.float64)
    R = quat2mat(np.asarray(obj_q, dtype=np.float64).reshape(4))

    extent = 0.0
    for axis_idx in range(3):
        axis_xy = np.asarray(R[:2, axis_idx], dtype=np.float64)
        extent += abs(float(np.dot(direction_xy, axis_xy))) * float(half_extents[axis_idx])

    min_extent = float(np.clip(min(size[0], size[1]) * 0.35, 0.01, 0.04))
    return float(max(extent, min_extent))



def compute_object_oriented_z_bounds(demo, args) -> tuple[float, float]:
    from transforms3d.quaternions import quat2mat

    size = real_mod.get_asset_box_size(args.sim_asset_file, args.sim_asset_scale)
    obj_p, obj_q = demo.get_obj_pose()
    center = np.asarray(obj_p, dtype=np.float32).reshape(3)
    half = np.asarray(size, dtype=np.float32).reshape(3) * 0.5
    corners = np.asarray(
        [
            [sx, sy, sz]
            for sx in (-half[0], half[0])
            for sy in (-half[1], half[1])
            for sz in (-half[2], half[2])
        ],
        dtype=np.float32,
    )
    R = quat2mat(np.asarray(obj_q, dtype=np.float32).reshape(4)).astype(np.float32)
    world = corners @ R.T + center.reshape(1, 3)
    return float(world[:, 2].min()), float(world[:, 2].max())


def estimate_contact_height(demo, args) -> float:
    try:
        object_bottom_z, object_top_z = compute_object_oriented_z_bounds(demo, args)
    except Exception:
        size = real_mod.get_asset_box_size(args.sim_asset_file, args.sim_asset_scale)
        obj_p, _ = demo.get_obj_pose()
        object_bottom_z = real_mod.get_object_bottom_z(demo, args)
        if object_bottom_z is None:
            object_bottom_z = float(obj_p[2] - 0.5 * size[2])
        object_top_z = float(object_bottom_z + size[2])

    world_height = max(float(object_top_z - object_bottom_z), 1e-4)
    body_height = float(np.clip(world_height * 0.45, 0.008, 0.02))
    lower_bound = float(object_bottom_z + min(world_height * 0.25, 0.008))
    upper_bound = float(max(object_bottom_z + 0.002, object_top_z - 0.003))
    contact_z = float(object_bottom_z + body_height + float(args.rearrangement_contact_lift))
    contact_z = float(np.clip(contact_z, lower_bound, upper_bound))
    safe_push_tcp_z = float(max(args.rearrangement_contact_height, contact_z))
    try:
        grasp_pose = demo.build_topdown_grasp_pose()
        pregrasp_pose = demo.build_pregrasp_pose(grasp_pose)
        grasp_pose, _, _ = real_mod.enforce_min_grasp_tcp_z(
            grasp_pose,
            pregrasp_pose,
            args.min_grasp_tcp_z,
        )
        safe_push_tcp_z = max(
            safe_push_tcp_z,
            float(pose_world_xyz(grasp_pose)[2]) + float(args.rearrangement_push_tcp_z_margin),
        )
    except Exception:
        pass
    return float(safe_push_tcp_z)



def build_push_tcp_pose(direction_xy: np.ndarray, position_xyz: np.ndarray, *, wrist_roll_deg: float = 0.0):
    from transforms3d.quaternions import mat2quat

    direction_xy = normalize_xy(direction_xy).astype(np.float64)
    # Keep the tool vertical and use the gripper's side face for lateral pushes.
    approaching = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    closing = np.array([direction_xy[0], direction_xy[1], 0.0], dtype=np.float64)
    closing_norm = np.linalg.norm(closing)
    if closing_norm < 1e-8:
        raise ValueError(f"Failed to build push orientation for direction {direction_xy}")
    closing = closing / closing_norm
    ortho = np.cross(closing, approaching)
    ortho_norm = np.linalg.norm(ortho)
    if ortho_norm < 1e-8:
        raise ValueError(f"Failed to build push orientation for direction {direction_xy}")
    ortho = ortho / ortho_norm
    closing = np.cross(approaching, ortho)
    closing = closing / max(np.linalg.norm(closing), 1e-8)
    R_tcp = np.stack([ortho, closing, approaching], axis=1)
    if abs(float(wrist_roll_deg)) > 1e-6:
        theta = math.radians(float(wrist_roll_deg))
        c = math.cos(theta)
        s = math.sin(theta)
        R_roll = np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        R_tcp = R_tcp @ R_roll
    q_wxyz = np.asarray(mat2quat(R_tcp), dtype=np.float32)
    p_xyz = np.asarray(position_xyz, dtype=np.float32).reshape(3)
    return real_mod.Pose.create_from_pq(p=p_xyz, q=q_wxyz)



def screen_push_stage_poses(demo, stage_poses, args) -> tuple[bool, float, str, list[np.ndarray | None]]:
    reachable_labels: list[str] = []
    terminal_qs: list[np.ndarray | None] = []
    for label, pose in stage_poses:
        q_terminal = demo.plan_terminal_q(pose, variant_name=args.variant)
        if q_terminal is not None:
            q_terminal = np.asarray(q_terminal, dtype=np.float32).reshape(-1)[:7].copy()
        terminal_qs.append(q_terminal)
        if q_terminal is not None:
            reachable_labels.append(label)
    ratio = float(len(reachable_labels) / max(len(stage_poses), 1))
    required = {"hover_pre", "contact"}
    if required.issubset(set(reachable_labels)) and ratio >= 0.75:
        return True, ratio, "ik_screen_pass", terminal_qs
    failed = [label for label, _ in stage_poses if label not in reachable_labels]
    return False, ratio, f"ik_screen_failed:{'+'.join(failed) if failed else 'none'}", terminal_qs



def plan_push_primitive_paths(demo, primitive: PushPrimitive, args) -> tuple[PushPlan | None, PushPlanningDiagnostics]:
    obj_p, _ = demo.get_obj_pose()
    obj_p = np.asarray(obj_p, dtype=np.float32)
    obj_xy = obj_p[:2].copy()
    direction_xy = normalize_xy(primitive.direction_xy)
    lateral_dir = np.array([-direction_xy[1], direction_xy[0]], dtype=np.float32)

    contact_extent = estimate_object_half_extent_xy_along_direction(demo, args, direction_xy)
    lateral_extent = estimate_object_half_extent_xy_along_direction(demo, args, lateral_dir)
    contact_radius = float(contact_extent + args.rearrangement_contact_margin)
    contact_z = float(max(
        args.rearrangement_contact_height,
        estimate_contact_height(demo, args) + float(getattr(primitive, "contact_height_offset", 0.0)),
    ))
    hover_z = float(contact_z + args.rearrangement_hover_height)
    lateral_offset = float(np.clip(lateral_extent * 0.55, 0.012, 0.04) * float(primitive.contact_lateral_bias))
    nominal_backoff = float(args.rearrangement_backoff) * float(primitive.backoff_scale)
    nominal_push_distance = float(primitive.push_distance) * float(primitive.push_distance_scale)

    geometry_attempts = [
        (nominal_backoff, nominal_push_distance),
        (float(max(nominal_backoff * 0.5, 0.0)), float(max(min(nominal_push_distance, 0.03), 0.02))),
        (0.0, float(max(min(nominal_push_distance, 0.025), 0.015))),
    ]

    start_q_seed = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    best_diag = PushPlanningDiagnostics(
        reachability_ratio=0.0,
        screen_reason="planning_not_started",
        geometry_attempt=None,
        contact_lateral_offset=lateral_offset,
        push_distance_used=nominal_push_distance,
        preview_stage_labels=[],
        preview_stage_terminal_qs=[],
    )
    seen_attempts: set[tuple[float, float]] = set()
    for attempt_idx, (backoff_distance, push_distance) in enumerate(geometry_attempts, start=1):
        key = (round(backoff_distance, 5), round(push_distance, 5))
        if key in seen_attempts:
            continue
        seen_attempts.add(key)

        start_xy = obj_xy - direction_xy * float(contact_radius + backoff_distance) + lateral_dir * lateral_offset
        contact_xy = obj_xy - direction_xy * float(contact_radius) + lateral_dir * lateral_offset
        end_xy = contact_xy + direction_xy * float(push_distance)
        print(
            f"[rearrangement] {primitive.name} geometry attempt={attempt_idx}, "
            f"contact_radius={contact_radius:.4f}, backoff={backoff_distance:.4f}, "
            f"push_distance={push_distance:.4f}, contact_z={contact_z:.4f}, "
            f"lateral_offset={lateral_offset:.4f}, wrist_roll={primitive.wrist_roll_deg:.1f}"
        )

        stage_positions = [
            ("hover_pre", np.array([start_xy[0], start_xy[1], hover_z], dtype=np.float32)),
            ("contact", np.array([contact_xy[0], contact_xy[1], contact_z], dtype=np.float32)),
            ("push", np.array([end_xy[0], end_xy[1], contact_z], dtype=np.float32)),
            ("hover_post", np.array([end_xy[0], end_xy[1], hover_z], dtype=np.float32)),
        ]
        stage_poses = [
            (label, build_push_tcp_pose(direction_xy, position_xyz, wrist_roll_deg=primitive.wrist_roll_deg))
            for label, position_xyz in stage_positions
        ]
        screen_ok, reachability_ratio, screen_reason, stage_terminal_qs = screen_push_stage_poses(demo, stage_poses, args)
        if reachability_ratio >= best_diag.reachability_ratio:
            best_diag = PushPlanningDiagnostics(
                reachability_ratio=reachability_ratio,
                screen_reason=screen_reason,
                geometry_attempt=attempt_idx,
                contact_lateral_offset=lateral_offset,
                push_distance_used=float(push_distance),
                preview_stage_labels=[label for label, _ in stage_poses],
                preview_stage_terminal_qs=[None if q is None else q.copy() for q in stage_terminal_qs],
            )
        if not screen_ok:
            continue

        stages: list[PushStagePlan] = []
        start_q = start_q_seed.copy()
        attempt_reason = "ik_screen_pass"
        for label, pose in stage_poses:
            q_path = real_mod.plan_pose_path(
                demo,
                pose,
                variant_name=args.variant,
                use_attach=False,
                label=f"{primitive.name}_{label}",
                planning_time=args.rearrangement_stage_planning_time,
                rrt_range=args.rearrangement_stage_rrt_range,
                start_q=start_q,
            )
            if not q_path:
                attempt_reason = f"{label}_planning_failed_after_screen"
                stages = []
                break
            q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in q_path]
            stages.append(PushStagePlan(label=label, pose=pose, q_path=q_path))
            start_q = q_path[-1]

        if stages:
            diagnostics = PushPlanningDiagnostics(
                reachability_ratio=reachability_ratio,
                screen_reason=screen_reason,
                geometry_attempt=attempt_idx,
                contact_lateral_offset=lateral_offset,
                push_distance_used=float(push_distance),
                preview_stage_labels=[label for label, _ in stage_poses],
                preview_stage_terminal_qs=[None if q is None else q.copy() for q in stage_terminal_qs],
            )
            return PushPlan(
                primitive=primitive,
                stages=stages,
                reference_pose=stages[0].pose,
                diagnostics=diagnostics,
            ), diagnostics

        if reachability_ratio >= best_diag.reachability_ratio:
            best_diag = PushPlanningDiagnostics(
                reachability_ratio=reachability_ratio,
                screen_reason=attempt_reason,
                geometry_attempt=attempt_idx,
                contact_lateral_offset=lateral_offset,
                push_distance_used=float(push_distance),
                preview_stage_labels=[label for label, _ in stage_poses],
                preview_stage_terminal_qs=[None if q is None else q.copy() for q in stage_terminal_qs],
            )

    return None, best_diag



def capture_eval_render_frame(demo) -> np.ndarray | None:
    env = getattr(demo, "env", None)
    render_sources = []
    if env is not None:
        render_sources.append(getattr(env, "unwrapped", None))
        render_sources.append(env)

    for source in render_sources:
        if source is None:
            continue
        render_fn = getattr(source, "render_rgb_array", None)
        if not callable(render_fn):
            continue
        try:
            frame = render_fn()
        except TypeError:
            try:
                frame = render_fn(camera_name=None)
            except Exception:
                continue
        except Exception:
            continue
        if frame is None:
            continue
        if hasattr(frame, "detach"):
            try:
                frame = frame.detach()
            except Exception:
                pass
        if hasattr(frame, "cpu"):
            try:
                frame = frame.cpu()
            except Exception:
                pass
        frame_np = np.asarray(frame)
        if frame_np.ndim == 4 and frame_np.shape[0] > 0:
            frame_np = frame_np[0]
        if frame_np.ndim != 3 or frame_np.shape[2] not in (3, 4):
            continue
        if frame_np.shape[2] == 4:
            frame_np = frame_np[:, :, :3]
        if frame_np.dtype != np.uint8:
            if np.issubdtype(frame_np.dtype, np.floating):
                max_value = float(np.nanmax(frame_np)) if frame_np.size > 0 else 0.0
                scale = 255.0 if max_value <= 1.5 else 1.0
                frame_np = np.clip(frame_np * scale, 0.0, 255.0).astype(np.uint8)
            else:
                frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
        return cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    return None



def collect_rollout_trace_sample(demo, stage_label: str, waypoint_index: int) -> RearrangementTraceSample:
    obj_p, obj_q = demo.get_obj_pose()
    frame_bgr = capture_eval_render_frame(demo)
    return RearrangementTraceSample(
        stage_label=str(stage_label),
        waypoint_index=int(waypoint_index),
        obj_p=np.asarray(obj_p, dtype=np.float32).copy(),
        obj_q=np.asarray(obj_q, dtype=np.float32).copy(),
        frame_bgr=None if frame_bgr is None else frame_bgr.copy(),
    )



def get_rearrangement_sim_gripper_value(args) -> float:
    custom_value = getattr(args, "rearrangement_sim_gripper_value", None)
    if custom_value is not None:
        return float(custom_value)
    return float(getattr(args, "sim_gripper_close", 1.0))



def get_rearrangement_real_gripper_value(args) -> float:
    custom_value = getattr(args, "rearrangement_real_gripper_value", None)
    if custom_value is not None:
        return float(custom_value)
    return float(getattr(args, "real_gripper_close", 1.0))



def rearrangement_push_gripper_closed(args) -> bool:
    return bool(get_rearrangement_sim_gripper_value(args) >= 0.0)



def rollout_push_plan_in_sim(demo, plan: PushPlan, args, *, visualize_motion: bool = False, record_trace: bool = False):
    original_sleep = getattr(demo.args, "sim_render_sleep", 0.0)
    original_render_mode = getattr(demo.args, "render_mode", None)
    trace_samples: list[RearrangementTraceSample] = []
    push_gripper_value = get_rearrangement_sim_gripper_value(args)
    try:
        if not visualize_motion:
            demo.args.sim_render_sleep = 0.0
            if original_render_mode == "human":
                demo.args.render_mode = "none"
        demo.hold_current_and_set_gripper(push_gripper_value, steps=12)
        if record_trace:
            trace_samples.append(collect_rollout_trace_sample(demo, "start", -1))
        for stage in plan.stages:
            q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in stage.q_path]
            for idx, q_target in enumerate(q_path):
                hold_steps = int(args.rearrangement_sim_hold_steps) if idx == len(q_path) - 1 else 0
                demo.execute_linear(
                    q_target,
                    gripper_value=push_gripper_value,
                    max_delta_per_step=args.rearrangement_sim_max_delta_per_step,
                    hold_steps=hold_steps,
                    tag=f"{plan.primitive.name}_{stage.label}_{idx:02d}",
                )
                if record_trace:
                    trace_samples.append(collect_rollout_trace_sample(demo, stage.label, idx))
    finally:
        demo.args.sim_render_sleep = original_sleep
        demo.args.render_mode = original_render_mode
    try:
        demo.refresh_runtime_handles(rebuild_visual=False)
    except Exception:
        pass
    real_mod.sync_planner_qpos_from_demo(demo)
    real_mod.update_attached_box_visual(demo)
    return trace_samples



def collect_failed_push_preview_trace(demo, planning_diag: PushPlanningDiagnostics, args) -> list[RearrangementTraceSample]:
    stage_labels = list(getattr(planning_diag, "preview_stage_labels", []) or [])
    stage_terminal_qs = list(getattr(planning_diag, "preview_stage_terminal_qs", []) or [])
    if not stage_labels or not stage_terminal_qs:
        return []

    preview_samples: list[RearrangementTraceSample] = []
    current_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7].copy()
    demo.hold_current_and_set_gripper(get_rearrangement_sim_gripper_value(args), steps=8)
    for stage_idx, (label, q_terminal) in enumerate(zip(stage_labels, stage_terminal_qs)):
        if q_terminal is None:
            preview_samples.append(collect_rollout_trace_sample(demo, f"{label}_fail", stage_idx))
            continue

        q_terminal = np.asarray(q_terminal, dtype=np.float32).reshape(-1)[:7].copy()
        delta = np.abs(q_terminal - current_q)
        interp_steps = int(np.clip(np.ceil(float(delta.max()) / 0.18), 2, 4))
        for waypoint_idx, alpha in enumerate(np.linspace(0.0, 1.0, interp_steps + 1, dtype=np.float32)[1:]):
            q_interp = ((1.0 - alpha) * current_q + alpha * q_terminal).astype(np.float32)
            real_mod.sync_demo_arm_qpos(demo, q_interp)
            try:
                demo.refresh_runtime_handles(rebuild_visual=False)
            except Exception:
                pass
            preview_samples.append(collect_rollout_trace_sample(demo, label, waypoint_idx))
        current_q = q_terminal

    real_mod.sync_planner_qpos_from_demo(demo)
    real_mod.update_attached_box_visual(demo)
    return preview_samples


def safe_metric_gain(current_value: float | None, baseline_value: float | None) -> float:
    current = 0.0 if current_value is None else float(current_value)
    baseline = 0.0 if baseline_value is None else float(baseline_value)
    return current - baseline

def evaluate_post_push_state(initial_obj_pose, final_obj_pose, initial_bottom_z, final_bottom_z, args) -> tuple[bool, str]:
    penetration_tolerance = float(max(getattr(args, "rearrangement_table_penetration_tolerance", 0.0), 0.0))
    if final_bottom_z is not None and final_bottom_z < -penetration_tolerance:
        return False, f"object_penetrates_table(bottom_z={final_bottom_z:.4f})"

    if initial_bottom_z is not None and final_bottom_z is not None:
        bottom_drop = float(initial_bottom_z - final_bottom_z)
        if bottom_drop > float(args.rearrangement_max_object_drop):
            return False, f"object_drop_exceeds_limit(drop={bottom_drop:.4f})"

    center_drop = float(initial_obj_pose[0][2] - final_obj_pose[0][2])
    if center_drop > float(args.rearrangement_max_object_drop):
        return False, f"object_center_drop_exceeds_limit(drop={center_drop:.4f})"

    return True, "valid"



def make_dris_from_scene(demo, args, grasp: GraspReadiness, snapshot, metadata: dict[str, Any] | None = None) -> DRIS:
    obj_p, obj_q = demo.get_obj_pose()
    observation = {
        "object_position": np.asarray(obj_p, dtype=np.float32).copy(),
        "object_quat_wxyz": np.asarray(obj_q, dtype=np.float32).copy(),
        "direct_grasp_feasible": bool(grasp.feasible),
        "pregrasp_feasible": bool(grasp.pregrasp_feasible),
        "direct_grasp_score": float(grasp.score),
    }
    context = {
        "object_name": getattr(args, "object_name", None),
        "selected_obstacle_object_names": list(getattr(args, "selected_obstacle_object_names", []) or []),
    }
    payload = dict(metadata or {})
    payload["snapshot"] = clone_state_tree(snapshot)
    payload["grasp"] = grasp
    return DRIS(
        observation=observation,
        state_space=None,
        context=context,
        representation_type="state",
        metadata=payload,
    )


class JiaobangSerialTSIP(TSIPBase):
    def __init__(self, demo, bridge_mod, args):
        super().__init__(name="JiaobangSerialTSIP")
        self.demo = demo
        self.bridge_mod = bridge_mod
        self.args = args
        self.scoring_args = argparse.Namespace(**vars(args).copy())
        self.scoring_args.render_mode = "none"

    def snapshot(self):
        return snapshot_demo_state(self.demo)

    def next(self, dris: DRIS, action, cage=None):
        if isinstance(action, list):
            return [self.next(dris, item, cage=cage) for item in action]

        primitive = action
        if not isinstance(primitive, PushPrimitive):
            raise TypeError(f"Expected PushPrimitive action, got {type(primitive)!r}")

        base_snapshot = dris.metadata.get("snapshot")
        if base_snapshot is None:
            base_snapshot = self.snapshot()
        baseline_grasp = dris.metadata.get("grasp")

        restore_demo_state(self.demo, base_snapshot)
        initial_obj_pose = tuple(np.asarray(x, dtype=np.float32).copy() for x in self.demo.get_obj_pose())
        initial_bottom_z = real_mod.get_object_bottom_z(self.demo, self.args)
        trace_samples = [collect_rollout_trace_sample(self.demo, "start", -1)] if self.args.rearrangement_export_grid_video else []

        plan, planning_diag = plan_push_primitive_paths(self.demo, primitive, self.args)
        if plan is None:
            reason = planning_diag.screen_reason or "planning_failed"
            if self.args.rearrangement_export_grid_video:
                trace_samples.extend(collect_failed_push_preview_trace(self.demo, planning_diag, self.args))
            grasp = invalidate_grasp_readiness(
                assess_direct_grasp(self.demo, self.bridge_mod, self.scoring_args, preview=False),
                reason,
            )
            failed_dris = make_dris_from_scene(
                self.demo,
                self.args,
                grasp,
                base_snapshot,
                metadata={
                    "primitive": primitive,
                    "plan": None,
                    "reason": reason,
                    "score": -1e9,
                    "initial_obj_pose": initial_obj_pose,
                    "final_obj_pose": initial_obj_pose,
                    "initial_bottom_z": initial_bottom_z,
                    "final_bottom_z": initial_bottom_z,
                    "object_motion_xy": 0.0,
                    "clearance_gain_xy": 0.0,
                    "corridor_clearance_gain_xy": 0.0,
                    "blocker_count_delta": 0,
                    "reachability_ratio": float(planning_diag.reachability_ratio),
                    "screen_reason": reason,
                    "trace_samples": trace_samples,
                },
            )
            restore_demo_state(self.demo, base_snapshot)
            return failed_dris

        trace_samples = rollout_push_plan_in_sim(
            self.demo,
            plan,
            self.args,
            visualize_motion=bool(
                getattr(self.args, "rearrangement_preview_rollout", False)
                and getattr(self.args, "render_mode", None) == "human"
            ),
            record_trace=bool(self.args.rearrangement_export_grid_video),
        )
        real_mod.lift_active_object_above_table_if_needed(
            self.demo,
            self.args,
            min_clearance=max(0.001, 0.5 * float(getattr(self.args, "min_object_center_z_margin", 0.0))),
        )
        final_obj_pose = tuple(np.asarray(x, dtype=np.float32).copy() for x in self.demo.get_obj_pose())
        final_bottom_z = real_mod.get_object_bottom_z(self.demo, self.args)
        grasp = assess_direct_grasp(self.demo, self.bridge_mod, self.scoring_args, preview=False)
        object_motion_xy = float(np.linalg.norm(final_obj_pose[0][:2] - initial_obj_pose[0][:2]))
        clearance_gain_xy = safe_metric_gain(
            grasp.min_obstacle_dist_xy,
            getattr(baseline_grasp, "min_obstacle_dist_xy", None),
        )
        corridor_clearance_gain_xy = safe_metric_gain(
            grasp.approach_corridor_clearance_xy,
            getattr(baseline_grasp, "approach_corridor_clearance_xy", None),
        )
        blocker_count_delta = int(getattr(baseline_grasp, "approach_blocker_count", 0) - grasp.approach_blocker_count)
        is_valid_state, validity_reason = evaluate_post_push_state(
            initial_obj_pose,
            final_obj_pose,
            initial_bottom_z,
            final_bottom_z,
            self.args,
        )
        if not is_valid_state:
            grasp = invalidate_grasp_readiness(grasp, validity_reason)
            score = -1e9
            reason = validity_reason
        else:
            score = float(
                grasp.score
                + clearance_gain_xy * float(self.args.rearrangement_clearance_gain_scale)
                + corridor_clearance_gain_xy * float(self.args.rearrangement_corridor_clearance_scale)
                + blocker_count_delta * float(self.args.rearrangement_blocker_gain_scale)
                + float(planning_diag.reachability_ratio) * float(self.args.rearrangement_reachability_scale)
                - object_motion_xy * float(self.args.rearrangement_motion_penalty_scale)
            )
            reason = grasp.reason
        next_snapshot = self.snapshot()
        next_dris = make_dris_from_scene(
            self.demo,
            self.args,
            grasp,
            next_snapshot,
            metadata={
                "primitive": primitive,
                "plan": plan,
                "reason": reason,
                "score": score,
                "initial_obj_pose": initial_obj_pose,
                "final_obj_pose": final_obj_pose,
                "initial_bottom_z": initial_bottom_z,
                "final_bottom_z": final_bottom_z,
                "object_motion_xy": object_motion_xy,
                "clearance_gain_xy": clearance_gain_xy,
                "corridor_clearance_gain_xy": corridor_clearance_gain_xy,
                "blocker_count_delta": blocker_count_delta,
                "reachability_ratio": float(planning_diag.reachability_ratio),
                "screen_reason": planning_diag.screen_reason,
                "trace_samples": trace_samples,
            },
        )
        restore_demo_state(self.demo, base_snapshot)
        return next_dris



def evaluate_rearrangement_candidate(tsip: JiaobangSerialTSIP, baseline_dris: DRIS, primitive: PushPrimitive) -> RearrangementEvaluation:
    candidate_dris = tsip.next(baseline_dris, primitive)
    metadata = dict(candidate_dris.metadata)
    grasp = metadata["grasp"]
    return RearrangementEvaluation(
        primitive=primitive,
        plan=metadata.get("plan"),
        grasp=grasp,
        score=float(metadata.get("score", -1e9)),
        reason=str(metadata.get("reason", "unknown")),
        initial_obj_pose=metadata.get("initial_obj_pose"),
        final_obj_pose=metadata.get("final_obj_pose"),
        object_motion_xy=float(metadata.get("object_motion_xy", 0.0)),
        clearance_gain_xy=float(metadata.get("clearance_gain_xy", 0.0)),
        corridor_clearance_gain_xy=float(metadata.get("corridor_clearance_gain_xy", 0.0)),
        blocker_count_delta=int(metadata.get("blocker_count_delta", 0)),
        reachability_ratio=float(metadata.get("reachability_ratio", 0.0)),
        screen_reason=str(metadata.get("screen_reason", "unknown")),
        trace_samples=list(metadata.get("trace_samples", []) or []),
    )



def print_search_summary(search: RearrangementSearchResult):
    baseline = search.baseline
    print(
        "[rearrangement] baseline:",
        f"score={baseline.score:.3f}, feasible={baseline.feasible}, pregrasp={baseline.pregrasp_feasible},",
        f"clearance={None if baseline.min_obstacle_dist_xy is None else round(baseline.min_obstacle_dist_xy, 4)},",
        f"corridor={None if baseline.approach_corridor_clearance_xy is None else round(baseline.approach_corridor_clearance_xy, 4)},",
        f"blockers={baseline.approach_blocker_count}, reason={baseline.reason}",
    )
    for evaluation in search.evaluations:
        print(
            "[rearrangement] candidate:",
            evaluation.primitive.name,
            f"score={evaluation.score:.3f}, feasible={evaluation.grasp.feasible},",
            f"pregrasp={evaluation.grasp.pregrasp_feasible}, reach={evaluation.reachability_ratio:.2f},",
            f"clear_gain={evaluation.clearance_gain_xy:+.4f}, corridor_gain={evaluation.corridor_clearance_gain_xy:+.4f},",
            f"blockers={evaluation.blocker_count_delta:+d}, motion_xy={evaluation.object_motion_xy:.4f},",
            f"reason={evaluation.reason}",
        )
    if search.selected is None:
        print("[rearrangement] selected: none")
    else:
        print(
            "[rearrangement] selected:",
            search.selected.primitive.name,
            f"score={search.selected.score:.3f}, reason={search.selected.reason}",
        )



def primitive_short_label(primitive: PushPrimitive) -> str:
    label = primitive.name
    replacements = [
        ("push_pos_x_pos_y", "+x+y"),
        ("push_pos_x_neg_y", "+x-y"),
        ("push_neg_x_pos_y", "-x+y"),
        ("push_neg_x_neg_y", "-x-y"),
        ("push_pos_x", "+x"),
        ("push_neg_x", "-x"),
        ("push_pos_y", "+y"),
        ("push_neg_y", "-y"),
    ]
    for prefix, short in replacements:
        if label.startswith(prefix):
            suffix = label[len(prefix):].lstrip("_")
            return short if not suffix else f"{short}/{suffix}"
    return label



def lookup_object_box_size(object_name: str | None, default_size: np.ndarray | None = None) -> np.ndarray:
    fallback = np.asarray(default_size if default_size is not None else [0.05, 0.05, 0.05], dtype=np.float32).reshape(3)
    if not object_name:
        return fallback
    spec = real_mod.get_object_spec(object_name)
    if spec is None:
        return fallback
    _, sim_asset_scale = real_mod.resolve_object_spec_scales(spec)
    asset_file = str(Path(spec.sim_asset_file or spec.mesh_file).expanduser())
    asset_scale = float(sim_asset_scale)
    try:
        return np.asarray(real_mod.get_asset_box_size(asset_file, asset_scale), dtype=np.float32).reshape(3)
    except Exception:
        return fallback



def quat_to_yaw_wxyz(q_wxyz: np.ndarray) -> float:
    from transforms3d.quaternions import quat2mat

    R = quat2mat(np.asarray(q_wxyz, dtype=np.float64).reshape(4))
    return float(math.atan2(R[1, 0], R[0, 0]))



def oriented_box_xy(center_xy: np.ndarray, yaw: float, size_xy: np.ndarray) -> np.ndarray:
    half = np.asarray(size_xy, dtype=np.float32).reshape(2) * 0.5
    corners = np.asarray(
        [
            [-half[0], -half[1]],
            [half[0], -half[1]],
            [half[0], half[1]],
            [-half[0], half[1]],
        ],
        dtype=np.float32,
    )
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    R = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    return corners @ R.T + np.asarray(center_xy, dtype=np.float32).reshape(1, 2)



def collect_search_xy_bounds(search: RearrangementSearchResult, args) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    for item in search.scene_obstacles:
        T_world_obj = np.asarray(item.get("T_world_obj"), dtype=np.float32)
        if T_world_obj.shape == (4, 4):
            points.append(T_world_obj[:2, 3].astype(np.float32))
    for evaluation in search.evaluations:
        points.append(np.asarray(evaluation.initial_obj_pose[0][:2], dtype=np.float32))
        points.append(np.asarray(evaluation.final_obj_pose[0][:2], dtype=np.float32))
        points.append(pose_world_xyz(evaluation.grasp.pregrasp_pose)[:2])
        points.append(pose_world_xyz(evaluation.grasp.grasp_pose)[:2])
        if evaluation.plan is not None:
            for stage in evaluation.plan.stages:
                points.append(pose_world_xyz(stage.pose)[:2])
        for sample in evaluation.trace_samples:
            points.append(np.asarray(sample.obj_p[:2], dtype=np.float32))
    if not points:
        points.append(np.zeros(2, dtype=np.float32))
    stacked = np.asarray(points, dtype=np.float32)
    min_xy = stacked.min(axis=0)
    max_xy = stacked.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-3)
    margin = float(max(span.max() * 0.2, 0.06))
    return min_xy - margin, max_xy + margin



def world_xy_to_canvas(center_xy: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], image_shape: tuple[int, int, int], pad: int = 18) -> tuple[int, int]:
    min_xy, max_xy = bounds
    h, w = image_shape[:2]
    usable_w = max(w - 2 * pad, 1)
    usable_h = max(h - 2 * pad, 1)
    span = np.maximum(max_xy - min_xy, 1e-6)
    scale = min(usable_w / float(span[0]), usable_h / float(span[1]))
    x = int(round(pad + (float(center_xy[0]) - float(min_xy[0])) * scale))
    y = int(round(h - pad - (float(center_xy[1]) - float(min_xy[1])) * scale))
    return x, y



def draw_world_polyline(canvas: np.ndarray, world_points: list[np.ndarray], bounds: tuple[np.ndarray, np.ndarray], color: tuple[int, int, int], thickness: int = 1):
    if len(world_points) < 2:
        return
    pts = np.asarray([world_xy_to_canvas(point[:2], bounds, canvas.shape) for point in world_points], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, color, thickness, lineType=cv2.LINE_AA)



def draw_world_oriented_box(canvas: np.ndarray, center_xy: np.ndarray, yaw: float, size_xy: np.ndarray, bounds, color, *, filled: bool = False, thickness: int = 2):
    corners_world = oriented_box_xy(center_xy, yaw, size_xy)
    corners_canvas = np.asarray([world_xy_to_canvas(point, bounds, canvas.shape) for point in corners_world], dtype=np.int32).reshape(-1, 1, 2)
    if filled:
        cv2.fillPoly(canvas, [corners_canvas], color, lineType=cv2.LINE_AA)
    else:
        cv2.polylines(canvas, [corners_canvas], True, color, thickness, lineType=cv2.LINE_AA)



def format_reason_text(reason: str, limit: int = 32) -> str:
    reason = str(reason or "")
    return reason if len(reason) <= limit else reason[: limit - 3] + "..."



def fit_frame_to_square(frame_bgr: np.ndarray | None, tile_size: int, *, fallback_color: int = 236) -> np.ndarray:
    if frame_bgr is None:
        return np.full((tile_size, tile_size, 3), fallback_color, dtype=np.uint8)

    frame = np.asarray(frame_bgr, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        return np.full((tile_size, tile_size, 3), fallback_color, dtype=np.uint8)

    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return np.full((tile_size, tile_size, 3), fallback_color, dtype=np.uint8)

    scale = max(float(tile_size) / float(width), float(tile_size) / float(height))
    resized_w = max(int(round(width * scale)), tile_size)
    resized_h = max(int(round(height * scale)), tile_size)
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)
    x0 = max((resized_w - tile_size) // 2, 0)
    y0 = max((resized_h - tile_size) // 2, 0)
    cropped = resized[y0 : y0 + tile_size, x0 : x0 + tile_size]
    if cropped.shape[0] != tile_size or cropped.shape[1] != tile_size:
        padded = np.full((tile_size, tile_size, 3), fallback_color, dtype=np.uint8)
        padded[: cropped.shape[0], : cropped.shape[1]] = cropped
        return padded
    return cropped



def draw_candidate_topdown_overlay(
    canvas: np.ndarray,
    evaluation: RearrangementEvaluation,
    search: RearrangementSearchResult,
    bounds: tuple[np.ndarray, np.ndarray],
    target_box_size: np.ndarray,
    current_sample: RearrangementTraceSample | None,
):
    for obstacle in search.scene_obstacles:
        T_world_obj = np.asarray(obstacle.get("T_world_obj"), dtype=np.float32)
        if T_world_obj.shape != (4, 4):
            continue
        obstacle_size = lookup_object_box_size(obstacle.get("object_name"), np.array([0.05, 0.05, 0.05], dtype=np.float32))
        yaw = float(math.atan2(float(T_world_obj[1, 0]), float(T_world_obj[0, 0])))
        draw_world_oriented_box(
            canvas,
            T_world_obj[:2, 3],
            yaw,
            obstacle_size[:2],
            bounds,
            (190, 190, 190),
            filled=True,
            thickness=1,
        )
        draw_world_oriented_box(canvas, T_world_obj[:2, 3], yaw, obstacle_size[:2], bounds, (120, 120, 120), filled=False, thickness=1)

    corridor_points = [pose_world_xyz(evaluation.grasp.pregrasp_pose), pose_world_xyz(evaluation.grasp.grasp_pose)]
    draw_world_polyline(canvas, corridor_points, bounds, (210, 205, 120), thickness=1)
    if evaluation.plan is not None:
        stage_points = [pose_world_xyz(stage.pose) for stage in evaluation.plan.stages]
        draw_world_polyline(canvas, stage_points, bounds, (90, 170, 220), thickness=1)

    initial_p, initial_q = evaluation.initial_obj_pose
    final_p, final_q = evaluation.final_obj_pose
    draw_world_oriented_box(canvas, initial_p[:2], quat_to_yaw_wxyz(initial_q), target_box_size[:2], bounds, (80, 130, 240), filled=False, thickness=2)
    final_outline_color = (60, 190, 90) if evaluation.grasp.feasible else (80, 80, 220)
    draw_world_oriented_box(canvas, final_p[:2], quat_to_yaw_wxyz(final_q), target_box_size[:2], bounds, final_outline_color, filled=False, thickness=2)
    if current_sample is not None:
        draw_world_oriented_box(canvas, current_sample.obj_p[:2], quat_to_yaw_wxyz(current_sample.obj_q), target_box_size[:2], bounds, (40, 130, 255), filled=True, thickness=2)
    else:
        draw_world_oriented_box(canvas, final_p[:2], quat_to_yaw_wxyz(final_q), target_box_size[:2], bounds, (40, 130, 255), filled=True, thickness=2)



def render_candidate_tile(
    evaluation: RearrangementEvaluation,
    search: RearrangementSearchResult,
    bounds: tuple[np.ndarray, np.ndarray],
    target_box_size: np.ndarray,
    sample_index: int,
    tile_size: int,
) -> np.ndarray:
    current_sample = None
    if evaluation.trace_samples:
        current_sample = evaluation.trace_samples[min(sample_index, len(evaluation.trace_samples) - 1)]

    frame_bgr = None
    if current_sample is not None and current_sample.frame_bgr is not None:
        frame_bgr = current_sample.frame_bgr
    elif search.baseline_frame_bgr is not None:
        frame_bgr = search.baseline_frame_bgr
    tile = fit_frame_to_square(frame_bgr, tile_size)

    overlay = tile.copy()
    top_panel_h = min(max(tile_size // 3, 92), 118)
    bottom_panel_h = min(max(tile_size // 4, 78), 100)
    cv2.rectangle(overlay, (0, 0), (tile_size - 1, top_panel_h), (18, 18, 18), -1)
    cv2.rectangle(overlay, (0, tile_size - bottom_panel_h), (tile_size - 1, tile_size - 1), (18, 18, 18), -1)
    blend_alpha = 0.45 if frame_bgr is not None else 0.12
    tile = cv2.addWeighted(overlay, blend_alpha, tile, 1.0 - blend_alpha, 0.0)

    mini_size = min(max(tile_size // 3, 92), 120)
    mini_map = np.full((mini_size, mini_size, 3), 244, dtype=np.uint8)
    draw_candidate_topdown_overlay(mini_map, evaluation, search, bounds, target_box_size, current_sample)
    mini_y0 = 10
    mini_x0 = tile_size - mini_size - 10
    tile[mini_y0 : mini_y0 + mini_size, mini_x0 : mini_x0 + mini_size] = mini_map
    cv2.rectangle(tile, (mini_x0, mini_y0), (mini_x0 + mini_size - 1, mini_y0 + mini_size - 1), (235, 235, 235), 2)

    if current_sample is not None:
        stage_text = f"stage={current_sample.stage_label}:{current_sample.waypoint_index}"
    else:
        stage_text = "stage=baseline"

    if search.selected is not None and evaluation.primitive.name == search.selected.primitive.name:
        border_color = (60, 180, 75)
    elif evaluation.plan is None:
        border_color = (55, 55, 200)
    else:
        border_color = (95, 95, 95)
    cv2.rectangle(tile, (2, 2), (tile_size - 3, tile_size - 3), border_color, 3)

    title_lines = [
        primitive_short_label(evaluation.primitive),
        stage_text,
    ]
    metric_lines = [
        f"score {evaluation.score:.1f} | feas {int(evaluation.grasp.feasible)} | reach {evaluation.reachability_ratio:.2f}",
        f"clear {evaluation.clearance_gain_xy:+.3f} | cor {evaluation.corridor_clearance_gain_xy:+.3f}",
        f"blk {evaluation.blocker_count_delta:+d} | move {evaluation.object_motion_xy:.3f}",
        format_reason_text(evaluation.reason, limit=52),
    ]

    y = 22
    for line in title_lines:
        cv2.putText(tile, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (245, 245, 245), 1, cv2.LINE_AA)
        y += 21

    y = tile_size - bottom_panel_h + 22
    for idx, line in enumerate(metric_lines):
        color = (245, 245, 245) if idx < 3 else (190, 210, 255)
        cv2.putText(tile, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, color, 1, cv2.LINE_AA)
        y += 18
    return tile



def search_summary_payload(search: RearrangementSearchResult) -> dict[str, Any]:
    return {
        "search_tag": search.search_tag,
        "baseline": {
            "score": float(search.baseline.score),
            "feasible": bool(search.baseline.feasible),
            "pregrasp_feasible": bool(search.baseline.pregrasp_feasible),
            "reason": str(search.baseline.reason),
            "min_obstacle_dist_xy": None if search.baseline.min_obstacle_dist_xy is None else float(search.baseline.min_obstacle_dist_xy),
            "approach_corridor_clearance_xy": None
            if search.baseline.approach_corridor_clearance_xy is None
            else float(search.baseline.approach_corridor_clearance_xy),
            "approach_blocker_count": int(search.baseline.approach_blocker_count),
        },
        "selected": None if search.selected is None else search.selected.primitive.name,
        "candidates": [
            {
                "name": evaluation.primitive.name,
                "score": float(evaluation.score),
                "feasible": bool(evaluation.grasp.feasible),
                "pregrasp_feasible": bool(evaluation.grasp.pregrasp_feasible),
                "reason": str(evaluation.reason),
                "clearance_gain_xy": float(evaluation.clearance_gain_xy),
                "corridor_clearance_gain_xy": float(evaluation.corridor_clearance_gain_xy),
                "blocker_count_delta": int(evaluation.blocker_count_delta),
                "object_motion_xy": float(evaluation.object_motion_xy),
                "reachability_ratio": float(evaluation.reachability_ratio),
                "screen_reason": str(evaluation.screen_reason),
            }
            for evaluation in search.evaluations
        ],
    }



def export_rearrangement_grid_video(search: RearrangementSearchResult, args):
    if not getattr(args, "rearrangement_export_grid_video", False) or not search.evaluations:
        return []

    out_dir = Path(args.rearrangement_video_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target_box_size = np.asarray(real_mod.get_asset_box_size(args.sim_asset_file, args.sim_asset_scale), dtype=np.float32).reshape(3)
    bounds = collect_search_xy_bounds(search, args)
    tile_size = int(max(getattr(args, "rearrangement_video_tile_size", 320), 180))
    fps = int(max(getattr(args, "rearrangement_video_fps", 8), 1))
    sample_repeat = int(max(getattr(args, "rearrangement_video_sample_repeat", 2), 1))
    header_h = 64
    page_paths = []

    payload = search_summary_payload(search)
    summary_path = out_dir / f"{search.search_tag}_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))

    evaluations = sorted(list(search.evaluations), key=candidate_rank_key, reverse=True)
    if len(evaluations) > 16:
        print(f"[rearrangement] video export keeps top 16/{len(evaluations)} candidates sorted by score")
    page = evaluations[:16]
    canvas_w = tile_size * 4
    canvas_h = header_h + tile_size * 4
    video_path = out_dir / f"{search.search_tag}_grid.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (canvas_w, canvas_h),
    )
    if not writer.isOpened():
        print(f"[warn] failed to open rearrangement evaluation video writer: {video_path}")
        return []

    max_samples = max((max(len(item.trace_samples), 1) for item in page), default=1)
    total_frames = max_samples * sample_repeat + fps
    for frame_idx in range(total_frames):
        logical_idx = min(frame_idx // sample_repeat, max_samples - 1)
        frame = np.full((canvas_h, canvas_w, 3), 250, dtype=np.uint8)
        selected_name = None if search.selected is None else search.selected.primitive.name
        header = f"{search.search_tag} | baseline={search.baseline.score:.1f} | selected={selected_name or 'none'} | top {len(page)} by score"
        cv2.putText(frame, header, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            "Each tile shows one push candidate: rendered rollout frames, current score, clutter gains, and final reason.",
            (16, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
        for idx, evaluation in enumerate(page):
            row = idx // 4
            col = idx % 4
            tile = render_candidate_tile(evaluation, search, bounds, target_box_size, logical_idx, tile_size)
            y0 = header_h + row * tile_size
            x0 = col * tile_size
            frame[y0 : y0 + tile_size, x0 : x0 + tile_size] = tile
        writer.write(frame)
    writer.release()
    page_paths.append(video_path)
    print(f"[rearrangement] wrote candidate grid video: {video_path}")
    print(f"[rearrangement] wrote candidate summary JSON: {summary_path}")
    return page_paths


def sim_grasp_summary_payload(evaluations: list[SimGraspEvaluation], selected_label: str | None, grasp_tag: str) -> dict[str, Any]:
    return {
        "grasp_tag": grasp_tag,
        "selected": selected_label,
        "candidates": [
            {
                "name": evaluation.candidate.label,
                "score": float(evaluation.score),
                "feasible": bool(evaluation.feasible),
                "grasped_after_close": bool(evaluation.grasped_after_close),
                "retained_after_lift": bool(evaluation.retained_after_lift),
                "object_lift_z": float(evaluation.object_lift_z),
                "close_gap": None if evaluation.close_gap is None else float(evaluation.close_gap),
                "lift_gap": None if evaluation.lift_gap is None else float(evaluation.lift_gap),
                "reason": str(evaluation.reason),
            }
            for evaluation in evaluations
        ],
    }


def render_sim_grasp_candidate_tile(
    evaluation: SimGraspEvaluation,
    baseline_frame_bgr: np.ndarray | None,
    sample_index: int,
    tile_size: int,
    selected_label: str | None,
) -> np.ndarray:
    current_sample = None
    if evaluation.trace_samples:
        current_sample = evaluation.trace_samples[min(sample_index, len(evaluation.trace_samples) - 1)]

    frame_bgr = None
    if current_sample is not None and current_sample.frame_bgr is not None:
        frame_bgr = current_sample.frame_bgr
    elif baseline_frame_bgr is not None:
        frame_bgr = baseline_frame_bgr
    tile = fit_frame_to_square(frame_bgr, tile_size)

    overlay = tile.copy()
    top_panel_h = min(max(tile_size // 3, 92), 118)
    bottom_panel_h = min(max(tile_size // 4, 78), 104)
    cv2.rectangle(overlay, (0, 0), (tile_size - 1, top_panel_h), (18, 18, 18), -1)
    cv2.rectangle(overlay, (0, tile_size - bottom_panel_h), (tile_size - 1, tile_size - 1), (18, 18, 18), -1)
    blend_alpha = 0.45 if frame_bgr is not None else 0.12
    tile = cv2.addWeighted(overlay, blend_alpha, tile, 1.0 - blend_alpha, 0.0)

    if current_sample is not None:
        stage_text = f"stage={current_sample.stage_label}:{current_sample.waypoint_index}"
    else:
        stage_text = "stage=baseline"

    if selected_label is not None and evaluation.candidate.label == selected_label:
        border_color = (60, 180, 75)
    elif evaluation.retained_after_lift:
        border_color = (70, 150, 220)
    elif evaluation.grasped_after_close:
        border_color = (70, 170, 210)
    elif evaluation.feasible:
        border_color = (120, 120, 120)
    else:
        border_color = (55, 55, 200)
    cv2.rectangle(tile, (2, 2), (tile_size - 3, tile_size - 3), border_color, 3)

    title_lines = [
        evaluation.candidate.label,
        stage_text,
    ]
    metric_lines = [
        f"score {evaluation.score:.1f} | feas {int(evaluation.feasible)}",
        f"close {int(evaluation.grasped_after_close)} | lift {int(evaluation.retained_after_lift)}",
        f"lift_z {evaluation.object_lift_z:+.3f} | gap {('n/a' if evaluation.close_gap is None else f'{evaluation.close_gap:.3f}')}",
        format_reason_text(evaluation.reason, limit=52),
    ]

    y = 22
    for line in title_lines:
        cv2.putText(tile, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (245, 245, 245), 1, cv2.LINE_AA)
        y += 21

    y = tile_size - bottom_panel_h + 22
    for idx, line in enumerate(metric_lines):
        color = (245, 245, 245) if idx < 3 else (190, 210, 255)
        cv2.putText(tile, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, color, 1, cv2.LINE_AA)
        y += 18
    return tile


def export_sim_grasp_grid_video(
    evaluations: list[SimGraspEvaluation],
    baseline_frame_bgr: np.ndarray | None,
    args,
    *,
    grasp_tag: str,
    selected_label: str | None,
):
    if not bool(getattr(args, "sim_grasp_tsip_export_grid_video", False)) or not evaluations:
        return []

    out_dir = Path(args.sim_grasp_tsip_video_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tile_size = int(max(getattr(args, "rearrangement_video_tile_size", 320), 180))
    fps = int(max(getattr(args, "rearrangement_video_fps", 8), 1))
    sample_repeat = int(max(getattr(args, "rearrangement_video_sample_repeat", 2), 1))
    header_h = 64
    page_paths = []

    payload = sim_grasp_summary_payload(evaluations, selected_label, grasp_tag)
    summary_path = out_dir / f"{grasp_tag}_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))

    evaluations_sorted = sorted(
        list(evaluations),
        key=lambda item: (
            float(item.score),
            int(item.retained_after_lift),
            int(item.grasped_after_close),
            float(item.object_lift_z),
        ),
        reverse=True,
    )
    if len(evaluations_sorted) > 16:
        print(f"[grasp-tsip] video export keeps top 16/{len(evaluations_sorted)} candidates sorted by score")
    page = evaluations_sorted[:16]
    canvas_w = tile_size * 4
    canvas_h = header_h + tile_size * 4
    video_path = out_dir / f"{grasp_tag}_grid.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (canvas_w, canvas_h),
    )
    if not writer.isOpened():
        print(f"[warn] failed to open grasp evaluation video writer: {video_path}")
        return []

    max_samples = max((max(len(item.trace_samples), 1) for item in page), default=1)
    total_frames = max_samples * sample_repeat + fps
    for frame_idx in range(total_frames):
        logical_idx = min(frame_idx // sample_repeat, max_samples - 1)
        frame = np.full((canvas_h, canvas_w, 3), 250, dtype=np.uint8)
        header = f"{grasp_tag} | selected={selected_label or 'none'} | top {len(page)} by score"
        cv2.putText(frame, header, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            "Each tile shows one grasp candidate: pregrasp, grasp, close, short lift, and grasp-retention score.",
            (16, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
        for idx, evaluation in enumerate(page):
            row = idx // 4
            col = idx % 4
            tile = render_sim_grasp_candidate_tile(evaluation, baseline_frame_bgr, logical_idx, tile_size, selected_label)
            y0 = header_h + row * tile_size
            x0 = col * tile_size
            frame[y0 : y0 + tile_size, x0 : x0 + tile_size] = tile
        writer.write(frame)
    writer.release()
    page_paths.append(video_path)
    print(f"[grasp-tsip] wrote candidate grid video: {video_path}")
    print(f"[grasp-tsip] wrote candidate summary JSON: {summary_path}")
    return page_paths


def search_best_rearrangement(
    demo,
    bridge_mod,
    args,
    *,
    force_rearrangement: bool = False,
    search_tag: str = "search",
) -> RearrangementSearchResult:
    # Rearrangement search assumes an open hand baseline. Some controller-side
    # gripper targets are not fully restored by physics snapshots alone, so
    # explicitly reopen before taking the search snapshot.
    real_mod.sync_demo_gripper_state(demo, closed=False, steps=4)
    baseline_snapshot = snapshot_demo_state(demo)
    baseline = assess_direct_grasp(demo, bridge_mod, args, preview=False)
    baseline_frame_bgr = capture_eval_render_frame(demo)
    scene_obstacles = clone_state_tree(list(getattr(demo, "scene_obstacles", []) or []))
    baseline_dris = make_dris_from_scene(
        demo,
        args,
        baseline,
        baseline_snapshot,
        metadata={
            "primitive": None,
            "plan": None,
            "reason": baseline.reason,
            "score": baseline.score,
            "initial_obj_pose": tuple(np.asarray(x, dtype=np.float32).copy() for x in demo.get_obj_pose()),
            "final_obj_pose": tuple(np.asarray(x, dtype=np.float32).copy() for x in demo.get_obj_pose()),
            "object_motion_xy": 0.0,
            "clearance_gain_xy": 0.0,
            "corridor_clearance_gain_xy": 0.0,
            "blocker_count_delta": 0,
            "reachability_ratio": 1.0,
            "screen_reason": "baseline",
            "trace_samples": [],
        },
    )

    if baseline.feasible and not force_rearrangement:
        result = RearrangementSearchResult(
            baseline=baseline,
            baseline_dris=baseline_dris,
            evaluations=[],
            selected=None,
            scene_obstacles=scene_obstacles,
            baseline_frame_bgr=None if baseline_frame_bgr is None else baseline_frame_bgr.copy(),
            search_tag=search_tag,
        )
        print_search_summary(result)
        return result

    tsip = JiaobangSerialTSIP(demo, bridge_mod, args)
    evaluations = [evaluate_rearrangement_candidate(tsip, baseline_dris, primitive) for primitive in make_push_primitives(args)]

    if evaluations and not bool(getattr(args, "disable_rearrangement_local_refine", False)):
        topk = max(int(getattr(args, "rearrangement_local_refine_topk", 1)), 0)
        coarse_winners = sorted(evaluations, key=candidate_rank_key, reverse=True)[:topk]
        refined_primitives: list[PushPrimitive] = []
        seen_names = {evaluation.primitive.name for evaluation in evaluations}
        for coarse_eval in coarse_winners:
            for primitive in make_local_refinement_primitives(coarse_eval.primitive, args):
                if primitive.name in seen_names:
                    continue
                seen_names.add(primitive.name)
                refined_primitives.append(primitive)
        if refined_primitives:
            print(
                f"[rearrangement] local refinement: seeding {len(coarse_winners)} coarse winner(s) "
                f"with {len(refined_primitives)} nearby variants"
            )
            evaluations.extend(
                evaluate_rearrangement_candidate(tsip, baseline_dris, primitive)
                for primitive in refined_primitives
            )

    selected = select_rearrangement_candidate(evaluations, baseline, args)

    result = RearrangementSearchResult(
        baseline=baseline,
        baseline_dris=baseline_dris,
        evaluations=evaluations,
        selected=selected,
        scene_obstacles=scene_obstacles,
        baseline_frame_bgr=None if baseline_frame_bgr is None else baseline_frame_bgr.copy(),
        search_tag=search_tag,
    )
    print_search_summary(result)
    restore_demo_state(demo, baseline_snapshot)
    real_mod.sync_demo_gripper_state(demo, closed=False, steps=4)
    export_rearrangement_grid_video(result, args)
    return result

def align_real_robot_to_demo_start(demo, bridge_mod, real_exec, args) -> bool:
    if real_exec is None:
        return True

    start_q = demo.current_arm_qpos()
    print("\n[real robot setup for rearrangement]")
    if args.render_mode == "human":
        bridge_mod.render_preview(demo.env, repeats=5)
    if not real_mod.confirm_simple_action(
        "move real robot to the current simulation start configuration with gripper open before rearrangement",
        args,
        bridge_mod=bridge_mod,
        env=demo.env,
        repeats=8,
    ):
        print("[abort] user cancelled before aligning the real robot for rearrangement")
        return False

    real_exec.set_gripper(args.real_gripper_open)
    real_mod.sync_demo_gripper_state(demo, closed=False, steps=4)
    if not args.no_move_real_to_sim_start:
        q_sent = real_exec.move_linear(
            start_q,
            gripper_pos=args.real_gripper_open,
            max_delta_per_step=args.real_max_delta_per_step,
            hz=args.real_control_hz,
            hold_steps=args.real_hold_steps,
            label="sim_start_rearrangement",
        )
        real_mod.sync_demo_arm_qpos(demo, q_sent)
        if args.render_mode == "human":
            bridge_mod.render_preview(demo.env, repeats=5)
    else:
        print("[real robot setup for rearrangement] skipped moving the real robot to the simulation start pose")
    return True



def execute_rearrangement_plan(demo, bridge_mod, real_exec, evaluation: RearrangementEvaluation, args) -> bool:
    if evaluation.plan is None:
        print(f"[rearrangement] selected primitive {evaluation.primitive.name} has no executable plan")
        return False

    print(f"\n[rearrangement] executing primitive: {evaluation.primitive.name}")
    q_return = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7].copy()
    if real_exec is None:
        if not real_mod.confirm_simple_action(
            f"simulate rearrangement primitive {evaluation.primitive.name}",
            args,
            bridge_mod=bridge_mod,
            env=demo.env,
            repeats=8,
        ):
            print("[abort] user cancelled before simulated rearrangement execution")
            return False
        rollout_push_plan_in_sim(demo, evaluation.plan, args, visualize_motion=True)
        obj_p, obj_q = demo.get_obj_pose()
        print("[rearrangement] post-push object pose:", np.round(obj_p, 6), np.round(obj_q, 6))
        print("[rearrangement] returning arm to the pre-push start configuration")
        demo.execute_linear(
            q_return,
            gripper_value=get_rearrangement_sim_gripper_value(args),
            max_delta_per_step=args.rearrangement_sim_max_delta_per_step,
            hold_steps=int(args.rearrangement_sim_hold_steps),
            tag="rearrangement_return",
        )
        refresh_demo_after_sim_motion(demo)
        real_mod.lift_active_object_above_table_if_needed(
            demo,
            args,
            min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
        )
        return True

    if not align_real_robot_to_demo_start(demo, bridge_mod, real_exec, args):
        return False
    push_gripper_value = get_rearrangement_real_gripper_value(args)
    real_exec.set_gripper(push_gripper_value)
    real_mod.sync_demo_gripper_state(demo, closed=rearrangement_push_gripper_closed(args), steps=4)
    for stage in evaluation.plan.stages:
        ok, _ = real_mod.execute_pose_path_stage(
            demo,
            bridge_mod,
            real_exec,
            f"{evaluation.primitive.name}_{stage.label}",
            stage.pose,
            stage.q_path,
            push_gripper_value,
            args,
        )
        if not ok:
            return False
    q_sent = real_exec.move_linear(
        q_return,
        gripper_pos=push_gripper_value,
        max_delta_per_step=args.real_max_delta_per_step,
        hz=args.real_control_hz,
        hold_steps=args.real_hold_steps,
        label=f"{evaluation.primitive.name}_return",
    )
    real_mod.sync_demo_arm_qpos(demo, q_sent)
    real_mod.recapture_active_object_pose_from_foundationpose(demo, bridge_mod, args)
    return True



def refresh_demo_after_sim_motion(demo):
    try:
        demo.refresh_runtime_handles(rebuild_visual=False)
    except Exception:
        pass
    real_mod.sync_planner_qpos_from_demo(demo)
    real_mod.update_attached_box_visual(demo)



def get_sim_gripper_blocked_after_close(demo, args) -> tuple[float | None, bool | None]:
    gap = real_mod.get_sim_gripper_pad_gap(demo)
    if gap is None:
        return None, None
    blocked = bool(gap > float(args.sim_gripper_blocked_gap_threshold))
    return gap, blocked


def _safe_entity_name_local(entity) -> str:
    if entity is None:
        return "None"
    for attr in ["get_name", "name"]:
        try:
            value = getattr(entity, attr)
            if callable(value):
                value = value()
            if value:
                return str(value)
        except Exception:
            pass
    return entity.__class__.__name__


def get_sim_object_obstacle_contact_names(demo) -> list[str]:
    obj = getattr(getattr(demo, "base_env", None), "obj", None)
    if obj is None:
        return []
    env_unwrapped = getattr(getattr(demo, "env", None), "unwrapped", None)
    obstacle_actors = list(getattr(env_unwrapped, "_scene_obstacle_actors", []) or [])
    if not obstacle_actors:
        return []

    same_entity = getattr(demo, "_same_entity", None)
    get_contacts = getattr(demo, "_get_scene_contacts", None)
    get_contact_actors = getattr(demo, "_get_contact_actors", None)
    if not callable(same_entity) or not callable(get_contacts) or not callable(get_contact_actors):
        return []

    names: list[str] = []
    contacts = list(get_contacts() or [])
    for contact in contacts:
        actor0, actor1 = get_contact_actors(contact)
        for obstacle in obstacle_actors:
            if (same_entity(actor0, obj) and same_entity(actor1, obstacle)) or (same_entity(actor1, obj) and same_entity(actor0, obstacle)):
                names.append(_safe_entity_name_local(obstacle))
                break
    return sorted(set(names))


def execute_sim_linear_guarded(
    demo,
    q_target,
    *,
    gripper_value: float,
    max_delta_per_step: float,
    hold_steps: int,
    tag: str,
    abort_on_object_obstacle_contact: bool = False,
):
    q_start = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    q_target = np.asarray(q_target, dtype=np.float32).reshape(-1)[:7]
    q_delta = (q_target - q_start + np.pi) % (2 * np.pi) - np.pi
    q_target_short = q_start + q_delta
    num_steps = int(np.ceil(np.max(np.abs(q_delta)) / max(float(max_delta_per_step), 1e-6)))
    num_steps = max(1, num_steps)
    print(f"[{tag}] execute_linear num_steps={num_steps}")

    for alpha in np.linspace(0.0, 1.0, num_steps):
        q = (1.0 - alpha) * q_start + alpha * q_target_short
        action = demo.compose_action(q, gripper_value)
        demo.step_and_render(action, tag=tag)
        if abort_on_object_obstacle_contact:
            contact_names = get_sim_object_obstacle_contact_names(demo)
            if contact_names:
                print(
                    f"[FAIL] {tag}: carried object contacted scene obstacle(s) during transport: "
                    + ", ".join(contact_names)
                )
                refresh_demo_after_sim_motion(demo)
                return False

    final_action = demo.compose_action(q_target_short, gripper_value)
    for _ in range(int(max(0, hold_steps))):
        demo.step_and_render(final_action, tag=f"{tag}_hold")
        if abort_on_object_obstacle_contact:
            contact_names = get_sim_object_obstacle_contact_names(demo)
            if contact_names:
                print(
                    f"[FAIL] {tag}: carried object contacted scene obstacle(s) during hold: "
                    + ", ".join(contact_names)
                )
                refresh_demo_after_sim_motion(demo)
                return False

    refresh_demo_after_sim_motion(demo)
    return True


def execute_sim_target_stage(demo, bridge_mod, label: str, pose, q_target, gripper_value: float, args, *, hold_steps: int | None = None):
    q_target = np.asarray(q_target, dtype=np.float32).reshape(-1)[:7]
    if not bridge_mod.confirm_planned_motion(demo, label, pose, q_target, args):
        print(f"[abort] user cancelled before executing {label}")
        return False, None
    stage_hold_steps = int(args.sim_exec_hold_steps if hold_steps is None else hold_steps)
    demo.execute_linear(
        q_target,
        gripper_value=gripper_value,
        max_delta_per_step=args.sim_exec_max_delta_per_step,
        hold_steps=stage_hold_steps,
        tag=label,
    )
    refresh_demo_after_sim_motion(demo)
    return True, q_target



def execute_sim_pose_path_stage(demo, bridge_mod, label: str, pose, q_path, gripper_value: float, args, *, abort_on_object_obstacle_contact: bool = False):
    q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in q_path]
    if not q_path:
        q_current = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
        print(f"[planner] {label} is a zero-length pose path; skipping execution")
        return True, q_current
    if not bridge_mod.confirm_planned_motion(demo, label, pose, q_path[-1], args):
        print(f"[abort] user cancelled before executing {label}")
        return False, None
    for idx, q_target in enumerate(q_path):
        hold_steps = int(args.sim_exec_hold_steps) if idx == len(q_path) - 1 else 0
        ok = execute_sim_linear_guarded(
            demo,
            q_target,
            gripper_value=gripper_value,
            max_delta_per_step=args.sim_exec_max_delta_per_step,
            hold_steps=hold_steps,
            tag=f"{label}_{idx:02d}",
            abort_on_object_obstacle_contact=abort_on_object_obstacle_contact,
        )
        if not ok:
            return False, None
    return True, q_path[-1]



def execute_sim_joint_path_stage(demo, bridge_mod, label: str, q_path, gripper_value: float, args, *, abort_on_object_obstacle_contact: bool = False):
    q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in q_path]
    if not q_path:
        q_current = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
        print(f"[planner] {label} is a zero-length joint path; skipping execution")
        return True, q_current
    if not real_mod.confirm_joint_path_motion(demo, bridge_mod, label, q_path, args):
        print(f"[abort] user cancelled before executing {label}")
        return False, None
    for idx, q_target in enumerate(q_path):
        hold_steps = int(args.sim_exec_hold_steps) if idx == len(q_path) - 1 else 0
        ok = execute_sim_linear_guarded(
            demo,
            q_target,
            gripper_value=gripper_value,
            max_delta_per_step=args.sim_exec_max_delta_per_step,
            hold_steps=hold_steps,
            tag=f"{label}_{idx:02d}",
            abort_on_object_obstacle_contact=abort_on_object_obstacle_contact,
        )
        if not ok:
            return False, None
    return True, q_path[-1]



def execute_sim_gripper_action(demo, bridge_mod, *, closed: bool, label: str, args) -> bool:
    if not real_mod.confirm_simple_action(label, args, bridge_mod=bridge_mod, env=demo.env, repeats=6):
        print(f"[abort] user cancelled before {label}")
        return False
    real_mod.sync_demo_gripper_state(demo, closed=closed, steps=args.sim_gripper_settle_steps)
    refresh_demo_after_sim_motion(demo)
    gap = real_mod.get_sim_gripper_pad_gap(demo)
    if gap is not None:
        print(f"[sim] gripper gap after {'close' if closed else 'open'}: {gap:.4f} m")
    try:
        flags = demo.print_step_flags(prefix='[sim] ')
    except Exception:
        flags = None
    if closed and not args.disable_sim_grasp_check:
        gap, blocked = get_sim_gripper_blocked_after_close(demo, args)
        if gap is not None and blocked is not None:
            print(f"[sim] close proxy: blocked_before_full_close={blocked} gap={gap:.4f} m")
            if not blocked:
                print("[FAIL] simulated gripper fully closed after the close command")
                return False
        else:
            print("[warn] simulated gripper opening proxy is unavailable; skipping the close check")
    if (not closed) and flags is not None:
        print(f"[sim] post-open success={flags.get('success')} is_obj_placed={flags.get('is_obj_placed')}")
    return True



def execute_sim_gripper_close_attempt(demo, bridge_mod, *, label: str, args) -> tuple[bool, bool]:
    if not real_mod.confirm_simple_action(label, args, bridge_mod=bridge_mod, env=demo.env, repeats=6):
        print(f"[abort] user cancelled before {label}")
        return False, False
    real_mod.sync_demo_gripper_state(demo, closed=True, steps=args.sim_gripper_settle_steps)
    refresh_demo_after_sim_motion(demo)
    gap = real_mod.get_sim_gripper_pad_gap(demo)
    if gap is not None:
        print(f"[sim] gripper gap after close: {gap:.4f} m")
    gap, blocked = get_sim_gripper_blocked_after_close(demo, args)
    if gap is not None and blocked is not None:
        print(f"[sim] close proxy: blocked_before_full_close={blocked} gap={gap:.4f} m")
        return True, blocked
    print("[warn] simulated gripper opening proxy is unavailable; treating the close attempt as accepted")
    return True, True



def build_sim_grasp_retry_candidates(grasp_pose, args):
    candidates = []
    seen = set()

    def _pose_key(pose):
        p = tuple(np.round(flatten_np(pose.p)[:3], 5).tolist())
        q = tuple(np.round(flatten_np(pose.q)[:4], 5).tolist())
        return p + q

    def _append(label, pose):
        key = _pose_key(pose)
        if key in seen:
            return
        seen.add(key)
        candidates.append((label, pose))

    z_lifts = [0.004]
    z_lifts.extend(float(v) for v in getattr(args, "grasp_fallback_z_lifts", []) or [] if float(v) > 1e-6)
    backoffs = [float(v) for v in getattr(args, "grasp_fallback_approach_backoffs", []) or [] if float(v) > 1e-6]

    for dz in z_lifts:
        _append(f"grasp_retry_raise_{int(round(dz * 1000))}mm", real_mod.shift_pose_world_z(grasp_pose, dz))
    for backoff in backoffs:
        _append(
            f"grasp_retry_backoff_{int(round(backoff * 1000))}mm",
            real_mod.shift_pose_along_approach(grasp_pose, backoff),
        )
    for dz in z_lifts:
        for backoff in backoffs:
            pose = real_mod.shift_pose_world_z(real_mod.shift_pose_along_approach(grasp_pose, backoff), dz)
            _append(
                f"grasp_retry_raise_{int(round(dz * 1000))}mm_backoff_{int(round(backoff * 1000))}mm",
                pose,
            )

    return candidates


def make_pose_from_pq(p_xyz: np.ndarray, q_wxyz: np.ndarray):
    return real_mod.Pose.create_from_pq(
        p=np.asarray(p_xyz, dtype=np.float32).reshape(3),
        q=np.asarray(q_wxyz, dtype=np.float32).reshape(4),
    )


def shift_pose_local_axis(pose, axis_index: int, distance: float):
    from transforms3d.quaternions import quat2mat

    p = flatten_np(pose.p)[:3].copy()
    q = flatten_np(pose.q)[:4].copy()
    axis = np.asarray(quat2mat(q)[:, int(axis_index)], dtype=np.float32)
    p = p + axis * float(distance)
    return make_pose_from_pq(p, q)


def rotate_pose_about_approach(pose, yaw_deg: float):
    from transforms3d.axangles import axangle2mat
    from transforms3d.quaternions import mat2quat, quat2mat

    p = flatten_np(pose.p)[:3].copy()
    q = flatten_np(pose.q)[:4].copy()
    if abs(float(yaw_deg)) <= 1e-6:
        return make_pose_from_pq(p, q)
    R_pose = quat2mat(q).astype(np.float64)
    approach_axis = np.asarray(R_pose[:, 2], dtype=np.float64)
    R_delta = axangle2mat(approach_axis, math.radians(float(yaw_deg))).astype(np.float64)
    R_new = R_delta @ R_pose
    return make_pose_from_pq(p, np.asarray(mat2quat(R_new), dtype=np.float32))


def build_sim_grasp_candidates(demo, args) -> list[SimGraspCandidate]:
    base_grasp_pose = demo.build_topdown_grasp_pose()
    base_pregrasp_pose = demo.build_pregrasp_pose(base_grasp_pose)
    base_grasp_pose, base_pregrasp_pose, _ = real_mod.enforce_min_grasp_tcp_z(
        base_grasp_pose,
        base_pregrasp_pose,
        args.min_grasp_tcp_z,
    )

    candidates: list[SimGraspCandidate] = []
    seen: set[tuple[float, ...]] = set()

    def _append(label: str, grasp_pose) -> None:
        pregrasp_pose = demo.build_pregrasp_pose(grasp_pose)
        grasp_pose_adj, pregrasp_pose_adj, _ = real_mod.enforce_min_grasp_tcp_z(
            grasp_pose,
            pregrasp_pose,
            args.min_grasp_tcp_z,
        )
        key = tuple(
            np.round(
                np.concatenate([flatten_np(grasp_pose_adj.p)[:3], flatten_np(grasp_pose_adj.q)[:4]]),
                5,
            ).tolist()
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append(SimGraspCandidate(label=label, grasp_pose=grasp_pose_adj, pregrasp_pose=pregrasp_pose_adj))

    _append("grasp_base", base_grasp_pose)

    for dz in [float(v) for v in getattr(args, "sim_grasp_tsip_z_offsets", []) or []]:
        if abs(dz) <= 1e-6:
            continue
        _append(f"grasp_z_{int(round(dz * 1000))}mm", real_mod.shift_pose_world_z(base_grasp_pose, dz))

    backoffs = [0.0]
    backoffs.extend(float(v) for v in getattr(args, "grasp_fallback_approach_backoffs", []) or [] if float(v) > 1e-6)
    for backoff in backoffs:
        if backoff <= 1e-6:
            continue
        _append(
            f"grasp_backoff_{int(round(backoff * 1000))}mm",
            real_mod.shift_pose_along_approach(base_grasp_pose, backoff),
        )

    for lateral in [float(v) for v in getattr(args, "sim_grasp_tsip_lateral_offsets", []) or []]:
        if abs(lateral) <= 1e-6:
            continue
        _append(f"grasp_lat_{int(round(lateral * 1000))}mm", shift_pose_local_axis(base_grasp_pose, 0, lateral))

    for closing in [float(v) for v in getattr(args, "sim_grasp_tsip_closing_offsets", []) or []]:
        if abs(closing) <= 1e-6:
            continue
        _append(f"grasp_closing_{int(round(closing * 1000))}mm", shift_pose_local_axis(base_grasp_pose, 1, closing))

    for yaw_deg in [float(v) for v in getattr(args, "sim_grasp_tsip_yaw_offsets_deg", []) or []]:
        if abs(yaw_deg) <= 1e-6:
            continue
        _append(f"grasp_yaw_{int(round(yaw_deg))}deg", rotate_pose_about_approach(base_grasp_pose, yaw_deg))

    return candidates


def _begin_internal_demo_eval(demo):
    original_render_mode = getattr(demo.args, "render_mode", None)
    original_sleep = getattr(demo.args, "sim_render_sleep", 0.0)
    if original_render_mode == "human":
        demo.args.render_mode = "none"
    demo.args.sim_render_sleep = 0.0
    return original_render_mode, original_sleep


def _end_internal_demo_eval(demo, state) -> None:
    original_render_mode, original_sleep = state
    demo.args.render_mode = original_render_mode
    demo.args.sim_render_sleep = original_sleep


def execute_sim_joint_path_silent(demo, q_path, gripper_value: float, args, *, tag_prefix: str, final_hold_steps: int = 0):
    q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in q_path]
    state = _begin_internal_demo_eval(demo)
    try:
        for idx, q_target in enumerate(q_path):
            hold_steps = int(final_hold_steps) if idx == len(q_path) - 1 else 0
            demo.execute_linear(
                q_target,
                gripper_value=gripper_value,
                max_delta_per_step=args.sim_exec_max_delta_per_step,
                hold_steps=hold_steps,
                tag=f"{tag_prefix}_{idx:02d}",
            )
    finally:
        _end_internal_demo_eval(demo, state)
    refresh_demo_after_sim_motion(demo)


def execute_sim_joint_path_silent_with_trace(
    demo,
    q_path,
    gripper_value: float,
    args,
    *,
    tag_prefix: str,
    stage_label: str,
    trace_samples: list[RearrangementTraceSample] | None,
    final_hold_steps: int = 0,
):
    q_path = [np.asarray(q, dtype=np.float32).reshape(-1)[:7] for q in q_path]
    state = _begin_internal_demo_eval(demo)
    try:
        for idx, q_target in enumerate(q_path):
            hold_steps = int(final_hold_steps) if idx == len(q_path) - 1 else 0
            demo.execute_linear(
                q_target,
                gripper_value=gripper_value,
                max_delta_per_step=args.sim_exec_max_delta_per_step,
                hold_steps=hold_steps,
                tag=f"{tag_prefix}_{idx:02d}",
            )
            if trace_samples is not None:
                trace_samples.append(collect_rollout_trace_sample(demo, stage_label, idx))
    finally:
        _end_internal_demo_eval(demo, state)
    refresh_demo_after_sim_motion(demo)


def evaluate_sim_grasp_candidate(demo, args, candidate: SimGraspCandidate, snapshot) -> SimGraspEvaluation:
    restore_demo_state(demo, snapshot)
    initial_obj_p, _ = demo.get_obj_pose()
    initial_obj_p = np.asarray(initial_obj_p, dtype=np.float32).reshape(3)
    trace_samples = [collect_rollout_trace_sample(demo, "start", -1)] if bool(getattr(args, "sim_grasp_tsip_export_grid_video", False)) else []

    q_pre_path = real_mod.plan_pose_path(
        demo,
        candidate.pregrasp_pose,
        variant_name=args.variant,
        use_attach=False,
        label=f"{candidate.label}_pregrasp_eval",
        planning_time=args.sim_grasp_tsip_planning_time,
        rrt_range=args.sim_grasp_tsip_rrt_range,
    )
    if q_pre_path is None:
        restore_demo_state(demo, snapshot)
        return SimGraspEvaluation(candidate, -1e9, "pregrasp_path_failed", False, False, False, 0.0, None, None, trace_samples=trace_samples)

    execute_sim_joint_path_silent_with_trace(
        demo,
        q_pre_path,
        args.sim_gripper_open,
        args,
        tag_prefix=f"{candidate.label}_pre",
        stage_label="pregrasp",
        trace_samples=trace_samples,
    )

    q_grasp_path = real_mod.plan_pose_path(
        demo,
        candidate.grasp_pose,
        variant_name=args.variant,
        use_attach=False,
        label=f"{candidate.label}_grasp_eval",
        planning_time=args.sim_grasp_tsip_planning_time,
        rrt_range=args.sim_grasp_tsip_rrt_range,
    )
    if q_grasp_path is None:
        restore_demo_state(demo, snapshot)
        return SimGraspEvaluation(candidate, -1e9, "grasp_path_failed", False, False, False, 0.0, None, None, q_pre_path=q_pre_path, trace_samples=trace_samples)

    execute_sim_joint_path_silent_with_trace(
        demo,
        q_grasp_path,
        args.sim_gripper_open,
        args,
        tag_prefix=f"{candidate.label}_grasp",
        stage_label="grasp",
        trace_samples=trace_samples,
        final_hold_steps=int(args.sim_grasp_hold_steps),
    )

    state = _begin_internal_demo_eval(demo)
    try:
        real_mod.sync_demo_gripper_state(demo, closed=True, steps=args.sim_gripper_settle_steps)
    finally:
        _end_internal_demo_eval(demo, state)
    refresh_demo_after_sim_motion(demo)
    if trace_samples:
        trace_samples.append(collect_rollout_trace_sample(demo, "close", 0))
    close_gap, grasped_after_close = get_sim_gripper_blocked_after_close(demo, args)

    q_lift_path: list[np.ndarray] = []
    retained_after_lift = False
    lift_gap = None
    if grasped_after_close:
        lift_pose = real_mod.make_lifted_tcp_pose(demo, args.sim_grasp_tsip_lift_height)
        q_lift_path = real_mod.plan_lift_path(
            demo,
            lift_pose,
            variant_name=args.variant,
            use_attach=False,
            label=f"{candidate.label}_lift_eval",
            planning_time=args.sim_grasp_tsip_planning_time,
            rrt_range=args.sim_grasp_tsip_rrt_range,
            start_q=np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7],
            allow_pose_rrt_fallback=False,
        ) or []
        if q_lift_path:
            execute_sim_joint_path_silent_with_trace(
                demo,
                q_lift_path,
                args.sim_gripper_close,
                args,
                tag_prefix=f"{candidate.label}_lift",
                stage_label="lift",
                trace_samples=trace_samples,
            )
            lift_gap, retained_after_lift = get_sim_gripper_blocked_after_close(demo, args)

    final_obj_p, _ = demo.get_obj_pose()
    final_obj_p = np.asarray(final_obj_p, dtype=np.float32).reshape(3)
    object_lift_z = float(final_obj_p[2] - initial_obj_p[2])

    score = 10.0
    score += 8.0 if q_pre_path else 0.0
    score += 15.0 if q_grasp_path else 0.0
    if grasped_after_close:
        score += 55.0
    if q_lift_path:
        score += 12.0
    if retained_after_lift:
        score += 70.0
    score += max(object_lift_z, 0.0) * 1200.0

    reason_parts = []
    if not grasped_after_close:
        reason_parts.append("close_not_blocked")
    if grasped_after_close and not q_lift_path:
        reason_parts.append("lift_path_failed")
    if q_lift_path and not retained_after_lift:
        reason_parts.append("retention_failed_after_lift")
    if retained_after_lift:
        reason_parts.append("retained_after_lift")
    if not reason_parts:
        reason_parts.append("path_only")

    evaluation = SimGraspEvaluation(
        candidate=candidate,
        score=float(score),
        reason=",".join(reason_parts),
        feasible=bool(q_pre_path and q_grasp_path),
        grasped_after_close=bool(grasped_after_close),
        retained_after_lift=bool(retained_after_lift),
        object_lift_z=float(object_lift_z),
        close_gap=None if close_gap is None else float(close_gap),
        lift_gap=None if lift_gap is None else float(lift_gap),
        q_pre_path=list(q_pre_path),
        q_grasp_path=list(q_grasp_path),
        q_lift_path=list(q_lift_path),
        trace_samples=list(trace_samples),
    )
    restore_demo_state(demo, snapshot)
    return evaluation


def select_best_sim_grasp_candidate(demo, args) -> list[SimGraspEvaluation]:
    snapshot = snapshot_demo_state(demo)
    baseline_frame_bgr = capture_eval_render_frame(demo) if bool(getattr(args, "sim_grasp_tsip_export_grid_video", False)) else None
    candidates = build_sim_grasp_candidates(demo, args)
    print(f"[grasp-tsip] evaluating {len(candidates)} grasp candidates")
    evaluations = [evaluate_sim_grasp_candidate(demo, args, candidate, snapshot) for candidate in candidates]
    evaluations.sort(
        key=lambda item: (
            float(item.score),
            int(item.retained_after_lift),
            int(item.grasped_after_close),
            float(item.object_lift_z),
        ),
        reverse=True,
    )
    for evaluation in evaluations:
        print(
            "[grasp-tsip] candidate:",
            evaluation.candidate.label,
            f"score={evaluation.score:.3f}, feasible={evaluation.feasible},",
            f"close={evaluation.grasped_after_close}, lift={evaluation.retained_after_lift},",
            f"lift_z={evaluation.object_lift_z:+.4f}, reason={evaluation.reason}",
        )
    selected_label = next(
        (
            item.candidate.label
            for item in evaluations
            if item.feasible and item.grasped_after_close
        ),
        None,
    )
    cycle_tag = int(getattr(args, "_cycle_idx", 1))
    object_tag = real_mod.normalize_object_name(getattr(args, "object_name", "object")) or "object"
    grasp_tag = f"cycle{cycle_tag:02d}_grasp_{object_tag}"
    export_sim_grasp_grid_video(
        evaluations,
        baseline_frame_bgr,
        args,
        grasp_tag=grasp_tag,
        selected_label=selected_label,
    )
    return evaluations



def ensure_sim_grasp_retained(demo, label: str, args) -> bool:
    if args.disable_sim_grasp_check:
        return True
    gap, blocked = get_sim_gripper_blocked_after_close(demo, args)
    if gap is None or blocked is None:
        print(f"[warn] simulated gripper opening proxy is unavailable after {label}; skipping the retention check")
        return True
    print(f"[sim] retention proxy after {label}: blocked_before_full_close={blocked} gap={gap:.4f} m")
    if blocked:
        return True
    print(f"[FAIL] simulated gripper fully closed again after {label}")
    return False



def run_fullsim_episode(demo, bridge_mod, args) -> bool:
    print("\n[episode] planning from FoundationPose-initialized object pose")
    print("\n[full-sim] --execute-real was not provided, so motions will be executed inside ManiSkill")
    episode_start_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    real_mod.lift_active_object_above_table_if_needed(
        demo,
        args,
        min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
    )

    if not execute_sim_gripper_action(
        demo,
        bridge_mod,
        closed=False,
        label="open the simulated gripper before starting the episode",
        args=args,
    ):
        return False
    real_mod.lift_active_object_above_table_if_needed(
        demo,
        args,
        min_clearance=max(0.001, 0.5 * float(getattr(args, "min_object_center_z_margin", 0.0))),
    )

    grasp_pose = demo.build_topdown_grasp_pose()
    pregrasp_pose = demo.build_pregrasp_pose(grasp_pose)
    grasp_pose, pregrasp_pose, grasp_tcp_raise = real_mod.enforce_min_grasp_tcp_z(
        grasp_pose,
        pregrasp_pose,
        args.min_grasp_tcp_z,
    )
    if grasp_tcp_raise > 0:
        print(
            f"[safety] raised grasp/pregrasp TCP z by {grasp_tcp_raise:.4f} m "
            f"to satisfy min_grasp_tcp_z={args.min_grasp_tcp_z:.4f}"
        )

    selected_grasp_eval = None
    if not bool(getattr(args, "disable_sim_grasp_tsip_selection", False)):
        grasp_evaluations = select_best_sim_grasp_candidate(demo, args)
        selected_grasp_eval = next((item for item in grasp_evaluations if item.feasible and item.grasped_after_close), None)
        if selected_grasp_eval is not None:
            grasp_pose = selected_grasp_eval.candidate.grasp_pose
            pregrasp_pose = selected_grasp_eval.candidate.pregrasp_pose
            print(
                f"[grasp-tsip] selected candidate: {selected_grasp_eval.candidate.label} "
                f"score={selected_grasp_eval.score:.3f} reason={selected_grasp_eval.reason}"
            )
        else:
            print("[grasp-tsip] no candidate passed close-and-lift screening; falling back to rule-based grasp planning")

    print("\n[poses]")
    print("object p:", np.round(demo.get_obj_pose()[0], 6), "object q:", np.round(demo.get_obj_pose()[1], 6))
    print("tcp grasp p:", np.round(flatten_np(grasp_pose.p)[:3], 6), "q:", np.round(flatten_np(grasp_pose.q)[:4], 6))
    print("tcp pregrasp p:", np.round(flatten_np(pregrasp_pose.p)[:3], 6), "q:", np.round(flatten_np(pregrasp_pose.q)[:4], 6))

    print("\n[move to pregrasp]")
    demo.preview_target_pose(pregrasp_pose)
    if args.render_mode == "human":
        bridge_mod.render_preview(demo.env, repeats=10)
    if selected_grasp_eval is not None and selected_grasp_eval.q_pre_path:
        ok, _ = execute_sim_pose_path_stage(
            demo,
            bridge_mod,
            "pregrasp",
            pregrasp_pose,
            selected_grasp_eval.q_pre_path,
            args.sim_gripper_open,
            args,
        )
    else:
        q_pre = demo.plan_terminal_q(pregrasp_pose, variant_name=args.variant)
        if q_pre is None:
            print("[FAIL] pregrasp planning failed")
            real_mod.inspect_failed_pose(demo, bridge_mod, "pregrasp", args, pose=pregrasp_pose, gripper_closed=False)
            return False
        ok, _ = execute_sim_target_stage(
            demo,
            bridge_mod,
            "pregrasp",
            pregrasp_pose,
            q_pre,
            args.sim_gripper_open,
            args,
            hold_steps=0,
        )
    if not ok:
        return False
    retreat_pregrasp_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]

    print("\n[move to grasp]")
    if selected_grasp_eval is not None and selected_grasp_eval.q_grasp_path:
        grasp_label = selected_grasp_eval.candidate.label
        selected_grasp_pose = selected_grasp_eval.candidate.grasp_pose
        grasp_plan = None
    else:
        grasp_plan = real_mod.plan_grasp_pose_with_fallbacks(demo, bridge_mod, grasp_pose, args)
        if grasp_plan is None:
            print("[FAIL] grasp planning failed")
            real_mod.inspect_failed_pose(demo, bridge_mod, "grasp", args, pose=grasp_pose, gripper_closed=False)
            return False
        grasp_label = grasp_plan.label
        selected_grasp_pose = grasp_plan.final_grasp_pose
        if grasp_label != "grasp":
            print(f"[planner] selected grasp fallback approach: {grasp_label}")
    retreat_pregrasp_pose = demo.build_pregrasp_pose(selected_grasp_pose)
    _, retreat_pregrasp_pose, _ = real_mod.enforce_min_grasp_tcp_z(
        selected_grasp_pose,
        retreat_pregrasp_pose,
        args.min_grasp_tcp_z,
    )
    if selected_grasp_eval is not None and selected_grasp_eval.q_grasp_path:
        ok, _ = execute_sim_pose_path_stage(
            demo,
            bridge_mod,
            grasp_label or "grasp",
            selected_grasp_pose,
            selected_grasp_eval.q_grasp_path,
            args.sim_gripper_open,
            args,
        )
    else:
        if grasp_plan.final_grasp_path is not None:
            approach_path = real_mod.plan_pose_path(
                demo,
                grasp_plan.approach_pose,
                variant_name=args.variant,
                label=grasp_label or "grasp",
            )
            if approach_path is not None:
                ok, _ = execute_sim_pose_path_stage(
                    demo,
                    bridge_mod,
                    grasp_label or "grasp",
                    grasp_plan.approach_pose,
                    approach_path,
                    args.sim_gripper_open,
                    args,
                )
            else:
                print(f"[planner] {grasp_label or 'grasp'} pose-path planning failed; falling back to terminal joint target")
                ok, _ = execute_sim_target_stage(
                    demo,
                    bridge_mod,
                    grasp_label or "grasp",
                    grasp_plan.approach_pose,
                    grasp_plan.approach_q,
                    args.sim_gripper_open,
                    args,
                    hold_steps=0,
                )
            if ok:
                ok, _ = execute_sim_pose_path_stage(
                    demo,
                    bridge_mod,
                    f"{grasp_label}_to_true_grasp",
                    selected_grasp_pose,
                    grasp_plan.final_grasp_path,
                    args.sim_gripper_open,
                    args,
                )
        else:
            ok, _ = execute_sim_target_stage(
                demo,
                bridge_mod,
                grasp_label or "grasp",
                selected_grasp_pose,
                grasp_plan.approach_q,
                args.sim_gripper_open,
                args,
                hold_steps=args.sim_grasp_hold_steps,
            )
    if not ok:
        return False

    print("\n[close gripper]")
    close_ok, grasped = execute_sim_gripper_close_attempt(
        demo,
        bridge_mod,
        label="close the simulated gripper",
        args=args,
    )
    if not close_ok:
        return False
    if not grasped:
        if args.disable_sim_grasp_retries:
            print("[FAIL] simulated gripper fully closed and sim grasp retries are disabled")
            return False
        retry_candidates = build_sim_grasp_retry_candidates(demo.build_topdown_grasp_pose(), args)
        if retry_candidates:
            print("[sim grasp] primary close fully closed; trying pose retries")
        for retry_label, retry_grasp_pose in retry_candidates:
            if not execute_sim_gripper_action(
                demo,
                bridge_mod,
                closed=False,
                label=f"re-open the simulated gripper before {retry_label}",
                args=args,
            ):
                return False
            retry_pregrasp_pose = demo.build_pregrasp_pose(retry_grasp_pose)
            retry_grasp_pose, retry_pregrasp_pose, retry_raise = real_mod.enforce_min_grasp_tcp_z(
                retry_grasp_pose,
                retry_pregrasp_pose,
                args.min_grasp_tcp_z,
            )
            if retry_raise > 0:
                print(
                    f"[sim grasp] {retry_label} raised TCP z by {retry_raise:.4f} m "
                    f"to satisfy min_grasp_tcp_z={args.min_grasp_tcp_z:.4f}"
                )
            print(f"[sim grasp] trying {retry_label}")
            print(f"[sim grasp] {retry_label} pose p:", np.round(flatten_np(retry_grasp_pose.p)[:3], 6))
            print(f"[sim grasp] {retry_label} pose q:", np.round(flatten_np(retry_grasp_pose.q)[:4], 6))

            q_retry_pre = demo.plan_terminal_q(retry_pregrasp_pose, variant_name=args.variant)
            if q_retry_pre is None:
                print(f"[sim grasp] skipped {retry_label}: pregrasp planning failed")
                continue
            ok, _ = execute_sim_target_stage(
                demo,
                bridge_mod,
                f"{retry_label}_pregrasp",
                retry_pregrasp_pose,
                q_retry_pre,
                args.sim_gripper_open,
                args,
                hold_steps=0,
            )
            if not ok:
                return False

            retry_pregrasp_q = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
            q_retry_grasp = demo.plan_terminal_q(retry_grasp_pose, variant_name=args.variant)
            if q_retry_grasp is None:
                print(f"[sim grasp] skipped {retry_label}: grasp planning failed")
                continue
            ok, _ = execute_sim_target_stage(
                demo,
                bridge_mod,
                retry_label,
                retry_grasp_pose,
                q_retry_grasp,
                args.sim_gripper_open,
                args,
                hold_steps=args.sim_grasp_hold_steps,
            )
            if not ok:
                return False

            close_ok, grasped = execute_sim_gripper_close_attempt(
                demo,
                bridge_mod,
                label=f"close the simulated gripper ({retry_label})",
                args=args,
            )
            if not close_ok:
                return False
            if grasped:
                retreat_pregrasp_pose = retry_pregrasp_pose
                retreat_pregrasp_q = retry_pregrasp_q
                break
        if not grasped:
            print("[FAIL] simulated gripper fully closed after all grasp retries")
            return False

    if args.skip_goal_motion:
        print("[done] skipped goal motion as requested")
        return True

    if args.use_sim_goal_motion:
        print("\n[move to goal]")
        goal_tcp_pose, q_goal = demo.refresh_goal_and_plan(announce=True)
        if q_goal is None:
            print("[FAIL] goal planning failed")
            real_mod.inspect_failed_pose(demo, bridge_mod, "goal", args, pose=goal_tcp_pose, gripper_closed=True, use_attach=True)
            return False
        ok, _ = execute_sim_target_stage(demo, bridge_mod, "goal", goal_tcp_pose, q_goal, args.sim_gripper_close, args)
        if not ok:
            return False
        if not ensure_sim_grasp_retained(demo, "goal motion", args):
            return False
        try:
            demo.print_step_flags(prefix='[sim goal] ')
        except Exception:
            pass
        print("[done] completed one simulated goal motion on the current sampled goal pose")
        return True

    print("\n[move to fixed goal joints]")
    q_goal = np.deg2rad(np.asarray(args.fixed_goal_joints_deg, dtype=np.float32))
    target_box_size = real_mod.get_asset_box_size(args.sim_asset_file, args.sim_asset_scale)
    attached_box_scale = float(np.clip(args.transport_attached_box_scale, 0.5, 2.0))
    attached_box_size = np.maximum(target_box_size * attached_box_scale, 1e-4).astype(np.float32)
    attach_pose = real_mod.make_attached_box_pose(demo, attached_box_size)
    fixed_goal_use_attach = not bool(args.fixed_goal_no_attach)
    print(
        "[planner] fixed_goal attached box size:",
        np.round(attached_box_size, 6),
        f"(scale={attached_box_scale:.3f})",
    )
    print(f"[planner] fixed_goal use_attach={fixed_goal_use_attach}, linear_fallback={not args.disable_linear_joint_fallback}")
    try:
        demo.planner.update_attached_box(attached_box_size.tolist(), attach_pose.tolist())
        real_mod.setup_attached_box_visual(demo, demo.env, attached_box_size, attach_pose)
    except Exception as exc:
        print(f"[warn] failed to register attached target box for fixed-goal planning: {exc}")
    prelift_pose = None
    prelift_path = None
    q_start_transport = np.asarray(demo.current_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
    print("[planner] starting fixed-goal planning directly from the current post-grasp arm q")
    if bool(getattr(args, "fixed_goal_rrt_probe_no_attach", False)) and bool(getattr(args, "debug", 0)):
        real_mod.run_fixed_goal_rrt_probe_suite(
            demo,
            q_start_transport,
            q_goal,
            planning_time=args.fixed_goal_planning_time,
            rrt_range=args.fixed_goal_rrt_range,
        )
    q_goal_path = real_mod.plan_joint_path(
        demo,
        q_goal,
        use_attach=fixed_goal_use_attach,
        label="fixed_goal",
        planning_time=args.fixed_goal_planning_time,
        rrt_range=args.fixed_goal_rrt_range,
        start_q=q_start_transport,
        allow_linear_fallback=not args.disable_linear_joint_fallback,
        allow_reverse_rrt_fallback=True,
    )
    lift_pose = None
    lift_path = None
    if q_goal_path is None:
        print("[FAIL] direct fixed-goal transport planning failed from the current post-grasp state")
        real_mod.print_failure_diagnostics(demo, args, "fixed_goal_direct", q_target=q_goal, use_attach=fixed_goal_use_attach)
        return False

    if prelift_path is not None and prelift_pose is not None:
        ok, _ = execute_sim_pose_path_stage(demo, bridge_mod, "pre_transport_lift", prelift_pose, prelift_path, args.sim_gripper_close, args)
        if not ok:
            return False
        if not ensure_sim_grasp_retained(demo, "pre-transport lift", args):
            return False
    ok, _ = execute_sim_joint_path_stage(
        demo,
        bridge_mod,
        "fixed_goal",
        q_goal_path,
        args.sim_gripper_close,
        args,
        abort_on_object_obstacle_contact=True,
    )
    if not ok:
        return False
    if not ensure_sim_grasp_retained(demo, "fixed-goal transport", args):
        return False

    print("\n[open gripper at fixed goal]")
    if not execute_sim_gripper_action(
        demo,
        bridge_mod,
        closed=False,
        label="open the simulated gripper at the fixed goal",
        args=args,
    ):
        return False
    demo._attached_box_visual_visible = False
    demo._attached_object_visual_active = False
    real_mod.update_attached_box_visual(demo, visible=False)
    print("[done] completed one simulated fixed post-grasp joint motion and opened the gripper")
    return True



def run_cycle(args, bridge_mod, planner_mod, real_exec) -> bool:
    env = None
    demo = None
    executed_rearrangement = False
    try:
        env, demo = real_mod.create_demo(args, bridge_mod, planner_mod)
        rearrangement_budget = max(int(args.max_rearrangement_steps), 0)

        for step_idx in range(rearrangement_budget):
            force_current = bool(args.force_rearrangement and step_idx == 0)
            print(f"\n[rearrangement] search step {step_idx + 1}/{rearrangement_budget}")
            cycle_tag = int(getattr(args, '_cycle_idx', 1))
            object_tag = real_mod.normalize_object_name(getattr(args, 'object_name', 'object')) or 'object'
            search_tag = f"cycle{cycle_tag:02d}_step{step_idx + 1:02d}_{object_tag}"
            search = search_best_rearrangement(
                demo,
                bridge_mod,
                args,
                force_rearrangement=force_current,
                search_tag=search_tag,
            )
            baseline_snapshot = search.baseline_dris.metadata.get("snapshot")
            if baseline_snapshot is not None:
                restore_demo_state(demo, baseline_snapshot)
            real_mod.sync_demo_gripper_state(demo, closed=False, steps=4)

            if search.baseline.feasible and not force_current:
                print("[rearrangement] direct grasp already feasible; skipping rearrangement")
                break
            if search.selected is None:
                print("[rearrangement] no beneficial push primitive found; falling back to direct grasp")
                break
            if not execute_rearrangement_plan(demo, bridge_mod, real_exec, search.selected, args):
                return False
            executed_rearrangement = True
            if args.rearrangement_only:
                print("[rearrangement] stopping after rearrangement_only as requested")
                return True

            if real_exec is not None:
                real_mod.close_env_quietly(env)
                env = None
                demo = None
                gc.collect()
                env, demo = real_mod.create_demo(args, bridge_mod, planner_mod)

        if args.rearrangement_only and not executed_rearrangement:
            print("[rearrangement] no rearrangement action was executed; stopping before grasp because rearrangement_only was requested")
            return True
        if real_exec is None:
            return run_fullsim_episode(demo, bridge_mod, args)
        return real_mod.run_real_episode(demo, bridge_mod, real_exec, args)
    finally:
        real_mod.close_env_quietly(env)
        gc.collect()



def main():
    args = parse_args()
    if int(args.repeat_count) < 1:
        raise ValueError("--repeat-count must be >= 1")
    real_mod.maybe_print_and_exit_object_specs(args)
    base_args = argparse.Namespace(**vars(args).copy())

    bridge_mod = real_mod.load_module_from_path("jiaobang_fp_bridge", args.bridge_script_path)
    planner_mod = real_mod.load_module_from_path("jiaobang_planner_impl", args.pick_script_path)

    print(f"Using bridge script: {Path(args.bridge_script_path).resolve()}")
    print(f"Using planner script: {Path(args.pick_script_path).resolve()}")
    print(f"Using ManiDreams root: {Path(args.manidreams_root).expanduser().resolve()}")
    print(f"Using camera extrinsic from: {args.camera_extrinsic_opencv_path}")
    if args.use_direct_camera_extrinsic:
        print("Using camera extrinsic convention: T_base_obj = T_base_cam @ T_cam_obj")
    else:
        print("Using camera extrinsic convention: T_base_obj = inv(camera_extrinsic_opencv) @ T_cam_obj")
    print(
        "Rearrangement mode:",
        f"max_steps={args.max_rearrangement_steps}, push_distance={args.rearrangement_push_distance:.3f},",
        f"force={bool(args.force_rearrangement)}, diagonals={bool(args.rearrangement_include_diagonals)}",
    )
    if args.use_sim_goal_motion:
        print("Post-grasp goal mode: simulation-sampled goal")
    else:
        print(f"Post-grasp goal mode: fixed joint goal (deg) = {args.fixed_goal_joints_deg}")
    if args.repeat_forever:
        print("Repeat mode: forever")
    else:
        print(f"Repeat mode: {args.repeat_count} cycle(s)")

    real_exec = None
    final_ok = True
    cycle_idx = 0
    try:
        if args.execute_real:
            real_exec = real_mod.RealmanJointExecutor(args)
            if args.reset_real_before_start:
                print("\n[real robot pre-reset]")
                if not real_mod.confirm_simple_action(
                    "reset the real robot to its hardware home pose before FoundationPose initialization",
                    args,
                ):
                    print("[abort] user cancelled before the pre-FoundationPose real robot reset")
                    return
                real_exec.reset_robot(gripper_pos=args.real_gripper_open)

        while True:
            cycle_idx += 1
            ok = False
            selected_name = real_mod.prompt_cycle_object_name(base_args, cycle_idx)
            if selected_name is None:
                final_ok = False
                print(f"[abort] user cancelled object selection for cycle {cycle_idx}")
                break
            cycle_args, spec = real_mod.make_cycle_args(base_args, selected_name)
            selected_obstacles = real_mod.prompt_cycle_obstacle_names(cycle_args, cycle_idx, selected_name)
            if selected_obstacles is None:
                final_ok = False
                print(f"[abort] user cancelled obstacle selection for cycle {cycle_idx}")
                break
            cycle_args.selected_obstacle_object_names = list(selected_obstacles)
            cycle_args._cycle_idx = int(cycle_idx)

            print(f"\n[cycle {cycle_idx}] using object spec: {spec.name}")
            print(f"[cycle {cycle_idx}] mesh file: {cycle_args.mesh_file}")
            print(f"[cycle {cycle_idx}] simulation asset file: {cycle_args.sim_asset_file}")
            print(f"[cycle {cycle_idx}] simulation asset scale: {cycle_args.sim_asset_scale}")
            print(f"[cycle {cycle_idx}] GroundingDINO target: {cycle_args.target_object_name}")
            print(f"[cycle {cycle_idx}] selected obstacles: {cycle_args.selected_obstacle_object_names}")
            print(f"\n================ cycle {cycle_idx} ================")

            ok = run_cycle(cycle_args, bridge_mod, planner_mod, real_exec)
            print(f"\ncycle {cycle_idx} success = {ok}")
            if not ok:
                final_ok = False
                break
            if not args.repeat_forever and cycle_idx >= int(args.repeat_count):
                break
            if real_exec is not None:
                print(f"\n[cycle {cycle_idx}] resetting real robot before the next cycle")
                real_exec.reset_robot(gripper_pos=args.real_gripper_open)
            print(f"[cycle {cycle_idx}] ready for the next cycle")

        print("\nfinal success =", final_ok)
    finally:
        if real_exec is not None:
            real_exec.close()


if __name__ == "__main__":
    main()
