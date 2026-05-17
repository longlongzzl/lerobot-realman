from __future__ import annotations

import argparse
import contextlib
import json
import os
import runpy
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PYTHON = os.environ.get("BETA_DEMO_PYTHON", sys.executable)
DEFAULT_SCRIPT = str(Path(__file__).resolve().parent / "plan_multi_wall_path_200step.py")
DEFAULT_OUT_DIR = str(Path(__file__).resolve().parent / "v6_standard_four_wall_retry_build_v1")
DEFAULT_ROLES = "right_wall,back_wall,left_wall,front_wall"
DEFAULT_TORCH_EXTENSIONS_DIR = str(Path(__file__).resolve().parent / ".curobo_torch_extensions")
DEFAULT_CUROBO_ROOT = os.environ.get("BETA_DEMO_CUROBO_ROOT", "/home/zhangzhao/PycharmProjects/curobo")
DEFAULT_ROBOT_CFG = str(Path(__file__).resolve().parent / "curobo_rm75_config" / "rm75.yml")
DEFAULT_LEROBOT_ROOT = "/home/zhangzhao/Desktop/lerobot"
DEFAULT_LEROBOT_SIM2REAL_ROOT = "/home/zhangzhao/Desktop/lerobot/lerobot-sim2real"
OPEN_GRIPPER = -1.0
CLOSED_GRIPPER = 1.0
RM75_HOME_BASE_YAW_DEG = float(os.environ.get("JIMU_RM75_HOME_BASE_YAW_DEG", "90.0"))
RM75_HOME = np.asarray(
    [np.deg2rad(RM75_HOME_BASE_YAW_DEG), 0.0, 0.0, -np.pi / 2.0, 0.0, -np.pi / 2.0, np.pi / 3.0],
    dtype=np.float32,
)
DEFAULT_MAGNET = {
    "drive_force_limit": "13.0",
    "attach_distance": "0.010",
    "attract_distance": "0.024",
    "detach_distance": "0.035",
    "edge_sample_half_span": "0.0185",
    "connect_edge_sample_half_span": "0.0185",
    "edge_sample_offsets": "0.028,0.006",
    "connect_edge_sample_offsets": "0.0185",
    "attract_stiffness": "9.0",
    "attract_force_limit": "4.2",
    "attract_torque_stiffness": "0.36",
    "attract_torque_limit": "0.28",
    "attract_normal_torque_stiffness": "0.88",
    "attract_normal_torque_limit": "0.56",
    "active_stiffness": "1.20",
    "active_damping": "0.28",
    "active_force_limit": "0.82",
    "active_edge_torque_scale": "0.90",
    "active_edge_torque_min_scale": "0.35",
    "active_normal_torque_scale": "1.65",
    "active_normal_torque_min_scale": "0.85",
    "active_torque_delay_steps": "6",
    "active_torque_ramp_steps": "18",
    "active_torque_max_point_error": "0.0",
    "floor_support_score": "10.0",
    "support_connection_score_scale": "1.6",
    "multi_connection_support_bonus": "2.4",
    "drive_stiffness": "165.0",
    "drive_damping": "32.0",
    "drive_angular_stiffness": "0.0",
    "drive_angular_damping": "0.0",
    "drive_angular_force_limit": "0.0",
}
DEFAULT_STAGE_ENV = {
    "JIMU_STAGE_LAYOUT": "right_side_row",
    "JIMU_STAGE_ROW_X_START": "-0.70",
    "JIMU_STAGE_ROW_X_STEP": "0.10",
    "JIMU_STAGE_ROW_Y": "-0.19",
    "JIMU_STAGE_YAW_DEG": "90",
}
FIRST_LAYER_WALL_TARGET_OFFSETS = {
    "right_wall": (0.104, 0.0),
    "back_wall": (0.0, 0.104),
    "left_wall": (-0.104, 0.0),
    "front_wall": (0.0, -0.104),
}
ROBOT_BASE_XY = (-0.615, 0.0)

STRATEGY_PRESET_VALUES = {
    "pair_first_speed_v1": {
        "fast_screen_max_release_candidates": 14,
        "fast_screen_max_grasp_candidates": 14,
        "fast_screen_ik_seeds": 12,
        "fast_chain_top_grasp_candidates": 4,
        "fast_chain_top_release_candidates": 5,
        "fast_chain_ik_seeds": 8,
        "fast_chain_pair_probe_candidates": 2,
        "fast_chain_center_distance_weight": 0.28,
        "fast_chain_release_probe_weight": 0.6,
        "close_steps": 6,
        "open_steps": 6,
        "pre_open_hold_steps": 4,
        "move_steps": 8,
        "short_steps": 4,
        "release_steps": 14,
        "stability_steps": 8,
        "final_all_roles_stability_steps": 10,
        "max_joint_step": 0.095,
        "max_existing_path_waypoints": 36,
        "loaded_state_settle_steps": 4,
        "return_neutral_steps": 6,
        "direct_multi_return_neutral_steps": 10,
        "fallback_conservative_execution": False,
    },
    "pair_first_speed_v2": {
        "fast_screen_max_release_candidates": 10,
        "fast_screen_max_grasp_candidates": 14,
        "fast_screen_ik_seeds": 12,
        "fast_chain_top_grasp_candidates": 4,
        "fast_chain_top_release_candidates": 10,
        "fast_chain_ik_seeds": 8,
        "fast_chain_pair_probe_candidates": 2,
        "close_steps": 6,
        "open_steps": 6,
        "pre_open_hold_steps": 4,
        "move_steps": 8,
        "short_steps": 4,
        "release_steps": 14,
        "stability_steps": 6,
        "final_all_roles_stability_steps": 8,
        "max_joint_step": 0.095,
        "max_existing_path_waypoints": 24,
        "loaded_state_settle_steps": 3,
        "return_neutral_steps": 3,
        "direct_multi_return_neutral_steps": 3,
        "return_neutral_max_joint_step": 0.12,
        "return_neutral_skip_final_role": True,
        "release_score_preplace_joint_weight": 0.005,
        "release_score_place_joint_weight": 0.0015,
        "fallback_conservative_execution": False,
    },
    "pair_first_speed_v3": {
        "fast_screen_max_release_candidates": 10,
        "fast_screen_max_grasp_candidates": 14,
        "fast_screen_ik_seeds": 12,
        "fast_chain_top_grasp_candidates": 4,
        "fast_chain_top_release_candidates": 6,
        "fast_chain_ik_seeds": 8,
        "fast_chain_pair_probe_candidates": 2,
        "close_steps": 6,
        "open_steps": 6,
        "pre_open_hold_steps": 4,
        "move_steps": 8,
        "short_steps": 4,
        "release_steps": 14,
        "stability_steps": 6,
        "final_all_roles_stability_steps": 8,
        "max_joint_step": 0.095,
        "max_existing_path_waypoints": 24,
        "loaded_state_settle_steps": 3,
        "return_neutral_steps": 3,
        "direct_multi_return_neutral_steps": 3,
        "return_neutral_max_joint_step": 0.12,
        "return_neutral_skip_final_role": True,
        "release_score_preplace_joint_weight": 0.005,
        "release_score_place_joint_weight": 0.0015,
        "release_robust_actor_to_tcp_y_offsets": "0.0,0.006,-0.006",
        "release_robust_actor_to_tcp_z_offsets": "0.0,-0.001,-0.002",
        "release_robust_actor_to_tcp_yaw_deg_offsets": "0.0,1.5,-1.5",
        "release_robust_min_variant_successes": 2,
        "release_robust_score_weight": 0.35,
        "use_cuda_graph_batch_ik": False,
        "cuda_graph_batch_ik_max_batch": 128,
        "cuda_graph_batch_ik_fixed_batch_size": 0,
        "free_space_motion_window": False,
        "free_space_pregrasp_max_waypoints": 12,
        "free_space_return_neutral_steps": 3,
        "free_space_return_neutral_max_joint_step": 0.12,
        "next_cycle_plan_prefetch": False,
        "fallback_conservative_execution": False,
    },
    "pair_first_fixed_mixed_v1": {
        "max_release_candidates": 30,
        "fast_screen_max_release_candidates": 30,
        "fast_screen_max_grasp_candidates": 14,
        "fast_screen_ik_seeds": 32,
        "fast_chain_top_grasp_candidates": 4,
        "fast_chain_top_release_candidates": 5,
        "fast_chain_ik_seeds": 32,
        "fast_chain_pair_probe_candidates": 2,
        "close_steps": 6,
        "open_steps": 6,
        "pre_open_hold_steps": 20,
        "move_steps": 8,
        "short_steps": 4,
        "release_steps": 14,
        "stability_steps": 6,
        "final_all_roles_stability_steps": 8,
        "max_joint_step": 0.095,
        "max_existing_path_waypoints": 24,
        "loaded_state_settle_steps": 3,
        "return_neutral_steps": 3,
        "direct_multi_return_neutral_steps": 3,
        "return_neutral_max_joint_step": 0.12,
        "return_neutral_skip_final_role": True,
        "release_score_preplace_joint_weight": 0.005,
        "release_score_place_joint_weight": 0.0015,
        "release_robust_actor_to_tcp_y_offsets": "0.0,0.006,-0.006",
        "release_robust_min_variant_successes": 2,
        "release_robust_score_weight": 0.35,
        "release_robust_require_same_release_index": True,
        "release_screen_max_preplace_joint_delta": 0.0,
        "adaptive_role_order": True,
        "capture_after_open_min_connection_potential": 3,
        "use_cuda_graph_batch_ik": False,
        "cuda_graph_batch_ik_max_batch": 128,
        "cuda_graph_batch_ik_fixed_batch_size": 0,
        "free_space_motion_window": False,
        "free_space_pregrasp_max_waypoints": 12,
        "free_space_return_neutral_steps": 3,
        "free_space_return_neutral_max_joint_step": 0.12,
        "next_cycle_plan_prefetch": False,
        "fallback_conservative_execution": False,
        "ik_seeds": 64,
        "wall_release_profile": "fixed_top_down",
        "fixed_top_down_open_gap": 0.006,
        "fixed_single_connection_release_min_clearance": 0.030,
        "require_active_connection_before_open": False,
    },
    "pair_first_stable_v1": {
        "max_release_candidates": 30,
        "fast_screen_max_release_candidates": 30,
        "fast_screen_max_grasp_candidates": 14,
        "fast_screen_ik_seeds": 32,
        "fast_chain_top_grasp_candidates": 8,
        "fast_chain_top_release_candidates": 5,
        "fast_chain_ik_seeds": 32,
        "fast_chain_pair_probe_candidates": 8,
        "fast_chain_center_distance_weight": 0.28,
        "fast_chain_release_probe_weight": 0.6,
        "close_steps": 6,
        "open_steps": 6,
        "pre_open_hold_steps": 20,
        "move_steps": 8,
        "short_steps": 4,
        "release_steps": 14,
        "stability_steps": 6,
        "final_all_roles_stability_steps": 8,
        "max_joint_step": 0.095,
        "max_existing_path_waypoints": 24,
        "loaded_state_settle_steps": 3,
        "return_neutral_steps": 3,
        "direct_multi_return_neutral_steps": 3,
        "return_neutral_max_joint_step": 0.12,
        "return_neutral_skip_final_role": True,
        "release_score_preplace_joint_weight": 0.005,
        "release_score_place_joint_weight": 0.0015,
        "release_robust_actor_to_tcp_y_offsets": "0.0,0.008,-0.008",
        "release_robust_actor_to_tcp_x_offsets": "0.0,0.008,-0.008",
        "release_robust_actor_to_tcp_z_offsets": "0.0",
        "release_robust_actor_to_tcp_yaw_deg_offsets": "0.0",
        "release_robust_min_variant_successes": 2,
        "release_robust_score_weight": 0.35,
        "release_robust_require_same_release_index": False,
        "release_screen_max_preplace_joint_delta": 0.0,
        "adaptive_role_order": True,
        "capture_after_open_min_connection_potential": 3,
        "use_cuda_graph_batch_ik": False,
        "cuda_graph_batch_ik_max_batch": 128,
        "cuda_graph_batch_ik_fixed_batch_size": 0,
        "free_space_motion_window": False,
        "free_space_pregrasp_max_waypoints": 12,
        "free_space_return_neutral_steps": 3,
        "free_space_return_neutral_max_joint_step": 0.12,
        "next_cycle_plan_prefetch": False,
        "fallback_conservative_execution": False,
        "ik_seeds": 64,
        "wall_release_profile": "fixed_top_down",
        "fixed_top_down_open_gap": 0.006,
        "require_active_connection_before_open": False,
    },
    "pair_first_mined_light_v1": {
        "max_release_candidates": 30,
        "fast_screen_max_release_candidates": 30,
        "fast_screen_max_grasp_candidates": 14,
        "fast_screen_ik_seeds": 24,
        "fast_chain_top_grasp_candidates": 6,
        "fast_chain_top_release_candidates": 4,
        "fast_chain_ik_seeds": 16,
        "fast_chain_pair_probe_candidates": 4,
        "fast_chain_stop_pair_probe_after_success": True,
        "fast_chain_min_successful_pair_probes": 0,
        "fast_chain_center_distance_weight": 0.28,
        "fast_chain_release_probe_weight": 0.6,
        "fast_chain_supported_wall_max_actor_to_tcp_y": 0.0,
        "release_candidate_indices": (
            "right_wall:16;17;15;2;3;5;12;14;1;28,"
            "back_wall:5;16;3;2;17;15;12;14;1;28,"
            "left_wall:16;5;2;17;15;3;12;14;1;28,"
            "front_wall:16;2;5;17;15;3;12;14;1;28"
        ),
        "wall_grasp_prior_mode": "mined_success_v1",
        "wall_grasp_prior_max_candidates": 10,
        "release_robust_actor_to_tcp_y_offsets": "0.0,0.008,-0.008",
        "release_robust_actor_to_tcp_x_offsets": "0.0",
        "release_robust_actor_to_tcp_z_offsets": "0.0",
        "release_robust_actor_to_tcp_yaw_deg_offsets": "0.0",
        "release_robust_min_variant_successes": 1,
        "release_robust_score_weight": 0.0,
        "release_robust_require_same_release_index": False,
        "release_robust_prefer_best_variant": True,
        "release_robust_preferred_actor_to_tcp_y_offset": 0.008,
        "release_robust_require_preferred_actor_to_tcp_y_offset": True,
        "release_robust_require_preferred_max_connection_potential": 0,
        "release_robust_early_stop_on_first_success": True,
        "release_screen_max_preplace_joint_delta": 0.0,
        "release_screen_single_connection_max_place_joint_delta": 3.0,
        "release_screen_max_predicted_actor_position_error": 0.010,
        "release_screen_max_predicted_actor_orientation_error_deg": 8.0,
        "release_prefer_candidate_index_order_for_multi_connection": True,
        "release_ignore_roles_as_collision_exclusions": True,
        "role_order_policy": "given",
        "use_cached_pair_release": True,
        "attach_held_payload_for_release_planning": True,
        "runtime_collision_monitor": True,
        "runtime_collision_monitor_period": 5,
        "runtime_collision_monitor_force_threshold": 0.05,
        "runtime_collision_monitor_max_events": 24,
        "runtime_collision_monitor_include_floor": False,
        "lock_held_actor_after_grasp": True,
        "held_actor_lock_reference": "live",
        "held_actor_unlock_stage": "after_open_gripper",
        "live_actor_to_tcp_after_lift_min_connection_potential": 2,
        "allow_nominal_release_fallback": True,
        "allow_nominal_release_fallback_max_connection_potential": 1,
        "release_prediction_gate_fallback_min_connection_potential": 2,
        "release_motion_plan_on_branch_jump": True,
        "release_motion_plan_on_branch_jump_min_connection_potential": 2,
        "enable_release_yaw_candidates": True,
        "release_open_gripper_value": 0.33,
        "return_home_steps": 10,
        "full_open_after_retreat_steps": 6,
        "wall_post_open_pullback_retreat": True,
        "wall_post_open_pullback_distance": 0.035,
        "wall_post_open_pullback_lift": 0.025,
        "grasp_quality_max_tcp_position_delta": 0.012,
        "grasp_quality_max_tcp_orientation_delta_deg": 12.0,
        "grasp_quality_allow_live_lock_recovery": True,
        "grasp_quality_live_lock_max_tcp_position_delta": 0.018,
        "close_steps": 6,
        "open_steps": 6,
        "pre_open_hold_steps": 20,
        "move_steps": 8,
        "short_steps": 4,
        "release_steps": 14,
        "stability_steps": 6,
        "final_all_roles_stability_steps": 8,
        "max_joint_step": 0.095,
        "max_existing_path_waypoints": 24,
        "loaded_state_settle_steps": 3,
        "return_neutral_steps": 3,
        "direct_multi_return_neutral_steps": 3,
        "return_neutral_max_joint_step": 0.12,
        "return_neutral_skip_final_role": True,
        "release_score_preplace_joint_weight": 0.005,
        "release_score_place_joint_weight": 0.0015,
        "adaptive_role_order": True,
        "capture_after_open_min_connection_potential": 3,
        "use_cuda_graph_batch_ik": False,
        "cuda_graph_batch_ik_max_batch": 128,
        "cuda_graph_batch_ik_fixed_batch_size": 0,
        "free_space_motion_window": False,
        "free_space_pregrasp_max_waypoints": 12,
        "free_space_return_neutral_steps": 3,
        "free_space_return_neutral_max_joint_step": 0.12,
        "next_cycle_plan_prefetch": False,
        "fallback_conservative_execution": False,
        "ik_seeds": 64,
        "wall_release_profile": "fixed_top_down",
        "fixed_top_down_open_gap": 0.006,
        "require_active_connection_before_open": False,
    },
    "pair_first_robust_fast_v1": {
        "max_release_candidates": 30,
        "fast_screen_max_release_candidates": 30,
        "fast_screen_max_grasp_candidates": 14,
        "fast_screen_ik_seeds": 24,
        "fast_chain_top_grasp_candidates": 6,
        "fast_chain_top_release_candidates": 4,
        "fast_chain_ik_seeds": 16,
        "fast_chain_pair_probe_candidates": 4,
        "fast_chain_stop_pair_probe_after_success": True,
        "fast_chain_min_successful_pair_probes": 0,
        "fast_chain_center_distance_weight": 0.28,
        "fast_chain_release_probe_weight": 0.6,
        "fast_chain_supported_wall_max_actor_to_tcp_y": 0.0,
        "release_candidate_indices": (
            "right_wall:16;17;15;2;3;5;12;14;1;28,"
            "back_wall:5;16;3;2;17;15;12;14;1;28,"
            "left_wall:16;5;2;17;15;3;12;14;1;28,"
            "front_wall:16;2;5;17;15;3;12;14;1;28"
        ),
        "wall_grasp_prior_mode": "mined_success_v1",
        "wall_grasp_prior_max_candidates": 10,
        "release_robust_actor_to_tcp_y_offsets": "0.0,0.008,-0.008",
        "release_robust_actor_to_tcp_x_offsets": "0.0",
        "release_robust_actor_to_tcp_z_offsets": "0.0",
        "release_robust_actor_to_tcp_yaw_deg_offsets": "0.0",
        "release_robust_min_variant_successes": 1,
        "release_robust_score_weight": 0.0,
        "release_robust_require_same_release_index": False,
        "release_robust_prefer_best_variant": True,
        "release_robust_preferred_actor_to_tcp_y_offset": 0.008,
        "release_robust_require_preferred_actor_to_tcp_y_offset": True,
        "release_robust_require_preferred_max_connection_potential": 0,
        "release_robust_early_stop_on_first_success": True,
        "release_screen_max_preplace_joint_delta": 3.0,
        "release_screen_single_connection_max_place_joint_delta": 0.0,
        "release_screen_max_predicted_actor_position_error": 0.010,
        "release_screen_max_predicted_actor_orientation_error_deg": 8.0,
        "candidate_screen_max_release_failures": 4,
        "candidate_screen_max_no_robust_failures": 3,
        "release_prefer_candidate_index_order_for_multi_connection": True,
        "release_ignore_roles_as_collision_exclusions": True,
        "role_order_policy": "given",
        "use_cached_pair_release": True,
        "attach_held_payload_for_release_planning": True,
        "runtime_collision_monitor": True,
        "runtime_collision_monitor_period": 5,
        "runtime_collision_monitor_force_threshold": 0.05,
        "runtime_collision_monitor_max_events": 24,
        "runtime_collision_monitor_include_floor": False,
        "lock_held_actor_after_grasp": True,
        "held_actor_lock_reference": "live",
        "held_actor_unlock_stage": "after_open_gripper",
        "live_actor_to_tcp_after_lift_min_connection_potential": 2,
        "allow_nominal_release_fallback": True,
        "allow_nominal_release_fallback_max_connection_potential": 1,
        "release_prediction_gate_fallback_min_connection_potential": 2,
        "release_motion_plan_on_branch_jump": True,
        "release_motion_plan_on_branch_jump_min_connection_potential": 2,
        "enable_release_yaw_candidates": True,
        "release_open_gripper_value": 0.33,
        "return_home_steps": 10,
        "full_open_after_retreat_steps": 6,
        "grasp_quality_max_tcp_position_delta": 0.012,
        "grasp_quality_max_tcp_orientation_delta_deg": 12.0,
        "grasp_quality_allow_live_lock_recovery": True,
        "grasp_quality_live_lock_max_tcp_position_delta": 0.018,
        "close_steps": 6,
        "open_steps": 6,
        "pre_open_hold_steps": 20,
        "move_steps": 8,
        "short_steps": 4,
        "release_steps": 14,
        "stability_steps": 6,
        "final_all_roles_stability_steps": 8,
        "max_joint_step": 0.095,
        "max_existing_path_waypoints": 24,
        "loaded_state_settle_steps": 3,
        "return_neutral_steps": 3,
        "direct_multi_return_neutral_steps": 3,
        "return_neutral_max_joint_step": 0.12,
        "return_neutral_skip_final_role": True,
        "release_score_preplace_joint_weight": 0.005,
        "release_score_place_joint_weight": 0.0015,
        "adaptive_role_order": True,
        "capture_after_open_min_connection_potential": 3,
        "inline_child_planner": True,
        "use_cuda_graph_batch_ik": True,
        "cuda_graph_batch_ik_max_batch": 16,
        "cuda_graph_batch_ik_fixed_batch_size": 16,
        "free_space_motion_window": False,
        "free_space_pregrasp_max_waypoints": 12,
        "free_space_return_neutral_steps": 3,
        "free_space_return_neutral_max_joint_step": 0.12,
        "next_cycle_plan_prefetch": False,
        "fallback_conservative_execution": False,
        "ik_seeds": 64,
        "wall_release_profile": "fixed_top_down",
        "fixed_top_down_open_gap": 0.006,
        "require_active_connection_before_open": False,
    },
    "pair_first_speed_v4": {
        "max_release_candidates": 10,
        "fast_screen_max_release_candidates": 10,
        "fast_screen_max_grasp_candidates": 14,
        "fast_screen_ik_seeds": 12,
        "fast_chain_top_grasp_candidates": 4,
        "fast_chain_top_release_candidates": 4,
        "fast_chain_ik_seeds": 8,
        "fast_chain_pair_probe_candidates": 2,
        "fast_chain_center_distance_weight": 1.0,
        "fast_chain_release_probe_weight": 1.2,
        "close_steps": 6,
        "open_steps": 6,
        "pre_open_hold_steps": 4,
        "move_steps": 8,
        "short_steps": 4,
        "release_steps": 14,
        "stability_steps": 6,
        "final_all_roles_stability_steps": 8,
        "max_joint_step": 0.095,
        "max_existing_path_waypoints": 24,
        "loaded_state_settle_steps": 3,
        "return_neutral_steps": 2,
        "direct_multi_return_neutral_steps": 2,
        "return_neutral_max_joint_step": 0.12,
        "return_neutral_skip_final_role": True,
        "release_score_preplace_joint_weight": 0.005,
        "release_score_place_joint_weight": 0.0015,
        "release_robust_actor_to_tcp_y_offsets": "0.0",
        "release_robust_actor_to_tcp_z_offsets": "0.0",
        "release_robust_actor_to_tcp_yaw_deg_offsets": "0.0",
        "release_robust_min_variant_successes": 1,
        "release_robust_score_weight": 0.0,
        "release_robust_require_same_release_index": False,
        "release_screen_max_preplace_joint_delta": 0.0,
        "capture_after_open_min_connection_potential": 3,
        "use_cuda_graph_batch_ik": True,
        "cuda_graph_batch_ik_max_batch": 8,
        "cuda_graph_batch_ik_fixed_batch_size": 8,
        "free_space_motion_window": True,
        "free_space_pregrasp_max_waypoints": 10,
        "free_space_return_neutral_steps": 2,
        "free_space_return_neutral_max_joint_step": 0.12,
        "next_cycle_plan_prefetch": True,
        "fallback_conservative_execution": False,
        "ik_seeds": 12,
        "wall_release_profile": "generic",
        "pre_open_connection_correction_trigger_pose_error": 0.010,
        "pre_open_connection_correction_trigger_orientation_error_deg": 5.0,
        "adaptive_role_order": True,
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.name}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate unique output directory near {path}")


def _split_roles(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _adaptive_first_layer_roles(args: argparse.Namespace, roles: list[str]) -> list[str]:
    role_order_policy = str(getattr(args, "role_order_policy", "adaptive") or "adaptive").strip().lower()
    if role_order_policy == "given":
        return roles
    if role_order_policy == "far_from_robot":
        expected = {"right_wall", "back_wall", "left_wall", "front_wall"}
        if set(roles) != expected or len(roles) != 4:
            return roles
        dx = float(getattr(args, "initial_assembly_offset_x", 0.0) or 0.0)
        dy = float(getattr(args, "initial_assembly_offset_y", 0.0) or 0.0)
        robot_x, robot_y = ROBOT_BASE_XY
        original_index = {role: index for index, role in enumerate(roles)}

        def distance_key(role: str) -> tuple[float, int]:
            offset_x, offset_y = FIRST_LAYER_WALL_TARGET_OFFSETS[role]
            target_x = dx + offset_x
            target_y = dy + offset_y
            distance_sq = (target_x - robot_x) ** 2 + (target_y - robot_y) ** 2
            return distance_sq, -original_index.get(role, 0)

        return sorted(roles, key=distance_key, reverse=True)
    if not bool(getattr(args, "adaptive_role_order", False)):
        return roles
    expected = {"right_wall", "back_wall", "left_wall", "front_wall"}
    if set(roles) != expected or len(roles) != 4:
        return roles
    dx = float(getattr(args, "initial_assembly_offset_x", 0.0))
    if dx > 0.005:
        return ["left_wall", "back_wall", "right_wall", "front_wall"]
    return roles


def _parser_default_map(parser: argparse.ArgumentParser) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for action in parser._actions:
        if not action.dest or action.dest == argparse.SUPPRESS:
            continue
        defaults[action.dest] = action.default
    return defaults


def _apply_strategy_preset(args: argparse.Namespace, defaults: dict[str, Any]) -> None:
    preset_name = str(getattr(args, "strategy_preset", "baseline") or "baseline")
    preset_values = STRATEGY_PRESET_VALUES.get(preset_name)
    if not preset_values:
        return
    for key, value in preset_values.items():
        if key not in defaults:
            continue
        if getattr(args, key, defaults[key]) == defaults[key]:
            setattr(args, key, value)
    if preset_name == "pair_first_robust_fast_v1":
        dx = float(getattr(args, "initial_assembly_offset_x", 0.0) or 0.0)
        dy = float(getattr(args, "initial_assembly_offset_y", 0.0) or 0.0)
        if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
            args.fast_chain_stop_pair_probe_after_success = False
            args.release_robust_early_stop_on_first_success = False
            args.release_motion_plan_on_branch_jump_min_connection_potential = 1
            if float(getattr(args, "release_screen_max_preplace_joint_delta", 0.0) or 0.0) == 3.0:
                args.release_screen_max_preplace_joint_delta = 4.0
            if int(getattr(args, "fast_screen_ik_seeds", 24) or 24) == 24:
                args.fast_screen_ik_seeds = 64
            if bool(getattr(args, "release_motion_plan_on_branch_jump", False)):
                args.release_motion_plan_on_branch_jump = False
        elif dx > 0.005 and str(getattr(args, "execution_mode", "direct-first") or "direct-first") == "direct-first":
            args.execution_mode = "retry"
        elif dx < -0.001 and int(getattr(args, "fast_screen_ik_seeds", 24) or 24) == 24:
            args.fast_screen_ik_seeds = 64
        if (
            dx > 0.005
            and -0.001 <= dy <= 0.035
            and not bool(getattr(args, "next_cycle_plan_prefetch", False))
        ):
            args.next_cycle_plan_prefetch = True
        if int(getattr(args, "pre_open_hold_steps", 20) or 20) == 20:
            if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
                args.pre_open_hold_steps = 10
            elif dx > 0.005 and abs(dy) <= 0.035:
                args.pre_open_hold_steps = 10
        if int(getattr(args, "wall_grasp_prior_max_candidates", 10) or 10) == 10:
            if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
                args.wall_grasp_prior_max_candidates = 6
        if dx > 0.005 and dy < -0.001 and bool(getattr(args, "use_cuda_graph_batch_ik", False)):
            args.cuda_graph_batch_ik_max_batch = 8
            args.cuda_graph_batch_ik_fixed_batch_size = 8
        if (
            (abs(dx) > 1e-9 or abs(dy) > 1e-9)
            and int(getattr(args, "release_motion_plan_on_branch_jump_min_connection_potential", 1) or 1) == 1
        ):
            args.release_motion_plan_on_branch_jump_min_connection_potential = 3
        if dx > 0.005 and float(getattr(args, "fast_chain_center_distance_weight", 0.28)) == 0.28:
            args.fast_chain_center_distance_weight = 1.0


def _role_supports(role: str, completed: list[str]) -> list[str]:
    neighbors = {
        "right_wall": {"back_wall", "front_wall"},
        "back_wall": {"right_wall", "left_wall"},
        "left_wall": {"back_wall", "front_wall"},
        "front_wall": {"left_wall", "right_wall"},
    }
    supports = ["floor"]
    for item in completed:
        if item in neighbors.get(role, set()) and item not in supports:
            supports.append(item)
    return supports


def _child_args_for_role(args: argparse.Namespace, role: str, completed: list[str]) -> argparse.Namespace:
    role_args = argparse.Namespace(**vars(args))
    return role_args


def _classify_failure(summary: dict[str, Any] | None, returncode: int) -> dict[str, Any]:
    if returncode == 124:
        return {"kind": "attempt_timeout", "returncode": returncode}
    if summary is None:
        return {"kind": "process_or_summary_failure", "returncode": returncode}
    final = summary.get("final", {})
    reports = summary.get("reports", [])
    last_report = reports[-1] if reports else {}
    failed_at = str(last_report.get("failed_at") or "")
    if returncode != 0:
        kind = "process_failed"
    elif final.get("success"):
        kind = "success"
    elif failed_at:
        kind = failed_at
    elif not reports:
        kind = "no_role_report"
    else:
        kind = "final_stability_or_quality_check"
    detail: dict[str, Any] = {
        "kind": kind,
        "returncode": returncode,
        "failed_at": failed_at or None,
        "completed_roles": final.get("completed_roles", []),
        "active_connection_count": None,
        "video": final.get("video"),
    }
    if last_report:
        detail["role"] = last_report.get("role")
        detail["role_success"] = last_report.get("success")
        detail["release_selected"] = (
            last_report.get("release_screen", {}).get("selected")
            if isinstance(last_report.get("release_screen"), dict)
            else None
        )
        detail["active_connection_count"] = last_report.get("active_connection_count")
    all_role_stability = final.get("all_role_stability")
    if isinstance(all_role_stability, dict):
        detail["all_role_stability_success"] = all_role_stability.get("success")
        detail["all_role_stability_roles"] = all_role_stability.get("roles")
    return detail


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}


def _load_realman_base_module() -> Any:
    pick_dir = Path(__file__).resolve().parents[1] / "pick_jiaobang"
    if str(pick_dir) not in sys.path:
        sys.path.insert(0, str(pick_dir))
    import rm75_jiaobang_pick_real_with_foundationpose as real_base

    return real_base


def _sim_gripper_to_real(gripper: float, args: argparse.Namespace) -> float:
    alpha = (float(gripper) - OPEN_GRIPPER) / max(float(CLOSED_GRIPPER - OPEN_GRIPPER), 1e-6)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    open_value = float(getattr(args, "real_gripper_open", 0.0))
    close_value = float(getattr(args, "real_gripper_close", 0.91))
    return float((1.0 - alpha) * open_value + alpha * close_value)


def _densify_joint_path_for_real(path: np.ndarray, max_delta: float) -> np.ndarray:
    path = np.asarray(path, dtype=np.float32).reshape(-1, 7)
    if path.shape[0] <= 1 or max_delta <= 0.0:
        return path
    rows = [path[0]]
    for target in path[1:]:
        previous = rows[-1]
        delta = target - previous
        steps = int(np.ceil(float(np.max(np.abs(delta))) / max(float(max_delta), 1e-6)))
        for index in range(1, max(steps, 1) + 1):
            rows.append((previous + delta * (index / max(steps, 1))).astype(np.float32))
    return np.asarray(rows, dtype=np.float32).reshape(-1, 7)


def _confirm_real_manifest_execution(args: argparse.Namespace, label: str) -> None:
    if bool(getattr(args, "auto_execute", False)):
        return
    try:
        print(
            f"[real] simulation succeeded for {label}. Press Enter to execute this validated trajectory on RM75, "
            "or type q then Enter to abort: ",
            flush=True,
        )
        answer = input().strip().lower()
    except EOFError:
        answer = "q"
    if answer in {"q", "quit", "n", "no"}:
        raise RuntimeError("user aborted before real manifest execution")


def _execute_real_from_summary(
    *,
    args: argparse.Namespace,
    summary_path: Path,
    log_path: Path,
    label: str,
) -> dict[str, Any]:
    summary = _read_json(summary_path)
    if not summary or not bool(summary.get("final", {}).get("success")):
        return {"success": False, "reason": "summary_not_success", "summary_path": str(summary_path)}
    manifest_path = Path(str(summary.get("manifest", ""))).expanduser()
    arrays_path = Path(str(summary.get("arrays", ""))).expanduser()
    manifest = _read_json(manifest_path) if manifest_path.exists() else None
    segments = summary.get("segments")
    if not isinstance(segments, list):
        segments = manifest.get("segments") if isinstance(manifest, dict) else None
    if not isinstance(segments, list):
        return {"success": False, "reason": "missing_segments", "summary_path": str(summary_path)}
    if not arrays_path.exists():
        return {"success": False, "reason": "missing_arrays", "arrays": str(arrays_path)}

    _confirm_real_manifest_execution(args, label)
    _append_log(log_path, f"[{_now()}] real replay start label={label} summary={summary_path}")
    arrays = np.load(arrays_path)
    real_exec = None
    started = time.perf_counter()
    sent_steps = 0
    segment_reports: list[dict[str, Any]] = []

    def send(q: np.ndarray, gripper: float) -> None:
        nonlocal sent_steps
        real_exec.send_action(
            arm_q=np.asarray(q, dtype=np.float32).reshape(-1)[:7],
            gripper_pos=_sim_gripper_to_real(float(gripper), args),
        )
        sent_steps += 1
        hz = float(max(getattr(args, "real_control_hz", 30.0), 1e-3))
        time.sleep(1.0 / hz)

    try:
        real_base = _load_realman_base_module()
        real_exec = real_base.RealmanJointExecutor(args)
        real_exec.set_gripper(float(getattr(args, "real_gripper_open", 0.0)))
        if bool(getattr(args, "reset_real_before_start", True)):
            _append_log(log_path, f"[{_now()}] real replay reset RM75 before label={label}")
            real_exec.reset_robot(gripper_pos=float(getattr(args, "real_gripper_open", 0.0)))
        last_q = np.asarray(real_exec.get_arm_qpos(), dtype=np.float32).reshape(-1)[:7]
        q_delta = (RM75_HOME - last_q + np.pi) % (2.0 * np.pi) - np.pi
        max_start_delta = float(np.max(np.abs(q_delta)))
        max_allowed = float(getattr(args, "real_start_max_delta", 0.12))
        _append_log(log_path, f"[{_now()}] real replay start delta to RM75_HOME={max_start_delta:.4f} rad label={label}")
        if max_allowed > 0.0 and max_start_delta > max_allowed:
            raise RuntimeError(
                f"real robot start is too far from RM75_HOME: max_delta={max_start_delta:.4f} rad > {max_allowed:.4f}"
            )

        max_delta_per_step = float(getattr(args, "real_max_delta_per_step", 0.1))
        for segment in segments:
            seg_type = str(segment.get("type") or "")
            seg_name = str(segment.get("name") or seg_type)
            before_steps = sent_steps
            if seg_type == "joint_path":
                key = str(segment.get("array_key") or "")
                if key not in arrays:
                    raise RuntimeError(f"missing joint path array: {key}")
                path = _densify_joint_path_for_real(np.asarray(arrays[key], dtype=np.float32).reshape(-1, 7), max_delta_per_step)
                gripper = float(segment.get("gripper", OPEN_GRIPPER))
                repeat = max(int(segment.get("action_repeat", 1)), 1)
                for q in path:
                    for _ in range(repeat):
                        send(q, gripper)
                last_q = path[-1, :7].astype(np.float32)
                for _ in range(max(int(segment.get("final_hold", 0)), 0)):
                    send(last_q, gripper)
            elif seg_type == "hold":
                gripper = float(segment.get("gripper", OPEN_GRIPPER))
                for _ in range(max(int(segment.get("steps", 0)), 0)):
                    send(last_q, gripper)
            elif seg_type == "gripper_ramp":
                steps = max(int(segment.get("steps", 0)), 0)
                start_gripper = float(segment.get("start_gripper", OPEN_GRIPPER))
                end_gripper = float(segment.get("end_gripper", OPEN_GRIPPER))
                for gripper in np.linspace(start_gripper, end_gripper, max(steps, 1), dtype=np.float32)[:steps]:
                    send(last_q, float(gripper))
            elif seg_type == "settle_zero_action":
                for _ in range(max(int(segment.get("steps", 0)), 0)):
                    send(last_q, OPEN_GRIPPER)
            else:
                segment_reports.append({"name": seg_name, "type": seg_type, "skipped": True, "reason": "unknown_segment_type"})
                continue
            segment_reports.append(
                {
                    "name": seg_name,
                    "type": seg_type,
                    "sent_steps": int(sent_steps - before_steps),
                }
            )
        return {
            "success": True,
            "label": label,
            "summary_path": str(summary_path),
            "manifest": str(manifest_path),
            "arrays": str(arrays_path),
            "sent_steps": int(sent_steps),
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "segments": segment_reports,
        }
    except Exception as exc:
        _append_log(log_path, f"[{_now()}] real replay failed label={label}: {type(exc).__name__}: {exc}")
        return {
            "success": False,
            "label": label,
            "summary_path": str(summary_path),
            "sent_steps": int(sent_steps),
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
            "segments": segment_reports,
        }
    finally:
        if real_exec is not None:
            try:
                real_exec.set_gripper(float(getattr(args, "real_gripper_open", 0.0)))
            except Exception:
                pass
            try:
                real_exec.close()
            except Exception:
                pass
        _append_log(log_path, f"[{_now()}] real replay finish label={label} sent_steps={sent_steps}")


def _planning_budget_for_attempt(args: argparse.Namespace, role: str, attempt_index: int) -> dict[str, Any]:
    use_fast_screen = attempt_index < max(int(getattr(args, "fast_screen_attempts", 0)), 0)
    if not use_fast_screen:
        return {
            "mode": "full",
            "max_release_candidates": int(args.max_release_candidates),
            "max_grasp_candidates": int(args.max_grasp_candidates),
            "ik_seeds": int(args.ik_seeds),
            "diversify_wall_grasp_candidates": True,
        }
    return {
        "mode": "fast_screen",
        "max_release_candidates": min(int(args.max_release_candidates), int(args.fast_screen_max_release_candidates)),
        "max_grasp_candidates": min(int(args.max_grasp_candidates), int(args.fast_screen_max_grasp_candidates)),
        "ik_seeds": min(int(args.ik_seeds), int(args.fast_screen_ik_seeds)),
        "diversify_wall_grasp_candidates": not (
            role == "front_wall" and bool(getattr(args, "fast_screen_disable_diversified_front_grasp", True))
        ),
    }


def _execution_budget_for_attempt(args: argparse.Namespace, attempt_index: int) -> dict[str, Any]:
    requested_max_joint_step = float(args.max_joint_step)
    if bool(getattr(args, "execute_real", False)):
        real_max_delta = float(getattr(args, "real_max_delta_per_step", requested_max_joint_step) or 0.0)
        if real_max_delta > 0.0:
            requested_max_joint_step = min(requested_max_joint_step, real_max_delta)
    if attempt_index <= 0 or not bool(getattr(args, "fallback_conservative_execution", True)):
        return {
            "mode": "fast",
            "close_steps": int(args.close_steps),
            "open_steps": int(args.open_steps),
            "pre_open_hold_steps": int(args.pre_open_hold_steps),
            "move_steps": int(args.move_steps),
            "short_steps": int(args.short_steps),
            "release_steps": int(args.release_steps),
            "stability_steps": int(args.stability_steps),
            "final_all_roles_stability_steps": int(args.final_all_roles_stability_steps),
            "max_joint_step": requested_max_joint_step,
            "max_segment_steps": int(args.max_segment_steps),
            "max_existing_path_waypoints": int(args.max_existing_path_waypoints),
            "loaded_state_settle_steps": int(args.loaded_state_settle_steps),
        }
    return {
        "mode": "conservative_retry",
        "close_steps": 12,
        "open_steps": 12,
        "pre_open_hold_steps": 16,
        "move_steps": 16,
        "short_steps": 12,
        "release_steps": 28,
        "stability_steps": 20,
        "final_all_roles_stability_steps": 24,
        "max_joint_step": min(0.050, requested_max_joint_step) if requested_max_joint_step > 0.0 else 0.050,
        "max_segment_steps": int(args.max_segment_steps),
        "max_existing_path_waypoints": 0,
        "loaded_state_settle_steps": 40,
    }


def _base_child_args(args: argparse.Namespace, planning_budget: dict[str, Any], execution_budget: dict[str, Any]) -> list[str]:
    child_args = [
        "--robot-cfg",
        str(args.robot_cfg),
        "--curobo-root",
        str(args.curobo_root),
        "--fps",
        str(args.fps),
        "--record-every",
        str(args.record_every),
        "--wall-release-profile",
        str(getattr(args, "wall_release_profile", "generic")),
        "--fixed-top-down-open-gap",
        str(getattr(args, "fixed_top_down_open_gap", 0.0)),
        "--wall-grasp-extra-tilt-degs",
        str(getattr(args, "wall_grasp_extra_tilt_degs", "")),
        "--wall-grasp-extra-tilt-max-abs-deg",
        str(getattr(args, "wall_grasp_extra_tilt_max_abs_deg", 30.0)),
        "--fixed-top-down-extra-tilt-degs",
        str(getattr(args, "fixed_top_down_extra_tilt_degs", "")),
        "--fixed-top-down-extra-tilt-max-abs-deg",
        str(getattr(args, "fixed_top_down_extra_tilt_max_abs_deg", 30.0)),
        "--fixed-top-down-extra-tilt-lift-mms",
        str(getattr(args, "fixed_top_down_extra_tilt_lift_mms", "1.0,2.0")),
        "--fixed-top-down-extra-tilt-normal-bias-mms",
        str(getattr(args, "fixed_top_down_extra_tilt_normal_bias_mms", "0.0,1.5,-1.5")),
        "--fixed-single-connection-release-min-clearance",
        str(getattr(args, "fixed_single_connection_release_min_clearance", 0.0)),
        "--max-release-candidates",
        str(planning_budget["max_release_candidates"]),
        "--max-grasp-candidates",
        str(planning_budget["max_grasp_candidates"]),
        "--ik-seeds",
        str(planning_budget["ik_seeds"]),
        "--ik-position-threshold",
        str(getattr(args, "ik_position_threshold", 0.008)),
        "--ik-rotation-threshold",
        str(getattr(args, "ik_rotation_threshold", 0.12)),
        "--fast-chain-screening" if bool(getattr(args, "fast_chain_screening", True)) else "--no-fast-chain-screening",
        "--fast-chain-top-grasp-candidates",
        str(args.fast_chain_top_grasp_candidates),
        "--fast-chain-top-release-candidates",
        str(args.fast_chain_top_release_candidates),
        "--fast-chain-ik-seeds",
        str(args.fast_chain_ik_seeds),
        "--fast-chain-allow-fallback" if bool(getattr(args, "fast_chain_allow_fallback", True)) else "--no-fast-chain-allow-fallback",
        "--fast-chain-pair-probe-candidates",
        str(getattr(args, "fast_chain_pair_probe_candidates", 0)),
        "--fast-chain-stop-pair-probe-after-success"
        if bool(getattr(args, "fast_chain_stop_pair_probe_after_success", False))
        else "--no-fast-chain-stop-pair-probe-after-success",
        "--fast-chain-min-successful-pair-probes",
        str(getattr(args, "fast_chain_min_successful_pair_probes", 0)),
        "--fast-chain-center-distance-weight",
        str(getattr(args, "fast_chain_center_distance_weight", 0.28)),
        "--fast-chain-release-probe-weight",
        str(getattr(args, "fast_chain_release_probe_weight", 0.6)),
        "--fast-chain-supported-wall-max-actor-to-tcp-y",
        str(getattr(args, "fast_chain_supported_wall_max_actor_to_tcp_y", 0.0)),
        "--use-cuda-graph-batch-ik" if bool(getattr(args, "use_cuda_graph_batch_ik", False)) else "--no-use-cuda-graph-batch-ik",
        "--cuda-graph-batch-ik-max-batch",
        str(getattr(args, "cuda_graph_batch_ik_max_batch", 128)),
        "--cuda-graph-batch-ik-fixed-batch-size",
        str(getattr(args, "cuda_graph_batch_ik_fixed_batch_size", 0)),
        "--close-steps",
        str(execution_budget["close_steps"]),
        "--open-steps",
        str(execution_budget["open_steps"]),
        "--pre-open-hold-steps",
        str(execution_budget["pre_open_hold_steps"]),
        "--move-steps",
        str(execution_budget["move_steps"]),
        "--short-steps",
        str(execution_budget["short_steps"]),
        "--release-steps",
        str(execution_budget["release_steps"]),
        "--stability-steps",
        str(execution_budget["stability_steps"]),
        "--final-all-roles-stability-steps",
        str(execution_budget["final_all_roles_stability_steps"]),
        "--max-joint-step",
        str(execution_budget["max_joint_step"]),
        "--max-segment-steps",
        str(execution_budget["max_segment_steps"]),
        "--max-existing-path-waypoints",
        str(execution_budget["max_existing_path_waypoints"]),
        "--free-space-motion-window" if bool(getattr(args, "free_space_motion_window", False)) else "--no-free-space-motion-window",
        "--free-space-pregrasp-max-waypoints",
        str(getattr(args, "free_space_pregrasp_max_waypoints", 12)),
        "--free-space-return-neutral-steps",
        str(getattr(args, "free_space_return_neutral_steps", 4)),
        "--free-space-return-neutral-max-joint-step",
        str(getattr(args, "free_space_return_neutral_max_joint_step", 0.12)),
        "--grasp-max-joint-delta",
        str(args.grasp_max_joint_delta),
        "--release-max-joint-delta",
        str(args.release_max_joint_delta),
        "--release-score-preplace-joint-weight",
        str(args.release_score_preplace_joint_weight),
        "--release-score-place-joint-weight",
        str(args.release_score_place_joint_weight),
        "--release-motion-plan-preplace" if bool(getattr(args, "release_motion_plan_preplace", False)) else "--no-release-motion-plan-preplace",
        "--release-motion-plan-on-branch-jump" if bool(getattr(args, "release_motion_plan_on_branch_jump", False)) else "--no-release-motion-plan-on-branch-jump",
        "--release-motion-plan-on-branch-jump-min-connection-potential",
        str(getattr(args, "release_motion_plan_on_branch_jump_min_connection_potential", 0)),
        "--release-motion-plan-timeout",
        str(getattr(args, "release_motion_plan_timeout", 4.0)),
        "--release-motion-plan-max-waypoints",
        str(getattr(args, "release_motion_plan_max_waypoints", 24)),
        "--enable-release-yaw-candidates" if bool(getattr(args, "enable_release_yaw_candidates", False)) else "--no-enable-release-yaw-candidates",
        "--release-candidate-indices",
        str(getattr(args, "release_candidate_indices", "")),
        (
            "--release-prefer-candidate-index-order-for-multi-connection"
            if bool(getattr(args, "release_prefer_candidate_index_order_for_multi_connection", False))
            else "--no-release-prefer-candidate-index-order-for-multi-connection"
        ),
        "--release-robust-actor-to-tcp-y-offsets",
        str(getattr(args, "release_robust_actor_to_tcp_y_offsets", "0.0")),
        "--release-robust-actor-to-tcp-x-offsets",
        str(getattr(args, "release_robust_actor_to_tcp_x_offsets", "0.0")),
        "--release-robust-actor-to-tcp-z-offsets",
        str(getattr(args, "release_robust_actor_to_tcp_z_offsets", "0.0")),
        "--release-robust-actor-to-tcp-yaw-deg-offsets",
        str(getattr(args, "release_robust_actor_to_tcp_yaw_deg_offsets", "0.0")),
        "--release-robust-min-variant-successes",
        str(getattr(args, "release_robust_min_variant_successes", 1)),
        "--release-robust-score-weight",
        str(getattr(args, "release_robust_score_weight", 0.0)),
        (
            "--release-robust-require-same-release-index"
            if bool(getattr(args, "release_robust_require_same_release_index", False))
            else "--no-release-robust-require-same-release-index"
        ),
        (
            "--release-robust-prefer-best-variant"
            if bool(getattr(args, "release_robust_prefer_best_variant", False))
            else "--no-release-robust-prefer-best-variant"
        ),
        "--release-robust-preferred-actor-to-tcp-y-offset",
        str(getattr(args, "release_robust_preferred_actor_to_tcp_y_offset", 0.0)),
        (
            "--release-robust-require-preferred-actor-to-tcp-y-offset"
            if bool(getattr(args, "release_robust_require_preferred_actor_to_tcp_y_offset", False))
            else "--no-release-robust-require-preferred-actor-to-tcp-y-offset"
        ),
        "--release-robust-require-preferred-max-connection-potential",
        str(getattr(args, "release_robust_require_preferred_max_connection_potential", 0)),
        (
            "--release-robust-early-stop-on-first-success"
            if bool(getattr(args, "release_robust_early_stop_on_first_success", False))
            else "--no-release-robust-early-stop-on-first-success"
        ),
        "--release-screen-max-preplace-joint-delta",
        str(getattr(args, "release_screen_max_preplace_joint_delta", 0.0)),
        "--release-screen-single-connection-max-place-joint-delta",
        str(getattr(args, "release_screen_single_connection_max_place_joint_delta", 0.0)),
        "--release-screen-max-predicted-actor-position-error",
        str(getattr(args, "release_screen_max_predicted_actor_position_error", 0.0)),
        "--release-screen-max-predicted-actor-orientation-error-deg",
        str(getattr(args, "release_screen_max_predicted_actor_orientation_error_deg", 0.0)),
        "--candidate-screen-max-release-failures",
        str(getattr(args, "candidate_screen_max_release_failures", 0)),
        "--candidate-screen-max-no-robust-failures",
        str(getattr(args, "candidate_screen_max_no_robust_failures", 0)),
        "--release-prediction-gate-fallback-min-connection-potential",
        str(getattr(args, "release_prediction_gate_fallback_min_connection_potential", 0)),
        (
            "--release-ignore-roles-as-collision-exclusions"
            if bool(getattr(args, "release_ignore_roles_as_collision_exclusions", True))
            else "--no-release-ignore-roles-as-collision-exclusions"
        ),
        "--role-order-policy",
        str(getattr(args, "role_order_policy", "adaptive")),
        (
            "--use-cached-pair-release"
            if bool(getattr(args, "use_cached_pair_release", False))
            else "--no-use-cached-pair-release"
        ),
        "--attach-held-payload-for-release-planning" if bool(getattr(args, "attach_held_payload_for_release_planning", False)) else "--no-attach-held-payload-for-release-planning",
        "--runtime-collision-monitor" if bool(getattr(args, "runtime_collision_monitor", False)) else "--no-runtime-collision-monitor",
        "--runtime-collision-monitor-period",
        str(getattr(args, "runtime_collision_monitor_period", 5)),
        "--runtime-collision-monitor-force-threshold",
        str(getattr(args, "runtime_collision_monitor_force_threshold", 0.05)),
        "--runtime-collision-monitor-max-events",
        str(getattr(args, "runtime_collision_monitor_max_events", 24)),
        "--runtime-collision-monitor-include-floor" if bool(getattr(args, "runtime_collision_monitor_include_floor", False)) else "--no-runtime-collision-monitor-include-floor",
        "--lock-held-actor-after-grasp" if bool(getattr(args, "lock_held_actor_after_grasp", False)) else "--no-lock-held-actor-after-grasp",
        "--held-actor-lock-reference",
        str(getattr(args, "held_actor_lock_reference", "live")),
        "--held-actor-unlock-stage",
        str(getattr(args, "held_actor_unlock_stage", None) or "before_open_gripper"),
        "--live-actor-to-tcp-after-lift" if bool(getattr(args, "use_live_actor_to_tcp_after_lift", True)) else "--no-live-actor-to-tcp-after-lift",
        "--live-actor-to-tcp-after-lift-min-connection-potential",
        str(getattr(args, "live_actor_to_tcp_after_lift_min_connection_potential", 0)),
        "--allow-nominal-release-fallback" if bool(getattr(args, "allow_nominal_release_fallback", False)) else "--no-allow-nominal-release-fallback",
        "--allow-nominal-release-fallback-max-connection-potential",
        str(getattr(args, "allow_nominal_release_fallback_max_connection_potential", 0)),
        "--enable-capture-after-open-roles",
        str(getattr(args, "enable_capture_after_open_roles", "")),
        "--capture-after-open-min-connection-potential",
        str(getattr(args, "capture_after_open_min_connection_potential", 0)),
        "--grasp-quality-max-tcp-position-delta",
        str(args.grasp_quality_max_tcp_position_delta),
        "--grasp-quality-max-tcp-orientation-delta-deg",
        str(args.grasp_quality_max_tcp_orientation_delta_deg),
        "--grasp-quality-min-finger-force",
        str(args.grasp_quality_min_finger_force),
        "--grasp-quality-max-force-balance-error",
        str(args.grasp_quality_max_force_balance_error),
        "--grasp-quality-allow-live-lock-recovery"
        if bool(getattr(args, "grasp_quality_allow_live_lock_recovery", False))
        else "--no-grasp-quality-allow-live-lock-recovery",
        "--grasp-quality-live-lock-max-tcp-position-delta",
        str(getattr(args, "grasp_quality_live_lock_max_tcp_position_delta", 0.0)),
        "--wall-grasp-prior-mode",
        str(getattr(args, "wall_grasp_prior_mode", "none")),
        "--wall-grasp-prior-max-candidates",
        str(getattr(args, "wall_grasp_prior_max_candidates", 0)),
        "--loaded-state-settle-steps",
        str(execution_budget["loaded_state_settle_steps"]),
        "--wall-release-approach-mode",
        str(getattr(args, "wall_release_approach_mode", "auto")),
        "--wall-release-side-approach-distance",
        str(args.wall_release_side_approach_distance),
        "--wall-release-side-approach-lift",
        str(args.wall_release_side_approach_lift),
        "--disable-held-role-magnets-until-release",
        "--disable-unbuilt-role-magnets",
        "--anchor-floor-during-build",
        "--magnet-attach-distance",
        str(args.magnet_attach_distance),
        "--magnet-attract-distance",
        str(args.magnet_attract_distance),
        "--magnet-detach-distance",
        str(args.magnet_detach_distance),
        "--magnet-edge-sample-half-span",
        str(args.magnet_edge_sample_half_span),
        "--magnet-connect-edge-sample-half-span",
        str(args.magnet_connect_edge_sample_half_span),
        "--magnet-edge-sample-offsets",
        str(args.magnet_edge_sample_offsets),
        "--magnet-connect-edge-sample-offsets",
        str(args.magnet_connect_edge_sample_offsets),
        "--magnet-attract-stiffness",
        str(args.magnet_attract_stiffness),
        "--magnet-attract-force-limit",
        str(args.magnet_attract_force_limit),
        "--magnet-attract-torque-stiffness",
        str(args.magnet_attract_torque_stiffness),
        "--magnet-attract-torque-limit",
        str(args.magnet_attract_torque_limit),
        "--magnet-attract-normal-torque-stiffness",
        str(args.magnet_attract_normal_torque_stiffness),
        "--magnet-attract-normal-torque-limit",
        str(args.magnet_attract_normal_torque_limit),
        "--magnet-active-stiffness",
        str(args.magnet_active_stiffness),
        "--magnet-active-damping",
        str(args.magnet_active_damping),
        "--magnet-active-force-limit",
        str(args.magnet_active_force_limit),
        "--magnet-active-edge-torque-scale",
        str(args.magnet_active_edge_torque_scale),
        "--magnet-active-edge-torque-min-scale",
        str(args.magnet_active_edge_torque_min_scale),
        "--magnet-active-normal-torque-scale",
        str(args.magnet_active_normal_torque_scale),
        "--magnet-active-normal-torque-min-scale",
        str(args.magnet_active_normal_torque_min_scale),
        "--magnet-active-torque-delay-steps",
        str(args.magnet_active_torque_delay_steps),
        "--magnet-active-torque-ramp-steps",
        str(args.magnet_active_torque_ramp_steps),
        "--magnet-active-torque-max-point-error",
        str(args.magnet_active_torque_max_point_error),
        "--magnet-floor-support-score",
        str(args.magnet_floor_support_score),
        "--magnet-support-connection-score-scale",
        str(args.magnet_support_connection_score_scale),
        "--magnet-multi-connection-support-bonus",
        str(args.magnet_multi_connection_support_bonus),
        "--magnet-drive-stiffness",
        str(args.magnet_drive_stiffness),
        "--magnet-drive-damping",
        str(args.magnet_drive_damping),
        "--magnet-drive-force-limit",
        str(args.magnet_drive_force_limit),
        "--magnet-drive-angular-stiffness",
        str(args.magnet_drive_angular_stiffness),
        "--magnet-drive-angular-damping",
        str(args.magnet_drive_angular_damping),
        "--magnet-drive-angular-force-limit",
        str(args.magnet_drive_angular_force_limit),
        "--max-position-error",
        str(args.max_position_error),
        "--max-orientation-error-deg",
        str(args.max_orientation_error_deg),
        "--all-roles-max-position-error",
        str(args.all_roles_max_position_error),
        "--all-roles-max-orientation-error-deg",
        str(args.all_roles_max_orientation_error_deg),
        "--all-roles-max-drift-position",
        str(args.all_roles_max_drift_position),
        "--all-roles-max-drift-orientation-deg",
        str(args.all_roles_max_drift_orientation_deg),
        "--all-roles-max-linear-speed",
        str(args.all_roles_max_linear_speed),
        "--all-roles-max-angular-speed",
        str(args.all_roles_max_angular_speed),
        "--final-max-connection-point-error",
        str(args.final_max_connection_point_error),
        "--final-max-connection-normal-angle-deg",
        str(args.final_max_connection_normal_angle_deg),
        "--final-max-connection-edge-angle-deg",
        str(args.final_max_connection_edge_angle_deg),
        "--release-open-gripper-value",
        str(args.release_open_gripper_value),
        "--return-home-steps",
        str(args.return_home_steps),
        "--full-open-after-retreat-steps",
        str(args.full_open_after_retreat_steps),
        "--pre-open-min-active-connections",
        str(args.pre_open_min_active_connections),
        "--pre-open-connection-correction-attempts",
        str(args.pre_open_connection_correction_attempts),
        "--pre-open-connection-correction-max-pose-error",
        str(args.pre_open_connection_correction_max_pose_error),
        "--pre-open-connection-correction-max-orientation-error-deg",
        str(args.pre_open_connection_correction_max_orientation_error_deg),
        "--pre-open-connection-correction-trigger-pose-error",
        str(args.pre_open_connection_correction_trigger_pose_error),
        "--pre-open-connection-correction-trigger-orientation-error-deg",
        str(args.pre_open_connection_correction_trigger_orientation_error_deg),
        "--require-active-connection-before-open"
        if bool(args.require_active_connection_before_open)
        else "--no-require-active-connection-before-open",
        "--release-correction-attempts",
        str(args.release_correction_attempts),
        "--next-cycle-plan-prefetch" if bool(getattr(args, "next_cycle_plan_prefetch", False)) else "--no-next-cycle-plan-prefetch",
        (
            "--return-neutral-motion-plan"
            if bool(getattr(args, "return_neutral_motion_plan", True))
            else "--no-return-neutral-motion-plan"
        ),
        "--return-neutral-motion-plan-timeout",
        str(getattr(args, "return_neutral_motion_plan_timeout", 2.5)),
        "--return-neutral-motion-plan-max-attempts",
        str(getattr(args, "return_neutral_motion_plan_max_attempts", 1)),
        (
            "--return-neutral-motion-plan-enable-graph"
            if bool(getattr(args, "return_neutral_motion_plan_enable_graph", True))
            else "--no-return-neutral-motion-plan-enable-graph"
        ),
        "--return-neutral-motion-plan-graph-seeds",
        str(getattr(args, "return_neutral_motion_plan_graph_seeds", 1)),
        "--return-neutral-motion-plan-max-waypoints",
        str(getattr(args, "return_neutral_motion_plan_max_waypoints", 0)),
    ]
    if bool(getattr(args, "record_live", True)):
        child_args.insert(0, "--record-live")
    child_args.extend(["--render-mode", str(getattr(args, "render_mode", "none") or "none")])
    child_args.extend(["--human-render-every", str(max(int(getattr(args, "human_render_every", 1) or 1), 1))])
    if getattr(args, "wait_before_close", None) is not None:
        child_args.append("--wait-before-close" if bool(args.wait_before_close) else "--no-wait-before-close")
    if bool(getattr(args, "return_neutral_after_role", True)):
        child_args.extend(["--return-neutral-after-role", "--return-neutral-steps", str(args.return_neutral_steps)])
        child_args.extend(["--return-neutral-max-joint-step", str(args.return_neutral_max_joint_step)])
        if bool(getattr(args, "return_neutral_skip_final_role", False)):
            child_args.append("--return-neutral-skip-final-role")
    if bool(getattr(args, "enable_pregrasp_graph_recovery", True)):
        child_args.extend(["--pregrasp-enable-graph", "--pregrasp-graph-fallback-only"])
    if float(getattr(args, "initial_actor_jitter_xy", 0.0) or 0.0) > 0.0:
        child_args.extend(
            [
                "--initial-actor-jitter-xy",
                str(args.initial_actor_jitter_xy),
                "--initial-actor-jitter-seed",
                str(args.initial_actor_jitter_seed),
                "--initial-actor-jitter-roles",
                str(args.initial_actor_jitter_roles),
                "--initial-actor-jitter-min-start-distance",
                str(args.initial_actor_jitter_min_start_distance),
                "--initial-actor-jitter-min-target-distance",
                str(args.initial_actor_jitter_min_target_distance),
                "--initial-actor-jitter-max-sample-attempts",
                str(args.initial_actor_jitter_max_sample_attempts),
            ]
        )
    return child_args


def _candidate_schedule(
    role: str,
    attempt_index: int,
    completed: list[str] | None = None,
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    schedule = [
        {
            "name": "generic_any_auto",
            "grasp_axis": "any",
            "release_index": -1,
            "restore_robot_qpos": False,
            "release_max_joint_delta": None,
            "preplace_height": None,
            "return_neutral": False,
        },
        {
            "name": "generic_x_auto",
            "grasp_axis": "x",
            "release_index": -1,
            "restore_robot_qpos": False,
            "release_max_joint_delta": None,
            "preplace_height": None,
            "return_neutral": False,
        },
        {
            "name": "generic_y_auto",
            "grasp_axis": "y",
            "release_index": -1,
            "restore_robot_qpos": False,
            "release_max_joint_delta": None,
            "preplace_height": None,
            "return_neutral": False,
        },
        {
            "name": "generic_any_exact_target",
            "grasp_axis": "any",
            "release_index": 0,
            "restore_robot_qpos": False,
            "release_max_joint_delta": None,
            "preplace_height": None,
            "return_neutral": False,
        },
        {
            "name": "generic_x_exact_target",
            "grasp_axis": "x",
            "release_index": 0,
            "restore_robot_qpos": False,
            "release_max_joint_delta": None,
            "preplace_height": None,
            "return_neutral": False,
        },
        {
            "name": "generic_any_auto_wider_release",
            "grasp_axis": "any",
            "release_index": -1,
            "restore_robot_qpos": False,
            "release_max_joint_delta": 2.8,
            "preplace_height": None,
            "return_neutral": False,
        },
        {
            "name": "generic_x_auto_restore_checkpoint_q",
            "grasp_axis": "x",
            "release_index": -1,
            "restore_robot_qpos": True,
            "release_max_joint_delta": None,
            "preplace_height": None,
            "return_neutral": False,
        },
        {
            "name": "generic_any_auto_higher_preplace",
            "grasp_axis": "any",
            "release_index": -1,
            "restore_robot_qpos": False,
            "release_max_joint_delta": None,
            "preplace_height": 0.065,
            "return_neutral": False,
        },
    ]
    completed_roles = list(completed or [])
    wall_support_count = len([item for item in _role_supports(role, completed_roles) if item != "floor"])
    dx = float(getattr(args, "initial_assembly_offset_x", 0.0) or 0.0) if args is not None else 0.0
    dy = float(getattr(args, "initial_assembly_offset_y", 0.0) or 0.0) if args is not None else 0.0
    if role in {"right_wall", "back_wall", "left_wall", "front_wall"} and wall_support_count >= 2:
        preferred = [3, 4, 1, 0, 5, 7, 6, 2]
    elif role == "right_wall" and wall_support_count >= 1:
        if dx > 0.005 and dy < -0.001:
            preferred = [1, 0, 3, 4, 5, 7, 6, 2]
        elif dx > 0.005 and dy > 0.035:
            preferred = [1, 3, 4, 0, 5, 7, 6, 2]
        elif dx > 0.005 and dy > 0.001:
            preferred = [4, 3, 1, 0, 5, 7, 6, 2]
        elif dx > 0.005:
            preferred = [3, 1, 0, 4, 5, 7, 6, 2]
        else:
            preferred = [0, 1, 3, 4, 5, 7, 6, 2]
    elif role == "front_wall":
        preferred = [4, 1, 0, 3, 6, 5, 7, 2]
    elif role == "right_wall":
        preferred = [1, 0, 4, 3, 5, 7, 6, 2]
    elif role in {"back_wall", "left_wall"}:
        preferred = [0, 1, 2, 3, 5, 7, 4, 6]
    else:
        preferred = [0, 1, 2, 3, 5, 7, 4, 6]
    selected = schedule[preferred[attempt_index % len(preferred)]]
    return dict(selected)


def _build_command(
    *,
    args: argparse.Namespace,
    role: str,
    attempt_dir: Path,
    state_in: Path | None,
    state_out: Path,
    completed: list[str],
    attempt_index: int,
) -> tuple[list[str], dict[str, Any]]:
    candidate = _candidate_schedule(role, attempt_index, completed, args)
    role_args = _child_args_for_role(args, role, completed)
    tilt_start_role_attempt = int(getattr(args, "retry_extra_tilt_start_role_attempt", 1) or 1)
    enable_retry_extra_tilt = tilt_start_role_attempt > 0 and attempt_index >= max(tilt_start_role_attempt - 1, 0)
    if enable_retry_extra_tilt:
        retry_grasp_tilts = str(getattr(args, "retry_wall_grasp_extra_tilt_degs", "") or "")
        if retry_grasp_tilts:
            role_args.wall_grasp_extra_tilt_degs = retry_grasp_tilts
            role_args.wall_grasp_extra_tilt_max_abs_deg = float(
                getattr(args, "retry_wall_grasp_extra_tilt_max_abs_deg", 30.0)
            )
        retry_release_tilts = str(getattr(args, "retry_fixed_top_down_extra_tilt_degs", "") or "")
        if retry_release_tilts:
            role_args.fixed_top_down_extra_tilt_degs = retry_release_tilts
            role_args.fixed_top_down_extra_tilt_max_abs_deg = float(
                getattr(args, "retry_fixed_top_down_extra_tilt_max_abs_deg", 30.0)
            )
            role_args.fixed_top_down_extra_tilt_lift_mms = str(
                getattr(args, "retry_fixed_top_down_extra_tilt_lift_mms", "1.0,2.0")
            )
            role_args.fixed_top_down_extra_tilt_normal_bias_mms = str(
                getattr(args, "retry_fixed_top_down_extra_tilt_normal_bias_mms", "0.0,1.5,-1.5")
            )
    planning_budget = _planning_budget_for_attempt(role_args, role, attempt_index)
    execution_budget = _execution_budget_for_attempt(role_args, attempt_index)
    release_ignore_roles = ",".join(_role_supports(role, completed))
    command = [
        role_args.python,
        role_args.planner_script,
        "--out-dir",
        str(attempt_dir),
        "--roles",
        role,
        "--save-assembly-state",
        str(state_out),
        "--save-assembly-roles",
        "floor,right_wall,back_wall,left_wall,front_wall",
        "--save-assembly-completed-roles",
        ",".join([*completed, role]),
        "--restore-loaded-magnetic-connections",
        "--release-ignore-roles",
        release_ignore_roles,
        "--release-candidate-index",
        str(candidate["release_index"]),
        "--square-wall-grasp-edge-axis",
        str(candidate["grasp_axis"]),
        "--prefer-center-wall-grasp" if bool(role_args.prefer_center_wall_grasp) else "--no-prefer-center-wall-grasp",
        "--wall-grasp-center-offset",
        str(role_args.wall_grasp_center_offset),
        "--wall-grasp-center-axis-keep-radius",
        str(role_args.wall_grasp_center_axis_keep_radius),
        "--max-center-wall-grasp-candidates",
        str(role_args.max_center_wall_grasp_candidates),
        "--wall-grasp-diversify-pool-size",
        str(role_args.wall_grasp_diversify_pool_size),
        *_base_child_args(role_args, planning_budget, execution_budget),
    ]
    if attempt_index > 0 and bool(getattr(args, "release_robust_require_preferred_actor_to_tcp_y_offset", False)):
        command.append("--no-release-robust-require-preferred-actor-to-tcp-y-offset")
    if not bool(planning_budget.get("diversify_wall_grasp_candidates", True)):
        command.append("--no-diversify-wall-grasp-candidates")
    if state_in is not None:
        command.extend(["--load-assembly-state", str(state_in)])
        if any(
            abs(float(value)) > 0.0
            for value in (
                args.loaded_state_perturb_dx,
                args.loaded_state_perturb_dy,
                args.loaded_state_perturb_dz,
                args.loaded_state_perturb_yaw_deg,
            )
        ):
            command.extend(
                [
                    "--loaded-state-perturb-dx",
                    str(args.loaded_state_perturb_dx),
                    "--loaded-state-perturb-dy",
                    str(args.loaded_state_perturb_dy),
                    "--loaded-state-perturb-dz",
                    str(args.loaded_state_perturb_dz),
                    "--loaded-state-perturb-yaw-deg",
                    str(args.loaded_state_perturb_yaw_deg),
                    "--loaded-state-perturb-origin-role",
                    args.loaded_state_perturb_origin_role,
                    "--loaded-state-perturb-roles",
                    args.loaded_state_perturb_roles,
                    "--loaded-state-perturb-target-roles",
                    args.loaded_state_perturb_target_roles,
                ]
            )
    if bool(candidate["restore_robot_qpos"]):
        command.append("--restore-loaded-robot-qpos")
    if candidate["release_max_joint_delta"] is not None:
        command.extend(["--release-max-joint-delta", str(candidate["release_max_joint_delta"])])
    if candidate["preplace_height"] is not None:
        command.extend(["--preplace-height", str(candidate["preplace_height"])])
    if (
        not bool(getattr(args, "return_neutral_after_role", True))
        and bool(candidate["return_neutral"])
    ):
        command.extend(["--return-neutral-after-role", "--return-neutral-steps", str(args.return_neutral_steps)])
        command.extend(["--return-neutral-max-joint-step", str(args.return_neutral_max_joint_step)])
        if bool(getattr(args, "return_neutral_skip_final_role", False)):
            command.append("--return-neutral-skip-final-role")
    if bool(args.no_rear_collision_wall):
        command.append("--no-add-rear-collision-wall")
    else:
        command.extend(
            [
                "--rear-collision-wall-frame",
                args.rear_collision_wall_frame,
                "--rear-collision-wall-distance",
                str(args.rear_collision_wall_distance),
                "--rear-collision-wall-width",
                str(args.rear_collision_wall_width),
                "--rear-collision-wall-height",
                str(args.rear_collision_wall_height),
            ]
        )
    if bool(args.add_overhead_collision_wall):
        command.append("--add-overhead-collision-wall")
    metadata = {
        "candidate": candidate,
        "planning_budget": planning_budget,
        "execution_budget": execution_budget,
        "prefer_center_wall_grasp": bool(role_args.prefer_center_wall_grasp),
        "wall_grasp_prior_max_candidates": int(getattr(role_args, "wall_grasp_prior_max_candidates", 0) or 0),
        "release_ignore_roles": release_ignore_roles,
        "state_in": str(state_in) if state_in is not None else None,
        "state_out": str(state_out),
    }
    return command, metadata


def _release_ignore_mapping_for_roles(roles: list[str], completed: list[str]) -> str:
    mapping: list[str] = []
    completed_so_far = list(completed)
    for role in roles:
        supports = _role_supports(role, completed_so_far)
        mapping.append(f"{role}:{','.join(supports)}")
        if role not in completed_so_far:
            completed_so_far.append(role)
    return ";".join(mapping)


def _build_direct_command(
    *,
    args: argparse.Namespace,
    roles: list[str],
    attempt_dir: Path,
    state_in: Path | None,
    state_out: Path,
    completed: list[str],
) -> tuple[list[str], dict[str, Any]]:
    direct_args = argparse.Namespace(**vars(args))
    dx = float(getattr(args, "initial_assembly_offset_x", 0.0) or 0.0)
    dy = float(getattr(args, "initial_assembly_offset_y", 0.0) or 0.0)
    if (
        str(getattr(args, "strategy_preset", "") or "") == "pair_first_robust_fast_v1"
        and dx > 0.005
        and dy > 0.035
        and len(roles) >= 2
        and int(getattr(args, "wall_grasp_prior_max_candidates", 10) or 10) == 10
    ):
        direct_args.wall_grasp_prior_max_candidates = 6
    planning_budget = {
        "mode": "direct_fast_screen",
        "max_release_candidates": min(int(direct_args.max_release_candidates), int(direct_args.fast_screen_max_release_candidates)),
        "max_grasp_candidates": min(int(direct_args.max_grasp_candidates), int(direct_args.fast_screen_max_grasp_candidates)),
        "ik_seeds": min(int(direct_args.ik_seeds), int(direct_args.fast_screen_ik_seeds)),
        "diversify_wall_grasp_candidates": True,
    }
    execution_budget = _execution_budget_for_attempt(direct_args, 0)
    release_ignore_mapping = _release_ignore_mapping_for_roles(roles, completed)
    command = [
        direct_args.python,
        direct_args.planner_script,
        "--out-dir",
        str(attempt_dir),
        "--roles",
        ",".join(roles),
        "--adaptive-role-order" if bool(getattr(direct_args, "adaptive_role_order", True)) else "--no-adaptive-role-order",
        "--save-assembly-state",
        str(state_out),
        "--save-assembly-roles",
        "floor,right_wall,back_wall,left_wall,front_wall",
        "--restore-loaded-magnetic-connections",
        "--release-candidate-index",
        "-1",
        "--release-ignore-roles-by-role",
        release_ignore_mapping,
        "--square-wall-grasp-edge-axis",
        "x",
        "--prefer-center-wall-grasp" if bool(direct_args.prefer_center_wall_grasp) else "--no-prefer-center-wall-grasp",
        "--wall-grasp-center-offset",
        str(direct_args.wall_grasp_center_offset),
        "--wall-grasp-center-axis-keep-radius",
        str(direct_args.wall_grasp_center_axis_keep_radius),
        "--max-center-wall-grasp-candidates",
        str(direct_args.max_center_wall_grasp_candidates),
        "--wall-grasp-diversify-pool-size",
        str(direct_args.wall_grasp_diversify_pool_size),
        *_base_child_args(direct_args, planning_budget, execution_budget),
    ]
    if any(
        abs(float(value)) > 0.0
        for value in (
            args.initial_assembly_offset_x,
            args.initial_assembly_offset_y,
            args.initial_assembly_offset_z,
        )
    ):
        command.extend(
            [
                "--initial-assembly-offset-x",
                str(args.initial_assembly_offset_x),
                "--initial-assembly-offset-y",
                str(args.initial_assembly_offset_y),
                "--initial-assembly-offset-z",
                str(args.initial_assembly_offset_z),
                "--initial-assembly-offset-actor-roles",
                args.initial_assembly_offset_actor_roles,
                "--initial-assembly-offset-target-roles",
                args.initial_assembly_offset_target_roles,
            ]
        )
    if state_in is not None:
        command.extend(["--load-assembly-state", str(state_in)])
        if any(
            abs(float(value)) > 0.0
            for value in (
                args.loaded_state_perturb_dx,
                args.loaded_state_perturb_dy,
                args.loaded_state_perturb_dz,
                args.loaded_state_perturb_yaw_deg,
            )
        ):
            command.extend(
                [
                    "--loaded-state-perturb-dx",
                    str(args.loaded_state_perturb_dx),
                    "--loaded-state-perturb-dy",
                    str(args.loaded_state_perturb_dy),
                    "--loaded-state-perturb-dz",
                    str(args.loaded_state_perturb_dz),
                    "--loaded-state-perturb-yaw-deg",
                    str(args.loaded_state_perturb_yaw_deg),
                    "--loaded-state-perturb-origin-role",
                    args.loaded_state_perturb_origin_role,
                    "--loaded-state-perturb-roles",
                    args.loaded_state_perturb_roles,
                    "--loaded-state-perturb-target-roles",
                    args.loaded_state_perturb_target_roles,
                ]
            )
    if bool(args.no_rear_collision_wall):
        command.append("--no-add-rear-collision-wall")
    else:
        command.extend(
            [
                "--rear-collision-wall-frame",
                args.rear_collision_wall_frame,
                "--rear-collision-wall-distance",
                str(args.rear_collision_wall_distance),
                "--rear-collision-wall-width",
                str(args.rear_collision_wall_width),
                "--rear-collision-wall-height",
                str(args.rear_collision_wall_height),
            ]
        )
    if bool(args.add_overhead_collision_wall):
        command.append("--add-overhead-collision-wall")
    if (
        not bool(getattr(args, "return_neutral_after_role", True))
        and bool(getattr(args, "direct_multi_return_neutral_after_role", True))
        and len(roles) > 1
    ):
        command.extend(["--return-neutral-after-role", "--return-neutral-steps", str(args.direct_multi_return_neutral_steps)])
        command.extend(["--return-neutral-max-joint-step", str(args.return_neutral_max_joint_step)])
        if bool(getattr(args, "return_neutral_skip_final_role", False)):
            command.append("--return-neutral-skip-final-role")
    metadata = {
        "candidate": {"name": "direct_multi_auto", "roles": roles, "release_index": -1, "grasp_axis": "any"},
        "planning_budget": planning_budget,
        "execution_budget": execution_budget,
        "prefer_center_wall_grasp": bool(direct_args.prefer_center_wall_grasp),
        "wall_grasp_prior_max_candidates": int(getattr(direct_args, "wall_grasp_prior_max_candidates", 0) or 0),
        "release_ignore_roles_by_role": release_ignore_mapping,
        "return_neutral_after_role": bool(getattr(args, "direct_multi_return_neutral_after_role", True)) and len(roles) > 1,
        "state_in": str(state_in) if state_in is not None else None,
        "state_out": str(state_out),
    }
    return command, metadata


def _append_log(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _run_child_process(
    *,
    command: list[str],
    attempt_dir: Path,
    cwd: Path,
    env: dict[str, str],
    timeout_sec: float,
    inline: bool = False,
    stream: bool = False,
) -> tuple[int, bool, Path, Path]:
    child_stdout = attempt_dir / "subprocess_stdout.txt"
    child_stderr = attempt_dir / "subprocess_stderr.txt"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    with child_stdout.open("w", encoding="utf-8") as stdout_handle, child_stderr.open("w", encoding="utf-8") as stderr_handle:
        if inline:
            old_argv = sys.argv[:]
            old_cwd = Path.cwd()
            old_env = os.environ.copy()
            try:
                os.environ.clear()
                os.environ.update(env)
                os.chdir(cwd)
                sys.argv = [command[1], *command[2:]]
                with contextlib.redirect_stdout(stdout_handle), contextlib.redirect_stderr(stderr_handle):
                    try:
                        runpy.run_path(command[1], run_name="__main__")
                        return 0, False, child_stdout, child_stderr
                    except SystemExit as exc:
                        code = exc.code if isinstance(exc.code, int) else 1
                        return int(code), False, child_stdout, child_stderr
                    except Exception:
                        traceback.print_exc(file=stderr_handle)
                        return 1, False, child_stdout, child_stderr
            finally:
                sys.argv = old_argv
                os.chdir(old_cwd)
                os.environ.clear()
                os.environ.update(old_env)
        try:
            if stream:
                process = subprocess.Popen(
                    command,
                    cwd=str(cwd),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )

                def pump(pipe: Any, handle: Any, mirror: Any) -> None:
                    try:
                        for line in iter(pipe.readline, ""):
                            handle.write(line)
                            handle.flush()
                            mirror.write(line)
                            mirror.flush()
                    finally:
                        try:
                            pipe.close()
                        except Exception:
                            pass

                stdout_thread = threading.Thread(
                    target=pump,
                    args=(process.stdout, stdout_handle, sys.stdout),
                    daemon=True,
                )
                stderr_thread = threading.Thread(
                    target=pump,
                    args=(process.stderr, stderr_handle, sys.stderr),
                    daemon=True,
                )
                stdout_thread.start()
                stderr_thread.start()
                try:
                    returncode = process.wait(timeout=timeout_sec)
                    stdout_thread.join(timeout=2.0)
                    stderr_thread.join(timeout=2.0)
                    return int(returncode), False, child_stdout, child_stderr
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout_thread.join(timeout=2.0)
                    stderr_thread.join(timeout=2.0)
                    return 124, True, child_stdout, child_stderr
            completed_process = subprocess.run(
                command,
                cwd=str(cwd),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
                timeout=timeout_sec,
            )
            return int(completed_process.returncode), False, child_stdout, child_stderr
        except subprocess.TimeoutExpired:
            return 124, True, child_stdout, child_stderr


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = _unique_dir(Path(args.out_dir).resolve()) if args.unique_out_dir else Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir = out_dir / "attempts"
    checkpoints_dir = out_dir / "checkpoints"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "standard_build_log.txt"
    report_path = out_dir / "standard_build_report.json"
    requested_roles = _split_roles(args.roles)
    roles = _adaptive_first_layer_roles(args, requested_roles)
    env = os.environ.copy()
    env["JIMU_DRIVE_FORCE_LIMIT"] = str(args.drive_force_limit)
    if bool(getattr(args, "stream_child_output", False)):
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTHONIOENCODING", "utf-8")
    torch_extensions_dir = Path(args.torch_extensions_dir).expanduser().resolve()
    torch_extensions_dir.mkdir(parents=True, exist_ok=True)
    env["TORCH_EXTENSIONS_DIR"] = str(torch_extensions_dir)
    for key, value in DEFAULT_STAGE_ENV.items():
        env[key] = value
    report: dict[str, Any] = {
        "started_at": _now(),
        "out_dir": str(out_dir),
        "planner_script": args.planner_script,
        "python": args.python,
        "robot_cfg": args.robot_cfg,
        "curobo_root": args.curobo_root,
        "roles": roles,
        "requested_roles": requested_roles,
        "execution_mode": args.execution_mode,
        "strategy_preset": str(getattr(args, "strategy_preset", "baseline")),
        "torch_extensions_dir": str(torch_extensions_dir),
        "magnet_physics": {
            "drive_force_limit": str(args.drive_force_limit),
            "attach_distance": str(args.magnet_attach_distance),
            "attract_distance": str(args.magnet_attract_distance),
            "detach_distance": str(args.magnet_detach_distance),
            "edge_sample_half_span": str(args.magnet_edge_sample_half_span),
            "connect_edge_sample_half_span": str(args.magnet_connect_edge_sample_half_span),
            "edge_sample_offsets": str(args.magnet_edge_sample_offsets),
            "connect_edge_sample_offsets": str(args.magnet_connect_edge_sample_offsets),
            "attract_stiffness": str(args.magnet_attract_stiffness),
            "attract_force_limit": str(args.magnet_attract_force_limit),
            "active_stiffness": str(args.magnet_active_stiffness),
            "active_force_limit": str(args.magnet_active_force_limit),
            "active_edge_torque_scale": str(args.magnet_active_edge_torque_scale),
            "active_edge_torque_min_scale": str(args.magnet_active_edge_torque_min_scale),
            "active_normal_torque_scale": str(args.magnet_active_normal_torque_scale),
            "active_normal_torque_min_scale": str(args.magnet_active_normal_torque_min_scale),
            "active_torque_delay_steps": str(args.magnet_active_torque_delay_steps),
            "active_torque_ramp_steps": str(args.magnet_active_torque_ramp_steps),
            "active_torque_max_point_error": str(args.magnet_active_torque_max_point_error),
            "floor_support_score": str(args.magnet_floor_support_score),
            "support_connection_score_scale": str(args.magnet_support_connection_score_scale),
            "multi_connection_support_bonus": str(args.magnet_multi_connection_support_bonus),
            "drive_stiffness": str(args.magnet_drive_stiffness),
            "drive_force_limit_override": str(args.magnet_drive_force_limit),
        },
        "planning": {
            "record_live": bool(getattr(args, "record_live", True)),
            "fast_screen_attempts": int(args.fast_screen_attempts),
            "fast_screen_max_release_candidates": int(args.fast_screen_max_release_candidates),
            "fast_screen_max_grasp_candidates": int(args.fast_screen_max_grasp_candidates),
            "fast_screen_ik_seeds": int(args.fast_screen_ik_seeds),
            "fast_screen_disable_diversified_front_grasp": bool(args.fast_screen_disable_diversified_front_grasp),
            "fast_chain_screening": bool(args.fast_chain_screening),
            "fast_chain_top_grasp_candidates": int(args.fast_chain_top_grasp_candidates),
            "fast_chain_top_release_candidates": int(args.fast_chain_top_release_candidates),
            "fast_chain_ik_seeds": int(args.fast_chain_ik_seeds),
            "fast_chain_pair_probe_candidates": int(getattr(args, "fast_chain_pair_probe_candidates", 0)),
            "fast_chain_stop_pair_probe_after_success": bool(getattr(args, "fast_chain_stop_pair_probe_after_success", False)),
            "fast_chain_min_successful_pair_probes": int(getattr(args, "fast_chain_min_successful_pair_probes", 0)),
            "fast_chain_allow_fallback": bool(args.fast_chain_allow_fallback),
            "fast_chain_supported_wall_max_actor_to_tcp_y": float(
                getattr(args, "fast_chain_supported_wall_max_actor_to_tcp_y", 0.0) or 0.0
            ),
            "prefer_center_wall_grasp": bool(args.prefer_center_wall_grasp),
            "wall_grasp_center_offset": float(args.wall_grasp_center_offset),
            "wall_grasp_center_axis_keep_radius": float(args.wall_grasp_center_axis_keep_radius),
            "max_center_wall_grasp_candidates": int(args.max_center_wall_grasp_candidates),
            "wall_grasp_diversify_pool_size": int(args.wall_grasp_diversify_pool_size),
            "wall_grasp_prior_mode": str(getattr(args, "wall_grasp_prior_mode", "none")),
            "wall_grasp_prior_max_candidates": int(getattr(args, "wall_grasp_prior_max_candidates", 0)),
            "grasp_quality_max_tcp_position_delta": float(args.grasp_quality_max_tcp_position_delta),
            "grasp_quality_max_tcp_orientation_delta_deg": float(args.grasp_quality_max_tcp_orientation_delta_deg),
            "grasp_quality_min_finger_force": float(args.grasp_quality_min_finger_force),
            "grasp_quality_max_force_balance_error": float(args.grasp_quality_max_force_balance_error),
            "grasp_quality_allow_live_lock_recovery": bool(getattr(args, "grasp_quality_allow_live_lock_recovery", False)),
            "grasp_quality_live_lock_max_tcp_position_delta": float(
                getattr(args, "grasp_quality_live_lock_max_tcp_position_delta", 0.0)
            ),
            "fallback_max_release_candidates": int(args.max_release_candidates),
            "fallback_max_grasp_candidates": int(args.max_grasp_candidates),
            "fallback_ik_seeds": int(args.ik_seeds),
            "return_neutral_steps": int(args.return_neutral_steps),
            "direct_multi_return_neutral_steps": int(args.direct_multi_return_neutral_steps),
            "direct_first_max_roles": int(args.direct_first_max_roles),
            "return_neutral_max_joint_step": float(args.return_neutral_max_joint_step),
            "return_neutral_skip_final_role": bool(args.return_neutral_skip_final_role),
            "release_score_preplace_joint_weight": float(args.release_score_preplace_joint_weight),
            "release_score_place_joint_weight": float(args.release_score_place_joint_weight),
            "release_motion_plan_preplace": bool(args.release_motion_plan_preplace),
            "release_motion_plan_on_branch_jump": bool(getattr(args, "release_motion_plan_on_branch_jump", False)),
            "release_motion_plan_on_branch_jump_min_connection_potential": int(
                getattr(args, "release_motion_plan_on_branch_jump_min_connection_potential", 0) or 0
            ),
            "release_motion_plan_timeout": float(args.release_motion_plan_timeout),
            "release_motion_plan_max_waypoints": int(args.release_motion_plan_max_waypoints),
            "enable_release_yaw_candidates": bool(getattr(args, "enable_release_yaw_candidates", False)),
            "release_candidate_indices": str(getattr(args, "release_candidate_indices", "")),
            "release_prefer_candidate_index_order_for_multi_connection": bool(
                getattr(args, "release_prefer_candidate_index_order_for_multi_connection", False)
            ),
            "release_robust_actor_to_tcp_y_offsets": str(args.release_robust_actor_to_tcp_y_offsets),
            "release_robust_actor_to_tcp_x_offsets": str(args.release_robust_actor_to_tcp_x_offsets),
            "release_robust_actor_to_tcp_z_offsets": str(args.release_robust_actor_to_tcp_z_offsets),
            "release_robust_actor_to_tcp_yaw_deg_offsets": str(args.release_robust_actor_to_tcp_yaw_deg_offsets),
            "release_robust_min_variant_successes": int(args.release_robust_min_variant_successes),
            "release_robust_score_weight": float(args.release_robust_score_weight),
            "release_robust_require_same_release_index": bool(args.release_robust_require_same_release_index),
            "release_robust_prefer_best_variant": bool(getattr(args, "release_robust_prefer_best_variant", False)),
            "release_robust_preferred_actor_to_tcp_y_offset": float(
                getattr(args, "release_robust_preferred_actor_to_tcp_y_offset", 0.0)
            ),
            "release_robust_require_preferred_actor_to_tcp_y_offset": bool(
                getattr(args, "release_robust_require_preferred_actor_to_tcp_y_offset", False)
            ),
            "release_robust_require_preferred_max_connection_potential": int(
                getattr(args, "release_robust_require_preferred_max_connection_potential", 0)
            ),
            "release_robust_early_stop_on_first_success": bool(
                getattr(args, "release_robust_early_stop_on_first_success", False)
            ),
            "release_screen_max_preplace_joint_delta": float(args.release_screen_max_preplace_joint_delta),
            "release_screen_single_connection_max_place_joint_delta": float(
                getattr(args, "release_screen_single_connection_max_place_joint_delta", 0.0)
            ),
            "release_screen_max_predicted_actor_position_error": float(args.release_screen_max_predicted_actor_position_error),
            "release_screen_max_predicted_actor_orientation_error_deg": float(args.release_screen_max_predicted_actor_orientation_error_deg),
            "candidate_screen_max_release_failures": int(getattr(args, "candidate_screen_max_release_failures", 0) or 0),
            "candidate_screen_max_no_robust_failures": int(
                getattr(args, "candidate_screen_max_no_robust_failures", 0) or 0
            ),
            "release_prediction_gate_fallback_min_connection_potential": int(
                getattr(args, "release_prediction_gate_fallback_min_connection_potential", 0) or 0
            ),
            "release_ignore_roles_as_collision_exclusions": bool(getattr(args, "release_ignore_roles_as_collision_exclusions", True)),
            "attach_held_payload_for_release_planning": bool(getattr(args, "attach_held_payload_for_release_planning", False)),
            "lock_held_actor_after_grasp": bool(getattr(args, "lock_held_actor_after_grasp", False)),
            "held_actor_lock_reference": str(getattr(args, "held_actor_lock_reference", "live")),
            "held_actor_unlock_stage": str(getattr(args, "held_actor_unlock_stage", None) or "before_open_gripper"),
            "use_live_actor_to_tcp_after_lift": bool(getattr(args, "use_live_actor_to_tcp_after_lift", True)),
            "live_actor_to_tcp_after_lift_min_connection_potential": int(
                getattr(args, "live_actor_to_tcp_after_lift_min_connection_potential", 0) or 0
            ),
            "allow_nominal_release_fallback": bool(getattr(args, "allow_nominal_release_fallback", False)),
            "allow_nominal_release_fallback_max_connection_potential": int(
                getattr(args, "allow_nominal_release_fallback_max_connection_potential", 0) or 0
            ),
            "enable_capture_after_open_roles": str(args.enable_capture_after_open_roles),
            "capture_after_open_min_connection_potential": int(args.capture_after_open_min_connection_potential),
            "adaptive_role_order": bool(args.adaptive_role_order),
            "role_order_policy": str(getattr(args, "role_order_policy", "adaptive")),
            "use_cached_pair_release": bool(getattr(args, "use_cached_pair_release", False)),
            "pre_open_min_active_connections": int(args.pre_open_min_active_connections),
            "pre_open_connection_correction_max_pose_error": float(args.pre_open_connection_correction_max_pose_error),
            "pre_open_connection_correction_max_orientation_error_deg": float(
                args.pre_open_connection_correction_max_orientation_error_deg
            ),
            "pre_open_connection_correction_trigger_pose_error": float(
                args.pre_open_connection_correction_trigger_pose_error
            ),
            "pre_open_connection_correction_trigger_orientation_error_deg": float(
                args.pre_open_connection_correction_trigger_orientation_error_deg
            ),
            "require_active_connection_before_open": bool(args.require_active_connection_before_open),
            "final_max_connection_point_error": float(args.final_max_connection_point_error),
            "final_max_connection_normal_angle_deg": float(args.final_max_connection_normal_angle_deg),
            "final_max_connection_edge_angle_deg": float(args.final_max_connection_edge_angle_deg),
            "initial_assembly_offset": {
                "x": float(args.initial_assembly_offset_x),
                "y": float(args.initial_assembly_offset_y),
                "z": float(args.initial_assembly_offset_z),
                "actor_roles": str(args.initial_assembly_offset_actor_roles),
                "target_roles": str(args.initial_assembly_offset_target_roles),
            },
            "initial_actor_jitter": {
                "xy": float(getattr(args, "initial_actor_jitter_xy", 0.0) or 0.0),
                "seed": int(getattr(args, "initial_actor_jitter_seed", 0) or 0),
                "roles": str(getattr(args, "initial_actor_jitter_roles", "")),
                "min_start_distance": float(getattr(args, "initial_actor_jitter_min_start_distance", 0.0) or 0.0),
                "min_target_distance": float(getattr(args, "initial_actor_jitter_min_target_distance", 0.0) or 0.0),
                "max_sample_attempts": int(getattr(args, "initial_actor_jitter_max_sample_attempts", 0) or 0),
            },
        },
        "stage_env": DEFAULT_STAGE_ENV,
        "attempts": [],
        "successes": [],
        "final": {"success": False, "completed_roles": []},
    }
    state_in: Path | None = Path(args.initial_state).resolve() if args.initial_state else None
    completed: list[str] = []
    attempt_number = 0
    direct_failed_role: str | None = None
    direct_failed_completed_count = 0
    started = time.perf_counter()
    if args.execution_mode in {"direct-first", "direct-only"} and roles:
        attempt_number += 1
        direct_roles = [role for role in roles if role not in completed]
        direct_first_max_roles = max(int(getattr(args, "direct_first_max_roles", 0) or 0), 0)
        if args.execution_mode == "direct-first" and direct_first_max_roles > 0:
            direct_roles = direct_roles[:direct_first_max_roles]
        attempt_dir = attempts_dir / f"attempt_{attempt_number:03d}_direct_multi"
        state_out = checkpoints_dir / f"attempt_{attempt_number:03d}_direct_multi_state.json"
        command, metadata = _build_direct_command(
            args=args,
            roles=direct_roles,
            attempt_dir=attempt_dir,
            state_in=state_in,
            state_out=state_out,
            completed=completed,
        )
        attempt_record: dict[str, Any] = {
            "attempt_number": attempt_number,
            "role": "__direct_multi__",
            "roles": direct_roles,
            "role_attempt": 1,
            "started_at": _now(),
            "attempt_dir": str(attempt_dir),
            "command": command,
            **metadata,
        }
        _append_log(log_path, f"[{_now()}] start attempt={attempt_number} mode=direct_multi roles={','.join(direct_roles)}")
        returncode, timed_out, child_stdout, child_stderr = _run_child_process(
            command=command,
            attempt_dir=attempt_dir,
            cwd=Path(args.planner_script).resolve().parent,
            env=env,
            timeout_sec=args.attempt_timeout_sec,
            inline=bool(getattr(args, "inline_child_planner", False)),
            stream=bool(getattr(args, "stream_child_output", False)),
        )
        summary_path = attempt_dir / "multi_wall_summary.json"
        summary = _read_json(summary_path)
        failure = _classify_failure(summary, returncode)
        direct_failed_role = str(failure.get("role") or "") or None
        direct_completed = [
            role
            for role in roles
            if role in set((summary or {}).get("final", {}).get("completed_roles", []))
        ]
        direct_failed_completed_count = len(direct_completed)
        success = bool(summary and summary.get("final", {}).get("success") and returncode == 0 and state_out.exists())
        attempt_record.update(
            {
                "finished_at": _now(),
                "elapsed_since_start_sec": round(time.perf_counter() - started, 3),
                "returncode": returncode,
                "timed_out": timed_out,
                "summary_path": str(summary_path) if summary_path.exists() else None,
                "stdout_path": str(child_stdout),
                "stderr_path": str(child_stderr),
                "video": str(attempt_dir / "multi_wall_live.mp4") if (attempt_dir / "multi_wall_live.mp4").exists() else None,
                "success": success,
                "classification": failure,
                "direct_completed_roles": direct_completed,
            }
        )
        report["attempts"].append(attempt_record)
        report_path.write_text(json.dumps(_json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")
        _append_log(log_path, f"[{_now()}] finish attempt={attempt_number} mode=direct_multi success={success} class={failure.get('kind')}")
        if success:
            if bool(getattr(args, "execute_real", False)):
                real_report = _execute_real_from_summary(
                    args=args,
                    summary_path=summary_path,
                    log_path=log_path,
                    label=f"direct_multi_attempt_{attempt_number:03d}",
                )
                attempt_record["real_execution"] = real_report
                report_path.write_text(json.dumps(_json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")
                if not bool(real_report.get("success")):
                    report["final"] = {
                        "success": False,
                        "completed_roles": completed,
                        "failed_role": direct_roles[0] if direct_roles else None,
                        "latest_stable_state": str(state_in) if state_in is not None else None,
                        "real_execution": real_report,
                    }
                    report["finished_at"] = _now()
                    report_path.write_text(json.dumps(_json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")
                    return report
            stable_state = checkpoints_dir / f"stable_direct_{len(direct_roles):02d}_{direct_roles[-1]}_state.json"
            shutil.copy2(state_out, stable_state)
            completed = [role for role in roles if role in {*completed, *direct_roles}]
            state_in = stable_state
            report["successes"].append(
                {
                    "roles": direct_roles,
                    "attempt_number": attempt_number,
                    "stable_state": str(stable_state),
                    "video": attempt_record["video"],
                    "summary_path": attempt_record["summary_path"],
                    "candidate": metadata["candidate"],
                }
            )
            report["final"] = {
                "success": len(completed) == len(roles),
                "completed_roles": completed,
                "latest_stable_state": str(state_in),
                "latest_video": attempt_record["video"],
            }
            if len(completed) == len(roles):
                report["finished_at"] = _now()
                report_path.write_text(json.dumps(_json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")
                return report
            report_path.write_text(json.dumps(_json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")
        if direct_completed and state_out.exists() and not bool(getattr(args, "execute_real", False)):
            completed = direct_completed
            stable_state = checkpoints_dir / f"stable_direct_partial_{len(completed):02d}_{completed[-1]}_state.json"
            shutil.copy2(state_out, stable_state)
            state_in = stable_state
            report["successes"].append(
                {
                    "roles": direct_completed,
                    "attempt_number": attempt_number,
                    "stable_state": str(stable_state),
                    "video": attempt_record["video"],
                    "summary_path": attempt_record["summary_path"],
                    "candidate": metadata["candidate"],
                    "partial_direct_pass": True,
                }
            )
            report["final"] = {
                "success": False,
                "completed_roles": completed,
                "latest_stable_state": str(state_in),
                "latest_video": attempt_record["video"],
            }
            report_path.write_text(json.dumps(_json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")
        if args.execution_mode == "direct-only":
            report["final"] = {
                "success": False,
                "completed_roles": completed,
                "failed_role": next((role for role in roles if role not in completed), None),
                "latest_stable_state": str(state_in) if state_in is not None else None,
                "latest_video": attempt_record["video"],
            }
            report["finished_at"] = _now()
            report_path.write_text(json.dumps(_json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")
            return report
    for role in roles:
        if role in completed:
            continue
        role_success = False
        for role_attempt in range(args.max_attempts_per_role):
            attempt_number += 1
            direct_failure_skip = 0
            if role == direct_failed_role:
                direct_failure_skip = 1
            positive_x_initial_right_skip = (
                role == "right_wall"
                and not completed
                and float(getattr(args, "initial_assembly_offset_x", 0.0) or 0.0) > 0.005
            )
            schedule_attempt_index = role_attempt + max(direct_failure_skip, 1 if positive_x_initial_right_skip else 0)
            attempt_dir = attempts_dir / f"attempt_{attempt_number:03d}_{role}"
            state_out = checkpoints_dir / f"attempt_{attempt_number:03d}_{role}_state.json"
            command, metadata = _build_command(
                args=args,
                role=role,
                attempt_dir=attempt_dir,
                state_in=state_in,
                state_out=state_out,
                completed=completed,
                attempt_index=schedule_attempt_index,
            )
            attempt_record: dict[str, Any] = {
                "attempt_number": attempt_number,
                "role": role,
                "role_attempt": role_attempt + 1,
                "schedule_attempt_index": schedule_attempt_index,
                "started_at": _now(),
                "attempt_dir": str(attempt_dir),
                "command": command,
                **metadata,
            }
            _append_log(log_path, f"[{_now()}] start attempt={attempt_number} role={role} candidate={metadata['candidate']['name']}")
            returncode, timed_out, child_stdout, child_stderr = _run_child_process(
                command=command,
                attempt_dir=attempt_dir,
                cwd=Path(args.planner_script).resolve().parent,
                env=env,
                timeout_sec=args.attempt_timeout_sec,
                inline=bool(getattr(args, "inline_child_planner", False)),
                stream=bool(getattr(args, "stream_child_output", False)),
            )
            summary_path = attempt_dir / "multi_wall_summary.json"
            summary = _read_json(summary_path)
            failure = _classify_failure(summary, returncode)
            success = bool(summary and summary.get("final", {}).get("success") and returncode == 0 and state_out.exists())
            attempt_record.update(
                {
                    "finished_at": _now(),
                    "elapsed_since_start_sec": round(time.perf_counter() - started, 3),
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "summary_path": str(summary_path) if summary_path.exists() else None,
                    "stdout_path": str(child_stdout),
                    "stderr_path": str(child_stderr),
                    "video": str(attempt_dir / "multi_wall_live.mp4") if (attempt_dir / "multi_wall_live.mp4").exists() else None,
                    "success": success,
                    "classification": failure,
                }
            )
            report["attempts"].append(attempt_record)
            report_path.write_text(json.dumps(_json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")
            _append_log(log_path, f"[{_now()}] finish attempt={attempt_number} role={role} success={success} class={failure.get('kind')}")
            if success:
                if bool(getattr(args, "execute_real", False)):
                    real_report = _execute_real_from_summary(
                        args=args,
                        summary_path=summary_path,
                        log_path=log_path,
                        label=f"{role}_attempt_{attempt_number:03d}",
                    )
                    attempt_record["real_execution"] = real_report
                    report_path.write_text(json.dumps(_json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")
                    if not bool(real_report.get("success")):
                        report["final"] = {
                            "success": False,
                            "completed_roles": completed,
                            "failed_role": role,
                            "latest_stable_state": str(state_in) if state_in is not None else None,
                            "real_execution": real_report,
                        }
                        report["finished_at"] = _now()
                        report_path.write_text(json.dumps(_json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")
                        return report
                stable_state = checkpoints_dir / f"stable_{len(completed) + 1:02d}_{role}_state.json"
                shutil.copy2(state_out, stable_state)
                completed.append(role)
                state_in = stable_state
                report["successes"].append(
                    {
                        "role": role,
                        "attempt_number": attempt_number,
                        "stable_state": str(stable_state),
                        "video": attempt_record["video"],
                        "summary_path": attempt_record["summary_path"],
                        "candidate": metadata["candidate"],
                    }
                )
                report["final"] = {
                    "success": len(completed) == len(roles),
                    "completed_roles": completed,
                    "latest_stable_state": str(state_in),
                    "latest_video": attempt_record["video"],
                }
                report_path.write_text(json.dumps(_json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")
                role_success = True
                break
        if not role_success:
            report["final"] = {
                "success": False,
                "completed_roles": completed,
                "failed_role": role,
                "latest_stable_state": str(state_in) if state_in is not None else None,
            }
            report["finished_at"] = _now()
            report_path.write_text(json.dumps(_json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")
            return report
    report["finished_at"] = _now()
    report["final"]["success"] = len(completed) == len(roles)
    report_path.write_text(json.dumps(_json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy-preset",
        choices=[
            "baseline",
            "pair_first_speed_v1",
            "pair_first_speed_v2",
            "pair_first_speed_v3",
            "pair_first_speed_v4",
            "pair_first_fixed_mixed_v1",
            "pair_first_stable_v1",
            "pair_first_mined_light_v1",
            "pair_first_robust_fast_v1",
        ],
        default="baseline",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--planner-script", default=DEFAULT_SCRIPT)
    parser.add_argument("--robot-cfg", default=DEFAULT_ROBOT_CFG)
    parser.add_argument("--curobo-root", default=DEFAULT_CUROBO_ROOT)
    parser.add_argument("--roles", default=DEFAULT_ROLES)
    parser.add_argument("--initial-state", default="")
    parser.add_argument("--execution-mode", choices=["direct-first", "direct-only", "retry"], default="direct-first")
    parser.add_argument("--torch-extensions-dir", default=DEFAULT_TORCH_EXTENSIONS_DIR)
    parser.add_argument("--return-neutral-after-role", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--direct-multi-return-neutral-after-role", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--direct-multi-return-neutral-steps", type=int, default=16)
    parser.add_argument("--direct-first-max-roles", type=int, default=0)
    parser.add_argument("--return-neutral-max-joint-step", type=float, default=0.080)
    parser.add_argument("--return-neutral-skip-final-role", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-pregrasp-graph-recovery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback-conservative-execution", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-attempts-per-role", type=int, default=2)
    parser.add_argument("--attempt-timeout-sec", type=float, default=600.0)
    parser.add_argument("--unique-out-dir", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inline-child-planner", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--record-live", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-mode", choices=["none", "rgb_array", "human"], default="none")
    parser.add_argument("--human-render-every", type=int, default=1)
    parser.add_argument("--wait-before-close", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stream-child-output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--execute-real", action="store_true", help="Connect to RM75 and stream accepted child trajectories to the real robot.")
    parser.add_argument("--auto-execute", action="store_true", help="Skip the child startup confirmation prompt for --execute-real.")
    parser.add_argument("--robot-ip", type=str, default=None)
    parser.add_argument("--lerobot-root", type=str, default=DEFAULT_LEROBOT_ROOT)
    parser.add_argument("--lerobot-sim2real-root", type=str, default=DEFAULT_LEROBOT_SIM2REAL_ROOT)
    parser.add_argument("--real-gripper-open", type=float, default=0.0)
    parser.add_argument("--real-gripper-close", type=float, default=0.91)
    parser.add_argument("--reset-real-before-start", dest="reset_real_before_start", action="store_true", default=True)
    parser.add_argument("--no-reset-real-before-start", dest="reset_real_before_start", action="store_false")
    parser.add_argument("--real-start-max-delta", type=float, default=0.12)
    parser.add_argument("--real-control-hz", type=float, default=30.0)
    parser.add_argument("--real-max-delta-per-step", type=float, default=0.1)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--record-every", type=int, default=6)
    parser.add_argument("--max-release-candidates", type=int, default=18)
    parser.add_argument("--max-grasp-candidates", type=int, default=32)
    parser.add_argument("--ik-seeds", type=int, default=32)
    parser.add_argument("--fast-screen-attempts", type=int, default=1)
    parser.add_argument("--fast-screen-max-release-candidates", type=int, default=12)
    parser.add_argument("--fast-screen-max-grasp-candidates", type=int, default=16)
    parser.add_argument("--fast-screen-ik-seeds", type=int, default=16)
    parser.add_argument("--fast-screen-disable-diversified-front-grasp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-chain-screening", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-chain-top-grasp-candidates", type=int, default=4)
    parser.add_argument("--fast-chain-top-release-candidates", type=int, default=4)
    parser.add_argument("--fast-chain-ik-seeds", type=int, default=12)
    parser.add_argument("--fast-chain-allow-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-chain-pair-probe-candidates", type=int, default=0)
    parser.add_argument("--fast-chain-stop-pair-probe-after-success", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fast-chain-min-successful-pair-probes", type=int, default=0)
    parser.add_argument("--fast-chain-center-distance-weight", type=float, default=0.28)
    parser.add_argument("--fast-chain-release-probe-weight", type=float, default=0.6)
    parser.add_argument("--fast-chain-supported-wall-max-actor-to-tcp-y", type=float, default=0.0)
    parser.add_argument("--use-cuda-graph-batch-ik", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cuda-graph-batch-ik-max-batch", type=int, default=128)
    parser.add_argument("--cuda-graph-batch-ik-fixed-batch-size", type=int, default=0)
    parser.add_argument("--prefer-center-wall-grasp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wall-grasp-center-offset", type=float, default=0.004)
    parser.add_argument("--wall-grasp-center-axis-keep-radius", type=float, default=0.012)
    parser.add_argument("--max-center-wall-grasp-candidates", type=int, default=8)
    parser.add_argument("--wall-grasp-diversify-pool-size", type=int, default=256)
    parser.add_argument("--wall-grasp-prior-mode", choices=["none", "mined_success_v1"], default="none")
    parser.add_argument("--wall-grasp-prior-max-candidates", type=int, default=0)
    parser.add_argument("--grasp-quality-max-tcp-position-delta", type=float, default=0.030)
    parser.add_argument("--grasp-quality-max-tcp-orientation-delta-deg", type=float, default=35.0)
    parser.add_argument("--grasp-quality-min-finger-force", type=float, default=1.0)
    parser.add_argument("--grasp-quality-max-force-balance-error", type=float, default=0.85)
    parser.add_argument("--grasp-quality-allow-live-lock-recovery", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grasp-quality-live-lock-max-tcp-position-delta", type=float, default=0.0)
    parser.add_argument("--close-steps", type=int, default=6)
    parser.add_argument("--open-steps", type=int, default=6)
    parser.add_argument("--pre-open-hold-steps", type=int, default=6)
    parser.add_argument("--move-steps", type=int, default=10)
    parser.add_argument("--short-steps", type=int, default=6)
    parser.add_argument("--release-steps", type=int, default=14)
    parser.add_argument("--stability-steps", type=int, default=8)
    parser.add_argument("--final-all-roles-stability-steps", type=int, default=12)
    parser.add_argument("--max-joint-step", type=float, default=0.080)
    parser.add_argument("--max-segment-steps", type=int, default=500)
    parser.add_argument("--max-existing-path-waypoints", type=int, default=48)
    parser.add_argument("--free-space-motion-window", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--free-space-pregrasp-max-waypoints", type=int, default=12)
    parser.add_argument("--free-space-return-neutral-steps", type=int, default=4)
    parser.add_argument("--free-space-return-neutral-max-joint-step", type=float, default=0.12)
    parser.add_argument("--grasp-max-joint-delta", type=float, default=3.0)
    parser.add_argument("--release-max-joint-delta", type=float, default=2.2)
    parser.add_argument("--ik-position-threshold", type=float, default=0.008)
    parser.add_argument("--ik-rotation-threshold", type=float, default=0.12)
    parser.add_argument("--release-score-preplace-joint-weight", type=float, default=0.0050)
    parser.add_argument("--release-score-place-joint-weight", type=float, default=0.0015)
    parser.add_argument("--release-motion-plan-preplace", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-motion-plan-on-branch-jump", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-motion-plan-on-branch-jump-min-connection-potential", type=int, default=0)
    parser.add_argument("--release-motion-plan-timeout", type=float, default=4.0)
    parser.add_argument("--release-motion-plan-max-waypoints", type=int, default=24)
    parser.add_argument("--enable-release-yaw-candidates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-candidate-indices", default="")
    parser.add_argument("--release-prefer-candidate-index-order-for-multi-connection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-robust-actor-to-tcp-y-offsets", default="0.0")
    parser.add_argument("--release-robust-actor-to-tcp-x-offsets", default="0.0")
    parser.add_argument("--release-robust-actor-to-tcp-z-offsets", default="0.0")
    parser.add_argument("--release-robust-actor-to-tcp-yaw-deg-offsets", default="0.0")
    parser.add_argument("--release-robust-min-variant-successes", type=int, default=1)
    parser.add_argument("--release-robust-score-weight", type=float, default=0.0)
    parser.add_argument("--release-robust-require-same-release-index", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--release-robust-prefer-best-variant", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-robust-preferred-actor-to-tcp-y-offset", type=float, default=0.0)
    parser.add_argument("--release-robust-require-preferred-actor-to-tcp-y-offset", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-robust-require-preferred-max-connection-potential", type=int, default=0)
    parser.add_argument("--release-robust-early-stop-on-first-success", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-screen-max-preplace-joint-delta", type=float, default=0.0)
    parser.add_argument("--release-screen-single-connection-max-place-joint-delta", type=float, default=0.0)
    parser.add_argument("--release-screen-max-predicted-actor-position-error", type=float, default=0.0)
    parser.add_argument("--release-screen-max-predicted-actor-orientation-error-deg", type=float, default=0.0)
    parser.add_argument("--candidate-screen-max-release-failures", type=int, default=0)
    parser.add_argument("--candidate-screen-max-no-robust-failures", type=int, default=0)
    parser.add_argument("--release-prediction-gate-fallback-min-connection-potential", type=int, default=0)
    parser.add_argument("--release-ignore-roles-as-collision-exclusions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--attach-held-payload-for-release-planning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--runtime-collision-monitor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--runtime-collision-monitor-period", type=int, default=5)
    parser.add_argument("--runtime-collision-monitor-force-threshold", type=float, default=0.05)
    parser.add_argument("--runtime-collision-monitor-max-events", type=int, default=24)
    parser.add_argument("--runtime-collision-monitor-include-floor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lock-held-actor-after-grasp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--held-actor-lock-reference", choices=["live", "nominal"], default="live")
    parser.add_argument("--held-actor-unlock-stage", choices=["before_open_gripper", "after_open_gripper"], default=None)
    parser.add_argument("--live-actor-to-tcp-after-lift", dest="use_live_actor_to_tcp_after_lift", action="store_true")
    parser.add_argument("--no-live-actor-to-tcp-after-lift", dest="use_live_actor_to_tcp_after_lift", action="store_false")
    parser.set_defaults(use_live_actor_to_tcp_after_lift=True)
    parser.add_argument("--live-actor-to-tcp-after-lift-min-connection-potential", type=int, default=0)
    parser.add_argument("--allow-nominal-release-fallback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-nominal-release-fallback-max-connection-potential", type=int, default=0)
    parser.add_argument("--enable-capture-after-open-roles", default="")
    parser.add_argument("--capture-after-open-min-connection-potential", type=int, default=None)
    parser.add_argument("--adaptive-role-order", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--role-order-policy", choices=["adaptive", "given", "far_from_robot"], default="adaptive")
    parser.add_argument("--use-cached-pair-release", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loaded-state-settle-steps", type=int, default=8)
    parser.add_argument("--wall-release-profile", choices=["generic", "fixed_top_down"], default="generic")
    parser.add_argument("--fixed-top-down-open-gap", type=float, default=None)
    parser.add_argument("--wall-grasp-extra-tilt-degs", default="")
    parser.add_argument("--wall-grasp-extra-tilt-max-abs-deg", type=float, default=30.0)
    parser.add_argument("--fixed-top-down-extra-tilt-degs", default="")
    parser.add_argument("--fixed-top-down-extra-tilt-max-abs-deg", type=float, default=30.0)
    parser.add_argument("--fixed-top-down-extra-tilt-lift-mms", default="1.0,2.0")
    parser.add_argument("--fixed-top-down-extra-tilt-normal-bias-mms", default="0.0,1.5,-1.5")
    parser.add_argument("--retry-wall-grasp-extra-tilt-degs", default="")
    parser.add_argument("--retry-wall-grasp-extra-tilt-max-abs-deg", type=float, default=30.0)
    parser.add_argument("--retry-fixed-top-down-extra-tilt-degs", default="")
    parser.add_argument("--retry-fixed-top-down-extra-tilt-max-abs-deg", type=float, default=30.0)
    parser.add_argument("--retry-fixed-top-down-extra-tilt-lift-mms", default="1.0,2.0")
    parser.add_argument("--retry-fixed-top-down-extra-tilt-normal-bias-mms", default="0.0,1.5,-1.5")
    parser.add_argument("--retry-extra-tilt-start-role-attempt", type=int, default=1)
    parser.add_argument("--fixed-single-connection-release-min-clearance", type=float, default=0.0)
    parser.add_argument("--wall-release-approach-mode", choices=["auto", "top_down", "side_push", "both"], default="auto")
    parser.add_argument("--wall-release-side-approach-distance", type=float, default=0.035)
    parser.add_argument("--wall-release-side-approach-lift", type=float, default=0.010)
    parser.add_argument("--drive-force-limit", default=DEFAULT_MAGNET["drive_force_limit"])
    parser.add_argument("--magnet-attach-distance", default=DEFAULT_MAGNET["attach_distance"])
    parser.add_argument("--magnet-attract-distance", default=DEFAULT_MAGNET["attract_distance"])
    parser.add_argument("--magnet-detach-distance", default=DEFAULT_MAGNET["detach_distance"])
    parser.add_argument("--magnet-edge-sample-half-span", default=DEFAULT_MAGNET["edge_sample_half_span"])
    parser.add_argument("--magnet-connect-edge-sample-half-span", default=DEFAULT_MAGNET["connect_edge_sample_half_span"])
    parser.add_argument("--magnet-edge-sample-offsets", default=DEFAULT_MAGNET["edge_sample_offsets"])
    parser.add_argument("--magnet-connect-edge-sample-offsets", default=DEFAULT_MAGNET["connect_edge_sample_offsets"])
    parser.add_argument("--magnet-attract-stiffness", default=DEFAULT_MAGNET["attract_stiffness"])
    parser.add_argument("--magnet-attract-force-limit", default=DEFAULT_MAGNET["attract_force_limit"])
    parser.add_argument("--magnet-attract-torque-stiffness", default=DEFAULT_MAGNET["attract_torque_stiffness"])
    parser.add_argument("--magnet-attract-torque-limit", default=DEFAULT_MAGNET["attract_torque_limit"])
    parser.add_argument("--magnet-attract-normal-torque-stiffness", default=DEFAULT_MAGNET["attract_normal_torque_stiffness"])
    parser.add_argument("--magnet-attract-normal-torque-limit", default=DEFAULT_MAGNET["attract_normal_torque_limit"])
    parser.add_argument("--magnet-active-stiffness", default=DEFAULT_MAGNET["active_stiffness"])
    parser.add_argument("--magnet-active-damping", default=DEFAULT_MAGNET["active_damping"])
    parser.add_argument("--magnet-active-force-limit", default=DEFAULT_MAGNET["active_force_limit"])
    parser.add_argument("--magnet-active-edge-torque-scale", default=DEFAULT_MAGNET["active_edge_torque_scale"])
    parser.add_argument("--magnet-active-edge-torque-min-scale", default=DEFAULT_MAGNET["active_edge_torque_min_scale"])
    parser.add_argument("--magnet-active-normal-torque-scale", default=DEFAULT_MAGNET["active_normal_torque_scale"])
    parser.add_argument("--magnet-active-normal-torque-min-scale", default=DEFAULT_MAGNET["active_normal_torque_min_scale"])
    parser.add_argument("--magnet-active-torque-delay-steps", default=DEFAULT_MAGNET["active_torque_delay_steps"])
    parser.add_argument("--magnet-active-torque-ramp-steps", default=DEFAULT_MAGNET["active_torque_ramp_steps"])
    parser.add_argument("--magnet-active-torque-max-point-error", default=DEFAULT_MAGNET["active_torque_max_point_error"])
    parser.add_argument("--magnet-floor-support-score", default=DEFAULT_MAGNET["floor_support_score"])
    parser.add_argument("--magnet-support-connection-score-scale", default=DEFAULT_MAGNET["support_connection_score_scale"])
    parser.add_argument("--magnet-multi-connection-support-bonus", default=DEFAULT_MAGNET["multi_connection_support_bonus"])
    parser.add_argument("--magnet-drive-stiffness", default=DEFAULT_MAGNET["drive_stiffness"])
    parser.add_argument("--magnet-drive-damping", default=DEFAULT_MAGNET["drive_damping"])
    parser.add_argument("--magnet-drive-force-limit", default=DEFAULT_MAGNET["drive_force_limit"])
    parser.add_argument("--magnet-drive-angular-stiffness", default=DEFAULT_MAGNET["drive_angular_stiffness"])
    parser.add_argument("--magnet-drive-angular-damping", default=DEFAULT_MAGNET["drive_angular_damping"])
    parser.add_argument("--magnet-drive-angular-force-limit", default=DEFAULT_MAGNET["drive_angular_force_limit"])
    parser.add_argument("--loaded-state-perturb-dx", type=float, default=0.0)
    parser.add_argument("--loaded-state-perturb-dy", type=float, default=0.0)
    parser.add_argument("--loaded-state-perturb-dz", type=float, default=0.0)
    parser.add_argument("--loaded-state-perturb-yaw-deg", type=float, default=0.0)
    parser.add_argument("--loaded-state-perturb-origin-role", default="floor")
    parser.add_argument("--loaded-state-perturb-roles", default="floor,right_wall,back_wall,left_wall")
    parser.add_argument("--loaded-state-perturb-target-roles", default="floor,right_wall,back_wall,left_wall,front_wall")
    parser.add_argument("--initial-assembly-offset-x", type=float, default=0.0)
    parser.add_argument("--initial-assembly-offset-y", type=float, default=0.0)
    parser.add_argument("--initial-assembly-offset-z", type=float, default=0.0)
    parser.add_argument("--initial-assembly-offset-actor-roles", default="floor")
    parser.add_argument("--initial-assembly-offset-target-roles", default="floor,right_wall,back_wall,left_wall,front_wall")
    parser.add_argument("--initial-actor-jitter-xy", type=float, default=0.0)
    parser.add_argument("--initial-actor-jitter-seed", type=int, default=0)
    parser.add_argument("--initial-actor-jitter-roles", default="right_wall,back_wall,left_wall,front_wall")
    parser.add_argument("--initial-actor-jitter-min-start-distance", type=float, default=0.055)
    parser.add_argument("--initial-actor-jitter-min-target-distance", type=float, default=0.035)
    parser.add_argument("--initial-actor-jitter-max-sample-attempts", type=int, default=100)
    parser.add_argument("--max-position-error", type=float, default=0.035)
    parser.add_argument("--max-orientation-error-deg", type=float, default=35.0)
    parser.add_argument("--all-roles-max-position-error", type=float, default=0.035)
    parser.add_argument("--all-roles-max-orientation-error-deg", type=float, default=35.0)
    parser.add_argument("--all-roles-max-drift-position", type=float, default=0.008)
    parser.add_argument("--all-roles-max-drift-orientation-deg", type=float, default=5.0)
    parser.add_argument("--all-roles-max-linear-speed", type=float, default=0.08)
    parser.add_argument("--all-roles-max-angular-speed", type=float, default=1.0)
    parser.add_argument("--final-max-connection-point-error", type=float, default=0.010)
    parser.add_argument("--final-max-connection-normal-angle-deg", type=float, default=8.0)
    parser.add_argument("--final-max-connection-edge-angle-deg", type=float, default=8.0)
    parser.add_argument("--release-open-gripper-value", type=float, default=0.03)
    parser.add_argument("--return-home-steps", type=int, default=0)
    parser.add_argument("--full-open-after-retreat-steps", type=int, default=0)
    parser.add_argument("--pre-open-min-active-connections", type=int, default=1)
    parser.add_argument("--pre-open-connection-correction-attempts", type=int, default=1)
    parser.add_argument("--pre-open-connection-correction-max-pose-error", type=float, default=0.08)
    parser.add_argument("--pre-open-connection-correction-max-orientation-error-deg", type=float, default=70.0)
    parser.add_argument("--pre-open-connection-correction-trigger-pose-error", type=float, default=0.012)
    parser.add_argument("--pre-open-connection-correction-trigger-orientation-error-deg", type=float, default=5.0)
    parser.add_argument("--require-active-connection-before-open", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--release-correction-attempts", type=int, default=1)
    parser.add_argument("--return-neutral-steps", type=int, default=8)
    parser.add_argument("--return-neutral-motion-plan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return-neutral-motion-plan-timeout", type=float, default=2.5)
    parser.add_argument("--return-neutral-motion-plan-max-attempts", type=int, default=1)
    parser.add_argument("--return-neutral-motion-plan-enable-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return-neutral-motion-plan-graph-seeds", type=int, default=1)
    parser.add_argument("--return-neutral-motion-plan-max-waypoints", type=int, default=0)
    parser.add_argument("--next-cycle-plan-prefetch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-rear-collision-wall", action="store_true")
    parser.add_argument("--rear-collision-wall-frame", choices=["world_back_x", "robot_back", "floor_back_y"], default="world_back_x")
    parser.add_argument("--rear-collision-wall-distance", type=float, default=0.35)
    parser.add_argument("--rear-collision-wall-width", type=float, default=1.80)
    parser.add_argument("--rear-collision-wall-height", type=float, default=1.20)
    parser.add_argument("--add-overhead-collision-wall", action="store_true")
    defaults = _parser_default_map(parser)
    args = parser.parse_args()
    _apply_strategy_preset(args, defaults)
    if bool(getattr(args, "execute_real", False)):
        print("[real] validating attempts in simulation first; only a successful manifest will be replayed on RM75", flush=True)
    if bool(getattr(args, "stream_child_output", False)) and bool(getattr(args, "inline_child_planner", False)):
        args.inline_child_planner = False
    if args.fixed_top_down_open_gap is None:
        args.fixed_top_down_open_gap = 0.008 if str(args.wall_release_profile) == "fixed_top_down" else 0.0
    if args.capture_after_open_min_connection_potential is None:
        args.capture_after_open_min_connection_potential = 0
    if args.release_robust_require_same_release_index is None:
        args.release_robust_require_same_release_index = False
    if args.require_active_connection_before_open is None:
        args.require_active_connection_before_open = str(args.wall_release_profile) != "fixed_top_down"
    if str(args.strategy_preset) == "pair_first_stable_v1" and float(args.attempt_timeout_sec) == float(
        defaults["attempt_timeout_sec"]
    ):
        args.attempt_timeout_sec = 900.0
    result = run(args)
    print(json.dumps(_json_ready({"report": str(Path(result["out_dir"]) / "standard_build_report.json"), "final": result.get("final")}), ensure_ascii=False, indent=2))
    sys.exit(0 if result.get("final", {}).get("success") else 1)


if __name__ == "__main__":
    main()
