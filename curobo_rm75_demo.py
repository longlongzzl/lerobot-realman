#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal cuRobo RM75 demo. This version uses a URDF-only robot config with the "
            "gripper joints locked and collision spheres disabled, so MotionGen runs in free space."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("fk", "ik", "motiongen"),
        default="ik",
        help="Which demo mode to run.",
    )
    parser.add_argument(
        "--curobo-root",
        type=Path,
        default=DEFAULT_CUROBO_ROOT,
        help="Path to the local cuRobo repository root.",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_RM75_URDF,
        help="Path to the RM75 URDF.",
    )
    parser.add_argument(
        "--base-link",
        type=str,
        default="base_link",
        help="Base link name passed to cuRobo.",
    )
    parser.add_argument(
        "--ee-link",
        type=str,
        default="gripper_tcp",
        help="End-effector link name passed to cuRobo.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device to use, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--torch-extensions-dir",
        type=Path,
        default=DEFAULT_TORCH_EXTENSIONS_DIR,
        help="Writable directory used by torch JIT extensions when cuRobo compiles CUDA kernels.",
    )
    parser.add_argument(
        "--cuda-arch-list",
        type=str,
        default=None,
        help="Optional TORCH_CUDA_ARCH_LIST override, for example 8.6 or 8.9+PTX.",
    )
    parser.add_argument(
        "--gripper-lock",
        type=float,
        default=0.6,
        help="Joint value used to lock all six gripper joints.",
    )
    parser.add_argument(
        "--start-q",
        type=float,
        nargs=7,
        default=None,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"),
        help="Optional start arm joint configuration in radians.",
    )
    parser.add_argument(
        "--delta-x",
        type=float,
        default=0.0,
        help="Goal pose offset in x relative to the start FK pose, in meters.",
    )
    parser.add_argument(
        "--delta-y",
        type=float,
        default=0.0,
        help="Goal pose offset in y relative to the start FK pose, in meters.",
    )
    parser.add_argument(
        "--delta-z",
        type=float,
        default=0.05,
        help="Goal pose offset in z relative to the start FK pose, in meters.",
    )
    parser.add_argument(
        "--goal-pose",
        type=float,
        nargs=7,
        default=None,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
        help="Optional absolute goal pose. If omitted, a goal is built from FK(start_q)+delta.",
    )
    parser.add_argument(
        "--num-ik-seeds",
        type=int,
        default=64,
        help="Number of seeds to use for IK.",
    )
    parser.add_argument(
        "--num-trajopt-seeds",
        type=int,
        default=1,
        help="Number of trajectory optimization seeds to use inside MotionGen.",
    )
    parser.add_argument(
        "--num-graph-seeds",
        type=int,
        default=1,
        help="Number of graph planner seeds to use inside MotionGen.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum MotionGen attempts.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="MotionGen timeout in seconds.",
    )
    parser.add_argument(
        "--enable-graph",
        action="store_true",
        help="Enable graph planning inside MotionGen.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Warm up MotionGen before solving. Disabled by default for this minimal demo.",
    )
    return parser


def ensure_curobo_on_path(curobo_root: Path) -> Path:
    src_dir = curobo_root.expanduser().resolve() / "src"
    if not src_dir.is_dir():
        raise FileNotFoundError(f"cuRobo src directory was not found: {src_dir}")
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return src_dir


def prepare_runtime_env(args) -> None:
    torch_extensions_dir = args.torch_extensions_dir.expanduser().resolve()
    torch_extensions_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(torch_extensions_dir))
    if args.cuda_arch_list:
        os.environ["TORCH_CUDA_ARCH_LIST"] = str(args.cuda_arch_list)
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

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False in the current session. "
            "cuRobo needs a working NVIDIA driver / CUDA runtime, and its CUDA extensions "
            "must compile against a visible GPU architecture. "
            f"Current TORCH_EXTENSIONS_DIR={os.environ['TORCH_EXTENSIONS_DIR']}. "
            "If you are in a remote or sandboxed shell, run this demo in a GPU-enabled environment "
            "or pass --cuda-arch-list with the target architecture after the NVIDIA driver is available."
        )
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"


def import_curobo_modules():
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.types.state import JointState
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

    return {
        "torch": torch,
        "TensorDeviceType": TensorDeviceType,
        "Pose": Pose,
        "RobotConfig": RobotConfig,
        "JointState": JointState,
        "IKSolver": IKSolver,
        "IKSolverConfig": IKSolverConfig,
        "MotionGen": MotionGen,
        "MotionGenConfig": MotionGenConfig,
        "MotionGenPlanConfig": MotionGenPlanConfig,
    }


def build_rm75_robot_cfg_dict(args) -> dict:
    urdf_path = args.urdf.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"RM75 URDF was not found: {urdf_path}")

    lock_joints = {joint_name: float(args.gripper_lock) for joint_name in GRIPPER_JOINT_NAMES}
    robot_root = urdf_path.parent.parent
    return {
        "robot_cfg": {
            "kinematics": {
                "use_usd_kinematics": False,
                "urdf_path": str(urdf_path),
                "asset_root_path": str(robot_root),
                "base_link": str(args.base_link),
                "ee_link": str(args.ee_link),
                "link_names": [str(args.ee_link)] + list(TRACKED_LINK_NAMES),
                "collision_link_names": None,
                "collision_spheres": None,
                "collision_sphere_buffer": 0.0,
                "extra_collision_spheres": None,
                "self_collision_ignore": None,
                "self_collision_buffer": None,
                "use_global_cumul": True,
                "mesh_link_names": None,
                "lock_joints": lock_joints,
                "extra_links": None,
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


def make_tensor_args(torch_mod, TensorDeviceType, device_str: str):
    device = torch_mod.device(device_str)
    return TensorDeviceType(device=device)


def get_start_q(args) -> np.ndarray:
    values = DEFAULT_RETRACT_CONFIG if args.start_q is None else [float(v) for v in args.start_q]
    q = np.asarray(values, dtype=np.float32).reshape(7)
    return q


def make_start_state(torch_mod, JointState, tensor_args, start_q: np.ndarray):
    q_tensor = torch_mod.as_tensor(start_q, device=tensor_args.device, dtype=tensor_args.dtype).view(1, -1)
    return JointState.from_position(q_tensor, joint_names=list(ARM_JOINT_NAMES))


def build_goal_pose(args, PoseCls, tensor_args, current_pose):
    if args.goal_pose is not None:
        goal = [float(v) for v in args.goal_pose]
        return PoseCls.from_list(goal, tensor_args=tensor_args)

    goal_position = current_pose.position.clone()
    goal_position[0, 0] += float(args.delta_x)
    goal_position[0, 1] += float(args.delta_y)
    goal_position[0, 2] += float(args.delta_z)
    goal_quaternion = current_pose.quaternion.clone()
    return PoseCls(position=goal_position, quaternion=goal_quaternion)


def first_success_solution(torch_mod, solution, success):
    if solution is None or success is None:
        return None
    success_idx = torch_mod.nonzero(success.reshape(-1), as_tuple=False).view(-1)
    if success_idx.numel() == 0:
        return None
    flat_solution = solution.reshape(-1, solution.shape[-1])
    return flat_solution[int(success_idx[0])]


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def print_pose(label: str, pose) -> None:
    position = np.round(to_numpy(pose.position).reshape(-1, 3)[0], 6)
    quaternion = np.round(to_numpy(pose.quaternion).reshape(-1, 4)[0], 6)
    print(f"{label} position (m): {position.tolist()}")
    print(f"{label} quaternion (wxyz): {quaternion.tolist()}")


def run_fk_demo(motion_gen, start_state) -> int:
    current_pose = motion_gen.compute_kinematics(start_state).ee_pose
    print_pose("FK", current_pose)
    return 0


def run_ik_demo(mods, robot_cfg, tensor_args, start_state, goal_pose, args) -> int:
    ik_config = mods["IKSolverConfig"].load_from_robot_config(
        robot_cfg,
        None,
        position_threshold=0.005,
        rotation_threshold=0.05,
        num_seeds=int(args.num_ik_seeds),
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=False,
    )
    ik_solver = mods["IKSolver"](ik_config)
    result = ik_solver.solve_single(
        goal_pose,
        retract_config=start_state.position.clone(),
        seed_config=start_state.position.view(1, 1, -1).clone(),
        num_seeds=int(args.num_ik_seeds),
    )

    print("IK success:", bool(result.success.item()))
    print("IK solve_time (s):", float(result.solve_time))
    print("IK position_error (m):", float(result.position_error.reshape(-1)[0].item()))
    print("IK rotation_error:", float(result.rotation_error.reshape(-1)[0].item()))

    best_q = first_success_solution(mods["torch"], result.solution, result.success)
    if best_q is not None:
        print("IK solution q (rad):", np.round(to_numpy(best_q), 6).tolist())
        return 0

    print("No feasible IK solution was returned.")
    return 1


def run_motion_gen_demo(mods, robot_cfg, tensor_args, start_state, goal_pose, args) -> int:
    motion_gen_config = mods["MotionGenConfig"].load_from_robot_config(
        robot_cfg,
        None,
        tensor_args=tensor_args,
        num_ik_seeds=int(args.num_ik_seeds),
        num_graph_seeds=int(args.num_graph_seeds),
        num_trajopt_seeds=int(args.num_trajopt_seeds),
        interpolation_dt=0.02,
        use_cuda_graph=False,
        self_collision_check=False,
        self_collision_opt=False,
    )
    motion_gen = mods["MotionGen"](motion_gen_config)
    if bool(args.warmup):
        motion_gen.warmup(enable_graph=bool(args.enable_graph), warmup_js_trajopt=False)
    motion_gen.reset_seed()
    ik_result = motion_gen.solve_ik(
        goal_pose,
        retract_config=start_state.position.clone(),
        seed_config=start_state.position.view(1, 1, -1).clone(),
        return_seeds=int(args.num_trajopt_seeds),
        num_seeds=int(args.num_ik_seeds),
        use_nn_seed=False,
    )
    motiongen_ik_success = int(mods["torch"].count_nonzero(ik_result.success).item())
    print("MotionGen internal IK successes:", motiongen_ik_success)
    print("MotionGen internal IK solve_time (s):", float(ik_result.solve_time))

    plan_config = mods["MotionGenPlanConfig"](
        enable_graph=bool(args.enable_graph),
        enable_opt=True,
        max_attempts=int(args.max_attempts),
        timeout=float(args.timeout),
        num_ik_seeds=int(args.num_ik_seeds),
        num_graph_seeds=int(args.num_graph_seeds),
        num_trajopt_seeds=int(args.num_trajopt_seeds),
    )
    motion_gen.reset_seed()
    result = motion_gen.plan_single(start_state, goal_pose, plan_config)

    success = bool(result.success.reshape(-1)[0].item())
    print("MotionGen success:", success)
    print("MotionGen status:", result.status)
    print("MotionGen solve_time (s):", float(result.solve_time))
    print("MotionGen ik_time (s):", float(result.ik_time))
    print("MotionGen trajopt_time (s):", float(result.trajopt_time))

    if not success:
        return 1

    traj = result.get_interpolated_plan()
    q_path = to_numpy(traj.position)
    print("Trajectory waypoints:", int(q_path.shape[0]))
    print("Trajectory dt (s):", float(result.interpolation_dt))
    print("Trajectory first q (rad):", np.round(q_path[0], 6).tolist())
    print("Trajectory last q (rad):", np.round(q_path[-1], 6).tolist())
    return 0


def main() -> int:
    args = build_arg_parser().parse_args()
    ensure_curobo_on_path(args.curobo_root)
    prepare_runtime_env(args)
    print(
        "[info] cuRobo may JIT-compile several CUDA extensions on first run. "
        "That is expected and can take a few minutes."
    )
    mods = import_curobo_modules()

    robot_cfg_dict = build_rm75_robot_cfg_dict(args)
    tensor_args = make_tensor_args(mods["torch"], mods["TensorDeviceType"], args.device)
    robot_cfg = mods["RobotConfig"].from_dict(robot_cfg_dict, tensor_args=tensor_args)

    motion_gen = mods["MotionGen"](
        mods["MotionGenConfig"].load_from_robot_config(
            robot_cfg,
            None,
            tensor_args=tensor_args,
            interpolation_dt=0.02,
            use_cuda_graph=False,
            self_collision_check=False,
            self_collision_opt=False,
        )
    )

    start_q = get_start_q(args)
    start_state = make_start_state(mods["torch"], mods["JointState"], tensor_args, start_q)
    current_pose = motion_gen.compute_kinematics(start_state).ee_pose
    goal_pose = build_goal_pose(args, mods["Pose"], tensor_args, current_pose)

    print("Using cuRobo:", str((args.curobo_root.expanduser().resolve() / "src")))
    print("Using RM75 URDF:", str(args.urdf.expanduser().resolve()))
    print("Torch extensions dir:", os.environ["TORCH_EXTENSIONS_DIR"])
    print("TORCH_CUDA_ARCH_LIST:", os.environ["TORCH_CUDA_ARCH_LIST"])
    print("Mode:", args.mode)
    print("Base link:", args.base_link)
    print("EE link:", args.ee_link)
    print("Arm joint names:", ARM_JOINT_NAMES)
    print("Locked gripper joints:", GRIPPER_JOINT_NAMES)
    print("Start q (rad):", np.round(start_q, 6).tolist())
    print("IK seeds:", int(args.num_ik_seeds))
    print("TrajOpt seeds:", int(args.num_trajopt_seeds))
    print("Graph seeds:", int(args.num_graph_seeds))
    print("Warmup enabled:", bool(args.warmup))
    print_pose("Start FK", current_pose)
    print_pose("Goal", goal_pose)
    print(
        "[note] This demo uses lock_joints for the gripper and collision_spheres=null, "
        "so MotionGen is free-space only."
    )

    if args.mode == "fk":
        return run_fk_demo(motion_gen, start_state)
    if args.mode == "ik":
        return run_ik_demo(mods, robot_cfg, tensor_args, start_state, goal_pose, args)
    return run_motion_gen_demo(mods, robot_cfg, tensor_args, start_state, goal_pose, args)


if __name__ == "__main__":
    raise SystemExit(main())
