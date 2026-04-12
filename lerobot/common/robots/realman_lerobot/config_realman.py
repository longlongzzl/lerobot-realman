from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional

from .robot_base import RobotConfig

# RealMan SDK enum lives in Robotic_Arm.rm_robot_interface
try:
    from Robotic_Arm.rm_robot_interface import rm_thread_mode_e  # type: ignore
except Exception:
    rm_thread_mode_e = None  # allow import even if SDK not installed


@dataclass
class RealManArmConfig(RobotConfig):
    ip: str = "192.168.1.18"
    port: int = 8080

    # thread mode: doc commonly uses RM_TRIPLE_MODE_E
    thread_mode: Any = getattr(rm_thread_mode_e, "RM_TRIPLE_MODE_E", None)

    # RealMan joint IO is degrees; expose degrees or radians to upper layer
    use_degrees: bool = True

    # rm_movej params
    v: int = 20
    r: int = 0
    connect: int = 0
    block: int = 0  # in multi-thread mode: 0 non-blocking, 1 blocking

    # joint naming
    joint_names: Optional[List[str]] = None

    # safety: cap each step
    max_relative_target: Optional[float] = None  # in same unit as exposed action (deg or rad)

    # gripper
    enable_gripper: bool = True
    gripper_normed: bool = True  # expose gripper.pos in [0,1]
    gripper_block: bool = False
    gripper_timeout_s: int = 10

    # optional: do a ping read at connect to validate link
    validate_on_connect: bool = True
