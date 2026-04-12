from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

from .config_realman import RealManArmConfig
from .errors import DeviceAlreadyConnectedError, DeviceNotConnectedError, RobotAPIError
from .robot_base import Robot
from .utils import ensure_safe_goal_position

# RealMan SDK
from Robotic_Arm.rm_robot_interface import *  # type: ignore
class RealManArm(Robot):
    """
    RealMan(无线方舟) -> LeRobot style wrapper.

    Observation keys:
      - "joint1.pos" .. "jointN.pos" (float)
      - "gripper.pos" (float, optional)
    Same for action keys.

    Units:
      - RealMan SDK uses degrees for joints.
      - If config.use_degrees=False, this wrapper exposes radians to caller.
    """

    config_class = RealManArmConfig
    name = "realman_arm"

    def __init__(self, config: RealManArmConfig):
        super().__init__(config)
        self.config = config

        self.arm: Optional[RoboticArm] = None
        self._connected: bool = False

        self._joint_names: List[str] = list(config.joint_names) if config.joint_names else []
        self._dof: Optional[int] = None
        self._last_gripper_value: float = 0.0
        self._last_gripper_counts: int = 0

    # ---------------- helpers ----------------
    def _check_ok(self, status: int, what: str) -> None:
        if int(status) != 0:
            raise RobotAPIError(f"RealMan API call failed: {what}, status={status}")

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

    def _deg2rad(self, x: float) -> float:
        return x * math.pi / 180.0

    def _rad2deg(self, x: float) -> float:
        return x * 180.0 / math.pi

    def _list_deg_to_exposed(self, deg_list: List[float]) -> List[float]:
        if self.config.use_degrees:
            return [float(x) for x in deg_list]
        return [self._deg2rad(float(x)) for x in deg_list]

    def _list_exposed_to_deg(self, xs: List[float]) -> List[float]:
        if self.config.use_degrees:
            return [float(x) for x in xs]
        return [self._rad2deg(float(x)) for x in xs]

    def _infer_dof_and_names(self) -> None:
        """
        Infer dof by reading joint degrees once.
        If no joint_names provided, generate joint1..jointN.
        """
        assert self.arm is not None
        code, deg = self.arm.rm_get_joint_degree()
        self._check_ok(code, "rm_get_joint_degree (infer dof)")
        deg_list = list(deg)
        self._dof = len(deg_list)

        if not self._joint_names:
            self._joint_names = [f"joint{i+1}" for i in range(self._dof)]

        # If user provided joint_names but length mismatch, fail early
        if len(self._joint_names) != self._dof:
            raise ValueError(
                f"joint_names length ({len(self._joint_names)}) != robot dof ({self._dof}). "
                f"Either provide correct names or leave joint_names=None to auto-generate."
            )

    def _normed_from_gripper_value(self, value: int) -> float:
        value = max(0, min(9000, int(value)))
        return max(0.0, min(1.0, value / 9000.0))


    def _gripper_value_from_normed(self, g: float) -> int:
        g = max(0.0, min(1.0, g))
        # 0=open, 1=fully closed
        return int(round(g * 9000.0))

    def _get_gripper_modbus(self) -> float:
        assert self.arm is not None
        # Read holding register 258 (status/trigger) then 259 (position)
        param = rm_peripheral_read_write_params_t(1, 258, 1)
        code, _ = self.arm.rm_read_holding_registers(param)
        self._check_ok(int(code), "rm_read_holding_registers (gripper status)")

        param = rm_peripheral_read_write_params_t(1, 259, 1)
        code, value = self.arm.rm_read_holding_registers(param)
        self._check_ok(int(code), "rm_read_holding_registers (gripper position)")
        return self._normed_from_gripper_value(int(value))

    def _set_gripper_modbus(self, g: float) -> None:
        assert self.arm is not None
        gripper_value_cmd = self._gripper_value_from_normed(g)

        # Set target position
        param = rm_peripheral_read_write_params_t(1, 258, 1, 2)
        code = self.arm.rm_write_registers(param, [0, gripper_value_cmd, 0, 0])
        self._check_ok(int(code), "rm_write_registers (gripper target)")

        # Execute
        param = rm_peripheral_read_write_params_t(1, 264, 1)
        code = self.arm.rm_write_single_register(param, 1)
        self._check_ok(int(code), "rm_write_single_register (gripper execute)")

    # ---------------- LeRobot-like feature specs ----------------
    @property
    def observation_features(self) -> Dict[str, Any]:
        ft: Dict[str, Any] = {f"{jn}.pos": float for jn in self._joint_names}
        if self.config.enable_gripper:
            ft["gripper.pos"] = float
        return ft

    @property
    def action_features(self) -> Dict[str, Any]:
        return self.observation_features

    # ---------------- connection ----------------
    @property
    def is_connected(self) -> bool:
        return self._connected and (self.arm is not None)

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Create SDK instance
        if self.config.thread_mode is None:
            raise RuntimeError(
                "RealMan rm_thread_mode_e not available. "
                "Check RealMan SDK installation/import."
            )

        self.arm = RoboticArm(self.config.thread_mode)

        # create arm handle
        # 根据官方API文档：rm_create_robot_arm(ip, port) 返回 handle 对象
        handle = self.arm.rm_create_robot_arm(self.config.ip, self.config.port)
        # handle.id exists in docs, but we don't strictly need it
        # 注意：如果连接失败，SDK可能会抛出异常或返回None，这里依赖SDK的错误处理

        self._connected = True

        # Infer dof & joint names and optionally validate link
        self._infer_dof_and_names()

        if self.config.validate_on_connect:
            # do a quick state read to validate communication
            _ = self.get_observation()

        self.configure()

    def configure(self) -> None:
        # Keep as no-op by default.
        # You can add: set init pose, speed limits, etc. if needed.
        return None

    # ---------------- observation ----------------
    def get_observation(self) -> Dict[str, Any]:
        self._require_connected()
        assert self.arm is not None
        assert self._dof is not None

        code, deg = self.arm.rm_get_joint_degree()
        self._check_ok(code, "rm_get_joint_degree")
        deg_list = list(deg)
        if len(deg_list) != self._dof:
            # SDK/robot changed? keep robust.
            self._dof = len(deg_list)
            if len(self._joint_names) != self._dof:
                self._joint_names = [f"joint{i+1}" for i in range(self._dof)]

        xs = self._list_deg_to_exposed(deg_list)

        obs: Dict[str, Any] = {f"{jn}.pos": float(xs[i]) for i, jn in enumerate(self._joint_names)}

        # if self.config.enable_gripper:
        #     # (code, dict)
        #     try:
        #         st, g = self.arm.rm_get_gripper_state()
        #         if int(st) == 0 and isinstance(g, dict):
        #             # documented field: actpos (actual position)
        #             actpos = g.get("actpos", None)
        #             if actpos is None:
        #                 obs["gripper.pos"] = float("nan")
        #             else:
        #                 actpos = float(actpos)  # typically 0..1000
        #                 obs["gripper.pos"] = (actpos / 1000.0) if self.config.gripper_normed else actpos
        #         else:
        #             obs["gripper.pos"] = float("nan")
        #     except Exception:
        #         obs["gripper.pos"] = float("nan")
        if self.config.enable_gripper:
            try:
                if self.config.gripper_normed:
                    obs["gripper.pos"] = float(self._get_gripper_modbus())
                else:
                    # If not normed, still return 0..9000 raw value for debugging
                    param = rm_peripheral_read_write_params_t(1, 259, 1)
                    code, value = self.arm.rm_read_holding_registers(param)
                    self._check_ok(int(code), "rm_read_holding_registers (gripper position raw)")
                    obs["gripper.pos"] = float(int(value))
            except Exception:
                obs["gripper.pos"] = float("nan")

        return obs

    # ---------------- action ----------------
    # def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
    #     print("real man get action",action)
    #     self._require_connected()
    #     assert self.arm is not None
    #     assert self._dof is not None
    #
    #     sent: Dict[str, Any] = {}
    #
    #     # ---------------- joints ----------------
    #     joint_keys = [f"{jn}.pos" for jn in self._joint_names]
    #     all_joint = all(k in action for k in joint_keys)
    #     if not all_joint:
    #         # 你说你不会传少于 7 个，这里直接报错更安全
    #         raise ValueError("send_action expects all 7 joint targets; partial joint action is not supported.")
    #
    #     goal_exposed = [float(action[f"{jn}.pos"]) for jn in self._joint_names]
    #
    #     # 可选：用“上一次目标”做每步限幅（不读真机，适合 follow/交互）
    #     if getattr(self.config, "max_relative_target", None) is not None:
    #         if getattr(self, "_last_goal_exposed", None) is None:
    #             self._last_goal_exposed = goal_exposed[:]  # 初始化
    #         gp = {
    #             self._joint_names[i]: (goal_exposed[i], self._last_goal_exposed[i])
    #             for i in range(self._dof)
    #         }
    #         clipped = ensure_safe_goal_position(gp, self.config.max_relative_target)
    #         goal_exposed = [float(clipped[self._joint_names[i]]) for i in range(self._dof)]
    #         self._last_goal_exposed = goal_exposed[:]
    #
    #     # SDK: rm_movej_follow 期望的是“角度”
    #     goal_deg = self._list_exposed_to_deg(goal_exposed)
    #
    #     rc = self.arm.rm_movej_follow(goal_deg)
    #     if int(rc) != 0:
    #         print(f"[WARN] rm_movej_follow failed rc={rc}, goal_deg={goal_deg}")
    #     else:
    #         for i, jn in enumerate(self._joint_names):
    #             sent[f"{jn}.pos"] = float(goal_exposed[i])
    #
    #     # ---------------- gripper ----------------
    #     # if self.config.enable_gripper and ("gripper.pos" in action):
    #     #     obst=self.arm.rm_get_gripper_state()
    #     #     print("obst:",obst)
    #     #     g_raw = float(action["gripper.pos"])
    #     #     # 你的新输入范围：0 ~ 0.91
    #     #     G_MAX_NORM = 0.91
    #     #     G_MAX_COUNTS = 9000.0
    #     #     # g：归一化(0~0.91) 或 counts
    #     #     if getattr(self.config, "gripper_normed", False):
    #     #         # clamp 到 0~0.91
    #     #         g_raw = max(0.0, min(G_MAX_NORM, g_raw))
    #     #         # 映射到 0~1
    #     #         g01 = g_raw / G_MAX_NORM
    #     #         # 保留你原来的反向逻辑：0 -> 9000, 0.91 -> 0
    #     #         # g01 = 1.0 - g01
    #     #         # 放大到 0~9000
    #     #         pos = int(round(g01 * G_MAX_COUNTS))
    #     #     else:
    #     #         # 非 normed：认为直接传 counts
    #     #         pos = int(round(g_raw))
    #     #     # 允许 0~9000（你原来是 1~9000）
    #     #     pos = max(0, min(9000, pos))
    #     #     print("moving gripper pos=", pos)
    #     #     obst = self.arm.rm_get_gripper_state()
    #     #     print("obst:", obst)
    #         try:
    #             obst = self.arm.rm_get_gripper_state()
    #             print("obst:", obst)
    #         except Exception as exc:
    #             obst = None
    #             print("[WARN] rm_get_gripper_state failed:", exc)
    #
    #         g_raw = float(action["gripper.pos"])
    #
    #         G_MAX_COUNTS = 9000.0  # 满行程 counts（按你之前的定义）
    #
    #         # --- 1) 把输入解释为“增量”，允许负值 ---
    #         if getattr(self.config, "gripper_normed", False):
    #             # normed: 认为 g_raw 是比例增量（比如 ±0.1），映射到 counts 增量
    #             g_raw = max(-1.0, min(1.0, g_raw))  # 保护一下，允许负值
    #             delta_counts = int(round(g_raw * G_MAX_COUNTS))
    #         else:
    #             # non-normed: 认为直接传 counts 增量（允许负值）
    #             delta_counts = int(round(g_raw))
    #         print("delta_counts:", delta_counts)
    #         #增量测试：
    #         print("modbus设置结果", self.arm.rm_set_modbus_mode(1, 115200, 2))
    #         param = rm_peripheral_read_write_params_t(1, 259, 1)
    #         try:
    #             ret, val = self.arm.rm_read_holding_registers(param)
    #             if int(ret) == 0 and isinstance(val, (int, float)):
    #                 print("读保持寄存器:", val)
    #                 current_counts = int(val)
    #                 self._last_gripper_counts = current_counts
    #             else:
    #                 print(f"[WARN] read holding register failed ret={ret}, val={val}, fallback to last={self._last_gripper_counts}")
    #                 current_counts = self._last_gripper_counts
    #         except Exception as exc:
    #             print(f"[WARN] rm_read_holding_registers exception: {exc}; fallback to last={self._last_gripper_counts}")
    #             current_counts = self._last_gripper_counts
    #
    #         print("current_counts delta_counts:",current_counts,delta_counts)
    #         pos = current_counts + delta_counts
    #         print("pos:", pos)
    #         self._last_gripper_counts = pos
    #         # 可选：夹爪 deadband，避免每帧都发（减少抖动/阻塞风险）
    #         deadband = float(getattr(self.config, "gripper_deadband_counts", 0))  # 例如 20~50
    #         last_pos = getattr(self, "_last_gripper_pos", None)
    #         # if last_pos is None or abs(pos - last_pos) > deadband:
    #         if True:
    #             gripper_block = bool(getattr(self.config, "gripper_block", False))
    #             timeout_s = int(getattr(self.config, "gripper_timeout_s", 2))
    #             rcg = None
    #             for attempt in range(3):
    #                 rcg = self.arm.rm_set_gripper_position(pos, gripper_block, timeout_s)
    #                 print("set gripper result=pos", rcg,pos)
    #                 if int(rcg) == 0:
    #                     break
    #                 print(f"[WARN] rm_set_gripper_position failed status={rcg}, retry {attempt + 1}/3")
    #             self._check_ok(int(rcg), "rm_set_gripper_position")
    #             self._last_gripper_pos = pos
    #
    #         sent["gripper.pos"] = float(action["gripper.pos"])
    #
    #     return sent

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._require_connected()
        assert self.arm is not None
        assert self._dof is not None

        sent: Dict[str, Any] = {}

        # ---- joints ----
        # Determine if any joint targets are provided
        joint_keys = [f"{jn}.pos" for jn in self._joint_names]
        provided_joint = any(k in action for k in joint_keys)

        if provided_joint:
            # Read current position (degrees) for merging unspecified joints + for safety clipping
            code, cur_deg = self.arm.rm_get_joint_degree()
            self._check_ok(code, "rm_get_joint_degree (merge action)")
            cur_deg_list = list(cur_deg)

            # Build goal in "exposed unit"
            cur_exposed = self._list_deg_to_exposed(cur_deg_list)

            goal_exposed: List[float] = []
            for i, jn in enumerate(self._joint_names):
                k = f"{jn}.pos"
                if k in action:
                    goal_exposed.append(float(action[k]))
                else:
                    goal_exposed.append(float(cur_exposed[i]))

            # Optional safety: clip per-step delta (in exposed units)
            if self.config.max_relative_target is not None:
                gp = {
                    self._joint_names[i]: (goal_exposed[i], cur_exposed[i])
                    for i in range(self._dof)
                }
                clipped = ensure_safe_goal_position(gp, self.config.max_relative_target)
                goal_exposed = [float(clipped[self._joint_names[i]]) for i in range(self._dof)]

            # Convert to degrees for SDK
            goal_deg = self._list_exposed_to_deg(goal_exposed)

            # Call movej: rm_movej(joint, v, r, connect, block) -> int
            rc = self.arm.rm_movej(goal_deg, self.config.v, self.config.r, self.config.connect, self.config.block)
            if int(rc) != 0:
                print(f"[WARN] rm_movej failed rc={rc}, goal_deg={goal_deg}, cur_deg={cur_deg_list}")
                return sent  # avoid hard crash; caller can decide how to handle

            for i, jn in enumerate(self._joint_names):
                sent[f"{jn}.pos"] = float(goal_exposed[i])

        # ---- gripper ----
        if self.config.enable_gripper and ("gripper.pos" in action):
            g = float(action["gripper.pos"])
            if self.config.gripper_normed:
                self._set_gripper_modbus(g)
            else:
                # If not normed, treat input as raw 0..9000 gripper value
                g_raw = max(0.0, min(9000.0, g))
                self._set_gripper_modbus(self._normed_from_gripper_value(int(g_raw)))

            sent["gripper.pos"] = float(action["gripper.pos"])

        return sent

    # ---------------- disconnect ----------------
    def disconnect(self) -> None:
        self._require_connected()
        assert self.arm is not None

        # Prefer per-arm deletion
        try:
            self.arm.rm_delete_robot_arm()
        except Exception:
            # Fallback: some SDK variants require destroy/close
            # doc shows rm_destroy has inconsistency; try both styles.
            try:
                self.arm.rm_destroy()  # instance method form
            except Exception:
                try:
                    RoboticArm.rm_destroy()  # static form
                except Exception:
                    pass

        self._connected = False
        self.arm = None
        self._dof = None

