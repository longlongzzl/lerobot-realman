from __future__ import annotations

import math,torch
import time
from typing import Any, Dict, List, Optional

from .config_realman import RealManArmConfig
from .errors import DeviceAlreadyConnectedError, DeviceNotConnectedError, RobotAPIError
from .robot_base import Robot
from .utils import ensure_safe_goal_position

# RealMan SDK
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_peripheral_read_write_params_t  # type: ignore


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

    def _gripper_value_from_normed(self, g: float) -> int:
        g = max(0.0, min(1.0, g))
        # 0=open, 1=fully closed
        return int(round(g * 1000.0))

    # def _gripper_value_from_normed(self, g: float) -> int:
    #     g = float(g)
    #
    #     # 保留原逻辑：负值 -> 0，上限截断到 0.91
    #     g = max(0.0, min(0.91, g))
    #
    #     # 线性映射：0..0.91 -> 0..9000
    #     return int(round(g / 0.91 * 9000.0))

    def _normed_from_gripper_value(self, value: int) -> float:
        value = max(0, min(1000, int(value)))
        return max(0.0, min(1.0, value / 1000.0))

    def _get_gripper_modbus(self) -> float:
        assert self.arm is not None
        # Read holding register 258 (status/trigger) then 259 (position)
        param = rm_peripheral_read_write_params_t(1, 258, 1)
        code, _ = self.arm.rm_read_holding_registers(param)
        self._check_ok(int(code), "rm_read_holding_registers (gripper status)")

        param = rm_peripheral_read_write_params_t(1, 259, 1)
        code, value = self.arm.rm_read_holding_registers(param)
        self._check_ok(int(code), "rm_read_holding_registers (gripper position)")
        # print("_get_gripper_modbus get position:", value)
        return self._normed_from_gripper_value(int(value))

    def _set_gripper_modbus(self, g: float) -> None:
        assert self.arm is not None
        print("_set_gripper_modbus get g:", g)
        gripper_value_cmd = self._gripper_value_from_normed(g)
        # # gripper_value_cmd = g
        # print("_set_gripper_modbus get g after gripper_value_cmd:", gripper_value_cmd)
        #
        # # Set target position
        # param = rm_peripheral_read_write_params_t(1, 258, 1, 2)
        # # code = self.arm.rm_set_gripper_position(gripper_value_cmd, True, 10)
        # code = self.arm.rm_write_registers(param, [0, gripper_value_cmd, 0, 0])
        # self._check_ok(int(code), "rm_write_registers (gripper target)")
        gripper_value_cmd = max(0, min(9000, gripper_value_cmd))  # 安全限幅
        # print("_set_gripper_modbus get g after gripper_value_cmd:", gripper_value_cmd)

        # 转 16 进制（用于打印/对照）
        hex_str = f"{gripper_value_cmd:08X}"  # 固定 8 位，表示 4 字节
        # print(f"gripper_value_cmd hex = 0x{hex_str}")

        # 拆成 4 个“十进制寄存器值”（0~255），大端：MSB -> LSB
        regs = [
            (gripper_value_cmd >> 24) & 0xFF,
            (gripper_value_cmd >> 16) & 0xFF,
            (gripper_value_cmd >> 8) & 0xFF,
            gripper_value_cmd & 0xFF,
        ]
        # print("rm_write_registers regs =", regs)

        # Set target position
        param = rm_peripheral_read_write_params_t(1, 258, 1, 2)
        code = self.arm.rm_write_registers(param, regs)
        self._check_ok(int(code), "rm_write_registers (gripper target)")

        # Execute
        param = rm_peripheral_read_write_params_t(1, 264, 1)
        code = self.arm.rm_write_single_register(param, 1)
        self._check_ok(int(code), "rm_write_single_register (gripper execute)")
        # print("gripper positon after set :", self._get_gripper_modbus())

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
            self._joint_names = [f"joint{i + 1}" for i in range(self._dof)]

        # If user provided joint_names but length mismatch, fail early
        if len(self._joint_names) != self._dof:
            raise ValueError(
                f"joint_names length ({len(self._joint_names)}) != robot dof ({self._dof}). "
                f"Either provide correct names or leave joint_names=None to auto-generate."
            )

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
        if self.config.enable_gripper:
            assert self.arm is not None
            # Modbus init for gripper control
            code = self.arm.rm_set_modbus_mode(1, 115200, 2)
            self._check_ok(int(code), "rm_set_modbus_mode")

            # Set gripper speed (register 260)
            param = rm_peripheral_read_write_params_t(1, 260, 1)
            code = self.arm.rm_write_single_register(param, 100)
            self._check_ok(int(code), "rm_write_single_register (gripper speed)")
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
                self._joint_names = [f"joint{i + 1}" for i in range(self._dof)]

        xs = self._list_deg_to_exposed(deg_list)

        obs: Dict[str, Any] = {f"{jn}.pos": float(xs[i]) for i, jn in enumerate(self._joint_names)}

        if self.config.enable_gripper:
            try:
                # if self.config.gripper_normed:
                obs["gripper.pos"] = (1.0-float(self._get_gripper_modbus()))*0.91
                # else:
                #     # If not normed, still return 0..9000 raw value for debugging
                #     param = rm_peripheral_read_write_params_t(1, 259, 1)
                #     code, value = self.arm.rm_read_holding_registers(param)
                #     self._check_ok(int(code), "rm_read_holding_registers (gripper position raw)")
                #     obs["gripper.pos"] = float(int(value))
            except Exception:
                obs["gripper.pos"] = float("nan")
        # print("return obs=", obs)#0-0.91
        return obs

    # ---------------- action ----------------
    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        t1=time.time()
        print("realman arm get action:", action)
        self._require_connected()
        assert self.arm is not None
        assert self._dof is not None

        sent: Dict[str, Any] = {}

        # ---- joints ----
        # Determine if any joint targets are provided
        joint_keys = [f"{jn}.pos" for jn in self._joint_names]
        provided_joint = any(k in action for k in joint_keys)
        print("time1=",time.time()-t1)
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
            print("time2=", time.time() - t1)

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
            print("time3=", time.time() - t1)

            for i, jn in enumerate(self._joint_names):
                sent[f"{jn}.pos"] = float(goal_exposed[i])

        # ---- gripper ----
        if self.config.enable_gripper and ("gripper.pos" in action):
            g = (0.91- float(action["gripper.pos"]))/0.91 #为0-1处理的函数 但是这里要放大一波 实际上传入的是0-0.91

            # delta_norm = g / 0.91
            # g_now=self._get_gripper_modbus()
            # print("g_now g=",g_now,g)
            # g = max(0.0, min(1.0, g_now + delta_norm))  # 0~1

            # target_rad = float(action["gripper.pos"])
            # target_norm = max(0.0, min(1.0, target_rad / 0.91))  # 0~1
            # g=target_norm
            print("time4=", time.time() - t1)

            if self.config.gripper_normed:
                # print("moving gripper to g", g)
                self._set_gripper_modbus(g)
            else:
                # If not normed, treat input as raw 0..9000 gripper value
                g_raw = max(0.0, min(1000.0, g))
                print("moving gripper to g_raw", g_raw)
                self._set_gripper_modbus(self._normed_from_gripper_value(int(g_raw)))
            print("time5=", time.time() - t1)

            sent["gripper.pos"] = float(action["gripper.pos"])
        print("send_action take second:",time.time() - t1)
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
