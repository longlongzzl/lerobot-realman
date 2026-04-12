"""
RealMan Robot Environment

支持两种模式：
1. 同步模式 (默认) - 兼容原有接口
2. 异步模式 - 用于高频遥操作场景

修改记录：
-----------
2026-01-28: TCP 偏移优化
    - 修改: T_TCP2REALMANEEF 中的 Z 偏移从 0.235m 改为 0.215m
    - 原因: 原 TCP 原点距离夹爪尖端较远，导致抓取位置不准确
    - 方法: 通过手眼标定数据分析，将 TCP 原点沿 X 轴负方向移动 20mm，
            使其更接近两个夹爪尖端之间的连线中心
    - 标定数据: data/20260127_234921/
    - 验证: TCP 一致性标准差 < 3mm，标定结果可靠
"""

from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e, rm_peripheral_read_write_params_t
import numpy as np
from pytransform3d.transformations import transform_from
from pytransform3d.rotations import active_matrix_from_angle
import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any


def T_from_realman_xyzrpy(xyzrpy):
    """将 RealMan 的 xyzrpy (位置+欧拉角) 转换为 4x4 变换矩阵"""
    x, y, z, rx, ry, rz = xyzrpy

    T = np.eye(4)
    Rx = np.array([[1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                [0, 1, 0],
                [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz), np.cos(rz), 0],
                [0, 0, 1]])
    T[:3, :3] = Rz @ Ry @ Rx  # 先绕 x轴旋转 再绕y轴旋转 最后绕z轴旋转
    T[:3, 3] = [x, y, z]
    return T


def realman_xyzrpy_from_T(T):
    """将 4x4 变换矩阵转换为 RealMan 的 xyzrpy"""
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]
    ry = np.arcsin(np.clip(-T[2, 0], -1, 1))
    if np.cos(ry) != 0:
        rx = np.arctan2(T[2, 1]/np.cos(ry), T[2, 2]/np.cos(ry))
        rz = np.arctan2(T[1, 0]/np.cos(ry), T[0, 0]/np.cos(ry))
    else:
        rx = 0
        rz = np.arctan2(-T[0, 1], T[1, 1])
    return np.array([x, y, z, rx, ry, rz])


def realman_gripper_value_from_width(width: float) -> int:
    """将夹爪宽度 (m) 转换为 RealMan 夹爪值"""
    return int(9000 - int(width * 1e5))


def width_from_realman_gripper_value(gripper_value: int) -> float:
    """将 RealMan 夹爪值转换为夹爪宽度 (m)"""
    return (9000 - gripper_value) * 1e-5


# TCP 到 RealMan 末端执行器的变换矩阵
# 优化后: TCP 原点沿 X 轴负方向移动 20mm，更接近夹爪尖端
T_TCP2REALMANEEF = transform_from(
    active_matrix_from_angle(2, -np.pi / 3) @ np.array([
        [0, 0, 1],
        [0, -1, 0],
        [1, 0, 0],
    ]),
    np.array([0, 0, 0.215])  # 原值 0.235，减少 20mm
)


@dataclass
class RobotState:
    """机器人状态快照"""
    Ttcp2base: np.ndarray
    gripper_open: float
    joint: np.ndarray
    timestamp: float


class RealmanEnv:
    """
    RealMan 机器人环境
    
    Args:
        robot_ip: 机械臂 IP 地址
        safety_mode: 是否启用安全模式（限制速度/加速度）
        async_mode: 是否启用异步模式（用于高频遥操作）
    """
    
    def __init__(
        self, 
        robot_ip: str = "192.168.101.20",
        safety_mode: bool = False,
        async_mode: bool = False,
        min_cmd_interval: float = 0.02,
    ):
        self.robot_ip = robot_ip
        self.safety_mode = safety_mode
        self.async_mode = async_mode
        self.min_cmd_interval = min_cmd_interval
        
        # 实例化 RoboticArm 类
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        
        # 创建机械臂连接
        handle = self.arm.rm_create_robot_arm(robot_ip, 8080)
        assert handle.id > 0, f"机械臂连接失败，检查ip是否正确: {robot_ip}"
        
        # 设置 Modbus
        ret = self.arm.rm_set_modbus_mode(1, 115200, 2)
        assert ret == 0, "机械臂modbus设置失败"
        
        # 设置夹爪运动速度
        param = rm_peripheral_read_write_params_t(1, 260, 1)
        ret = self.arm.rm_write_single_register(param, 100)
        assert ret == 0, "写夹爪运动速度失败"
        
        # 安全模式
        if safety_mode:
            print(f"[RealmanEnv] 启用安全模式 ({robot_ip})")
            self.arm.rm_set_arm_max_line_speed(0.1)
            self.arm.rm_set_arm_max_line_acc(0.5)
            self.arm.rm_set_arm_max_angular_speed(0.5)
            self.arm.rm_set_arm_max_angular_acc(1.0)
        
        # 连接状态
        self.connected = True
        
        # 机械臂访问锁 (防止多线程同时访问)
        self._arm_lock = threading.RLock()
        
        # 异步模式相关
        if async_mode:
            self._init_async_mode()
        
        print(f"[RealmanEnv] 连接成功: {robot_ip}")
    
    def _init_async_mode(self):
        """初始化异步模式"""
        self._stop_event = threading.Event()
        
        # 状态缓存
        self._state_lock = threading.Lock()
        self._latest_state: Optional[RobotState] = None
        self._cached_gripper_for_state = 0.09  # 夹爪缓存
        
        # 命令缓存
        self._cmd_lock = threading.Lock()
        self._pending_pose: Optional[np.ndarray] = None
        self._pending_gripper: Optional[float] = None
        self._last_cmd_time: float = 0
        
        # 统计
        self._stats_lock = threading.Lock()
        self._cmd_count = 0
        self._cmd_latency_sum = 0.0
        self._state_count = 0
        self._last_stats_time = time.time()
        self._state_rate = 0.0
        self._avg_latency = 0.0
        
        # 启动后台线程
        self._cmd_thread = threading.Thread(target=self._cmd_loop, daemon=True)
        self._state_thread = threading.Thread(target=self._state_loop, daemon=True)
        self._cmd_thread.start()
        self._state_thread.start()
        
        # 等待首次状态
        for _ in range(50):
            if self._latest_state is not None:
                break
            time.sleep(0.02)
    
    def _state_loop(self):
        """状态读取线程"""
        # 缓存的夹爪值 (从命令中更新)
        self._cached_gripper_for_state = 0.09
        
        while not self._stop_event.is_set():
            try:
                # 直接调用，RealMan SDK TRIPLE_MODE 应该是线程安全的
                ret, state = self.arm.rm_get_current_arm_state()
                
                if ret == 0 and state is not None:
                    robot_state = RobotState(
                        Ttcp2base=T_from_realman_xyzrpy(state["pose"]) @ T_TCP2REALMANEEF,
                        gripper_open=self._cached_gripper_for_state,
                        joint=np.radians(state["joint"]),
                        timestamp=time.time(),
                    )
                    with self._state_lock:
                        self._latest_state = robot_state
                    with self._stats_lock:
                        self._state_count += 1
                else:
                    # 通信失败时增加延迟
                    time.sleep(0.05)
                    continue
                
                # 控制读取频率
                time.sleep(0.02)  # 50 Hz
                
            except:
                time.sleep(0.1)
    
    def _cmd_loop(self):
        """命令发送线程"""
        while not self._stop_event.is_set():
            try:
                with self._cmd_lock:
                    pose = self._pending_pose
                    gripper = self._pending_gripper
                    self._pending_pose = None
                    self._pending_gripper = None
                
                if pose is None and gripper is None:
                    time.sleep(0.005)
                    continue
                
                # 最小发送间隔
                now = time.time()
                elapsed = now - self._last_cmd_time
                if elapsed < self.min_cmd_interval:
                    time.sleep(self.min_cmd_interval - elapsed)
                
                start = time.time()
                
                # 直接调用，不使用锁 (TRIPLE_MODE 线程安全)
                if pose is not None:
                    xyzrpy = realman_xyzrpy_from_T(pose @ np.linalg.inv(T_TCP2REALMANEEF))
                    ret = self.arm.rm_movep_follow(xyzrpy)
                    # 调试：如果移动失败，打印一次
                    if ret != 0 and self._cmd_count % 50 == 0:
                        print(f"[RealmanEnv] movep_follow 失败: ret={ret}")
                
                if gripper is not None:
                    gripper_value = realman_gripper_value_from_width(gripper)
                    param = rm_peripheral_read_write_params_t(1, 258, 1, 2)
                    self.arm.rm_write_registers(param, [0, gripper_value, 0, 0])
                    param = rm_peripheral_read_write_params_t(1, 264, 1)
                    self.arm.rm_write_single_register(param, 1)
                    self._cached_gripper_for_state = gripper
                
                latency = time.time() - start
                self._last_cmd_time = time.time()
                
                with self._stats_lock:
                    self._cmd_count += 1
                    self._cmd_latency_sum += latency
                    
            except Exception as e:
                time.sleep(0.01)
    
    def _update_stats(self):
        """更新统计信息"""
        if not self.async_mode:
            return
        now = time.time()
        with self._stats_lock:
            dt = now - self._last_stats_time
            if dt >= 1.0:
                self._state_rate = self._state_count / dt
                if self._cmd_count > 0:
                    self._avg_latency = (self._cmd_latency_sum / self._cmd_count) * 1000
                self._state_count = 0
                self._cmd_count = 0
                self._cmd_latency_sum = 0
                self._last_stats_time = now
    
    # ========== 原有同步接口 (完全兼容) ==========
    
    def _get_gripper(self, retries: int = 3) -> float:
        """获取夹爪当前开度 (带重试)"""
        for attempt in range(retries):
            try:
                param = rm_peripheral_read_write_params_t(1, 258, 1)
                ret, _ = self.arm.rm_read_holding_registers(param)
                param = rm_peripheral_read_write_params_t(1, 259, 1)
                ret, gripper_value_state = self.arm.rm_read_holding_registers(param)
                if ret == 0:
                    return width_from_realman_gripper_value(gripper_value_state)
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(0.1)
                    continue
        # 返回默认值
        return 0.09
    
    def _set_gripper(self, gripper_open: float, retries: int = 3) -> bool:
        """设置夹爪位置 (带重试)"""
        gripper_value_cmd = realman_gripper_value_from_width(gripper_open)
        
        for attempt in range(retries):
            try:
                # 设置夹爪目标位置
                param = rm_peripheral_read_write_params_t(1, 258, 1, 2)
                ret = self.arm.rm_write_registers(param, [0, gripper_value_cmd, 0, 0])
                if ret != 0:
                    if attempt < retries - 1:
                        time.sleep(0.1)
                        continue
                    print(f"[RealmanEnv] 夹爪设置失败: write_registers ret={ret}")
                    return False
                
                # 执行
                param = rm_peripheral_read_write_params_t(1, 264, 1)
                ret = self.arm.rm_write_single_register(param, 1)
                if ret != 0:
                    if attempt < retries - 1:
                        time.sleep(0.1)
                        continue
                    print(f"[RealmanEnv] 夹爪执行失败: write_single_register ret={ret}")
                    return False
                
                return True
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(0.1)
                    continue
                print(f"[RealmanEnv] 夹爪异常: {e}")
                return False
        
        return False
    
    def compute_observation(self, retries: int = 3) -> dict:
        """获取当前机器人状态 (同步，带重试)"""
        for attempt in range(retries):
            try:
                ret, state = self.arm.rm_get_current_arm_state()
                if ret == 0 and state is not None:
                    return {
                        "Ttcp2base": T_from_realman_xyzrpy(state["pose"]) @ T_TCP2REALMANEEF,
                        "gripper_open": self._get_gripper(),
                        "joint": np.radians(state["joint"]),
                    }
            except Exception as e:
                pass
            if attempt < retries - 1:
                time.sleep(0.05)
        
        # 返回默认值
        print("[RealmanEnv] 警告: 获取状态失败，返回默认值")
        return {
            "Ttcp2base": np.eye(4),
            "gripper_open": 0.09,
            "joint": np.zeros(7),
        }
    
    def reset(self, target_gripper: float = 0.09, max_attempts: int = 100) -> dict:
        """
        重置机械臂到初始位置
        
        Args:
            target_gripper: 目标夹爪开度 (m)，默认完全打开
            max_attempts: 最大尝试次数
        """
        target_joints = np.array([90, 0, 0, -90, 0, -90, 60])
        print(f"[RealmanEnv] 复位中...")
        
        for attempt in range(max_attempts):
            try:
                ret = self.arm.rm_movej_follow(target_joints)
                if ret != 0:
                    print(f"[RealmanEnv] movej_follow 失败: {ret}")
                    time.sleep(0.1)
                    continue
                
                ret, state = self.arm.rm_get_current_arm_state()
                if ret != 0 or state is None:
                    print(f"[RealmanEnv] 获取状态失败")
                    time.sleep(0.1)
                    continue
                
                # 设置夹爪（容错）
                self._set_gripper(target_gripper)
                
                err = np.linalg.norm(state["joint"] - target_joints)
                err_gripper = abs(self._get_gripper() - target_gripper)
                
                if err < 0.1 and err_gripper < 0.01:
                    print(f"[RealmanEnv] 复位完成")
                    break
                
                if attempt % 10 == 0:
                    print(f"waiting for reset... joint_err: {err:.4f}, gripper_err: {err_gripper:.4f}")
                    
            except Exception as e:
                print(f"[RealmanEnv] 复位异常: {e}")
                time.sleep(0.1)
        
        return self.compute_observation()
    
    def step(self, action: dict) -> dict:
        """
        执行动作 (同步)
        
        允许单独传位姿和夹爪的开合程度
        """
        if "Ttcp2base" in action:
            pose_target = realman_xyzrpy_from_T(action["Ttcp2base"] @ np.linalg.inv(T_TCP2REALMANEEF))
            self.arm.rm_movep_follow(pose_target)
        
        if "gripper_open" in action:
            self._set_gripper(action["gripper_open"])
        
        return self.compute_observation()
    
    # ========== 异步模式接口 (用于遥操作) ==========
    
    def get_state(self) -> Optional[Dict[str, Any]]:
        """获取状态 (非阻塞，返回缓存)"""
        if not self.async_mode:
            return self.compute_observation()
        
        self._update_stats()
        with self._state_lock:
            if self._latest_state is None:
                return None
            return {
                "Ttcp2base": self._latest_state.Ttcp2base.copy(),
                "gripper_open": self._latest_state.gripper_open,
                "joint": self._latest_state.joint.copy(),
                "timestamp": self._latest_state.timestamp,
            }
    
    def get_pose(self) -> Optional[np.ndarray]:
        """获取位姿 (非阻塞)"""
        if not self.async_mode:
            obs = self.compute_observation()
            return obs["Ttcp2base"]
        
        with self._state_lock:
            if self._latest_state is None:
                return None
            return self._latest_state.Ttcp2base.copy()
    
    def send_pose(self, Ttcp2base: np.ndarray):
        """发送位姿命令 (非阻塞)"""
        if not self.async_mode:
            self.step({"Ttcp2base": Ttcp2base})
            return
        
        with self._cmd_lock:
            self._pending_pose = Ttcp2base.copy()
    
    def send_gripper(self, gripper_open: float):
        """发送夹爪命令 (非阻塞)"""
        if not self.async_mode:
            self._set_gripper(gripper_open)
            return
        
        with self._cmd_lock:
            self._pending_gripper = gripper_open
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """获取通信统计"""
        if not self.async_mode:
            return {"connected": self.connected, "async_mode": False}
        
        with self._stats_lock:
            return {
                "connected": self.connected,
                "async_mode": True,
                "state_update_rate": self._state_rate,
                "avg_latency_ms": self._avg_latency,
            }
    
    def close(self):
        """关闭连接"""
        if self.async_mode:
            self._stop_event.set()
            if hasattr(self, '_cmd_thread'):
                self._cmd_thread.join(timeout=1.0)
            if hasattr(self, '_state_thread'):
                self._state_thread.join(timeout=1.0)
        
        self.arm.rm_delete_robot_arm()
        print(f"[RealmanEnv] 已关闭: {self.robot_ip}")


# ========== 测试代码 ==========

if __name__ == "__main__":
    # Test
    assert np.allclose(
        realman_xyzrpy_from_T(T_from_realman_xyzrpy([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])),
        [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
    )
    assert np.allclose(width_from_realman_gripper_value(realman_gripper_value_from_width(0.09)), 0.09)
    
    env = RealmanEnv()
    try:
        obs = env.reset()
        for _ in range(1000):
            print(obs["Ttcp2base"][:3, 3])
            print(obs["Ttcp2base"][:3, :3])
            obs["gripper_open"] = 0
            obs = env.step(obs)
    finally:
        env.close()
