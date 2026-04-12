"""
Code based on https://github.com/huggingface/lerobot for supporting real robot control via the unified LeRobot interface.
"""

import time
from typing import List, Optional

import numpy as np
import torch

from mani_skill.agents.base_real_agent import BaseRealAgent
from mani_skill.utils import common
from mani_skill.utils.structs.types import Array

try:
    from lerobot.common.cameras.camera import Camera
    from lerobot.common.motors.motors_bus import MotorNormMode
    from lerobot.common.robots.robot import Robot
    from lerobot.common.utils.robot_utils import busy_wait
except ImportError:
    pass


class LeRobotRealAgent(BaseRealAgent):
    """
    LeRobotRealAgent is a general class for controlling real robots via the LeRobot system. You simply just pass in the Robot instance you create via LeRobot and pass it here to make it work with ManiSkill Sim2Real environment interfaces.

    Args:
        robot (Robot): The Robot instance you create via LeRobot.
        use_cached_qpos (bool): Whether to cache the fetched qpos values. If True, the qpos will be
            read from the cache instead of the real robot when possible. This cache is only invalidated when
            set_target_qpos or set_target_qvel is called. This can be useful if you want to easily have higher frequency (> 30Hz) control since qpos reading from the robot is
            currently the slowest part of LeRobot for some of the supported motors.
    """

    def __init__(self, robot: Robot, use_cached_qpos: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._captured_sensor_data = None
        self.real_robot = robot
        self.use_cached_qpos = use_cached_qpos
        self._cached_qpos = None
        self._motor_keys: List[str] = None

        if self.real_robot.name == "so100_follower" or self.real_robot.name == "so101_follower" :
            print("===========================================================")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ", self.real_robot.name)
            print("===========================================================")
            #assert  1==2
            self.real_robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES

    def start(self):
        self.real_robot.connect()

    def stop(self):
        self.real_robot.disconnect()

    def set_target_qpos(self, qpos: Array):


        print("current qpos", qpos)
        #assert  1==2
        self._cached_qpos = None
        qpos = common.to_cpu_tensor(qpos).flatten()
        #find_delta = np.array([ 0. ,  -0.06, -0.12, -0.01,  0. ,   3.0291e-03 ])

        #print("before", qpos)
        # for i in range(len(qpos)):
        #     print(i, "=", qpos[i], find_delta[0][i], qpos[i] - find_delta[0][i])
        qpos = qpos #-find_delta
        #print("after", qpos)



        qpos = torch.rad2deg(qpos)
        #print(qpos)
        #print(len(qpos))

        qpos = {f"{self._motor_keys[i]}.pos": qpos[i] for i in range(len(qpos))}


        # print(self._motor_keys)
        # assert  1==2
        # NOTE (stao): It seems the calibration from LeRobot has some offsets in some joints. We fix reading them here to match the expected behavior
        if self.real_robot.name == "so100_follower" :
            qpos["elbow_flex.pos"] = qpos["elbow_flex.pos"] + 6.8
        # # if self.real_robot.name == "so101_follower" :
        #     qpos["wrist_flex.pos"] = qpos["wrist_flex.pos"] + 3.5
        # qpos -= np.array([[ 0. ,  -0.06, -0.12, -0.01,  0. ,   3.0291e-03 ]])
        self.real_robot.send_action(qpos)

    def reset(self, qpos: Array):
        qpos = common.to_cpu_tensor(qpos)
        freq = 30
        target_pos = self.qpos
        max_rad_per_step = 0.025
        for _ in range(int(20 * freq)):
            start_loop_t = time.perf_counter()
            delta_step = (qpos - target_pos).clip(
                min=-max_rad_per_step, max=max_rad_per_step
            )
            if np.linalg.norm(delta_step) <= 1e-4:
                break
            target_pos += delta_step

            self.set_target_qpos(target_pos)
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / freq - dt_s)

    def capture_sensor_data(self, sensor_names: Optional[List[str]] = None):
        sensor_obs = dict()
        cameras: dict[str, Camera] = self.real_robot.cameras
        if sensor_names is None:
            sensor_names = list(cameras.keys())
        for name in sensor_names:
            data = cameras[name].async_read()
            # until https://github.com/huggingface/lerobot/issues/860 is resolved we temporarily assume this is RGB data only otherwise need to write a few extra if statements to check
            # if isinstance(cameras[name], IntelRealSenseCamera):
            sensor_obs[name] = dict(rgb=(common.to_tensor(data)).unsqueeze(0))
        self._captured_sensor_data = sensor_obs

    def get_sensor_data(self, sensor_names: Optional[List[str]] = None):
        if self._captured_sensor_data is None:
            raise RuntimeError(
                "No sensor data captured yet. Please call capture_sensor_data() first."
            )
        if sensor_names is None:
            return self._captured_sensor_data
        else:
            return {
                k: v for k, v in self._captured_sensor_data.items() if k in sensor_names
            }

    def get_qpos(self):
        # NOTE (stao): the slowest part of inference is reading the qpos from the robot. Each time it takes about 5-6 milliseconds, meaning control frequency is capped at 200Hz.
        # and if you factor in other operations like policy inference etc. the max control frequency is typically more like 30-60 Hz.
        # Moreover on the rare occassions reading qpos can take 40 milliseconds which causes the control step to fall behind the desired control frequency.
        if self.use_cached_qpos and self._cached_qpos is not None:
            #print("????")
            return self._cached_qpos.clone()
        qpos_deg = self.real_robot.bus.sync_read("Present_Position")

        # NOTE (stao): It seems the calibration from LeRobot has some offsets in some joints. We fix reading them here to match the expected behavior
        if self.real_robot.name == "so100_follower":
            qpos_deg["elbow_flex"] = qpos_deg["elbow_flex"] - 6.8
        # if self.real_robot.name == "so101_follower" :
        #     qpos_deg["wrist_flex"] = qpos_deg["wrist_flex"] + 3.5
        if self._motor_keys is None:
            self._motor_keys = list(qpos_deg.keys())
        print("self._motor_keys", self._motor_keys)
        qpos_deg = common.flatten_state_dict(qpos_deg)
        qpos = torch.deg2rad(torch.tensor(qpos_deg)).unsqueeze(0)
        #qpos += np.array([[ 0. ,  -0.06, -0.12, -0.01,  0. ,   3.0291e-03 ]])
        self._cached_qpos = qpos
        return qpos


    def get_rad(self):
        # NOTE (stao): the slowest part of inference is reading the qpos from the robot. Each time it takes about 5-6 milliseconds, meaning control frequency is capped at 200Hz.
        # and if you factor in other operations like policy inference etc. the max control frequency is typically more like 30-60 Hz.
        # Moreover on the rare occassions reading qpos can take 40 milliseconds which causes the control step to fall behind the desired control frequency.
        # if self.use_cached_qpos and self._cached_qpos is not None:
        #     print("????")
        #     return self._cached_qpos.clone()
        qpos_deg = self.real_robot.bus.sync_read("Present_Position")

        # NOTE (stao): It seems the calibration from LeRobot has some offsets in some joints. We fix reading them here to match the expected behavior
        if self.real_robot.name == "so100_follower":
            qpos_deg["elbow_flex"] = qpos_deg["elbow_flex"] - 6.8
        # if self.real_robot.name == "so101_follower":
        #     qpos_deg["wrist_flex"] = qpos_deg["wrist_flex"] - 3.5


        # if self.real_robot.name == "so101_follower":
        #     qpos_deg["elbow_flex"] = qpos_deg["elbow_flex"] - 2.7272791048227187
        # #
        #     qpos_deg["wrist_flex"] = qpos_deg["wrist_flex"] - 0.8791464408487352
        #     qpos_deg["shoulder_pan"] = qpos_deg["shoulder_pan"] + 0.5714279978178238
        # delta [ 0.   -0.05 -0.04 -0.03  0.    0.  ]
        if self._motor_keys is None:
            self._motor_keys = list(qpos_deg.keys())
        #print("self._motor_keys", self._motor_keys)
        qpos_deg = common.flatten_state_dict(qpos_deg)
        rad = torch.deg2rad(torch.tensor(qpos_deg)).unsqueeze(0)
      #  print(rad)
       # rad += np.array([[ 0.00383589, -0.0069046 , -0.05048029 ,-0.01687791, -0.00613742  ,0.00460307]])
      #  rad += np.array([[0.00143001, 0.00187127, 0.00935312, 0.01498931, 0.01138775, 0.00275124]])
        #rad += np.array([[ 0. ,  -0.06, -0.12, -0.01,  0. ,   3.0291e-03 ]])

        return rad

    def get_qvel(self):
        """
        获取当前关节速度（弧度/秒）。
        优先尝试从硬件读取 Present_Velocity，如果不支持则返回零速度。

        注意：
        - 仿真中 SAPIEN 的 get_qvel() 返回弧度/秒 (rad/s)
        - LeRobot 的 sync_read("Present_Velocity") 返回的值：
          * Present_Velocity 不在 NORMALIZED_DATA 列表中，返回原始编码值
          * 对于 Dynamixel: 单位通常是 0.229 rpm (需要转换为 rad/s)
          * 对于 Feetech: 可能是度/秒或其他单位
        - 当前实现假设 sync_read 返回的值已经是度/秒，然后用 deg2rad 转换
        - 如果实际单位不同，需要根据硬件文档调整转换逻辑

        单位转换链：
        1. 硬件原始值 -> (可能需要转换) -> 度/秒
        2. 度/秒 -> deg2rad -> 弧度/秒 (与仿真对齐)
        """
        try:
            # 尝试从硬件读取速度（原始值，未归一化）
            qvel_raw = self.real_robot.bus.sync_read("Present_Velocity")


            if self._motor_keys is None:
                self._motor_keys = list(qvel_raw.keys())

            qvel_raw = common.flatten_state_dict(qvel_raw)
            qvel_raw = torch.tensor(qvel_raw)*2*np.pi/4096


            return qvel_raw.unsqueeze(0)

        except (KeyError, AttributeError, NotImplementedError):
            # 如果硬件不支持 Present_Velocity，返回零速度
            # 这样可以避免 NotImplementedError，但速度信息为 0
            raise "error"
    #
    # def get_qvel(self):
    #     raise NotImplementedError
