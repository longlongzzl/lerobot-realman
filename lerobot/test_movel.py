from Robotic_Arm.rm_robot_interface import *

# 实例化RoboticArm类
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
# 创建机械臂连接，打印连接id
handle = arm.rm_create_robot_arm("192.168.101.20", 8080)
print(handle.id)

# print(arm.rm_movej_p([-0.12, -0.299, 0.297, 3.141, 0, 0], 20, 0, 0, 1))
print(arm.rm_movej_p([-0.14297763724152412, -0.22993167327883207, 0.2951923966632054,3.141,0 ,0], 20, 0, 0, 1))

arm.rm_delete_robot_arm()