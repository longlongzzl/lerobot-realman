from Robotic_Arm.rm_robot_interface import *
import struct

# 实例化RoboticArm类
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

# 创建机械臂连接，打印连接id
handle = arm.rm_create_robot_arm("192.168.101.20", 8080)
print(handle.id)

# 设置灵巧手各手指角度
# print(arm.rm_set_hand_angle([0,100,200,300,400,1000],True,1))
# print(arm.rm_set_hand_angle([1000,1000,1000,1000,1000,1000],True,1))



# 配置控制器RS485端口为RTU主站
print(arm.rm_set_modbus_mode(0,115200,2))

values = [50, 600, 700, 800, 0, 1000]
data = list(struct.pack('>6h', *values))

write_params = rm_peripheral_read_write_params_t(1, 1486, 1, 6)
print(arm.rm_write_registers(write_params, data))

# 读取
read_params = rm_peripheral_read_write_params_t(1, 1486, 1, 6)
code, raw_data = arm.rm_read_multiple_holding_registers(read_params)

print("code:", code)
print("raw_data:", raw_data)

if code == 0 and raw_data is not None and len(raw_data) >= 12:
    values = struct.unpack('>6h', bytes(raw_data[:12]))
    print("ANGLE_SET:", values)
else:
    print("读取失败或返回字节数不足")

arm.rm_delete_robot_arm()
