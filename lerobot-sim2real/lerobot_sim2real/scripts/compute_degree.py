import numpy as np, math

q = np.array([0.552123, 0.468534, -0.446436, 0.525672])
q = q / np.linalg.norm(q)          # 归一化（必须！）

# 把下面两行换成自己的顺序
x, y, z, w = q                     # xyzw 顺序示例
# w, x, y, z = q                  # 如果是 wxyz 就把这一行取消注释

sinr_cosp = 2*(w*x + y*z)
cosr_cosp = 1 - 2*(x*x + y*y)
roll  = math.atan2(sinr_cosp, cosr_cosp)

sinp = 2*(w*y - z*x)
pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

siny_cosp = 2*(w*z + x*y)
cosy_cosp = 1 - 2*(y*y + z*z)
yaw   = math.atan2(siny_cosp, cosy_cosp)

print(np.degrees([roll, pitch, yaw]))

print(np.deg2rad([90, 90, 90]))
print(np.deg2rad(np.degrees([roll, pitch, yaw])))

print("delta", np.deg2rad(np.degrees([roll, pitch, yaw])) - np.deg2rad([90, 90, 90]))