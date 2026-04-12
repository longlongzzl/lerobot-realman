import numpy as np

def quat_normalize(q):
    return q / np.linalg.norm(q)

def quat_to_rotmat(q):
    q = quat_normalize(q)
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1-2*(x*x+z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x+y*y)]
    ])
    return R

def local_axes_in_world(q):
    R = quat_to_rotmat(q)
    vx, vy, vz = R[:,0], R[:,1], R[:,2]
    return vx, vy, vz

def angle_between(u, v):
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    cosv = np.clip(np.dot(u, v), -1.0, 1.0)
    return np.degrees(np.arccos(cosv))

def euler_zyx_from_quat(q):
    q = quat_normalize(q)
    w,x,y,z = q
    yaw = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    # pitch (注意数值裁剪)
    s = 2*(w*y - z*x)
    s = np.clip(s, -1.0, 1.0)
    pitch = np.degrees(np.arcsin(s))
    roll = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    return yaw, pitch, roll





# 仿真：第一个关节，
w = -0.501
x = 0.483
y = -0.517
z = -0.499

# 仿真：第二个关节，
w = -0.02
x = -0.005
y = -0.707
z = -0.707

# 仿真：第0个关节，
w = 0
x = 0.022
y = 1.000
z = 0.000

# 真机：第一个关节，
w = -0.511
x = 0.500
y = -0.500
z = -0.488

# 真机：第二个关节，
w = -0.028
x = 0.012
y =-0.707
z = -0.707


# 仿真：第0个关节，
w = 0
x = 0.012
y = 1.000
z = 0.000


# SO100

w = 0.511
x = 0.511
y = 0.489
z = 0.489
q = np.array([w, x, y, z], dtype=float)  # 你的四元数

vx, vy, vz = local_axes_in_world(q)
print("局部 x 轴 (world):", vx)
print("局部 y 轴 (world):", vy)
print("局部 z 轴 (world):", vz)

# 与世界轴夹角
ax_x = angle_between(vx, np.array([1,0,0]))
ay_y = angle_between(vy, np.array([0,1,0]))
az_z = angle_between(vz, np.array([0,0,1]))
print(f"局部x与世界X夹角: {ax_x:.3f}°")
print(f"局部y与世界Y夹角: {ay_y:.3f}°")
print(f"局部z与世界Z夹角: {az_z:.3f}°")

# 欧拉角 (Z-Y-X)
yaw, pitch, roll = euler_zyx_from_quat(q)
print(f"Yaw={yaw:.3f}°, Pitch={pitch:.3f}°, Roll={roll:.3f}°")