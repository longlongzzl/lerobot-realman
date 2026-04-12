import numpy as np


def normalize(q):
    return q / np.linalg.norm(q)


def quat_mul(q2, q1):
    w2, x2, y2, z2 = q2
    w1, x1, y1, z1 = q1
    return np.array([
        w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
        w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
        w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
        w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1
    ])


def quat_inv(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z]) / np.dot(q, q)


def joint0_offset(q_meas, q_ref, joint_axis=None):
    """q_meas, q_ref: 四元数 (w,x,y,z); joint_axis: 3D 单位向量，如果需要有符号角"""
    q_meas = normalize(q_meas)
    q_ref = normalize(q_ref)
    q_rel = quat_mul(q_meas, quat_inv(q_ref))
    # 统一符号，保证 w >= 0（可选）
    if q_rel[0] < 0:
        q_rel = -q_rel
    w, x, y, z = q_rel
    w = np.clip(w, -1.0, 1.0)
    angle = 2 * np.arccos(w)  # 最小旋转角 (rad)
    angle_deg = np.degrees(angle)

    signed_angle = None
    if joint_axis is not None:
        v = np.array([x, y, z])
        sin_half = np.linalg.norm(v)
        if sin_half < 1e-12:
            signed_angle = 0.0
        else:
            axis = v / sin_half
            sgn = np.sign(np.dot(axis, joint_axis / np.linalg.norm(joint_axis)))
            signed_angle = sgn * angle
    return angle, angle_deg, signed_angle

# 仿真：第0个关节，
w0r = 0
x0r = 0.012
y0r = 1.000
z0r = 0.000


# 仿真：第0个关节，
w0m = 0
x0m = 0.022
y0m = 1.000
z0m = 0.000
# ====== 把下面换成你的实际数值 ======
q0_meas = np.array([w0m, x0m, y0m, z0m], dtype=float)  # joint0 当前读数对应四元数
q0_ref = np.array([w0r, x0r, y0r, z0r], dtype=float)  # joint0 参考零位四元数
joint_axis = np.array([0, 0, 1], dtype=float)  # 例如 [0,0,1]，若已知

angle_rad, angle_deg, signed_angle = joint0_offset(q0_meas, q0_ref, joint_axis)

print(f"Joint0 几何最小偏差角: {angle_deg:.6f}°")
if signed_angle is not None:
    print(f"Joint0 对关节轴的有符号偏差: {np.degrees(signed_angle):.6f}°")
