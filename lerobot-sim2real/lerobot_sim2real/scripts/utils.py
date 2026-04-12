
import numpy as np
import matplotlib.pyplot as plt
def plot_action_sequence(actions, timestep=1.0, joint_labels=None, figsize=(10, 6)):
    """
    根据机械臂 action 序列绘制各维度扭矩随时间变化曲线。

    Parameters
    ----------
    actions : array‑like, shape (T, D)
        动作序列，T 为时间步数，D 为关节（维度）数。
    timestep : float | array‑like, default=1.0
        相邻动作之间的时间间隔（秒）。若传入一维数组，则作为横坐标，长度需等于 T。
    joint_labels : list[str], optional
        每个关节的名称；未指定时按 “Joint 0, 1, 2, ...” 自动生成。
    figsize : tuple, default=(10, 6)
        传递给 matplotlib 的画布大小。
    """
    actions = np.asarray(actions)
    if actions.ndim == 1:                     # 单维 action 也兼容
        actions = actions[:, None]

    T, D = actions.shape

    # 构造时间轴
    if np.isscalar(timestep):
        t = np.arange(T) * float(timestep)
    else:
        t = np.asarray(timestep)
        if t.shape[0] != T:
            raise ValueError("timestep 数组长度必须与动作步数一致")

    # 默认关节标签
    if joint_labels is None:
        joint_labels = [f"Joint {i}" for i in range(D)]
    elif len(joint_labels) != D:
        raise ValueError("joint_labels 数量需等于关节（维度）数")

    # 绘图
    plt.figure(figsize=figsize)
    for d in range(D):
        plt.plot(t, actions[:, d], label=joint_labels[d])
    plt.xlabel("Time (s)")
    plt.ylabel("Torque")
    plt.title("Torque Action Sequence per Joint")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()