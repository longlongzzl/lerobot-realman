from ..realman_lerobot import RealManArm, RealManArmConfig

def main():
    """
    演示如何使用RealMan机械臂API。
    支持带夹爪或不带夹爪的机械臂，会自动检测。
    """
    cfg = RealManArmConfig(
        id="rm1",
        ip="192.168.1.18",
        port=8080,
        use_degrees=True,          # False -> 对外用弧度
        v=20, r=0, connect=0,
        block=0,                   # 高频控制建议 0（非阻塞）
        max_relative_target=5.0,   # 每次最多动 5 度（use_degrees=True 时）
        enable_gripper=True,       # 如果机械臂没有夹爪，会在连接后自动检测并处理
        gripper_normed=True,       # 夹爪位置归一化到 [0,1]
    )

    robot = RealManArm(cfg)
    robot.connect()

    obs = robot.get_observation()
    print("obs keys:", list(obs.keys()))
    print("first obs:", obs)

    # 检查是否有夹爪（如果返回NaN说明没有夹爪或检测失败）
    import math
    has_gripper = (
        "gripper.pos" in obs and 
        isinstance(obs["gripper.pos"], (int, float)) and 
        not math.isnan(obs["gripper.pos"])
    )
    
    if has_gripper:
        print("\n检测到夹爪，可以控制夹爪")
        # 让所有关节保持不变，只动夹爪（演示）
        action = {"gripper.pos": 0.2}  # 0.0=完全闭合, 1.0=完全打开
        sent = robot.send_action(action)
        print("sent:", sent)
    else:
        print("\n未检测到夹爪，跳过夹爪控制")
        # 如果只有关节控制，可以这样：
        # action = {"joint1.pos": 10.0}  # 示例：移动第一个关节
        # sent = robot.send_action(action)
        # print("sent:", sent)

    robot.disconnect()
    print("done")

if __name__ == "__main__":
    main()
