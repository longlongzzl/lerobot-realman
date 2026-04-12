from lerobot.common.robots.realman_lerobot import RealManArm, RealManArmConfig
from lerobot.common.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.common.cameras import ColorMode  # 如需改色彩模式

def create_real_robot() -> tuple[RealManArm, RealSenseCamera]:
    # 机械臂配置：根据你的设备 IP/端口、速度参数调整
    robot_cfg = RealManArmConfig(
        id="rm1",
        ip="192.168.101.20",
        port=8080,
        use_degrees=True,   # 外部接口是否用角度；False 则用弧度
        v=60, r=0, connect=0, block=0,
        max_relative_target=5.0,
        enable_gripper=True,
        gripper_normed=True,
    )
    robot = RealManArm(robot_cfg)

    # 相机配置：替换成你的 RealSense 序列号或名称
    cam_cfg = RealSenseCameraConfig(
        serial_number_or_name="342522073637",
        fps=30,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        use_depth=False,   # 如需深度改 True
    )
    camera = RealSenseCamera(cam_cfg)

    return robot, camera

if __name__ == "__main__":
    robot, camera = create_real_robot()
    robot.connect()
    camera.connect()
    print("obs:", robot.get_observation())
    frame = camera.read()
    print("frame shape:", frame.shape)
    camera.disconnect()
    robot.disconnect()
