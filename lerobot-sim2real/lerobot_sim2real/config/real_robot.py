import subprocess

from lerobot.common.robots.realman_lerobot import RealManArm, RealManArmConfig
# from lerobot.common.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.common.cameras.realsense import RealSenseCamera, RealSenseCameraConfig


def _prefetch_camera_profile(device: str, width: int | None, height: int | None, fps: int | None, fourcc: str | None) -> None:
    """Use v4l2-ctl to force the camera into a desired mode if available."""
    if width is None or height is None or fps is None:
        return
    args = ["v4l2-ctl", f"--device={device}", f"--set-fmt-video=width={width},height={height}"]
    if fourcc:
        args[-1] += f",pixelformat={fourcc}"
    args.append(f"--set-parm={fps}")
    try:
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass  # 如果系统没有 v4l2-ctl 或设置失败，继续走 OpenCV 的设置流程

def create_real_robot(auto_connect: bool = True) -> tuple[RealManArm, RealSenseCamera]:
    # 机械臂配置：根据你的设备 IP/端口、速度参数调整
    robot_cfg = RealManArmConfig(
        id="rm1",
        ip="192.168.101.20",
        port=8080,
        use_degrees=True,   # 外部接口是否用角度；False 则用弧度
        v=20, r=0, connect=0, block=1,  # block 改回阻塞模式，避免 rm_movej 返回状态 1
        max_relative_target=6,
        enable_gripper=True,
        gripper_normed=True,
    )

    robot = RealManArm(robot_cfg)
    # 相机配置：使用 RealSense 彩色流
    cam_cfg = RealSenseCameraConfig(
        serial_number_or_name="342522073637",
        width=None,
        height=None,
        fps=None,
    )
    camera = RealSenseCamera(cam_cfg)

    # Attach cameras to robot for LeRobotRealAgent sensor access.
    robot.cameras = {"base_camera": camera}

    if auto_connect:
        robot.connect()
        camera.connect()

    return robot, camera

if __name__ == "__main__":
    robot, camera = create_real_robot()
    print("RealMan config:", robot.config)
    print("RealSense config:", camera.config)
    robot.connect()
    try:
        obs = robot.get_observation()
        print("RealMan observation keys:", list(obs.keys()))
    finally:
        robot.disconnect()
