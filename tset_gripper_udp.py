# 下面是一个如何注册UDP机械臂实时状态主动上报信息回调函数的示例：
# 在这个示例中，我们定义了一个名为`arm_state_func`的函数，用于处理机械臂实时上报的数据，并将其注册为回调函数。
# `arm_state_func`函数会按照UDP接口设置的周期被调用，该函数接收一个rm_realtime_arm_joint_state_t的对象作为参数
from Robotic_Arm.rm_robot_interface import *
import time

_printed_fields = False
_last_ts = None
_dt_sum = 0.0
_dt_count = 0
_joint_pos_last = None
_gripper_pos_last = None
_hand_pos_last = None
def arm_state_func(data):
    global _last_ts, _dt_sum, _dt_count, _joint_pos_last, _gripper_pos_last, _hand_pos_last
    # One-time introspection to find joint fields in this SDK build (disabled)
    global _printed_fields
    if not _printed_fields:
        _printed_fields = True
    # Joint status (rm_joint_status_t): capture joint positions
    if hasattr(data, "joint_status"):
        try:
            js = data.joint_status
            if hasattr(js, "to_dict"):
                jd = js.to_dict()
                _joint_pos_last = jd.get("joint_position")
            else:
                n = None
                for attr in ("joint_num", "dof", "joint_count", "arm_dof"):
                    if hasattr(data, attr):
                        try:
                            n = int(getattr(data, attr))
                            break
                        except Exception:
                            pass
                if n is None:
                    try:
                        n = len(js)
                    except Exception:
                        n = 7

                joint_pos = []
                for i in range(n):
                    try:
                        one = js[i]
                    except Exception:
                        break
                    if hasattr(one, "pos"):
                        joint_pos.append(one.pos)

                _joint_pos_last = joint_pos
        except Exception:
            pass
    elif hasattr(data, "joint_pos"):
        try:
            _joint_pos_last = data.joint_pos
        except Exception:
            pass
    elif hasattr(data, "joint"):
        try:
            _joint_pos_last = data.joint
        except Exception:
            pass
    # 灵巧手信息 handState 字段
    if hasattr(data, "handState"):
        try:
            hand_state = data.handState
            if hasattr(hand_state, "to_dict"):
                _hand_pos_last = hand_state.to_dict()
            else:
                _hand_pos_last = {
                    "hand_pos": list(getattr(hand_state, "hand_pos", [])),
                    "hand_angle": list(getattr(hand_state, "hand_angle", [])),
                    "hand_force": list(getattr(hand_state, "hand_force", [])),
                    "hand_state": list(getattr(hand_state, "hand_state", [])),
                    "hand_err": getattr(hand_state, "hand_err", None),
                }
        except Exception:
            pass
    # End-effector (gripper) info via plus_state
    if hasattr(data, "plus_state_info"):
        try:
            ps = data.plus_state_info
            _gripper_pos_last = ps.pos[0]
        except Exception:
            pass

    # Timestamp + interval (ms): print average every 100 samples
    now = time.perf_counter()
    if _last_ts is not None:
        dt_ms = (now - _last_ts) * 1000.0
        _dt_sum += dt_ms
        _dt_count += 1
        if _dt_count >= 100:
            avg_dt = _dt_sum / _dt_count
            print(f"avg_dt_ms(100): {avg_dt:.2f}")
            if _joint_pos_last is not None:
                print("joint_pos:", _joint_pos_last)
            if _gripper_pos_last is not None:
                print("pos[0]:", _gripper_pos_last)
            if _hand_pos_last is not None:
                print("hand_state:", _hand_pos_last)
            _dt_sum = 0.0
            _dt_count = 0
    _last_ts = now
# 初始化为三线程模式
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

# 创建机械臂连接，打印连接id
handle = arm.rm_create_robot_arm("192.168.101.20", 8080)
print(handle.id)

# 设置UDP端口，广播周期500ms，使能，广播端口号8089，力数据坐标系使用传感器坐标系，上报目标IP为"192.168.1.104"
# 自定义上报项均设置关闭，用户可根据实际情况修改这些配置
custom = rm_udp_custom_config_t()
custom.joint_speed = 1
custom.lift_state = 0
custom.expand_state = 0
custom.hand_state = 1
custom.arm_current_status = 1
custom.plus_base = 1
custom.plus_state = 1
custom.hand_pos=1
config = rm_realtime_push_config_t(1, True, 8089, 0, "192.168.101.27", custom)
print(arm.rm_set_realtime_push(config))
print(arm.rm_get_realtime_push())
arm.rm_set_rm_plus_mode(115200)
arm_state_callback = rm_realtime_arm_state_callback_ptr(arm_state_func)
arm.rm_realtime_arm_state_call_back(arm_state_callback)

# 关节运动
# ret = arm.rm_movej([0, 30, 60, 0, 90, 0], 30, 0, 0, 1)
# print("movej: ", ret)

# 删除指定机械臂对象
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    arm.rm_delete_robot_arm()
