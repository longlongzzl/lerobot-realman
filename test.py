from lerobot.common.robots.realman_lerobot import RealManArm, RealManArmConfig
r = RealManArm(RealManArmConfig(ip="192.168.101.19", port=8080, use_degrees=True))
r.connect()
print("obs:", r.get_observation())               # 能否正常读角度
print("move rc:", r.send_action({"joint1.pos":0.0}))  # 原位小幅度指令
r.disconnect()
