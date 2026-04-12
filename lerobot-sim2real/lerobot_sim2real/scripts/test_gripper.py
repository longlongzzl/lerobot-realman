import socket
import time


# def send_cmd(client, cmd_6axis):
#     client.send(cmd_6axis.encode('utf-8'))
#     # Optional: Receive a response from the server
#     _ = client.recv(1024).decode()
#     print(_)
#     return True

def send_cmd_nr(client, cmd_6axis):
    client.send(cmd_6axis.encode('utf-8'))
    # Optional: Receive a response from the server
    # _ = client.recv(1024).decode()
    # print(_)
    return True


# IP and port configuration
ip = '192.168.101.20'

port_no = 8080

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((ip, port_no))
print("机械臂第一次连接", ip)

time.sleep(1)



# point6_00 = '{"command":"set_arm_power","arm_power":0}\r\n'
# _ = send_cmd(client, cmd_6axis=point6_00)
# time.sleep(0.5)
# print("获取工具坐标系")

while True:
#获取机械臂当前状态
    point6_00 = '{"command":"hand_follow_pos","hand_pos":[100]}\r\n'
    point6_01 = '{"command":"hand_follow_pos","hand_pos":[9000]}\r\n'
    move6_00='{"command":"movej_canfd","joint":[90, 0, 0, -90, 0, -90, 60],"follow":false,"expand":0,"trajectory_mode":0,"radio":50}\r\n'
    # move6_01='{"command":"movej_canfd","joint":[100, 5, 0, -90, 0, -90, 60],"follow":false,"expand":1000,"trajectory_mode":1,"radio":50}\r\n'
    t1=time.time()
    _ = send_cmd_nr(client, cmd_6axis=point6_00)
    print("time1",time.time()-t1)
    # time.sleep(2)
    t2=time.time()
    _ = send_cmd_nr(client, cmd_6axis=point6_01)
    print("time2",time.time()-t2)
    # time.sleep(2)
    t3=time.time()
    _ = send_cmd_nr(client, cmd_6axis=move6_00)
    _ = send_cmd_nr(client, cmd_6axis=move6_00)
    print("time3",time.time()-t3)
    time.sleep(2)
    t4=time.time()
    # _ = send_cmd_nr(client, cmd_6axis=move6_01)
    # print("time4",time.time()-t4)
    # time.sleep(2)
    print("获取工具坐标系")