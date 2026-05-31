# `jimu_14tray_5base_only.json` 坐标说明

这个文件只描述 19 个积木片的位置：

- 桌面底板 5 片：`floor`, `floor_right_support`, `floor_back_support`, `floor_left_support`, `floor_front_support`
- 料盘槽位 14 片：`tray_slot_roles` 里的 14 个角色
- 不包含整体定位物体：没有 `jimu_base_assembly`，也没有 `jimu_liaoban`

最容易出错的一点：`translation_m` 不是机器人/仿真世界坐标，它是相机坐标。仿真或机器人 base 下的位置要用 `T_base_obj` 或 `base_position_m`。

## 字段含义

每个 `results[]` 条目里关键字段如下：

| 字段 | 含义 | 是否建议下游使用 |
| --- | --- | --- |
| `T_base_obj` | 4x4 齐次矩阵，物体局部坐标到机器人 base 坐标：`p_base = T_base_obj @ p_obj` | 推荐 |
| `base_position_m` | `T_base_obj[:3, 3]` 的冗余展开，单位 m | 推荐只读位置 |
| `T_cam_obj` | 4x4 齐次矩阵，物体局部坐标到相机坐标：`p_cam = T_cam_obj @ p_obj` | 只给 SAM6D/相机链路兼容用 |
| `translation_m` | `T_cam_obj[:3, 3]`，也就是相机系位置 | 不要当世界坐标用 |
| `camera_position_m` | `T_cam_obj[:3, 3]` 的显式名字 | 不要当世界坐标用 |
| `mesh_file` / `sim_asset_file` | 渲染/仿真用 mesh | 可用 |
| `mesh_scale` / `sim_asset_scale` | mesh 缩放 | 可用 |

如果别人加载后位置不对，第一件事就是检查是不是用了 `translation_m`。正确示例：

```python
import json
import numpy as np

data = json.load(open("jimu_14tray_5base_only.json"))
for item in data["results"]:
    name = item["object_name"]
    T_base_obj = np.asarray(item["T_base_obj"], dtype=np.float32)
    p_base = T_base_obj[:3, 3]
    R_base_obj = T_base_obj[:3, :3]
```

SAPIEN 里如果世界坐标就等于机器人 base 坐标，可以直接：

```python
# T_base_obj maps object local frame to SAPIEN/robot-base world frame.
actor.set_pose(sapien.Pose(T_base_obj[:3, 3], quat_wxyz_from_matrix(T_base_obj[:3, :3])))
```

注意 SAPIEN 的四元数顺序是 `wxyz`。ROS 常见是 `xyzw`，不要混用。

## 矩阵方向

本文件里的 `T_base_obj` 是“从 object 到 base”，不是“从 base 到 object”。

正确：

```python
p_base_h = T_base_obj @ p_obj_h
```

如果你的程序需要 `T_obj_base`，要显式取逆：

```python
T_obj_base = np.linalg.inv(T_base_obj)
```

不要转置矩阵；平移在最后一列：

```python
x = T_base_obj[0][3]
y = T_base_obj[1][3]
z = T_base_obj[2][3]
```

## 坐标系来源

这个 JSON 是从 `current_apriltag_scene_two_square_layers_roof4.json` 的 `jimu_expanded_scene` 抽出来的。

源代码逻辑在：

```text
Beta_demo-codex-v0.9/rm75_jimu_four_wall_portable.py
  _build_jimu_expanded_scene_export()
  _jimu_export_pose_entry()
  _jimu_base_support_local_poses()
  _jimu_tray_slot_local_poses()
```

整体流程：

1. 用 AprilTag 或定位结果得到两个整体锚点：
   - `jimu_base_assembly`
   - `jimu_liaoban`
2. 把底板锚点 canonical 成中间底板 `floor` 的姿态。
3. 把料盘锚点 canonical 成料盘坐标系。
4. 用固定局部偏移生成 5 个桌面底板片和 14 个料盘槽位片。
5. 导出时保留每个片的 `T_base_obj`，同时为了兼容 SAM6D 风格结果，也写入 `T_cam_obj` 和 `translation_m`。

相机和 base 的转换在本项目里默认是：

```python
T_base_obj = inv(T_base_cam) @ T_cam_obj
```

如果程序启用了 `--use-direct-camera-extrinsic`，才是：

```python
T_base_obj = T_base_cam @ T_cam_obj
```

但这个 `jimu_14tray_5base_only.json` 已经写好了 `T_base_obj`，下游通常不需要再从 `T_cam_obj` 转一次。

## 底板五片的生成逻辑

当前逻辑片尺寸是：

```text
plate_extents_m = [0.074, 0.006, 0.074]
```

也就是 74 mm x 74 mm，厚度 6 mm。这里厚度轴是积木片局部 `+Y`。

底板中心高度：

```python
table_top_z = curobo_table_z_offset + 0.5 * curobo_table_thickness
floor_center_z = table_top_z + 0.5 * plate_thickness
```

当前默认 `table_top_z = 0.0`，所以底板中心 `z = 0.003 m`。

四周支撑片相对 `floor` 的局部偏移：

```text
floor_right_support: +0.074 m along floor local X
floor_left_support : -0.074 m along floor local X
floor_back_support : +0.074 m along floor local Z
floor_front_support: -0.074 m along floor local Z
```

在当前 JSON 里，这 5 个中心点的 base 坐标是：

| role | base_position_m |
| --- | --- |
| `floor` | `[0.1210035, -0.2112420, 0.0030000]` |
| `floor_right_support` | `[0.1950035, -0.2112420, 0.0030000]` |
| `floor_back_support` | `[0.1210035, -0.2852420, 0.0030000]` |
| `floor_left_support` | `[0.0470035, -0.2112420, 0.0030000]` |
| `floor_front_support` | `[0.1210035, -0.1372420, 0.0030000]` |

## 料盘 14 片的生成逻辑

料盘本体没有导出到这个 JSON，只导出了槽里的 14 个片。槽位来自料盘 mesh bounds 和参数：

```text
jimu_tray_slot_rows = 2
jimu_tray_slot_columns = 7
jimu_tray_slot_x_margin_m = 0.00875
jimu_tray_slot_insertion_depth_m = 0.012
```

局部槽位坐标：

```python
x_values = linspace(tray_min_x + 0.00875, tray_max_x - 0.00875, 7)
y_values = [
    tray_min_y + 0.20 * (tray_max_y - tray_min_y),
    tray_min_y + 0.80 * (tray_max_y - tray_min_y),
]
z_center = tray_max_z + 0.5 * plate_size - insertion_depth
```

槽内片的局部姿态约定：

```text
plate local X -> tray -Y
plate local Y/thickness -> tray +X
plate local Z -> tray +Z/up
```

也就是代码里的：

```python
R_tray_plate = [
    [0,  1, 0],
    [-1, 0, 0],
    [0,  0, 1],
]
```

最终：

```python
T_base_slot = T_base_tray @ T_tray_slot
```

当前 JSON 里，料盘 14 片的 base 坐标是：

| role | base_position_m |
| --- | --- |
| `right_wall` | `[-0.2351502, -0.1772885, 0.0500000]` |
| `back_wall` | `[-0.2351502, -0.1397886, 0.0500000]` |
| `left_wall` | `[-0.2351502, -0.1022886, 0.0500000]` |
| `front_wall` | `[-0.2351502, -0.0647886, 0.0500000]` |
| `right_roof_triangle` | `[-0.2351502, -0.0272886, 0.0805000]` |
| `back_roof_triangle` | `[-0.2351502, 0.0102114, 0.0805000]` |
| `left_roof_triangle` | `[-0.2351502, 0.0477114, 0.0805000]` |
| `jimu_spare_slot_09` | `[-0.3551502, -0.1772885, 0.0500000]` |
| `right_second_wall` | `[-0.3551502, -0.1397886, 0.0500000]` |
| `back_second_wall` | `[-0.3551502, -0.1022886, 0.0500000]` |
| `left_second_wall` | `[-0.3551502, -0.0647886, 0.0500000]` |
| `front_second_wall` | `[-0.3551502, -0.0272886, 0.0500000]` |
| `front_roof_triangle` | `[-0.3551502, 0.0102114, 0.0805000]` |
| `jimu_spare_slot_10` | `[-0.3551502, 0.0477114, 0.0500000]` |

## 发给别人时怎么重定位

这个 JSON 的 `T_base_obj` 是在我当前机器人 base 坐标系下的绝对位姿。别人电脑或仿真里如果世界原点、桌面位置、机器人 base 不一样，直接用这些数当然会偏。

如果在别人那里“相对布局是对的，但整套场景一起旋转了 90/180 度”，通常不是每个物体的姿态坏了，而是少乘了一个全局坐标系对齐矩阵：

```text
你这里的 base/world 坐标系  !=  对方程序里的 world 坐标系
```

这种情况不要逐个改物体姿态，要先求一个统一的 `T_world_from_json`，再左乘到所有物体：

```python
T_world_obj = T_world_from_json @ T_json_obj
```

如果别人只是想保留整套相对布局，但换一个世界位置，可以用一个新锚点整体搬过去。例如以 `floor` 为锚点：

```python
T_json_floor = results_by_role["floor"]["T_base_obj"]
T_world_floor_desired = ...  # 对方希望 floor 在自己世界里的位姿

T_world_from_json = T_world_floor_desired @ np.linalg.inv(T_json_floor)

for role, item in results_by_role.items():
    T_json_obj = np.asarray(item["T_base_obj"])
    T_world_obj = T_world_from_json @ T_json_obj
```

如果只知道差了一个绕桌面法线的 yaw，比如差 90 度，可以先临时绕 `floor` 中心整体旋转验证：

```python
def Rz(theta_deg):
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T

def Trans(p):
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = np.asarray(p, dtype=np.float32)
    return T

T_json_floor = np.asarray(results_by_role["floor"]["T_base_obj"], dtype=np.float32)
p = T_json_floor[:3, 3]
T_fix = Trans(p) @ Rz(90.0) @ Trans(-p)

for role, item in results_by_role.items():
    T_world_obj = T_fix @ np.asarray(item["T_base_obj"], dtype=np.float32)
```

这个只适合调试。正式做法还是用对方自己的 `floor`、底板 AprilTag 或料盘 AprilTag 求 `T_world_from_json`。

如果桌面底板和料盘在对方场景里是两个独立摆放的东西，不应该用同一个 `floor` 锚点整体搬运。应该分两组：

- 5 个底板片：用 `floor` 或底板 AprilTag 作为锚点重定位
- 14 个料盘片：用料盘 AprilTag 或料盘整体 pose 作为锚点重定位

这个 `only` JSON 删除了 `jimu_liaoban` 和 `jimu_base_assembly`，所以它不适合让别人从两个整体锚点重新展开槽位。如果需要跨机器复现，建议同时发原始完整文件：

```text
current_apriltag_scene_two_square_layers_roof4.json
```

或者重新导出一个“含两个 anchor + 19 个片”的版本。

## 常见错误排查

1. 把 `translation_m` 当世界坐标用：错。它是相机系。
2. 把 `T_base_obj` 取反了：错。它已经是 object 到 base。
3. 把矩阵转置了：错。平移在最后一列。
4. 单位当成 mm：错。所有坐标单位都是 m。
5. 四元数顺序用错：SAPIEN 是 `wxyz`，ROS 通常是 `xyzw`。
6. 别人的世界原点和 RM75 base 不一致：需要用上面的锚点重定位。
7. 想看料盘本体：这个 JSON 没有料盘本体，只包含槽里的 14 个片。
