# cuRobo Collision World 维护流程

这份文档只总结当前 Jimu 搭建脚本里的碰撞世界维护逻辑，重点是 mesh/box/attached payload 怎么进入 cuRobo。需要先区分两套世界：

- ManiSkill 仿真世界：负责渲染、物理、Actor、真实执行预览。
- cuRobo 规划世界：负责 IK、MotionGen、轨迹碰撞检查。

两者会同步姿态和可视化，但不是同一个碰撞数据结构。

## 结论

当前默认不是所有 mesh 都直接进 cuRobo。

- 大多数已放置/未轮到的积木：以 `cuboid` 进入 cuRobo world。
- 被抓住的当前目标物体：以 `attached_object` 的球模型挂到机器人 TCP 链路上。
- 只有显式加入 `args.curobo_world_mesh_object_names` 的物体：才以真实 mesh 进入 cuRobo world。
- 屋顶三角片可以通过 `--jimu-roof-curobo-mesh-obstacles` 开启真实 mesh 障碍；默认是关闭的。
- 活动目标物体通常不作为 scene obstacle 放进 world，抓住后由 attached payload 表示，释放后再作为 placed obstacle 加回去。

## 关键文件

- `Beta_demo-codex-v0.9/rm75_jimu_four_wall_portable.py`
  - Jimu 场景、角色、物体规格、attached payload 球模型。
  - 会 monkey patch direct 脚本里的 attach 逻辑。
- `Beta_demo-codex-v0.9/rm75_jimu_triangle_roof_apriltag_portable.py`
  - 三角屋顶角色、三角 mesh 开关、屋顶碰撞缩放。
- `pick_jiaobang/rm75_jiaobang_pick_place_targeted_curobo_direct_pre_place.py`
  - 主 pick/place 流程、cuRobo world refresh、放置后回撤/回初始位时的障碍维护。
- `pick_jiaobang/rm75_jiaobang_pick_place_targeted_curobo.py`
  - `demo.scene_obstacles -> cuRobo cuboids/meshes` 的转换函数。
- `Beta_demo-codex-v0.9/curobo_rm75_planner.py`
  - `WorldConfig` 构建、`motion_gen.update_world()`、`ik_solver.update_world()`、attached spheres。

## 物体来源

每个逻辑物体先来自 object spec 或 portable wrapper 的覆盖配置：

```text
object_name
mesh_file
mesh_scale
sim_asset_file
sim_asset_scale
scene_obstacle_box_scale
```

含义：

- `mesh_file`：通常给 SAM6D/FoundationPose/CAD 模板用。
- `sim_asset_file`：通常给 ManiSkill 渲染/仿真、尺寸估计、planner box 用。
- `scene_obstacle_box_scale`：把 mesh 的 AABB 缩放成 planner box 时使用。
- Jimu 当前默认尺寸主要按 `74mm x 6mm x 74mm` 的逻辑板尺寸维护。

## 场景定位到 scene cache

定位阶段会得到物体位姿，Jimu 当前常用两种来源：

- SAM3/SAM6D 多实例定位。
- AprilTag anchor 定位，然后从底板/料盘整体位姿推导每个槽位和每个积木角色的 `T_world_obj`。

这些位姿进入 `scene_capture_cache`，后续单场景循环从里面取：

```text
object_name -> T_world_obj / score / label / placed
```

## demo.scene_obstacles 的维护

每一轮切换 active target 时，会同步 `demo.scene_obstacles`。

核心逻辑：

```text
selected target = 当前要抓的物体
desired obstacles = 当前轮规划要避开的物体
selected target 从 desired obstacles 中移除
每个 obstacle 生成一个 obstacle entry
demo.scene_obstacles = obstacle_entries + virtual_entries
```

典型 obstacle entry：

```python
{
    "object_name": "right_wall",
    "actor_name": "scene_obstacle_right_wall",
    "T_world_obj": T_world_obj,
    "placed": True/False,
    "planner_collision": True,
    "asset_file": sim_asset_file,
    "asset_scale": sim_asset_scale,
    "visual_box_size": mesh_aabb_size,
    "planner_box_size": visual_box_size * box_scale,
}
```

关键点：

- `planner_collision=False` 的物体不会进 cuRobo world。
- 当前 active object 通常不在 `demo.scene_obstacles`，避免抓取时把目标物体当障碍。
- 已放置物体会在后续轮次作为 obstacle 加回 `demo.scene_obstacles`。
- `floor`、底板四周支撑片、已搭好的墙、第二层、屋顶都通过这条机制进入 obstacle 列表。

## scene_obstacles 转 cuRobo world

转换入口：

```python
curobo_wrapper._scene_obstacles_to_curobo_world(demo, args, exclude_object_names=...)
```

规则：

1. 遍历 `demo.scene_obstacles`。
2. 跳过 `planner_collision=False` 的 entry。
3. 跳过 `exclude_object_names` 里的对象。
4. 读取 `T_world_obj`，转换成 `[x, y, z, qw, qx, qy, qz]`。
5. 如果 `object_name in args.curobo_world_mesh_object_names` 且存在 `asset_file`：
   - 输出到 `meshes`：

```python
{
    "name": actor_name,
    "file_path": asset_file,
    "scale": [asset_scale, asset_scale, asset_scale],
    "pose": pose_world,
}
```

6. 否则输出到 `cuboids`：

```python
{
    "name": actor_name,
    "dims": planner_box_size,
    "pose": pose_world,
}
```

所以 mesh 进 cuRobo 的唯一普通路径是：

```text
demo.scene_obstacles entry
-> object_name 命中 curobo_world_mesh_object_names
-> asset_file 存在
-> _scene_obstacles_to_curobo_world 输出 meshes
-> _refresh_curobo_world 写入 planner
```

## 坐标系转换

`demo.scene_obstacles` 里的 pose 是 world frame。

cuRobo planner 使用 robot base frame，所以刷新 world 前会做：

```python
curobo_wrapper._transform_curobo_world_to_robot_base(cuboids, meshes, demo)
```

转换逻辑：

```text
T_base_obj = inv(T_world_base) @ T_world_obj
```

转换后再交给 planner。

## _refresh_curobo_world 的刷新流程

主入口：

```python
direct._refresh_curobo_world(
    planner,
    demo,
    args,
    label=...,
    include_active_object=False,
    include_table=False,
    exclude_object_names=None,
    extra_scene_obstacles=None,
)
```

流程：

1. 从 `demo.scene_obstacles` 生成 `cuboids, meshes`。
2. 如果有 `extra_scene_obstacles`，额外生成一批 cuboids/meshes 并追加。
3. 如果 `include_active_object=True`，把当前 active object 加入 world。
4. 如果 `include_table=True`，追加虚拟桌面 cuboid。
5. 计算 world signature。
6. 如果 signature 没变，复用旧 world，避免重复更新。
7. 如果变了：
   - 同步 debug 可视化 actor。
   - world frame 转 robot base frame。
   - 调 `planner.set_world_from_obstacles(cuboids=..., meshes=...)`。
   - 更新 `planner._persistent_world_signature`。

这也是日志里这些信息的来源：

```text
[curobo] updated world with N cuboid and M mesh obstacles
[curobo] reused persistent world
[curobo] label world: excluded scene obstacle(s): [...]
```

## planner 内部怎么接收 mesh

入口：

```python
RM75CuRoboPlanner.set_world_from_obstacles(cuboids=..., meshes=...)
```

内部：

1. `build_world_from_obstacles()` 构造 `WorldConfig.from_dict(world_dict)`。
2. cuboid 写入：

```python
world_dict["cuboid"][name] = {
    "dims": dims,
    "pose": pose,
}
```

3. mesh 写入：

```python
world_dict["mesh"][name] = {
    "file_path": file_path,
    "scale": [sx, sy, sz],
    "pose": pose,
}
```

4. 如果第一次出现 mesh world：
   - 重建 `ik_solver`。
   - 重建 `motion_gen`。
   - 标记 `_mesh_world_initialized=True`。

原因是 cuRobo mesh world 的缓存结构和纯 cuboid world 不一样，第一次切到 mesh 时需要重新初始化。

后续更新：

```python
motion_gen.update_world(world_cfg)
ik_solver.update_world(world_cfg)
_update_cuda_graph_batch_ik_world(world_cfg)
```

## 当前 Jimu 的 attached payload

抓住目标后，目标物体不再作为 world obstacle 存在，而是挂到机器人上：

```text
active source object
-> 从 obstacle world 排除
-> force_active_object_to_attached_pose()
-> attach_transport_payload_to_curobo_jimu()
-> attached_object link spheres
```

Jimu wrapper 覆盖了原始 direct attach 逻辑，用平面球阵列表示被抓的片：

```python
_attach_jimu_planar_spheres_to_robot(...)
planner.motion_gen.attach_spheres_to_robot(..., link_name="attached_object")
planner.ik_solver.attach_object_to_robot(..., link_name="attached_object")
```

普通方片：

- 根据 `jimu_plate_size_m`、`jimu_plate_thickness_m` 或 asset AABB 得到尺寸。
- 默认使用 `plate_corners_edges` 布局。
- 典型日志：

```text
[jimu-curobo] attached planar Jimu payload grid:
layout=plate_corners_edges, dims=[74.0, 6.0, 74.0]mm, spheres=6/6, radius=8.0mm
```

三角屋顶片：

- 使用 `triangle_inset_vertices_edges` 布局。
- 球心在三角形内部，球边覆盖角点和边中点。
- 避免之前球心直接放角点导致碰撞体过度外扩。

attached payload 的关键参数：

```text
--jimu-attached-sphere-radius-m
--jimu-attached-sphere-long-count
--jimu-attached-sphere-wide-count
--jimu-attached-sphere-span-scale
--jimu-attached-sphere-dim-scale
--curobo-attach-world-z-offset-m
--curobo-attach-min-start-clearance-m
```

## 抓取/运输/放置各阶段的碰撞状态

### start -> pregrasp

- 通常 world 包含非当前目标的 scene obstacles。
- 当前 active target 不作为普通 obstacle。
- 某些 pregrasp MotionGen 诊断/路径会临时 `include_active_object=True`，用于检查靠近目标时是否撞目标。

### pregrasp -> grasp

- 使用受限直线规划。
- 为了允许夹爪接触目标，有些规划会临时放松 gripper links 的 world collision。
- 目标本身仍不是 attached payload，直到夹爪闭合后。

### grasp close 后

- 程序记录抓取时 `T_tcp_obj`。
- 强制同步 active object 到夹爪 attached pose。
- 调 `attach_transport_payload_to_curobo_jimu()`。
- cuRobo 里出现 `attached_object` 球模型。

### transport hover

- 抓住的物体由 `attached_object` 表示。
- 原来的 source object 从 world obstacle 排除，避免重复碰撞。
- 其他已放置/未抓物体仍在 scene obstacles 中。
- 如果起点已经撞 world，会先做 world-Z lift，再重试运输规划。

### final contact / place

- 目标物体仍是 `attached_object`。
- 最后一段贴合通常用受限直线。
- 为了允许真实接触，部分 link 会临时 disabled，但 attached payload 仍用于整体避障。

### open gripper 后

- `planner.detach_object_from_robot()`。
- active object 从 attached payload 中移除。
- 当前物体的释放位姿会构造成 placed obstacle：

```python
_build_current_target_placed_obstacle(args, T_world_obj)
```

这个 placed obstacle 会加入 `extra_scene_obstacles`，用于：

- post-place clearance。
- return-to-start 预验证。
- 后续轮次障碍。

## extra_scene_obstacles 的作用

`extra_scene_obstacles` 是临时追加到 cuRobo world 的障碍，不一定已经写回 `demo.scene_obstacles`。

典型用途：

- 刚放下当前物体后，还没进入下一轮，但回撤/回初始位必须把它当障碍。
- post-place clearance 预检查。
- return-to-start 预检查。

也就是说：

```text
demo.scene_obstacles = 已存在场景障碍
extra_scene_obstacles = 当前刚放好的物体/临时障碍
cuRobo world = 两者合并
```

## exclude_object_names 的作用

有些规划段必须临时排除某些物体：

- 抓取源物体已经 attached 后，原 world 里的同名 obstacle 必须排除。
- final contact 时，有时目标连接对象需要特殊处理。
- 起步 lift 时可能排除 active/source，避免当前接触状态被误判成不可规划。

`_refresh_curobo_world()` 会打印：

```text
[curobo] <label> world: excluded scene obstacle(s): [...]
```

如果排除失败，会打印 warning。

## 屋顶 mesh 障碍

默认：

```text
DEFAULT_ROOF_CUROBO_MESH_OBSTACLES = False
```

即屋顶三角片默认还是 cuboid，只是 box scale 较小：

```text
--jimu-roof-scene-obstacle-box-scale
```

如果开启：

```bash
--jimu-roof-curobo-mesh-obstacles
```

会把屋顶角色加入：

```python
args.curobo_world_mesh_object_names
```

然后这些 role 的 scene obstacle 会走 mesh 路径：

```text
scene_obstacle right_roof_triangle
-> object_name 命中 curobo_world_mesh_object_names
-> asset_file = red_triangle.glb
-> meshes.append(...)
-> WorldConfig["mesh"]
```

注意：mesh world 第一次启用会触发 cuRobo planner/IK solver 重建，可能比 cuboid 慢。

## 虚拟桌面

如果 `include_table=True` 且 `--curobo-table-collision` 开启，会追加：

```python
{
    "name": "virtual_table_plane",
    "dims": [table_size_x, table_size_y, table_thickness],
    "pose": [table_x, table_y, table_z, 1, 0, 0, 0],
}
```

这是 cuboid，不是 mesh。

## Debug 和确认方法

常用日志：

```text
[single_scene] active=..., scene obstacle(s)=[...]
[curobo] updated world with N cuboid and M mesh obstacles for <label>
[curobo] reused persistent world for <label>
[jimu-curobo] attached planar Jimu payload grid: ...
[curobo] detached object from link=attached_object
```

建议排查顺序：

1. 先看 `[single_scene] active=..., scene obstacle(s)=...`，确认谁被放进 `demo.scene_obstacles`。
2. 打开 `--curobo-debug`，看 `updated world with N cuboid and M mesh obstacles`。
3. 如果 M=0，说明没有任何物体按 mesh 进入 cuRobo。
4. 如果被抓物体碰撞异常，看 `[jimu-curobo] attached planar Jimu payload grid` 的布局、半径、sphere 数量。
5. 如果刚放好的物体回撤/回初始位穿模，重点看 `extra_scene_obstacles` 和 `return_to_start` 的 world refresh 是否包含 placed obstacle。

## 当前维护原则

- 规划世界优先用 cuboid，速度快、稳定。
- 对三角屋顶这类 AABB 明显不准的物体，必要时才启用 mesh obstacle。
- 被抓物体不用 mesh world 表示，而用 attached spheres 表示，因为 cuRobo 的 attached object 本质就是挂到机器人链路上的球集合。
- 放置完成后必须 detach，再把物体以 placed obstacle 加回 world，否则回撤和下一轮可能穿模。
- 每次 world 成员、尺寸、姿态变化都通过 signature 控制刷新，避免无意义地重建 cuRobo world。
