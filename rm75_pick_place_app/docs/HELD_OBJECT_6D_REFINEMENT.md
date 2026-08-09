# 被抓物体 6D 精调接口

这条能力默认关闭。`PickPlaceCoordinator` 只有在注入
`held_object_refiner` 时，才会在物体抬升并静止后运行一次局部 6D 精调；
未配置腕带时，原来的抓取、规划与放置行为不变。

## 数据约定

- `T_tcp_camera`：腕带相机坐标系在 TCP 坐标系下的位姿，来自手眼标定。
- `prior_T_tcp_object`：抓取后物体在 TCP 坐标系下的初始位姿。
- 局部后端输入：`T_camera_object = inv(T_tcp_camera) @ T_tcp_object`。
- 局部后端输出会转换回 `T_tcp_object`，因此机械臂移动时不需要依赖固定的
  `T_base_camera`。

现有 RRTrack 的 `FoundationPoseRefiner` 已满足局部后端协议。腕带接入时只需实现
`HeldObjectFrameSource.capture()`，返回同步 RGB、深度、内参、物体掩码和
`T_tcp_camera`。没有硬件时可使用 `CachedHeldObjectFrameSource` 对录制帧离线验证。

## 执行与安全门限

精调发生在 `lift` 和 `preplace` 之间。默认门限为：掩码至少 80 像素、相对先验
平移不超过 35 mm、旋转不超过 18 度；也可以配置后端最低分数。结果被接受后，
Curobo2 会按新的 `T_tcp_object` 更新附着碰撞体，再规划后续运输和放置。

精调被拒绝或后端报错时采用 fail-open：保留抓取时的附着位姿并继续原流程，结果
会写入 `PickPlaceRunResult.held_object_refinement`，Curobo2 入口也会将它记录到
`result.json`。

## 组合方式

```python
backend = FoundationPoseRefiner(...)
source = MyWristFrameSource(...)  # 实现 HeldObjectFrameSource
refiner = HeldObjectPoseRefiner(backend, HeldObjectRefinementConfig(...))
hook = LiftRefinementHook(refiner, source, initial_T_tcp_object)

coordinator = PickPlaceCoordinator(
    planner,
    executor,
    held_object_refiner=hook,
)
```

腕带到位后仍需完成两项现场工作：可靠的 TCP-相机外参标定，以及抬升静止帧里被抓
物体掩码的提供（建议复用 Cutie 跟踪结果）。这两项都被隔离在 frame source 中，
不会再改主任务状态机。
