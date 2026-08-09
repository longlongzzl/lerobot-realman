# 未见物体动态几何入口

这条入口解决“没有 CAD 模型的刚体如何进入现有 pick-place/curobo 主线”，不生成抓取动作，也不替代 cuRobo。它把外观感知和运动规划之间的边界固定成版本化碰撞几何。

```text
SAM3 初始实例 mask / CUTIE 后续 mask
                 ↓
初帧 RGB-D → RaySt3R 生成背面先验点云
                 ↓
      对象模型坐标系的双层点图
      ├─ generated prior：低权重、补不可见面
      └─ observed RGB-D：高权重、覆盖邻近生成点
                 ↓
腕带新视角 → 运动学预测位姿 → ICP 残差精修 → 门控
                 ↓
信息增益触发重建
      ├─ visual_mesh.ply：Poisson（无 Open3D 时退化到凸包）
      └─ collision_mesh.obj：膨胀的保守凸包
                 ↓
latest_curobo_obstacle.json → 仅在下一次 replan 前更新 cuRobo world
```

## 为什么分两层几何

RaySt3R 官方输出是经过置信度过滤的补全点云，不是可直接用于碰撞检查的 mesh。生成区域只能当遮挡面的先验；腕带相机真正看到的新深度具有更高可信度，会删除邻域内的生成点并写入观测图。这样可以在初始单视角时保守补全，也可以随新视角逐步纠正形状。

碰撞 mesh 使用膨胀凸包，避免 Poisson 小孔或生成噪声导致规划穿模。visual mesh 独立保存，允许以后替换成更精细的 TSDF/神经重建而不改变规划契约。

## 运行

初始目录需要 `rgb.png`、`depth.png|depth.npy`、`mask.png|sam6d_binary_mask.png` 和 `camera.json`：

```bash
python -m rm75_app openworld-geometry -- \
  --instance-id unknown_01 \
  --initial-frame-dir <initial_frame> \
  --initial-T-base-camera <T_base_camera.npy> \
  --rayst3r-root runtime_data/third_party/RaySt3R \
  --rayst3r-python <rayst3r_env_python> \
  --updates-manifest <updates.json>
```

没有 RaySt3R 环境时，可以先验证实测表面融合和 mesh 交接：

```bash
python -m rm75_app openworld-geometry -- \
  --instance-id unknown_01 \
  --initial-frame-dir <initial_frame> \
  --provider observed
```

更新清单示例：

```json
[
  {
    "frame_dir": "wrist_0001",
    "T_base_camera": "wrist_0001/T_base_camera.npy",
    "predicted_T_base_model": "wrist_0001/T_base_tcp_times_T_tcp_obj.npy"
  }
]
```

桌面阶段不填 `predicted_T_base_model` 时，以上一帧对象位姿作为预测。抓取之后必须从机器人 FK 和抓取关系提供 `T_base_tcp · T_tcp_obj`，ICP 只负责修正运动学/夹持误差。`--trust-camera-poses` 会跳过 ICP，仅用于标定回放或调试。

## 接入 cuRobo

每次重建产生不可变目录 `v0001/`、`v0002/`；`latest_state.json` 和 `latest_curobo_obstacle.json` 原子切换到最新可用状态。规划编排层使用 `DynamicGeometryWorld.stage(snapshot)` 暂存一个或多个对象，在当前轨迹完成/取消后、下一次规划前调用 `apply_before_replan(planner)`。不要在 MotionGen 正在执行的轨迹中更新 world。

## 当前边界

- 仅支持刚体；软物体、关节物体需要不同的形变模型。
- mask 由 SAM3/CUTIE 提供，本入口不会自行决定实例身份。
- RaySt3R 保持外部依赖，不复制或修改其源码；其官方许可限定为学术/非营利的非商业研究，商业部署必须另行解决授权或替换 provider。
- 官方 RaySt3R 点云没有逐点导出原始置信度，因此适配器把阈值过滤后的点统一作为低权重先验。
