# RM75 Pick-Place App

这个目录现在是独立运行工程：pick-place 默认主线已切换为分层 Curobo2，感知、动态几何、候选生成、批量 IK/TrajOpt、状态编排和执行适配均有独立边界。旧 direct/SAM6D 单体流程只保留兼容，不再是默认主线。

## 目录结构

```text
rm75_pick_place_app/
  rm75_app/
    core/          # 与硬件/仿真无关的任务、阶段、结果契约
    tasks/         # pickplace / jimu / lego 任务适配器和注册表
    pipeline/      # 任务解析、校验、计划编译和后端边界
    pickplace/     # pick-place 默认配置、后端选择、层级绑定和 runner
    legacy/        # 尚未迁移的 Jimu/Lego 旧后端路径
    runtime/       # direct pick-place、SAM6D pick-place、FoundationPose scene pipeline
    perception/    # SAM3/SAM6D providers、RRTrack closed-loop tracker
    planning/      # planner-independent contracts + Curobo2 batch backend
    execution/     # 可移植轨迹包和真机/仿真执行适配器
    placement/     # 放置规则
    assets/        # object specs
    llm/           # LLM pick-place orchestrator
    web/           # Web 控制台
    config/        # 默认对象和通用参数
    paths.py       # 本目录内路径定义
  assets/
    meshs/
    curobo_rm75_config/
    test_scenes/
    robot_models/
    calibration/
    lego/          # Lego task JSON schema and smoke task
  runtime_data/
    planning_profile_logs/
    sam6d_grasp_scene_runs/
    sam6d_groundingdino_runs/
    sam6d_template_cache/
    sam6d_pem_feature_cache/
    failure_renders/
    llm_pick_place_runs/
```

## 主线分层

`rm75_pick_place_app` 是当前唯一重构主线。向上依赖关系固定为：

```text
任务适配器（pickplace / jimu / lego）
                 ↓
        core contracts + pipeline
                 ↓
 perception → candidate/place policy → planning → execution → validation
                 ↓
       Curobo2 规划 / ManiSkill 回放 / 可替换执行器
```

- `core/` 只定义任务请求、阶段、计划、命令和结果，不导入 NumPy、ManiSkill、cuRobo 或硬件 SDK。
- `tasks/` 只描述任务差异。抓取放置、Jimu 装配、Lego snap 都通过同一套阶段契约接入。
- `perception/`、`planning/`、`execution/` 是可替换能力层；新主线不导入旧的超大运行脚本。
- `pickplace/coordinator.py` 显式管理 approach、grasp、attach、lift、preplace、place、detach 和 retreat；Curobo2 张量不会越过 `planning/` 边界。
- 可提前求解的抓取/放置候选 IK 按批处理；attach/detach 改变碰撞世界，因此对应轨迹优化仍保持顺序边界。
- 1.0.24 已删除旧的 2 万行 direct 单体执行器及其 cuRobo v1/targeted/SAM6D/wrist 包装链；历史实现仍可从 Git 提交 `8aa9fae` 恢复。
- 1.0.25 补齐多物体原子任务、LLM 清单执行、前端场景工作台与 geometry/cuRobo2/ManiSkill 三级验证，并将 cuRobo2 候选筛选和分段轨迹规划切到常驻批量求解与 CUDA Graph 快速路径。
- 1.0.25 的固定场景 geometry 与 cuRobo2 验证已通过；ManiSkill 场景预览已接通，但物理夹取仍需继续校准 TCP、夹爪碰撞体和控制跟踪，当前不作为真机流程的通过依据。
- `direct` 只作为 `curobo2` 的命令别名；`sam6d` 现在只运行位姿感知，不再隐式启动旧抓取执行器。
- Jimu 和 Lego 现在是明确标记为 `compatibility` 的适配器：统一入口已经在主线，具体旧执行器还没有搬进来，避免把旧耦合伪装成完成迁移。

查看任务注册表和编译后的兼容命令：

```bash
python -m rm75_app tasks list
python -m rm75_app tasks info pickplace
python -m rm75_app tasks info jimu
python -m rm75_app tasks command pickplace --mode curobo2 -- --help
python -m rm75_app tasks command jimu --mode four-wall
python -m rm75_app tasks command lego --mode dry-run
python -m rm75_app pickplace --mode curobo2 -- --help
```

完整的边界和迁移顺序见 `docs/TASK_ARCHITECTURE.md`。

## 运行方式

```bash
cd /home/zhangzhao/Desktop/lerobot/rm75_pick_place_app
python -m rm75_app --help
python -m rm75_app pickplace -- --help
python -m rm75_app curobo2-pickplace -- --cached-pose-result <sam6d_pose_result.json>
python -m rm75_app curobo2-sim-replay -- --manifest <run_dir/execution.json>
python -m rm75_app direct -- --help  # curobo2 compatibility alias
python -m rm75_app sam6d -- --help  # perception only
python -m rm75_app rrtrack -- --help
python -m rm75_app openworld-geometry -- --help
python -m rm75_app tabletop-refine -- --help
python -m rm75_app web -- --host 127.0.0.1 --port 7860
python -m rm75_app llm -- --help
python -m rm75_app verify
```

本机 Curobo2 与 ManiSkill 依赖位于不同 Python 环境。规划器输出小型 `execution.json` 清单和每阶段压缩 NPZ，仿真/真机进程只消费这个可移植轨迹包，不加载 Curobo2。Web 控制台提供“Curobo2 缓存全链路”和“仿真回放最新规划”两个入口。

## 场景工作台

Web 默认首页现在是开放世界场景工作台。启动后访问 `http://127.0.0.1:7860`：

```bash
python -m rm75_app web -- --host 127.0.0.1 --port 7860
```

“扫描桌面并定位”在同一张 RGB-D 帧上运行两路感知：

```text
ObjectSpec prompts → SAM3 多实例 mask → SAM6D pose → 已见/不确定
桌面物体通用 prompt → SAM3 类别无关候选 ─┘ mask 去重 → 未见
```

扫描结果持久化到 `runtime_data/scene_workbench/state.json`。每个动作绑定
`snapshot_id + scene_version + instance_id`；计划校验通过后快照被冻结，避免重新扫描后静默串号。

- 已见：资产和 6D 位姿均可用，可以加入 pick-place 动作。
- 不确定：分割存在但位姿未通过，或人工指定了资产候选；必须重新扫描验证。
- 未见：通用桌面候选未与已知资产 mask 匹配，只能先生成几何处理任务。
- 忽略：保留在快照中，但不参与计划。

未见任务包包含初始 `rgb.png`、`depth.png`、`camera.json` 和 `mask.png`。页面可以直接启动
`observed` 或 `RaySt3R` 几何入口；只有任务变为 `geometry_ready` 后，未知实例才允许通过动作计划校验。
RaySt3R 会占用 GPU，启动前页面要求先停止 SAM3/SAM6D 常驻模型。几何更新仍只在下一次 replan
边界交给 cuRobo，不会在正在执行的轨迹中修改碰撞世界。

当前低层 pick-place 执行器仍按资产名寻址，因此工作台会阻止同一资产的多个实例同时进入可执行计划；
场景清单和 SAM6D 初始化已经保留多实例，后续只需把执行契约升级为 `instance_id`，无需再改扫描层。

## RRTrack 风格可恢复感知入口

`rrtrack` 是独立的新感知入口，不替换原有 `sam6d`。它依据 RRTrack 论文公开方法实现：

```text
SAM3 + SAM6D 初始化
        ↓
CUTIE mask 传播（先预测、禁止未验证写回）
        ↓
FoundationPose 局部 6D refine
        ↓
观测 mask / CAD 渲染 mask 的 P、S 一致性门控
        ├─ 局部漂移：mask 中心 + 中值深度 snap，再 refine
        ├─ 稳定：分别更新 CUTIE 短期/有界长期记忆和在线 DINO 库
        └─ 丢失/停滞：DINOv2 离线+在线双库各 Top-3
                       → FoundationPose 批量 refine + ranker 选优 → P 验收
                       ├─ CUTIE 无 mask：SAM3 全图多候选后走同一身份/几何验收
                       └─ 无候选：FoundationPose 球面注册
```

实时入口会先用现有 SAM3/SAM6D 获取初始 mask 和位姿，再接管 RealSense 视频：

```bash
python -m rm75_app rrtrack -- --object-name carriot
```

整桌多物体只在开始时做一次全局初始化；之后只有被抓对象进入 CUTIE/6D 连续跟踪，其他实例保存在本次运行的 `scene_registry.json` 中。相同类别用 `--active-instance-index` 选择：

```bash
python -m rm75_app rrtrack -- \
  --object-name redcube \
  --active-instance-index 1 \
  --scene-object-names redcube redcube tennis carriot
```

复用已经生成的 SAM6D 初始结果：

```bash
python -m rm75_app rrtrack -- \
  --object-name carriot \
  --init-result-json <sam6d_pose_result.json>
```

离线 RGB-D 序列要求 `rgb/`、`depth/` 和 `camera.json`；深度已经是米时保持默认 scale，uint16 毫米深度使用 `--sequence-depth-scale 0.001`：

```bash
python -m rm75_app rrtrack -- \
  --object-name carriot \
  --init-result-json <sam6d_pose_result.json> \
  --sequence-dir <sequence_dir> \
  --sequence-depth-scale 0.001
```

RRTrack 论文使用 256 个全球离线模板（128 个视角加 180° 平面内增强）。当前标准 SAM-6D renderer 实际生成 level0 的 42 个视角，因此项目使用与图片严格匹配的 `cam_poses_level0.npy`，生成 84 个有效条目；不能用 level2 姿态错误地解释这 42 张图。入口默认按对象名从 `runtime_data/rrtrack_banks/` 复用离线库，也可以手工建立或通过 `--offline-bank` 固定指定：

```bash
python -m rm75_app rrtrack-build-bank -- \
  --templates-dir <sam6d_run/templates> \
  --output runtime_data/rrtrack_banks/carriot_dinov2_vits14.npz

# 为 ObjectSpec 中所有已知物体批量生成；共用 mesh+缩放的对象只渲染一次
python -m rm75_app rrtrack-build-all-banks

python -m rm75_app rrtrack -- \
  --object-name carriot \
  --offline-bank runtime_data/rrtrack_banks/carriot_dinov2_vits14.npz
```

批量结果记录在 `runtime_data/rrtrack_banks/manifest.json`。当前 24 个 ObjectSpec 按 mesh+真实缩放去重为 18 套模板，所有对象名各有一个可自动发现的 `<object>_dinov2_vits14.npz`。

CUTIE 是外部模型依赖，不提交进项目源码；当前机器的官方 MIT 版本位于被忽略的 `runtime_data/third_party/Cutie`，也可通过 `--cutie-root` 指定其他安装目录。入口只需要官方 `cutie-base-mega.pth`，默认从 `<cutie-root>/weights/` 加载，也可用 `--cutie-weights` 指定。DINOv2 默认优先使用本机 torch hub 缓存。SAM3 只在 CUTIE 持续空 mask 时按需启动，可用 `--disable-sam3-recovery` 关闭。论文没有公开所有门控阈值，未公开项集中在 `rm75_app/perception/rrtrack/config.py`，先使用保守项目默认值，再只用稳定跟踪帧做 EMA–MAD 自适应。

当前 `pickplace --mode rrtrack` 指向这条感知入口，输出 `trajectory.jsonl`、`latest_pose.json`、`scene_registry.json` 和 `run_config.json`；它暂不自动触发机械臂运动，只有经过门控的 `latest_pose.json` 才应交给规划层。

阈值或状态策略可通过 `--rrtrack-config <json>` 覆盖，运行目录中的 `run_config.json` 会保存最终生效配置，便于按真实遮挡序列回放调参。

## 未见物体动态几何入口

`openworld-geometry` 为没有 CAD 的刚体建立 RaySt3R 生成先验，并用后续腕带 RGB-D 新视角动态覆盖、融合和版本化重建。它输出独立的 visual mesh 与保守 collision mesh；规划仍由现有 cuRobo 完成，不引入另一套 grasp/motion planner。

```bash
python -m rm75_app openworld-geometry -- \
  --instance-id unknown_01 \
  --initial-frame-dir <rgbd_mask_frame> \
  --initial-T-base-camera <T_base_camera.npy> \
  --updates-manifest <wrist_frames.json>
```

详细的数据格式、抓取后运动学预测位姿和 cuRobo 重规划交接见 `docs/OPENWORLD_DYNAMIC_GEOMETRY.md`。

## 桌面物体位姿精修

SAM6D 全局定位之后，可以单独运行桌面物体 `x/y/yaw` 精修入口：

```bash
python -m rm75_app tabletop-refine -- --sam6d-result <full_scene_pose_results.json>
```

这个入口不会重新跑 SAM3/SAM6D，也不会影响原始 SAM6D 输出。它读取 SAM6D 保存的 `shared_frame/rgb.png`、`depth.png`、`camera.json` 和每个物体的 `sam6d_binary_mask.png`，把 SAM6D 结果当粗定位，只在机器人基座/桌面平面内搜索 `x`、`y` 和 `yaw`，锁定 `z/roll/pitch`，用 CAD 投影 mask IoU、bbox/中心误差和 depth residual 打分。输出写到：

```text
runtime_data/tabletop_pose_refine_runs/<timestamp>_tabletop_refine/tabletop_refined_scene_results.json
```

输出仍保留 `results[].T_cam_obj` 字段，因此可以继续作为 Curobo2 缓存场景输入：

```bash
python -m rm75_app curobo2 -- --cached-pose-result <tabletop_refined_scene_results.json>
```

常用调参：

```bash
python -m rm75_app tabletop-refine -- --sam6d-result <full_scene_pose_results.json> --object-names carriot gluestick lvmukuai --xy-search-radius-m 0.05 --yaw-search-radius-deg 45
```

如果某些物体不是稳定桌面状态，或需要 full 6D，不要放进这个精修入口；它默认只解决桌面物体的平面误差。

## 腕带相机抓取后精修

腕带相机的持物 6D 精调目前保留为默认关闭的接口，见
`docs/HELD_OBJECT_6D_REFINEMENT.md`。1.0.24 不包含真实腕带采集、夹爪遮挡处理或腕带执行入口。

## 相机标定入口

新标定代码在 `rm75_app/calibration/`，结果默认写到 `runtime_data/calibration_runs/`。

推荐标定栈是先用主相机固定全局关系，再用同一块固定 ChArUco 板约束腕带相机：

```bash
# 1. 主/固定相机：复用 EasyHEC 思路，用机械臂本体 mask + URDF 渲染做视觉优化
python -m rm75_app calib-base -- --config assets/calibration/rm75_calibration_config.example.json --execute-real

# 2. 固定板锚定：主相机观测固定 ChArUco 板，输出 T_R_P
python -m rm75_app calib-board-anchor -- --config assets/calibration/rm75_calibration_config.example.json --base-camera-run-dir <base_run_dir> --manual-capture --manual-count 20 --preview

# 3. 腕带锚定：腕带相机从多姿态看同一块固定板，用 T_R_P 优化 T_E_Cw
python -m rm75_app calib-wrist-anchor -- --config assets/calibration/rm75_calibration_config.example.json --board-anchor-run-dir <board_anchor_run_dir> --manual-capture --manual-count 40 --preview

# 4. 可选双相机联合优化：同时优化/校验 T_R_P 和 T_E_Cw
python -m rm75_app calib-joint-board -- --config assets/calibration/rm75_calibration_config.example.json --global-board-run-dir <board_anchor_run_dir> --wrist-board-run-dir <wrist_run_dir> --base-camera-run-dir <base_run_dir>

# 5. 可选腕带本体轮廓 sanity check，小幅视觉微调，不作为主标定路线
/home/zhangzhao/anaconda3/envs/foundationpose310/bin/python -m rm75_app calib-wrist-visual-refine -- --wrist-camera-run-dir <wrist_run_dir>

# 6. 最终联合校验：两个相机看同一块标定板，用主相机外参、腕带外参和机械臂 FK 互相检查
python -m rm75_app calib-check -- --config assets/calibration/rm75_calibration_config.example.json --execute-real --base-camera-run-dir <base_run_dir> --wrist-camera-run-dir <wrist_run_dir> --preview
```

现有经典腕带 ChArUco 手眼标定仍可作为可信 fallback 或无主相机板锚时使用：

```bash
# 腕带相机：相机看桌面 ChArUco 标定板，自动采集多姿态并解 ee_T_wrist_camera
python -m rm75_app calib-wrist -- --config assets/calibration/rm75_calibration_config.example.json --execute-real

# 先离线生成腕带采样计划，确认姿态范围再真机执行
python -m rm75_app calib-wrist -- --config assets/calibration/rm75_calibration_config.example.json --auto-generate-samples --dry-run-plan

# 自动从 seed pose 抖动生成 18 个姿态，采集时显示实时叠图
python -m rm75_app calib-wrist -- --config assets/calibration/rm75_calibration_config.example.json --execute-real --auto-generate-samples --preview

# 手动采集：实时显示角点/坐标轴，你自己摆姿态，按回车或空格拍照，不自动移动机械臂
/home/zhangzhao/anaconda3/envs/foundationpose310/bin/python -m rm75_app calib-wrist -- --config assets/calibration/rm75_calibration_config.example.json --manual-capture --manual-count 18
```

主相机视觉标定需要机器人 mask，可通过 `--mask-npy` 或 `--mask-dir` 提供；已有 EasyHEC 旧数据时可用 `--use-previous-run` 复用 `image_dataset.npy`、`link_poses_dataset.npy` 和 `mask.npy`。

每次运行会写一个带 `schema_version`、`run_kind`、`accepted`、`transforms` 和 `metrics` 的报告 JSON，以及 `report.html`。腕带标定会自动尝试多种 OpenCV 手眼算法并选择残差最小的结果；ChArUco 坏帧按角点数量、角点覆盖率和重投影误差过滤。腕带视觉微调不会覆盖标定板结果，只有明确 accepted 的微调才适合作为 runtime 外参。综合校验默认要求同帧主/腕相机误差不超过 `3cm / 5deg`，阈值可通过 `--max-main-wrist-translation-m` 和 `--max-main-wrist-rotation-deg` 调整。

插入前/入孔前的局部视觉伺服是后续独立层，不属于这套全局相机和腕带相机标定栈。

更完整的运行顺序见 `rm75_app/calibration/README.md`。

## 独立性边界

pick-place direct/SAM6D 主链不依赖外层旧代码。腕带关系的可选 overlay 仍保留一个旧算法兼容桥；它不参与默认 direct、SAM6D 和 LLM/batch 流程。仍然作为外部环境使用的是第三方框架、模型和硬件：

- conda 环境和 Python 依赖
- cuRobo / ManiSkill / SAPIEN / RealMan SDK
- SAM-6D 项目和权重
- SAM3 权重
- 真实机械臂和相机设备

这些属于运行环境，不是本项目源码的一部分。
