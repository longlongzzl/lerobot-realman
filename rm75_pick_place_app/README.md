# RM75 Pick-Place App

这个目录现在是独立运行工程：pick-place 的 direct/SAM6D 主链、核心源码、mesh、cuRobo 配置、测试场景、机器人 URDF、相机标定副本和运行输出都在本目录内，不再从外层旧工程目录加载代码或资产。

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
    perception/    # SAM3 mask provider、SAM6D pose provider、resident worker
    planning/      # cuRobo planner、full-chain helper、fast-chain notes
    execution/     # 真机/仿真桥接执行脚本
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
       仿真 / 真机 / 旧脚本兼容后端
```

- `core/` 只定义任务请求、阶段、计划、命令和结果，不导入 NumPy、ManiSkill、cuRobo 或硬件 SDK。
- `tasks/` 只描述任务差异。抓取放置、Jimu 装配、Lego snap 都通过同一套阶段契约接入。
- `perception/`、`planning/`、`execution/` 是可替换能力层；任务层不应该直接导入旧的超大运行脚本。
- `pickplace/` 是 pick-place 的主线编排边界；`runtime/` 保留具体的 direct/SAM6D 运行实现，避免 CLI、任务定义和重型依赖互相混在一起。
- `direct`、`sam6d`、`wrist` 已经通过 `pickplace` 适配器和 runner 选择后端；direct/SAM6D 的运行源码不再从外层 `pick_jiaobang` 加载。
- Jimu 和 Lego 现在是明确标记为 `compatibility` 的适配器：统一入口已经在主线，具体旧执行器还没有搬进来，避免把旧耦合伪装成完成迁移。

查看任务注册表和编译后的兼容命令：

```bash
python -m rm75_app tasks list
python -m rm75_app tasks info pickplace
python -m rm75_app tasks info jimu
python -m rm75_app tasks command pickplace --mode sam6d -- --help
python -m rm75_app tasks command jimu --mode four-wall
python -m rm75_app tasks command lego --mode dry-run
python -m rm75_app pickplace --mode direct -- --help
```

完整的边界和迁移顺序见 `docs/TASK_ARCHITECTURE.md`。

## 运行方式

```bash
cd /home/zhangzhao/Desktop/lerobot/rm75_pick_place_app
python -m rm75_app --help
python -m rm75_app pickplace --mode direct -- --help
python -m rm75_app direct -- --help
python -m rm75_app sam6d -- --help
python -m rm75_app tabletop-refine -- --help
python -m rm75_app roof -- --help
python -m rm75_app web -- --host 127.0.0.1 --port 7860
python -m rm75_app llm -- --help
python -m rm75_app print-command sam6d --execute-real
python -m rm75_app verify
```

常用真机 SAM6D 命令可直接生成：

```bash
python -m rm75_app print-command sam6d --execute-real
```

## 桌面物体位姿精修

SAM6D 全局定位之后，可以单独运行桌面物体 `x/y/yaw` 精修入口：

```bash
python -m rm75_app tabletop-refine -- --sam6d-result <full_scene_pose_results.json>
```

这个入口不会重新跑 SAM3/SAM6D，也不会影响原始 SAM6D 输出。它读取 SAM6D 保存的 `shared_frame/rgb.png`、`depth.png`、`camera.json` 和每个物体的 `sam6d_binary_mask.png`，把 SAM6D 结果当粗定位，只在机器人基座/桌面平面内搜索 `x`、`y` 和 `yaw`，锁定 `z/roll/pitch`，用 CAD 投影 mask IoU、bbox/中心误差和 depth residual 打分。输出写到：

```text
runtime_data/tabletop_pose_refine_runs/<timestamp>_tabletop_refine/tabletop_refined_scene_results.json
```

输出仍保留 `results[].T_cam_obj` 字段，因此可以作为新的固定场景结果继续传给 SAM6D 抓取入口：

```bash
python -m rm75_app sam6d -- --sam6d-fixed-scene-result-file <tabletop_refined_scene_results.json>
```

常用调参：

```bash
python -m rm75_app tabletop-refine -- --sam6d-result <full_scene_pose_results.json> --object-names carriot gluestick lvmukuai --xy-search-radius-m 0.05 --yaw-search-radius-deg 45
```

如果某些物体不是稳定桌面状态，或需要 full 6D，不要放进这个精修入口；它默认只解决桌面物体的平面误差。

## 腕带相机抓取后精修

腕带相机的持物关系估计适配层在 `rm75_app/perception/wrist_relation.py`，新入口是：

```bash
python -m rm75_app wrist -- --execute-real
```

它等价于 direct pick-place，但默认打开 `--wrist-relation-refine-after-grasp`。抓爪闭合后会异步运行腕带 RGB-D 定位；进入 targeted place/pre_place 规划前，如果结果可用，就更新当前 `T_tcp_obj` 并重新生成放置候选。旧的 `direct`、`sam6d`、LLM/batch 入口默认不启用这个 hook。

## 盖亭子入口

亭子底座加四根柱子的独立入口是：

```bash
python -m rm75_app roof -- --execute-real
# 等价别名
python -m rm75_app tingzi -- --execute-real
```

它复用 direct pick-place 主流程，但默认按顺序执行 `tingzi_base`、`tingzi_pillar_front_left/right`、`tingzi_pillar_back_left/right`。第一步把底座放到桌面中心附近；后四步共用 `tingzi_pillar1.obj`，把四根柱子按底座本地坐标的四个角插入。入口默认启用腕带相机抓取后 `T_tcp_obj` 精修；原来的 `direct`、`sam6d` 和 batch/LLM 入口不会自动启用这套亭子参数。

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
