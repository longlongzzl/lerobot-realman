# 重构交付清单

## 现有能力模块

## 当前主线分层

| 层级 | 位置 | 责任 |
| --- | --- | --- |
| Core contracts | `rm75_app/core/` | 任务请求、阶段、计划、命令和结果，不依赖硬件/仿真 |
| Task adapters | `rm75_app/tasks/` | `pickplace`、`jimu`、`lego` 的任务差异和注册表 |
| Pipeline boundary | `rm75_app/pipeline/` | 任务校验、计划编译、后端选择 |
| Pick-place mainline | `rm75_app/pickplace/` | 默认参数、模式后端、层级绑定和 runner |
| Compatibility boundary | `rm75_app/legacy/` | Jimu/Lego 尚未迁移的旧后端路径 |

`direct`、`sam6d`、`wrist` CLI 已通过 `pickplace` 任务适配器和 runner 选择 app 内 runtime；Jimu 和 Lego 先以兼容适配器接入，具体执行器分阶段迁移。详细边界见 `docs/TASK_ARCHITECTURE.md`。

| 功能 | 新位置 |
| --- | --- |
| 统一 CLI | `rm75_app/cli.py` |
| 路径与运行目录 | `rm75_app/paths.py` |
| 默认对象/参数 | `rm75_app/config/defaults.py` |
| 命令构造 | `rm75_app/commands.py` |
| Direct pick-place 主流程 | `rm75_app/runtime/direct_pre_place.py` |
| SAM6D pick-place 主流程 | `rm75_app/runtime/sam6d_pick_place.py` |
| SAM6D 后处理桌面位姿精修 | `rm75_app/runtime/tabletop_pose_refine.py`, `rm75_app/entrypoints/tabletop_pose_refine.py` |
| FoundationPose 场景/仿真桥 | `rm75_app/runtime/foundationpose_scene.py` |
| Targeted place 规则流程 | `rm75_app/runtime/targeted_place.py` |
| cuRobo targeted wrapper | `rm75_app/runtime/targeted_curobo.py` |
| cuRobo planner | `rm75_app/planning/curobo_planner.py` |
| cuRobo full-chain helper | `rm75_app/planning/curobo_fullchain.py` |
| PyRoki bridge | `rm75_app/planning/pyroki_trajopt_bridge.py` |
| 对象规格 | `rm75_app/assets/object_specs.py` |
| 放置规则 | `rm75_app/placement/place_rules.py` |
| SAM3 provider/resident | `rm75_app/perception/sam3_mask_provider.py`, `rm75_app/perception/sam3_resident_worker.py` |
| SAM6D provider/resident | `rm75_app/perception/sam6d_pose_provider.py`, `rm75_app/perception/sam6d_resident_worker.py` |
| 腕带相机持物关系精修 | `rm75_app/perception/wrist_relation.py`, `rm75_app/perception/wrist_gripper_object_relation.py`, `rm75_app/runtime/wrist_refined_pick_place.py` |
| 盖亭子装配入口 | `rm75_app/runtime/roof_assembly_pick_place.py`, `rm75_app/assets/object_specs.py`, `rm75_app/placement/place_rules.py` |
| 真机/仿真执行桥 | `rm75_app/execution/rm75_bridge.py`, `rm75_app/execution/pick_move_v10.py` |
| Web 控制台 | `rm75_app/web/control_panel.py` |
| LLM orchestrator | `rm75_app/llm/orchestrator.py` |

## 本地资产

| 内容 | 新位置 |
| --- | --- |
| mesh 资产 | `assets/meshs/` |
| cuRobo 配置 | `assets/curobo_rm75_config/` |
| 测试场景 | `assets/test_scenes/` |
| RM75 URDF 副本 | `assets/robot_models/` |
| 相机标定副本 | `assets/calibration/` |
| 运行输出 | `runtime_data/` |

## 验证命令

```bash
cd /home/zhangzhao/Desktop/lerobot/rm75_pick_place_app
python -m compileall -q rm75_app
python -m rm75_app --help
python -m rm75_app tasks list
python -m rm75_app tasks info jimu
python -m rm75_app tasks command pickplace --mode sam6d -- --help
python -m rm75_app pickplace --mode direct -- --help
python -m rm75_app print-command direct
python -m rm75_app print-command roof --execute-real
python -m rm75_app print-command sam6d --execute-real
python -m rm75_app tabletop-refine -- --help
python -m rm75_app llm -- --fixed-scene-pose-file assets/test_scenes/current_table.json --command "把网球放进笔筒" --llm-provider mock
python -m rm75_app verify
```
