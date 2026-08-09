# Task Architecture

## 目标

`rm75_pick_place_app` 是 RM75 任务的主线工程。任务差异放在 `rm75_app/tasks/`，pick-place 的主线编排放在 `rm75_app/pickplace/`，感知、规划和执行能力放在对应能力层。

当前注册的任务：

| 任务 key | 任务类型 | 状态 | 入口模式 |
| --- | --- | --- | --- |
| `pickplace` | 通用抓取放置 | active/mainline | `curobo2`, `rrtrack`, `openworld-geometry`, `tabletop-refine` |
| `jimu` | 磁吸积木装配 | compatibility | `four-wall`, `triangle-roof` |
| `lego` | 固定连接头 Lego snap | compatibility | `real`, `dry-run` |

## 层级边界

### `core`

只放稳定的数据契约：

- `TaskRequest`：任务名、模式、任务文件和透传参数；
- `TaskDefinition`：能力、阶段、后端状态和别名；
- `TaskPlan` / `TaskStep`：统一阶段计划；
- `CommandSpec`：不依赖 shell 的兼容启动命令；
- `PipelineContext` / `StageResult`：未来真机、仿真和离线执行共用的运行状态。

这一层不能导入相机、机器人、仿真器、cuRobo 或大模型。

### `tasks`

任务适配器只做三件事：

1. 校验任务输入和任务专属模式；
2. 把任务差异转换为统一 `TaskPlan`；
3. 选择一个能力后端或兼容入口。

`PickPlaceTask` 不包含 SAM6D 实现，`JimuTask` 不包含四墙轨迹细节，`LegoTask` 不包含 cuRobo 规划细节。这样新的任务只需新增适配器，不要复制整个 `rm75_app`。

### `pickplace`

这是当前已经收敛的 pick-place 主线：

- `config.py`：默认对象、跟踪障碍物和安全运行参数；
- `backends.py`：`curobo2/rrtrack/openworld-geometry/tabletop-refine` 到 in-app runtime 的唯一映射；
- `layers.py`：任务、感知、放置、规划、执行和编排的责任边界；
- `runner.py`：统一的本地模块启动器。

旧的 direct/SAM6D 单体执行链已从 1.0.24 工作树删除；外部 FoundationPose、SAM-6D、Curobo2 和 ManiSkill 仍属于允许的第三方运行环境。

### 能力层

- `perception/`：场景、目标、障碍物和持物关系；
- `placement/`：放置语义和候选生成策略；
- `planning/`：批量 IK、候选配对和 Curobo2 轨迹；
- `execution/`：仿真或 RM75 真机执行；
- `runtime/`：感知、规划和回放的薄入口。

能力层通过 `core/interfaces.py` 暴露协议，不把任务类型硬编码进通用 planner/executor。

## 迁移顺序

1. `pickplace` 已完成 Curobo2 主线入口收敛；`direct` 仅作为 Curobo2 别名，SAM6D 为独立感知入口。
2. 把 Jimu 的对象角色、装配状态和四墙/三角屋顶步骤整理成任务 JSON/状态模型；旧 portable 脚本变成后端。
3. 把 Lego 的 grid task、snap action 和固定连接头状态迁入主线；复用主线 perception/planning/execution，不再复制整套 app。
4. 腕带硬件到位后，实现 `HeldObjectFrameSource`，不再恢复旧 wrist/direct 单体流程。
5. Jimu/Lego 验证完成后，再清理 `legacy/backends.py` 中对应的外部路径。

## 当前刻意保留的兼容边界

Jimu 和 Lego 的 `tasks command` 会输出旧后端命令，并明确标记 `compatibility`。这让旧代码继续可运行，同时避免任务层继续直接 `import` pick-place 巨型脚本。真正迁移执行器时，保持任务 key、模式和阶段契约不变，替换 `command()` 和能力绑定即可。
