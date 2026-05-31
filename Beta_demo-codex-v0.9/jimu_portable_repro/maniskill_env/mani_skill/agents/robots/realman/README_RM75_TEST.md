# RM75机械臂测试说明

## 文件说明

- `test_rm75_simple.py` - 简单快速测试脚本（推荐先用这个）
- `test_rm75.py` - 完整测试脚本（包含多种测试场景）

## 运行测试

### 方法1：简单测试（推荐）

```bash
# 在ManiSkill根目录下运行
python -m mani_skill.agents.robots.realman.test_rm75_simple
```

或者直接运行：

```bash
cd ManiSkill-main/ManiSkill-main/mani_skill/agents/robots/realman
python test_rm75_simple.py
```

### 方法2：完整测试

```bash
python -m mani_skill.agents.robots.realman.test_rm75
```

## 测试内容

### test_rm75_simple.py
- 创建PickCube环境并加载RM75机械臂
- 执行100步随机动作
- 显示GUI界面（如果设置了render_mode="human"）

### test_rm75.py
包含三个测试场景：
1. **GUI模式测试** - 带图形界面的基本功能测试
2. **无头模式测试** - 不需要GUI的测试
3. **并行环境测试** - GPU并行模拟测试

## 注意事项

1. **确保URDF文件存在**：
   - 路径：`~/.maniskill/data/robots/RM75_gripper/RM75-B/urdf/RM75-B.urdf`
   - 如果不存在，需要先下载或放置URDF文件

2. **GUI渲染**：
   - 如果设置了`render_mode="human"`，需要图形界面
   - 在服务器上运行可以注释掉`render_mode="human"`或使用无头模式

3. **控制模式**：
   - 当前使用`pd_joint_delta_pos`（关节位置增量控制）
   - 也可以尝试其他控制模式，如`pd_ee_delta_pose`（末端执行器位姿控制）

4. **观察模式**：
   - 当前使用`state`模式（状态观察）
   - 也可以尝试`rgbd`（RGB-D图像）或`pointcloud`（点云）

## 常见问题

### 问题1：找不到RM75机械臂
**错误**：`ValueError: Robot uid 'RM75' not found`

**解决**：
- 确保`realman_with_gripper.py`中的类已正确注册
- 检查`uid = "RM75"`是否正确设置
- 确保`__init__.py`中已导入相关类

### 问题2：找不到URDF文件
**错误**：`FileNotFoundError: .../RM75-B.urdf`

**解决**：
- 检查URDF文件路径是否正确
- 确保文件存在于：`~/.maniskill/data/robots/RM75_gripper/RM75-B/urdf/RM75-B.urdf`

### 问题3：关节名称不匹配
**错误**：`KeyError: 'joint_1'` 或类似错误

**解决**：
- 检查URDF文件中的关节名称是否与代码中的`arm_joint_names`匹配
- 确保所有gripper关节名称已正确更新

## 修改测试参数

如果需要测试不同的配置，可以修改测试文件中的参数：

```python
env = gym.make(
    "PickCube-v1",
    robot_uids="RM75",
    num_envs=1,  # 并行环境数量
    obs_mode="state",  # 观察模式：state, rgbd, pointcloud等
    control_mode="pd_joint_delta_pos",  # 控制模式
    render_mode="human",  # 渲染模式：human, None等
)
```

## 参考文档

- [ManiSkill快速开始](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html)
- [ManiSkill任务列表](https://maniskill.readthedocs.io/en/latest/user_guide/tasks/index.html)
- [ManiSkill机器人](https://maniskill.readthedocs.io/en/latest/user_guide/robots/index.html)
