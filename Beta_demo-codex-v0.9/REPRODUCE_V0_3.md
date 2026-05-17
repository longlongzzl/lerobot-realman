# Reproduce v0.3

## 1. 复现边界

`v0.3` 不是“只拉 `Beta_demo` 脚本本体就能直接跑”的版本。

原因很简单：

- 当前运行环境里的 `mani_skill` 不是普通 site-packages 副本
- 它是 editable install
- 实际加载路径是 `D:\Project\Scaling\ManiSkill-main\ManiSkill-main`
- 其中 `RM75` 相关实现已经被本地改过

所以如果不同时处理这些外部依赖，`Beta_demo` 的当前结果不能稳定复现。

## 2. 必须同步的 ManiSkill 文件

仓库已经收录了当前运行所需的关键补丁镜像：

- `repro\maniskill_patch\mani_skill\agents\robots\realman\__init__.py`
- `repro\maniskill_patch\mani_skill\agents\robots\realman\realman_with_gripper.py`

这两个文件需要覆盖到你的 editable `ManiSkill` 源树中对应位置。

目标路径示例：

- `<MANISKILL_EDITABLE_ROOT>\mani_skill\agents\robots\realman\__init__.py`
- `<MANISKILL_EDITABLE_ROOT>\mani_skill\agents\robots\realman\realman_with_gripper.py`

## 3. 环境文件

仓库内已经保存：

- 精确环境导出：`repro\env\sim2real_curobo_v0.3_full.yml`
- 历史指令导出：`repro\env\sim2real_curobo_v0.3_from_history.yml`
- 原始 `pip freeze`：`repro\env\pip_freeze_v0.3.txt`
- 去掉本地 editable 行后的 pip 依赖：`repro\env\pip_runtime_v0.3.txt`

其中 `pip_freeze_v0.3.txt` 明确记录了当前两个本地 editable 依赖：

```text
# Editable Git install with no remote (nvidia_curobo==0.7.8.post1.dev0+dirty)
-e c:\users\administrator\appdata\local\temp\curobo-v078

# Editable install with no version control (mani_skill==3.0.0b21)
-e d:\project\scaling\maniskill-main\maniskill-main
```

因此 `full.yml` 只适合作为参考快照，不适合作为唯一重建入口。当前主推荐路线是：

- `from_history.yml` 创建 conda 基础环境
- `pip_runtime_v0.3.txt` 安装常规 pip 依赖
- 仓库内 `repro\curobo_snapshot` 作为 CuRobo 运行快照
- editable `ManiSkill` + 仓库内 `maniskill_patch`

## 4. 创建 test_env

推荐直接运行：

```powershell
powershell -ExecutionPolicy Bypass -File D:\Project\Scaling\Beta_demo\repro\setup_test_env.ps1 `
  -EnvName test_env `
  -Recreate `
  -ManiSkillEditableRoot D:\Project\Scaling\ManiSkill-main\ManiSkill-main
```

这个脚本会做五件事：

1. 用 `repro\env\sim2real_curobo_v0.3_from_history.yml` 创建 `test_env`
2. 用 conda 安装 `torch==2.5.1`、`torchvision==0.20.1`、`pytorch-cuda=12.1`
3. 用 `repro\env\pip_runtime_v0.3.txt` 以 `--no-deps` 方式安装其余 pip 依赖，避免把 `torch/numpy` 错误升级
4. 在 `test_env` 中安装 editable `mani_skill`
5. 把仓库内补丁覆盖到 editable `ManiSkill` 源树
6. 验证运行时加载位置、补丁哈希、以及仓库内 CuRobo snapshot 是否可导入

如果你只想手动做，可以按下面步骤执行。

## 5. 手动步骤

### 5.1 创建环境

```powershell
conda env create -n test_env -f D:\Project\Scaling\Beta_demo\repro\env\sim2real_curobo_v0.3_from_history.yml
```

### 5.2 安装常规 pip 依赖

先装 torch：

```powershell
conda install -n test_env pytorch=2.5.1 torchvision=0.20.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

然后安装其余 pip 依赖：

```powershell
D:\MiniConda\envs\test_env\python.exe -m pip install --no-deps -r D:\Project\Scaling\Beta_demo\repro\env\pip_runtime_v0.3.txt
```

### 5.3 安装 editable ManiSkill

```powershell
D:\MiniConda\envs\test_env\python.exe -m pip install --no-deps -e D:\Project\Scaling\ManiSkill-main\ManiSkill-main
```

### 5.4 应用补丁

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
D:\MiniConda\envs\test_env\python.exe D:\Project\Scaling\Beta_demo\repro\apply_maniskill_patch.py `
  --target-root D:\Project\Scaling\ManiSkill-main\ManiSkill-main
```

### 5.5 验证补丁和 CuRobo snapshot

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
D:\MiniConda\envs\test_env\python.exe D:\Project\Scaling\Beta_demo\repro\verify_runtime_setup.py `
  --target-root D:\Project\Scaling\ManiSkill-main\ManiSkill-main
```

## 6. 运行 v0.3 冒烟测试

默认 smoke 指令使用当前推荐四墙配置：

- `--strategy-preset pair_first_robust_fast_v1`
- `--execution-mode direct-first`
- `--initial-assembly-offset-x/y/z 0.0`

```powershell
powershell -ExecutionPolicy Bypass -File D:\Project\Scaling\Beta_demo\repro\run_v0_3_smoke.ps1 `
  -CondaEnv test_env `
  -OutDir D:\Project\Scaling\Beta_demo\repro_runs\v0_3_test_env_smoke
```

等价命令：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
D:\MiniConda\envs\test_env\python.exe D:\Project\Scaling\Beta_demo\standard_four_wall_retry_build.py `
  --out-dir D:\Project\Scaling\Beta_demo\repro_runs\v0_3_test_env_smoke `
  --no-unique-out-dir `
  --strategy-preset pair_first_robust_fast_v1 `
  --execution-mode direct-first `
  --initial-assembly-offset-x 0.0 `
  --initial-assembly-offset-y 0.0 `
  --initial-assembly-offset-z 0.0
```

## 7. 通过标准

最低通过标准：

- `verify_runtime_setup.py` 通过
- `JimuPickCube-v1` 能被成功注册
- 仓库内 `repro\curobo_snapshot\src\curobo` 能被成功导入
- `standard_four_wall_retry_build.py` 能在 `test_env` 中启动并生成输出目录

如果你要追求严格的实验复现，还需要继续检查：

- 最终 `summary.json`
- `attempts\attempt_001_direct_multi\multi_wall_live.mp4`
- 连接数和最终稳定性是否符合当前基线

## 8. 当前验证摘要

仓库里保留了小体积摘要文件：

- `repro\validation\rand10_pullback_summary_20260513.json`
- `repro\validation\rand10_pullback_extra_summary_20260513.json`
- `repro\validation\stage_jitter_summary_20260513.json`

这些文件用于说明当前版本不是只在单一 case 上通过。

另外，这次仓库内还补了一个冷启动复现摘要：

- `repro\validation\test_env_smoke_v0_3_summary_20260513.json`

它对应的是：

- 全新 `test_env`
- 使用仓库内 `curobo_snapshot`
- 应用仓库内 `maniskill_patch`
- 由 `test_env` 自己作为外层与内层规划解释器完成的一次完整四墙冒烟成功
