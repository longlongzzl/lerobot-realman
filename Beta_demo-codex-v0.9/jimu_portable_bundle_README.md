# Jimu Portable Bundle

解压位置：`lerobot` 仓库根目录。

这个包包含：

- `Beta_demo-codex-v0.9/rm75_jimu_four_wall_portable.py`
- `Beta_demo-codex-v0.9/jimu_portable_repro/`
- `pick_jiaobang/` 里的 direct 主流程、targeted place、place rules、object specs、cuRobo planner、SAM3/SAM6D provider 等核心代码
- `pick_jiaobang/curobo_rm75_config/`
- `pick_jiaobang/meshs/`
- `RM75_gripper/`
- `lerobot-sim2real/lerobot_sim2real/config/real_robot.py` 的最小真机接口副本
- `lerobot/common/robots/realman_lerobot/` 和 `lerobot/common/cameras/realsense/` 的最小真机依赖副本
- `rm75_pick_place_app/rm75_app/` 和 `rm75_pick_place_app/docs/FAST_CHAIN_IK_BATCH_FLOW.md`

默认运行会使用包里的固定 9 物体 JSON、Jimu mesh、portable ManiSkill `Two_finger_PickJiaobang-v1` 和 RM75 资产，不会调用相机/SAM3/SAM6D。

单层测试：

```bash
python Beta_demo-codex-v0.9/rm75_jimu_four_wall_portable.py \
  --render-mode human \
  --jimu-build-layers first \
  --auto-execute
```

两层测试：

```bash
python Beta_demo-codex-v0.9/rm75_jimu_four_wall_portable.py \
  --render-mode human \
  --jimu-build-layers two \
  --auto-execute
```

真机执行：

```bash
python Beta_demo-codex-v0.9/rm75_jimu_four_wall_portable.py \
  --render-mode human \
  --jimu-build-layers two \
  --auto-execute \
  --execute-real \
  --real-control-hz 30 \
  --real-max-delta-per-step 0.1
```

如果要重新拍照定位，额外加：

```bash
--jimu-live-sam6d
```

如果想禁用包里的 ManiSkill env，额外加：

```bash
--no-jimu-portable-maniskill-env
```
