# Jimu Portable Repro

This folder contains the offline repro inputs for `../rm75_jimu_four_wall_portable.py`.

Bundled files:

- `assets/red_jimu_cube.glb`: Jimu block mesh used for all roles.
- `scenes/jimu_9objects_default_sam6d.json`: fixed 9-object SAM6D-style scene. It contains `floor` plus eight wall blocks, so the runtime skips camera capture, SAM3, and SAM6D.
- `maniskill_env/`: portable ManiSkill task/robot bundle. It force-registers `Two_finger_PickJiaobang-v1` with the no-fixture PickJiaobang scene, plus the RM75 robot module and URDF assets.

Copy `rm75_jimu_four_wall_portable.py` and this whole `jimu_portable_repro/` folder together. Copying only the script will fall back to that machine's local ManiSkill task.

Run first layer in simulation:

```bash
python Beta_demo-codex-v0.9/rm75_jimu_four_wall_portable.py \
  --render-mode human \
  --jimu-build-layers first \
  --auto-execute
```

Run two layers in simulation:

```bash
python Beta_demo-codex-v0.9/rm75_jimu_four_wall_portable.py \
  --render-mode human \
  --jimu-build-layers two \
  --auto-execute
```

Run live SAM3/SAM6D instead of the bundled scene:

```bash
python Beta_demo-codex-v0.9/rm75_jimu_four_wall_portable.py \
  --render-mode human \
  --jimu-live-sam6d \
  --jimu-build-layers two
```

If this script is copied outside the lerobot tree, pass:

```bash
--lerobot-root /path/to/lerobot
```
