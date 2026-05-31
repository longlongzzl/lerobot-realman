# GitHub Update List - Jimu Portable Build

Date: 2026-05-31

This update records the current Jimu stacking work around the portable four-wall/two-layer/roof pipeline, AprilTag anchor localization, cuRobo collision-world fixes, and the real/sim execution bridge.

## Main Changes

- Added a portable Jimu build entrypoint:
  - `rm75_jimu_four_wall_portable.py`
  - `rm75_jimu_triangle_roof_apriltag_portable.py`
- Added Jimu-specific SAM6D/AprilTag pose provider:
  - `jimu_sam6d_pose_provider.py`
  - Supports AprilTag anchor localization for tray/base assemblies.
  - Supports fixed-scene JSON replay for simulation-only debugging.
- Added portable reproducibility assets under `jimu_portable_repro/`:
  - Jimu square plate assets.
  - old/new tray assets.
  - 5-plate base assembly asset.
  - default fixed scenes.
  - portable ManiSkill/RM75 runtime subset.
- Added fixed/random scene generation and rendering helpers:
  - `generate_jimu_9object_fixed_scene.py`
  - `generate_jimu_assembly_anchor_random_scene.py`
  - `render_jimu_plate_scene_json.py`
  - `tools/fit_apriltag_tray_offset.py`
- Added cuRobo collision-world documentation:
  - `CUROBO_COLLISION_WORLD_FLOW.md`

## Planning And Execution Updates

- Integrated Jimu build logic with the existing direct pick/place pipeline.
- Added one-scene multi-cycle execution: switch active object in place instead of recreating the environment every role.
- Added first-layer, second-layer, and roof role ordering.
- Added source retry: if a tray slot cannot produce a valid grasp/place chain, try another same-family source slot.
- Added partial-open gripper behavior:
  - pregrasp uses partial open instead of full open.
  - place release keeps partial open during post-place clearance.
  - avoids hitting neighboring tray slots and placed panels.
- Added world-Z constrained short motions for:
  - pregrasp to grasp,
  - post-grasp lift,
  - final contact,
  - post-place clearance.
- Updated strict short-line validation:
  - non-contact short segment tolerance is now `8mm`.
  - final contact remains stricter.
- Added/kept return-to-start after placement, including final roof role.

## cuRobo Collision Updates

- Scene obstacles are rebuilt per cycle from the current scene cache and active role.
- Current active source is excluded from ordinary world obstacles once it becomes the picked target.
- Held object is represented as `attached_object` collision spheres during transport.
- Jimu square plates use a planar attached-sphere grid.
- Triangle roof pieces use inset vertex/edge sphere centers so the collision model stays inside the panel while still covering the boundary.
- Newly placed objects are added back as temporary `extra_scene_obstacles` before post-place clearance and return-to-start checks.
- Roof pieces can optionally use true mesh obstacles in cuRobo:
  - `--jimu-roof-curobo-mesh-obstacles`
- The RM75 planning collision config was cleaned by removing the extra camera/wrist spheres that caused false `gripper_base_link <-> link_1` self-collisions.

## Localization Updates

- Added assembly-anchor localization path:
  - tray assembly localized from AprilTag `tag0`.
  - base assembly localized from AprilTag `tag1`.
- Updated current tag sizes:
  - tray tag: `60mm`.
  - base tag: `52mm`.
- Added tray/base top-surface Z handling for anchor-derived poses.
- Added scene export helpers for sharing 14 tray slots + 5 base plates.

## Useful Commands

Simulation with fixed scene JSON:

```bash
python Beta_demo-codex-v0.9/rm75_jimu_triangle_roof_apriltag_portable.py \
  --sam6d-fixed-scene-result-file Beta_demo-codex-v0.9/jimu_portable_repro/scenes/jimu_assembly_anchors_default_sam6d.json \
  --render-mode human \
  --auto-execute
```

Live AprilTag/SAM6D localization, simulation only:

```bash
python Beta_demo-codex-v0.9/rm75_jimu_triangle_roof_apriltag_portable.py \
  --render-mode human \
  --jimu-apriltag-anchor-localization \
  --auto-execute
```

Live localization + real robot execution:

```bash
python Beta_demo-codex-v0.9/rm75_jimu_triangle_roof_apriltag_portable.py \
  --render-mode human \
  --jimu-apriltag-anchor-localization \
  --auto-execute \
  --execute-real \
  --real-control-hz 30 \
  --real-max-delta-per-step 0.1
```

## Suggested Files To Include

Core Jimu portable code:

```text
.gitignore
Beta_demo-codex-v0.9/rm75_jimu_four_wall_portable.py
Beta_demo-codex-v0.9/rm75_jimu_triangle_roof_apriltag_portable.py
Beta_demo-codex-v0.9/jimu_sam6d_pose_provider.py
Beta_demo-codex-v0.9/generate_jimu_9object_fixed_scene.py
Beta_demo-codex-v0.9/generate_jimu_assembly_anchor_random_scene.py
Beta_demo-codex-v0.9/render_jimu_plate_scene_json.py
Beta_demo-codex-v0.9/tools/fit_apriltag_tray_offset.py
Beta_demo-codex-v0.9/CUROBO_COLLISION_WORLD_FLOW.md
Beta_demo-codex-v0.9/GITHUB_UPDATE_LIST.md
Beta_demo-codex-v0.9/jimu_portable_bundle_README.md
Beta_demo-codex-v0.9/jimu_portable_repro/
```

Direct planner/runtime changes:

```text
pick_jiaobang/rm75_jiaobang_pick_place_targeted_curobo_direct_pre_place.py
pick_jiaobang/rm75_jiaobang_pick_place_targeted_curobo_direct_pre_place_sam6d.py
pick_jiaobang/rm75_jiaobang_pick_real_with_foundationpose.py
pick_jiaobang/curobo_rm75_planner.py
pick_jiaobang/object_specs.py
pick_jiaobang/sam3_mask_provider.py
pick_jiaobang/sam3_resident_worker.py
pick_jiaobang/sam6d_groundingdino_pose_provider.py
pick_jiaobang/curobo_rm75_config/rm75.yml
```

Assets that may be needed by the portable demo:

```text
FoundationPose/assets/red_jimu_cube.glb
pick_jiaobang/meshs/red_triangle.glb
```

Optional standalone comparison implementation:

```text
Demo_Triangle/
run_demo_triangle_github_local.sh
run_demo_triangle_github_v16.sh
run_demo_triangle_github_safe_one_roof.sh
run_demo_triangle_github_safe_full_roof.sh
run_demo_triangle_github_safe_full_roof_real_tray.py
run_demo_triangle_github_safe_full_roof_real_tray.sh
```

## Do Not Include

These are local generated outputs or caches:

```text
planning_profile.jsonl
planning_profile_logs/
pick_jiaobang/failure_renders/
pick_jiaobang/sam6d_groundingdino_runs/
pick_jiaobang/sam6d_grasp_scene_runs/
pick_jiaobang/sam6d_template_cache/
pick_jiaobang/sam6d_pem_feature_cache/
Beta_demo-codex-v0.9/localization_debug/
Beta_demo-codex-v0.9/sam6d_jimu_direct_runs/
Beta_demo-codex-v0.9/v6_standard_four_wall_retry_build*/
Beta_demo-codex-v0.9/*.zip
Demo_Triangle/runs/
Demo_Triangle/reports/
__pycache__/
*.pyc
```

## Validation Notes

- `py_compile` passed for the current portable Jimu entrypoints and the direct planner after the latest strict-line tolerance change.
- Cached-scene dry-run validated `right_wall -> back_wall -> left_wall` after raising non-contact short-line tolerance to `8mm`.
- Latest relevant dry-run profile:

```text
planning_profile_logs/20260531_030113_right_wall_pid49882.jsonl
```

## Suggested Commit Message

```text
Add portable Jimu stacking pipeline with AprilTag anchors
```
