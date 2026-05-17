# Jimu v9 Building Blocks Baseline - Agent Handoff

This file is a handoff note for a future Codex/agent session. It summarizes the
current repository state, the known successful baseline, the intended workflow,
and the constraints that should be preserved.

## Current Repository

- GitHub repository: `https://github.com/yeshenpy/Building_Blocks_System`
- Current branch: `codex/v9-jimu-second-layer-baseline`
- Local checkout: `D:\Project\Scaling\Building_Blocks_System`
- Source snapshot copied from:
  `D:\Project\Scaling\jimu\v9_best_force8_order_retry_v1\code`
- Reference success run:
  `D:\Project\Scaling\jimu\v9_repro_62`
- Reference success video:
  `D:\Project\Scaling\jimu\v9_repro_62\second_layer_triangles\triangle_wall_live.mp4`

This repository is intentionally compact. It contains the best known Jimu v9
code baseline for building four first-layer square walls and then placing four
vertical triangle panels as second-layer walls. It does not contain historical
probe folders, videos, large logs, caches, or run artifacts.

## Goal Of This Codebase

The current baseline simulates magnetic building-block assembly with an RM75
robot and CuRobo planning. The successful target structure is:

1. A first layer made from square panels:
   - `floor`
   - `right_wall`
   - `back_wall`
   - `left_wall`
   - `front_wall`
2. A second layer made from four vertical triangle panels:
   - `front_second_triangle`
   - `right_second_triangle`
   - `left_second_triangle`
   - `back_second_triangle`

The main future objective is robustness: the second-layer triangle walls should
still be buildable when the first-layer square wall structure has moderate
position and yaw perturbation.

## Most Important Files

- `jimu_pick_cube_env.py`
  - Creates the Jimu simulation environment, panels, triangles, robot, table,
    collision walls, and rendering setup.
- `magnetic_snap.py`
  - Implements magnetic edge attraction, active connection creation, detach
    logic, suspended roles, and edge alignment rules.
- `curobo_rm75_planner.py`
  - CuRobo wrapper for RM75 IK and motion planning.
- `plan_multi_wall_path_200step.py`
  - Planner/executor for first-layer square wall placement.
- `standard_four_wall_retry_build.py`
  - Checkpointed first-layer four-wall builder with role-generic retry logic.
- `plan_second_layer_triangle_wall_path.py`
  - Main second-layer triangle-wall placement planner.
- `run_v9_full_physical_build.py`
  - Wrapper for the full first-layer plus second-layer build.
- `run_v7_attach10mm_robustness_matrix.py`
  - Matrix runner for first-layer perturbation cases followed by second-layer
    triangle placement.
- `assembly_planner/full_build.py`
  - Orchestrates full build stages and triangle retry orders.
- `assembly_planner/second_layer_matrix.py`
  - Defines perturbation cases, physical environment overrides, commands,
    result extraction, video manifest generation, and matrix execution.

## Known Successful Baseline

Best code baseline:

- `D:\Project\Scaling\jimu\v9_best_force8_order_retry_v1`

Best full evidence run:

- `D:\Project\Scaling\jimu\v9_repro_62`

The v9 best-force baseline is preferred over earlier v6/v7 versions because:

- It preserves a strict 10 mm active attach boundary.
- It uses stronger global established-edge holding force.
- It includes multi-order retry support.
- It completed all four second-layer triangle walls.
- Final stability passed.
- No wrong active connections were present in the successful reference.

Earlier versions mean different things:

- `v6` mainly solved first-layer four-square-wall building and checkpointed
  retry. Some v6 runs used looser magnetic distances such as `attach=0.026 m`,
  so they are not the best strict-physics second-layer baseline.
- `v7` introduced attach-10mm second-layer perturbation matrix testing, but the
  inspected matrix success rates were low, for example 1/4 or 2/8 in older
  matrix folders.
- `v9_best_force8_order_retry_v1` is the current best code baseline for the
  second-layer triangle-wall task.

## Physical Constraints To Preserve

Do not casually loosen these just to pass one scene:

- `magnet_attach_distance = 0.010 m`
- `magnet_attract_distance = 0.024 m`
- `magnet_detach_distance = 0.015 m`

Important rule: the active connection boundary should remain 1 cm. If stability
needs improvement, prefer global physical parameter changes such as established
edge force, damping, friction, or generic candidate search improvements. Do not
add role-specific magnet hacks.

Known v9/v10 success notes:

- The successful change was not expanding attach distance.
- Improvements came from stronger pre-contact attraction, contact friction,
  release gating, watchdog validation, and better global placement order.
- The best-force snapshot notes a global established-edge drive force increase
  from 5 N to 8 N and angular force from 0.35 to 0.55.

## Recommended Commands

Use the CuRobo conda Python:

```powershell
D:\MiniConda\envs\sim2real-curobo\python.exe
```

Run full build:

```powershell
D:\MiniConda\envs\sim2real-curobo\python.exe run_v9_full_physical_build.py `
  --out-dir D:\Project\Scaling\jimu\github_v9_full_build
```

Run second-layer robustness matrix:

```powershell
D:\MiniConda\envs\sim2real-curobo\python.exe run_v7_attach10mm_robustness_matrix.py `
  --out-root D:\Project\Scaling\jimu\github_v9_second_layer_matrix
```

Run second-layer planner directly from a stable first-layer checkpoint:

```powershell
D:\MiniConda\envs\sim2real-curobo\python.exe plan_second_layer_triangle_wall_path.py `
  --out-dir D:\Project\Scaling\jimu\github_v9_second_layer_direct `
  --load-assembly-state D:\Project\Scaling\jimu\v6_standard_four_wall_retry_build_v1\checkpoints\stable_04_front_wall_state.json `
  --roles front_second_triangle,left_second_triangle,right_second_triangle,back_second_triangle `
  --record-live
```

Example direct perturbation flags:

```powershell
--loaded-state-perturb-dx 0.020 `
--loaded-state-perturb-dy 0.000 `
--loaded-state-perturb-yaw-deg 5.0
```

## Current Recommended Development Direction

The next work should not start with triangle initial staging randomization.
The user clarified that the immediate priority is:

1. Start from a stable first-layer four-wall checkpoint.
2. Apply position/yaw perturbation to the completed first layer.
3. Build the four second-layer vertical triangle walls.
4. Save success and failure videos.
5. Only after this is robust should triangle staging randomization be revisited.

Suggested validation order:

1. Nominal reproduction:
   - `dx=0`
   - `dy=0`
   - `yaw=0`
2. Moderate fixed perturbations:
   - `dx=+0.020, dy=0.000, yaw=+5 deg`
   - `dx=-0.020, dy=0.000, yaw=-5 deg`
   - `dx=0.000, dy=+0.020, yaw=+5 deg`
   - `dx=0.000, dy=-0.020, yaw=-5 deg`
   - `dx=+0.030, dy=-0.020, yaw=+8 deg`
3. If those pass, run random moderate perturbations:
   - `dx, dy` in `[-0.030 m, +0.030 m]`
   - `yaw` in `[-8 deg, +8 deg]`

Acceptance criteria for a successful case:

- All four second-layer triangle roles complete.
- Intended parent-top-edge to triangle-base-edge connections are active.
- No wrong active connections.
- Final 50-step stability passes.
- Completed first-layer structure is not dragged, opened, or destabilized.
- Final triangle base edge error stays within the configured strict threshold.

## Important Design Rules

- Keep changes generic. Do not add `if role == ...` fixes for a single wall.
- Role-specific geometry is allowed only when derived from the assembly spec:
  target edge, target normal, parent role, or target pose.
- Do not enlarge attach distance beyond 1 cm unless the user explicitly changes
  the physical model requirement.
- If a placed part is disturbed later, classify and log which phase caused it.
- Keep success and failure videos for important runs.
- Do not commit videos or heavy artifacts to this repository unless explicitly
  requested.
- Prefer checkpoint retry at the current block rather than restarting from
  scratch for every failure.
- If a long path causes the robot to whip the held object, increase execution
  steps or tighten max joint step/path filters before changing physics.

## Known Useful Artifacts Outside This Repo

These are local reference artifacts and are not committed here:

- Best source snapshot:
  `D:\Project\Scaling\jimu\v9_best_force8_order_retry_v1`
- Complete success run:
  `D:\Project\Scaling\jimu\v9_repro_62`
- Success documentation:
  `D:\Project\Scaling\jimu\SUCCESS_V10_V62.md`
- First-layer stable checkpoint often used by second-layer tests:
  `D:\Project\Scaling\jimu\v6_standard_four_wall_retry_build_v1\checkpoints\stable_04_front_wall_state.json`

## Minimal Repo Hygiene

This repository should stay compact. Keep excluding:

- `run_artifacts/`
- videos such as `*.mp4`
- logs
- `__pycache__/`
- `.npz` / `.npy`
- one-off probe directories

When adding new reusable scripts, keep them in the root or `assembly_planner/`
unless there is a strong reason to introduce a new package.

