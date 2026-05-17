# Jimu Assembly Planner

This package is the modular control layer above the Jimu magnetic tile physics and the CuRobo motion planner.

The physical layer stays in:

- `D:\Project\Scaling\jimu\jimu_pick_cube_env.py`
- `D:\Project\Scaling\jimu\magnetic_snap.py`

The current long-horizon robot assembly entry stays in:

- `D:\Project\Scaling\jimu\plan_second_layer_triangle_wall_path.py`

This package adds reusable orchestration pieces around that entry:

- stable physical run profiles
- perturbation matrices
- case retries
- failure classification
- video manifests
- shared trajectory scoring
- success profile caches
- full-build orchestration
- best-version registry
- graph-style assembly specs
- future candidate-level and case-level parallelism

## Current Standard

The hard magnetic connection boundary remains:

```text
attach_distance = 0.010m
```

This runner does not increase that boundary. Soft magnetic attraction remains a near-contact field effect, not a hard connection.

## Current Flow

```text
four-wall checkpoint
  -> base pose perturbation
  -> staged triangle fixtures
  -> role sequence
  -> CuRobo motion planning inside plan_second_layer_triangle_wall_path.py
  -> joint qpos step execution
  -> magnetic capture check
  -> 50-step final stability check
  -> summary/video/checkpoint artifacts
```

## Failure Classes

The runner normalizes raw failure strings into:

```text
runtime_error
scene_state_or_checkpoint
curobo_planning
collision_or_retreat
magnetic_capture
post_release_stability
unknown
```

This is intentionally coarse. The next optimization stage should use these classes to select local retry actions before restarting the full case.

## Parallelism Policy

`--parallel-workers` enables case-level parallelism for offline robustness matrices. Keep it low, normally `1` to `3`, because each worker owns a SAPIEN simulation and may compete for GPU resources.

Candidate-level CuRobo batching is the next intended optimization, but it should be added after the failure reports are stable.

## Standard Entrypoints

Run the second-layer robustness matrix:

```powershell
D:\MiniConda\envs\sim2real-curobo\python.exe D:\Project\Scaling\jimu\run_v7_attach10mm_robustness_matrix.py `
  --out-root D:\Project\Scaling\jimu\v9_modular_matrix_full `
  --case-retries 2 `
  --parallel-workers 1
```

Run the full physical build from an existing four-wall state:

```powershell
D:\MiniConda\envs\sim2real-curobo\python.exe D:\Project\Scaling\jimu\run_v9_full_physical_build.py `
  --out-dir D:\Project\Scaling\jimu\v9_full_physical_build_from_checkpoint
```

Run the full physical build from scratch:

```powershell
D:\MiniConda\envs\sim2real-curobo\python.exe D:\Project\Scaling\jimu\run_v9_full_physical_build.py `
  --out-dir D:\Project\Scaling\jimu\v9_full_physical_build_from_scratch `
  --from-scratch
```

All standard v9 entrypoints keep hard magnetic attachment at `0.010m`.

Record a confirmed best version:

```powershell
D:\MiniConda\envs\sim2real-curobo\python.exe D:\Project\Scaling\jimu\record_best_version.py `
  --name v9_best_candidate `
  --source-run D:\Project\Scaling\jimu\v9_full_physical_build_from_checkpoint `
  --note "Four walls plus vertical triangle layer, hard attach boundary 10mm."
```

## Trajectory Score

`trajectory_score.py` defines the shared fitness shape for future candidate selection:

```text
CuRobo success
robot/gripper collision
held object displacement
release edge error
release center error
edge and normal angle error
active magnetic connection
post-release stability
built structure motion
joint path distance
execution steps
```

This gives the planner a general quality function instead of role-specific placement hacks.

## Assembly Spec

`assembly_spec.py` defines a small graph-style structure format:

```text
PanelSpec(role, shape, stage_group)
ConnectionSpec(child, child_edge, parent, parent_edge, relation, order)
AssemblySpec(name, panels, connections)
```

The current standard task is `standard_two_layer_wall_spec()`. More complex scenes, such as castles or multi-floor buildings, should add assembly specs first and then reuse the same planning, retry, reporting, and versioning layers.
