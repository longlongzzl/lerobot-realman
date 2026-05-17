# Beta_demo v0.6

This repository snapshot records the current Jimu first-layer four-wall build planner.

The v0.6 default is the best tested order and release behavior from the current local validation:

- fixed wall order: `right_wall,back_wall,left_wall,front_wall`
- order policy: `given`
- strategy preset: `pair_first_robust_fast_v1`
- execution mode: `direct-first`
- grasp-release binding: enabled with cached pair release
- release preplace branch gate: `release_screen_max_preplace_joint_delta=3.0`
- positive-`x` speed guard: skips the previously failing first `right_wall` fast-screen branch and starts that role from the robust retry budget
- release behavior: partially open, unlock, then fully open before retreat for first-layer walls

## Best Command

Run this from PowerShell on the local Windows machine:

```powershell
D:\MiniConda\envs\sim2real-curobo\python.exe D:\v0.6\standard_four_wall_retry_build.py `
  --out-dir D:\v0.6\repro_runs\v0_6_best_no_offset `
  --no-unique-out-dir `
  --strategy-preset pair_first_robust_fast_v1 `
  --execution-mode direct-first `
  --roles right_wall,back_wall,left_wall,front_wall `
  --role-order-policy given `
  --use-cached-pair-release `
  --initial-assembly-offset-x 0.0 `
  --initial-assembly-offset-y 0.0 `
  --initial-assembly-offset-z 0.0
```

This command uses the current best default behavior. The important pieces are:

- `--strategy-preset pair_first_robust_fast_v1`
- `--execution-mode direct-first`
- `--roles right_wall,back_wall,left_wall,front_wall`
- `--role-order-policy given`
- `--use-cached-pair-release`
- `--release-screen-max-preplace-joint-delta 3.0` is included in the `pair_first_robust_fast_v1` preset
- zero initial floor offset for the no-offset reference run

The script records live videos by default through the wrapper command path.

## Current Validation Summary

The fixed order `right_wall -> back_wall -> left_wall -> front_wall` was the best of the tested order policies on the same eight random floor offsets in `x/y` within `+-0.08m`:

- fixed `right,back,left,front`: `6/8`, mean `91.7s`, total `733.9s`
- adaptive / connection-style order: `6/8`, mean `129.7s`, total `1037.5s`
- far-from-robot order: `4/8`, mean `160.7s`, total `1285.4s`

The current best 8-case summary is stored locally at:

```text
D:\v0.6\repro_runs\v0_6_xy8cm_8case_fixed_rblf_strict\xy8cm_8case_fixed_rblf_strict_summary.json
```

The no-offset validation run for the fixed default is stored locally at:

```text
D:\v0.6\repro_runs\v0_6_fixed_rblf_default_no_offset
```

After enabling cached pair release, the two previously failing fixed-order cases were re-tested directly and both passed:

- case 2 offset `(x=+0.026603, y=+0.025583)`: passed at `D:\v0.6\repro_runs\v0_6_pair_bound_case02`
- case 8 offset `(x=-0.013580, y=+0.072477)`: passed at `D:\v0.6\repro_runs\v0_6_pair_bound_case08`

The pure cached-pair setting was faster but failed three positive-`x` cases because it allowed a large `q_lift -> q_preplace` joint branch jump. After adding the release preplace branch gate, the full same-seed 8-case sweep passed:

```text
D:\v0.6\repro_runs\v0_6_pair_bound_preplace_gate3_xy8cm_8case\xy8cm_8case_pair_bound_preplace_gate3_summary.json
```

Result: `8/8`, mean `92.5s`, total `740.4s`.

The latest speed-optimized preset keeps that robustness and skips the stable-failing positive-`x` first `right_wall` fast-screen branch. On the same same-seed 8-case sweep:

```text
D:\v0.6\repro_runs\v0_6_speed_opt_positive_x_skip_fast_screen_xy8cm_8case\xy8cm_8case_speed_opt_summary.json
```

Result: `8/8`, mean `71.9s`, total `575.2s`, min `45.3s`, max `96.4s`.

Two additional random `+-0.08m` floor-offset sweeps passed with the same best preset:

- seed `714`: `10/10`, mean `69.8s`
- seed `715`: `10/10`, mean `81.9s`

Combined speed-optimized floor-offset validation: `28/28`, mean `74.7s`.

## Initial Actor Jitter Test

The planner can also test randomly perturbed initial positions for the loose wall actors while keeping the final target assembly fixed. This is disabled by default. Example command:

```powershell
D:\MiniConda\envs\sim2real-curobo\python.exe D:\v0.6\standard_four_wall_retry_build.py `
  --out-dir D:\v0.6\repro_runs\v0_6_best_actor_jitter_example `
  --no-unique-out-dir `
  --strategy-preset pair_first_robust_fast_v1 `
  --execution-mode direct-first `
  --roles right_wall,back_wall,left_wall,front_wall `
  --role-order-policy given `
  --use-cached-pair-release `
  --initial-actor-jitter-xy 0.03 `
  --initial-actor-jitter-seed 901 `
  --initial-actor-jitter-roles right_wall,back_wall,left_wall,front_wall `
  --initial-actor-jitter-min-start-distance 0.055 `
  --initial-actor-jitter-min-target-distance 0.035
```

The jitter samples each listed actor within a `3cm` XY radius and rejects samples that put loose pieces too close to each other or too close to other target centers. A 3-case smoke sweep passed:

```text
D:\v0.6\repro_runs\v0_6_best_actor_jitter_3cm_3case_seed901\actor_jitter_3cm_3case_summary.json
```

Result: `3/3`, mean `80.7s`.

## Environment

The validated local Python interpreter is:

```text
D:\MiniConda\envs\sim2real-curobo\python.exe
```

CUDA was available in the validated environment with `torch 2.5.1+cu121`.

`repro\verify_runtime_setup.py` may report a ManiSkill editable patch hash mismatch in this local setup. The current smoke and planner runs were still able to execute; treat that mismatch as external dependency drift unless a runtime command itself fails.

## Important Files

- `standard_four_wall_retry_build.py`: top-level retry wrapper and default v0.6 strategy.
- `plan_multi_wall_path_200step.py`: per-role CuRobo path planning and release execution.
- `repro\curobo_snapshot`: CuRobo runtime snapshot used by this repo.
- `repro\maniskill_patch`: mirrored ManiSkill RealMan patch files from the original reproducibility setup.

## What Changed In v0.6

- The default first-layer order is fixed to `right_wall,back_wall,left_wall,front_wall`.
- `pair_first_robust_fast_v1` keeps the provided role order through `--role-order-policy given`.
- The lower-level planner also respects `--role-order-policy given`, so it does not re-sort the wall sequence internally.
- `pair_first_robust_fast_v1` enables cached pair release, so a grasp candidate that passed release screening reuses the same grasp-release pair instead of picking a different release after the object is already lifted.
- `pair_first_robust_fast_v1` rejects cached grasp-release pairs with `q_lift -> q_preplace` joint distance above `3.0`, preventing the large preplace branch jumps that visually appear as the arm rotating in place after grasping.
- For positive `initial_assembly_offset_x`, `pair_first_robust_fast_v1` now bypasses the direct-first attempt and starts the initial `right_wall` from the robust retry schedule. This avoids a repeatable failed fast-screen branch that cost about `40-50s` on the tested positive-`x` offsets.
- Optional initial actor jitter was added for loose wall actors. It perturbs the initial actor poses only, leaves final targets fixed, and enforces minimum spacing before the build starts.
- First-layer wall release now fully opens the gripper before retreat when using the partial release-open value. This avoids the earlier issue where a released object could still appear carried upward during retreat.

## Known Remaining Failure Modes

No failure remains in the current same-seed eight-case `+-8cm` sweep or in the two additional 10-case floor-offset sweeps. The main remaining risk is coverage: the initial actor jitter test has only been smoke-tested on three cases, so larger jitter sweeps may still expose grasp reachability or clutter-spacing edge cases.
