# Task 03: Two-Camera Joint Optimization

## Objective

Implement a higher-ceiling calibration mode that jointly uses:

- global/main camera board observations;
- wrist camera board observations across many robot poses;
- the EasyHeC/global camera extrinsic as a prior.

Variables:

```text
T_E_Cw   wrist camera relative to end effector
T_R_P    board pose in robot base
```

Optional variable if the implementation is stable:

```text
small delta on T_R_Cg with a strong prior
```

## Suggested Command

```bash
python -m rm75_app calib-joint-board -- \
  --config assets/calibration/rm75_calibration_config.example.json \
  --global-board-run-dir <global_board_anchor_run> \
  --wrist-board-run-dir <wrist_board_or_anchor_run> \
  --base-camera-run-dir <base_camera_run_dir>
```

## Cost Function

Use raw ChArUco corner reprojection errors.

Global camera terms:

```text
T_Cg_P = inv(T_R_Cg) @ T_R_P
uv_pred_g = project(K_Cg, dist_Cg, T_Cg_P, board_points_P)
```

Wrist camera terms:

```text
T_Cw_P_i = inv(T_R_E_i @ T_E_Cw) @ T_R_P
uv_pred_w_i = project(K_Cw, dist_Cw, T_Cw_P_i, board_points_P)
```

Priors:

```text
T_R_P near global-board-anchor result
T_E_Cw near wrist-anchor or classical hand-eye result
optional T_R_Cg near EasyHeC result
```

Use robust loss and clear per-term weights. Defaults should make this conservative:

```text
global board prior: strong
wrist reprojection: strong
global reprojection: medium
T_R_Cg delta: fixed by default
```

## Required Outputs

Run directory:

```text
runtime_data/calibration_runs/<timestamp>_two_camera_joint_board/
```

Write:

```text
ee_T_wrist_camera_joint.npy/json
R_T_board_joint.npy/json
delta_board_anchor_to_joint.npy/json
delta_wrist_initial_to_joint.npy/json
joint_optimization_report.json
report.html
global_overlays/
wrist_overlays/
```

Report must include:

- initial vs final reprojection RMSE for both cameras;
- per-frame wrist errors;
- global board pose delta in mm/deg;
- wrist extrinsic delta in mm/deg;
- whether the solution was accepted or rejected by quality gates.

## Quality Gates

Reject or warn if:

- final wrist reprojection gets worse;
- optimized `T_R_P` drifts too far from global anchor;
- optimized `T_E_Cw` drifts too far from wrist-anchor initial;
- only a narrow set of wrist poses is present;
- global board observations are inconsistent across frames.

Suggested defaults:

```text
max T_R_P drift: 10 mm / 2 deg
max T_E_Cw drift from initial: 30 mm / 5 deg
minimum wrist frames: 12
recommended wrist frames: 30+
```

## Acceptance Criteria

- Can run fully offline from two existing run directories.
- Produces accepted/rejected status and does not silently publish bad results.
- HTML report shows both camera overlays.
- Does not overwrite normal wrist calibration unless explicitly requested.
- `python -m py_compile` passes.

## Suggested Write Scope

- Add a new file under `rm75_app/calibration/`, for example `joint_board_optimization.py`.
- Small command registration edits in `cli.py` and `commands.py`.
- May reuse helper functions from Tasks 01/02 after they land.

