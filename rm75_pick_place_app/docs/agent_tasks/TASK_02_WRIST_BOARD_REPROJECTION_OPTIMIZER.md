# Task 02: Wrist Board Reprojection Optimizer

## Objective

Implement the main wrist-camera calibration route based on a fixed board anchor `T_R_P` and raw wrist ChArUco corner observations.

This should improve over simple per-frame PnP averaging / classical hand-eye by optimizing `T_E_Cw` directly against corner reprojection error:

```text
T_Cw_P_i = inv(T_R_E_i @ T_E_Cw) @ T_R_P
uv_pred = project(K_Cw, dist_Cw, T_Cw_P_i, board_points_P)
minimize sum_i robust_loss(uv_pred_i - uv_obs_i)
```

## Suggested Command

```bash
python -m rm75_app calib-wrist-anchor -- \
  --config assets/calibration/rm75_calibration_config.example.json \
  --board-anchor-run-dir <global_board_anchor_run> \
  --manual-capture \
  --manual-count 40 \
  --preview
```

Reuse old images:

```bash
python -m rm75_app calib-wrist-anchor -- \
  --input-run <wrist_camera_board_or_anchor_run> \
  --board-anchor-run-dir <global_board_anchor_run>
```

## Required Inputs

- Fixed board anchor `T_R_P` from Task 01.
- Wrist camera intrinsics/distortion.
- Per-frame robot end-effector pose `T_R_E_i`.
- Per-frame wrist image ChArUco corner observations.
- Board 3D corner coordinates in `P`.

## Initialization

For every valid frame:

```text
T_Cw_P_i = solvePnP(board_points_P, uv_obs_i)
T_E_Cw_i = inv(T_R_E_i) @ T_R_P @ inv(T_Cw_P_i)
```

Then compute a robust initial `T_E_Cw_init`:

- reject obvious outliers by translation/rotation spread;
- use median translation;
- average rotations or use the current helper pattern in `common.average_transforms`.

## Optimization

Use `scipy.optimize.least_squares` with:

- parameterization: 6D delta around `T_E_Cw_init`;
- robust loss: `soft_l1` or `huber`;
- optional weak prior on delta from init;
- frame weighting by ChArUco quality:
  - more corners = higher confidence;
  - lower detection RMSE = higher confidence;
  - better image coverage = higher confidence.

Keep OpenCV distortion in projection with `cv2.projectPoints` unless the existing board helper already has a better projection API.

## Required Outputs

Run directory:

```text
runtime_data/calibration_runs/<timestamp>_wrist_camera_board_anchor/
```

Write:

```text
ee_T_wrist_camera.npy/json
wrist_camera_T_ee.npy/json
ee_T_wrist_camera_initial.npy/json
board_anchor_used.npy/json
observations.json
wrist_anchor_optimization_report.json
calibration_report.json
report.html
overlays/
```

Report fields:

```json
{
  "frames_total": 40,
  "frames_used": 35,
  "initial_reprojection_rmse_px": 0.45,
  "final_reprojection_rmse_px": 0.18,
  "initial_candidate_spread_translation_mm_median": 2.1,
  "final_frame_pose_residual_translation_mm_median": 1.2,
  "outlier_frames": []
}
```

## Important Behavior

- Do not overwrite the existing `*_wrist_camera_board` result unless a flag explicitly requests it.
- Add `--write-as-refined-to-wrist-run` or similar only if needed.
- Keep compatibility with `rm75_app/perception/wrist_relation.py`, which currently looks for recent board/refined transforms.

## Acceptance Criteria

- Works from a previous captured wrist board run and a board-anchor run.
- Produces `ee_T_wrist_camera.npy/json` compatible with runtime lookup.
- Saves overlays showing observed ChArUco corners and optimized projected corners.
- Compares against classical hand-eye result if the input run contains it.
- `python -m py_compile` passes.

## Suggested Write Scope

- Add a new file under `rm75_app/calibration/`, for example `wrist_camera_board_anchor_calibration.py`.
- Small command registration edits in `cli.py` and `commands.py`.
- Do not modify SAM3 body refine code in this task.

