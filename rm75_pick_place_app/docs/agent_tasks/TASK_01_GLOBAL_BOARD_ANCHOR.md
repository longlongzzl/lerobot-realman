# Task 01: Global Camera Board Anchor

## Objective

Add a calibration command that uses the already-calibrated main/global camera to estimate the fixed board pose in robot-base coordinates:

```text
T_R_P = T_R_Cg @ T_Cg_P
```

This produces a stable `T_R_P` board anchor that later wrist-camera optimization can use.

## Suggested Command

Add one of these, following existing naming style:

```bash
python -m rm75_app calib-board-anchor -- \
  --config assets/calibration/rm75_calibration_config.example.json \
  --base-camera-run-dir <base_camera_run_dir> \
  --manual-capture \
  --manual-count 20 \
  --preview
```

Also support reuse:

```bash
python -m rm75_app calib-board-anchor -- \
  --input-run <previous_anchor_or_image_run> \
  --base-camera-run-dir <base_camera_run_dir>
```

## Required Inputs

- Main camera intrinsics and distortion.
- Main camera external calibration `T_R_Cg`.
- ChArUco board config from `assets/calibration/rm75_calibration_config.example.json`.
- One or more global-camera images of the fixed board.

## Required Outputs

Create a run directory:

```text
runtime_data/calibration_runs/<timestamp>_global_board_anchor/
```

Write:

```text
R_T_board.npy/json
board_T_R.npy/json
camera_T_board_per_frame.npy
global_board_anchor_report.json
report.html
images/
overlays/
observations.json
```

Use existing naming helpers where possible. Include a metadata field with all transform aliases:

```json
{
  "T_R_P": "...",
  "T_R_Cg": "...",
  "T_Cg_P_per_frame": "...",
  "selected_frame_count": 18,
  "board_reprojection_rmse_px_mean": 0.12
}
```

## Implementation Notes

- Reuse ChArUco detection from `rm75_app/calibration/board.py`.
- Reuse matrix I/O and reports from `common.py` and `reporting.py`.
- Average multiple `T_R_P` estimates robustly:
  - reject frames with too few corners;
  - reject high reprojection RMSE;
  - use median translation plus rotation averaging or a small least-squares refinement.
- Add an HTML report with board axis overlay and per-frame error table.
- If multiple valid frames are captured from a fixed global camera, they should all agree; report translation/rotation spread.

## Acceptance Criteria

- The command runs in previous-run mode without hardware.
- Outputs `R_T_board.npy/json` as a 4x4 transform.
- Report includes per-frame ChArUco corner count and reprojection RMSE.
- Bad frames are rejected and listed with reasons.
- `python -m py_compile` passes for changed Python files.

## Suggested Write Scope

- Add a new file under `rm75_app/calibration/`, for example `global_board_anchor.py`.
- Edit `rm75_app/cli.py` and `rm75_app/commands.py` only to register the command.
- Avoid changing wrist-camera calibration logic in this task.

