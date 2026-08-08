# Calibration Agent Tasks Index

This folder splits the next calibration work into parallel agent-sized tasks. The scope intentionally excludes insertion-time local visual servoing. These tasks focus on global camera anchoring, wrist-camera board calibration, joint optimization, robot-body visual refinement, and shared reporting/data integration.

## Shared Goal

Build a stronger calibration stack for the RM75 wrist camera:

```text
1. Main/global camera is calibrated to robot base with EasyHeC-style visual alignment.
2. Main/global camera observes a fixed ChArUco board and anchors the board in robot-base coordinates.
3. Wrist camera observes the same fixed board from many arm poses.
4. Optimize wrist camera extrinsic from raw board corner reprojection, not only hand-eye pose averaging.
5. Optionally run joint two-camera optimization and cross-checks.
6. Keep robot-body/SAM3 visual refinement as an auxiliary sanity check, not as the main route.
```

Do not implement the insertion-before-hole local correction loop in this task set.

## Coordinate Convention

Use `T_A_B` to mean a transform that maps coordinates from frame `B` into frame `A`.

Frames:

```text
R  = robot base
E  = end effector / flange / configured wrist mount link
Cg = global/main camera
Cw = wrist camera
P  = board / calibration pattern
```

Important chain:

```text
T_R_P = T_R_Cg @ T_Cg_P
T_R_P = T_R_E_i @ T_E_Cw @ T_Cw_P_i
T_E_Cw_i = inv(T_R_E_i) @ T_R_P @ inv(T_Cw_P_i)
```

For reprojection optimization, predict wrist-camera board corners with:

```text
T_Cw_P_i = inv(T_R_E_i @ T_E_Cw) @ T_R_P
uv_pred = project(K_Cw, dist_Cw, T_Cw_P_i, board_points_P)
```

## Existing Code To Reuse

- `rm75_app/calibration/common.py`: transforms, JSON/matrix I/O, RealSense, RealMan, FK helpers.
- `rm75_app/calibration/board.py`: ChArUco / board detection helpers.
- `rm75_app/calibration/base_camera_visual_calibration.py`: main/global camera visual calibration.
- `rm75_app/calibration/wrist_camera_board_calibration.py`: current wrist ChArUco hand-eye calibration.
- `rm75_app/calibration/combined_calibration_check.py`: current two-camera board cross-check.
- `rm75_app/calibration/wrist_camera_visual_refine.py`: robot-body visual refinement with optional SAM3 mask targets.
- `rm75_app/calibration/reporting.py`: HTML reports.
- `rm75_app/cli.py` and `rm75_app/commands.py`: command registration.
- `assets/calibration/rm75_calibration_config.example.json`: shared config.

Remember the CLI argument separator:

```bash
python -m rm75_app <command> -- <command args>
```

## Current Trusted Data Point

Good wrist board calibration run:

```text
runtime_data/calibration_runs/20260525_161851_wrist_camera_board
```

Known quality:

```text
18/18 valid detections
selected DANIILIDIS hand-eye
mean translation residual about 3.30 mm
max translation residual about 6.18 mm
mean rotation residual about 0.355 deg
```

Recent wrist body visual-refine tests did not produce a trustworthy improvement. Treat robot-body visual refinement as auxiliary until reports prove otherwise.

## Parallel Task Files

1. `TASK_01_GLOBAL_BOARD_ANCHOR.md`
   Implement a main/global camera board-anchor run that outputs `T_R_P`.

2. `TASK_02_WRIST_BOARD_REPROJECTION_OPTIMIZER.md`
   Implement wrist-camera calibration using fixed board anchor and multi-frame raw ChArUco corner reprojection.

3. `TASK_03_TWO_CAMERA_JOINT_OPTIMIZATION.md`
   Implement optional joint optimization over `T_E_Cw` and `T_R_P` using main and wrist camera board observations.

4. `TASK_04_SAM3_BODY_REFINE_AUXILIARY.md`
   Harden the SAM3 robot-body refinement path as a sanity-check tool.

5. `TASK_05_CALIBRATION_DATA_AND_REPORTING.md`
   Unify schemas, reports, latest-result lookup, and docs across all calibration routes.

## Coordination Rules

- Prefer additive files or clearly scoped edits.
- Keep outputs under `runtime_data/calibration_runs/<timestamp>_<kind>/`.
- Never overwrite a trusted calibration in another run directory unless a CLI flag explicitly requests it.
- Any refined transform that should be consumed by runtime code must be written into a clearly named file and recorded in JSON metadata.
- Use `apply_patch` for manual edits.
- Run at least `python -m py_compile` on changed Python files.
- If hardware is required, include a dry-run or previous-run mode so the task can be tested without moving the arm.

