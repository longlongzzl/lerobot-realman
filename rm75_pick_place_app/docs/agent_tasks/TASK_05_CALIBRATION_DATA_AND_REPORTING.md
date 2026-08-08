# Task 05: Calibration Data, Lookup, And Reporting

## Objective

Unify the file formats, latest-result lookup, documentation, and visualization across the new calibration routes.

This task is glue work. It should make outputs from Tasks 01-04 easy to consume by later runtime code and by humans reading reports.

## Required Conventions

All calibration run reports should include:

```json
{
  "schema_version": 1,
  "run_kind": "...",
  "created_at": "...",
  "command": "...",
  "config_path": "...",
  "frames_total": 0,
  "frames_used": 0,
  "accepted": true,
  "rejection_reason": "",
  "transforms": {
    "T_R_Cg": null,
    "T_R_P": null,
    "T_E_Cw": null
  },
  "metrics": {}
}
```

Do not break existing consumers. Add aliases rather than removing old fields.

## Result Lookup

Review and extend:

```text
rm75_app/perception/wrist_relation.py
```

Needed behavior:

1. Prefer explicitly selected calibration paths from config/CLI.
2. Prefer accepted refined/joint results only if they are marked accepted.
3. Fall back to trusted board calibration.
4. Do not accidentally pick smoke-test visual-refine runs that kept identity or were rejected.

Suggested priority:

```text
explicit path
accepted *_wrist_camera_board_anchor/ee_T_wrist_camera.npy
accepted *_two_camera_joint_board/ee_T_wrist_camera_joint.npy
accepted refined transform copied into trusted wrist board run
normal *_wrist_camera_board/ee_T_wrist_camera.npy
```

## Reports

Update `rm75_app/calibration/reporting.py` if useful, but keep it simple.

Each report should show:

- transform matrices;
- accepted/rejected status;
- frame count and outlier list;
- reprojection metrics;
- transform deltas in mm/deg;
- image overlays.

## Documentation

Update:

```text
README.md
rm75_app/calibration/README.md
```

Explain the recommended sequence:

```text
1. calib-base                 main/global camera to robot base
2. calib-board-anchor         fixed board in robot base
3. calib-wrist-anchor         wrist camera from anchored board
4. calib-joint-board          optional two-camera joint refinement
5. calib-wrist-visual         optional robot-body sanity check
6. calib-check                final cross-check
```

Mention explicitly that insertion-time local servo is a separate future layer and is not required for this calibration stack.

## Acceptance Criteria

- New reports have a consistent `accepted` field.
- Runtime lookup ignores rejected calibration runs.
- README commands use the correct `python -m rm75_app <command> -- <args>` separator.
- Existing calibration commands remain documented.
- `python -m py_compile` passes for changed Python files.

## Suggested Write Scope

- `rm75_app/calibration/reporting.py`
- `rm75_app/perception/wrist_relation.py`
- `README.md`
- `rm75_app/calibration/README.md`
- Avoid heavy optimizer changes in this task.

