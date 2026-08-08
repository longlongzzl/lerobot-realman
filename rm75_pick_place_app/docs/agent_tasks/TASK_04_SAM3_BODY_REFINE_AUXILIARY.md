# Task 04: SAM3 Body Refine As Auxiliary

## Objective

Harden the existing wrist robot-body visual refinement route so it is useful as a diagnostic and optional small-delta refinement, not as the main calibration route.

The current preferred main route should be board-anchor plus wrist reprojection optimization. This task improves the SAM3/EasyHeC-like body silhouette path for cross-checking and debugging.

## Existing State

Relevant file:

```text
rm75_app/calibration/wrist_camera_visual_refine.py
```

Recent behavior:

- Without masks, it uses Canny edges inside projected robot ROI.
- With `--sam3-mask`, it can generate SAM3 masks in one resident batch.
- SAM3 candidates are checked against the current projected robot mask to avoid choosing unrelated white objects.
- nvdiffrast candidate is rejected if the validation metric gets worse.

Known issue:

```text
Wrist camera often sees only base_link/link_1/link_2/link_3.
Frames with wrong SAM3 masks or invisible links can make the optimizer worse.
```

## Improvements To Make

1. Improve mask quality selection:
   - Save all SAM3 candidates for rejected frames.
   - Add a report table with selected candidate score, projected overlap, mask area, and reason for rejection.
   - Allow per-frame manual override mask from `--mask-dir`.

2. Improve link visibility handling:
   - For each frame, estimate projected area per link.
   - Skip links with tiny projected area.
   - Report which links are actually used per frame.

3. Improve validation metrics:
   - Add mask IoU/Dice metric when mask targets exist.
   - Keep edge distance metric as secondary.
   - Reject candidate if mask metric worsens even when edge metric improves.

4. Improve visualization:
   - HTML sections for `sam3_overlays`, `initial_overlays`, `candidate_overlays`, `final_overlays`.
   - Include accepted/rejected status per frame.

## Suggested CLI Additions

```bash
python -m rm75_app calib-wrist-visual -- \
  --input-run <wrist_visual_run> \
  --sam3-mask \
  --sam3-max-masks-per-item 4 \
  --include-links base_link link_1 link_2 link_3
```

Optional:

```bash
--min-link-projected-area-px 200
--min-mask-dice-improvement 0.005
--save-sam3-candidates
```

## Acceptance Criteria

- Existing `calib-wrist-visual` command still works without `--sam3-mask`.
- With `--sam3-mask`, SAM3 loads once for the batch.
- Bad masks are rejected with explicit reasons.
- Report includes mask IoU/Dice when masks are used.
- A worse optimization does not write a refined transform unless `--allow-worse-refine` is passed.
- `python -m py_compile` passes.

## Suggested Write Scope

- Primarily `rm75_app/calibration/wrist_camera_visual_refine.py`.
- Small changes to `rm75_app/perception/sam3_mask_provider.py` only if needed.
- Do not implement board-anchor or joint-board optimization in this task.

