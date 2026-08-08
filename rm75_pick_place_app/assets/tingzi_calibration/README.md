# Tingzi calibration assets

This directory stores the camera extrinsic used by the Jimu/Tingzi assembly
portable runners.

It is intentionally separate from `../calibration/`. Re-running the generic
base-camera calibration with `calib-base --save-to-app-assets` updates
`../calibration/`, but should not overwrite the Tingzi-specific values here.

Files:

- `camera_extrinsic_opencv.npy`: OpenCV-style camera extrinsic consumed by the
  Tingzi/Jimu runner.
- `camera_extrinsic_opencv.json`: readable copy of the same matrix.
- `base_T_camera.npy`: base-to-camera matrix kept for inspection/conversion.
- `base_T_camera.json`: readable copy of the same matrix.

The portable Tingzi runner now uses this file by default:

```bash
--camera-extrinsic-opencv-path /home/zhangzhao/Desktop/lerobot/rm75_pick_place_app/assets/tingzi_calibration/camera_extrinsic_opencv.npy
```

You can still pass `--camera-extrinsic-opencv-path` manually to test another
extrinsic without modifying this directory.
