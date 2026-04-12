import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args():
    code_dir = Path(__file__).resolve().parent
    default_output = code_dir / "demo_data" / f"recorded_{time.strftime('%Y%m%d_%H%M%S')}"

    parser = argparse.ArgumentParser(
        description="Record an RGB-D sequence from Intel RealSense for FoundationPose run_demo.py."
    )
    parser.add_argument("--serial", type=str, default=342522073637, help="RealSense serial number. Omit to list devices.")
    parser.add_argument("--output_dir", type=str, default=str(default_output), help="Output scene directory.")
    parser.add_argument("--width", type=int, default=640, help="Color/depth width.")
    parser.add_argument("--height", type=int, default=480, help="Color/depth height.")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS.")
    parser.add_argument("--warmup", type=int, default=30, help="Frames to discard before preview.")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Maximum number of frames to record including frame 0. 0 means no limit.",
    )
    return parser.parse_args()


def list_devices():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("未检测到 RealSense 相机。")
        return
    print("检测到以下 RealSense 相机：")
    for idx, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        fw = dev.get_info(rs.camera_info.firmware_version)
        print(f"  [{idx}] {name}  serial={serial}  firmware={fw}")


def make_output_dirs(output_dir: Path):
    (output_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (output_dir / "depth").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)


def save_frame(output_dir: Path, frame_idx: int, color_bgr: np.ndarray, depth_mm: np.ndarray):
    stem = f"{frame_idx:06d}"
    cv2.imwrite(str(output_dir / "rgb" / f"{stem}.png"), color_bgr)
    cv2.imwrite(str(output_dir / "depth" / f"{stem}.png"), depth_mm)


def save_mask(output_dir: Path, bbox, image_shape):
    x, y, w, h = [int(v) for v in bbox]
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if w > 0 and h > 0:
        mask[y : y + h, x : x + w] = 255
    cv2.imwrite(str(output_dir / "masks" / "000000.png"), mask)
    return mask


def draw_overlay(image, text_lines, recording, frame_idx):
    vis = image.copy()
    y = 24
    for line in text_lines:
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y += 26
    if recording:
        cv2.circle(vis, (vis.shape[1] - 24, 24), 8, (0, 0, 255), -1)
        cv2.putText(
            vis,
            f"REC frame={frame_idx:06d}",
            (vis.shape[1] - 190, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return vis


def main():
    args = parse_args()

    if args.serial is None:
        list_devices()
        raise SystemExit("请用 --serial 指定要录制的 D435 序列号。")
    serial = str(args.serial)

    output_dir = Path(args.output_dir).resolve()
    make_output_dirs(output_dir)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    color_intrinsics = color_profile.get_intrinsics()
    K = np.array(
        [
            [color_intrinsics.fx, 0.0, color_intrinsics.ppx],
            [0.0, color_intrinsics.fy, color_intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.savetxt(output_dir / "cam_K.txt", K)

    metadata = {
        "serial": serial,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "depth_scale_m_per_unit": depth_scale,
        "depth_png_unit": "millimeter",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    timestamps = []
    frame_idx = 0
    recording = False
    first_bbox = None

    try:
        for _ in range(max(args.warmup, 0)):
            pipeline.wait_for_frames()

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_mm = np.round(depth_raw.astype(np.float32) * depth_scale * 1000.0).astype(np.uint16)

            lines = [
                "b: select bbox on current frame and start recording",
                "q: quit",
            ]
            if recording:
                lines = [
                    "recording... press q to stop and exit",
                    f"saved frames: {frame_idx}",
                ]
            preview = draw_overlay(color_bgr, lines, recording, frame_idx)
            cv2.imshow("FoundationPose RGB-D Recorder", preview)

            key = cv2.waitKey(1) & 0xFF
            if not recording and key == ord("b"):
                roi = cv2.selectROI("Select Target BBox", color_bgr, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select Target BBox")
                x, y, w, h = [int(v) for v in roi]
                if w <= 0 or h <= 0:
                    print("未选择有效框，继续预览。")
                    continue

                save_frame(output_dir, 0, color_bgr, depth_mm)
                save_mask(output_dir, roi, color_bgr.shape)
                timestamps.append(
                    {
                        "frame": 0,
                        "timestamp_ms": int(color_frame.get_timestamp()),
                    }
                )
                first_bbox = {"x": x, "y": y, "w": w, "h": h}
                frame_idx = 1
                recording = True
                print(f"开始录制到: {output_dir}")
                print("已保存 frame 000000 和 masks/000000.png")
                continue

            if key == ord("q"):
                break

            if recording:
                save_frame(output_dir, frame_idx, color_bgr, depth_mm)
                timestamps.append(
                    {
                        "frame": frame_idx,
                        "timestamp_ms": int(color_frame.get_timestamp()),
                    }
                )
                frame_idx += 1
                if args.max_frames > 0 and frame_idx >= args.max_frames:
                    print(f"达到 max_frames={args.max_frames}，停止录制。")
                    break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    metadata["num_frames"] = frame_idx
    metadata["first_bbox"] = first_bbox
    metadata["timestamps"] = timestamps
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if frame_idx == 0:
        print("未保存任何帧。")
    else:
        print(f"录制完成，共保存 {frame_idx} 帧。")
        print(f"可直接运行: python run_demo.py --test_scene_dir {output_dir}")


if __name__ == "__main__":
    main()
