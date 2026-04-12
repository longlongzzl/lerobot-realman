import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from lerobot_sim2real.config.real_robot import create_real_robot


try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


@dataclass
class Detection:
    xyxy: np.ndarray
    conf: float
    cls_id: int
    cls_name: str

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy
        return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="本地 YOLO 权重路径，如 best.pt")
    parser.add_argument("--image", type=str, default=None, help="单张图片测试路径")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--class-name", action="append", default=None, help="只保留这些类别名，可重复传入")
    parser.add_argument("--class-id", type=int, action="append", default=None, help="只保留这些类别 id，可重复传入")
    parser.add_argument("--min-count", type=int, default=1, help="至少检测到多少个目标才可能 ready")
    parser.add_argument("--max-count", type=int, default=None, help="至多允许多少个目标")
    parser.add_argument("--stable-frames", type=int, default=5, help="连续多少帧稳定才判 ready")
    parser.add_argument("--center-move-px", type=float, default=12.0, help="相邻帧中心点平均移动阈值（像素）")
    parser.add_argument("--roi", type=str, default=None, help="桌面 ROI: x1,y1,x2,y2")
    parser.add_argument("--show", action="store_true", help="显示检测窗口")
    parser.add_argument("--save", type=str, default=None, help="保存当前可视化图片路径")
    parser.add_argument("--camera-timeout-ms", type=int, default=200)
    return parser.parse_args()


def parse_roi(roi_str: str | None):
    if roi_str is None:
        return None
    vals = [int(x) for x in roi_str.split(",")]
    if len(vals) != 4:
        raise ValueError("--roi 格式必须是 x1,y1,x2,y2")
    x1, y1, x2, y2 = vals
    if not (x2 > x1 and y2 > y1):
        raise ValueError("--roi 需要满足 x2>x1 且 y2>y1")
    return x1, y1, x2, y2


def crop_with_roi(img: np.ndarray, roi):
    if roi is None:
        return img, (0, 0)
    x1, y1, x2, y2 = roi
    return img[y1:y2, x1:x2], (x1, y1)


def load_model(model_path: str):
    if YOLO is None:
        raise RuntimeError(
            "当前环境没有安装 ultralytics，无法运行 YOLO。"
            "先在当前环境安装后再运行这个脚本。"
        )
    return YOLO(model_path)


def run_detector(model, image: np.ndarray, conf: float, iou: float) -> list[Detection]:
    result = model.predict(image, conf=conf, iou=iou, verbose=False)[0]
    names = result.names
    detections: list[Detection] = []
    if result.boxes is None:
        return detections

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    clss = result.boxes.cls.detach().cpu().numpy().astype(np.int32)

    for box, score, cls_id in zip(xyxy, confs, clss):
        detections.append(
            Detection(
                xyxy=box.astype(np.float32),
                conf=float(score),
                cls_id=int(cls_id),
                cls_name=str(names[int(cls_id)]),
            )
        )
    return detections


def filter_detections(
    detections: list[Detection],
    class_names: set[str] | None,
    class_ids: set[int] | None,
    offset_xy=(0, 0),
) -> list[Detection]:
    ox, oy = offset_xy
    filtered: list[Detection] = []
    for det in detections:
        if class_names is not None and det.cls_name not in class_names:
            continue
        if class_ids is not None and det.cls_id not in class_ids:
            continue
        det = Detection(
            xyxy=det.xyxy + np.array([ox, oy, ox, oy], dtype=np.float32),
            conf=det.conf,
            cls_id=det.cls_id,
            cls_name=det.cls_name,
        )
        filtered.append(det)
    return filtered


def sort_centers(detections: Iterable[Detection]) -> np.ndarray:
    centers = np.array([det.center for det in detections], dtype=np.float32)
    if centers.size == 0:
        return centers.reshape(0, 2)
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    return centers[order]


def compute_motion(prev_dets: list[Detection] | None, cur_dets: list[Detection]) -> float | None:
    if prev_dets is None:
        return None
    prev_centers = sort_centers(prev_dets)
    cur_centers = sort_centers(cur_dets)
    if len(prev_centers) != len(cur_centers):
        return math.inf
    if len(cur_centers) == 0:
        return math.inf
    return float(np.linalg.norm(cur_centers - prev_centers, axis=1).mean())


def is_ready(
    detections: list[Detection],
    prev_detections: list[Detection] | None,
    min_count: int,
    max_count: int | None,
    center_move_px: float,
):
    count = len(detections)
    if count < min_count:
        return False, None, f"count<{min_count}"
    if max_count is not None and count > max_count:
        return False, None, f"count>{max_count}"

    motion = compute_motion(prev_detections, detections)
    if motion is None:
        return False, None, "warmup"
    if motion == math.inf:
        return False, motion, "count_changed"
    if motion > center_move_px:
        return False, motion, f"motion>{center_move_px:.1f}"
    return True, motion, "stable"


def draw_overlay(image: np.ndarray, detections: list[Detection], ready: bool, stable_hits: int, stable_need: int, msg: str):
    canvas = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.xyxy.astype(int).tolist()
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{det.cls_name} {det.conf:.2f}"
        cv2.putText(canvas, text, (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    title = f"{'READY' if ready else 'NOT_READY'} stable={stable_hits}/{stable_need} {msg}"
    color = (0, 200, 0) if ready else (0, 0, 255)
    cv2.putText(canvas, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return canvas


def read_image_from_camera(timeout_ms: int):
    robot, camera = create_real_robot(auto_connect=False)
    try:
        camera.connect()
        frame = camera.read(timeout_ms=timeout_ms)
        return frame, camera, robot
    except Exception:
        try:
            camera.disconnect()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        raise


def run_single_image(args, model):
    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"无法读取图片: {args.image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    roi = parse_roi(args.roi)
    cropped, offset = crop_with_roi(image, roi)
    detections = run_detector(model, cropped, args.conf, args.iou)
    detections = filter_detections(
        detections,
        set(args.class_name) if args.class_name else None,
        set(args.class_id) if args.class_id else None,
        offset_xy=offset,
    )
    ready = len(detections) >= args.min_count and (args.max_count is None or len(detections) <= args.max_count)
    msg = f"count={len(detections)}"
    canvas = draw_overlay(image, detections, ready, int(ready), 1, msg)
    print("READY" if ready else "NOT_READY", msg)
    if args.save:
        out = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.save, out)
    if args.show:
        cv2.imshow("yolo_ready", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_live(args, model):
    roi = parse_roi(args.roi)
    class_names = set(args.class_name) if args.class_name else None
    class_ids = set(args.class_id) if args.class_id else None

    robot, camera = create_real_robot(auto_connect=False)
    stable_hits = 0
    prev_detections: list[Detection] | None = None

    try:
        camera.connect()
        print(
            f"[YOLOReady] live start model={args.model} min_count={args.min_count} "
            f"stable_frames={args.stable_frames} move_px={args.center_move_px}"
        )
        while True:
            frame = camera.read(timeout_ms=args.camera_timeout_ms)
            cropped, offset = crop_with_roi(frame, roi)
            detections = run_detector(model, cropped, args.conf, args.iou)
            detections = filter_detections(detections, class_names, class_ids, offset_xy=offset)

            stable_ok, motion, msg = is_ready(
                detections,
                prev_detections,
                min_count=args.min_count,
                max_count=args.max_count,
                center_move_px=args.center_move_px,
            )
            prev_detections = detections

            if stable_ok:
                stable_hits += 1
            else:
                stable_hits = 0

            ready = stable_hits >= args.stable_frames
            if motion is not None and motion is not math.inf:
                msg = f"{msg} motion={motion:.1f}px count={len(detections)}"
            else:
                msg = f"{msg} count={len(detections)}"

            print(f"[YOLOReady] ready={ready} stable={stable_hits}/{args.stable_frames} {msg}")

            canvas = draw_overlay(frame, detections, ready, stable_hits, args.stable_frames, msg)
            if args.save:
                cv2.imwrite(args.save, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            if args.show:
                cv2.imshow("yolo_ready", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            if ready:
                print("[YOLOReady] READY")
                break
    finally:
        try:
            camera.disconnect()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        if args.show:
            cv2.destroyAllWindows()


def main():
    args = parse_args()
    model_path = Path(args.model)
    # Support either a local weights path or a model name such as "yolo11n.pt"
    # that Ultralytics can resolve/download automatically.
    if model_path.exists():
        model_ref = str(model_path)
    else:
        if model_path.parent != Path(".") or model_path.is_absolute():
            raise RuntimeError(f"模型文件不存在: {model_path}")
        model_ref = args.model

    model = load_model(model_ref)
    if args.image is not None:
        run_single_image(args, model)
    else:
        run_live(args, model)


if __name__ == "__main__":
    main()
