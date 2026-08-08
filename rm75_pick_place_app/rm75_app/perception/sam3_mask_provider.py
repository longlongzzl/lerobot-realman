#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


DEFAULT_SAM3_CHECKPOINT_PATH = "/home/zhangzhao/Downloads/sam3.pt"


def _sanitize_proxy_env_for_hf():
    for key in ("ALL_PROXY", "all_proxy"):
        if str(os.environ.get(key, "")).lower().startswith("socks://"):
            os.environ.pop(key, None)


def _clip_box_xyxy(box_xyxy, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in np.asarray(box_xyxy, dtype=np.float32).reshape(-1)[:4]]
    x1 = min(max(x1, 0.0), float(width - 1))
    y1 = min(max(y1, 0.0), float(height - 1))
    x2 = min(max(x2, x1 + 1.0), float(width))
    y2 = min(max(y2, y1 + 1.0), float(height))
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _xyxy_to_norm_cxcywh(box_xyxy, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in _clip_box_xyxy(box_xyxy, width, height)]
    return [
        ((x1 + x2) * 0.5) / max(float(width), 1.0),
        ((y1 + y2) * 0.5) / max(float(height), 1.0),
        max(x2 - x1, 1.0) / max(float(width), 1.0),
        max(y2 - y1, 1.0) / max(float(height), 1.0),
    ]


def _box_mask(shape: tuple[int, int], box_xyxy) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = [int(round(v)) for v in _clip_box_xyxy(box_xyxy, width, height)]
    mask = np.zeros((height, width), dtype=bool)
    mask[max(y1, 0) : min(y2, height), max(x1, 0) : min(x2, width)] = True
    return mask


def _mask_bbox_xyxy(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(np.asarray(mask > 0, dtype=bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _bbox_iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in np.asarray(a, dtype=np.float32).reshape(-1)[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in np.asarray(b, dtype=np.float32).reshape(-1)[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / max(area_a + area_b - inter, 1e-6))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.asarray(mask > 0, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if count <= 1:
        return mask_u8.astype(bool)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(text))


def _rank_masks(masks: np.ndarray, boxes: np.ndarray, scores: np.ndarray, select_box, min_area: int) -> list[tuple[int, dict]]:
    if masks.ndim == 4:
        masks = masks[:, 0]
    masks = np.asarray(masks > 0, dtype=bool)
    if masks.ndim != 3 or masks.shape[0] <= 0:
        raise RuntimeError("SAM3 returned no masks")

    height, width = masks.shape[1:3]
    select_region = _box_mask((height, width), select_box)
    select_area = max(float(np.count_nonzero(select_region)), 1.0)
    candidates: list[tuple[int, dict]] = []
    for idx, mask in enumerate(masks):
        area = float(np.count_nonzero(mask))
        if area < float(min_area):
            continue
        overlap = float(np.count_nonzero(mask & select_region))
        if overlap <= 0.0:
            continue
        mask_box = boxes[idx] if boxes is not None and idx < len(boxes) else _mask_bbox_xyxy(mask)
        if mask_box is None:
            continue
        score = float(scores[idx]) if idx < len(scores) else 0.0
        cov = overlap / select_area
        inbox = overlap / max(area, 1.0)
        iou = _bbox_iou_xyxy(mask_box, select_box)
        rank = score + (2.0 * cov) + (1.5 * inbox) + (0.8 * iou)
        candidates.append(
            (
                idx,
                {
                    "model_score": score,
                    "coverage": cov,
                    "mask_in_box": inbox,
                    "bbox_iou": iou,
                    "area": area,
                    "rank": rank,
                    "box": np.asarray(mask_box, dtype=np.float32).tolist(),
                },
            )
        )

    if not candidates:
        raise RuntimeError("SAM3 masks did not overlap the selected box")
    candidates.sort(key=lambda item: item[1].get("rank", 0.0), reverse=True)
    return candidates


def _save_overlay(rgb: np.ndarray, box_xyxy, mask: np.ndarray, out_path: Path, label: str):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    overlay[np.asarray(mask > 0, dtype=bool)] = (0, 180, 255)
    canvas = cv2.addWeighted(overlay, 0.35, bgr, 0.65, 0.0)
    x1, y1, x2, y2 = [int(round(v)) for v in np.asarray(box_xyxy, dtype=np.float32).reshape(-1)[:4]]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(canvas, label[:80], (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.imwrite(str(out_path), canvas)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM3 image segmentation for one RGB frame or a batch of frames.")
    parser.add_argument("--rgb-path", required=False)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--items-json", type=str, default=None, help="Optional list of prompt items for one-image multi-object SAM3 inference.")
    parser.add_argument("--batch-frames-json", type=str, default=None, help="Optional list of frames. Loads SAM3 once, then runs all frame prompts.")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--mode", choices=["text", "box", "text_box"], default="text_box")
    parser.add_argument("--box", type=float, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--select-box", type=float, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_SAM3_CHECKPOINT_PATH)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--min-mask-area", type=int, default=64)
    parser.add_argument("--morph-kernel", type=int, default=3)
    parser.add_argument("--sam3-max-masks-per-item", type=int, default=1, help="Maximum SAM3 candidates to output per item.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _run_one_item(processor, base_state: dict, rgb: np.ndarray, args, item: dict, item_output_dir: Path) -> list[dict]:
    height, width = rgb.shape[:2]
    mode = str(item.get("mode") or args.mode)
    prompt = item.get("prompt", args.prompt)
    box = item.get("box", args.box)
    select_box = item.get("select_box", box)
    if select_box is None:
        select_box = [0.0, 0.0, float(width), float(height)]
    select_box = _clip_box_xyxy(select_box, width, height)

    if mode in {"box", "text_box"} and box is None:
        raise ValueError("--box is required for SAM3 box or text_box mode")
    if mode in {"text", "text_box"} and not prompt:
        raise ValueError("--prompt is required for SAM3 text or text_box mode")

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if str(args.device).startswith("cuda")
        else contextlib.nullcontext()
    )
    infer_t0 = time.perf_counter()
    state = dict(base_state)
    processor.reset_all_prompts(state)
    with amp_ctx:
        if mode == "text":
            state = processor.set_text_prompt(str(prompt), state)
        elif mode == "box":
            norm_box = _xyxy_to_norm_cxcywh(box, width, height)
            state = processor.add_geometric_prompt(norm_box, True, state)
        else:
            state = processor.set_text_prompt(str(prompt), state)
            norm_box = _xyxy_to_norm_cxcywh(box, width, height)
            state = processor.add_geometric_prompt(norm_box, True, state)
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
    infer_elapsed_ms = (time.perf_counter() - infer_t0) * 1000.0

    masks = state["masks"].detach().cpu().numpy()
    if masks.ndim == 4:
        masks = masks[:, 0]
    boxes = state["boxes"].detach().float().cpu().numpy() if "boxes" in state else None
    scores = state["scores"].detach().float().cpu().numpy() if "scores" in state else np.zeros((masks.shape[0],), dtype=np.float32)

    candidates = _rank_masks(masks, boxes, scores, select_box, int(args.min_mask_area))
    max_keep = max(1, int(getattr(args, "sam3_max_masks_per_item", 1)))
    selected_candidates = candidates[:max_keep]

    item_output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for rank_idx, (best_idx, meta) in enumerate(selected_candidates, start=1):
        best_mask = np.asarray(masks[best_idx], dtype=bool)
        if best_mask.ndim == 3:
            best_mask = best_mask[0]
        best_mask = _largest_component(best_mask)
        kernel_size = max(int(args.morph_kernel), 0)
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            best_mask = cv2.morphologyEx(best_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
            best_mask = cv2.morphologyEx(best_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)

        if len(selected_candidates) == 1:
            mask_path = item_output_dir / "sam3_mask.png"
            overlay_path = item_output_dir / "sam3_overlay.png"
        else:
            mask_path = item_output_dir / f"sam3_mask_{rank_idx:02d}.png"
            overlay_path = item_output_dir / f"sam3_overlay_{rank_idx:02d}.png"

        cv2.imwrite(str(mask_path), best_mask.astype(np.uint8) * 255)
        _save_overlay(rgb, select_box, best_mask, overlay_path, f"SAM3 {mode} r{rank_idx} {meta['model_score']:.3f}")
        mask_bbox = _mask_bbox_xyxy(best_mask)
        outputs.append(
            {
                "ok": True,
                "id": item.get("id"),
                "object_name": item.get("object_name"),
                "mode": mode,
                "prompt": prompt,
                "box": list(box) if box is not None else None,
                "select_box": select_box.tolist(),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "mask_pixels": int(np.count_nonzero(best_mask)),
                "mask_bbox": mask_bbox.tolist() if mask_bbox is not None else None,
                "selected_index": int(best_idx),
                "candidate_rank": int(rank_idx),
                "selected": meta,
                "candidate_count": int(len(candidates)),
                "infer_elapsed_ms": float(infer_elapsed_ms),
            }
        )
    return outputs


def _build_processor(args):
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    load_t0 = time.perf_counter()
    model = build_sam3_image_model(
        checkpoint_path=str(Path(args.checkpoint_path).expanduser()) if args.checkpoint_path else None,
        load_from_HF=args.checkpoint_path is None,
        device=str(args.device),
        eval_mode=True,
        enable_segmentation=True,
    )
    processor = Sam3Processor(
        model,
        resolution=int(args.resolution),
        device=str(args.device),
        confidence_threshold=float(args.confidence_threshold),
    )
    load_elapsed_ms = (time.perf_counter() - load_t0) * 1000.0
    return processor, load_elapsed_ms


def _set_image(processor, image: Image.Image, args) -> tuple[dict, float]:
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if str(args.device).startswith("cuda")
        else contextlib.nullcontext()
    )
    image_t0 = time.perf_counter()
    with amp_ctx:
        base_state = processor.set_image(image)
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
    image_elapsed_ms = (time.perf_counter() - image_t0) * 1000.0
    return base_state, image_elapsed_ms


def _load_prompt_items_for_frame(frame_item: dict, args) -> list[dict]:
    raw_items = frame_item.get("items")
    if isinstance(raw_items, list):
        return [dict(item) for item in raw_items]
    return [
        {
            "id": frame_item.get("id"),
            "object_name": frame_item.get("object_name"),
            "mode": frame_item.get("mode", args.mode),
            "prompt": frame_item.get("prompt", args.prompt),
            "box": frame_item.get("box", args.box),
            "select_box": frame_item.get("select_box", args.select_box),
        }
    ]


def _run_batch_frames(processor, args, output_dir: Path, load_elapsed_ms: float) -> None:
    with open(Path(args.batch_frames_json).expanduser(), "r") as f:
        frames = json.load(f)
    if not isinstance(frames, list):
        raise ValueError("--batch-frames-json must contain a JSON list")

    batch_results = []
    image_elapsed_total_ms = 0.0
    prompt_elapsed_total_ms = 0.0
    for frame_idx, frame_item in enumerate(frames):
        if not isinstance(frame_item, dict):
            batch_results.append({"ok": False, "frame_index": frame_idx, "error": "frame item is not an object"})
            continue
        frame_id = str(frame_item.get("id") or f"frame_{frame_idx:04d}")
        frame_output_dir = Path(frame_item.get("output_dir") or (output_dir / _safe_name(frame_id))).expanduser()
        frame_output_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = Path(frame_item.get("rgb_path", "")).expanduser()
        frame_results = []
        image_elapsed_ms = 0.0
        try:
            if not str(rgb_path):
                raise ValueError("frame item missing rgb_path")
            image = Image.open(rgb_path).convert("RGB")
            rgb = np.asarray(image, dtype=np.uint8)
            base_state, image_elapsed_ms = _set_image(processor, image, args)
            image_elapsed_total_ms += float(image_elapsed_ms)
            prompt_items = _load_prompt_items_for_frame(frame_item, args)
            for item_idx, prompt_item in enumerate(prompt_items):
                safe = _safe_name(str(prompt_item.get("id") or prompt_item.get("object_name") or f"item_{item_idx:02d}"))
                item_output_dir = frame_output_dir if len(prompt_items) == 1 else frame_output_dir / "items" / safe
                candidates = _run_one_item(processor, base_state, rgb, args, prompt_item, item_output_dir)
                for candidate in candidates:
                    candidate.update(
                        {
                            "frame_id": frame_id,
                            "frame_index": int(frame_idx),
                            "rgb_path": str(rgb_path),
                            "image_elapsed_ms": float(image_elapsed_ms),
                        }
                    )
                    prompt_elapsed_total_ms += float(candidate.get("infer_elapsed_ms", 0.0))
                frame_results.extend(candidates)
            if len(frame_results) == 1:
                frame_payload = dict(frame_results[0])
                frame_payload.update({"load_elapsed_ms": float(load_elapsed_ms), "image_elapsed_ms": float(image_elapsed_ms)})
            else:
                frame_payload = {
                    "ok": bool(frame_results),
                    "frame_id": frame_id,
                    "frame_index": int(frame_idx),
                    "rgb_path": str(rgb_path),
                    "item_count": len(frame_results),
                    "results": frame_results,
                    "load_elapsed_ms": float(load_elapsed_ms),
                    "image_elapsed_ms": float(image_elapsed_ms),
                    "total_prompt_infer_elapsed_ms": float(sum(float(item.get("infer_elapsed_ms", 0.0)) for item in frame_results)),
                }
            with open(frame_output_dir / "sam3_result.json", "w") as f:
                json.dump(frame_payload, f, indent=2)
            batch_results.append(frame_payload)
            print(f"[sam3-batch] frame {frame_id}: items={len(frame_results)} image={image_elapsed_ms:.1f}ms")
        except Exception as exc:
            error_payload = {
                "ok": False,
                "frame_id": frame_id,
                "frame_index": int(frame_idx),
                "rgb_path": str(rgb_path),
                "output_dir": str(frame_output_dir),
                "error": repr(exc),
                "load_elapsed_ms": float(load_elapsed_ms),
                "image_elapsed_ms": float(image_elapsed_ms),
            }
            with open(frame_output_dir / "sam3_result.json", "w") as f:
                json.dump(error_payload, f, indent=2)
            batch_results.append(error_payload)
            print(f"[sam3-batch] frame {frame_id}: failed {exc!r}")

    result = {
        "ok": True,
        "frame_count": len(frames),
        "results": batch_results,
        "load_elapsed_ms": float(load_elapsed_ms),
        "total_image_elapsed_ms": float(image_elapsed_total_ms),
        "total_prompt_infer_elapsed_ms": float(prompt_elapsed_total_ms),
    }
    result_path = output_dir / "sam3_batch_frames_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


def main():
    args = parse_args()
    _sanitize_proxy_env_for_hf()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    processor, load_elapsed_ms = _build_processor(args)

    if args.batch_frames_json:
        _run_batch_frames(processor, args, output_dir, load_elapsed_ms)
        return

    if not args.rgb_path:
        raise ValueError("--rgb-path is required unless --batch-frames-json is used")
    rgb_path = Path(args.rgb_path).expanduser()
    image = Image.open(rgb_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    base_state, image_elapsed_ms = _set_image(processor, image, args)

    if args.items_json:
        with open(Path(args.items_json).expanduser(), "r") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError("--items-json must contain a JSON list")
        results = []
        for idx, item in enumerate(items):
            safe = _safe_name(str(item.get("id") or item.get("object_name") or f"item_{idx:02d}"))
            try:
                candidates = _run_one_item(processor, base_state, rgb, args, item, output_dir / "items" / safe)
                results.extend(candidates)
            except Exception as exc:
                results.append(
                    {
                        "ok": False,
                        "id": item.get("id"),
                        "object_name": item.get("object_name"),
                        "mode": item.get("mode", args.mode),
                        "prompt": item.get("prompt", args.prompt),
                        "error": repr(exc),
                    }
                )
        result = {
            "ok": True,
            "rgb_path": str(rgb_path),
            "item_count": len(results),
            "results": results,
            "load_elapsed_ms": float(load_elapsed_ms),
            "image_elapsed_ms": float(image_elapsed_ms),
            "total_prompt_infer_elapsed_ms": float(sum(float(item.get("infer_elapsed_ms", 0.0)) for item in results)),
        }
        result_path = output_dir / "sam3_batch_result.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
        return

    item = {
        "id": "single",
        "mode": args.mode,
        "prompt": args.prompt,
        "box": args.box,
        "select_box": args.select_box,
    }
    candidates = _run_one_item(processor, base_state, rgb, args, item, output_dir)
    if not candidates:
        raise RuntimeError("SAM3 did not return any candidate masks")
    result = candidates[0]
    if len(candidates) > 1:
        result["all_candidates"] = candidates
    result.update(
        {
            "rgb_path": str(rgb_path),
            "load_elapsed_ms": float(load_elapsed_ms),
            "image_elapsed_ms": float(image_elapsed_ms),
        }
    )
    result_path = output_dir / "sam3_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
