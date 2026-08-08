#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from rm75_app.paths import RUNTIME_DIR
from rm75_app.perception import wrist_relation


DEFAULT_FRAME = Path(
    "/tmp/wrist_tingzi_pillar_split_live_verify/"
    "20260526_005917_wrist_live_overlay/cycle_0000/frame/frame_rgbd.npz"
)
DEFAULT_MASK = Path(
    "/tmp/wrist_tingzi_pillar_split_live_verify/"
    "20260526_005917_wrist_live_overlay/cycle_0000/sam3/sam3_union_mask.png"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _default_prior() -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray([0.020, 0.0, 0.020], dtype=np.float64)
    return T


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline simulation for the pavilion flow: start wrist held-object refinement after grasp, "
            "let transport time elapse, then check whether the refined relation is ready by pre_place."
        )
    )
    parser.add_argument("--object-name", default="tingzi_pillar_front_left")
    parser.add_argument("--frame-npz", type=Path, default=DEFAULT_FRAME if DEFAULT_FRAME.exists() else None)
    parser.add_argument("--mask-path", type=Path, default=DEFAULT_MASK if DEFAULT_MASK.exists() else None)
    parser.add_argument("--transport-duration-s", type=float, nargs="*", default=[1.0, 2.0, 4.0, 6.0, 8.0])
    parser.add_argument("--wait-at-pre-place-s", type=float, default=0.75)
    parser.add_argument("--timeout-s", type=float, default=18.0)
    parser.add_argument("--output-dir", type=Path, default=RUNTIME_DIR / "wrist_transport_refine_sim")
    parser.add_argument("--init-T-gripper-obj-path", type=Path, default=None)
    parser.add_argument("--T-gripper-cam-path", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--extra-arg", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.frame_npz is None or not Path(args.frame_npz).exists():
        raise FileNotFoundError("Provide --frame-npz; no default saved wrist frame was found.")
    if args.mask_path is None or not Path(args.mask_path).exists():
        raise FileNotFoundError("Provide --mask-path; no default saved wrist mask was found.")

    run_dir = Path(args.output_dir).expanduser().resolve() / time.strftime("%Y%m%d_%H%M%S_transport_refine_sim")
    run_dir.mkdir(parents=True, exist_ok=True)
    prior_path = Path(args.init_T_gripper_obj_path).expanduser() if args.init_T_gripper_obj_path else run_dir / "prior_T_gripper_obj.json"
    if args.init_T_gripper_obj_path is None:
        wrist_relation.write_matrix_json(prior_path, "T_gripper_obj", _default_prior())

    relation_out = run_dir / "wrist_relation_latest.json"
    extra_args = [
        "--frame-npz",
        str(Path(args.frame_npz).expanduser()),
        "--mask-path",
        str(Path(args.mask_path).expanduser()),
        "--pose-backend",
        "gripper_plane",
        "--mask-refine-mode",
        "gripper",
        "--mask-max-refine-candidates",
        "1",
        "--gripper-plane-refine-method",
        "smooth",
        "--gripper-plane-opt-maxiter",
        "30",
        "--gripper-plane-opt-point-count",
        "700",
        "--sam3-max-masks-per-item",
        "5",
        "--sam3-component-mode",
        "all",
    ]
    for item in list(args.extra_arg or []):
        extra_args.extend(shlex.split(str(item)))

    config = wrist_relation.WristRelationConfig(
        object_name=str(args.object_name),
        python=str(args.python),
        output_dir=run_dir / "wrist_relation_runs",
        relation_out=relation_out,
        hand_eye_path=Path(args.T_gripper_cam_path).expanduser() if args.T_gripper_cam_path else None,
        init_T_gripper_obj_path=prior_path,
        timeout_s=float(args.timeout_s),
        min_confidence=0.25,
        min_projection_iou=0.08,
        require_held=True,
        extra_args=tuple(extra_args),
    )

    t0 = time.perf_counter()
    job = wrist_relation.start_async(config)
    readiness: list[dict[str, Any]] = []
    sorted_durations = sorted(max(float(v), 0.0) for v in args.transport_duration_s)
    last_t = 0.0
    for duration in sorted_durations:
        sleep_s = max(0.0, duration - last_t)
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        last_t = duration
        readiness.append(
            {
                "transport_duration_s": duration,
                "ready_at_arrival": bool(job.done()),
                "elapsed_s": float(time.perf_counter() - t0),
            }
        )

    result = None
    wait_started = time.perf_counter()
    try:
        result = job.wait(timeout_s=float(args.wait_at_pre_place_s))
        wait_error = None
    except Exception as exc:
        wait_error = f"{type(exc).__name__}: {exc}"
    total_elapsed = float(time.perf_counter() - t0)
    wait_elapsed = float(time.perf_counter() - wait_started)
    estimator_elapsed = job.elapsed_s()
    for item in readiness:
        if estimator_elapsed is None:
            item["ready_with_pre_place_wait"] = False
        else:
            item["ready_with_pre_place_wait"] = bool(
                float(estimator_elapsed) <= float(item["transport_duration_s"]) + float(args.wait_at_pre_place_s)
            )
    summary = {
        "ok": bool(result is not None and result.accepted) if wait_error is None else False,
        "object_name": str(args.object_name),
        "run_dir": run_dir,
        "frame_npz": Path(args.frame_npz),
        "mask_path": Path(args.mask_path),
        "prior_path": prior_path,
        "relation_out": relation_out,
        "transport_durations": sorted_durations,
        "readiness": readiness,
        "wait_at_pre_place_s": float(args.wait_at_pre_place_s),
        "wait_elapsed_s": wait_elapsed,
        "total_elapsed_s": total_elapsed,
        "estimator_elapsed_s": estimator_elapsed,
        "wait_error": wait_error,
        "result_ready_after_wait": bool(result is not None),
        "accepted": bool(result.accepted) if result is not None else False,
        "confidence": float(result.confidence) if result is not None else None,
        "projection_iou": float(result.projection_iou) if result is not None else None,
        "grasp_state": result.grasp_state if result is not None else None,
        "reason": result.reason if result is not None else None,
        "T_gripper_obj": result.T_gripper_obj if result is not None else None,
        "command": wrist_relation.build_command(config),
    }
    summary_path = _write_json(run_dir / "summary.json", summary)
    print(f"[sim] summary={summary_path}")
    print(
        "[sim] result "
        f"accepted={summary['accepted']} conf={summary['confidence']} iou={summary['projection_iou']} "
        f"total_elapsed_s={total_elapsed:.3f} wait_elapsed_s={wait_elapsed:.3f}"
    )
    for item in readiness:
        print(
            "[sim] transport "
            f"{item['transport_duration_s']:.2f}s ready={item['ready_at_arrival']} "
            f"ready_plus_wait={item['ready_with_pre_place_wait']} "
            f"elapsed={item['elapsed_s']:.3f}s"
        )
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
