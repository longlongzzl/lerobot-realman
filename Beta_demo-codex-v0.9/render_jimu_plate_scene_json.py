#!/usr/bin/env python3
"""Render an explicit Jimu plate-position JSON in a minimal SAPIEN scene."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import sapien


DEFAULT_SCENE = Path(__file__).resolve().parent / "jimu_exported_scenes" / "jimu_14tray_5base_only.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "jimu_exported_scenes" / "jimu_14tray_5base_only_preview.png"


def _mat_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 1e-12)) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 1e-12)) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 1e-12)) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    quat = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return quat.astype(np.float32)


def _look_at_pose(eye: np.ndarray, target: np.ndarray) -> sapien.Pose:
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    forward = target - eye
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    left = np.cross(np.asarray([0.0, 0.0, 1.0]), forward)
    if np.linalg.norm(left) < 1e-9:
        left = np.asarray([0.0, 1.0, 0.0])
    left /= max(float(np.linalg.norm(left)), 1e-12)
    up = np.cross(forward, left)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    rotation = np.stack([forward, left, up], axis=1)
    return sapien.Pose(eye.astype(np.float32), _mat_to_quat_wxyz(rotation))


def _load_objects(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{path} has no top-level results list")
    objects = [item for item in results if isinstance(item, dict) and item.get("ok", True)]
    if not objects:
        raise ValueError(f"{path} has no ok object entries")
    return objects


def _build_scene(objects: list[dict], *, show_ground: bool):
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    scene = engine.create_scene(sapien.SceneConfig())
    scene.set_timestep(1.0 / 60.0)
    scene.set_ambient_light([0.58, 0.58, 0.58])
    scene.add_directional_light([0.4, 0.6, -1.0], [1.0, 1.0, 1.0], shadow=True)
    scene.add_point_light([0.2, -0.4, 0.6], [0.7, 0.7, 0.7], shadow=False)
    if show_ground:
        scene.add_ground(0.0)

    positions = []
    for item in objects:
        mesh_file = str(Path(str(item.get("sim_asset_file") or item.get("mesh_file") or "")).expanduser())
        if not mesh_file or not Path(mesh_file).exists():
            raise FileNotFoundError(f"mesh not found for {item.get('object_name')}: {mesh_file}")
        scale = float(item.get("sim_asset_scale") or item.get("mesh_scale") or 1.0)
        T = np.asarray(item.get("T_base_obj"), dtype=np.float32).reshape(4, 4)
        builder = scene.create_actor_builder()
        builder.add_visual_from_file(mesh_file, scale=[scale, scale, scale])
        actor = builder.build_static(name=str(item.get("object_name") or "jimu_plate"))
        actor.set_pose(sapien.Pose(T[:3, 3], _mat_to_quat_wxyz(T[:3, :3])))
        positions.append(T[:3, 3].astype(np.float32))
    return scene, np.asarray(positions, dtype=np.float32)


def _save_preview(scene, points: np.ndarray, out_path: Path, width: int, height: int) -> None:
    center = points.mean(axis=0)
    span = np.ptp(points, axis=0)
    radius = max(float(np.linalg.norm(span[:2])), 0.35)
    eye = center + np.asarray([radius * 0.85, -radius * 0.95, max(0.38, radius * 0.75)], dtype=np.float32)
    target = center + np.asarray([0.0, 0.0, 0.035], dtype=np.float32)
    camera = scene.add_camera("preview_camera", width, height, 0.8, 0.01, 10.0)
    camera.set_pose(_look_at_pose(eye, target))
    scene.update_render()
    camera.take_picture()
    rgba = np.clip(camera.get_picture("Color"), 0.0, 1.0)
    image = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(image).save(out_path)


def _run_viewer(scene, points: np.ndarray, seconds: float | None) -> None:
    from sapien.utils.viewer import Viewer

    viewer = Viewer()
    viewer.set_scene(scene)
    center = points.mean(axis=0)
    span = np.ptp(points, axis=0)
    radius = max(float(np.linalg.norm(span[:2])), 0.35)
    viewer.set_camera_xyz(
        float(center[0] + radius * 0.75),
        float(center[1] - radius * 0.95),
        float(center[2] + max(0.35, radius * 0.75)),
    )
    viewer.set_camera_rpy(0.0, -0.58, 2.35)
    start = time.time()
    while not viewer.closed:
        scene.update_render()
        viewer.render()
        if seconds is not None and time.time() - start >= seconds:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-json", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--viewer", action="store_true", help="Open a live SAPIEN viewer after saving the preview image.")
    parser.add_argument("--viewer-seconds", type=float, default=0.0, help="Viewer lifetime; <=0 keeps it open until closed.")
    parser.add_argument("--no-ground", dest="show_ground", action="store_false", default=True)
    args = parser.parse_args()

    objects = _load_objects(args.scene_json.expanduser())
    scene, points = _build_scene(objects, show_ground=bool(args.show_ground))
    _save_preview(scene, points, args.out.expanduser(), int(args.width), int(args.height))
    print(f"[jimu-render] loaded objects: {len(objects)}")
    print(f"[jimu-render] saved preview: {args.out.expanduser()}")
    if bool(args.viewer):
        seconds = None if float(args.viewer_seconds) <= 0.0 else float(args.viewer_seconds)
        _run_viewer(scene, points, seconds)


if __name__ == "__main__":
    main()
