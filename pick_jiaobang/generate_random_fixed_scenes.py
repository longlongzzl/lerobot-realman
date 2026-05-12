#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_SCENES = (
    THIS_DIR / "test_scenes" / "current_table.json",
    THIS_DIR / "test_scenes" / "gluestick_desk_regression.json",
)
DEFAULT_OUTPUT_DIR = THIS_DIR / "test_scenes" / "generated_random_batch_001"
DEFAULT_MOVABLE_OBJECTS = ("bi", "carriot", "gluestick", "hongshupian", "lvmukuai", "shuazi", "tennis")
DEFAULT_FIXED_OBSTACLES = ("bitong", "desk")
DEFAULT_WORKSPACE_XY = (-0.58, 0.08, -0.34, 0.20)


# Conservative fallback dimensions in object local frame, used if mesh loading is unavailable.
FALLBACK_DIMS_M = {
    "bi": (0.023, 0.138, 0.023),
    "bitong": (0.085, 0.095, 0.085),
    "carriot": (0.030, 0.210, 0.030),
    "desk": (0.430, 0.100, 0.240),
    "gluestick": (0.032, 0.110, 0.032),
    "hongshupian": (0.070, 0.150, 0.070),
    "lvmukuai": (0.060, 0.060, 0.060),
    "shuazi": (0.105, 0.055, 0.030),
    "tennis": (0.070, 0.070, 0.070),
}


SWAP_GROUPS = (
    ("bi", "gluestick", "carriot", "shuazi"),
    ("lvmukuai", "tennis"),
    ("hongshupian",),
)


@dataclass(frozen=True)
class Aabb2:
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def inflated(self, margin: float) -> "Aabb2":
        return Aabb2(
            self.min_x - margin,
            self.max_x + margin,
            self.min_y - margin,
            self.max_y + margin,
        )

    def overlaps(self, other: "Aabb2") -> bool:
        return (
            self.min_x <= other.max_x
            and self.max_x >= other.min_x
            and self.min_y <= other.max_y
            and self.max_y >= other.min_y
        )


@dataclass(frozen=True)
class Footprint2:
    points: np.ndarray


def _as_matrix(entry: dict) -> np.ndarray:
    return np.asarray(entry["T_world_obj"], dtype=np.float64).reshape(4, 4)


def _to_json_matrix(T: np.ndarray) -> list[list[float]]:
    return np.asarray(T, dtype=np.float64).reshape(4, 4).tolist()


def _rot_z(deg: float) -> np.ndarray:
    theta = math.radians(float(deg))
    c = math.cos(theta)
    s = math.sin(theta)
    R = np.eye(3, dtype=np.float64)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    return R


def _apply_world_yaw(T: np.ndarray, yaw_deg: float) -> np.ndarray:
    out = np.asarray(T, dtype=np.float64).reshape(4, 4).copy()
    out[:3, :3] = _rot_z(float(yaw_deg)) @ out[:3, :3]
    return out


def _set_xy_keep_z(T: np.ndarray, xy: np.ndarray) -> np.ndarray:
    out = np.asarray(T, dtype=np.float64).reshape(4, 4).copy()
    out[0, 3] = float(xy[0])
    out[1, 3] = float(xy[1])
    return out


def _sample_yaw(rng: random.Random, yaw_step_deg: float) -> float:
    step = float(yaw_step_deg)
    if step <= 0.0:
        return rng.uniform(-180.0, 180.0)
    count = int(round(360.0 / step))
    values = [(-180.0 + i * step) for i in range(count)]
    return float(rng.choice(values))


def _load_scene(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    objects = data.get("objects")
    if not isinstance(objects, dict):
        raise ValueError(f"{path} does not contain top-level objects dict")
    for name, entry in objects.items():
        if not isinstance(entry, dict) or "T_world_obj" not in entry:
            raise ValueError(f"{path} object {name!r} is missing T_world_obj")
        _as_matrix(entry)
    return data


def _try_mesh_dims(name: str) -> tuple[float, float, float] | None:
    try:
        import trimesh
        from object_specs import get_object_spec, resolve_object_spec_scales

        spec = get_object_spec(name)
        if spec is None:
            return None
        asset = Path(str(spec.sim_asset_file or spec.mesh_file)).expanduser()
        _, sim_scale = resolve_object_spec_scales(spec)
        loaded = trimesh.load(str(asset), force="scene")
        bounds = loaded.bounds
        if bounds is None:
            return None
        extents = np.asarray(bounds[1] - bounds[0], dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(extents)) or float(np.max(extents)) <= 1e-9:
            return None
        dims = tuple(float(v) * float(sim_scale) for v in extents)
        return dims
    except Exception:
        return None


def _resolve_dims(names: Iterable[str]) -> dict[str, tuple[float, float, float]]:
    dims = {}
    for name in names:
        mesh_dims = _try_mesh_dims(name)
        dims[name] = mesh_dims or FALLBACK_DIMS_M.get(name, (0.08, 0.08, 0.08))
    return dims


def _cross_2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    pts = sorted({(round(float(p[0]), 9), round(float(p[1]), 9)) for p in np.asarray(points, dtype=np.float64)})
    if len(pts) <= 1:
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    arr = [np.asarray(p, dtype=np.float64) for p in pts]
    lower: list[np.ndarray] = []
    for p in arr:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in reversed(arr):
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=np.float64).reshape(-1, 2)


def _footprint_poly(T: np.ndarray, dims: tuple[float, float, float]) -> Footprint2:
    half = np.asarray(dims, dtype=np.float64).reshape(3) * 0.5
    corners = np.array(
        [
            [sx * half[0], sy * half[1], sz * half[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    pts = (np.asarray(T[:3, :3], dtype=np.float64) @ corners.T).T + np.asarray(T[:3, 3], dtype=np.float64)
    hull = _convex_hull_2d(pts[:, :2])
    return Footprint2(hull)


def _polys_overlap(poly_a: Footprint2, poly_b: Footprint2, clearance_m: float) -> bool:
    pts_a = np.asarray(poly_a.points, dtype=np.float64).reshape(-1, 2)
    pts_b = np.asarray(poly_b.points, dtype=np.float64).reshape(-1, 2)
    if len(pts_a) == 0 or len(pts_b) == 0:
        return False
    axes = []
    for pts in (pts_a, pts_b):
        for i in range(len(pts)):
            edge = pts[(i + 1) % len(pts)] - pts[i]
            norm = float(np.linalg.norm(edge))
            if norm <= 1e-9:
                continue
            axis = np.asarray([-edge[1], edge[0]], dtype=np.float64) / norm
            axes.append(axis)
    if not axes:
        return False
    clearance = float(max(clearance_m, 0.0))
    for axis in axes:
        proj_a = pts_a @ axis
        proj_b = pts_b @ axis
        if float(np.max(proj_a)) + clearance < float(np.min(proj_b)):
            return False
        if float(np.max(proj_b)) + clearance < float(np.min(proj_a)):
            return False
    return True


def _validate_layout(
    scene: dict,
    *,
    movable_names: tuple[str, ...],
    fixed_names: tuple[str, ...],
    dims: dict[str, tuple[float, float, float]],
    workspace_xy: tuple[float, float, float, float],
    clearance_m: float,
) -> tuple[bool, str]:
    objects = scene["objects"]
    check_names = [name for name in (*movable_names, *fixed_names) if name in objects]
    footprints: dict[str, Footprint2] = {}
    for name in check_names:
        T = _as_matrix(objects[name])
        if name in movable_names:
            x_min, x_max, y_min, y_max = workspace_xy
            center = T[:2, 3]
            if center[0] < x_min or center[0] > x_max or center[1] < y_min or center[1] > y_max:
                return False, f"{name} center out of workspace xy={center.tolist()}"
        footprints[name] = _footprint_poly(T, dims[name])

    for i, name_a in enumerate(check_names):
        for name_b in check_names[i + 1 :]:
            if _polys_overlap(footprints[name_a], footprints[name_b], float(clearance_m)):
                return False, f"footprint overlap: {name_a} vs {name_b}"
    return True, "ok"


def _write_scene(path: Path, scene: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scene, indent=2, ensure_ascii=False) + "\n")


def _base_name(path: Path) -> str:
    try:
        data = json.loads(path.read_text())
        value = data.get("scene_name")
        if value:
            return str(value)
    except Exception:
        pass
    return path.stem


def _make_scene(
    base_data: dict,
    *,
    mode: str,
    rng: random.Random,
    movable_names: tuple[str, ...],
    yaw_step_deg: float,
    jitter_m: float,
) -> dict:
    scene = copy.deepcopy(base_data)
    objects = scene["objects"]

    if mode in {"yaw_only", "jitter_yaw"}:
        for name in movable_names:
            if name not in objects:
                continue
            T = _as_matrix(objects[name])
            if mode == "jitter_yaw":
                jitter = np.asarray(
                    [rng.uniform(-jitter_m, jitter_m), rng.uniform(-jitter_m, jitter_m)],
                    dtype=np.float64,
                )
                T = _set_xy_keep_z(T, T[:2, 3] + jitter)
            T = _apply_world_yaw(T, _sample_yaw(rng, yaw_step_deg))
            objects[name]["T_world_obj"] = _to_json_matrix(T)
        return scene

    if mode == "same_class_swap_jitter":
        new_xy: dict[str, np.ndarray] = {}
        for group in SWAP_GROUPS:
            present = [name for name in group if name in objects and name in movable_names]
            if not present:
                continue
            src_positions = [np.asarray(_as_matrix(objects[name])[:2, 3], dtype=np.float64) for name in present]
            shuffled = src_positions[:]
            if len(present) > 1:
                for _ in range(20):
                    rng.shuffle(shuffled)
                    if any(np.linalg.norm(shuffled[i] - src_positions[i]) > 1e-6 for i in range(len(present))):
                        break
            for name, xy in zip(present, shuffled):
                jitter = np.asarray(
                    [rng.uniform(-jitter_m, jitter_m), rng.uniform(-jitter_m, jitter_m)],
                    dtype=np.float64,
                )
                new_xy[name] = xy + jitter
        for name in movable_names:
            if name not in objects:
                continue
            T = _as_matrix(objects[name])
            if name in new_xy:
                T = _set_xy_keep_z(T, new_xy[name])
            T = _apply_world_yaw(T, _sample_yaw(rng, yaw_step_deg))
            objects[name]["T_world_obj"] = _to_json_matrix(T)
        return scene

    raise ValueError(f"Unsupported mode: {mode}")


def _generate_for_mode(
    *,
    mode: str,
    count: int,
    base_scene_paths: tuple[Path, ...],
    base_scenes: dict[Path, dict],
    rng: random.Random,
    output_dir: Path,
    movable_names: tuple[str, ...],
    fixed_names: tuple[str, ...],
    dims: dict[str, tuple[float, float, float]],
    workspace_xy: tuple[float, float, float, float],
    clearance_m: float,
    yaw_step_deg: float,
    jitter_m: float,
    max_attempts_per_scene: int,
) -> list[dict]:
    generated = []
    per_base = [count // len(base_scene_paths)] * len(base_scene_paths)
    for i in range(count % len(base_scene_paths)):
        per_base[i] += 1

    for base_idx, (base_path, n_from_base) in enumerate(zip(base_scene_paths, per_base)):
        base_name = _base_name(base_path)
        base_data = base_scenes[base_path]
        for local_idx in range(n_from_base):
            last_reason = ""
            for attempt in range(1, int(max_attempts_per_scene) + 1):
                scene = _make_scene(
                    base_data,
                    mode=mode,
                    rng=rng,
                    movable_names=movable_names,
                    yaw_step_deg=yaw_step_deg,
                    jitter_m=jitter_m,
                )
                valid, reason = _validate_layout(
                    scene,
                    movable_names=movable_names,
                    fixed_names=fixed_names,
                    dims=dims,
                    workspace_xy=workspace_xy,
                    clearance_m=clearance_m,
                )
                last_reason = reason
                if not valid:
                    continue
                global_idx = len(generated)
                scene_name = f"random_{mode}_{base_name}_{global_idx:03d}"
                scene["version"] = int(scene.get("version", 1) or 1)
                scene["scene_name"] = scene_name
                scene["source"] = "generated_random_fixed_scene"
                scene["generated_metadata"] = {
                    "mode": mode,
                    "base_scene_file": str(base_path),
                    "base_scene_name": base_name,
                    "attempt": attempt,
                    "clearance_m": float(clearance_m),
                    "jitter_m": float(jitter_m),
                    "yaw_step_deg": float(yaw_step_deg),
                    "movable_objects": list(movable_names),
                    "fixed_obstacles": list(fixed_names),
                }
                filename = f"{mode}_{base_name}_{base_idx:02d}_{local_idx:02d}.json"
                out_path = output_dir / filename
                _write_scene(out_path, scene)
                generated.append(
                    {
                        "file": str(out_path),
                        "mode": mode,
                        "base_scene": str(base_path),
                        "scene_name": scene_name,
                        "attempt": attempt,
                    }
                )
                break
            else:
                raise RuntimeError(
                    f"Failed to generate valid scene for mode={mode}, base={base_path}, "
                    f"local_idx={local_idx}, last_reason={last_reason}"
                )
    return generated


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate collision-checked randomized fixed-scene JSON files.")
    parser.add_argument("--base-scenes", nargs="+", type=Path, default=list(DEFAULT_BASE_SCENES))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--count-per-mode", type=int, default=20)
    parser.add_argument(
        "--count-total",
        type=int,
        default=0,
        help="If >0, generate exactly this many scenes distributed across modes instead of count-per-mode for each mode.",
    )
    parser.add_argument("--modes", nargs="+", default=["yaw_only", "jitter_yaw", "same_class_swap_jitter"])
    parser.add_argument("--movable-objects", nargs="+", default=list(DEFAULT_MOVABLE_OBJECTS))
    parser.add_argument("--fixed-obstacles", nargs="+", default=list(DEFAULT_FIXED_OBSTACLES))
    parser.add_argument("--yaw-step-deg", type=float, default=15.0)
    parser.add_argument("--jitter-m", type=float, default=0.03)
    parser.add_argument("--swap-jitter-m", type=float, default=0.025)
    parser.add_argument("--clearance-m", type=float, default=0.010)
    parser.add_argument("--workspace-xy", nargs=4, type=float, default=list(DEFAULT_WORKSPACE_XY), metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"))
    parser.add_argument("--max-attempts-per-scene", type=int, default=2000)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rng = random.Random(int(args.seed))
    base_scene_paths = tuple(Path(p).expanduser().resolve() for p in args.base_scenes)
    base_scenes = {path: _load_scene(path) for path in base_scene_paths}
    movable_names = tuple(str(x) for x in args.movable_objects)
    fixed_names = tuple(str(x) for x in args.fixed_obstacles)
    all_names = tuple(dict.fromkeys((*movable_names, *fixed_names)))
    dims = _resolve_dims(all_names)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = list(args.modes)
    if int(args.count_total) > 0:
        total = int(args.count_total)
        base_count = total // max(len(modes), 1)
        extra = total % max(len(modes), 1)
        mode_counts = {
            str(mode): base_count + (1 if idx < extra else 0)
            for idx, mode in enumerate(modes)
        }
    else:
        mode_counts = {str(mode): int(args.count_per_mode) for mode in modes}

    generated = []
    for mode in modes:
        mode = str(mode)
        jitter = float(args.swap_jitter_m) if mode == "same_class_swap_jitter" else float(args.jitter_m)
        if mode == "yaw_only":
            jitter = 0.0
        generated.extend(
            _generate_for_mode(
                mode=mode,
                count=int(mode_counts.get(mode, 0)),
                base_scene_paths=base_scene_paths,
                base_scenes=base_scenes,
                rng=rng,
                output_dir=output_dir,
                movable_names=movable_names,
                fixed_names=fixed_names,
                dims=dims,
                workspace_xy=tuple(float(v) for v in args.workspace_xy),
                clearance_m=float(args.clearance_m),
                yaw_step_deg=float(args.yaw_step_deg),
                jitter_m=jitter,
                max_attempts_per_scene=int(args.max_attempts_per_scene),
            )
        )

    manifest = {
        "seed": int(args.seed),
        "count": len(generated),
        "base_scenes": [str(p) for p in base_scene_paths],
        "output_dir": str(output_dir),
        "modes": modes,
        "count_per_mode": int(args.count_per_mode),
        "count_total": int(args.count_total),
        "mode_counts": mode_counts,
        "movable_objects": list(movable_names),
        "fixed_obstacles": list(fixed_names),
        "workspace_xy": [float(v) for v in args.workspace_xy],
        "clearance_m": float(args.clearance_m),
        "yaw_step_deg": float(args.yaw_step_deg),
        "dimensions_m": {name: [float(v) for v in values] for name, values in dims.items()},
        "scenes": generated,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"[generate_random_fixed_scenes] wrote {len(generated)} scene(s) to {output_dir}")
    print(f"[generate_random_fixed_scenes] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
