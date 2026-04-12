"""
Asset scale calibrator (no CLI args).

Place this script under: mani_skill/envs/tasks/tabletop/
Run:
  python asset_scale_calibrator.py

Default demo asset: jiaobang.glb in the same folder.

What it does:
1) Loads the mesh and reports its current AABB (L/W/H) in meters.
2) Prompts you to enter the real-world L/W/H (supports cm or m input).
3) Computes per-axis scale factors (sx, sy, sz).
4) Optionally validates by loading into a minimal SAPIEN scene and reading the collision/visual bounds after scaling.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
asset_path = os.path.join(here, "../holder.glb")

@dataclass
class Dims:
    x: float
    y: float
    z: float

    def as_np(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


def _fmt_dims(d: Dims) -> str:
    return f"(x={d.x:.6f} m, y={d.y:.6f} m, z={d.z:.6f} m)"


def _prompt_real_dims() -> Dims:
    print("\nEnter REAL-WORLD dimensions for the object.")
    print("You can type in cm or m, examples:")
    print("  12 3 2 cm")
    print("  0.12 0.03 0.02 m")
    while True:
        s = input("Real L W H + unit (cm/m): ").strip().lower()
        parts = s.replace(",", " ").split()
        if len(parts) != 4:
            print("Expected 4 tokens: <L> <W> <H> <cm|m>")
            continue
        try:
            l = float(parts[0])
            w = float(parts[1])
            h = float(parts[2])
        except ValueError:
            print("Failed to parse numbers. Try again.")
            continue
        unit = parts[3]
        if unit not in ("cm", "m"):
            print("Unit must be 'cm' or 'm'. Try again.")
            continue
        if unit == "cm":
            l, w, h = l / 100.0, w / 100.0, h / 100.0
        if min(l, w, h) <= 0:
            print("All dims must be > 0.")
            continue
        return Dims(l, w, h)


def _load_mesh_extents_trimesh(asset_path: str) -> Dims:
    try:
        import trimesh  # optional but recommended
    except Exception as e:
        raise RuntimeError(
            "This script requires trimesh. Install with: pip install trimesh"
        ) from e

    scene_or_mesh = trimesh.load(asset_path, force="scene")
    if hasattr(scene_or_mesh, "geometry") and len(scene_or_mesh.geometry) > 0:
        # Combine all geometries into one AABB in its loaded frame
        bounds_list = []
        for g in scene_or_mesh.geometry.values():
            if g.bounds is None:
                continue
            bounds_list.append(g.bounds)
        if len(bounds_list) == 0:
            raise RuntimeError("No geometry bounds found in asset.")
        bounds = np.array(bounds_list)
        bmin = bounds[:, 0, :].min(axis=0)
        bmax = bounds[:, 1, :].max(axis=0)
    else:
        mesh = scene_or_mesh
        if mesh.bounds is None:
            raise RuntimeError("No mesh bounds found in asset.")
        bmin, bmax = mesh.bounds

    ext = (bmax - bmin).astype(np.float64)  # (x, y, z)
    return Dims(float(ext[0]), float(ext[1]), float(ext[2]))


def _validate_in_sapien(asset_path: str, scale_xyz: np.ndarray) -> Tuple[Dims, Dims]:
    """
    Returns (visual_extents_m, collision_extents_m) if possible.
    If collision bounds are unavailable, collision_extents_m is zeros.
    """
    try:
        import sapien
        from mani_skill.utils.structs.actor import Actor
    except Exception as e:
        raise RuntimeError(
            "SAPIEN + ManiSkill must be importable to validate in simulation."
        ) from e

    engine = sapien.Engine()
    scene = engine.create_scene()
    scene.set_timestep(1 / 240)
    # Add a light so visuals exist (not strictly required for bounds)
    scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=False)

    builder = scene.create_actor_builder()
    builder.initial_pose = sapien.Pose(p=[0, 0, 0])
    builder.add_visual_from_file(asset_path, scale=scale_xyz.tolist())
    # Try add collision; if it fails for GLB, we'll still keep visuals for bounds
    try:
        builder.add_nonconvex_collision_from_file(asset_path, scale=scale_xyz.tolist())
    except Exception:
        try:
            builder.add_multiple_convex_collisions_from_file(
                asset_path, decomposition="coacd", scale=scale_xyz.tolist()
            )
        except Exception:
            pass

    obj = builder.build(name="asset")
    actor = Actor([obj], scene=scene)  # lightweight wrapper for bounds helpers

    visual_ext = Dims(0.0, 0.0, 0.0)
    coll_ext = Dims(0.0, 0.0, 0.0)

    # Visual bounds: use render shapes AABB in world frame (fallback to zero if not accessible)
    try:
        comp = obj.entity.find_component_by_type(sapien.render.RenderBodyComponent)
        if comp is not None and len(comp.render_shapes) > 0:
            mins = []
            maxs = []
            for rs in comp.render_shapes:
                aabb = rs.bounding_box
                mins.append(np.array(aabb.min, dtype=np.float64))
                maxs.append(np.array(aabb.max, dtype=np.float64))
            bmin = np.min(np.stack(mins, axis=0), axis=0)
            bmax = np.max(np.stack(maxs, axis=0), axis=0)
            ext = bmax - bmin
            visual_ext = Dims(float(ext[0]), float(ext[1]), float(ext[2]))
    except Exception:
        pass

    # Collision bounds: via PhysX collision mesh if available (ManiSkill Actor helper)
    try:
        cm = actor.get_first_collision_mesh()
        if cm is not None:
            b = cm.bounding_box.bounds
            ext = (b[1] - b[0]).astype(np.float64)
            coll_ext = Dims(float(ext[0]), float(ext[1]), float(ext[2]))
    except Exception:
        pass

    return visual_ext, coll_ext


def main() -> int:

    if not os.path.exists(asset_path):
        print(f"Asset not found: {asset_path}")
        return 1

    print(f"Asset: {asset_path}")
    cur = _load_mesh_extents_trimesh(asset_path)
    print("\nCurrent AABB extents from mesh (loaded units):")
    print(f"  current: {_fmt_dims(cur)}")

    real = _prompt_real_dims()
    print("\nReal-world target extents:")
    print(f"  real:    {_fmt_dims(real)}")

    cur_np = cur.as_np()
    real_np = real.as_np()
    scale = real_np / np.maximum(cur_np, 1e-12)

    print("\nPer-axis scale factors to match real dims:")
    print(f"  scale_xyz = [{scale[0]:.6f}, {scale[1]:.6f}, {scale[2]:.6f}]")
    print("\nTip: If you want a single uniform scale, use something like mean(scale_xyz).")

    try:
        vis_ext, col_ext = _validate_in_sapien(asset_path, scale)
        print("\nValidation in a minimal SAPIEN scene (approx AABB after scaling):")
        if (vis_ext.x + vis_ext.y + vis_ext.z) > 0:
            print(f"  visual:    {_fmt_dims(vis_ext)}")
        else:
            print("  visual:    <unavailable>")
        if (col_ext.x + col_ext.y + col_ext.z) > 0:
            print(f"  collision: {_fmt_dims(col_ext)}")
        else:
            print("  collision: <unavailable>")
    except Exception as e:
        print("\n(Skipped SAPIEN validation)")
        print(f"Reason: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

