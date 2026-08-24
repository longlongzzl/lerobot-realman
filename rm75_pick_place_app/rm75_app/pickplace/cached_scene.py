"""Convert cached perception output into planning-domain inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from rm75_app.assets.object_specs import get_object_spec, resolve_object_spec_scales
from rm75_app.assets.collision_proxy import build_automatic_collision_proxy
from rm75_app.planning.contracts import CollisionObject, PlanningScene, Pose, PoseCandidate


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale,
             (rotation[0, 2] - rotation[2, 0]) / scale,
             (rotation[1, 0] - rotation[0, 1]) / scale]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.asarray([(rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale,
                               (rotation[0, 1] + rotation[1, 0]) / scale,
                               (rotation[0, 2] + rotation[2, 0]) / scale])
        elif axis == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = np.asarray([(rotation[0, 2] - rotation[2, 0]) / scale,
                               (rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                               (rotation[1, 2] + rotation[2, 1]) / scale])
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = np.asarray([(rotation[1, 0] - rotation[0, 1]) / scale,
                               (rotation[0, 2] + rotation[2, 0]) / scale,
                               (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale])
    return quat / np.linalg.norm(quat)


def _mesh_extents(path: Path, scale: float) -> np.ndarray:
    loaded = trimesh.load(path, force="scene", process=False)
    bounds = np.asarray(loaded.bounds, dtype=np.float64)
    return (bounds[1] - bounds[0]) * float(scale)


def _clamp_to_table(T_base_object: np.ndarray, extents: np.ndarray, table_z: float) -> np.ndarray:
    result = np.asarray(T_base_object, dtype=np.float64).reshape(4, 4).copy()
    half_world_height = 0.5 * float(np.abs(result[2, :3]) @ extents)
    result[2, 3] = max(float(result[2, 3]), float(table_z) + half_world_height + 0.002)
    return result


@dataclass(frozen=True)
class CachedKnownObjectScene:
    object_name: str
    rgb_path: Path
    depth_path: Path
    mask_path: Path
    T_base_object: np.ndarray
    object_collision: CollisionObject
    scene: PlanningScene


def load_cached_known_object_scene(
    result_json: str | Path,
    camera_extrinsic: str | Path,
    *,
    use_direct_camera_extrinsic: bool = False,
    table_z: float = -0.01,
) -> CachedKnownObjectScene:
    result_path = Path(result_json).expanduser().resolve()
    data = json.loads(result_path.read_text(encoding="utf-8"))
    object_name = str(data["object_name"])
    spec = get_object_spec(object_name)
    if spec is None:
        raise ValueError(f"cached object has no registered asset: {object_name}")
    run_dir = Path(data.get("run_dir") or result_path.parent).expanduser().resolve()
    rgb_path = run_dir / "rgb.png"
    depth_path = run_dir / "depth.png"
    mask_path = run_dir / "sam6d_binary_mask.png"
    for path in (rgb_path, depth_path, mask_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    T_camera_object = np.asarray(data["T_cam_obj"], dtype=np.float64).reshape(4, 4)
    extrinsic = np.asarray(np.load(Path(camera_extrinsic).expanduser()), dtype=np.float64).reshape(4, 4)
    T_base_camera = extrinsic if use_direct_camera_extrinsic else np.linalg.inv(extrinsic)
    T_base_object = T_base_camera @ T_camera_object

    mesh_scale, _ = resolve_object_spec_scales(spec)
    mesh_path = Path(spec.mesh_file).expanduser().resolve()
    extents = _mesh_extents(mesh_path, mesh_scale)
    T_base_object = _clamp_to_table(T_base_object, extents, table_z)
    collision = build_automatic_collision_proxy(
        object_name,
        spec,
        T_base_object,
    )
    table = CollisionObject(
        "virtual_table_plane",
        "cuboid",
        Pose([0.0, 0.0, table_z - 0.01], [1.0, 0.0, 0.0, 0.0]),
        dimensions=[1.2, 1.2, 0.02],
    )
    return CachedKnownObjectScene(
        object_name=object_name,
        rgb_path=rgb_path,
        depth_path=depth_path,
        mask_path=mask_path,
        T_base_object=T_base_object,
        object_collision=collision,
        scene=PlanningScene((table, collision), revision=result_path.stem),
    )


def topdown_grasp_candidates(
    scene: CachedKnownObjectScene,
    *,
    z_offset: float = 0.0,
    yaw_offsets_deg: tuple[float, ...] = (0.0, -12.0, 12.0, 90.0),
) -> tuple[PoseCandidate, ...]:
    rotation = scene.T_base_object[:3, :3]
    spec = get_object_spec(scene.object_name)
    mesh_scale, _ = resolve_object_spec_scales(spec)
    extents = _mesh_extents(Path(spec.mesh_file), mesh_scale)
    local_long_axis = np.zeros(3)
    local_long_axis[int(np.argmax(extents))] = 1.0
    long_axis = rotation @ local_long_axis
    long_xy = np.asarray([long_axis[0], long_axis[1], 0.0])
    if np.linalg.norm(long_xy) < 1e-8:
        long_xy = np.asarray([1.0, 0.0, 0.0])
    long_xy /= np.linalg.norm(long_xy)
    base_closing = np.asarray([-long_xy[1], long_xy[0], 0.0])
    approaching = np.asarray([0.0, 0.0, -1.0])
    output = []
    for index, yaw_degrees in enumerate(yaw_offsets_deg):
        angle = np.deg2rad(yaw_degrees)
        c, s = np.cos(angle), np.sin(angle)
        closing = np.asarray(
            [c * base_closing[0] - s * base_closing[1],
             s * base_closing[0] + c * base_closing[1], 0.0]
        )
        orthogonal = np.cross(closing, approaching)
        orthogonal /= np.linalg.norm(orthogonal)
        closing = np.cross(approaching, orthogonal)
        tcp_rotation = np.stack([orthogonal, closing, approaching], axis=1)
        position = scene.T_base_object[:3, 3].copy()
        position[2] += float(z_offset)
        output.append(
            PoseCandidate(
                f"grasp_{index:02d}_{yaw_degrees:+.0f}deg",
                Pose(position, matrix_to_quaternion_wxyz(tcp_rotation)),
                score=1.0 - 0.05 * index,
                metadata={"yaw_offset_deg": yaw_degrees, "source": "cached_known_mesh"},
            )
        )
    return tuple(output)
