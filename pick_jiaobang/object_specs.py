from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import trimesh


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    grounding_prompt: str
    mesh_file: str
    mesh_scale: float | None = None
    sim_asset_file: str | None = None
    sim_asset_scale: float | None = None
    real_longest_axis_m: float | None = None
    foundationpose_position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    foundationpose_local_rotation_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pregrasp_height: float | None = None
    grasp_z_offset: float | None = None
    goal_z_offset: float | None = None
    fixed_goal_joints_deg: tuple[float, float, float, float, float, float, float] | None = None
    grasp_mode: str = "object_normal"
    sim_static_friction: float | None = None
    sim_dynamic_friction: float | None = None
    sim_restitution: float | None = None
    sim_linear_damping: float | None = None
    sim_angular_damping: float | None = None
    scene_obstacle_box_scale: float | None = None


DEFAULT_FIXED_GOAL_JOINTS_DEG = (178.0, -5.0, 0.0, -70.0, 0.0, -102.0, 60.0)


OBJECT_SPECS: Dict[str, ObjectSpec] = {
    "shuazi": ObjectSpec(
        name="shuazi",
        grounding_prompt="white plastic brush.",
        mesh_file="~/Desktop/lerobot/pick_jiaobang/meshs/shuazi.glb",
        sim_asset_file="~/Desktop/lerobot/pick_jiaobang/meshs/shuazi.glb",
        real_longest_axis_m=0.105,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        scene_obstacle_box_scale=1.25,
    ),
    "bitong": ObjectSpec(
        name="bitong",
        grounding_prompt="beige cylindrical cup.",
        mesh_file="~/Desktop/lerobot/pick_jiaobang/meshs/holder_85x95.glb",
        sim_asset_file="~/Desktop/lerobot/pick_jiaobang/meshs/holder_sim_85x95.glb",
        real_longest_axis_m=0.095,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
    ),
    "bi": ObjectSpec(
        name="bi",
        grounding_prompt="red pen.",
        mesh_file="~/Desktop/lerobot/pick_jiaobang/meshs/pen.glb",
        sim_asset_file="~/Desktop/lerobot/pick_jiaobang/meshs/pen.glb",
        real_longest_axis_m=0.138,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        grasp_mode="pen_topdown_insert_ready",
        grasp_z_offset=0.004,
        sim_static_friction=2.5,
        sim_dynamic_friction=2.0,
        sim_restitution=0.0,
        sim_linear_damping=0.15,
        sim_angular_damping=8.0,
    ),
    "beizi": ObjectSpec(
        name="cup",
        grounding_prompt="stainless steel cup.",
        mesh_file="~/Desktop/lerobot/pick_jiaobang/meshs/cup.glb",
        sim_asset_file="~/Desktop/lerobot/pick_jiaobang/meshs/cup_sim.glb",
        real_longest_axis_m=0.09,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        grasp_mode="topdown_long_axis",
    ),
    "haimian": ObjectSpec(
        name="haimian",
        grounding_prompt="dishwashing sponge.",
        mesh_file="~/Desktop/lerobot/pick_jiaobang/meshs/haimian.glb",
        sim_asset_file="~/Desktop/lerobot/pick_jiaobang/meshs/haimian.glb",
        real_longest_axis_m=0.099316,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
    ),
    "lvmukuai": ObjectSpec(
        name="lvmukuai",
        grounding_prompt="green cube.",
        mesh_file="~/Desktop/lerobot/pick_jiaobang/meshs/lvmukuai.glb",
        sim_asset_file="~/Desktop/lerobot/pick_jiaobang/meshs/lvmukuai.glb",
        real_longest_axis_m=0.06,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        grasp_mode="topdown_long_axis",
    ),
    "hongshupian": ObjectSpec(
        name="hongshupian",
        grounding_prompt="Red Potato Chip Can.",
        mesh_file="~/Desktop/lerobot/pick_jiaobang/meshs/hongshupian.glb",
        sim_asset_file="~/Desktop/lerobot/pick_jiaobang/meshs/hongshupian.glb",
        real_longest_axis_m=0.15,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        grasp_mode="long_axis_adaptive",
    ),
    "desk": ObjectSpec(
        name="desk",
        grounding_prompt="small wooden tabletop platform.",
        mesh_file="~/Desktop/lerobot/pick_jiaobang/meshs/desk.glb",
        sim_asset_file="~/Desktop/lerobot/pick_jiaobang/meshs/desk_sim.glb",
        real_longest_axis_m=0.43,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        grasp_mode="topdown_long_axis",
        scene_obstacle_box_scale=1.1,
    ),
    "carriot": ObjectSpec(
        name="carriot",
        grounding_prompt="orange carrot.",
        mesh_file="~/Desktop/lerobot/pick_jiaobang/meshs/carriot.glb",
        sim_asset_file="~/Desktop/lerobot/pick_jiaobang/meshs/carriot_sim.glb",
        real_longest_axis_m=0.21,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        grasp_mode="long_axis_adaptive",
    ),
    "cafe": ObjectSpec(
        name="cafe",
        grounding_prompt="coffee can.",
        mesh_file="~/Desktop/lerobot/pick_jiaobang/meshs/cafe.glb",
        sim_asset_file="~/Desktop/lerobot/pick_jiaobang/meshs/cafe_sim.glb",
        real_longest_axis_m=0.175,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        grasp_mode="long_axis_adaptive",
    ),
    "redcube": ObjectSpec(
        name="redcube",
        grounding_prompt="red wooden cube.",
        mesh_file="~/Desktop/lerobot/pick_jiaobang/meshs/redcube.glb",
        sim_asset_file="~/Desktop/lerobot/pick_jiaobang/meshs/redcube_sim.glb",
        real_longest_axis_m=0.03,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        grasp_mode="topdown_long_axis",
    ),
    "tennis": ObjectSpec(
        name="tennis",
        grounding_prompt="tennis ball.",
        mesh_file="~/Desktop/lerobot/pick_jiaobang/meshs/tennis.glb",
        sim_asset_file="~/Desktop/lerobot/pick_jiaobang/meshs/tennis_sim.glb",
        real_longest_axis_m=0.07,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        grasp_mode="topdown_long_axis",
    ),
    "greenpen": ObjectSpec(
        name="greenpen",
        grounding_prompt="green marker pen.",
        mesh_file="~/Desktop/lerobot/pick_jiaobang/meshs/greenpen.glb",
        sim_asset_file="~/Desktop/lerobot/pick_jiaobang/meshs/greenpen_sim.glb",
        real_longest_axis_m=0.137,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        grasp_mode="long_axis_adaptive",
        sim_static_friction=2.5,
        sim_dynamic_friction=2.0,
        sim_restitution=0.0,
        sim_linear_damping=0.15,
        sim_angular_damping=8.0,
    ),
    "gluestick": ObjectSpec(
        name="gluestick",
        grounding_prompt="a small glue stick.",
        mesh_file="~/Desktop/lerobot/pick_jiaobang/meshs/gluestick.glb",
        sim_asset_file="~/Desktop/lerobot/pick_jiaobang/meshs/gluestick_sim.glb",
        real_longest_axis_m=0.11,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        grasp_mode="long_axis_adaptive",
        sim_static_friction=2.2,
        sim_dynamic_friction=1.8,
        sim_restitution=0.0,
        sim_linear_damping=0.12,
        sim_angular_damping=6.0,
    ),
    # ---------------  roof assembly  ---------------
    "red_bricks_cube": ObjectSpec(
        name="red_bricks_cube",
        grounding_prompt="small square plastic building block.",
        mesh_file="~/Desktop/lerobot/FoundationPose/assets/red_jimu_cube.glb",
        mesh_scale=0.1,
        sim_asset_file="~/Desktop/lerobot/FoundationPose/assets/red_jimu_cube.glb",
        sim_asset_scale=0.1,
        real_longest_axis_m=0.105,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
    ),
    "red_triangle_front": ObjectSpec(
        name="red_triangle_front",
        grounding_prompt="red triangle panel.",
        mesh_file="~/Desktop/lerobot/pick_jiaobang/meshs/red_triangle.glb",
        sim_asset_file="~/Desktop/lerobot/pick_jiaobang/meshs/red_triangle.glb",
        real_longest_axis_m=0.12,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        grasp_mode="topdown_long_axis",
    ),
    "red_triangle_back": ObjectSpec(
        name="red_triangle_back",
        grounding_prompt="red triangle panel.",
        mesh_file="~/Desktop/lerobot/pick_jiaobang/meshs/red_triangle.glb",
        sim_asset_file="~/Desktop/lerobot/pick_jiaobang/meshs/red_triangle.glb",
        real_longest_axis_m=0.12,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        grasp_mode="topdown_long_axis",
    ),
    "red_triangle_left": ObjectSpec(
        name="red_triangle_left",
        grounding_prompt="red triangle panel.",
        mesh_file="~/Desktop/lerobot/pick_jiaobang/meshs/red_triangle.glb",
        sim_asset_file="~/Desktop/lerobot/pick_jiaobang/meshs/red_triangle.glb",
        real_longest_axis_m=0.12,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        grasp_mode="topdown_long_axis",
    ),
    "red_triangle_right": ObjectSpec(
        name="red_triangle_right",
        grounding_prompt="red triangle panel.",
        mesh_file="~/Desktop/lerobot/pick_jiaobang/meshs/red_triangle.glb",
        sim_asset_file="~/Desktop/lerobot/pick_jiaobang/meshs/red_triangle.glb",
        real_longest_axis_m=0.12,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        grasp_mode="topdown_long_axis",
    ),
    # ---------------  pavilion base + pillars  ---------------
    "tingzi_base": ObjectSpec(
        name="tingzi_base",
        grounding_prompt="small square pavilion base.",
        mesh_file="~/Desktop/lerobot/rm75_pick_place_app/assets/meshs/tingzi_base.obj",
        sim_asset_file="~/Desktop/lerobot/rm75_pick_place_app/assets/meshs/tingzi_base.obj",
        real_longest_axis_m=0.068,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        grasp_mode="topdown_long_axis",
        scene_obstacle_box_scale=1.05,
    ),
    "tingzi_pillar_front_left": ObjectSpec(
        name="tingzi_pillar_front_left",
        grounding_prompt="small vertical pavilion pillar.",
        mesh_file="~/Desktop/lerobot/rm75_pick_place_app/assets/meshs/tingzi_pillar1.obj",
        sim_asset_file="~/Desktop/lerobot/rm75_pick_place_app/assets/meshs/tingzi_pillar1.obj",
        real_longest_axis_m=0.094,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        grasp_mode="long_axis_adaptive",
        sim_static_friction=2.5,
        sim_dynamic_friction=2.0,
        sim_restitution=0.0,
        sim_linear_damping=0.15,
        sim_angular_damping=8.0,
        scene_obstacle_box_scale=1.5,
    ),
    "tingzi_pillar_front_right": ObjectSpec(
        name="tingzi_pillar_front_right",
        grounding_prompt="small vertical pavilion pillar.",
        mesh_file="~/Desktop/lerobot/rm75_pick_place_app/assets/meshs/tingzi_pillar1.obj",
        sim_asset_file="~/Desktop/lerobot/rm75_pick_place_app/assets/meshs/tingzi_pillar1.obj",
        real_longest_axis_m=0.094,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        grasp_mode="long_axis_adaptive",
        sim_static_friction=2.5,
        sim_dynamic_friction=2.0,
        sim_restitution=0.0,
        sim_linear_damping=0.15,
        sim_angular_damping=8.0,
        scene_obstacle_box_scale=1.5,
    ),
    "tingzi_pillar_back_left": ObjectSpec(
        name="tingzi_pillar_back_left",
        grounding_prompt="small vertical pavilion pillar.",
        mesh_file="~/Desktop/lerobot/rm75_pick_place_app/assets/meshs/tingzi_pillar1.obj",
        sim_asset_file="~/Desktop/lerobot/rm75_pick_place_app/assets/meshs/tingzi_pillar1.obj",
        real_longest_axis_m=0.094,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        grasp_mode="long_axis_adaptive",
        sim_static_friction=2.5,
        sim_dynamic_friction=2.0,
        sim_restitution=0.0,
        sim_linear_damping=0.15,
        sim_angular_damping=8.0,
        scene_obstacle_box_scale=1.5,
    ),
    "tingzi_pillar_back_right": ObjectSpec(
        name="tingzi_pillar_back_right",
        grounding_prompt="small vertical pavilion pillar.",
        mesh_file="~/Desktop/lerobot/rm75_pick_place_app/assets/meshs/tingzi_pillar1.obj",
        sim_asset_file="~/Desktop/lerobot/rm75_pick_place_app/assets/meshs/tingzi_pillar1.obj",
        real_longest_axis_m=0.094,
        fixed_goal_joints_deg=DEFAULT_FIXED_GOAL_JOINTS_DEG,
        grasp_mode="long_axis_adaptive",
        sim_static_friction=2.5,
        sim_dynamic_friction=2.0,
        sim_restitution=0.0,
        sim_linear_damping=0.15,
        sim_angular_damping=8.0,
        scene_obstacle_box_scale=1.5,
    ),
}


OBJECT_NAME_ALIASES: Dict[str, str] = {
    "brush": "shuazi",
    "holder": "bitong",
    "pen": "bi",
    "cup": "beizi",
    "sponge": "haimian",
    "green_cube": "lvmukuai",
    "potato_chips": "hongshupian",
    "chips": "hongshupian",
    "table": "desk",
    "zhuozi": "desk",
    "carrot": "carriot",
    "huluobo": "carriot",
    "coffee": "cafe",
    "coffee_can": "cafe",
    "kafei": "cafe",
    "red_cube": "redcube",
    "hongmukuai": "redcube",
    "wangqiu": "tennis",
    "green_pen": "greenpen",
    "lvsebi": "greenpen",
    "glue_stick": "gluestick",
    "jiaobang": "gluestick",
    # roof assembly aliases
    "roof_tri_front": "red_triangle_front",
    "roof_tri_back": "red_triangle_back",
    "roof_tri_left": "red_triangle_left",
    "roof_tri_right": "red_triangle_right",
    "red_triangle": "red_triangle_front",
    "bricks_cube": "red_bricks_cube",
    "pavilion_base": "tingzi_base",
    "tingzi": "tingzi_base",
    "tingzi_pillar": "tingzi_pillar_front_left",
    "pavilion_pillar": "tingzi_pillar_front_left",
    "pillar_front_left": "tingzi_pillar_front_left",
    "pillar_front_right": "tingzi_pillar_front_right",
    "pillar_back_left": "tingzi_pillar_back_left",
    "pillar_back_right": "tingzi_pillar_back_right",
}


def normalize_object_name(name: str | None) -> str | None:
    if name is None:
        return None
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return None
    return OBJECT_NAME_ALIASES.get(normalized, normalized)



def get_object_spec(name: str | None) -> ObjectSpec | None:
    normalized = normalize_object_name(name)
    if normalized is None:
        return None
    return OBJECT_SPECS.get(normalized)



def list_object_spec_names() -> list[str]:
    return sorted(OBJECT_SPECS.keys())



def iter_object_specs() -> Iterable[ObjectSpec]:
    for key in list_object_spec_names():
        yield OBJECT_SPECS[key]


_MESH_LONGEST_AXIS_CACHE: dict[tuple[str, int, int], float] = {}


def _load_mesh_longest_axis_m(asset_file: str) -> float:
    asset_path = str(Path(asset_file).expanduser())
    path = Path(asset_path)
    stat = path.stat()
    cache_key = (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    cached = _MESH_LONGEST_AXIS_CACHE.get(cache_key)
    if cached is not None:
        return float(cached)
    loaded = trimesh.load(asset_path, force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        bounds = loaded.bounds
    else:
        bounds = loaded.bounds
        if bounds is None:
            raise RuntimeError(f"No scene bounds found in asset: {asset_file}")
    extents = np.asarray(bounds[1] - bounds[0], dtype=np.float64).reshape(3)
    longest = float(np.max(extents))
    _MESH_LONGEST_AXIS_CACHE[cache_key] = longest
    return longest


def _resolve_uniform_scale(asset_file: str, explicit_scale: float | None, real_longest_axis_m: float | None) -> float:
    if explicit_scale is not None:
        return float(explicit_scale)
    if real_longest_axis_m is None:
        raise ValueError(f"Object spec for {asset_file} must define either an explicit scale or real_longest_axis_m")
    raw_longest_axis_m = _load_mesh_longest_axis_m(asset_file)
    if raw_longest_axis_m <= 1e-9:
        raise ValueError(f"Asset {asset_file} has a degenerate longest axis and cannot be scaled from real_longest_axis_m")
    return float(real_longest_axis_m) / float(raw_longest_axis_m)


def resolve_object_spec_scales(spec: ObjectSpec) -> tuple[float, float]:
    mesh_scale = _resolve_uniform_scale(spec.mesh_file, spec.mesh_scale, spec.real_longest_axis_m)
    sim_asset_file = spec.sim_asset_file or spec.mesh_file
    sim_scale = _resolve_uniform_scale(sim_asset_file, spec.sim_asset_scale, spec.real_longest_axis_m)
    return float(mesh_scale), float(sim_scale)



def describe_object_specs() -> str:
    lines = []
    for spec in iter_object_specs():
        mesh_name = Path(spec.mesh_file).name
        prompt = spec.grounding_prompt
        try:
            mesh_scale, sim_scale = resolve_object_spec_scales(spec)
        except Exception:
            mesh_scale, sim_scale = spec.mesh_scale, spec.sim_asset_scale
        lines.append(
            f"- {spec.name}: prompt={prompt!r}, mesh={mesh_name}, "
            f"real_longest_axis_m={spec.real_longest_axis_m}, "
            f"mesh_scale={mesh_scale}, sim_asset_scale={sim_scale}, grasp_mode={spec.grasp_mode}"
        )
    return "\n".join(lines)
